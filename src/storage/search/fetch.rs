//! Chunk row retrieval: the shared row mapper and chunk-type filter, the
//! id/path metadata getters that read `chunks` rows without ranking, and the
//! id-only source-navigation lookup (`get_chunks_for_from_target`).

use std::collections::HashMap;

use amici::storage::filter::{append_in_filter, append_like_prefix_filter};
use rusqlite::{Connection, Error as RusqliteError, Row, params, params_from_iter, types::ToSql};

use crate::storage::{
    Chunk, ChunkType, SourceKind, StorageError, anon_placeholders, in_placeholders,
};

/// Deserializes a [`Chunk`] from a row whose columns at `offset..=offset + 8`
/// are, in `chunks` table order: file_path, chunk_type, name, content,
/// start_line, end_line, parent_chunk_id, source_kind, injection_flags. Callers
/// that prefix extra columns pass the matching offset — `get_chunk_by_id`
/// selects the 9 columns at offset 0; `get_chunks_by_ids` and the FTS path
/// prefix `id` and pass offset 1.
pub(super) fn chunk_from_row(row: &Row<'_>, offset: usize) -> rusqlite::Result<Chunk> {
    let injection_flags_json: Option<String> = row.get(offset + 8)?;
    let injection_flags = injection_flags_json.and_then(|s| {
        serde_json::from_str::<Vec<String>>(&s)
            .inspect_err(|e| {
                tracing::warn!(error = %e, "injection_flags JSON parse failed, treating as None");
            })
            .ok()
    });
    Ok(Chunk {
        file_path: row.get(offset)?,
        chunk_type: ChunkType::from_db(row.get::<_, String>(offset + 1)?.as_ref()),
        name: row.get(offset + 2)?,
        content: row.get(offset + 3)?,
        start_line: row.get(offset + 4)?,
        end_line: row.get(offset + 5)?,
        parent_chunk_id: row.get(offset + 6)?,
        source_kind: row
            .get::<_, Option<String>>(offset + 7)?
            .map(|s| SourceKind::from_db(s.as_ref())),
        injection_flags,
    })
}

/// Appends a chunk-type `IN (...)` filter, treating both `None` and
/// `Some(&[])` as "no filter". Contrast with [`append_in_filter`], which
/// renders `Some(&[])` as `AND 1 = 0`.
///
/// Precondition: `sql` ends inside an open WHERE clause — callers anchor with
/// `MATCH ?` (FTS path) or a trailing `IN (...)` (vec path) so the leading
/// ` AND ` from amici helpers attaches cleanly.
pub(super) fn append_chunk_type_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    types: Option<&[ChunkType]>,
) {
    let Some(types) = types.filter(|t| !t.is_empty()) else {
        return;
    };
    let strings: Vec<String> = types.iter().map(|c| c.as_str().to_owned()).collect();
    append_in_filter(sql, params, column, Some(&strings));
}

pub fn get_chunk_by_id(conn: &Connection, chunk_id: i64) -> Result<Option<Chunk>, StorageError> {
    let mut stmt = conn.prepare_cached(
        "SELECT file_path, chunk_type, name, content, start_line, end_line, parent_chunk_id, source_kind, injection_flags
         FROM chunks WHERE id = ?1",
    )?;
    match stmt.query_row([chunk_id], |row| chunk_from_row(row, 0)) {
        Ok(chunk) => Ok(Some(chunk)),
        Err(RusqliteError::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Bulk-fetch chunk metadata by ids with optional type/path filters applied
/// at the SQL layer. Shared by [`super::vec_search`], [`super::vec_search_multi`], and
/// callers that need a plain id→chunk lookup.
pub fn get_chunks_by_ids(
    conn: &Connection,
    ids: &[i64],
    type_filter: Option<&[ChunkType]>,
    path_filter: &[String],
) -> Result<HashMap<i64, Chunk>, StorageError> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut sql = format!(
        "SELECT id, file_path, chunk_type, name, content, start_line, end_line, parent_chunk_id, source_kind, injection_flags \
         FROM chunks WHERE id IN ({})",
        anon_placeholders(ids.len())
    );
    let mut params: Vec<Box<dyn ToSql>> = ids
        .iter()
        .map(|id| Box::new(*id) as Box<dyn ToSql>)
        .collect();
    append_like_prefix_filter(&mut sql, &mut params, "file_path", path_filter);
    append_chunk_type_filter(&mut sql, &mut params, "chunk_type", type_filter);

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(params.iter()), |row| {
        Ok((row.get::<_, i64>(0)?, chunk_from_row(row, 1)?))
    })?;
    rows.collect::<Result<HashMap<_, _>, _>>()
        .map_err(Into::into)
}

/// Bulk-fetch every Chunk (with body content) belonging to the given files.
/// Output is sorted by `(file_path, start_line)` so brief expansion can apply
/// topological ordering on top without an extra sort pass.
pub fn get_chunks_for_files(conn: &Connection, paths: &[&str]) -> Result<Vec<Chunk>, StorageError> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = in_placeholders(paths.len());
    let sql = format!(
        "SELECT file_path, chunk_type, name, content, start_line, end_line, parent_chunk_id, source_kind, injection_flags
         FROM chunks WHERE file_path IN ({placeholders})
         ORDER BY file_path, start_line"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(paths.iter()), |row| chunk_from_row(row, 0))?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_chunks_for_from_target(
    conn: &Connection,
    file_path: &str,
    symbol: Option<&str>,
) -> Result<Vec<i64>, StorageError> {
    let mut sql = String::from(
        "SELECT id FROM chunks \
         WHERE file_path = ?1 AND chunk_type != 'inner_fn'",
    );
    if symbol.is_some() {
        sql.push_str(" AND name = ?2");
    }

    let mut stmt = conn.prepare(&sql)?;
    let ids = if let Some(sym) = symbol {
        stmt.query_map(params![file_path, sym], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?
    } else {
        stmt.query_map([file_path], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok(ids)
}
