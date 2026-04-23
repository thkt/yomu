use std::collections::{HashMap, HashSet};

use amici::storage::filter::{
    append_exclude_ids, append_in_filter, append_include_ids, append_like_prefix_filter,
};
use rurico::storage::{fts_quote, prepare_match_query};
use rusqlite::{Connection, Error as RusqliteError, Row, params, params_from_iter, types::ToSql};

use super::{
    Chunk, ChunkType, MatchSource, SearchResult, StorageError, anon_placeholders, f32_as_bytes,
    in_placeholders,
};

const VEC_MAXSIM_OVERSAMPLE: u32 = 10;

fn chunk_from_row(row: &Row<'_>, offset: usize) -> rusqlite::Result<Chunk> {
    Ok(Chunk {
        file_path: row.get(offset)?,
        chunk_type: ChunkType::from_db(row.get::<_, String>(offset + 1)?.as_ref()),
        name: row.get(offset + 2)?,
        content: row.get(offset + 3)?,
        start_line: row.get(offset + 4)?,
        end_line: row.get(offset + 5)?,
        parent_chunk_id: row.get(offset + 6)?,
    })
}

/// Appends a chunk-type `IN (...)` filter, treating both `None` and
/// `Some(&[])` as "no filter". Contrast with [`append_in_filter`], which
/// renders `Some(&[])` as `AND 1 = 0`.
///
/// Precondition: `sql` ends inside an open WHERE clause — callers anchor with
/// `MATCH ?` (FTS path) or a trailing `IN (...)` (vec path) so the leading
/// ` AND ` from amici helpers attaches cleanly.
fn append_chunk_type_filter(
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

pub fn vec_search(
    conn: &Connection,
    query_embedding: &[f32],
    limit: u32,
    type_filter: Option<&[ChunkType]>,
    path_filter: &[String],
) -> Result<Vec<SearchResult>, StorageError> {
    let query_bytes = f32_as_bytes(query_embedding);
    let k = limit.saturating_mul(VEC_MAXSIM_OVERSAMPLE);

    // KNN query — fetch only chunk_id + distance.
    // vec0 auxiliary columns (+chunk_id) cannot be used in JOIN conditions,
    // so we avoid a direct JOIN here.
    let knn_rows: Vec<(i64, f32)> = {
        let mut stmt = conn.prepare_cached(
            "SELECT chunk_id, distance FROM vec_chunks \
             WHERE embedding MATCH ?1 AND k = ?2 \
             ORDER BY distance",
        )?;
        stmt.query_map(params![query_bytes, k], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?
    };

    if knn_rows.is_empty() {
        return Ok(Vec::new());
    }

    // MaxSim: keep the sub-embedding with the smallest distance per chunk_id.
    let mut best: HashMap<i64, f32> = HashMap::new();
    for (chunk_id, distance) in &knn_rows {
        best.entry(*chunk_id)
            .and_modify(|d| {
                if *distance < *d {
                    *d = *distance;
                }
            })
            .or_insert(*distance);
    }

    let chunk_ids: Vec<i64> = best.keys().copied().collect();
    let mut sql = format!(
        "SELECT id, file_path, chunk_type, name, content, start_line, end_line, parent_chunk_id \
         FROM chunks WHERE id IN ({})",
        anon_placeholders(chunk_ids.len())
    );
    let mut params: Vec<Box<dyn ToSql>> = chunk_ids
        .iter()
        .map(|id| Box::new(*id) as Box<dyn ToSql>)
        .collect();
    append_like_prefix_filter(&mut sql, &mut params, "file_path", path_filter);
    append_chunk_type_filter(&mut sql, &mut params, "chunk_type", type_filter);

    let mut stmt2 = conn.prepare(&sql)?;
    let meta: HashMap<i64, Chunk> = stmt2
        .query_map(params_from_iter(params.iter()), |row| {
            Ok((
                row.get::<_, i64>(0)?,
                Chunk {
                    file_path: row.get(1)?,
                    chunk_type: ChunkType::from_db(row.get::<_, String>(2)?.as_ref()),
                    name: row.get(3)?,
                    content: row.get(4)?,
                    start_line: row.get(5)?,
                    end_line: row.get(6)?,
                    parent_chunk_id: row.get(7)?,
                },
            ))
        })?
        .collect::<Result<HashMap<_, _>, _>>()?;

    let mut results: Vec<SearchResult> = best
        .into_iter()
        .filter_map(|(chunk_id, distance)| {
            meta.get(&chunk_id).map(|chunk| SearchResult {
                chunk: chunk.clone(),
                chunk_id: Some(chunk_id),
                distance,
                match_source: MatchSource::Semantic,
                score: 1.0 / (1.0 + distance),
            })
        })
        .collect();

    results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    results.truncate(limit as usize);
    Ok(results)
}

#[allow(clippy::cast_possible_truncation)]
pub fn search_by_fts(
    conn: &Connection,
    keywords: &[&str],
    type_filter: Option<&[ChunkType]>,
    exclude_ids: &HashSet<i64>,
    include_ids: Option<&HashSet<i64>>,
    limit: u32,
    path_filter: &[String],
) -> Result<Vec<SearchResult>, StorageError> {
    if keywords.is_empty() {
        return Ok(Vec::new());
    }

    let parts: Vec<String> = keywords
        .iter()
        .filter_map(|k| match prepare_match_query(conn, k, "fts_chunks_vocab") {
            Ok(m) if !m.as_str().is_empty() => Some(m.into_string()),
            Err(e) if !k.trim().is_empty() => {
                tracing::debug!(keyword = %k, error = %e, "prepare_match_query failed, falling back to fts_quote");
                Some(fts_quote(k))
            }
            _ => None,
        })
        .collect();
    if parts.is_empty() {
        return Ok(Vec::new());
    }
    let fts_query = parts.join(" AND ");

    const BM25_EXPR: &str = "bm25(fts_chunks, 5.0, 1.0, 3.0)";

    let mut sql = format!(
        "SELECT c.id, c.file_path, c.chunk_type, c.name, c.content,
                c.start_line, c.end_line, c.parent_chunk_id,
                {BM25_EXPR}
         FROM fts_chunks f
         INNER JOIN chunks c ON c.id = f.rowid
         WHERE fts_chunks MATCH ?1",
    );

    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    params.push(Box::new(fts_query));

    append_chunk_type_filter(&mut sql, &mut params, "c.chunk_type", type_filter);
    append_exclude_ids(&mut sql, &mut params, "c.id", exclude_ids);
    append_include_ids(&mut sql, &mut params, "c.id", include_ids);
    append_like_prefix_filter(&mut sql, &mut params, "c.file_path", path_filter);
    sql.push_str(&format!(" ORDER BY {BM25_EXPR} LIMIT ?"));
    params.push(Box::new(limit));

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(params.iter()), |row| {
        let bm25_score: f64 = row.get(8)?;
        let abs = (bm25_score as f32).abs();
        let base_score = abs / (1.0 + abs);
        Ok(SearchResult {
            chunk: chunk_from_row(row, 1)?,
            chunk_id: Some(row.get(0)?),
            distance: f32::INFINITY,
            match_source: MatchSource::Fts,
            score: base_score,
        })
    })?;

    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_chunk_by_id(conn: &Connection, chunk_id: i64) -> Result<Option<Chunk>, StorageError> {
    let mut stmt = conn.prepare_cached(
        "SELECT file_path, chunk_type, name, content, start_line, end_line, parent_chunk_id
         FROM chunks WHERE id = ?1",
    )?;
    match stmt.query_row([chunk_id], |row| chunk_from_row(row, 0)) {
        Ok(chunk) => Ok(Some(chunk)),
        Err(RusqliteError::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

pub fn get_chunks_by_ids(
    conn: &Connection,
    ids: &[i64],
) -> Result<HashMap<i64, Chunk>, StorageError> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }
    let placeholders = in_placeholders(ids.len());
    let sql = format!(
        "SELECT id, file_path, chunk_type, name, content, start_line, end_line, parent_chunk_id
         FROM chunks WHERE id IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(ids.iter()), |row| {
        let id: i64 = row.get(0)?;
        let chunk = chunk_from_row(row, 1)?;
        Ok((id, chunk))
    })?;
    rows.collect::<Result<HashMap<_, _>, _>>()
        .map_err(Into::into)
}

pub fn get_keyword_doc_frequencies(
    conn: &Connection,
    keywords: &[&str],
    total_chunks: u32,
) -> Result<Vec<u32>, StorageError> {
    let mut dfs = Vec::with_capacity(keywords.len());
    for kw in keywords {
        let fts_term = fts_quote(kw);
        let count: u32 = match conn.query_row(
            "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH ?1",
            [&fts_term],
            |row| row.get(0),
        ) {
            Ok(c) => c,
            Err(e) => {
                tracing::debug!(keyword = %kw, error = %e, "FTS5 doc freq query failed, treating as neutral");
                total_chunks
            }
        };
        dfs.push(count);
    }
    Ok(dfs)
}

pub fn vec_search_multi(
    conn: &Connection,
    query_embeddings: &[&[f32]],
    limit: u32,
    type_filter: Option<&[ChunkType]>,
    path_filter: &[String],
) -> Result<Vec<SearchResult>, StorageError> {
    if query_embeddings.is_empty() {
        return Ok(Vec::new());
    }

    let mut best: HashMap<i64, SearchResult> = HashMap::new();
    for emb in query_embeddings {
        for result in vec_search(conn, emb, limit, type_filter, path_filter)? {
            if let Some(chunk_id) = result.chunk_id {
                best.entry(chunk_id)
                    .and_modify(|existing| {
                        if result.distance < existing.distance {
                            existing.distance = result.distance;
                            existing.score = result.score;
                        }
                    })
                    .or_insert(result);
            }
        }
    }

    let mut merged: Vec<SearchResult> = best.into_values().collect();
    merged.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    merged.truncate(limit as usize);
    Ok(merged)
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

pub fn get_sub_embeddings_for_chunks(
    conn: &Connection,
    chunk_ids: &[i64],
) -> Result<Vec<(i64, Vec<u8>)>, StorageError> {
    if chunk_ids.is_empty() {
        return Ok(Vec::new());
    }

    // Step 1: Get (chunk_id, vec_rowid) from embedded_chunk_ids, ordered by sub_idx.
    let placeholders = anon_placeholders(chunk_ids.len());
    let sql = format!(
        "SELECT chunk_id, vec_rowid FROM embedded_chunk_ids \
         WHERE chunk_id IN ({placeholders}) ORDER BY chunk_id, sub_idx"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mappings: Vec<(i64, i64)> = stmt
        .query_map(params_from_iter(chunk_ids.iter()), |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    if mappings.is_empty() {
        return Ok(Vec::new());
    }

    // Step 2: Batch-fetch embedding bytes (vec0 supports WHERE rowid IN (...)).
    let emb_placeholders = anon_placeholders(mappings.len());
    let emb_sql =
        format!("SELECT rowid, embedding FROM vec_chunks WHERE rowid IN ({emb_placeholders})");
    let mut by_rowid: HashMap<i64, Vec<u8>> = conn
        .prepare(&emb_sql)?
        .query_map(
            params_from_iter(mappings.iter().map(|(_, vid)| vid)),
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?
        .collect::<Result<_, _>>()?;

    let results: Vec<(i64, Vec<u8>)> = mappings
        .iter()
        .filter_map(|(chunk_id, vec_rowid)| by_rowid.remove(vec_rowid).map(|b| (*chunk_id, b)))
        .collect();
    Ok(results)
}
