use std::collections::{HashMap, HashSet};

use rusqlite::Connection;

use super::{
    Chunk, ChunkType, MatchSource, SearchResult, StorageError, f32_as_bytes, sql_placeholders,
};

fn chunk_from_row(row: &rusqlite::Row<'_>, offset: usize) -> rusqlite::Result<Chunk> {
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

pub fn search_similar(
    conn: &Connection,
    query_embedding: &[f32],
    limit: u32,
    offset: u32,
) -> Result<Vec<SearchResult>, StorageError> {
    let query_bytes = f32_as_bytes(query_embedding);

    let k = limit.saturating_add(offset);

    let mut stmt = conn.prepare(
        "SELECT c.file_path, c.chunk_type, c.name, c.content,
                c.start_line, c.end_line, c.parent_chunk_id, v.distance, c.id
         FROM vec_chunks v
         INNER JOIN chunks c ON c.id = v.chunk_id
         WHERE v.embedding MATCH ?1 AND k = ?2
               AND v.distance >= 0.0 -- filters NULL rows; vec0 disallows IS NOT NULL in KNN queries
         ORDER BY v.distance
         LIMIT ?3
         OFFSET ?4",
    )?;

    let rows = stmt.query_map(rusqlite::params![query_bytes, k, limit, offset], |row| {
        let distance: f32 = row.get(7)?;
        Ok(SearchResult {
            chunk: chunk_from_row(row, 0)?,
            chunk_id: Some(row.get(8)?),
            distance,
            match_source: MatchSource::Semantic,
            score: 1.0 / (1.0 + distance),
        })
    })?;

    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn search_by_fts(
    conn: &Connection,
    keywords: &[&str],
    type_filter: Option<&[ChunkType]>,
    exclude_ids: &HashSet<i64>,
    limit: u32,
) -> Result<Vec<SearchResult>, StorageError> {
    if keywords.is_empty() {
        return Ok(Vec::new());
    }

    let parts: Vec<String> = keywords
        .iter()
        .filter_map(|k| {
            match rurico::storage::prepare_match_query(conn, k) {
                Ok(m) if !m.as_str().is_empty() => Some(m.into_string()),
                Err(_) if !k.trim().is_empty() => Some(rurico::storage::fts_quote(k)),
                _ => None,
            }
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

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    params.push(Box::new(fts_query));

    append_type_filter(&mut sql, &mut params, "c.chunk_type", type_filter);
    append_exclude_ids(&mut sql, &mut params, "c.id", exclude_ids);
    sql.push_str(&format!(" ORDER BY {BM25_EXPR} LIMIT ?"));
    params.push(Box::new(limit));

    let mut stmt = conn.prepare(&sql)?;
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|b| b.as_ref()).collect();

    let rows = stmt.query_map(param_refs.as_slice(), |row| {
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
    let mut stmt = conn.prepare(
        "SELECT file_path, chunk_type, name, content, start_line, end_line, parent_chunk_id
         FROM chunks WHERE id = ?1",
    )?;
    match stmt.query_row([chunk_id], |row| chunk_from_row(row, 0)) {
        Ok(chunk) => Ok(Some(chunk)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
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
    let placeholders = super::sql_placeholders(ids.len());
    let sql = format!(
        "SELECT id, file_path, chunk_type, name, content, start_line, end_line, parent_chunk_id
         FROM chunks WHERE id IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> = ids
        .iter()
        .map(|id| id as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = stmt.query_map(params.as_slice(), |row| {
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
        let fts_term = rurico::storage::fts_quote(kw);
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

fn append_type_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn rusqlite::types::ToSql>>,
    column: &str,
    types: Option<&[ChunkType]>,
) {
    if let Some(types) = types
        && !types.is_empty()
    {
        sql.push_str(&format!(
            " AND {column} IN ({})",
            sql_placeholders(types.len())
        ));
        for t in types {
            params.push(Box::new(t.as_str().to_string()));
        }
    }
}

fn append_exclude_ids(
    sql: &mut String,
    params: &mut Vec<Box<dyn rusqlite::types::ToSql>>,
    column: &str,
    exclude_ids: &HashSet<i64>,
) {
    if !exclude_ids.is_empty() {
        sql.push_str(&format!(
            " AND {column} NOT IN ({})",
            sql_placeholders(exclude_ids.len())
        ));
        for id in exclude_ids {
            params.push(Box::new(*id));
        }
    }
}

