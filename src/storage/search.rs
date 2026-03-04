use std::collections::HashSet;

use rusqlite::Connection;

use super::{
    f32_as_bytes, sql_placeholders, Chunk, ChunkType, MatchSource, SearchResult, StorageError,
};

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
                c.start_line, c.end_line, v.distance, c.id
         FROM vec_chunks v
         INNER JOIN chunks c ON c.id = v.chunk_id
         WHERE v.embedding MATCH ?1 AND k = ?2
         ORDER BY v.distance
         LIMIT ?3
         OFFSET ?4",
    )?;

    let rows = stmt.query_map(rusqlite::params![query_bytes, k, limit, offset], |row| {
        let distance: f32 = row.get(6)?;
        Ok(SearchResult {
            chunk: Chunk {
                file_path: row.get(0)?,
                chunk_type: ChunkType::from_db(row.get::<_, String>(1)?.as_ref()),
                name: row.get(2)?,
                content: row.get(3)?,
                start_line: row.get(4)?,
                end_line: row.get(5)?,
            },
            chunk_id: Some(row.get(7)?),
            distance,
            match_source: MatchSource::Semantic,
            score: 1.0 / (1.0 + distance),
        })
    })?;

    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn search_by_name(
    conn: &Connection,
    keywords: &[&str],
    type_filter: Option<&[ChunkType]>,
    exclude_ids: &HashSet<i64>,
    limit: u32,
) -> Result<Vec<SearchResult>, StorageError> {
    if keywords.is_empty() {
        return Ok(Vec::new());
    }

    let name_clause = vec!["name LIKE ? ESCAPE '\\'"; keywords.len()].join(" OR ");
    let path_clause = vec!["file_path LIKE ? ESCAPE '\\'"; keywords.len()].join(" OR ");

    let mut sql = format!(
        "SELECT id, file_path, chunk_type, name, content, start_line, end_line \
         FROM chunks WHERE ((name IS NOT NULL AND ({name_clause})) OR ({path_clause}))",
    );

    let mut all_params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    for k in keywords {
        all_params.push(Box::new(format!("%{}%", escape_like(k))) as Box<dyn rusqlite::types::ToSql>);
    }
    for k in keywords {
        all_params.push(Box::new(format!("%{}%", escape_like(k))) as Box<dyn rusqlite::types::ToSql>);
    }

    append_type_filter(&mut sql, &mut all_params, "chunk_type", type_filter);
    append_exclude_ids(&mut sql, &mut all_params, "id", exclude_ids);
    sql.push_str(" LIMIT ?");
    all_params.push(Box::new(limit));

    let mut stmt = conn.prepare(&sql)?;
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = all_params
        .iter()
        .map(|b| b.as_ref())
        .collect();

    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        Ok(SearchResult {
            chunk: Chunk {
                file_path: row.get(1)?,
                chunk_type: ChunkType::from_db(row.get::<_, String>(2)?.as_ref()),
                name: row.get(3)?,
                content: row.get(4)?,
                start_line: row.get(5)?,
                end_line: row.get(6)?,
            },
            chunk_id: Some(row.get(0)?),
            distance: f32::INFINITY,
            match_source: MatchSource::NameMatch,
            score: 0.5,
        })
    })?;

    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

/// Keywords are double-quote escaped to prevent FTS5 syntax injection.
pub fn search_by_content(
    conn: &Connection,
    keywords: &[&str],
    type_filter: Option<&[ChunkType]>,
    exclude_ids: &HashSet<i64>,
    limit: u32,
) -> Result<Vec<SearchResult>, StorageError> {
    if keywords.is_empty() {
        return Ok(Vec::new());
    }

    let fts_query: String = keywords
        .iter()
        .map(|k| format!("\"{}\"", k.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" OR ");

    let mut sql = String::from(
        "SELECT c.id, c.file_path, c.chunk_type, c.name, c.content,
                c.start_line, c.end_line
         FROM fts_chunks f
         INNER JOIN chunks c ON c.id = f.rowid
         WHERE fts_chunks MATCH ?1",
    );

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    params.push(Box::new(fts_query));

    append_type_filter(&mut sql, &mut params, "c.chunk_type", type_filter);
    append_exclude_ids(&mut sql, &mut params, "c.id", exclude_ids);
    sql.push_str(" ORDER BY rank LIMIT ?");
    params.push(Box::new(limit));

    let mut stmt = conn.prepare(&sql)?;
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        params.iter().map(|b| b.as_ref()).collect();

    let rows = stmt.query_map(param_refs.as_slice(), |row| {
        Ok(SearchResult {
            chunk: Chunk {
                file_path: row.get(1)?,
                chunk_type: ChunkType::from_db(row.get::<_, String>(2)?.as_ref()),
                name: row.get(3)?,
                content: row.get(4)?,
                start_line: row.get(5)?,
                end_line: row.get(6)?,
            },
            chunk_id: Some(row.get(0)?),
            distance: f32::INFINITY,
            match_source: MatchSource::ContentMatch,
            score: 0.45,
        })
    })?;

    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_keyword_doc_frequencies(
    conn: &Connection,
    keywords: &[&str],
) -> Result<Vec<u32>, StorageError> {
    let mut dfs = Vec::with_capacity(keywords.len());
    for kw in keywords {
        let fts_term = format!("\"{}\"", kw.replace('"', "\"\""));
        let count: u32 = match conn.query_row(
            "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH ?1",
            [&fts_term],
            |row| row.get(0),
        ) {
            Ok(c) => c,
            Err(e) => {
                tracing::debug!(keyword = %kw, error = %e, "FTS5 doc freq query failed, treating as 0");
                0
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
        sql.push_str(&format!(" AND {column} IN ({})", sql_placeholders(types.len())));
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
        sql.push_str(&format!(" AND {column} NOT IN ({})", sql_placeholders(exclude_ids.len())));
        for id in exclude_ids {
            params.push(Box::new(*id));
        }
    }
}

fn escape_like(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len());
    for c in s.chars() {
        if matches!(c, '%' | '_' | '\\') {
            escaped.push('\\');
        }
        escaped.push(c);
    }
    escaped
}
