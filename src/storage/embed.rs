use std::collections::HashSet;

use rurico::embed::ChunkedEmbedding;
use rusqlite::Connection;

use super::{
    ChunkType, StorageError, anon_placeholders, as_sql_params, in_placeholders,
    insert_sub_embeddings,
};

pub fn add_chunked_embeddings(
    conn: &Connection,
    embeddings: &[(i64, ChunkedEmbedding)],
) -> Result<u32, StorageError> {
    if embeddings.is_empty() {
        return Ok(0);
    }
    let chunk_ids: Vec<i64> = embeddings.iter().map(|(id, _)| *id).collect();
    let existing = existing_embedded_ids(conn, &chunk_ids)?;
    let tx = conn.unchecked_transaction()?;
    let mut count = 0u32;
    for (chunk_id, chunked_emb) in embeddings {
        if existing.contains(chunk_id) {
            continue;
        }
        insert_sub_embeddings(&tx, *chunk_id, chunked_emb)?;
        count += 1;
    }
    tx.commit()?;
    Ok(count)
}

fn existing_embedded_ids(
    conn: &Connection,
    chunk_ids: &[i64],
) -> Result<HashSet<i64>, StorageError> {
    if chunk_ids.is_empty() {
        return Ok(HashSet::new());
    }
    let sql = format!(
        "SELECT chunk_id FROM embedded_chunk_ids WHERE chunk_id IN ({})",
        in_placeholders(chunk_ids.len())
    );
    let params = as_sql_params(chunk_ids);
    let mut stmt = conn.prepare(&sql)?;
    let ids = stmt
        .query_map(params.as_slice(), |row| row.get::<_, i64>(0))?
        .collect::<Result<HashSet<_>, _>>()?;
    Ok(ids)
}

pub fn get_unembedded_chunks_for_file(
    conn: &Connection,
    file_path: &str,
) -> Result<Vec<(i64, String, String, Option<String>, Option<String>)>, StorageError> {
    let mut stmt = conn.prepare_cached(
        "SELECT c.id, c.content, c.chunk_type, c.name, p.name
         FROM chunks c
         LEFT JOIN chunks p ON c.parent_chunk_id = p.id
         LEFT JOIN embedded_chunk_ids e ON c.id = e.chunk_id
         WHERE c.file_path = ?1 AND e.chunk_id IS NULL AND c.chunk_type != 'inner_fn'",
    )?;
    let rows = stmt.query_map([file_path], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<String>>(3)?,
            row.get::<_, Option<String>>(4)?,
        ))
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_imports_for_file(conn: &Connection, file_path: &str) -> Result<String, StorageError> {
    match conn.query_row(
        "SELECT imports_text FROM file_context WHERE file_path = ?1",
        [file_path],
        |row| row.get::<_, String>(0),
    ) {
        Ok(text) => Ok(text),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(String::new()),
        Err(e) => Err(e.into()),
    }
}

pub fn get_files_with_chunk_types(
    conn: &Connection,
    files: &[String],
    types: &[ChunkType],
) -> Result<HashSet<String>, StorageError> {
    if files.is_empty() || types.is_empty() {
        return Ok(HashSet::new());
    }
    let type_ph = anon_placeholders(types.len());
    let file_ph = anon_placeholders(files.len());
    let sql = format!(
        "SELECT DISTINCT file_path FROM chunks WHERE chunk_type IN ({type_ph}) AND file_path IN ({file_ph})"
    );
    let params: Vec<String> = types
        .iter()
        .map(|t| t.as_str().to_string())
        .chain(files.iter().cloned())
        .collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(as_sql_params(&params).as_slice(), |row| {
        row.get::<_, String>(0)
    })?;
    rows.collect::<Result<HashSet<_>, _>>().map_err(Into::into)
}

pub fn has_embeddings(conn: &Connection) -> bool {
    conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM embedded_chunk_ids)",
        [],
        |row| row.get(0),
    )
    .map_err(|e| {
        tracing::warn!(error = %e, "has_embeddings query failed, assuming no embeddings");
        e
    })
    .unwrap_or(false)
}

pub fn should_reindex(
    conn: &Connection,
    file_path: &str,
    current_hash: &str,
) -> Result<bool, StorageError> {
    match stored_hash_for_file(conn, file_path)? {
        None => Ok(true),
        Some(h) => Ok(h != current_hash),
    }
}

fn stored_hash_for_file(
    conn: &Connection,
    file_path: &str,
) -> Result<Option<String>, StorageError> {
    match conn.query_row(
        "SELECT file_hash FROM chunks WHERE file_path = ?1 LIMIT 1",
        [file_path],
        |row| row.get::<_, String>(0),
    ) {
        Ok(h) => Ok(Some(h)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
pub fn get_unembedded_file_paths(conn: &Connection) -> Result<Vec<(String, u32)>, StorageError> {
    let mut stmt = conn.prepare_cached(
        "SELECT c.file_path, COUNT(*) as chunk_count
         FROM chunks c
         LEFT JOIN embedded_chunk_ids e ON c.id = e.chunk_id
         WHERE e.chunk_id IS NULL AND c.chunk_type != 'inner_fn'
         GROUP BY c.file_path",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, u32>(1)?))
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

#[cfg(test)]
pub fn needs_embedding(
    conn: &Connection,
    file_path: &str,
    current_hash: &str,
) -> Result<bool, StorageError> {
    match stored_hash_for_file(conn, file_path)? {
        None => Ok(true),
        Some(h) if h != current_hash => Ok(true),
        Some(_) => {
            let has_unembedded: bool = conn.query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM chunks c
                    LEFT JOIN embedded_chunk_ids e ON c.id = e.chunk_id
                    WHERE c.file_path = ?1 AND e.chunk_id IS NULL
                )",
                [file_path],
                |row| row.get(0),
            )?;
            Ok(has_unembedded)
        }
    }
}
