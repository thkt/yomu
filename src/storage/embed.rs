use std::collections::HashSet;

use rurico::embed::ChunkedEmbedding;
use rusqlite::Connection;

use super::{
    ChunkType, StorageError, anon_placeholders, as_sql_params, collect_rows, embeddable_predicate,
    fetch_by_in_clause, insert_sub_embeddings,
};

fn existing_embedded_ids(
    conn: &Connection,
    chunk_ids: &[i64],
) -> Result<HashSet<i64>, StorageError> {
    fetch_by_in_clause(
        conn,
        chunk_ids,
        "SELECT chunk_id FROM embedded_chunk_ids WHERE chunk_id IN ({placeholders})",
        |row| row.get(0),
    )
}

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

pub struct UnembeddedChunk {
    pub id: i64,
    pub content: String,
    pub chunk_type: String,
    pub name: Option<String>,
    pub parent_name: Option<String>,
}

pub fn get_unembedded_chunks_for_file(
    conn: &Connection,
    file_path: &str,
) -> Result<Vec<UnembeddedChunk>, StorageError> {
    let mut stmt = conn.prepare_cached(&format!(
        "SELECT c.id, c.content, c.chunk_type, c.name, p.name
         FROM chunks c
         LEFT JOIN chunks p ON c.parent_chunk_id = p.id
         LEFT JOIN embedded_chunk_ids e ON c.id = e.chunk_id
         WHERE c.file_path = ?1 AND e.chunk_id IS NULL AND {}",
        embeddable_predicate("c")
    ))?;
    let rows = stmt.query_map([file_path], |row| {
        Ok(UnembeddedChunk {
            id: row.get(0)?,
            content: row.get(1)?,
            chunk_type: row.get(2)?,
            name: row.get(3)?,
            parent_name: row.get(4)?,
        })
    })?;
    collect_rows(rows)
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
        .map(|t| t.as_str().to_owned())
        .chain(files.iter().cloned())
        .collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(as_sql_params(&params).as_slice(), |row| {
        row.get::<_, String>(0)
    })?;
    collect_rows(rows)
}

/// Counts chunks on the embed worklist ([`embeddable_predicate`], the same
/// predicate as [`get_unembedded_chunks_for_file`]) that have no embedding. A
/// non-zero gap means indexing died mid-embed (#288): FTS rows are written
/// synchronously at chunk time, so the index looks complete while vector
/// search runs against a partial candidate space. `yomu index` re-runs fill
/// the gap incrementally.
pub fn embed_gap_count(conn: &Connection) -> Result<u32, StorageError> {
    conn.query_row(
        &format!(
            "SELECT COUNT(*)
             FROM chunks c
             LEFT JOIN embedded_chunk_ids e ON c.id = e.chunk_id
             WHERE e.chunk_id IS NULL AND {}",
            embeddable_predicate("c")
        ),
        [],
        |row| row.get(0),
    )
    .map_err(Into::into)
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
    collect_rows(rows)
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
