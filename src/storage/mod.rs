mod embed;
mod graph;
mod schema;
mod search;
mod types;

pub use embed::*;
pub use graph::*;
pub use schema::*;
pub use search::*;
pub use types::*;

use std::collections::HashSet;

use rusqlite::Connection;

pub type Db = Connection;

pub use rurico::embed::EMBEDDING_DIMS;
pub use rurico::storage::f32_as_bytes;

pub(crate) fn in_placeholders(len: usize) -> String {
    (1..=len)
        .map(|i| format!("?{i}"))
        .collect::<Vec<_>>()
        .join(", ")
}

// Use when multiple IN clauses share a parameter list: unnamed `?` avoids
// index collisions that numbered `?N` would cause when params are appended incrementally.
pub(crate) fn anon_placeholders(n: usize) -> String {
    vec!["?"; n].join(", ")
}

pub(crate) fn as_sql_params<T: rusqlite::types::ToSql>(
    values: &[T],
) -> Vec<&dyn rusqlite::types::ToSql> {
    values.iter().map(|v| v as &dyn rusqlite::types::ToSql).collect()
}

fn insert_chunk_row(
    conn: &Connection,
    file_path: &str,
    chunk: &NewChunk,
    file_hash: &str,
    parent_chunk_id: Option<i64>,
) -> Result<i64, StorageError> {
    debug_assert!(chunk.start_line <= chunk.end_line, "start_line > end_line");
    debug_assert!(!chunk.content.is_empty(), "empty chunk content");
    conn.execute(
        "INSERT INTO chunks (file_path, chunk_type, name, content, start_line, end_line, file_hash, parent_chunk_id)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        rusqlite::params![
            file_path,
            chunk.chunk_type.as_str(),
            chunk.name,
            chunk.content,
            chunk.start_line,
            chunk.end_line,
            file_hash,
            parent_chunk_id,
        ],
    )?;
    let chunk_id = conn.last_insert_rowid();

    let fts_name = chunk
        .name
        .map(|n| crate::text::split_identifier(n).join(" "))
        .unwrap_or_default();
    conn.execute(
        "INSERT INTO fts_chunks(rowid, name, content, file_path) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![chunk_id, fts_name, chunk.content, file_path],
    )?;

    Ok(chunk_id)
}

pub(crate) fn insert_sub_embeddings(
    conn: &Connection,
    chunk_id: i64,
    chunked_emb: &rurico::embed::ChunkedEmbedding,
) -> Result<(), StorageError> {
    for (sub_idx, emb_slice) in chunked_emb.chunks.iter().enumerate() {
        let bytes = f32_as_bytes(emb_slice);
        conn.execute(
            "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, ?2, ?3)",
            rusqlite::params![bytes, chunk_id, sub_idx as i64],
        )?;
        let vec_rowid = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO embedded_chunk_ids (chunk_id, sub_idx, vec_rowid) VALUES (?1, ?2, ?3)",
            rusqlite::params![chunk_id, sub_idx as i64, vec_rowid],
        )?;
    }
    Ok(())
}

pub fn insert_chunk(
    conn: &Connection,
    file_path: &str,
    chunk: &NewChunk,
    file_hash: &str,
    embedding: &rurico::embed::ChunkedEmbedding,
    parent_chunk_id: Option<i64>,
) -> Result<i64, StorageError> {
    let chunk_id = insert_chunk_row(conn, file_path, chunk, file_hash, parent_chunk_id)?;
    insert_sub_embeddings(conn, chunk_id, embedding)?;
    Ok(chunk_id)
}

fn write_file_metadata(
    tx: &Connection,
    file_path: &str,
    imports_text: &str,
    refs: &[Reference],
    mtime_epoch: Option<i64>,
) -> Result<(), StorageError> {
    tx.execute(
        "INSERT OR REPLACE INTO file_context (file_path, imports_text, mtime_epoch) VALUES (?1, ?2, ?3)",
        rusqlite::params![file_path, imports_text, mtime_epoch],
    )?;

    for r in refs {
        tx.execute(
            "INSERT INTO file_references (source_file, target_file, symbol_name, ref_kind)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                r.source_file,
                r.target_file,
                r.symbol_name,
                r.ref_kind.as_str()
            ],
        )?;
    }

    tx.execute(
        "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('last_indexed_at', datetime('now'))",
        [],
    )?;
    Ok(())
}

pub fn replace_file_chunks_with(
    conn: &Connection,
    data: &FileData,
    embeddings: &[rurico::embed::ChunkedEmbedding],
) -> Result<(), StorageError> {
    let embeddable_count = data
        .chunks
        .iter()
        .filter(|c| *c.chunk_type != ChunkType::InnerFn)
        .count();
    if embeddable_count != embeddings.len() {
        return Err(StorageError::LengthMismatch {
            chunks: embeddable_count,
            embeddings: embeddings.len(),
        });
    }
    replace_file_data(conn, data, Some(embeddings))
}

#[cfg(test)]
pub fn replace_file_chunks(
    conn: &Connection,
    file_path: &str,
    chunks: &[NewChunk],
    embeddings: &[rurico::embed::ChunkedEmbedding],
    file_hash: &str,
    imports_text: &str,
    refs: &[Reference],
) -> Result<(), StorageError> {
    let data = FileData {
        file_path,
        chunks,
        file_hash,
        imports_text,
        refs,
        mtime_epoch: None,
    };
    replace_file_chunks_with(conn, &data, embeddings)
}

#[cfg(test)]
pub fn replace_file_chunks_only(
    conn: &Connection,
    file_path: &str,
    chunks: &[NewChunk],
    file_hash: &str,
    imports_text: &str,
    refs: &[Reference],
    mtime_epoch: Option<i64>,
) -> Result<(), StorageError> {
    let data = FileData {
        file_path,
        chunks,
        file_hash,
        imports_text,
        refs,
        mtime_epoch,
    };
    replace_file_data(conn, &data, None)
}

pub(crate) fn replace_file_data(
    conn: &Connection,
    data: &FileData,
    embeddings: Option<&[rurico::embed::ChunkedEmbedding]>,
) -> Result<(), StorageError> {
    let tx = conn.unchecked_transaction()?;
    delete_file_chunks_in(&tx, data.file_path)?;

    let mut inserted_ids: Vec<i64> = Vec::with_capacity(data.chunks.len());
    let mut emb_idx = 0;

    for (i, chunk) in data.chunks.iter().enumerate() {
        let parent_chunk_id = resolve_parent(chunk, &inserted_ids, i);
        let is_embeddable = *chunk.chunk_type != ChunkType::InnerFn;
        let id = if is_embeddable && let Some(embs) = embeddings {
            let id = insert_chunk(
                &tx,
                data.file_path,
                chunk,
                data.file_hash,
                &embs[emb_idx],
                parent_chunk_id,
            )?;
            emb_idx += 1;
            id
        } else {
            insert_chunk_row(&tx, data.file_path, chunk, data.file_hash, parent_chunk_id)?
        };
        inserted_ids.push(id);
    }

    write_file_metadata(
        &tx,
        data.file_path,
        data.imports_text,
        data.refs,
        data.mtime_epoch,
    )?;
    tx.commit()?;
    Ok(())
}

fn resolve_parent(chunk: &NewChunk, inserted_ids: &[i64], current_index: usize) -> Option<i64> {
    chunk.parent_index.and_then(|pi| {
        if pi < current_index {
            inserted_ids.get(pi).copied()
        } else {
            tracing::warn!(
                parent_index = pi,
                current_index,
                "invalid parent_index, skipping"
            );
            None
        }
    })
}

pub fn file_exists_in_index(conn: &Connection, file_path: &str) -> Result<bool, StorageError> {
    let count: u32 = conn.query_row(
        "SELECT COUNT(*) FROM chunks WHERE file_path = ?1",
        [file_path],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

pub fn get_stats(conn: &Connection) -> Result<IndexStatus, StorageError> {
    let total_chunks: u32 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))?;

    let embeddable_chunks: u32 = conn.query_row(
        "SELECT COUNT(*) FROM chunks WHERE chunk_type != 'inner_fn'",
        [],
        |row| row.get(0),
    )?;

    let total_files: u32 =
        conn.query_row("SELECT COUNT(DISTINCT file_path) FROM chunks", [], |row| {
            row.get(0)
        })?;

    let embedded_chunks: u32 = conn.query_row(
        "SELECT COUNT(DISTINCT chunk_id) FROM embedded_chunk_ids",
        [],
        |row| row.get(0),
    )?;

    let last_indexed_at: Option<String> = match conn.query_row(
        "SELECT value FROM index_meta WHERE key = 'last_indexed_at'",
        [],
        |row| row.get(0),
    ) {
        Ok(val) => Some(val),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(e) => return Err(e.into()),
    };

    Ok(IndexStatus {
        total_files,
        total_chunks,
        embeddable_chunks,
        embedded_chunks,
        last_indexed_at,
    })
}

pub fn is_index_fresh(conn: &Connection, max_age_secs: u32) -> Result<bool, StorageError> {
    match conn.query_row(
        "SELECT strftime('%s', 'now') - strftime('%s', value) < ?1 FROM index_meta WHERE key = 'last_indexed_at'",
        [max_age_secs],
        |row| row.get(0),
    ) {
        Ok(fresh) => Ok(fresh),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(false),
        Err(e) => Err(e.into()),
    }
}

pub fn get_all_file_paths(conn: &Connection) -> Result<HashSet<String>, StorageError> {
    let mut stmt = conn.prepare("SELECT DISTINCT file_path FROM chunks")?;
    let paths = stmt.query_map([], |row| row.get::<_, String>(0))?;
    paths.collect::<Result<HashSet<_>, _>>().map_err(Into::into)
}

/// Must be called within a transaction.
pub fn delete_file_chunks_in(conn: &Connection, file_path: &str) -> Result<(), StorageError> {
    conn.execute(
        "DELETE FROM fts_chunks WHERE rowid IN (SELECT id FROM chunks WHERE file_path = ?1)",
        [file_path],
    )?;
    conn.execute(
        "DELETE FROM vec_chunks WHERE rowid IN \
         (SELECT vec_rowid FROM embedded_chunk_ids \
          WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?1))",
        [file_path],
    )?;
    conn.execute(
        "DELETE FROM embedded_chunk_ids \
         WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?1)",
        [file_path],
    )?;
    conn.execute("DELETE FROM chunks WHERE file_path = ?1", [file_path])?;
    conn.execute("DELETE FROM file_context WHERE file_path = ?1", [file_path])?;
    conn.execute(
        "DELETE FROM file_references WHERE source_file = ?1",
        [file_path],
    )?;
    Ok(())
}

#[cfg(test)]
pub fn delete_file_chunks(conn: &Connection, file_path: &str) -> Result<(), StorageError> {
    let tx = conn.unchecked_transaction()?;
    delete_file_chunks_in(&tx, file_path)?;
    tx.commit()?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn ce(v: Vec<f32>) -> rurico::embed::ChunkedEmbedding {
    rurico::embed::ChunkedEmbedding { chunks: vec![v] }
}

#[cfg(test)]
mod tests;
