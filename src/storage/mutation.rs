use rusqlite::Connection;

use rurico::embed::ChunkedEmbedding;
use rurico::storage::f32_as_bytes;

use crate::text::split_identifier;

use super::{ChunkType, FileData, NewChunk, Reference, StorageError};

pub(crate) fn insert_chunk_row(
    conn: &Connection,
    file_path: &str,
    chunk: &NewChunk,
    file_hash: &str,
    parent_chunk_id: Option<i64>,
) -> Result<i64, StorageError> {
    debug_assert!(chunk.start_line <= chunk.end_line, "start_line > end_line");
    debug_assert!(!chunk.content.is_empty(), "empty chunk content");
    conn.prepare_cached(
        "INSERT INTO chunks (file_path, chunk_type, name, content, start_line, end_line, file_hash, parent_chunk_id)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
    )?
    .execute(rusqlite::params![
        file_path,
        chunk.chunk_type.as_str(),
        chunk.name,
        chunk.content,
        chunk.start_line,
        chunk.end_line,
        file_hash,
        parent_chunk_id,
    ])?;
    let chunk_id = conn.last_insert_rowid();

    let fts_name = chunk
        .name
        .map(|n| split_identifier(n).join(" "))
        .unwrap_or_default();
    conn.prepare_cached(
        "INSERT INTO fts_chunks(rowid, name, content, file_path) VALUES (?1, ?2, ?3, ?4)",
    )?
    .execute(rusqlite::params![
        chunk_id,
        fts_name,
        chunk.content,
        file_path
    ])?;

    Ok(chunk_id)
}

pub(super) fn insert_sub_embeddings(
    conn: &Connection,
    chunk_id: i64,
    chunked_emb: &ChunkedEmbedding,
) -> Result<(), StorageError> {
    for (sub_idx, emb_slice) in chunked_emb.chunks.iter().enumerate() {
        let bytes = f32_as_bytes(emb_slice);
        conn.prepare_cached(
            "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, ?2, ?3)",
        )?
        .execute(rusqlite::params![bytes, chunk_id, sub_idx as i64])?;
        let vec_rowid = conn.last_insert_rowid();
        conn.prepare_cached(
            "INSERT INTO embedded_chunk_ids (chunk_id, sub_idx, vec_rowid) VALUES (?1, ?2, ?3)",
        )?
        .execute(rusqlite::params![chunk_id, sub_idx as i64, vec_rowid])?;
    }
    Ok(())
}

pub fn insert_chunk(
    conn: &Connection,
    file_path: &str,
    chunk: &NewChunk,
    file_hash: &str,
    embedding: &ChunkedEmbedding,
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
        tx.prepare_cached(
            "INSERT INTO file_references (source_file, target_file, symbol_name, ref_kind)
             VALUES (?1, ?2, ?3, ?4)",
        )?
        .execute(rusqlite::params![
            r.source_file,
            r.target_file,
            r.symbol_name,
            r.ref_kind.as_str()
        ])?;
    }

    tx.execute(
        "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('last_indexed_at', datetime('now'))",
        [],
    )?;
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

/// Must be called within a transaction.
pub(crate) fn delete_file_chunks_in(
    conn: &Connection,
    file_path: &str,
) -> Result<(), StorageError> {
    conn.execute(
        "WITH ids AS (SELECT id FROM chunks WHERE file_path = ?1)
         DELETE FROM fts_chunks WHERE rowid IN (SELECT id FROM ids)",
        [file_path],
    )?;
    conn.execute(
        "WITH ids AS (SELECT id FROM chunks WHERE file_path = ?1)
         DELETE FROM vec_chunks WHERE rowid IN \
         (SELECT vec_rowid FROM embedded_chunk_ids WHERE chunk_id IN (SELECT id FROM ids))",
        [file_path],
    )?;
    conn.execute(
        "WITH ids AS (SELECT id FROM chunks WHERE file_path = ?1)
         DELETE FROM embedded_chunk_ids WHERE chunk_id IN (SELECT id FROM ids)",
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

pub(crate) fn replace_file_data(
    conn: &Connection,
    data: &FileData,
    embeddings: Option<&[ChunkedEmbedding]>,
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

pub fn replace_file_chunks_with(
    conn: &Connection,
    data: &FileData,
    embeddings: &[ChunkedEmbedding],
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
    embeddings: &[ChunkedEmbedding],
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

#[cfg(test)]
pub fn delete_file_chunks(conn: &Connection, file_path: &str) -> Result<(), StorageError> {
    let tx = conn.unchecked_transaction()?;
    delete_file_chunks_in(&tx, file_path)?;
    tx.commit()?;
    Ok(())
}
