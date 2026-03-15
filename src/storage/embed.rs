use std::collections::HashSet;

use rusqlite::Connection;

use super::{ChunkType, StorageError, check_embedding_dims, f32_as_bytes, sql_placeholders};

pub fn add_embeddings(
    conn: &Connection,
    embeddings: &[(i64, Vec<f32>)],
) -> Result<u32, StorageError> {
    if embeddings.is_empty() {
        return Ok(0);
    }

    // vec0 virtual tables don't support INSERT OR IGNORE, so batch-fetch existing IDs.
    let ph = sql_placeholders(embeddings.len());
    let sql = format!("SELECT chunk_id FROM vec_chunks WHERE chunk_id IN ({ph})");
    let params: Vec<Box<dyn rusqlite::types::ToSql>> = embeddings
        .iter()
        .map(|(id, _)| Box::new(*id) as Box<dyn rusqlite::types::ToSql>)
        .collect();
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|b| b.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let existing: HashSet<i64> = stmt
        .query_map(param_refs.as_slice(), |row| row.get::<_, i64>(0))?
        .filter_map(|r| r.ok())
        .collect();

    let tx = conn.unchecked_transaction()?;
    let mut inserted = 0u32;

    for (chunk_id, embedding) in embeddings {
        if existing.contains(chunk_id) {
            continue;
        }
        check_embedding_dims(embedding)?;
        let bytes = f32_as_bytes(embedding);
        tx.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
            rusqlite::params![chunk_id, bytes],
        )?;
        inserted += 1;
    }

    tx.commit()?;
    Ok(inserted)
}

pub fn get_unembedded_file_paths(conn: &Connection) -> Result<Vec<(String, u32)>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT c.file_path, COUNT(*) as chunk_count
         FROM chunks c
         LEFT JOIN vec_chunks v ON c.id = v.chunk_id
         WHERE v.chunk_id IS NULL
         GROUP BY c.file_path",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, u32>(1)?))
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_unembedded_chunks_for_file(
    conn: &Connection,
    file_path: &str,
) -> Result<Vec<(i64, String)>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT c.id, c.content FROM chunks c
         LEFT JOIN vec_chunks v ON c.id = v.chunk_id
         WHERE c.file_path = ?1 AND v.chunk_id IS NULL",
    )?;
    let rows = stmt.query_map([file_path], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_files_with_chunk_types(
    conn: &Connection,
    files: &[String],
    types: &[ChunkType],
) -> Result<HashSet<String>, StorageError> {
    if files.is_empty() || types.is_empty() {
        return Ok(HashSet::new());
    }
    let type_ph = sql_placeholders(types.len());
    let file_ph = sql_placeholders(files.len());
    let sql = format!(
        "SELECT DISTINCT file_path FROM chunks WHERE chunk_type IN ({type_ph}) AND file_path IN ({file_ph})"
    );
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> =
        Vec::with_capacity(types.len() + files.len());
    for t in types {
        params.push(Box::new(t.as_str().to_string()));
    }
    for f in files {
        params.push(Box::new(f.clone()));
    }
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|b| b.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(param_refs.as_slice(), |row| row.get::<_, String>(0))?;
    rows.collect::<Result<HashSet<_>, _>>().map_err(Into::into)
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
                    LEFT JOIN vec_chunks v ON c.id = v.chunk_id
                    WHERE c.file_path = ?1 AND v.chunk_id IS NULL
                )",
                [file_path],
                |row| row.get(0),
            )?;
            Ok(has_unembedded)
        }
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
