mod embed;
mod graph;
mod mutation;
mod schema;
mod search;
mod types;

pub use embed::*;
pub use graph::*;
#[cfg(test)]
pub(crate) use mutation::insert_chunk_row;
pub(crate) use mutation::replace_file_data;
pub use mutation::*;
pub use schema::*;
pub use search::*;
pub use types::*;

use std::collections::HashSet;

use rusqlite::Connection;

pub type Db = Connection;

pub use rurico::embed::EMBEDDING_DIMS;
pub use rurico::storage::f32_as_bytes;

pub(crate) use amici::storage::{anon_placeholders, as_sql_params, in_placeholders};

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
    let mut stmt = conn.prepare_cached("SELECT DISTINCT file_path FROM chunks")?;
    let paths = stmt.query_map([], |row| row.get::<_, String>(0))?;
    paths.collect::<Result<HashSet<_>, _>>().map_err(Into::into)
}

#[cfg(test)]
use rurico::embed::ChunkedEmbedding;

#[cfg(test)]
pub(crate) fn ce(v: Vec<f32>) -> ChunkedEmbedding {
    ChunkedEmbedding { chunks: vec![v] }
}

#[cfg(test)]
mod tests;
