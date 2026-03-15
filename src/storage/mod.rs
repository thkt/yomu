mod embed;
mod graph;
mod schema;
mod search;

pub use embed::*;
pub use graph::*;
pub use schema::*;
pub use search::*;

use std::collections::HashSet;

use rusqlite::Connection;

pub type Db = Connection;

pub const EMBEDDING_DIMS: u32 = 768;

#[cfg(not(target_endian = "little"))]
compile_error!("yomu requires a little-endian target for f32↔u8 embedding storage");

pub(crate) fn f32_as_bytes(slice: &[f32]) -> &[u8] {
    bytemuck::cast_slice(slice)
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    Component,
    Hook,
    TypeDef,
    CssRule,
    HtmlElement,
    TestCase,
    Other,
}

impl ChunkType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Component => "component",
            Self::Hook => "hook",
            Self::TypeDef => "type_def",
            Self::CssRule => "css_rule",
            Self::HtmlElement => "html_element",
            Self::TestCase => "test_case",
            Self::Other => "other",
        }
    }

    pub fn from_db(s: &str) -> Self {
        match s {
            "component" => Self::Component,
            "hook" => Self::Hook,
            "type_def" => Self::TypeDef,
            "css_rule" => Self::CssRule,
            "html_element" => Self::HtmlElement,
            "test_case" => Self::TestCase,
            "other" => Self::Other,
            other => {
                tracing::warn!(
                    chunk_type = other,
                    "Unknown chunk_type in DB, defaulting to Other"
                );
                Self::Other
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub file_path: String,
    pub chunk_type: ChunkType,
    pub name: Option<String>,
    pub content: String,
    pub start_line: u32,
    pub end_line: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexStatus {
    pub total_files: u32,
    pub total_chunks: u32,
    pub embedded_chunks: u32,
    pub last_indexed_at: Option<String>,
}

impl IndexStatus {
    pub fn embed_percentage(&self) -> u32 {
        if self.total_chunks > 0 {
            (self.embedded_chunks as f64 / self.total_chunks as f64 * 100.0) as u32
        } else {
            0
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchSource {
    Semantic,
    NameMatch,
    ContentMatch,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub chunk_id: Option<i64>,
    pub distance: f32,
    pub match_source: MatchSource,
    /// Reranked score (higher = better).
    pub score: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("chunks/embeddings length mismatch: {chunks} chunks vs {embeddings} embeddings")]
    LengthMismatch { chunks: usize, embeddings: usize },
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

pub struct NewChunk<'a> {
    pub chunk_type: &'a ChunkType,
    pub name: Option<&'a str>,
    pub content: &'a str,
    pub start_line: u32,
    pub end_line: u32,
}

fn check_embedding_dims(embedding: &[f32]) -> Result<(), StorageError> {
    if embedding.len() != EMBEDDING_DIMS as usize {
        return Err(StorageError::DimensionMismatch {
            expected: EMBEDDING_DIMS as usize,
            actual: embedding.len(),
        });
    }
    Ok(())
}

fn insert_chunk_row(
    conn: &Connection,
    file_path: &str,
    chunk: &NewChunk,
    file_hash: &str,
) -> Result<i64, StorageError> {
    debug_assert!(chunk.start_line <= chunk.end_line, "start_line > end_line");
    debug_assert!(!chunk.content.is_empty(), "empty chunk content");
    conn.execute(
        "INSERT INTO chunks (file_path, chunk_type, name, content, start_line, end_line, file_hash)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        rusqlite::params![
            file_path,
            chunk.chunk_type.as_str(),
            chunk.name,
            chunk.content,
            chunk.start_line,
            chunk.end_line,
            file_hash,
        ],
    )?;
    let chunk_id = conn.last_insert_rowid();

    conn.execute(
        "INSERT INTO fts_chunks(rowid, content) VALUES (?1, ?2)",
        rusqlite::params![chunk_id, chunk.content],
    )?;

    Ok(chunk_id)
}

pub fn insert_chunk(
    conn: &Connection,
    file_path: &str,
    chunk: &NewChunk,
    file_hash: &str,
    embedding: &[f32],
) -> Result<i64, StorageError> {
    check_embedding_dims(embedding)?;
    let chunk_id = insert_chunk_row(conn, file_path, chunk, file_hash)?;

    let embedding_bytes = f32_as_bytes(embedding);
    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
        rusqlite::params![chunk_id, embedding_bytes],
    )?;

    Ok(chunk_id)
}

fn write_file_metadata(
    tx: &Connection,
    file_path: &str,
    imports_text: &str,
    refs: &[Reference],
) -> Result<(), StorageError> {
    tx.execute(
        "INSERT OR REPLACE INTO file_context (file_path, imports_text) VALUES (?1, ?2)",
        rusqlite::params![file_path, imports_text],
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

pub fn replace_file_chunks(
    conn: &Connection,
    file_path: &str,
    chunks: &[NewChunk],
    embeddings: &[Vec<f32>],
    file_hash: &str,
    imports_text: &str,
    refs: &[Reference],
) -> Result<(), StorageError> {
    if chunks.len() != embeddings.len() {
        return Err(StorageError::LengthMismatch {
            chunks: chunks.len(),
            embeddings: embeddings.len(),
        });
    }

    let tx = conn.unchecked_transaction()?;
    delete_file_chunks_in(&tx, file_path)?;

    for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
        insert_chunk(&tx, file_path, chunk, file_hash, embedding)?;
    }

    write_file_metadata(&tx, file_path, imports_text, refs)?;
    tx.commit()?;
    Ok(())
}

/// Stores chunks without embeddings (no API calls).
pub fn replace_file_chunks_only(
    conn: &Connection,
    file_path: &str,
    chunks: &[NewChunk],
    file_hash: &str,
    imports_text: &str,
    refs: &[Reference],
) -> Result<(), StorageError> {
    let tx = conn.unchecked_transaction()?;
    delete_file_chunks_in(&tx, file_path)?;

    for chunk in chunks {
        insert_chunk_row(&tx, file_path, chunk, file_hash)?;
    }

    write_file_metadata(&tx, file_path, imports_text, refs)?;
    tx.commit()?;
    Ok(())
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

    let total_files: u32 =
        conn.query_row("SELECT COUNT(DISTINCT file_path) FROM chunks", [], |row| {
            row.get(0)
        })?;

    let embedded_chunks: u32 =
        conn.query_row("SELECT COUNT(*) FROM vec_chunks", [], |row| row.get(0))?;

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

/// No transaction -- caller manages.
pub fn delete_file_chunks_in(conn: &Connection, file_path: &str) -> Result<(), StorageError> {
    conn.execute(
        "DELETE FROM fts_chunks WHERE rowid IN (SELECT id FROM chunks WHERE file_path = ?1)",
        [file_path],
    )?;
    conn.execute(
        "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?1)",
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefKind {
    Named,
    Default,
    Namespace,
    TypeOnly,
    SideEffect,
}

impl RefKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Named => "named",
            Self::Default => "default",
            Self::Namespace => "namespace",
            Self::TypeOnly => "type_only",
            Self::SideEffect => "side_effect",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    pub source_file: String,
    pub target_file: String,
    pub symbol_name: Option<String>,
    pub ref_kind: RefKind,
}

pub fn sql_placeholders(count: usize) -> String {
    std::iter::repeat_n("?", count)
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
mod tests;
