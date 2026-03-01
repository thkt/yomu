//! SQLite storage layer for chunks and vector embeddings.

mod graph;
mod search;

pub use graph::*;
pub use search::*;

use std::collections::HashSet;
use std::path::Path;

use rusqlite::Connection;
use rusqlite::ffi::sqlite3_auto_extension;
use sqlite_vec::sqlite3_vec_init;

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
                tracing::warn!(chunk_type = other, "Unknown chunk_type in DB, defaulting to Other");
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

pub fn open_db(path: &Path) -> Result<Connection, StorageError> {
    static INIT: std::sync::OnceLock<Result<(), i32>> = std::sync::OnceLock::new();
    let init_result = INIT.get_or_init(|| {
        // SAFETY: sqlite3_vec_init is the auto-extension entry point exported by sqlite-vec.
        // sqlite-vec exports it as `unsafe extern "C" fn()`, while rusqlite's
        // sqlite3_auto_extension expects the full init signature. Both are C fn pointers
        // with compatible calling conventions; SQLite calls it with the correct arguments.
        // Explicit source type ensures compile error if sqlite-vec changes its export type.
        let rc = unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute::<
                unsafe extern "C" fn(),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut std::os::raw::c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> std::os::raw::c_int,
            >(sqlite3_vec_init)))
        };
        if rc == 0 { Ok(()) } else { Err(rc) }
    });
    if let Err(rc) = init_result {
        return Err(StorageError::Io(std::io::Error::other(
            format!(
                "sqlite-vec extension failed to register (sqlite3 rc={rc}). \
                 This is a process-level initialization error that cannot be retried."
            ),
        )));
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let conn = match Connection::open(path) {
        Ok(c) => c,
        Err(e) => {
            // Stale WAL/SHM files can cause I/O errors; remove them and retry
            tracing::warn!(error = %e, "DB open failed, removing WAL/SHM and retrying");
            let path_str = path.to_string_lossy();
            let _ = std::fs::remove_file(format!("{path_str}-wal"));
            let _ = std::fs::remove_file(format!("{path_str}-shm"));
            Connection::open(path)?
        }
    };

    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA busy_timeout=5000;",
    )?;

    init_schema(&conn)?;
    Ok(conn)
}

const SCHEMA_VERSION: &str = "3";

const DDL: &str = "
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        file_path TEXT NOT NULL,
        chunk_type TEXT NOT NULL,
        name TEXT,
        content TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        file_hash TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
    CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(file_hash);

    CREATE TABLE IF NOT EXISTS index_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS file_context (
        file_path TEXT PRIMARY KEY,
        imports_text TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS file_references (
        id INTEGER PRIMARY KEY,
        source_file TEXT NOT NULL,
        target_file TEXT NOT NULL,
        symbol_name TEXT,
        ref_kind TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_refs_source ON file_references(source_file);
    CREATE INDEX IF NOT EXISTS idx_refs_target ON file_references(target_file);
    CREATE INDEX IF NOT EXISTS idx_refs_target_symbol ON file_references(target_file, symbol_name);
";

fn init_schema(conn: &Connection) -> Result<(), StorageError> {
    conn.execute_batch(DDL)?;

    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIMS}]
        )"
    ))?;

    conn.execute_batch(
        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
            content,
            content_rowid='rowid'
        )"
    )?;

    let stored: String = match conn.query_row(
        "SELECT value FROM index_meta WHERE key = 'schema_version'",
        [],
        |row| row.get(0),
    ) {
        Ok(v) => v,
        Err(rusqlite::Error::QueryReturnedNoRows) => "0".to_string(),
        Err(e) => return Err(e.into()),
    };

    if stored != SCHEMA_VERSION {
        migrate(conn, &stored)?;
        conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('schema_version', ?1)",
            [SCHEMA_VERSION],
        )?;
    }

    Ok(())
}

fn migrate(conn: &Connection, from_version: &str) -> Result<(), StorageError> {
    let from: u32 = match from_version.parse() {
        Ok(v) => v,
        Err(_) => {
            tracing::warn!(version = from_version, "Unparseable schema version, treating as 0");
            0
        }
    };
    let to: u32 = SCHEMA_VERSION.parse().expect("SCHEMA_VERSION must be a valid u32");
    if from >= to {
        return Ok(());
    }
    tracing::info!(from, to, "Migrating schema from v{from} to v{to}");

    // v2 → v3: populate fts_chunks from existing chunks
    if from < 3 {
        conn.execute_batch(
            "INSERT OR IGNORE INTO fts_chunks(rowid, content)
             SELECT id, content FROM chunks"
        )?;
    }

    Ok(())
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
            rusqlite::params![r.source_file, r.target_file, r.symbol_name, r.ref_kind.as_str()],
        )?;
    }

    tx.execute(
        "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('last_indexed_at', datetime('now'))",
        [],
    )?;
    Ok(())
}

/// Replaces all chunks + embeddings + references for a file in a single transaction.
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

/// Returns the number of newly inserted embeddings.
pub fn add_embeddings(
    conn: &Connection,
    embeddings: &[(i64, Vec<f32>)],
) -> Result<u32, StorageError> {
    let tx = conn.unchecked_transaction()?;
    let mut inserted = 0u32;

    for (chunk_id, embedding) in embeddings {
        check_embedding_dims(embedding)?;
        let exists: bool = tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM vec_chunks WHERE chunk_id = ?1)",
            [chunk_id],
            |row| row.get(0),
        )?;
        if exists {
            continue;
        }
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

pub fn get_unembedded_file_paths(
    conn: &Connection,
) -> Result<Vec<(String, u32)>, StorageError> {
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
    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::with_capacity(types.len() + files.len());
    for t in types {
        params.push(Box::new(t.as_str().to_string()));
    }
    for f in files {
        params.push(Box::new(f.clone()));
    }
    let param_refs: Vec<&dyn rusqlite::types::ToSql> =
        params.iter().map(|b| b.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt
        .query_map(param_refs.as_slice(), |row| row.get::<_, String>(0))?;
    rows.collect::<Result<HashSet<_>, _>>().map_err(Into::into)
}

/// Returns true if the file needs embedding:
/// new file → true, hash changed → true, has un-embedded chunks → true, else false.
pub fn needs_embedding(
    conn: &Connection,
    file_path: &str,
    current_hash: &str,
) -> Result<bool, StorageError> {
    let stored_hash: Option<String> = match conn.query_row(
        "SELECT file_hash FROM chunks WHERE file_path = ?1 LIMIT 1",
        [file_path],
        |row| row.get::<_, String>(0),
    ) {
        Ok(h) => Some(h),
        Err(rusqlite::Error::QueryReturnedNoRows) => None,
        Err(e) => return Err(e.into()),
    };

    match stored_hash {
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
    match conn.query_row(
        "SELECT file_hash FROM chunks WHERE file_path = ?1 LIMIT 1",
        [file_path],
        |row| row.get::<_, String>(0),
    ) {
        Ok(stored_hash) => Ok(stored_hash != current_hash),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(true),
        Err(e) => Err(e.into()),
    }
}

pub fn get_stats(conn: &Connection) -> Result<IndexStatus, StorageError> {
    let total_chunks: u32 = conn.query_row(
        "SELECT COUNT(*) FROM chunks",
        [],
        |row| row.get(0),
    )?;

    let total_files: u32 = conn.query_row(
        "SELECT COUNT(DISTINCT file_path) FROM chunks",
        [],
        |row| row.get(0),
    )?;

    let embedded_chunks: u32 = conn.query_row(
        "SELECT COUNT(*) FROM vec_chunks",
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
        embedded_chunks,
        last_indexed_at,
    })
}

/// Returns true if the index was updated within `max_age_secs` seconds.
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
    std::iter::repeat_n("?", count).collect::<Vec<_>>().join(",")
}

#[cfg(test)]
mod tests;
