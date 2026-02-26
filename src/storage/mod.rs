//! SQLite storage layer for chunks and vector embeddings.

use std::collections::HashSet;
use std::path::Path;

use rusqlite::Connection;
use rusqlite::ffi::sqlite3_auto_extension;
use sqlite_vec::sqlite3_vec_init;

pub type Db = Connection;

/// Dimensionality of embedding vectors (Gemini gemini-embedding-001).
pub const EMBEDDING_DIMS: u32 = 768;

#[cfg(not(target_endian = "little"))]
compile_error!("yomu requires a little-endian target for f32↔u8 embedding storage");

/// Reinterpret `&[f32]` as `&[u8]` for sqlite-vec's BLOB format (little-endian).
fn f32_as_bytes(slice: &[f32]) -> &[u8] {
    bytemuck::cast_slice(slice)
}

/// Semantic classification of a code chunk, determined by AST analysis.
#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    /// React/framework component (PascalCase function or arrow function)
    Component,
    /// React hook (`use` + PascalCase suffix, e.g. `useAuth`)
    Hook,
    /// TypeScript `interface` or `type` alias
    TypeDef,
    /// CSS rule set, `@media`, or `@keyframes`
    CssRule,
    /// HTML top-level element
    HtmlElement,
    /// Test case (`describe`, `it`, or `test` call)
    TestCase,
    /// Anything not matching the above patterns
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
            other => {
                tracing::warn!(chunk_type = other, "Unknown chunk_type in DB, defaulting to Other");
                Self::Other
            }
        }
    }
}

/// A stored code chunk read back from the database.
///
/// `file_path` is relative to the project root.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub file_path: String,
    pub chunk_type: ChunkType,
    pub name: Option<String>,
    pub content: String,
    pub start_line: u32,
    pub end_line: u32,
}

/// Aggregate statistics about the search index.
#[derive(Debug)]
pub struct IndexStatus {
    pub total_files: u32,
    pub total_chunks: u32,
    pub last_indexed_at: Option<String>,
}

/// A chunk with its L2 distance from the query embedding.
///
/// Smaller `distance` = more similar. Display uses `similarity = 1.0 / (1.0 + distance)` → [0, 1].
#[derive(Debug)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub distance: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("chunks/embeddings length mismatch: {chunks} chunks vs {embeddings} embeddings")]
    LengthMismatch { chunks: usize, embeddings: usize },
}

/// Opens or creates a SQLite database at the given path, initializing the schema.
///
/// Registers the sqlite-vec extension on first call via [`std::sync::Once`].
/// Creates parent directories if they don't exist.
pub fn open_db(path: &Path) -> Result<Connection, StorageError> {
    // sqlite3_auto_extension is process-global; OnceLock ensures we register exactly
    // once even when open_db() is called multiple times (e.g., in tests).
    static INIT: std::sync::OnceLock<Result<(), i32>> = std::sync::OnceLock::new();
    let init_result = INIT.get_or_init(|| {
        // SAFETY: sqlite3_vec_init is the auto-extension entry point exported by sqlite-vec.
        // Typed local `ext` ensures a compile-time error if sqlite3_vec_init's signature
        // changes. transmute converts into the `unsafe extern "C" fn()` expected by
        // sqlite3_auto_extension. This is the documented usage pattern for sqlite-vec.
        let rc = unsafe {
            let ext = std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut std::os::raw::c_char,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> std::os::raw::c_int,
            >(sqlite3_vec_init as *const ());
            sqlite3_auto_extension(Some(ext))
        };
        if rc == 0 { Ok(()) } else { Err(rc) }
    });
    if let Err(rc) = init_result {
        return Err(StorageError::Io(std::io::Error::other(
            format!("sqlite-vec extension failed to register (rc={rc})"),
        )));
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let conn = Connection::open(path)?;

    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA busy_timeout=5000;",
    )?;

    init_schema(&conn)?;
    Ok(conn)
}

const SCHEMA_VERSION: &str = "1";

fn init_schema(conn: &Connection) -> Result<(), StorageError> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS chunks (
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
        );",
    )?;

    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIMS}]
        )"
    ))?;

    conn.execute(
        "INSERT OR IGNORE INTO index_meta (key, value) VALUES ('schema_version', ?1)",
        [SCHEMA_VERSION],
    )?;

    Ok(())
}

/// Data for inserting a new chunk (borrowed references to avoid cloning).
pub struct NewChunk<'a> {
    pub chunk_type: &'a ChunkType,
    pub name: Option<&'a str>,
    pub content: &'a str,
    pub start_line: u32,
    pub end_line: u32,
}

/// Insert a single chunk with its embedding. Returns the new row id.
pub fn insert_chunk(
    conn: &Connection,
    file_path: &str,
    chunk: &NewChunk,
    file_hash: &str,
    embedding: &[f32],
) -> Result<i64, StorageError> {
    debug_assert_eq!(
        embedding.len(),
        EMBEDDING_DIMS as usize,
        "embedding dimension mismatch: expected {}, got {}",
        EMBEDDING_DIMS,
        embedding.len()
    );
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

    let embedding_bytes = f32_as_bytes(embedding);

    conn.execute(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
        rusqlite::params![chunk_id, embedding_bytes],
    )?;

    Ok(chunk_id)
}

/// Atomically replace all chunks for a file. Deletes old data and inserts new
/// chunks + embeddings in a single transaction.
pub fn replace_file_chunks(
    conn: &Connection,
    file_path: &str,
    chunks: &[NewChunk],
    embeddings: &[Vec<f32>],
    file_hash: &str,
) -> Result<(), StorageError> {
    if chunks.len() != embeddings.len() {
        return Err(StorageError::LengthMismatch {
            chunks: chunks.len(),
            embeddings: embeddings.len(),
        });
    }

    // unchecked_transaction: conn is &Connection (not &mut) because it's behind
    // parking_lot::Mutex. The Mutex guarantees single-writer access.
    let tx = conn.unchecked_transaction()?;

    tx.execute(
        "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?1)",
        [file_path],
    )?;
    tx.execute("DELETE FROM chunks WHERE file_path = ?1", [file_path])?;

    for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
        insert_chunk(&tx, file_path, chunk, file_hash, embedding)?;
    }

    tx.execute(
        "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('last_indexed_at', datetime('now'))",
        [],
    )?;

    tx.commit()?;
    Ok(())
}

/// Check whether a file needs re-indexing by comparing its content hash.
///
/// Returns `true` if the file is new or its hash has changed.
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

/// Return aggregate statistics about the current index.
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
        last_indexed_at,
    })
}

/// Return all distinct file paths currently stored in the index.
pub fn get_all_file_paths(conn: &Connection) -> Result<HashSet<String>, StorageError> {
    let mut stmt = conn.prepare("SELECT DISTINCT file_path FROM chunks")?;
    let paths = stmt.query_map([], |row| row.get::<_, String>(0))?;
    paths.collect::<Result<HashSet<_>, _>>().map_err(Into::into)
}

/// Delete a file's chunks and embeddings (no transaction -- caller manages).
pub fn delete_file_chunks_in(conn: &Connection, file_path: &str) -> Result<(), StorageError> {
    conn.execute(
        "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM chunks WHERE file_path = ?1)",
        [file_path],
    )?;
    conn.execute("DELETE FROM chunks WHERE file_path = ?1", [file_path])?;
    Ok(())
}

#[allow(dead_code)]
pub fn delete_file_chunks(conn: &Connection, file_path: &str) -> Result<(), StorageError> {
    let tx = conn.unchecked_transaction()?;
    delete_file_chunks_in(&tx, file_path)?;
    tx.commit()?;
    Ok(())
}

/// Find chunks whose embeddings are nearest to `query_embedding`.
///
/// sqlite-vec requires `k` (the ANN retrieval window) as a query parameter.
/// We set `k = limit + offset` so that after SQL OFFSET, enough candidates
/// remain to fill the requested `limit`.
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
                c.start_line, c.end_line, v.distance
         FROM vec_chunks v
         INNER JOIN chunks c ON c.id = v.chunk_id
         WHERE v.embedding MATCH ?1 AND k = ?2
         ORDER BY v.distance
         LIMIT ?3
         OFFSET ?4",
    )?;

    let rows = stmt.query_map(rusqlite::params![query_bytes, k, limit, offset], |row| {
        Ok(SearchResult {
            chunk: Chunk {
                file_path: row.get(0)?,
                chunk_type: ChunkType::from_db(row.get::<_, String>(1)?.as_ref()),
                name: row.get(2)?,
                content: row.get(3)?,
                start_line: row.get(4)?,
                end_line: row.get(5)?,
            },
            distance: row.get(6)?,
        })
    })?;

    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_db() -> (Connection, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = open_db(&db_path).unwrap();
        (conn, dir)
    }

    #[test]
    fn init_db_creates_tables() {
        let (conn, _dir) = test_db();

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);

        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn insert_and_read_chunk() {
        let (conn, _dir) = test_db();
        let embedding = vec![0.1_f32; 768];

        let id = insert_chunk(
            &conn,
            "src/Button.tsx",
            &NewChunk {
                chunk_type: &ChunkType::Component,
                name: Some("Button"),
                content: "function Button() { return <div/>; }",
                start_line: 1,
                end_line: 3,
            },
            "abc123",
            &embedding,
        )
        .unwrap();

        assert!(id > 0);

        let chunk: (String, String, String) = conn
            .query_row(
                "SELECT file_path, chunk_type, name FROM chunks WHERE id = ?1",
                [id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();

        assert_eq!(chunk.0, "src/Button.tsx");
        assert_eq!(chunk.1, "component");
        assert_eq!(chunk.2, "Button");
    }

    #[test]
    fn should_reindex_returns_false_for_same_hash() {
        let (conn, _dir) = test_db();
        let embedding = vec![0.0_f32; 768];

        insert_chunk(
            &conn,
            "src/App.tsx",
            &NewChunk {
                chunk_type: &ChunkType::Component,
                name: Some("App"),
                content: "function App() {}",
                start_line: 1,
                end_line: 1,
            },
            "hash_abc",
            &embedding,
        )
        .unwrap();

        assert!(!should_reindex(&conn, "src/App.tsx", "hash_abc").unwrap());
    }

    #[test]
    fn should_reindex_returns_true_for_different_hash() {
        let (conn, _dir) = test_db();
        let embedding = vec![0.0_f32; 768];

        insert_chunk(
            &conn,
            "src/App.tsx",
            &NewChunk {
                chunk_type: &ChunkType::Component,
                name: Some("App"),
                content: "function App() {}",
                start_line: 1,
                end_line: 1,
            },
            "hash_abc",
            &embedding,
        )
        .unwrap();

        assert!(should_reindex(&conn, "src/App.tsx", "hash_xyz").unwrap());
    }

    #[test]
    fn should_reindex_returns_true_for_new_file() {
        let (conn, _dir) = test_db();
        assert!(should_reindex(&conn, "src/New.tsx", "any_hash").unwrap());
    }

    #[test]
    fn get_stats_returns_counts() {
        let (conn, _dir) = test_db();
        let embedding = vec![0.0_f32; 768];

        insert_chunk(
            &conn, "src/A.tsx",
            &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code", start_line: 1, end_line: 5 },
            "h1", &embedding,
        ).unwrap();
        insert_chunk(
            &conn, "src/A.tsx",
            &NewChunk { chunk_type: &ChunkType::Hook, name: Some("useA"), content: "code", start_line: 6, end_line: 10 },
            "h1", &embedding,
        ).unwrap();
        insert_chunk(
            &conn, "src/B.tsx",
            &NewChunk { chunk_type: &ChunkType::Component, name: Some("B"), content: "code", start_line: 1, end_line: 3 },
            "h2", &embedding,
        ).unwrap();

        let stats = get_stats(&conn).unwrap();
        assert_eq!(stats.total_chunks, 3);
        assert_eq!(stats.total_files, 2);
    }

    #[test]
    fn get_all_file_paths_returns_distinct_paths() {
        let (conn, _dir) = test_db();
        let embedding = vec![0.0_f32; 768];

        insert_chunk(&conn, "src/A.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code", start_line: 1, end_line: 3 }, "h1", &embedding).unwrap();
        insert_chunk(&conn, "src/A.tsx", &NewChunk { chunk_type: &ChunkType::Hook, name: Some("useA"), content: "code", start_line: 4, end_line: 6 }, "h1", &embedding).unwrap();
        insert_chunk(&conn, "src/B.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("B"), content: "code", start_line: 1, end_line: 3 }, "h2", &embedding).unwrap();

        let paths = get_all_file_paths(&conn).unwrap();
        assert_eq!(paths.len(), 2);
        assert!(paths.contains("src/A.tsx"));
        assert!(paths.contains("src/B.tsx"));
    }

    #[test]
    fn delete_file_chunks_removes_all_chunks_for_file() {
        let (conn, _dir) = test_db();
        let embedding = vec![0.0_f32; 768];

        insert_chunk(&conn, "src/A.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code", start_line: 1, end_line: 3 }, "h1", &embedding).unwrap();
        insert_chunk(&conn, "src/B.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("B"), content: "code", start_line: 1, end_line: 3 }, "h2", &embedding).unwrap();

        delete_file_chunks(&conn, "src/A.tsx").unwrap();

        let stats = get_stats(&conn).unwrap();
        assert_eq!(stats.total_files, 1);
        assert_eq!(stats.total_chunks, 1);

        let paths = get_all_file_paths(&conn).unwrap();
        assert!(!paths.contains("src/A.tsx"));
        assert!(paths.contains("src/B.tsx"));
    }

    #[test]
    fn replace_file_chunks_replaces_existing() {
        let (conn, _dir) = test_db();
        let embedding = vec![0.0_f32; 768];

        insert_chunk(
            &conn, "src/A.tsx",
            &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "old code", start_line: 1, end_line: 5 },
            "h1", &embedding,
        ).unwrap();

        let new_chunks = vec![
            NewChunk { chunk_type: &ChunkType::Hook, name: Some("useA"), content: "new code", start_line: 1, end_line: 3 },
            NewChunk { chunk_type: &ChunkType::Component, name: Some("B"), content: "more code", start_line: 4, end_line: 8 },
        ];
        let embeddings = vec![embedding.clone(), embedding.clone()];

        replace_file_chunks(&conn, "src/A.tsx", &new_chunks, &embeddings, "h2").unwrap();

        let stats = get_stats(&conn).unwrap();
        assert_eq!(stats.total_chunks, 2);
        assert_eq!(stats.total_files, 1);
        assert!(stats.last_indexed_at.is_some());
    }

    #[test]
    fn search_similar_returns_ordered_results() {
        let (conn, _dir) = test_db();
        let mut emb_a = vec![0.0_f32; 768];
        emb_a[0] = 1.0;
        let mut emb_b = vec![0.0_f32; 768];
        emb_b[1] = 1.0;

        insert_chunk(&conn, "src/A.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code a", start_line: 1, end_line: 3 }, "h1", &emb_a).unwrap();
        insert_chunk(&conn, "src/B.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("B"), content: "code b", start_line: 1, end_line: 3 }, "h2", &emb_b).unwrap();

        let results = search_similar(&conn, &emb_a, 10, 0).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].chunk.name.as_deref(), Some("A"));
        assert!(results[0].distance <= results[1].distance);
    }

    #[test]
    fn chunk_type_roundtrip() {
        let variants = [
            ChunkType::Component,
            ChunkType::Hook,
            ChunkType::TypeDef,
            ChunkType::CssRule,
            ChunkType::HtmlElement,
            ChunkType::TestCase,
            ChunkType::Other,
        ];
        for variant in &variants {
            let s = variant.as_str();
            let restored = ChunkType::from_db(s);
            assert_eq!(&restored, variant, "roundtrip failed for {s}");
        }
    }

    #[test]
    fn chunk_type_from_db_unknown_defaults_to_other() {
        assert_eq!(ChunkType::from_db("nonexistent"), ChunkType::Other);
    }
}
