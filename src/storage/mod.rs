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
            "other" => Self::Other,
            other => {
                tracing::warn!(chunk_type = other, "Unknown chunk_type in DB, defaulting to Other");
                Self::Other
            }
        }
    }
}

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

#[derive(Debug)]
pub struct IndexStatus {
    pub total_files: u32,
    pub total_chunks: u32,
    pub embedded_chunks: u32,
    pub last_indexed_at: Option<String>,
}

/// Source of a search result match.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchSource {
    /// Vector similarity search result.
    Semantic,
    /// Name/type fallback search result.
    NameMatch,
}

/// Smaller `distance` = more similar. Display uses `similarity = 1.0 / (1.0 + distance)`.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub distance: f32,
    pub match_source: MatchSource,
    /// Reranked score (higher = better). Initialized from distance/match_source,
    /// then adjusted by rerank signals (type hints, import count, test path).
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
}

/// Registers the sqlite-vec extension on first call.
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

const SCHEMA_VERSION: &str = "2";

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
        CREATE INDEX IF NOT EXISTS idx_refs_target_symbol ON file_references(target_file, symbol_name);",
    )?;

    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
            chunk_id INTEGER PRIMARY KEY,
            embedding FLOAT[{EMBEDDING_DIMS}]
        )"
    ))?;

    // Read stored version and update if needed (INSERT OR REPLACE, not INSERT OR IGNORE)
    let stored: String = conn
        .query_row(
            "SELECT value FROM index_meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )
        .unwrap_or_else(|_| "0".to_string());
    if stored != SCHEMA_VERSION {
        conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('schema_version', ?1)",
            [SCHEMA_VERSION],
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

pub fn insert_chunk(
    conn: &Connection,
    file_path: &str,
    chunk: &NewChunk,
    file_hash: &str,
    embedding: &[f32],
) -> Result<i64, StorageError> {
    assert_eq!(
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

/// Deletes old data and inserts new chunks + embeddings + references in a
/// single transaction. Also stores file-level context (import statements)
/// in the `file_context` table.
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

    // unchecked_transaction: conn is &Connection (not &mut) because it's behind
    // parking_lot::Mutex. The Mutex guarantees single-writer access.
    let tx = conn.unchecked_transaction()?;

    delete_file_chunks_in(&tx, file_path)?;

    for (chunk, embedding) in chunks.iter().zip(embeddings.iter()) {
        insert_chunk(&tx, file_path, chunk, file_hash, embedding)?;
    }

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

    tx.commit()?;
    Ok(())
}

/// Stores chunks without embeddings. Deletes old data (including vec_chunks)
/// and inserts new chunks + file_context + file_references.
/// Used by chunk-only indexing (no Gemini API calls).
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
        tx.execute(
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
    }

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

    tx.commit()?;
    Ok(())
}

/// Add embeddings to existing chunks (INSERT OR IGNORE into vec_chunks).
/// Returns the number of newly inserted embeddings.
pub fn add_embeddings(
    conn: &Connection,
    embeddings: &[(i64, Vec<f32>)],
) -> Result<u32, StorageError> {
    let tx = conn.unchecked_transaction()?;
    let mut inserted = 0u32;

    for (chunk_id, embedding) in embeddings {
        assert_eq!(
            embedding.len(),
            EMBEDDING_DIMS as usize,
            "embedding dimension mismatch: expected {}, got {}",
            EMBEDDING_DIMS,
            embedding.len()
        );
        // vec0 virtual tables don't support INSERT OR IGNORE, so check first.
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

/// Returns file paths that have chunks but no embeddings, with chunk counts.
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

/// Returns true if the file needs embedding:
/// - file not in DB → true (new file)
/// - hash changed → true (re-chunk + re-embed)
/// - hash same but has un-embedded chunks → true (embed only)
/// - hash same and all chunks embedded → false (skip)
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

pub fn get_all_file_paths(conn: &Connection) -> Result<HashSet<String>, StorageError> {
    let mut stmt = conn.prepare("SELECT DISTINCT file_path FROM chunks")?;
    let paths = stmt.query_map([], |row| row.get::<_, String>(0))?;
    paths.collect::<Result<HashSet<_>, _>>().map_err(Into::into)
}

/// No transaction -- caller manages.
pub fn delete_file_chunks_in(conn: &Connection, file_path: &str) -> Result<(), StorageError> {
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

/// Search chunks by name using SQL LIKE.
/// Returns chunks whose `name` contains any of the keywords (case-insensitive).
/// Empty keywords returns empty results.
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

    let keyword_conditions: Vec<String> = keywords
        .iter()
        .enumerate()
        .map(|(i, _)| format!("name LIKE ?{}", i + 1))
        .collect();
    let keyword_clause = keyword_conditions.join(" OR ");

    let mut sql = format!(
        "SELECT id, file_path, chunk_type, name, content, start_line, end_line \
         FROM chunks WHERE name IS NOT NULL AND ({})",
        keyword_clause
    );

    if let Some(types) = type_filter {
        if !types.is_empty() {
            let type_list: Vec<String> = types.iter().map(|t| format!("'{}'", t.as_str())).collect();
            sql.push_str(&format!(" AND chunk_type IN ({})", type_list.join(",")));
        }
    }

    if !exclude_ids.is_empty() {
        let id_list: Vec<String> = exclude_ids.iter().map(|id| id.to_string()).collect();
        sql.push_str(&format!(" AND id NOT IN ({})", id_list.join(",")));
    }

    sql.push_str(&format!(" LIMIT {}", limit));

    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<String> = keywords.iter().map(|k| format!("%{k}%")).collect();
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params
        .iter()
        .map(|s| s as &dyn rusqlite::types::ToSql)
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
            distance: f32::INFINITY,
            match_source: MatchSource::NameMatch,
            score: 0.5,
        })
    })?;

    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

#[cfg(test)]
pub fn delete_file_chunks(conn: &Connection, file_path: &str) -> Result<(), StorageError> {
    let tx = conn.unchecked_transaction()?;
    delete_file_chunks_in(&tx, file_path)?;
    tx.commit()?;
    Ok(())
}

// ── Reference types (FR-005, FR-006) ──

/// Kind of import reference (named, default, namespace, type_only, or side_effect).
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

/// A resolved file-to-file reference stored in DB.
#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    pub source_file: String,
    pub target_file: String,
    pub symbol_name: Option<String>,
    pub ref_kind: RefKind,
}

/// A dependent file returned by impact queries.
#[derive(Debug, Clone, PartialEq)]

pub struct Dependent {
    pub file_path: String,
    pub depth: u32,
}

/// Atomically replace all references for a source file.
/// Deletes old references and inserts new ones.
///
/// Production code uses [`replace_file_chunks`] which handles references
/// atomically within the same transaction. This function is kept for tests
/// that need to set up references independently.
#[cfg(test)]
pub fn replace_file_references(
    conn: &Connection,
    source_file: &str,
    refs: &[Reference],
) -> Result<(), StorageError> {
    let tx = conn.unchecked_transaction()?;
    tx.execute(
        "DELETE FROM file_references WHERE source_file = ?1",
        [source_file],
    )?;
    for r in refs {
        tx.execute(
            "INSERT INTO file_references (source_file, target_file, symbol_name, ref_kind)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![r.source_file, r.target_file, r.symbol_name, r.ref_kind.as_str()],
        )?;
    }
    tx.commit()?;
    Ok(())
}

/// Get direct dependents of a target file (files that import it).
#[allow(dead_code)] // public API, used in tests
pub fn get_dependents(
    conn: &Connection,
    target_file: &str,
) -> Result<Vec<Dependent>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT source_file FROM file_references WHERE target_file = ?1",
    )?;
    let rows = stmt.query_map([target_file], |row| {
        Ok(Dependent {
            file_path: row.get(0)?,
            depth: 1,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

/// Get transitive dependents using recursive CTE.
/// Returns all files that directly or transitively depend on target_file,
/// with depth clamped to max_depth (max 10).
pub fn get_transitive_dependents(
    conn: &Connection,
    target_file: &str,
    max_depth: u32,
) -> Result<Vec<Dependent>, StorageError> {
    let max_depth = max_depth.min(10);
    let mut stmt = conn.prepare(
        "WITH RECURSIVE deps(file_path, depth, visited) AS (
            SELECT DISTINCT source_file, 1,
                   ',' || ?1 || ',' || source_file || ','
            FROM file_references
            WHERE target_file = ?1
          UNION
            SELECT r.source_file, d.depth + 1,
                   d.visited || r.source_file || ','
            FROM file_references r
            INNER JOIN deps d ON r.target_file = d.file_path
            WHERE d.depth < ?2
              AND INSTR(d.visited, ',' || r.source_file || ',') = 0
        )
        SELECT file_path, MIN(depth) as depth
        FROM deps GROUP BY file_path ORDER BY depth, file_path",
    )?;
    let rows = stmt.query_map(rusqlite::params![target_file, max_depth], |row| {
        Ok(Dependent {
            file_path: row.get(0)?,
            depth: row.get(1)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

/// Get direct dependents that reference a specific symbol from a target file.
pub fn get_symbol_dependents(
    conn: &Connection,
    target_file: &str,
    symbol_name: &str,
) -> Result<Vec<String>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT source_file FROM file_references
         WHERE target_file = ?1 AND symbol_name = ?2",
    )?;
    let rows = stmt.query_map(rusqlite::params![target_file, symbol_name], |row| {
        row.get::<_, String>(0)
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

/// Get the total number of references in the database.
pub fn get_reference_count(conn: &Connection) -> Result<u32, StorageError> {
    let count: u32 = conn.query_row(
        "SELECT COUNT(*) FROM file_references",
        [],
        |row| row.get(0),
    )?;
    Ok(count)
}

/// Returns un-embedded file paths ordered by import count (most-imported first).
/// Files with the same import count are ordered alphabetically.
pub fn get_files_by_import_count(
    conn: &Connection,
) -> Result<Vec<String>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT c.file_path
         FROM chunks c
         LEFT JOIN vec_chunks v ON c.id = v.chunk_id
         WHERE v.chunk_id IS NULL
         GROUP BY c.file_path
         ORDER BY (
             SELECT COUNT(*) FROM file_references r
             WHERE r.target_file = c.file_path
         ) + (
             SELECT CASE WHEN EXISTS(
                 SELECT 1 FROM chunks c2
                 WHERE c2.file_path = c.file_path
                 AND c2.chunk_type IN ('hook', 'component')
             ) THEN 3 ELSE 0 END
         ) DESC, c.file_path ASC",
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

/// Returns import counts for the given file paths.
/// Each count is the number of files that import the target file.
pub fn get_import_counts(
    conn: &Connection,
    file_paths: &[&str],
) -> Result<std::collections::HashMap<String, u32>, StorageError> {
    if file_paths.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let sql = format!(
        "SELECT fp.path, COALESCE(cnt.c, 0)
         FROM (SELECT value AS path FROM ({})) fp
         LEFT JOIN (
             SELECT target_file, COUNT(DISTINCT source_file) AS c
             FROM file_references
             GROUP BY target_file
         ) cnt ON cnt.target_file = fp.path",
        file_paths.iter().enumerate()
            .map(|(i, _)| format!("SELECT ?{} AS value", i + 1))
            .collect::<Vec<_>>()
            .join(" UNION ALL ")
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> = file_paths
        .iter()
        .map(|s| s as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, u32>(1)?))
    })?;
    rows.collect::<Result<std::collections::HashMap<_, _>, _>>()
        .map_err(Into::into)
}

#[derive(Debug, Clone)]
pub struct SiblingInfo {
    pub name: Option<String>,
    pub chunk_type: ChunkType,
    pub start_line: u32,
    pub end_line: u32,
}

/// Build a comma-separated list of `?` placeholders for SQL IN clauses.
fn sql_placeholders(count: usize) -> String {
    std::iter::repeat_n("?", count).collect::<Vec<_>>().join(",")
}

/// Files without file_context entries (e.g. old indexes) are absent from the result.
pub fn get_file_contexts(
    conn: &Connection,
    file_paths: &[&str],
) -> Result<std::collections::HashMap<String, String>, StorageError> {
    if file_paths.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let placeholders = sql_placeholders(file_paths.len());
    let sql = format!(
        "SELECT file_path, imports_text FROM file_context WHERE file_path IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> = file_paths
        .iter()
        .map(|s| s as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    rows.collect::<Result<std::collections::HashMap<_, _>, _>>()
        .map_err(Into::into)
}

/// Returns other chunks in the same file as search results ("siblings").
pub fn get_file_siblings(
    conn: &Connection,
    file_paths: &[&str],
) -> Result<std::collections::HashMap<String, Vec<SiblingInfo>>, StorageError> {
    if file_paths.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let placeholders = sql_placeholders(file_paths.len());
    let sql = format!(
        "SELECT file_path, name, chunk_type, start_line, end_line \
         FROM chunks WHERE file_path IN ({placeholders}) \
         ORDER BY file_path, start_line"
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> = file_paths
        .iter()
        .map(|s| s as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,
            SiblingInfo {
                name: row.get(1)?,
                chunk_type: ChunkType::from_db(row.get::<_, String>(2)?.as_ref()),
                start_line: row.get(3)?,
                end_line: row.get(4)?,
            },
        ))
    })?;
    let mut map: std::collections::HashMap<String, Vec<SiblingInfo>> =
        std::collections::HashMap::new();
    for row in rows {
        let (path, info) = row?;
        map.entry(path).or_default().push(info);
    }
    Ok(map)
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
            distance,
            match_source: MatchSource::Semantic,
            score: 1.0 / (1.0 + distance),
        })
    })?;

    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

#[cfg(test)]
mod tests;
