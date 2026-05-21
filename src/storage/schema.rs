use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;

use amici::migration::notify_schema_change;
use rurico::storage::ensure_sqlite_vec;
use rusqlite::Connection;
use rusqlite::ffi::{Error as FfiError, ErrorCode};

use super::{EMBEDDING_DIMS, StorageError};

pub fn open_db(path: &Path) -> Result<Connection, StorageError> {
    ensure_sqlite_vec().map_err(|e| StorageError::Io(io::Error::other(e)))?;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let conn = match Connection::open(path) {
        Ok(c) => c,
        Err(
            ref e @ rusqlite::Error::SqliteFailure(
                FfiError {
                    code: ErrorCode::CannotOpen | ErrorCode::SystemIoFailure,
                    ..
                },
                _,
            ),
        ) => {
            tracing::warn!(error = %e, "DB open failed (I/O), removing WAL/SHM and retrying");
            let path_str = path.to_string_lossy();
            let _ = fs::remove_file(format!("{path_str}-wal"));
            let _ = fs::remove_file(format!("{path_str}-shm"));
            Connection::open(path)?
        }
        Err(e) => return Err(e.into()),
    };

    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA busy_timeout=5000;",
    )?;

    init_schema(&conn, path)?;
    verify_required_columns(&conn, path)?;
    Ok(conn)
}

const SCHEMA_VERSION: u32 = 10;

const DDL_FTS_CHUNKS: &str = "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(name, content, file_path, tokenize='trigram')";

const DDL_FTS_CHUNKS_VOCAB: &str =
    "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks_vocab USING fts5vocab(fts_chunks, row)";

const DDL_EMBEDDED_CHUNK_IDS: &str = "\
    CREATE TABLE IF NOT EXISTS embedded_chunk_ids (\
        chunk_id INTEGER NOT NULL, \
        sub_idx INTEGER NOT NULL, \
        vec_rowid INTEGER NOT NULL, \
        PRIMARY KEY (chunk_id, sub_idx)\
    )";

fn ddl_vec_chunks() -> String {
    format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(\
             embedding FLOAT[{EMBEDDING_DIMS}], \
             +chunk_id INTEGER, \
             +sub_idx INTEGER\
         )"
    )
}

const DDL: &str = "
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        file_path TEXT NOT NULL,
        chunk_type TEXT NOT NULL,
        name TEXT,
        content TEXT NOT NULL,
        start_line INTEGER NOT NULL,
        end_line INTEGER NOT NULL,
        file_hash TEXT NOT NULL,
        parent_chunk_id INTEGER REFERENCES chunks(id),
        source_kind TEXT,
        injection_flags TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
    CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(file_hash);

    CREATE TABLE IF NOT EXISTS index_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS file_context (
        file_path TEXT PRIMARY KEY,
        imports_text TEXT NOT NULL,
        mtime_epoch INTEGER
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

fn init_schema(conn: &Connection, path: &Path) -> Result<(), StorageError> {
    conn.execute_batch(DDL)?;

    conn.execute_batch(&ddl_vec_chunks())?;
    conn.execute_batch(DDL_EMBEDDED_CHUNK_IDS)?;

    conn.execute_batch(DDL_FTS_CHUNKS)?;

    let stored: u32 = match conn.query_row(
        "SELECT value FROM index_meta WHERE key = 'schema_version'",
        [],
        |row| row.get::<_, String>(0),
    ) {
        Ok(v) => v.parse().unwrap_or_else(|e| {
            tracing::warn!(value = %v, error = %e, "Corrupt schema_version, treating as 0");
            0
        }),
        Err(rusqlite::Error::QueryReturnedNoRows) => 0,
        Err(e) => return Err(e.into()),
    };

    if stored != SCHEMA_VERSION {
        migrate(conn, stored, path)?;
        conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('schema_version', ?1)",
            [SCHEMA_VERSION.to_string()],
        )?;
    }

    Ok(())
}

fn verify_required_columns(conn: &Connection, path: &Path) -> Result<(), StorageError> {
    const REQUIRED: &[&str] = &[
        "id",
        "file_path",
        "chunk_type",
        "name",
        "content",
        "start_line",
        "end_line",
        "file_hash",
        "parent_chunk_id",
        "source_kind",
        "injection_flags",
    ];

    let mut stmt = conn.prepare("PRAGMA table_info(chunks)")?;
    let existing: HashSet<String> = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<Result<HashSet<String>, _>>()?;

    let missing: Vec<String> = REQUIRED
        .iter()
        .filter(|col| !existing.contains(**col))
        .map(|col| (*col).to_owned())
        .collect();

    if !missing.is_empty() {
        return Err(StorageError::SchemaMismatch {
            table: "chunks",
            missing,
            path: path.to_path_buf(),
        });
    }

    Ok(())
}

/// Returns `true` if the column probe query succeeds, `false` if SQLite reports "no such column".
fn column_exists(conn: &Connection, probe_sql: &str) -> Result<bool, StorageError> {
    match conn.prepare(probe_sql) {
        Ok(_) => Ok(true),
        Err(rusqlite::Error::SqliteFailure(_, Some(ref msg))) if msg.contains("no such column") => {
            Ok(false)
        }
        Err(e) => Err(e.into()),
    }
}

fn migrate(conn: &Connection, from: u32, path: &Path) -> Result<(), StorageError> {
    let to = SCHEMA_VERSION;
    if from >= to {
        return Ok(());
    }
    tracing::info!(from, to, "Migrating schema from v{from} to v{to}");

    // The v8 → v9 step below drops every chunks-related table, so v4 → v8
    // work on `chunks` / `fts_*` / `vec_chunks` is wiped. Only the v5 → v6
    // ALTER on `file_context` survives — v9 does not touch that table.

    // v4 → v5: add parent_chunk_id for subchunk extraction
    if from < 5 && !column_exists(conn, "SELECT parent_chunk_id FROM chunks LIMIT 0")? {
        conn.execute_batch(
            "ALTER TABLE chunks ADD COLUMN parent_chunk_id INTEGER REFERENCES chunks(id)",
        )?;
    }

    // v5 → v6: add mtime_epoch to file_context
    if from < 6 && !column_exists(conn, "SELECT mtime_epoch FROM file_context LIMIT 0")? {
        conn.execute_batch("ALTER TABLE file_context ADD COLUMN mtime_epoch INTEGER")?;
    }

    // v6 → v7: rebuild FTS with 3 columns (name, content, file_path).
    // Drop fts_chunks/vocab and let `yomu index` repopulate on the next run.
    // file_hash reset is handled by the v7 → v8 block below.
    if from < 7 {
        let _span = tracing::info_span!("migration_v7", path = %path.display()).entered();
        conn.execute_batch(
            "DROP TABLE IF EXISTS fts_chunks_vocab; DROP TABLE IF EXISTS fts_chunks;",
        )?;
        conn.execute_batch(DDL_FTS_CHUNKS)?;
        conn.execute_batch(DDL_FTS_CHUNKS_VOCAB)?;
        notify_schema_change("yomu", "FTS index", 0, "yomu index");
    }

    // v7 → v8: migrate vec_chunks to multi-sub-chunk schema, add embedded_chunk_ids
    if from < 8 {
        let _span = tracing::info_span!("migration_v8", path = %path.display()).entered();
        conn.execute_batch("DROP TABLE IF EXISTS vec_chunks")?;
        conn.execute_batch(&ddl_vec_chunks())?;
        conn.execute_batch(DDL_EMBEDDED_CHUNK_IDS)?;
        // Clear file hashes so unchanged files are re-embedded on the next `yomu index`.
        // Without this, should_reindex() would skip every file whose content hasn't changed,
        // leaving embedded_chunk_ids permanently empty after the migration.
        conn.execute_batch("UPDATE chunks SET file_hash = ''")?;
        notify_schema_change("yomu", "embeddings", 0, "yomu index");
    }

    // v8 → v9: add source_kind / injection_flags to chunks via drop-and-recreate.
    // ADR-0069 accepts the reindex cost in exchange for forbidding ALTER ADD
    // COLUMN. chunks.id restarts at 1 after recreation; old fts_chunks /
    // vec_chunks / embedded_chunk_ids rows would collide via rowid on the next
    // INSERT, so drop and recreate the dependent tables together.
    if from < 9 {
        let _span = tracing::info_span!("migration_v9", path = %path.display()).entered();
        conn.execute_batch(
            "DROP TABLE IF EXISTS chunks; \
             DROP TABLE IF EXISTS embedded_chunk_ids; \
             DROP TABLE IF EXISTS vec_chunks; \
             DROP TABLE IF EXISTS fts_chunks_vocab; \
             DROP TABLE IF EXISTS fts_chunks;",
        )?;
        // file_references rows reference chunks-side file_paths; wiping
        // chunks invalidates them. Leaving the table populated would leak
        // stale references across the reindex boundary into `impact` and
        // `status` queries (F-005). The next `yomu index` rebuilds entries
        // per surviving file via replace_file_references.
        conn.execute_batch("DELETE FROM file_references")?;
        conn.execute_batch(DDL)?;
        conn.execute_batch(&ddl_vec_chunks())?;
        conn.execute_batch(DDL_EMBEDDED_CHUNK_IDS)?;
        conn.execute_batch(DDL_FTS_CHUNKS)?;
        conn.execute_batch(DDL_FTS_CHUNKS_VOCAB)?;
        notify_schema_change("yomu", "chunks v9", 0, "yomu index");
    }

    // v9 → v10: close PR#1 contract loop. PR#1-era v9 DBs may contain rows
    // with NULL `source_kind` / `injection_flags`. Once PR#3 emits
    // `injection_check = "ran"` at the response level, those NULLs would lie
    // to consumers. Drop + recreate the same dependent set as v9 so the next
    // `yomu index` repopulates the columns uniformly.
    if from < 10 {
        let _span = tracing::info_span!("migration_v10", path = %path.display()).entered();
        conn.execute_batch(
            "DROP TABLE IF EXISTS chunks; \
             DROP TABLE IF EXISTS embedded_chunk_ids; \
             DROP TABLE IF EXISTS vec_chunks; \
             DROP TABLE IF EXISTS fts_chunks_vocab; \
             DROP TABLE IF EXISTS fts_chunks;",
        )?;
        // Mirror v9 file_references wipe: dropping chunks invalidates the
        // references; the next reindex repopulates per surviving file (F-005).
        conn.execute_batch("DELETE FROM file_references")?;
        conn.execute_batch(DDL)?;
        conn.execute_batch(&ddl_vec_chunks())?;
        conn.execute_batch(DDL_EMBEDDED_CHUNK_IDS)?;
        conn.execute_batch(DDL_FTS_CHUNKS)?;
        conn.execute_batch(DDL_FTS_CHUNKS_VOCAB)?;
        notify_schema_change("yomu", "chunks v10", 0, "yomu index");
    }

    Ok(())
}

const FTS5_AUTOMERGE_DEFAULT: i32 = 4;

pub fn fts_set_automerge(conn: &Connection, enabled: bool) -> Result<(), StorageError> {
    let value = if enabled { FTS5_AUTOMERGE_DEFAULT } else { 0 };
    conn.execute(
        "INSERT INTO fts_chunks(fts_chunks, rank) VALUES('automerge', ?1)",
        [value],
    )?;
    Ok(())
}

pub struct FtsAutomergeGuard<'a> {
    conn: &'a Connection,
}

impl<'a> FtsAutomergeGuard<'a> {
    pub fn new(conn: &'a Connection) -> Result<Self, StorageError> {
        fts_set_automerge(conn, false)?;
        Ok(Self { conn })
    }
}

impl Drop for FtsAutomergeGuard<'_> {
    fn drop(&mut self) {
        if let Err(e) = fts_set_automerge(self.conn, true) {
            tracing::error!(error = %e, "Failed to restore FTS automerge");
        }
    }
}

pub fn fts_optimize(conn: &Connection) -> Result<(), StorageError> {
    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('optimize')", [])?;
    Ok(())
}
