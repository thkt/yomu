use std::path::Path;

use rusqlite::Connection;
use rusqlite::ffi::sqlite3_auto_extension;
use sqlite_vec::sqlite3_vec_init;

use super::{StorageError, EMBEDDING_DIMS};

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
        Err(
            ref e @ rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error {
                    code: rusqlite::ffi::ErrorCode::CannotOpen | rusqlite::ffi::ErrorCode::SystemIoFailure,
                    ..
                },
                _,
            ),
        ) => {
            tracing::warn!(error = %e, "DB open failed (I/O), removing WAL/SHM and retrying");
            let path_str = path.to_string_lossy();
            let _ = std::fs::remove_file(format!("{path_str}-wal"));
            let _ = std::fs::remove_file(format!("{path_str}-shm"));
            Connection::open(path)?
        }
        Err(e) => return Err(e.into()),
    };

    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA busy_timeout=5000;",
    )?;

    init_schema(&conn)?;
    Ok(conn)
}

const SCHEMA_VERSION: u32 = 3;

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

    let stored: u32 = match conn.query_row(
        "SELECT value FROM index_meta WHERE key = 'schema_version'",
        [],
        |row| row.get::<_, String>(0),
    ) {
        Ok(v) => v.parse().unwrap_or(0),
        Err(rusqlite::Error::QueryReturnedNoRows) => 0,
        Err(e) => return Err(e.into()),
    };

    if stored != SCHEMA_VERSION {
        migrate(conn, stored)?;
        conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('schema_version', ?1)",
            [SCHEMA_VERSION.to_string()],
        )?;
    }

    Ok(())
}

fn migrate(conn: &Connection, from: u32) -> Result<(), StorageError> {
    let to = SCHEMA_VERSION;
    if from >= to {
        return Ok(());
    }
    tracing::info!(from, to, "Migrating schema from v{from} to v{to}");

    // v2 → v3: populate fts_chunks from existing chunks
    if from < 3 {
        fts_set_automerge(conn, false)?;
        conn.execute_batch(
            "INSERT OR IGNORE INTO fts_chunks(rowid, content)
             SELECT id, content FROM chunks"
        )?;
        fts_optimize(conn)?;
        fts_set_automerge(conn, true)?;
    }

    Ok(())
}

/// FTS5 default automerge threshold (merge when ≥4 b-tree segments exist).
const FTS5_AUTOMERGE_DEFAULT: i32 = 4;

pub fn fts_set_automerge(conn: &Connection, enabled: bool) -> Result<(), StorageError> {
    let value = if enabled { FTS5_AUTOMERGE_DEFAULT } else { 0 };
    conn.execute(
        "INSERT INTO fts_chunks(fts_chunks, rank) VALUES('automerge', ?1)",
        [value],
    )?;
    Ok(())
}

pub fn fts_optimize(conn: &Connection) -> Result<(), StorageError> {
    conn.execute("INSERT INTO fts_chunks(fts_chunks) VALUES('optimize')", [])?;
    Ok(())
}
