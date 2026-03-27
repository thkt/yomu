use std::path::Path;

use rusqlite::Connection;

use super::{EMBEDDING_DIMS, StorageError};

pub fn open_db(path: &Path) -> Result<Connection, StorageError> {
    rurico::storage::ensure_sqlite_vec().map_err(|e| StorageError::Io(std::io::Error::other(e)))?;

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let conn = match Connection::open(path) {
        Ok(c) => c,
        Err(
            ref e @ rusqlite::Error::SqliteFailure(
                rusqlite::ffi::Error {
                    code:
                        rusqlite::ffi::ErrorCode::CannotOpen | rusqlite::ffi::ErrorCode::SystemIoFailure,
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

const SCHEMA_VERSION: u32 = 5;

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
        parent_chunk_id INTEGER REFERENCES chunks(id)
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
        )",
    )?;

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
        let _automerge = FtsAutomergeGuard::new(conn)?;
        conn.execute_batch(
            "INSERT OR IGNORE INTO fts_chunks(rowid, content)
             SELECT id, content FROM chunks",
        )?;
        fts_optimize(conn)?;
    }

    // v3 → v4: add fts5vocab table for short-term expansion
    if from < 4 {
        conn.execute_batch(
            "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks_vocab USING fts5vocab(fts_chunks, row)",
        )?;
    }

    // v4 → v5: add parent_chunk_id for subchunk extraction
    if from < 5 {
        let has_column = match conn.prepare("SELECT parent_chunk_id FROM chunks LIMIT 0") {
            Ok(_) => true,
            Err(rusqlite::Error::SqliteFailure(_, Some(ref msg)))
                if msg.contains("no such column") =>
            {
                false
            }
            Err(e) => return Err(e.into()),
        };
        if !has_column {
            conn.execute_batch(
                "ALTER TABLE chunks ADD COLUMN parent_chunk_id INTEGER REFERENCES chunks(id)",
            )?;
        }
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
