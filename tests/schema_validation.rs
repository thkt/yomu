use std::collections::HashSet;

use rusqlite::Connection;
use tempfile::tempdir;

use yomu::storage::open_db;

// T-526: open_db_recovers_stale_v5_schema_via_v9_migration
// Per ADR-0069 the v9 migration drops `chunks` (and dependent tables) and
// recreates them, so opening a stale v5 DB self-recovers without manual
// intervention.
#[test]
fn open_db_recovers_stale_v5_schema_via_v9_migration() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("index.db");

    let conn = Connection::open(&db_path).unwrap();
    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         CREATE TABLE chunks (
             id INTEGER PRIMARY KEY,
             file_path TEXT NOT NULL,
             chunk_type TEXT NOT NULL,
             name TEXT,
             content TEXT NOT NULL,
             start_line INTEGER NOT NULL,
             end_line INTEGER NOT NULL,
             file_hash TEXT NOT NULL
         );
         CREATE TABLE index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
         INSERT INTO index_meta (key, value) VALUES ('schema_version', '5');",
    )
    .unwrap();
    drop(conn);

    let conn = open_db(&db_path).expect("v9 migration must self-recover a v5 DB");

    let cols: HashSet<String> = conn
        .prepare("PRAGMA table_info(chunks)")
        .unwrap()
        .query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    for required in ["parent_chunk_id", "source_kind", "injection_flags"] {
        assert!(
            cols.contains(required),
            "post-migration chunks must include `{required}`, got: {cols:?}"
        );
    }
}

// T-527: open_db_succeeds_with_current_schema
#[test]
fn open_db_succeeds_with_current_schema() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("index.db");

    let conn = open_db(&db_path);
    assert!(
        conn.is_ok(),
        "fresh open_db should succeed: {:?}",
        conn.err()
    );
}
