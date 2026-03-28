use yomu::storage::{StorageError, open_db};

#[test]
fn open_db_detects_missing_column_in_stale_schema() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("index.db");

    // Create a DB with old schema (no parent_chunk_id) and fake version = 5
    let conn = rusqlite::Connection::open(&db_path).unwrap();
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

    let err = open_db(&db_path).unwrap_err();
    let msg = err.to_string();
    assert!(
        matches!(err, StorageError::SchemaMismatch { .. }),
        "expected SchemaMismatch, got: {msg}"
    );
    assert!(msg.contains("parent_chunk_id"), "should name the missing column: {msg}");
    assert!(msg.contains("index.db"), "should include the DB path: {msg}");
    assert!(msg.contains("delete this file"), "should suggest deleting the file: {msg}");
}

#[test]
fn open_db_succeeds_with_current_schema() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("index.db");

    let conn = open_db(&db_path);
    assert!(conn.is_ok(), "fresh open_db should succeed: {:?}", conn.err());
}
