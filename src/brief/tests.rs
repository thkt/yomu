use rusqlite::Connection;
use tempfile::{TempDir, tempdir};

use super::*;
use crate::storage::{
    EMBEDDING_DIMS, NewChunk, RefKind, Reference, ce, insert_chunk, open_db,
    replace_file_references,
};

fn test_db() -> (Connection, TempDir) {
    let dir = tempdir().unwrap();
    let conn = open_db(&dir.path().join("test.db")).unwrap();
    (conn, dir)
}

fn seed_file(value: &str) -> Seed {
    Seed {
        kind: SeedKind::File,
        value: value.to_owned(),
    }
}

fn task_with_seeds(seeds: Vec<Seed>, depth: u32) -> TaskBrief {
    TaskBrief {
        task: "find search".to_owned(),
        seeds,
        depth,
        max_chunks: 80,
        max_bytes: 80_000,
    }
}

fn insert_test_chunk(conn: &Connection, file_path: &str, name: &'static str, start: u32) {
    insert_chunk(
        conn,
        file_path,
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some(name),
            content: "fn body() {}",
            start_line: start,
            end_line: start + 2,
            parent_index: None,
        },
        "h",
        &ce(vec![0.0_f32; EMBEDDING_DIMS]),
        None,
    )
    .unwrap();
}

// T-600: expand_plan_returns_seed_and_forward_chunks
#[test]
fn expand_plan_returns_seed_and_forward_chunks() {
    let (conn, _dir) = test_db();
    insert_test_chunk(&conn, "src/a.rs", "a", 1);
    insert_test_chunk(&conn, "src/b.rs", "b", 1);
    replace_file_references(
        &conn,
        "src/a.rs",
        &[Reference {
            source_file: "src/a.rs".into(),
            target_file: "src/b.rs".into(),
            symbol_name: None,
            ref_kind: RefKind::Named,
        }],
    )
    .unwrap();

    let task = task_with_seeds(vec![seed_file("src/a.rs")], 1);
    let output = expand_plan(&conn, &task).unwrap();

    assert_eq!(output.chunks.len(), 2);
    assert!(!output.degraded);

    let by_path: std::collections::HashMap<&str, &BriefChunk> = output
        .chunks
        .iter()
        .map(|c| (c.file_path.as_str(), c))
        .collect();
    assert_eq!(
        by_path["src/a.rs"].included_reason,
        ChunkInclusionReason::Seed
    );
    assert_eq!(
        by_path["src/b.rs"].included_reason,
        ChunkInclusionReason::Forward(1)
    );
}
