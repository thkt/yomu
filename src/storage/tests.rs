use tempfile::{TempDir, tempdir};

use super::*;

fn test_db() -> (Connection, TempDir) {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = open_db(&db_path).unwrap();
    (conn, dir)
}

// T-121: init_db_creates_tables
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

// T-122: insert_and_read_chunk
#[test]
fn insert_and_read_chunk() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.1_f32; EMBEDDING_DIMS];

    let id = insert_chunk(
        &conn,
        "src/Button.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("Button"),
            content: "function Button() { return <div/>; }",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "abc123",
        &ce(embedding.clone()),
        None,
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

// T-123: should_reindex_returns_false_for_same_hash
#[test]
fn should_reindex_returns_false_for_same_hash() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/App.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("App"),
            content: "function App() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        },
        "hash_abc",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    assert!(!should_reindex(&conn, "src/App.tsx", "hash_abc").unwrap());
}

// T-124: should_reindex_returns_true_for_different_hash
#[test]
fn should_reindex_returns_true_for_different_hash() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/App.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("App"),
            content: "function App() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        },
        "hash_abc",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    assert!(should_reindex(&conn, "src/App.tsx", "hash_xyz").unwrap());
}

// T-125: should_reindex_returns_true_for_new_file
#[test]
fn should_reindex_returns_true_for_new_file() {
    let (conn, _dir) = test_db();
    assert!(should_reindex(&conn, "src/New.tsx", "any_hash").unwrap());
}

// T-126: get_stats_returns_counts
#[test]
fn get_stats_returns_counts() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("A"),
            content: "code",
            start_line: 1,
            end_line: 5,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useA"),
            content: "code",
            start_line: 6,
            end_line: 10,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/B.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("B"),
            content: "code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h2",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    let stats = get_stats(&conn).unwrap();
    assert_eq!(stats.total_chunks, 3);
    assert_eq!(stats.total_files, 2);
}

// T-127: get_all_file_paths_returns_distinct_paths
#[test]
fn get_all_file_paths_returns_distinct_paths() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("A"),
            content: "code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useA"),
            content: "code",
            start_line: 4,
            end_line: 6,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/B.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("B"),
            content: "code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h2",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    let paths = get_all_file_paths(&conn).unwrap();
    assert_eq!(paths.len(), 2);
    assert!(paths.contains("src/A.tsx"));
    assert!(paths.contains("src/B.tsx"));
}

// T-128: delete_file_chunks_removes_all_chunks_for_file
#[test]
fn delete_file_chunks_removes_all_chunks_for_file() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("A"),
            content: "code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/B.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("B"),
            content: "code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h2",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    delete_file_chunks(&conn, "src/A.tsx").unwrap();

    let stats = get_stats(&conn).unwrap();
    assert_eq!(stats.total_files, 1);
    assert_eq!(stats.total_chunks, 1);

    let paths = get_all_file_paths(&conn).unwrap();
    assert!(!paths.contains("src/A.tsx"));
    assert!(paths.contains("src/B.tsx"));
}

// T-129: replace_file_chunks_replaces_existing
#[test]
fn replace_file_chunks_replaces_existing() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("A"),
            content: "old code",
            start_line: 1,
            end_line: 5,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    let new_chunks = vec![
        NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useA"),
            content: "new code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("B"),
            content: "more code",
            start_line: 4,
            end_line: 8,
            parent_index: None,
        },
    ];
    let embeddings = vec![ce(embedding.clone()), ce(embedding.clone())];

    replace_file_chunks(
        &conn,
        "src/A.tsx",
        &new_chunks,
        &embeddings,
        "h2",
        "import { x } from 'y'",
        &[],
    )
    .unwrap();

    let stats = get_stats(&conn).unwrap();
    assert_eq!(stats.total_chunks, 2);
    assert_eq!(stats.total_files, 1);
    assert!(stats.last_indexed_at.is_some());
}

// T-130: vec_search_returns_ordered_results
#[test]
fn vec_search_returns_ordered_results() {
    let (conn, _dir) = test_db();
    let mut emb_a = vec![0.0_f32; EMBEDDING_DIMS];
    emb_a[0] = 1.0;
    let mut emb_b = vec![0.0_f32; EMBEDDING_DIMS];
    emb_b[1] = 1.0;

    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("A"),
            content: "code a",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(emb_a.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/B.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("B"),
            content: "code b",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h2",
        &ce(emb_b),
        None,
    )
    .unwrap();

    let results = vec_search(&conn, &emb_a, 10, None, &[]).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].chunk.name.as_deref(), Some("A"));
    assert!(results[0].distance <= results[1].distance);
}

// T-131: chunk_type_roundtrip
#[test]
fn chunk_type_roundtrip() {
    let variants = [
        ChunkType::Component,
        ChunkType::Hook,
        ChunkType::TypeDef,
        ChunkType::CssRule,
        ChunkType::HtmlElement,
        ChunkType::TestCase,
        ChunkType::RustFn,
        ChunkType::RustStruct,
        ChunkType::RustEnum,
        ChunkType::RustTrait,
        ChunkType::RustImpl,
        ChunkType::MdSection,
        ChunkType::InnerFn,
        ChunkType::Other,
    ];
    for variant in &variants {
        // Exhaustive match: compile-time guard — adding a ChunkType variant without
        // updating `variants` above causes a compile error here.
        match variant {
            ChunkType::Component
            | ChunkType::Hook
            | ChunkType::TypeDef
            | ChunkType::CssRule
            | ChunkType::HtmlElement
            | ChunkType::TestCase
            | ChunkType::RustFn
            | ChunkType::RustStruct
            | ChunkType::RustEnum
            | ChunkType::RustTrait
            | ChunkType::RustImpl
            | ChunkType::MdSection
            | ChunkType::InnerFn
            | ChunkType::Other => {}
        }
        let s = variant.as_str();
        let restored = ChunkType::from_db(s);
        assert_eq!(&restored, variant, "roundtrip failed for {s}");
    }
}

// T-132: chunk_type_from_db_unknown_defaults_to_other
#[test]
fn chunk_type_from_db_unknown_defaults_to_other() {
    assert_eq!(ChunkType::from_db("nonexistent"), ChunkType::Other);
}

// T-133: delete_file_chunks_also_removes_file_context
#[test]
fn delete_file_chunks_also_removes_file_context() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];
    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("A"),
        content: "code",
        start_line: 1,
        end_line: 5,
        parent_index: None,
    }];
    let embeddings = vec![ce(embedding)];
    replace_file_chunks(
        &conn,
        "src/A.tsx",
        &chunks,
        &embeddings,
        "h1",
        "import React from 'react'",
        &[],
    )
    .unwrap();

    let contexts = get_file_contexts(&conn, &["src/A.tsx"]).unwrap();
    assert_eq!(
        contexts.len(),
        1,
        "pre-condition: file_context should exist"
    );

    delete_file_chunks(&conn, "src/A.tsx").unwrap();

    let contexts = get_file_contexts(&conn, &["src/A.tsx"]).unwrap();
    assert!(
        contexts.is_empty(),
        "file_context should be removed after delete: {contexts:?}"
    );
}

// T-134: replace_file_chunks_stores_file_context
#[test]
fn replace_file_chunks_stores_file_context() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];
    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("App"),
        content: "code",
        start_line: 1,
        end_line: 5,
        parent_index: None,
    }];
    let embeddings = vec![ce(embedding)];
    replace_file_chunks(
        &conn,
        "src/App.tsx",
        &chunks,
        &embeddings,
        "h1",
        "import { useState } from 'react'",
        &[],
    )
    .unwrap();

    let contexts = get_file_contexts(&conn, &["src/App.tsx"]).unwrap();
    assert_eq!(contexts.len(), 1);
    assert_eq!(contexts["src/App.tsx"], "import { useState } from 'react'");
}

// T-135: get_file_contexts_returns_empty_for_missing_files
#[test]
fn get_file_contexts_returns_empty_for_missing_files() {
    let (conn, _dir) = test_db();
    let contexts = get_file_contexts(&conn, &["src/Missing.tsx"]).unwrap();
    assert!(contexts.is_empty());
}

// T-136: get_file_contexts_returns_empty_for_empty_input
#[test]
fn get_file_contexts_returns_empty_for_empty_input() {
    let (conn, _dir) = test_db();
    let contexts = get_file_contexts(&conn, &[]).unwrap();
    assert!(contexts.is_empty());
}

// T-137: get_file_siblings_returns_all_chunks_for_file
#[test]
fn get_file_siblings_returns_all_chunks_for_file() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];
    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("App"),
            content: "code",
            start_line: 1,
            end_line: 5,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useAuth"),
            content: "code",
            start_line: 6,
            end_line: 10,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/B.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("Button"),
            content: "code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h2",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    let siblings = get_file_siblings(&conn, &["src/A.tsx"]).unwrap();
    assert_eq!(siblings.len(), 1);
    let a_siblings = &siblings["src/A.tsx"];
    assert_eq!(a_siblings.len(), 2);
    assert_eq!(a_siblings[0].name.as_deref(), Some("App"));
    assert_eq!(a_siblings[1].name.as_deref(), Some("useAuth"));
}

// T-138: get_file_siblings_returns_empty_for_empty_input
#[test]
fn get_file_siblings_returns_empty_for_empty_input() {
    let (conn, _dir) = test_db();
    let siblings = get_file_siblings(&conn, &[]).unwrap();
    assert!(siblings.is_empty());
}

// T-139: replace_file_references_stores_refs
#[test]
fn replace_file_references_stores_refs() {
    let (conn, _dir) = test_db();
    let refs = vec![
        Reference {
            source_file: "src/A.tsx".into(),
            target_file: "src/B.tsx".into(),
            symbol_name: Some("Button".into()),
            ref_kind: RefKind::Named,
        },
        Reference {
            source_file: "src/A.tsx".into(),
            target_file: "src/C.tsx".into(),
            symbol_name: None,
            ref_kind: RefKind::Namespace,
        },
        Reference {
            source_file: "src/A.tsx".into(),
            target_file: "src/D.tsx".into(),
            symbol_name: Some("useAuth".into()),
            ref_kind: RefKind::Named,
        },
    ];
    replace_file_references(&conn, "src/A.tsx", &refs).unwrap();

    let count = get_reference_count(&conn).unwrap();
    assert_eq!(count, 3);
}

// T-140: replace_file_references_replaces_existing
#[test]
fn replace_file_references_replaces_existing() {
    let (conn, _dir) = test_db();
    let old_refs = vec![
        Reference {
            source_file: "src/A.tsx".into(),
            target_file: "src/B.tsx".into(),
            symbol_name: Some("B".into()),
            ref_kind: RefKind::Named,
        },
        Reference {
            source_file: "src/A.tsx".into(),
            target_file: "src/C.tsx".into(),
            symbol_name: None,
            ref_kind: RefKind::Default,
        },
        Reference {
            source_file: "src/A.tsx".into(),
            target_file: "src/D.tsx".into(),
            symbol_name: Some("D".into()),
            ref_kind: RefKind::Named,
        },
    ];
    replace_file_references(&conn, "src/A.tsx", &old_refs).unwrap();

    let new_refs = vec![
        Reference {
            source_file: "src/A.tsx".into(),
            target_file: "src/E.tsx".into(),
            symbol_name: Some("E".into()),
            ref_kind: RefKind::Named,
        },
        Reference {
            source_file: "src/A.tsx".into(),
            target_file: "src/F.tsx".into(),
            symbol_name: None,
            ref_kind: RefKind::TypeOnly,
        },
    ];
    replace_file_references(&conn, "src/A.tsx", &new_refs).unwrap();

    let count = get_reference_count(&conn).unwrap();
    assert_eq!(count, 2);

    let dependents = get_dependents(&conn, "src/E.tsx").unwrap();
    assert_eq!(dependents.len(), 1);
    assert_eq!(dependents[0].file_path, "src/A.tsx");
}

// T-141: delete_file_chunks_also_removes_references
#[test]
fn delete_file_chunks_also_removes_references() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];
    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("A"),
        content: "code",
        start_line: 1,
        end_line: 5,
        parent_index: None,
    }];
    let embeddings = vec![ce(embedding)];
    replace_file_chunks(&conn, "src/A.tsx", &chunks, &embeddings, "h1", "", &[]).unwrap();

    let refs = vec![Reference {
        source_file: "src/A.tsx".into(),
        target_file: "src/B.tsx".into(),
        symbol_name: Some("B".into()),
        ref_kind: RefKind::Named,
    }];
    replace_file_references(&conn, "src/A.tsx", &refs).unwrap();

    delete_file_chunks(&conn, "src/A.tsx").unwrap();

    let count = get_reference_count(&conn).unwrap();
    assert_eq!(count, 0);
}

// T-142: get_transitive_dependents_chain
#[test]
fn get_transitive_dependents_chain() {
    let (conn, _dir) = test_db();
    replace_file_references(
        &conn,
        "src/A.tsx",
        &[Reference {
            source_file: "src/A.tsx".into(),
            target_file: "src/B.tsx".into(),
            symbol_name: None,
            ref_kind: RefKind::Named,
        }],
    )
    .unwrap();
    replace_file_references(
        &conn,
        "src/B.tsx",
        &[Reference {
            source_file: "src/B.tsx".into(),
            target_file: "src/C.tsx".into(),
            symbol_name: None,
            ref_kind: RefKind::Named,
        }],
    )
    .unwrap();

    let deps = get_transitive_dependents(&conn, "src/C.tsx", 3).unwrap();
    assert_eq!(deps.len(), 2);
    assert_eq!(
        deps[0],
        Dependent {
            file_path: "src/B.tsx".into(),
            depth: 1
        }
    );
    assert_eq!(
        deps[1],
        Dependent {
            file_path: "src/A.tsx".into(),
            depth: 2
        }
    );
}

// T-143: get_transitive_dependents_circular
#[test]
fn get_transitive_dependents_circular() {
    let (conn, _dir) = test_db();
    replace_file_references(
        &conn,
        "src/A.tsx",
        &[Reference {
            source_file: "src/A.tsx".into(),
            target_file: "src/B.tsx".into(),
            symbol_name: None,
            ref_kind: RefKind::Named,
        }],
    )
    .unwrap();
    replace_file_references(
        &conn,
        "src/B.tsx",
        &[Reference {
            source_file: "src/B.tsx".into(),
            target_file: "src/A.tsx".into(),
            symbol_name: None,
            ref_kind: RefKind::Named,
        }],
    )
    .unwrap();

    let deps = get_transitive_dependents(&conn, "src/A.tsx", 5).unwrap();
    assert_eq!(deps.len(), 1);
    assert_eq!(
        deps[0],
        Dependent {
            file_path: "src/B.tsx".into(),
            depth: 1
        }
    );
}

fn get_chunk_ids(conn: &Connection, file_path: &str) -> Vec<i64> {
    let mut stmt = conn
        .prepare("SELECT id FROM chunks WHERE file_path = ?1 ORDER BY id")
        .unwrap();
    stmt.query_map([file_path], |row| row.get::<_, i64>(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect()
}

fn vec_chunks_count(conn: &Connection) -> u32 {
    conn.query_row("SELECT COUNT(*) FROM vec_chunks", [], |row| row.get(0))
        .unwrap()
}

// T-144: replace_file_chunks_only_inserts_chunks_without_embeddings
#[test]
fn replace_file_chunks_only_inserts_chunks_without_embeddings() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("Button"),
            content: "function Button() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useClick"),
            content: "function useClick() {}",
            start_line: 5,
            end_line: 8,
            parent_index: None,
        },
    ];
    let refs: Vec<Reference> = vec![];

    replace_file_chunks_only(
        &conn,
        "src/Button.tsx",
        &chunks,
        "hash_a",
        "import React from 'react'",
        &refs,
        None,
    )
    .unwrap();

    let chunk_count: u32 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE file_path = 'src/Button.tsx'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(chunk_count, 2);

    assert_eq!(vec_chunks_count(&conn), 0);

    let contexts = get_file_contexts(&conn, &["src/Button.tsx"]).unwrap();
    assert_eq!(contexts["src/Button.tsx"], "import React from 'react'");

    let last: String = conn
        .query_row(
            "SELECT value FROM index_meta WHERE key = 'last_indexed_at'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(!last.is_empty());
}

// T-145: replace_file_chunks_only_deletes_old_embeddings
#[test]
fn replace_file_chunks_only_deletes_old_embeddings() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/App.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("App"),
            content: "old code",
            start_line: 1,
            end_line: 5,
            parent_index: None,
        },
        "hash_old",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();
    assert_eq!(vec_chunks_count(&conn), 1);

    let new_chunks = vec![NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("AppV2"),
        content: "new code",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/App.tsx", &new_chunks, "hash_new", "", &[], None).unwrap();

    assert_eq!(vec_chunks_count(&conn), 0);

    let chunk_count: u32 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE file_path = 'src/App.tsx'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(chunk_count, 1);

    let name: String = conn
        .query_row(
            "SELECT name FROM chunks WHERE file_path = 'src/App.tsx'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(name, "AppV2");
}

// T-146: add_embeddings_inserts_into_vec_chunks
#[test]
fn add_embeddings_inserts_into_vec_chunks() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("Card"),
            content: "function Card() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useCard"),
            content: "function useCard() {}",
            start_line: 5,
            end_line: 8,
            parent_index: None,
        },
    ];
    replace_file_chunks_only(&conn, "src/Card.tsx", &chunks, "hash_c", "", &[], None).unwrap();
    assert_eq!(vec_chunks_count(&conn), 0);

    let ids = get_chunk_ids(&conn, "src/Card.tsx");
    assert_eq!(ids.len(), 2);

    let mut emb1 = vec![0.0_f32; EMBEDDING_DIMS];
    emb1[0] = 1.0;
    let mut emb2 = vec![0.0_f32; EMBEDDING_DIMS];
    emb2[1] = 1.0;

    let embeddings = vec![(ids[0], ce(emb1)), (ids[1], ce(emb2))];
    let inserted = add_chunked_embeddings(&conn, &embeddings).unwrap();

    assert_eq!(inserted, 2);
    assert_eq!(vec_chunks_count(&conn), 2);
}

// T-147: add_embeddings_skips_already_embedded
#[test]
fn add_embeddings_skips_already_embedded() {
    let (conn, _dir) = test_db();

    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("Modal"),
        content: "function Modal() {}",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/Modal.tsx", &chunks, "hash_m", "", &[], None).unwrap();

    let ids = get_chunk_ids(&conn, "src/Modal.tsx");
    let emb = vec![0.5_f32; EMBEDDING_DIMS];
    let embeddings = vec![(ids[0], ce(emb.clone()))];

    let inserted = add_chunked_embeddings(&conn, &embeddings).unwrap();
    assert_eq!(inserted, 1);

    let embeddings_dup = vec![(ids[0], ce(emb))];
    let inserted_dup = add_chunked_embeddings(&conn, &embeddings_dup).unwrap();
    assert_eq!(inserted_dup, 0);

    assert_eq!(vec_chunks_count(&conn), 1);
}

// T-148: get_unembedded_file_paths_returns_only_unembedded
#[test]
fn get_unembedded_file_paths_returns_only_unembedded() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("A"),
            content: "code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    let chunks_b = vec![
        NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("B"),
            content: "code b",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useB"),
            content: "code b2",
            start_line: 5,
            end_line: 8,
            parent_index: None,
        },
    ];
    replace_file_chunks_only(&conn, "src/B.tsx", &chunks_b, "h2", "", &[], None).unwrap();

    let chunks_c = vec![NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("C"),
        content: "code c",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/C.tsx", &chunks_c, "h3", "", &[], None).unwrap();

    let unembedded = get_unembedded_file_paths(&conn).unwrap();

    let paths: Vec<&str> = unembedded.iter().map(|(p, _)| p.as_str()).collect();
    assert!(!paths.contains(&"src/A.tsx"));
    assert!(paths.contains(&"src/B.tsx"));
    assert!(paths.contains(&"src/C.tsx"));

    let b_count = unembedded.iter().find(|(p, _)| p == "src/B.tsx").unwrap().1;
    assert_eq!(b_count, 2);

    let c_count = unembedded.iter().find(|(p, _)| p == "src/C.tsx").unwrap().1;
    assert_eq!(c_count, 1);
}

// T-149: needs_embedding_returns_true_for_chunk_only_file
#[test]
fn needs_embedding_returns_true_for_chunk_only_file() {
    let (conn, _dir) = test_db();

    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("Nav"),
        content: "function Nav() {}",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/Nav.tsx", &chunks, "hash_nav", "", &[], None).unwrap();

    assert!(needs_embedding(&conn, "src/Nav.tsx", "hash_nav").unwrap());
}

// T-150: needs_embedding_returns_false_for_fully_embedded
#[test]
fn needs_embedding_returns_false_for_fully_embedded() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/Footer.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("Footer"),
            content: "function Footer() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "hash_footer",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    assert!(!needs_embedding(&conn, "src/Footer.tsx", "hash_footer").unwrap());
}

// T-151: needs_embedding_returns_true_when_hash_changed
#[test]
fn needs_embedding_returns_true_when_hash_changed() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/Header.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("Header"),
            content: "function Header() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "hash_v1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    assert!(needs_embedding(&conn, "src/Header.tsx", "hash_v2").unwrap());
}

// T-152: get_stats_reports_embedded_vs_total_chunks
#[test]
fn get_stats_reports_embedded_vs_total_chunks() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("A"),
            content: "code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useA"),
            content: "code",
            start_line: 5,
            end_line: 8,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    let chunks_b = vec![NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("B"),
        content: "code b",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/B.tsx", &chunks_b, "h2", "", &[], None).unwrap();

    let stats = get_stats(&conn).unwrap();
    assert_eq!(stats.total_chunks, 3);
    assert_eq!(stats.total_files, 2);
    assert_eq!(stats.embedded_chunks, 2);
    assert!(stats.embedded_chunks < stats.total_chunks);
}

// T-153: get_files_by_import_count_returns_most_imported_first
#[test]
fn get_files_by_import_count_returns_most_imported_first() {
    let (conn, _dir) = test_db();

    for (path, hash) in [
        ("src/utils.tsx", "h1"),
        ("src/Button.tsx", "h2"),
        ("src/App.tsx", "h3"),
    ] {
        let chunks = vec![NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("X"),
            content: "code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        }];
        replace_file_chunks_only(&conn, path, &chunks, hash, "", &[], None).unwrap();
    }

    replace_file_references(
        &conn,
        "src/App.tsx",
        &[
            Reference {
                source_file: "src/App.tsx".into(),
                target_file: "src/utils.tsx".into(),
                symbol_name: Some("format".into()),
                ref_kind: RefKind::Named,
            },
            Reference {
                source_file: "src/App.tsx".into(),
                target_file: "src/Button.tsx".into(),
                symbol_name: Some("Button".into()),
                ref_kind: RefKind::Named,
            },
        ],
    )
    .unwrap();
    replace_file_references(
        &conn,
        "src/Button.tsx",
        &[Reference {
            source_file: "src/Button.tsx".into(),
            target_file: "src/utils.tsx".into(),
            symbol_name: Some("cn".into()),
            ref_kind: RefKind::Named,
        }],
    )
    .unwrap();
    replace_file_references(
        &conn,
        "src/Other.tsx",
        &[Reference {
            source_file: "src/Other.tsx".into(),
            target_file: "src/utils.tsx".into(),
            symbol_name: None,
            ref_kind: RefKind::Namespace,
        }],
    )
    .unwrap();

    let ordered = get_files_by_import_count(&conn).unwrap();

    // utils (3 refs) > Button (1 ref) > App (0 refs)
    assert_eq!(ordered.len(), 3);
    assert_eq!(ordered[0], "src/utils.tsx");
    assert_eq!(ordered[1], "src/Button.tsx");
    assert_eq!(ordered[2], "src/App.tsx");
}

// T-154: get_files_by_import_count_boosts_hook_component_files
#[test]
fn get_files_by_import_count_boosts_hook_component_files() {
    let (conn, _dir) = test_db();

    // Hook file: 0 import refs
    let hook_chunks = vec![NewChunk {
        chunk_type: &ChunkType::Hook,
        name: Some("useAuth"),
        content: "function useAuth() {}",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/useAuth.tsx", &hook_chunks, "h1", "", &[], None).unwrap();

    // TypeDef file: 0 import refs
    let typedef_chunks = vec![NewChunk {
        chunk_type: &ChunkType::TypeDef,
        name: Some("AuthConfig"),
        content: "interface AuthConfig {}",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/types.tsx", &typedef_chunks, "h2", "", &[], None).unwrap();

    // Component file: 0 import refs
    let comp_chunks = vec![NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("LoginButton"),
        content: "function LoginButton() {}",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(
        &conn,
        "src/LoginButton.tsx",
        &comp_chunks,
        "h3",
        "",
        &[],
        None,
    )
    .unwrap();

    let ordered = get_files_by_import_count(&conn).unwrap();

    // hook/component files (+3 boost) should come before type_def (no boost)
    assert_eq!(ordered.len(), 3);
    // First two should be hook or component files (both have +3 boost)
    let boosted: Vec<&str> = ordered.iter().take(2).map(String::as_str).collect();
    assert!(
        boosted.contains(&"src/useAuth.tsx"),
        "hook file should be boosted: {ordered:?}"
    );
    assert!(
        boosted.contains(&"src/LoginButton.tsx"),
        "component file should be boosted: {ordered:?}"
    );
    // Last should be typedef
    assert_eq!(
        ordered[2], "src/types.tsx",
        "typedef file should be last: {ordered:?}"
    );
}

// T-155: vec_search_sets_semantic_match_source
#[test]
fn vec_search_sets_semantic_match_source() {
    let (conn, _dir) = test_db();
    let mut emb = vec![0.0_f32; EMBEDDING_DIMS];
    emb[0] = 1.0;

    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("A"),
            content: "code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(emb.clone()),
        None,
    )
    .unwrap();

    let results = vec_search(&conn, &emb, 10, None, &[]).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].match_source, MatchSource::Semantic);
}

// T-156: vec_search_sets_initial_score
#[test]
fn vec_search_sets_initial_score() {
    let (conn, _dir) = test_db();
    let mut emb = vec![0.0_f32; EMBEDDING_DIMS];
    emb[0] = 1.0;

    insert_chunk(
        &conn,
        "src/A.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("A"),
            content: "code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(emb.clone()),
        None,
    )
    .unwrap();

    let results = vec_search(&conn, &emb, 10, None, &[]).unwrap();
    assert_eq!(results.len(), 1);

    // score should be initialized to 1.0 / (1.0 + distance)
    let expected_score = 1.0 / (1.0 + results[0].distance);
    assert!(
        (results[0].score - expected_score).abs() < 1e-6,
        "expected score {expected_score}, got {}",
        results[0].score
    );
}

// T-157: get_import_counts_returns_correct_counts
#[test]
fn get_import_counts_returns_correct_counts() {
    let (conn, _dir) = test_db();

    // Set up files with references
    for (path, hash) in [
        ("src/utils.tsx", "h1"),
        ("src/Button.tsx", "h2"),
        ("src/App.tsx", "h3"),
    ] {
        let chunks = vec![NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("X"),
            content: "code",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        }];
        replace_file_chunks_only(&conn, path, &chunks, hash, "", &[], None).unwrap();
    }

    // utils is imported 3 times, Button 1 time, App 0 times
    replace_file_references(
        &conn,
        "src/App.tsx",
        &[
            Reference {
                source_file: "src/App.tsx".into(),
                target_file: "src/utils.tsx".into(),
                symbol_name: None,
                ref_kind: RefKind::Named,
            },
            Reference {
                source_file: "src/App.tsx".into(),
                target_file: "src/Button.tsx".into(),
                symbol_name: None,
                ref_kind: RefKind::Named,
            },
        ],
    )
    .unwrap();
    replace_file_references(
        &conn,
        "src/Button.tsx",
        &[Reference {
            source_file: "src/Button.tsx".into(),
            target_file: "src/utils.tsx".into(),
            symbol_name: None,
            ref_kind: RefKind::Named,
        }],
    )
    .unwrap();
    replace_file_references(
        &conn,
        "src/Other.tsx",
        &[Reference {
            source_file: "src/Other.tsx".into(),
            target_file: "src/utils.tsx".into(),
            symbol_name: None,
            ref_kind: RefKind::Named,
        }],
    )
    .unwrap();

    let counts =
        get_import_counts(&conn, &["src/utils.tsx", "src/Button.tsx", "src/App.tsx"]).unwrap();

    assert_eq!(counts.len(), 3);
    assert_eq!(counts["src/utils.tsx"], 3);
    assert_eq!(counts["src/Button.tsx"], 1);
    assert_eq!(counts["src/App.tsx"], 0);
}

// T-158: get_import_counts_returns_empty_for_empty_input
#[test]
fn get_import_counts_returns_empty_for_empty_input() {
    let (conn, _dir) = test_db();
    let counts = get_import_counts(&conn, &[]).unwrap();
    assert!(counts.is_empty());
}

// T-159: fts_chunks_deleted_with_file
#[test]
fn fts_chunks_deleted_with_file() {
    let (conn, _dir) = test_db();

    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Hook,
        name: Some("useX"),
        content: "function useX() { return state; }",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/x.tsx", &chunks, "h1", "", &[], None).unwrap();

    // Should find it before delete
    let results = search_by_fts(&conn, &["state"], None, &HashSet::new(), None, 10, &[]).unwrap();
    assert_eq!(results.len(), 1);

    // Delete file chunks
    delete_file_chunks(&conn, "src/x.tsx").unwrap();

    // Should not find it after delete
    let results = search_by_fts(&conn, &["state"], None, &HashSet::new(), None, 10, &[]).unwrap();
    assert!(results.is_empty());
}

// T-160: fts5_migration_from_v2
#[test]
fn fts5_migration_from_v2() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = open_db(&db_path).unwrap();

    // Insert a chunk (schema is v3, fts_chunks populated via insert_chunk_row)
    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Hook,
        name: Some("useX"),
        content: "function useX() { return loading; }",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/x.tsx", &chunks, "h1", "", &[], None).unwrap();

    // Simulate a v2 DB by clearing fts_chunks and rolling back schema_version
    conn.execute_batch("DELETE FROM fts_chunks").unwrap();
    conn.execute(
        "UPDATE index_meta SET value = '2' WHERE key = 'schema_version'",
        [],
    )
    .unwrap();

    // Verify FTS5 is empty
    let results = search_by_fts(&conn, &["loading"], None, &HashSet::new(), None, 10, &[]).unwrap();
    assert!(
        results.is_empty(),
        "fts_chunks should be empty before migration"
    );

    // Re-init triggers migration v2→v3
    drop(conn);
    let conn = open_db(&db_path).unwrap();

    // Migration should have populated fts_chunks from existing chunks
    let results = search_by_fts(&conn, &["loading"], None, &HashSet::new(), None, 10, &[]).unwrap();
    assert_eq!(results.len(), 1, "migration should populate fts_chunks");
    assert_eq!(results[0].chunk.name.as_deref(), Some("useX"));
}

// T-161: fts5_handles_special_characters_in_content
#[test]
fn fts5_handles_special_characters_in_content() {
    let (conn, _dir) = test_db();

    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("App"),
        content: "function App() { return <div className={`container ${active}`}>{data?.name}</div>; }",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/App.tsx", &chunks, "h1", "", &[], None).unwrap();

    let results =
        search_by_fts(&conn, &["container"], None, &HashSet::new(), None, 10, &[]).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.name.as_deref(), Some("App"));
}

// T-162: get_unembedded_chunks_for_file_returns_rows
#[test]
fn get_unembedded_chunks_for_file_returns_rows() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("Card"),
            content: "function Card() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useCard"),
            content: "function useCard() {}",
            start_line: 5,
            end_line: 8,
            parent_index: None,
        },
    ];
    replace_file_chunks_only(&conn, "src/Card.tsx", &chunks, "hash_card", "", &[], None).unwrap();

    let rows = get_unembedded_chunks_for_file(&conn, "src/Card.tsx").unwrap();
    assert_eq!(rows.len(), 2);

    let types: Vec<&str> = rows.iter().map(|r| r.chunk_type.as_str()).collect();
    assert!(types.contains(&"component"));
    assert!(types.contains(&"hook"));

    let contents: Vec<&str> = rows.iter().map(|r| r.content.as_str()).collect();
    assert!(contents.contains(&"function Card() {}"));
    assert!(contents.contains(&"function useCard() {}"));
}

// T-163: get_unembedded_chunks_for_file_excludes_embedded
#[test]
fn get_unembedded_chunks_for_file_excludes_embedded() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/Nav.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("Nav"),
            content: "function Nav() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    let rows = get_unembedded_chunks_for_file(&conn, "src/Nav.tsx").unwrap();
    assert!(rows.is_empty());
}

// T-164: get_imports_for_file_returns_stored_imports
#[test]
fn get_imports_for_file_returns_stored_imports() {
    let (conn, _dir) = test_db();

    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("App"),
        content: "function App() {}",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    let imports = "import { useState } from 'react'\nimport { Button } from './Button'";
    replace_file_chunks_only(&conn, "src/App.tsx", &chunks, "h1", imports, &[], None).unwrap();

    let result = get_imports_for_file(&conn, "src/App.tsx").unwrap();
    assert_eq!(result, imports);
}

// T-165: get_imports_for_file_returns_empty_for_missing
#[test]
fn get_imports_for_file_returns_empty_for_missing() {
    let (conn, _dir) = test_db();
    let result = get_imports_for_file(&conn, "src/nonexistent.tsx").unwrap();
    assert!(result.is_empty());
}

// --- FTS5 short term expansion ---

// T-166: fts_chunks_vocab_exists_on_new_db
#[test]
fn fts_chunks_vocab_exists_on_new_db() {
    let (conn, _dir) = test_db();

    let exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type = 'table' AND name = 'fts_chunks_vocab'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(exists, "fts_chunks_vocab table should exist after open_db");
}

// T-167: migration_v3_to_v4_creates_vocab_table
#[test]
fn migration_v3_to_v4_creates_vocab_table() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = open_db(&db_path).unwrap();

    // Seed data before simulating v3
    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Other,
        name: Some("migrationTest"),
        content: "function migrationTest() { return true; }",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/migrate.ts", &chunks, "h1", "", &[], None).unwrap();

    // Simulate a v3 DB by rolling back schema_version and dropping vocab table
    conn.execute(
        "UPDATE index_meta SET value = '3' WHERE key = 'schema_version'",
        [],
    )
    .unwrap();
    let _ = conn.execute_batch("DROP TABLE IF EXISTS fts_chunks_vocab");
    drop(conn);

    // Re-open triggers migration v3 -> v4
    let conn = open_db(&db_path).unwrap();

    let version: String = conn
        .query_row(
            "SELECT value FROM index_meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(version, "8", "schema_version should be 8 after migration");

    let exists: bool = conn
        .query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type = 'table' AND name = 'fts_chunks_vocab'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(
        exists,
        "fts_chunks_vocab should exist after v3->v4 migration"
    );

    // Verify data preserved: chunk should still be searchable
    let results = search_by_fts(
        &conn,
        &["migrationTest"],
        None,
        &HashSet::new(),
        None,
        10,
        &[],
    )
    .unwrap();
    assert!(
        !results.is_empty(),
        "seeded chunk should survive v3->v4 migration"
    );
    assert_eq!(
        results[0].chunk.name.as_deref(),
        Some("migrationTest"),
        "chunk name should be preserved after migration"
    );
}

// T-168: get_file_mtimes_returns_stored_mtime
#[test]
fn get_file_mtimes_returns_stored_mtime() {
    let (conn, _dir) = test_db();
    let now = 1711600000i64;
    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("A"),
        content: "function A() {}",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/A.tsx", &chunks, "h1", "", &[], Some(now)).unwrap();
    replace_file_chunks_only(&conn, "src/B.tsx", &chunks, "h2", "", &[], None).unwrap();

    let mtimes = get_file_mtimes(&conn, &["src/A.tsx", "src/B.tsx", "src/C.tsx"]).unwrap();
    assert_eq!(mtimes.len(), 1, "only A should have mtime");
    assert_eq!(mtimes["src/A.tsx"], now);
}

// T-169: get_file_mtimes_empty_input_returns_empty
#[test]
fn get_file_mtimes_empty_input_returns_empty() {
    let (conn, _dir) = test_db();
    let mtimes = get_file_mtimes(&conn, &[]).unwrap();
    assert!(mtimes.is_empty());
}

// T-170: replace_file_chunks_rejects_length_mismatch
#[test]
fn replace_file_chunks_rejects_length_mismatch() {
    let (conn, _dir) = test_db();
    let chunks = vec![
        NewChunk {
            chunk_type: &ChunkType::Other,
            name: Some("A"),
            content: "fn a() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        },
        NewChunk {
            chunk_type: &ChunkType::Other,
            name: Some("B"),
            content: "fn b() {}",
            start_line: 2,
            end_line: 2,
            parent_index: None,
        },
    ];
    let embeddings = vec![ce(vec![0.0_f32; EMBEDDING_DIMS])]; // 1 embedding for 2 chunks

    let result = replace_file_chunks(&conn, "src/test.rs", &chunks, &embeddings, "h1", "", &[]);
    assert!(result.is_err(), "should reject mismatched lengths");
    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("2") && err.contains("1"),
        "error should mention chunk/embedding counts, got: {err}"
    );
}

// T-171: insert_chunk_rejects_wrong_embedding_dims
#[test]
fn insert_chunk_rejects_wrong_embedding_dims() {
    let (conn, _dir) = test_db();
    let wrong_dims = vec![0.0_f32; 10]; // 10 instead of 768

    let result = insert_chunk(
        &conn,
        "src/test.rs",
        &NewChunk {
            chunk_type: &ChunkType::Other,
            name: Some("A"),
            content: "fn a() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        },
        "h1",
        &ce(wrong_dims),
        None,
    );
    assert!(result.is_err(), "should reject wrong embedding dimensions");
}

// T-172: fts_automerge_guard_restores_on_drop
#[test]
fn fts_automerge_guard_restores_on_drop() {
    let (conn, _dir) = test_db();

    replace_file_chunks_only(
        &conn,
        "src/A.tsx",
        &[NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("A"),
            content: "function A() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        }],
        "h1",
        "",
        &[],
        None,
    )
    .unwrap();

    {
        let _guard = FtsAutomergeGuard::new(&conn).unwrap();
        // guard drops here, restoring automerge
    }

    // FTS operations should work normally after guard drop
    replace_file_chunks_only(
        &conn,
        "src/B.tsx",
        &[NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("B"),
            content: "function B() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        }],
        "h2",
        "",
        &[],
        None,
    )
    .unwrap();
    let results =
        search_by_fts(&conn, &["function"], None, &HashSet::new(), None, 10, &[]).unwrap();
    assert!(
        results.len() >= 2,
        "FTS should work after guard drop, got {} results",
        results.len()
    );
}

// T-543: new_chunk_with_parent_index_stores_parent_chunk_id
#[test]
fn new_chunk_with_parent_index_stores_parent_chunk_id() {
    let (conn, _dir) = test_db();

    let parent = NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("Dashboard"),
        content: "function Dashboard() { return <div/>; }",
        start_line: 1,
        end_line: 60,
        parent_index: None,
    };
    let child = NewChunk {
        chunk_type: &ChunkType::InnerFn,
        name: Some("handleClick"),
        content: "const handleClick = () => {}",
        start_line: 10,
        end_line: 20,
        parent_index: Some(0),
    };

    replace_file_chunks_only(
        &conn,
        "src/Dashboard.tsx",
        &[parent, child],
        "hash1",
        "",
        &[],
        None,
    )
    .unwrap();

    let rows: Vec<(i64, String, Option<i64>)> = {
        let mut stmt = conn
            .prepare(
                "SELECT id, chunk_type, parent_chunk_id FROM chunks \
                 WHERE file_path = 'src/Dashboard.tsx' ORDER BY id",
            )
            .unwrap();
        stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    };

    assert_eq!(rows.len(), 2);
    let parent_id = rows[0].0;
    assert_eq!(rows[0].1, "component");
    assert_eq!(rows[0].2, None);
    assert_eq!(rows[1].1, "inner_fn");
    assert_eq!(rows[1].2, Some(parent_id));
}

// T-546: out_of_order_chunks_resolves_to_null
#[test]
fn out_of_order_chunks_resolves_to_null() {
    let (conn, _dir) = test_db();

    let child = NewChunk {
        chunk_type: &ChunkType::InnerFn,
        name: Some("handleClick"),
        content: "const handleClick = () => {}",
        start_line: 10,
        end_line: 20,
        parent_index: Some(1),
    };
    let parent = NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("Dashboard"),
        content: "function Dashboard() { return <div/>; }",
        start_line: 1,
        end_line: 60,
        parent_index: None,
    };

    replace_file_chunks_only(
        &conn,
        "src/Dashboard.tsx",
        &[child, parent],
        "hash1",
        "",
        &[],
        None,
    )
    .unwrap();

    let child_parent: Option<i64> = conn
        .query_row(
            "SELECT parent_chunk_id FROM chunks WHERE chunk_type = 'inner_fn' AND file_path = 'src/Dashboard.tsx'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        child_parent, None,
        "out-of-order parent_index should resolve to NULL"
    );
}

// T-549: innerfn_in_embed_path_not_in_vec_chunks_yes_in_fts
#[test]
fn innerfn_in_embed_path_not_in_vec_chunks_yes_in_fts() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    let parent = NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("Form"),
        content: "function Form() { return <form/>; }",
        start_line: 1,
        end_line: 60,
        parent_index: None,
    };
    let child = NewChunk {
        chunk_type: &ChunkType::InnerFn,
        name: Some("handleSubmit"),
        content: "const handleSubmit = () => {}",
        start_line: 10,
        end_line: 20,
        parent_index: Some(0),
    };

    let data = FileData {
        file_path: "src/Form.tsx",
        chunks: &[parent, child],
        file_hash: "hash_form",
        imports_text: "",
        refs: &[],
        mtime_epoch: None,
    };
    replace_file_chunks_with(&conn, &data, &[ce(embedding)]).unwrap();

    let total: u32 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE file_path = 'src/Form.tsx'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(total, 2);
    assert_eq!(vec_chunks_count(&conn), 1);

    let fts_count: u32 = conn
        .query_row("SELECT COUNT(*) FROM fts_chunks", [], |row| row.get(0))
        .unwrap();
    assert_eq!(fts_count, 2);
}

// T-560: get_stats_returns_embeddable_and_total_chunks
#[test]
fn get_stats_returns_embeddable_and_total_chunks() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("App"),
            content: "function App() { return <div/>; }",
            start_line: 1,
            end_line: 60,
            parent_index: None,
        },
        NewChunk {
            chunk_type: &ChunkType::InnerFn,
            name: Some("handleClick"),
            content: "const handleClick = () => {}",
            start_line: 10,
            end_line: 20,
            parent_index: Some(0),
        },
        NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useAuth"),
            content: "function useAuth() {}",
            start_line: 70,
            end_line: 80,
            parent_index: None,
        },
    ];
    replace_file_chunks_only(&conn, "src/App.tsx", &chunks, "h1", "", &[], None).unwrap();

    let stats = get_stats(&conn).unwrap();

    assert_eq!(stats.total_chunks, 3);
    assert_eq!(stats.embeddable_chunks, 2);
}

// T-017: embed_percentage_uses_embeddable_chunks
#[test]
fn embed_percentage_uses_embeddable_chunks() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/Card.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("Card"),
            content: "function Card() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    let stats_before = get_stats(&conn).unwrap();
    let pct_before = stats_before.embed_percentage();

    insert_chunk_row(
        &conn,
        "src/Card.tsx",
        &NewChunk {
            chunk_type: &ChunkType::InnerFn,
            name: Some("onClick"),
            content: "const onClick = () => {}",
            start_line: 5,
            end_line: 10,
            parent_index: Some(0),
        },
        "h1",
        None,
    )
    .unwrap();

    let stats_after = get_stats(&conn).unwrap();
    let pct_after = stats_after.embed_percentage();

    assert_eq!(stats_after.total_chunks, 2);
    assert_eq!(stats_after.embeddable_chunks, 1);
    assert_eq!(pct_before, pct_after);
    assert_eq!(pct_after, 100);
}

// T-018: embed_path_with_innerfn_no_length_mismatch
#[test]
fn embed_path_with_innerfn_no_length_mismatch() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS];

    let parent = NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("Panel"),
        content: "function Panel() { return <div/>; }",
        start_line: 1,
        end_line: 60,
        parent_index: None,
    };
    let inner1 = NewChunk {
        chunk_type: &ChunkType::InnerFn,
        name: Some("toggle"),
        content: "const toggle = () => {}",
        start_line: 10,
        end_line: 15,
        parent_index: Some(0),
    };
    let inner2 = NewChunk {
        chunk_type: &ChunkType::InnerFn,
        name: Some("reset"),
        content: "const reset = () => {}",
        start_line: 20,
        end_line: 25,
        parent_index: Some(0),
    };

    let data = FileData {
        file_path: "src/Panel.tsx",
        chunks: &[parent, inner1, inner2],
        file_hash: "hash_panel",
        imports_text: "",
        refs: &[],
        mtime_epoch: None,
    };

    let result = replace_file_chunks_with(&conn, &data, &[ce(embedding)]);
    assert!(result.is_ok(), "expected Ok, got: {result:?}");

    let total: u32 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE file_path = 'src/Panel.tsx'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(total, 3);
    assert_eq!(vec_chunks_count(&conn), 1);
}

// T-019: idf_uses_total_chunks_innerfn_included_non_negative
#[test]
fn idf_uses_total_chunks_innerfn_included_non_negative() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("List"),
            content: "function List() { return <ul/>; }",
            start_line: 1,
            end_line: 60,
            parent_index: None,
        },
        NewChunk {
            chunk_type: &ChunkType::InnerFn,
            name: Some("renderItem"),
            content: "const renderItem = (item) => <li>{item}</li>",
            start_line: 10,
            end_line: 20,
            parent_index: Some(0),
        },
    ];
    replace_file_chunks_only(&conn, "src/List.tsx", &chunks, "h1", "", &[], None).unwrap();

    let stats = get_stats(&conn).unwrap();
    assert_eq!(stats.total_chunks, 2);

    let keywords = ["renderitem"];
    let keyword_refs: Vec<&str> = keywords.to_vec();
    let dfs = get_keyword_doc_frequencies(&conn, &keyword_refs, stats.total_chunks).unwrap();

    let total = stats.total_chunks.max(1) as f32;
    for &df in &dfs {
        assert!(
            df <= stats.total_chunks,
            "df ({df}) > total_chunks ({})",
            stats.total_chunks
        );
        let idf = (total / (df.max(1) as f32)).ln();
        assert!(
            idf >= 0.0,
            "IDF should be non-negative, got {idf} for df={df}, total={total}"
        );
    }
}

// T-551: chunk_from_row_reads_parent_chunk_id
#[test]
fn chunk_from_row_reads_parent_chunk_id() {
    let (conn, _dir) = test_db();

    let parent = NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("UserForm"),
        content: "function UserForm() { return <form/>; }",
        start_line: 1,
        end_line: 60,
        parent_index: None,
    };
    let child = NewChunk {
        chunk_type: &ChunkType::InnerFn,
        name: Some("handleSubmit"),
        content: "const handleSubmit = (event) => { event.preventDefault(); submit(); }",
        start_line: 10,
        end_line: 15,
        parent_index: Some(0),
    };

    replace_file_chunks_only(
        &conn,
        "src/UserForm.tsx",
        &[parent, child],
        "hash_uf",
        "",
        &[],
        None,
    )
    .unwrap();

    let parent_id: i64 = conn
        .query_row(
            "SELECT id FROM chunks WHERE file_path = 'src/UserForm.tsx' AND chunk_type = 'component'",
            [],
            |row| row.get(0),
        )
        .unwrap();

    let results = search_by_fts(
        &conn,
        &["handlesubmit"],
        None,
        &HashSet::new(),
        None,
        10,
        &[],
    )
    .unwrap();

    assert!(
        !results.is_empty(),
        "should find handleSubmit via content search"
    );
    let innerfn_result = results
        .iter()
        .find(|r| r.chunk.chunk_type == ChunkType::InnerFn)
        .expect("should have an InnerFn result");

    assert_eq!(
        innerfn_result.chunk.parent_chunk_id,
        Some(parent_id),
        "chunk_from_row should read parent_chunk_id from DB, not hardcode None"
    );
}

// T-552: chunk_from_row_reads_parent_chunk_id_name_search
#[test]
fn chunk_from_row_reads_parent_chunk_id_name_search() {
    let (conn, _dir) = test_db();

    let parent = NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("UserForm"),
        content: "function UserForm() { return <form/>; }",
        start_line: 1,
        end_line: 60,
        parent_index: None,
    };
    let child = NewChunk {
        chunk_type: &ChunkType::InnerFn,
        name: Some("handleSubmit"),
        content: "const handleSubmit = (event) => { event.preventDefault(); }",
        start_line: 10,
        end_line: 15,
        parent_index: Some(0),
    };

    replace_file_chunks_only(
        &conn,
        "src/UserForm.tsx",
        &[parent, child],
        "hash_uf2",
        "",
        &[],
        None,
    )
    .unwrap();

    let parent_id: i64 = conn
        .query_row(
            "SELECT id FROM chunks WHERE file_path = 'src/UserForm.tsx' AND chunk_type = 'component'",
            [],
            |row| row.get(0),
        )
        .unwrap();

    let results = search_by_fts(
        &conn,
        &["handlesubmit"],
        None,
        &HashSet::new(),
        None,
        10,
        &[],
    )
    .unwrap();

    assert!(
        !results.is_empty(),
        "should find handleSubmit via name search"
    );
    let innerfn_result = results
        .iter()
        .find(|r| r.chunk.chunk_type == ChunkType::InnerFn)
        .expect("should have an InnerFn result");

    assert_eq!(
        innerfn_result.chunk.parent_chunk_id,
        Some(parent_id),
        "chunk_from_row should read parent_chunk_id for name search results"
    );
}

// T-554: parent_chunk_search_result_has_no_parent_chunk_id
#[test]
fn parent_chunk_search_result_has_no_parent_chunk_id() {
    let (conn, _dir) = test_db();

    let parent = NewChunk {
        chunk_type: &ChunkType::Component,
        name: Some("UserForm"),
        content: "function UserForm() { return <form/>; }",
        start_line: 1,
        end_line: 60,
        parent_index: None,
    };

    replace_file_chunks_only(
        &conn,
        "src/UserForm.tsx",
        &[parent],
        "hash_uf3",
        "",
        &[],
        None,
    )
    .unwrap();

    let results =
        search_by_fts(&conn, &["userform"], None, &HashSet::new(), None, 10, &[]).unwrap();

    assert!(!results.is_empty(), "should find UserForm via name search");
    let component_result = &results[0];

    assert_eq!(
        component_result.chunk.parent_chunk_id, None,
        "parent chunk should have parent_chunk_id = None in search results"
    );
}

// ── Phase 1: FTS 3-column unification (FR-001, FR-002, FR-010, FR-011) ──

// T-528: fts_stores_split_identifier_name
#[test]
fn fts_stores_split_identifier_name() {
    let (conn, _dir) = test_db();

    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Hook,
        name: Some("useAuthProvider"),
        content: "function useAuthProvider() { return auth; }",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/auth.tsx", &chunks, "h1", "", &[], None).unwrap();

    let fts_name: String = conn
        .query_row(
            "SELECT name FROM fts_chunks WHERE rowid = (SELECT id FROM chunks LIMIT 1)",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(
        fts_name, "use Auth Provider",
        "FTS name column should contain split-identifier output"
    );
}

// T-531: fts_stores_empty_name_for_none
#[test]
fn fts_stores_empty_name_for_none() {
    let (conn, _dir) = test_db();

    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Other,
        name: None,
        content: "const x = 42;",
        start_line: 1,
        end_line: 1,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/util.ts", &chunks, "h1", "", &[], None).unwrap();

    let fts_name: String = conn
        .query_row(
            "SELECT name FROM fts_chunks WHERE rowid = (SELECT id FROM chunks LIMIT 1)",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(
        fts_name, "",
        "FTS name column should be empty string when chunk name is None"
    );
}

// T-003: fts_stores_file_path
#[test]
fn fts_stores_file_path() {
    let (conn, _dir) = test_db();

    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::RustFn,
        name: Some("login"),
        content: "fn login() { authenticate(); }",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/auth/login.ts", &chunks, "h1", "", &[], None).unwrap();

    let fts_path: String = conn
        .query_row(
            "SELECT file_path FROM fts_chunks WHERE rowid = (SELECT id FROM chunks LIMIT 1)",
            [],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(
        fts_path, "src/auth/login.ts",
        "FTS file_path column should store the raw file path"
    );
}

// T-557: migration_v6_to_v7_preserves_fts_search
#[test]
fn migration_v6_to_v7_preserves_fts_search() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = open_db(&db_path).unwrap();

    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::RustFn,
        name: Some("processData"),
        content: "fn process_data() { transform(); }",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/data.rs", &chunks, "h1", "", &[], None).unwrap();

    // Simulate a v6 DB: roll back schema_version, drop FTS tables
    conn.execute(
        "UPDATE index_meta SET value = '6' WHERE key = 'schema_version'",
        [],
    )
    .unwrap();
    conn.execute_batch("DROP TABLE IF EXISTS fts_chunks")
        .unwrap();
    conn.execute_batch("DROP TABLE IF EXISTS fts_chunks_vocab")
        .unwrap();
    drop(conn);

    // Re-open triggers migration v6 -> v7
    let conn = open_db(&db_path).unwrap();

    let version: String = conn
        .query_row(
            "SELECT value FROM index_meta WHERE key = 'schema_version'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(version, "8", "schema_version should be 8 after migration");

    let results =
        search_by_fts(&conn, &["transform"], None, &HashSet::new(), None, 10, &[]).unwrap();
    assert_eq!(
        results.len(),
        1,
        "FTS search should return results after v6->v7 migration"
    );
    assert_eq!(results[0].chunk.name.as_deref(), Some("processData"));
}

// T-559: migration_populates_fts_name_with_split_identifier
#[test]
fn migration_populates_fts_name_with_split_identifier() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = open_db(&db_path).unwrap();

    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::RustFn,
        name: Some("getUserName"),
        content: "fn get_user_name() { return name; }",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/user.rs", &chunks, "h1", "", &[], None).unwrap();

    // Simulate pre-v7: roll back to v6, drop FTS tables
    conn.execute(
        "UPDATE index_meta SET value = '6' WHERE key = 'schema_version'",
        [],
    )
    .unwrap();
    conn.execute_batch("DROP TABLE IF EXISTS fts_chunks")
        .unwrap();
    conn.execute_batch("DROP TABLE IF EXISTS fts_chunks_vocab")
        .unwrap();
    drop(conn);

    // Re-open triggers migration to v7
    let conn = open_db(&db_path).unwrap();

    let fts_name: String = conn
        .query_row(
            "SELECT name FROM fts_chunks WHERE rowid = (SELECT id FROM chunks LIMIT 1)",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        fts_name, "get User Name",
        "migration should populate FTS name with split_identifier output"
    );
}

// === Phase 2: search_by_fts tests (T-004, T-005, T-007, T-016) ===

// T-004: search_by_fts_finds_camelcase_by_split_keywords
#[test]
fn search_by_fts_finds_camelcase_by_split_keywords() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("handleSubmit"),
            content: "function handleSubmit() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("submitForm"),
            content: "function submitForm() {}",
            start_line: 5,
            end_line: 7,
            parent_index: None,
        },
    ];
    replace_file_chunks_only(&conn, "src/form.tsx", &chunks, "h1", "", &[], None).unwrap();

    // Search with split keywords ["handle", "submit"] should find handleSubmit
    let results = search_by_fts(
        &conn,
        &["handle", "submit"],
        None,
        &HashSet::new(),
        None,
        10,
        &[],
    )
    .unwrap();
    let names: Vec<_> = results
        .iter()
        .filter_map(|r| r.chunk.name.as_deref())
        .collect();
    assert!(
        names.contains(&"handleSubmit"),
        "should find handleSubmit via split keywords, got: {names:?}"
    );
    assert!(
        !names.contains(&"submitForm"),
        "AND semantics: submitForm should NOT match [handle, submit], got: {names:?}"
    );

    // TC-7: verify FTS result fields
    let fts_result = &results[0];
    assert_eq!(
        fts_result.match_source,
        MatchSource::Fts,
        "search_by_fts should set match_source to Fts"
    );
    assert!(
        fts_result.distance.is_infinite(),
        "search_by_fts should set distance to INFINITY"
    );
    assert!(
        fts_result.score > 0.0,
        "search_by_fts bm25 score should be positive, got {}",
        fts_result.score
    );
}

// T-005: search_by_fts_finds_by_path_segment
#[test]
fn search_by_fts_finds_by_path_segment() {
    let (conn, _dir) = test_db();

    let chunks = vec![NewChunk {
        chunk_type: &ChunkType::Other,
        name: Some("login"),
        content: "function login() {}",
        start_line: 1,
        end_line: 3,
        parent_index: None,
    }];
    replace_file_chunks_only(&conn, "src/auth/login.ts", &chunks, "h1", "", &[], None).unwrap();

    let results = search_by_fts(&conn, &["auth"], None, &HashSet::new(), None, 10, &[]).unwrap();
    assert_eq!(results.len(), 1, "should find chunk by path segment 'auth'");
    assert_eq!(results[0].chunk.file_path, "src/auth/login.ts");
}

// T-544: search_by_fts_respects_type_filter
#[test]
fn search_by_fts_respects_type_filter() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useAuth"),
            content: "function useAuth() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("AuthButton"),
            content: "function AuthButton() {}",
            start_line: 5,
            end_line: 7,
            parent_index: None,
        },
    ];
    replace_file_chunks_only(&conn, "src/auth.tsx", &chunks, "h1", "", &[], None).unwrap();

    let results = search_by_fts(
        &conn,
        &["auth"],
        Some(&[ChunkType::Hook]),
        &HashSet::new(),
        None,
        10,
        &[],
    )
    .unwrap();
    assert_eq!(results.len(), 1, "type_filter should limit to hooks");
    assert_eq!(results[0].chunk.name.as_deref(), Some("useAuth"));
}

// T-562: vec_search_respects_type_filter
#[test]
fn vec_search_respects_type_filter() {
    let (conn, _dir) = test_db();
    let mut emb = vec![0.0_f32; EMBEDDING_DIMS];
    emb[0] = 1.0;

    // Hook chunk → passes Some(&[Hook]) filter
    insert_chunk(
        &conn,
        "src/hook.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useAuth"),
            content: "hook body",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(emb.clone()),
        None,
    )
    .unwrap();

    // Component chunk with identical embedding → violates Some(&[Hook]) filter
    insert_chunk(
        &conn,
        "src/comp.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("AuthBtn"),
            content: "component body",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h2",
        &ce(emb.clone()),
        None,
    )
    .unwrap();

    let results = vec_search(&conn, &emb, 10, Some(&[ChunkType::Hook]), &[]).unwrap();
    assert_eq!(
        results.len(),
        1,
        "type_filter should limit to hooks, got {:?}",
        results.iter().map(|r| &r.chunk.name).collect::<Vec<_>>()
    );
    assert_eq!(results[0].chunk.name.as_deref(), Some("useAuth"));
}

// T-563: vec_search_multi_respects_type_filter
#[test]
fn vec_search_multi_respects_type_filter() {
    let (conn, _dir) = test_db();
    let mut emb = vec![0.0_f32; EMBEDDING_DIMS];
    emb[0] = 1.0;

    insert_chunk(
        &conn,
        "src/hook.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useAuth"),
            content: "hook body",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(emb.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/comp.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("AuthBtn"),
            content: "component body",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h2",
        &ce(emb.clone()),
        None,
    )
    .unwrap();

    let embs: Vec<&[f32]> = vec![emb.as_slice()];
    let results = vec_search_multi(&conn, &embs, 10, Some(&[ChunkType::Hook]), &[]).unwrap();
    assert_eq!(results.len(), 1, "type_filter should limit to hooks");
    assert_eq!(results[0].chunk.name.as_deref(), Some("useAuth"));
}

// T-561: search_by_fts_excludes_ids
#[test]
fn search_by_fts_excludes_ids() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useAuth"),
            content: "function useAuth() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        NewChunk {
            chunk_type: &ChunkType::Hook,
            name: Some("useAuthProvider"),
            content: "function useAuthProvider() {}",
            start_line: 5,
            end_line: 7,
            parent_index: None,
        },
    ];
    replace_file_chunks_only(&conn, "src/auth.tsx", &chunks, "h1", "", &[], None).unwrap();

    let all = search_by_fts(&conn, &["auth"], None, &HashSet::new(), None, 10, &[]).unwrap();
    assert_eq!(all.len(), 2);

    let first_id = all[0].chunk_id.unwrap();
    let mut exclude = HashSet::new();
    exclude.insert(first_id);
    let filtered = search_by_fts(&conn, &["auth"], None, &exclude, None, 10, &[]).unwrap();
    assert_eq!(filtered.len(), 1, "excluded id should be filtered out");
    assert_ne!(filtered[0].chunk_id, Some(first_id));
}

// T-173: vec_chunks_embedding_retrievable_by_rowid
#[allow(clippy::cast_possible_truncation)]
#[test]
fn vec_chunks_embedding_retrievable_by_rowid() {
    let (conn, _dir) = test_db();
    let emb: Vec<f32> = (0..EMBEDDING_DIMS as u32)
        .map(|i| i as f32 / 768.0)
        .collect();
    let chunk_id = insert_chunk(
        &conn,
        "src/Foo.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some("foo"),
            content: "fn foo() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        },
        "h1",
        &ce(emb.clone()),
        None,
    )
    .unwrap();

    let vec_rowid: i64 = conn
        .query_row(
            "SELECT vec_rowid FROM embedded_chunk_ids WHERE chunk_id = ?1 AND sub_idx = 0",
            [chunk_id],
            |row| row.get(0),
        )
        .unwrap();

    let bytes: Vec<u8> = conn
        .query_row(
            "SELECT embedding FROM vec_chunks WHERE rowid = ?1",
            [vec_rowid],
            |row| row.get(0),
        )
        .unwrap();

    assert_eq!(bytes.len(), EMBEDDING_DIMS * 4);
    let recovered: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(recovered.len(), EMBEDDING_DIMS);
    assert!((recovered[0] - emb[0]).abs() < 1e-6);
    assert!((recovered[100] - emb[100]).abs() < 1e-6);
}

// T-529: get_chunks_for_from_target file-only returns non-inner_fn chunks
#[test]
fn get_chunks_for_from_target_file_only_excludes_inner_fn() {
    let (conn, _dir) = test_db();
    let emb = vec![0.0_f32; EMBEDDING_DIMS];

    // 2 non-inner_fn chunks
    insert_chunk(
        &conn,
        "src/foo.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some("foo"),
            content: "fn foo() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(emb.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/foo.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustStruct,
            name: Some("Bar"),
            content: "struct Bar {}",
            start_line: 5,
            end_line: 7,
            parent_index: None,
        },
        "h1",
        &ce(emb.clone()),
        None,
    )
    .unwrap();
    // 1 inner_fn chunk (should be excluded)
    insert_chunk_row(
        &conn,
        "src/foo.rs",
        &NewChunk {
            chunk_type: &ChunkType::InnerFn,
            name: Some("inner"),
            content: "fn inner() {}",
            start_line: 2,
            end_line: 2,
            parent_index: None,
        },
        "h1",
        None,
    )
    .unwrap();

    let ids = get_chunks_for_from_target(&conn, "src/foo.rs", None).unwrap();
    assert_eq!(ids.len(), 2);
}

// T-532: get_chunks_for_from_target returns empty for unknown file
#[test]
fn get_chunks_for_from_target_unknown_file_returns_empty() {
    let (conn, _dir) = test_db();
    let rows = get_chunks_for_from_target(&conn, "src/bar.rs", None).unwrap();
    assert!(rows.is_empty());
}

// T-534: get_chunks_for_from_target with symbol returns exact match
#[test]
fn get_chunks_for_from_target_with_symbol_returns_exact_match() {
    let (conn, _dir) = test_db();
    let emb = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/x.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some("foo"),
            content: "fn foo() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(emb.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/x.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some("bar"),
            content: "fn bar() {}",
            start_line: 5,
            end_line: 7,
            parent_index: None,
        },
        "h1",
        &ce(emb.clone()),
        None,
    )
    .unwrap();

    let foo_id = get_chunks_for_from_target(&conn, "src/x.rs", Some("foo")).unwrap();
    assert_eq!(foo_id.len(), 1);
    let bar_id = get_chunks_for_from_target(&conn, "src/x.rs", Some("bar")).unwrap();
    assert_eq!(bar_id.len(), 1);
    assert_ne!(
        foo_id[0], bar_id[0],
        "foo and bar should return different chunk IDs"
    );
}

// T-537: get_chunks_for_from_target symbol not found returns empty
#[test]
fn get_chunks_for_from_target_symbol_not_found_returns_empty() {
    let (conn, _dir) = test_db();
    let emb = vec![0.0_f32; EMBEDDING_DIMS];

    insert_chunk(
        &conn,
        "src/x.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some("foo"),
            content: "fn foo() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(emb.clone()),
        None,
    )
    .unwrap();

    let rows = get_chunks_for_from_target(&conn, "src/x.rs", Some("baz")).unwrap();
    assert!(rows.is_empty());
}

// T-541: get_sub_embeddings_for_chunks returns correct byte length
#[allow(clippy::cast_possible_truncation)]
#[test]
fn get_sub_embeddings_for_chunks_returns_correct_bytes() {
    let (conn, _dir) = test_db();
    let emb: Vec<f32> = (0..EMBEDDING_DIMS as u32)
        .map(|i| i as f32 / 768.0)
        .collect();

    let chunk_id = insert_chunk(
        &conn,
        "src/a.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some("a"),
            content: "fn a() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        },
        "h1",
        &ce(emb.clone()),
        None,
    )
    .unwrap();

    let results = get_sub_embeddings_for_chunks(&conn, &[chunk_id]).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, chunk_id);
    assert_eq!(results[0].1.len(), EMBEDDING_DIMS * 4); // 768 * 4 = 3072 bytes
}

// T-542: get_sub_embeddings_for_chunks with missing chunk returns empty
#[test]
fn get_sub_embeddings_for_chunks_missing_chunk_returns_empty() {
    let (conn, _dir) = test_db();
    let results = get_sub_embeddings_for_chunks(&conn, &[99]).unwrap();
    assert!(results.is_empty());
}

// T-545: vec_search_multi merges duplicate chunk_ids by min distance
#[test]
fn vec_search_multi_keeps_min_distance_for_duplicate_chunk() {
    let (conn, _dir) = test_db();
    let mut emb_target = vec![0.0_f32; EMBEDDING_DIMS];
    emb_target[0] = 1.0;
    let mut emb_other = vec![0.0_f32; EMBEDDING_DIMS];
    emb_other[1] = 1.0;

    insert_chunk(
        &conn,
        "src/target.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some("target"),
            content: "fn target() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        },
        "h1",
        &ce(emb_target.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/other.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some("other"),
            content: "fn other() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        },
        "h2",
        &ce(emb_other.clone()),
        None,
    )
    .unwrap();

    // query_a = exact match to target (distance ≈ 0)
    // query_b = far from target (distance ≈ √2)
    // min-distance merge should keep distance ≈ 0 for target
    let results = vec_search_multi(&conn, &[&emb_target, &emb_other], 10, None, &[]).unwrap();
    let target = results
        .iter()
        .find(|r| r.chunk.name.as_deref() == Some("target"))
        .unwrap();
    assert!(
        target.distance < 0.01,
        "min-distance merge should pick the closer query; got {}",
        target.distance
    );
}

// T-547: search_by_fts include_ids filters results to specified subset
#[test]
fn search_by_fts_with_include_ids_filters_to_subset() {
    let (conn, _dir) = test_db();
    let emb = vec![0.0_f32; EMBEDDING_DIMS];

    let id1 = insert_chunk(
        &conn,
        "src/a.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some("handle_error"),
            content: "fn handle_error() { error handling logic }",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h1",
        &ce(emb.clone()),
        None,
    )
    .unwrap();
    let _id2 = insert_chunk(
        &conn,
        "src/b.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some("handle_error2"),
            content: "fn handle_error2() { more error stuff }",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        },
        "h2",
        &ce(emb.clone()),
        None,
    )
    .unwrap();

    // Without include_ids: both chunks match "error"
    let all = search_by_fts(&conn, &["error"], None, &HashSet::new(), None, 10, &[]).unwrap();
    assert_eq!(all.len(), 2);

    // With include_ids = {id1}: only id1 is returned
    let include = HashSet::from([id1]);
    let filtered = search_by_fts(
        &conn,
        &["error"],
        None,
        &HashSet::new(),
        Some(&include),
        10,
        &[],
    )
    .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].chunk_id, Some(id1));
}

// T-008: vec_search_multi returns union of non-overlapping results
#[test]
fn vec_search_multi_returns_union() {
    let (conn, _dir) = test_db();
    let mut emb_a = vec![0.0_f32; EMBEDDING_DIMS];
    emb_a[0] = 1.0;
    let mut emb_b = vec![0.0_f32; EMBEDDING_DIMS];
    emb_b[1] = 1.0;

    insert_chunk(
        &conn,
        "src/a.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some("fn_a"),
            content: "fn a() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        },
        "h1",
        &ce(emb_a.clone()),
        None,
    )
    .unwrap();
    insert_chunk(
        &conn,
        "src/b.rs",
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some("fn_b"),
            content: "fn b() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        },
        "h2",
        &ce(emb_b.clone()),
        None,
    )
    .unwrap();

    let results = vec_search_multi(&conn, &[&emb_a, &emb_b], 10, None, &[]).unwrap();
    assert_eq!(results.len(), 2);
    let names: Vec<_> = results
        .iter()
        .filter_map(|r| r.chunk.name.as_deref())
        .collect();
    assert!(names.contains(&"fn_a"));
    assert!(names.contains(&"fn_b"));
}

// === SF-3: FTS prefix query fallback ===

// T-174: search_by_fts_fallback_to_quoted_literal_on_sanitize_error
#[test]
fn search_by_fts_fallback_to_quoted_literal_on_sanitize_error() {
    let (conn, _dir) = test_db();

    // Insert a chunk with "NOT" as a standalone word in content.
    // FTS5 unicode61 tokenizer splits on whitespace → "not" token exists in the index.
    replace_file_chunks_only(
        &conn,
        "src/guard.ts",
        &[NewChunk {
            chunk_type: &ChunkType::Other,
            name: Some("shouldNotCall"),
            content: "do NOT call this function directly",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        }],
        "h1",
        "",
        &[],
        None,
    )
    .unwrap();

    // "NOT" is an FTS5 operator → sanitize_fts_query returns NoSearchableTerms.
    // search_by_fts must fall back to fts_quote("NOT") = "\"NOT\"" and still find the chunk.
    let results = search_by_fts(&conn, &["NOT"], None, &HashSet::new(), None, 10, &[])
        .expect("fts fallback should succeed without error");
    assert!(
        !results.is_empty(),
        "fts_quote fallback should find content containing 'NOT', got empty"
    );
}
