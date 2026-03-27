use super::*;

use rurico::embed::{AlternatingEmbedder, FailingEmbedder, MismatchEmbedder, MockEmbedder};

#[test]
fn file_hash_is_deterministic() {
    let h1 = file_hash("hello world");
    let h2 = file_hash("hello world");
    assert_eq!(h1, h2);
}

#[test]
fn file_hash_changes_with_content() {
    let h1 = file_hash("hello");
    let h2 = file_hash("world");
    assert_ne!(h1, h2);
}

#[test]
fn enrich_for_embedding_with_imports() {
    let result = enrich_for_embedding(
        "src/App.tsx",
        "component",
        "import { useState } from 'react'\nimport { Button } from './Button'",
        "function App() {}",
    );
    assert!(result.starts_with("// File: src/App.tsx\n"));
    assert!(result.contains("// Type: component\n"));
    assert!(result.contains("// import { useState } from 'react'\n"));
    assert!(result.contains("// import { Button } from './Button'\n"));
    assert!(result.ends_with("function App() {}"));
}

#[test]
fn enrich_for_embedding_without_imports() {
    let result = enrich_for_embedding("src/utils.ts", "function", "", "function add() {}");
    assert_eq!(
        result,
        "// File: src/utils.ts\n// Type: function\nfunction add() {}"
    );
}

#[test]
fn enrich_for_embedding_empty_content() {
    let result = enrich_for_embedding("src/empty.ts", "other", "", "");
    assert_eq!(result, "// File: src/empty.ts\n// Type: other\n");
}

#[test]
fn to_rel_path_outside_root() {
    let result = to_rel_path(std::path::Path::new("/a"), std::path::Path::new("/b/c.tsx"));
    assert_eq!(result, "/b/c.tsx");
}

#[test]
fn read_source_too_large() {
    let dir = tempfile::tempdir().unwrap();
    let big_file = dir.path().join("big.ts");
    let content = "x".repeat(MAX_FILE_SIZE as usize + 1);
    std::fs::write(&big_file, content).unwrap();
    let result = read_source(&big_file);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), FileAction::Skip));
}

#[test]
fn read_source_nonexistent() {
    let result = read_source(std::path::Path::new("/nonexistent/file.ts"));
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), FileAction::Error));
}

fn test_db() -> (storage::Db, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    (conn, dir)
}

#[test]
fn run_index_with_mock_embedder() {
    let dir = tempfile::tempdir().unwrap();

    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("Button.tsx"),
        "function Button() { return <div/>; }",
    )
    .unwrap();
    std::fs::write(
        src_dir.join("App.tsx"),
        "function App() { return <main/>; }",
    )
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let result = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).unwrap();

    assert_eq!(result.files_processed, 2);
    assert!(result.chunks_created >= 2);
    assert_eq!(result.files_errored, 0);

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats.total_files, 2);
    assert!(stats.total_chunks >= 2);
}

#[test]
fn run_index_skips_unchanged_files() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(src_dir.join("A.tsx"), "function A() { return 1; }").unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    // First run: processes file
    let r1 = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).unwrap();
    assert_eq!(r1.files_processed, 1);

    // Second run: skips unchanged file
    let r2 = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).unwrap();
    assert_eq!(r2.files_processed, 0);
    assert_eq!(r2.files_skipped, 1);
}

#[test]
fn run_index_force_reindexes() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(src_dir.join("A.tsx"), "function A() { return 1; }").unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).unwrap();

    let r2 = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, true).unwrap();
    assert_eq!(r2.files_processed, 1);
    assert_eq!(r2.files_skipped, 0);
}

#[test]
fn run_index_removes_deleted_file_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(src_dir.join("A.tsx"), "function A() { return 1; }").unwrap();
    std::fs::write(src_dir.join("B.tsx"), "function B() { return 2; }").unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    // Index both files
    let r1 = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).unwrap();
    assert_eq!(r1.files_processed, 2);
    assert_eq!(
        storage::get_stats(&conn.lock().unwrap())
            .unwrap()
            .total_files,
        2
    );

    // Delete B.tsx from disk
    std::fs::remove_file(src_dir.join("B.tsx")).unwrap();

    // Re-index: should remove orphaned B.tsx chunks
    run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).unwrap();
    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats.total_files, 1, "orphaned file should be removed");
}

#[test]
fn build_references_named_import() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("B.tsx"), "").unwrap();

    let resolver = crate::resolver::Resolver::new(tmp.path());
    let imports = vec![chunker::ParsedImport {
        specifiers: vec![chunker::ImportSpecifier {
            name: "B".to_string(),
            alias: None,
            kind: chunker::ImportKind::Named,
        }],
        source: "./B".to_string(),
    }];
    let refs = build_references(&imports, "src/A.tsx", &resolver);
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].source_file, "src/A.tsx");
    assert_eq!(refs[0].target_file, "src/B.tsx");
    assert_eq!(refs[0].symbol_name, Some("B".to_string()));
    assert_eq!(refs[0].ref_kind, RefKind::Named);
}

#[test]
fn build_references_side_effect_import() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("styles.css"), "").unwrap();

    let resolver = crate::resolver::Resolver::new(tmp.path());
    let imports = vec![chunker::ParsedImport {
        specifiers: vec![],
        source: "./styles.css".to_string(),
    }];
    let refs = build_references(&imports, "src/A.tsx", &resolver);
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].symbol_name, None);
    assert_eq!(refs[0].ref_kind, RefKind::SideEffect);
}

#[test]
fn build_references_unresolvable_returns_empty() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();

    let resolver = crate::resolver::Resolver::new(tmp.path());
    let imports = vec![chunker::ParsedImport {
        specifiers: vec![chunker::ImportSpecifier {
            name: "useState".to_string(),
            alias: None,
            kind: chunker::ImportKind::Named,
        }],
        source: "react".to_string(),
    }];
    let refs = build_references(&imports, "src/A.tsx", &resolver);
    assert!(refs.is_empty());
}

#[test]
fn build_references_rust_crate_import() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    let foo = src.join("foo");
    std::fs::create_dir_all(&foo).unwrap();
    std::fs::write(foo.join("bar.rs"), "").unwrap();

    let rust_resolver = crate::rust_resolver::RustResolver::new(tmp.path());
    let imports = vec![chunker::ParsedImport {
        specifiers: vec![chunker::ImportSpecifier {
            name: "Bar".to_string(),
            alias: None,
            kind: chunker::ImportKind::Named,
        }],
        source: "crate::foo::bar".to_string(),
    }];
    let refs = build_references(&imports, "src/lib.rs", &rust_resolver);
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].source_file, "src/lib.rs");
    assert_eq!(refs[0].target_file, "src/foo/bar.rs");
    assert_eq!(refs[0].symbol_name, Some("Bar".to_string()));
    assert_eq!(refs[0].ref_kind, RefKind::Named);
}

#[test]
fn run_index_stores_imports_in_file_context() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("App.tsx"),
        "import { useState } from 'react';\nimport { useAuth } from './useAuth';\nfunction App() { return <div/>; }",
    ).unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).unwrap();

    let contexts = storage::get_file_contexts(&conn.lock().unwrap(), &["src/App.tsx"]).unwrap();
    assert_eq!(contexts.len(), 1);
    let imports = &contexts["src/App.tsx"];
    assert!(
        imports.contains("import { useState } from 'react'"),
        "expected useState import, got: {imports}"
    );
    assert!(
        imports.contains("import { useAuth } from './useAuth'"),
        "expected useAuth import, got: {imports}"
    );
}

#[test]
fn run_chunk_only_index_stores_chunks_without_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("App.tsx"),
        "import { Button } from './Button';\nfunction App() { return <Button/>; }",
    )
    .unwrap();
    std::fs::write(
        src_dir.join("Button.tsx"),
        "export function Button() { return <div/>; }",
    )
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let result = run_chunk_only_index(Arc::clone(&conn), dir.path()).unwrap();

    assert_eq!(result.files_processed, 2);
    assert!(result.chunks_created >= 2);
    assert_eq!(result.files_errored, 0);

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats.total_files, 2);
    assert!(stats.total_chunks >= 2);
    assert_eq!(stats.embedded_chunks, 0);

    let ref_count = storage::get_reference_count(&conn.lock().unwrap()).unwrap();
    assert!(
        ref_count >= 1,
        "expected at least 1 reference from App→Button, got {ref_count}"
    );
}

#[test]
fn run_incremental_embed_empty_db_returns_zero() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let result = run_incremental_embed(Arc::clone(&conn), &MockEmbedder, 50, None).unwrap();

    assert_eq!(result.chunks_embedded, 0);
    assert_eq!(result.files_completed, 0);
    assert!(!result.budget_exhausted);
}

#[test]
fn run_incremental_embed_within_budget() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    for name in ["A.tsx", "B.tsx", "C.tsx"] {
        std::fs::write(
            src_dir.join(name),
            format!("function {}() {{ return 1; }}", &name[..1]),
        )
        .unwrap();
    }

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    run_chunk_only_index(Arc::clone(&conn), dir.path()).unwrap();
    let stats_before = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats_before.embedded_chunks, 0);

    let result = run_incremental_embed(Arc::clone(&conn), &MockEmbedder, 50, None).unwrap();

    assert!(result.chunks_embedded >= 3);
    assert_eq!(result.files_completed, 3);
    assert!(!result.budget_exhausted);

    let stats_after = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats_after.embedded_chunks, stats_after.total_chunks);
}

#[test]
fn run_incremental_embed_exhausts_budget() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    for i in 0..5 {
        std::fs::write(
            src_dir.join(format!("F{i}.tsx")),
            format!("function F{i}() {{ return {i}; }}"),
        )
        .unwrap();
    }

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    run_chunk_only_index(Arc::clone(&conn), dir.path()).unwrap();

    let result = run_incremental_embed(Arc::clone(&conn), &MockEmbedder, 2, None).unwrap();

    assert!(result.budget_exhausted);
    assert!(result.chunks_embedded <= 2);

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert!(stats.embedded_chunks < stats.total_chunks);
}

#[test]
fn run_incremental_embed_prioritizes_type_hints() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();

    storage::replace_file_chunks_only(
        &conn,
        "src/types.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::TypeDef,
            name: Some("AuthConfig"),
            content: "interface AuthConfig {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        }],
        "h1",
        "",
        &[],
    )
    .unwrap();

    storage::replace_file_chunks_only(
        &conn,
        "src/useAuth.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Hook,
            name: Some("useAuth"),
            content: "function useAuth() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        }],
        "h2",
        "",
        &[],
    )
    .unwrap();

    let conn = Arc::new(Mutex::new(conn));

    let result = run_incremental_embed(
        Arc::clone(&conn),
        &MockEmbedder,
        1,
        Some(&[storage::ChunkType::Hook]),
    )
    .unwrap();

    assert_eq!(result.files_completed, 1);
    assert!(result.chunks_embedded >= 1);

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert!(stats.embedded_chunks >= 1);

    let hook_embedded: bool = conn
        .lock()
        .unwrap()
        .query_row(
            "SELECT EXISTS(
            SELECT 1 FROM vec_chunks v
            INNER JOIN chunks c ON c.id = v.chunk_id
            WHERE c.file_path = 'src/useAuth.tsx'
        )",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(hook_embedded, "hook file should be embedded first");
}

#[test]
fn run_incremental_embed_none_hints_preserves_order() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();

    storage::replace_file_chunks_only(
        &conn,
        "src/B.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("B"),
            content: "function B() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        }],
        "h1",
        "",
        &[],
    )
    .unwrap();
    storage::replace_file_chunks_only(
        &conn,
        "src/A.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("A"),
            content: "function A() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        }],
        "h2",
        "",
        &[],
    )
    .unwrap();

    let conn = Arc::new(Mutex::new(conn));

    let result = run_incremental_embed(Arc::clone(&conn), &MockEmbedder, 50, None).unwrap();

    assert_eq!(result.files_completed, 2);
    assert!(result.chunks_embedded >= 2);
    assert!(!result.budget_exhausted);
}

#[test]
fn run_incremental_embed_recovers_after_intermittent_failure() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();

    // Insert 6 files with 1 chunk each
    for i in 0..6 {
        storage::replace_file_chunks_only(
            &conn,
            &format!("src/R{i}.tsx"),
            &[storage::NewChunk {
                chunk_type: &storage::ChunkType::Component,
                name: Some(&format!("R{i}")),
                content: &format!("function R{i}() {{}}"),
                start_line: 1,
                end_line: 1,
                parent_index: None,
            }],
            &format!("r{i}"),
            "",
            &[],
        )
        .unwrap();
    }

    let conn = Arc::new(Mutex::new(conn));

    // AlternatingEmbedder: call 0 fails, 1 succeeds, 2 fails, 3 succeeds, ...
    // consecutive_errors resets to 0 on each success, so it never reaches MAX (5).
    let result = run_incremental_embed(Arc::clone(&conn), &AlternatingEmbedder::new(), 50, None);

    assert!(
        result.is_ok(),
        "should not abort when failures are intermittent (counter resets on success)"
    );
    let outcome = result.unwrap();
    assert!(
        outcome.files_completed > 0,
        "some files should embed successfully"
    );
}

#[test]
fn run_incremental_embed_aborts_after_consecutive_failures() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();

    for i in 0..6 {
        storage::replace_file_chunks_only(
            &conn,
            &format!("src/F{i}.tsx"),
            &[storage::NewChunk {
                chunk_type: &storage::ChunkType::Component,
                name: Some(&format!("F{i}")),
                content: &format!("function F{i}() {{}}"),
                start_line: 1,
                end_line: 1,
                parent_index: None,
            }],
            &format!("h{i}"),
            "",
            &[],
        )
        .unwrap();
    }

    let conn = Arc::new(Mutex::new(conn));

    let result = run_incremental_embed(
        Arc::clone(&conn),
        &FailingEmbedder::all_fail("mock failure"),
        50,
        None,
    );

    assert!(
        result.is_err(),
        "should abort after {MAX_CONSECUTIVE_EMBED_ERRORS} consecutive failures"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("mock failure"), "got: {err_msg}");
}

#[test]
fn embed_and_store_aborts_after_consecutive_failures() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    for i in 0..6 {
        std::fs::write(
            src_dir.join(format!("F{i}.tsx")),
            format!("function F{i}() {{ return {i}; }}"),
        )
        .unwrap();
    }

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let result = run_index(
        Arc::clone(&conn),
        dir.path(),
        &FailingEmbedder::all_fail("mock failure"),
        false,
    );

    assert!(
        result.is_err(),
        "should abort after {MAX_CONSECUTIVE_EMBED_ERRORS} consecutive failures"
    );
}

#[test]
fn order_files_for_embedding_most_imported_first() {
    let (conn, _dir) = test_db();

    // B is imported by A, so B should come first
    storage::replace_file_chunks_only(
        &conn,
        "src/A.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("A"),
            content: "function A() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        }],
        "h1",
        "",
        &[storage::Reference {
            source_file: "src/A.tsx".to_string(),
            target_file: "src/B.tsx".to_string(),
            symbol_name: Some("B".to_string()),
            ref_kind: storage::RefKind::Named,
        }],
    )
    .unwrap();
    storage::replace_file_chunks_only(
        &conn,
        "src/B.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("B"),
            content: "function B() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        }],
        "h2",
        "",
        &[],
    )
    .unwrap();

    let ordered = order_files_for_embedding(&conn, None).unwrap();
    assert_eq!(ordered.len(), 2);
    assert_eq!(
        ordered[0], "src/B.tsx",
        "most-imported file should come first"
    );
    assert_eq!(ordered[1], "src/A.tsx");
}

#[test]
fn order_files_for_embedding_type_hints_prioritize() {
    let (conn, _dir) = test_db();

    storage::replace_file_chunks_only(
        &conn,
        "src/types.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::TypeDef,
            name: Some("Config"),
            content: "interface Config {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        }],
        "h1",
        "",
        &[],
    )
    .unwrap();
    storage::replace_file_chunks_only(
        &conn,
        "src/App.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("App"),
            content: "function App() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        }],
        "h2",
        "",
        &[],
    )
    .unwrap();

    let ordered = order_files_for_embedding(&conn, Some(&[storage::ChunkType::TypeDef])).unwrap();
    assert_eq!(
        ordered[0], "src/types.tsx",
        "type hint file should be prioritized"
    );
}

#[test]
fn order_files_for_embedding_empty_hints_no_reorder() {
    let (conn, _dir) = test_db();

    storage::replace_file_chunks_only(
        &conn,
        "src/A.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("A"),
            content: "function A() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        }],
        "h1",
        "",
        &[],
    )
    .unwrap();

    let ordered = order_files_for_embedding(&conn, Some(&[])).unwrap();
    assert_eq!(ordered.len(), 1);
    assert_eq!(ordered[0], "src/A.tsx");
}

#[test]
fn embed_and_store_catches_count_mismatch_via_storage() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    // File with multiple top-level functions → multiple chunks
    std::fs::write(
        src_dir.join("Multi.tsx"),
        "function A() { return 1; }\nfunction B() { return 2; }\nfunction C() { return 3; }",
    )
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    // MismatchEmbedder returns 1 vector regardless of input count.
    // Storage layer catches embeddings.len() != chunks.len() via LengthMismatch.
    let result = run_index(Arc::clone(&conn), dir.path(), &MismatchEmbedder, false);

    if let Ok(outcome) = &result {
        let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
        assert_eq!(
            stats.embedded_chunks, 0,
            "mismatched embeddings should not be stored"
        );
        assert!(
            outcome.files_errored > 0 || outcome.chunks_created == 0,
            "mismatch should be detected: errored={}, embedded={}",
            outcome.files_errored,
            outcome.chunks_created
        );
    }
    // If Err, the mismatch was caught — also acceptable
}

#[test]
fn run_incremental_embed_skips_count_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();

    // Insert 2 files with 1 chunk each — embedder will return 1 vector for any input
    for i in 0..2 {
        storage::replace_file_chunks_only(
            &conn,
            &format!("src/F{i}.tsx"),
            &[storage::NewChunk {
                chunk_type: &storage::ChunkType::Component,
                name: Some(&format!("F{i}")),
                content: &format!("function F{i}() {{}}"),
                start_line: 1,
                end_line: 1,
                parent_index: None,
            }],
            &format!("h{i}"),
            "",
            &[],
        )
        .unwrap();
    }

    let conn = Arc::new(Mutex::new(conn));

    // MismatchEmbedder always returns 1 vector, but each file has 1 chunk,
    // so for single-chunk files there's no mismatch. Add a multi-chunk file.
    {
        let c = conn.lock().unwrap();
        storage::replace_file_chunks_only(
            &c,
            "src/Multi.tsx",
            &[
                storage::NewChunk {
                    chunk_type: &storage::ChunkType::Component,
                    name: Some("A"),
                    content: "function A() {}",
                    start_line: 1,
                    end_line: 1,
                    parent_index: None,
                },
                storage::NewChunk {
                    chunk_type: &storage::ChunkType::Component,
                    name: Some("B"),
                    content: "function B() {}",
                    start_line: 2,
                    end_line: 2,
                    parent_index: None,
                },
            ],
            "h_multi",
            "",
            &[],
        )
        .unwrap();
    }

    let result = run_incremental_embed(Arc::clone(&conn), &MismatchEmbedder, 50, None).unwrap();

    // Multi.tsx should be skipped (2 chunks, 1 embedding), single-chunk files should succeed
    assert!(
        result.files_completed >= 2,
        "single-chunk files should embed: {}",
        result.files_completed
    );

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert!(
        stats.embedded_chunks < stats.total_chunks,
        "multi-chunk file should be skipped due to count mismatch"
    );
}

#[test]
fn run_chunk_only_index_handles_rust_files() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("lib.rs"),
        "pub struct Config { name: String }\npub fn init() {}",
    )
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let result = run_chunk_only_index(Arc::clone(&conn), dir.path()).unwrap();

    assert_eq!(result.files_processed, 1);
    assert_eq!(result.chunks_created, 2);
    assert_eq!(result.files_errored, 0);

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats.total_files, 1);
    assert_eq!(stats.total_chunks, 2);

    let ref_count = storage::get_reference_count(&conn.lock().unwrap()).unwrap();
    assert_eq!(ref_count, 0, "Rust files should have no import references");
}

#[cfg(unix)]
#[test]
fn run_index_counts_unreadable_files_as_errors() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    // Normal file
    std::fs::write(src_dir.join("Good.tsx"), "function Good() { return 1; }").unwrap();

    // Unreadable file (permission 000)
    let bad_path = src_dir.join("Bad.tsx");
    std::fs::write(&bad_path, "function Bad() { return 2; }").unwrap();
    std::fs::set_permissions(&bad_path, std::fs::Permissions::from_mode(0o000)).unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let result = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).unwrap();

    // Restore permissions for cleanup
    std::fs::set_permissions(&bad_path, std::fs::Permissions::from_mode(0o644)).unwrap();

    assert!(
        result.files_errored > 0,
        "unreadable file should be counted as error, got files_errored={}",
        result.files_errored
    );
    assert!(
        result.files_processed > 0,
        "readable file should still be processed"
    );
}
