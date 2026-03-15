use super::*;
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

use crate::indexer::embedder::MockEmbedder;

fn test_db() -> (storage::Db, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    (conn, dir)
}

#[tokio::test]
async fn run_index_with_mock_embedder() {
    let dir = tempfile::tempdir().unwrap();

    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("Button.tsx"),
        "function Button() { return <div/>; }",
    ).unwrap();
    std::fs::write(
        src_dir.join("App.tsx"),
        "function App() { return <main/>; }",
    ).unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let result = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false)
        .await
        .unwrap();

    assert_eq!(result.files_processed, 2);
    assert!(result.chunks_created >= 2);
    assert_eq!(result.files_errored, 0);

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats.total_files, 2);
    assert!(stats.total_chunks >= 2);
}

#[tokio::test]
async fn run_index_skips_unchanged_files() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(src_dir.join("A.tsx"), "function A() { return 1; }").unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    // First run: processes file
    let r1 = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).await.unwrap();
    assert_eq!(r1.files_processed, 1);

    // Second run: skips unchanged file
    let r2 = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).await.unwrap();
    assert_eq!(r2.files_processed, 0);
    assert_eq!(r2.files_skipped, 1);
}

#[tokio::test]
async fn run_index_force_reindexes() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(src_dir.join("A.tsx"), "function A() { return 1; }").unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).await.unwrap();

    // Force reindex
    let r2 = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, true).await.unwrap();
    assert_eq!(r2.files_processed, 1);
    assert_eq!(r2.files_skipped, 0);
}

#[tokio::test]
async fn run_index_removes_deleted_file_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(src_dir.join("A.tsx"), "function A() { return 1; }").unwrap();
    std::fs::write(src_dir.join("B.tsx"), "function B() { return 2; }").unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    // Index both files
    let r1 = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).await.unwrap();
    assert_eq!(r1.files_processed, 2);
    assert_eq!(storage::get_stats(&conn.lock().unwrap()).unwrap().total_files, 2);

    // Delete B.tsx from disk
    std::fs::remove_file(src_dir.join("B.tsx")).unwrap();

    // Re-index: should remove orphaned B.tsx chunks
    run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).await.unwrap();
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

#[tokio::test]
async fn run_index_stores_imports_in_file_context() {
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

    run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).await.unwrap();

    let contexts = storage::get_file_contexts(&conn.lock().unwrap(), &["src/App.tsx"]).unwrap();
    assert_eq!(contexts.len(), 1);
    let imports = &contexts["src/App.tsx"];
    assert!(imports.contains("import { useState } from 'react'"), "expected useState import, got: {imports}");
    assert!(imports.contains("import { useAuth } from './useAuth'"), "expected useAuth import, got: {imports}");
}

#[tokio::test]
async fn run_chunk_only_index_stores_chunks_without_embeddings() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("App.tsx"),
        "import { Button } from './Button';\nfunction App() { return <Button/>; }",
    ).unwrap();
    std::fs::write(
        src_dir.join("Button.tsx"),
        "export function Button() { return <div/>; }",
    ).unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let result = run_chunk_only_index(Arc::clone(&conn), dir.path()).await.unwrap();

    assert_eq!(result.files_processed, 2);
    assert!(result.chunks_created >= 2);
    assert_eq!(result.files_errored, 0);

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats.total_files, 2);
    assert!(stats.total_chunks >= 2);
    assert_eq!(stats.embedded_chunks, 0);

    let ref_count = storage::get_reference_count(&conn.lock().unwrap()).unwrap();
    assert!(ref_count >= 1, "expected at least 1 reference from App→Button, got {ref_count}");
}

#[tokio::test]
async fn run_incremental_embed_within_budget() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    for name in ["A.tsx", "B.tsx", "C.tsx"] {
        std::fs::write(
            src_dir.join(name),
            format!("function {}() {{ return 1; }}", &name[..1]),
        ).unwrap();
    }

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    run_chunk_only_index(Arc::clone(&conn), dir.path()).await.unwrap();
    let stats_before = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats_before.embedded_chunks, 0);

    let result = run_incremental_embed(
        Arc::clone(&conn), &MockEmbedder, 50, None,
    ).await.unwrap();

    assert!(result.chunks_embedded >= 3);
    assert_eq!(result.files_completed, 3);
    assert!(!result.budget_exhausted);

    let stats_after = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats_after.embedded_chunks, stats_after.total_chunks);
}

#[tokio::test]
async fn run_incremental_embed_exhausts_budget() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    for i in 0..5 {
        std::fs::write(
            src_dir.join(format!("F{i}.tsx")),
            format!("function F{i}() {{ return {i}; }}"),
        ).unwrap();
    }

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    run_chunk_only_index(Arc::clone(&conn), dir.path()).await.unwrap();

    let result = run_incremental_embed(
        Arc::clone(&conn), &MockEmbedder, 2, None,
    ).await.unwrap();

    assert!(result.budget_exhausted);
    assert!(result.chunks_embedded <= 2);

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert!(stats.embedded_chunks < stats.total_chunks);
}

#[tokio::test]
async fn run_incremental_embed_prioritizes_type_hints() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();

    storage::replace_file_chunks_only(
        &conn, "src/types.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::TypeDef,
            name: Some("AuthConfig"),
            content: "interface AuthConfig {}",
            start_line: 1, end_line: 3,
        }],
        "h1", "", &[],
    ).unwrap();

    storage::replace_file_chunks_only(
        &conn, "src/useAuth.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Hook,
            name: Some("useAuth"),
            content: "function useAuth() {}",
            start_line: 1, end_line: 3,
        }],
        "h2", "", &[],
    ).unwrap();

    let conn = Arc::new(Mutex::new(conn));

    let result = run_incremental_embed(
        Arc::clone(&conn), &MockEmbedder, 1,
        Some(&[storage::ChunkType::Hook]),
    ).await.unwrap();

    assert_eq!(result.files_completed, 1);
    assert!(result.chunks_embedded >= 1);

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert!(stats.embedded_chunks >= 1);

    let hook_embedded: bool = conn.lock().unwrap().query_row(
        "SELECT EXISTS(
            SELECT 1 FROM vec_chunks v
            INNER JOIN chunks c ON c.id = v.chunk_id
            WHERE c.file_path = 'src/useAuth.tsx'
        )", [], |row| row.get(0),
    ).unwrap();
    assert!(hook_embedded, "hook file should be embedded first");
}

#[tokio::test]
async fn run_incremental_embed_none_hints_preserves_order() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();

    storage::replace_file_chunks_only(
        &conn, "src/B.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("B"),
            content: "function B() {}",
            start_line: 1, end_line: 3,
        }],
        "h1", "", &[],
    ).unwrap();
    storage::replace_file_chunks_only(
        &conn, "src/A.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("A"),
            content: "function A() {}",
            start_line: 1, end_line: 3,
        }],
        "h2", "", &[],
    ).unwrap();

    let conn = Arc::new(Mutex::new(conn));

    let result = run_incremental_embed(
        Arc::clone(&conn), &MockEmbedder, 50, None,
    ).await.unwrap();

    assert_eq!(result.files_completed, 2);
    assert!(result.chunks_embedded >= 2);
    assert!(!result.budget_exhausted);
}

use crate::indexer::embedder::FailingEmbedder;

#[tokio::test]
async fn run_incremental_embed_aborts_after_consecutive_failures() {
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
            }],
            &format!("h{i}"),
            "",
            &[],
        )
        .unwrap();
    }

    let conn = Arc::new(Mutex::new(conn));

    let result = run_incremental_embed(Arc::clone(&conn), &FailingEmbedder::all_fail(500, "mock failure"), 50, None).await;

    assert!(result.is_err(), "should abort after {MAX_CONSECUTIVE_EMBED_ERRORS} consecutive failures");
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("mock failure"), "got: {err_msg}");
}

#[tokio::test]
async fn embed_and_store_aborts_after_consecutive_failures() {
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

    let result = run_index(Arc::clone(&conn), dir.path(), &FailingEmbedder::all_fail(500, "mock failure"), false).await;

    assert!(result.is_err(), "should abort after {MAX_CONSECUTIVE_EMBED_ERRORS} consecutive failures");
}

#[test]
fn order_files_for_embedding_most_imported_first() {
    let (conn, _dir) = test_db();

    // B is imported by A, so B should come first
    storage::replace_file_chunks_only(
        &conn, "src/A.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("A"),
            content: "function A() {}",
            start_line: 1, end_line: 1,
        }],
        "h1", "", &[storage::Reference {
            source_file: "src/A.tsx".to_string(),
            target_file: "src/B.tsx".to_string(),
            symbol_name: Some("B".to_string()),
            ref_kind: storage::RefKind::Named,
        }],
    ).unwrap();
    storage::replace_file_chunks_only(
        &conn, "src/B.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("B"),
            content: "function B() {}",
            start_line: 1, end_line: 1,
        }],
        "h2", "", &[],
    ).unwrap();

    let ordered = order_files_for_embedding(&conn, None).unwrap();
    assert_eq!(ordered.len(), 2);
    assert_eq!(ordered[0], "src/B.tsx", "most-imported file should come first");
    assert_eq!(ordered[1], "src/A.tsx");
}

#[test]
fn order_files_for_embedding_type_hints_prioritize() {
    let (conn, _dir) = test_db();

    storage::replace_file_chunks_only(
        &conn, "src/types.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::TypeDef,
            name: Some("Config"),
            content: "interface Config {}",
            start_line: 1, end_line: 1,
        }],
        "h1", "", &[],
    ).unwrap();
    storage::replace_file_chunks_only(
        &conn, "src/App.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("App"),
            content: "function App() {}",
            start_line: 1, end_line: 1,
        }],
        "h2", "", &[],
    ).unwrap();

    let ordered = order_files_for_embedding(
        &conn, Some(&[storage::ChunkType::TypeDef]),
    ).unwrap();
    assert_eq!(ordered[0], "src/types.tsx", "type hint file should be prioritized");
}

#[test]
fn order_files_for_embedding_empty_hints_no_reorder() {
    let (conn, _dir) = test_db();

    storage::replace_file_chunks_only(
        &conn, "src/A.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("A"),
            content: "function A() {}",
            start_line: 1, end_line: 1,
        }],
        "h1", "", &[],
    ).unwrap();

    let ordered = order_files_for_embedding(&conn, Some(&[])).unwrap();
    assert_eq!(ordered.len(), 1);
    assert_eq!(ordered[0], "src/A.tsx");
}
