use std::fs;
use std::path::Path;

use super::*;
use crate::resolver::Resolver;
use crate::rust_resolver::RustResolver;

use rurico::embed::{
    AlternatingEmbedder, FailingEmbedder, MismatchEmbedder, MockChunkedEmbedder, MockEmbedder,
};
use tempfile::{TempDir, tempdir};

// T-362: file_hash_is_deterministic
#[test]
fn file_hash_is_deterministic() {
    let h1 = file_hash("hello world");
    let h2 = file_hash("hello world");
    assert_eq!(h1, h2);
}

// T-363: file_hash_changes_with_content
#[test]
fn file_hash_changes_with_content() {
    let h1 = file_hash("hello");
    let h2 = file_hash("world");
    assert_ne!(h1, h2);
}

// T-364: enrich_for_embedding_with_imports
#[test]
fn enrich_for_embedding_with_imports() {
    let result = enrich_for_embedding(
        "src/App.tsx",
        "component",
        None,
        None,
        "import { useState } from 'react'\nimport { Button } from './Button'",
        "function App() {}",
    );
    assert!(result.starts_with("// File: src/App.tsx\n"));
    assert!(result.contains("// Type: component\n"));
    assert!(result.contains("// import { useState } from 'react'\n"));
    assert!(result.contains("// import { Button } from './Button'\n"));
    assert!(result.ends_with("function App() {}"));
}

// T-365: enrich_for_embedding_without_imports
#[test]
fn enrich_for_embedding_without_imports() {
    let result = enrich_for_embedding(
        "src/utils.ts",
        "function",
        None,
        None,
        "",
        "function add() {}",
    );
    assert_eq!(
        result,
        "// File: src/utils.ts\n// Type: function\nfunction add() {}"
    );
}

// T-366: enrich_for_embedding_empty_content
#[test]
fn enrich_for_embedding_empty_content() {
    let result = enrich_for_embedding("src/empty.ts", "other", None, None, "", "");
    assert_eq!(result, "// File: src/empty.ts\n// Type: other\n");
}

// T-367: enrich_for_embedding_with_name
#[test]
fn enrich_for_embedding_with_name() {
    let result = enrich_for_embedding(
        "src/slack.rs",
        "rust_fn",
        Some("handle_rate_limit"),
        None,
        "",
        "fn handle_rate_limit() {}",
    );
    assert!(result.contains("// Name: handle_rate_limit\n"));
    assert!(!result.contains("// Parent:"));
}

// T-368: enrich_for_embedding_with_name_and_parent
#[test]
fn enrich_for_embedding_with_name_and_parent() {
    let result = enrich_for_embedding(
        "src/client.rs",
        "rust_fn",
        Some("handle_rate_limit"),
        Some("SlackClient"),
        "",
        "fn handle_rate_limit() {}",
    );
    assert!(result.contains("// Name: handle_rate_limit\n"));
    assert!(result.contains("// Parent: SlackClient\n"));
    let name_pos = result.find("// Name:").unwrap();
    let parent_pos = result.find("// Parent:").unwrap();
    assert!(name_pos < parent_pos);
}

// T-369: to_rel_path_outside_root
#[test]
fn to_rel_path_outside_root() {
    let result = to_rel_path(Path::new("/a"), Path::new("/b/c.tsx"));
    assert_eq!(result, "/b/c.tsx");
}

// T-370: read_source_too_large
#[allow(clippy::cast_possible_truncation)]
#[test]
fn read_source_too_large() {
    let dir = tempdir().unwrap();
    let big_file = dir.path().join("big.ts");
    let content = "x".repeat(MAX_FILE_SIZE as usize + 1);
    fs::write(&big_file, content).unwrap();
    let result = read_source(&big_file);
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), FileAction::Skip));
}

// T-371: read_source_nonexistent
#[test]
fn read_source_nonexistent() {
    let result = read_source(Path::new("/nonexistent/file.ts"));
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), FileAction::Error));
}

fn test_db() -> (storage::Db, TempDir) {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    (conn, dir)
}

// T-377: build_references_named_import
#[test]
fn build_references_named_import() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("B.tsx"), "").unwrap();

    let resolver = Resolver::new(tmp.path());
    let imports = vec![chunker::ParsedImport {
        specifiers: vec![chunker::ImportSpecifier {
            name: "B".to_owned(),
            alias: None,
            kind: chunker::ImportKind::Named,
        }],
        source: "./B".to_owned(),
    }];
    let refs = build_references(&imports, "src/A.tsx", &resolver);
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].source_file, "src/A.tsx");
    assert_eq!(refs[0].target_file, "src/B.tsx");
    assert_eq!(refs[0].symbol_name, Some("B".to_owned()));
    assert_eq!(refs[0].ref_kind, RefKind::Named);
}

// T-378: build_references_side_effect_import
#[test]
fn build_references_side_effect_import() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("styles.css"), "").unwrap();

    let resolver = Resolver::new(tmp.path());
    let imports = vec![chunker::ParsedImport {
        specifiers: vec![],
        source: "./styles.css".to_owned(),
    }];
    let refs = build_references(&imports, "src/A.tsx", &resolver);
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].symbol_name, None);
    assert_eq!(refs[0].ref_kind, RefKind::SideEffect);
}

// T-379: build_references_unresolvable_returns_empty
#[test]
fn build_references_unresolvable_returns_empty() {
    let tmp = tempdir().unwrap();
    fs::create_dir_all(tmp.path().join("src")).unwrap();

    let resolver = Resolver::new(tmp.path());
    let imports = vec![chunker::ParsedImport {
        specifiers: vec![chunker::ImportSpecifier {
            name: "useState".to_owned(),
            alias: None,
            kind: chunker::ImportKind::Named,
        }],
        source: "react".to_owned(),
    }];
    let refs = build_references(&imports, "src/A.tsx", &resolver);
    assert!(refs.is_empty());
}

// T-380: build_references_rust_crate_import
#[test]
fn build_references_rust_crate_import() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    let foo = src.join("foo");
    fs::create_dir_all(&foo).unwrap();
    fs::write(foo.join("bar.rs"), "").unwrap();

    let rust_resolver = RustResolver::new(tmp.path());
    let imports = vec![chunker::ParsedImport {
        specifiers: vec![chunker::ImportSpecifier {
            name: "Bar".to_owned(),
            alias: None,
            kind: chunker::ImportKind::Named,
        }],
        source: "crate::foo::bar".to_owned(),
    }];
    let refs = build_references(&imports, "src/lib.rs", &rust_resolver);
    assert_eq!(refs.len(), 1);
    assert_eq!(refs[0].source_file, "src/lib.rs");
    assert_eq!(refs[0].target_file, "src/foo/bar.rs");
    assert_eq!(refs[0].symbol_name, Some("Bar".to_owned()));
    assert_eq!(refs[0].ref_kind, RefKind::Named);
}

// T-382: run_chunk_only_index_stores_chunks_without_embeddings
#[test]
fn run_chunk_only_index_stores_chunks_without_embeddings() {
    let dir = tempdir().unwrap();
    let src_dir = dir.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    fs::write(
        src_dir.join("App.tsx"),
        "import { Button } from './Button';\nfunction App() { return <Button/>; }",
    )
    .unwrap();
    fs::write(
        src_dir.join("Button.tsx"),
        "export function Button() { return <div/>; }",
    )
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let result = run_chunk_only_index(&conn, dir.path(), false).unwrap();

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

// T-383: run_incremental_embed_empty_db_returns_zero
#[test]
fn run_incremental_embed_empty_db_returns_zero() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let result = run_incremental_embed(&conn, &MockEmbedder::default(), 50, None).unwrap();

    assert_eq!(result.chunks_embedded, 0);
    assert_eq!(result.files_completed, 0);
}

// T-384: run_incremental_embed_within_budget
#[test]
fn run_incremental_embed_within_budget() {
    let dir = tempdir().unwrap();
    let src_dir = dir.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    for name in ["A.tsx", "B.tsx", "C.tsx"] {
        fs::write(
            src_dir.join(name),
            format!("function {}() {{ return 1; }}", &name[..1]),
        )
        .unwrap();
    }

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    run_chunk_only_index(&conn, dir.path(), false).unwrap();
    let stats_before = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats_before.embedded_chunks, 0);

    let result = run_incremental_embed(&conn, &MockEmbedder::default(), 50, None).unwrap();

    assert!(result.chunks_embedded >= 3);
    assert_eq!(result.files_completed, 3);

    let stats_after = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats_after.embedded_chunks, stats_after.total_chunks);
}

// T-385: run_incremental_embed_exhausts_budget
#[test]
fn run_incremental_embed_exhausts_budget() {
    let dir = tempdir().unwrap();
    let src_dir = dir.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    for i in 0..5 {
        fs::write(
            src_dir.join(format!("F{i}.tsx")),
            format!("function F{i}() {{ return {i}; }}"),
        )
        .unwrap();
    }

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    run_chunk_only_index(&conn, dir.path(), false).unwrap();

    let result = run_incremental_embed(&conn, &MockEmbedder::default(), 2, None).unwrap();

    assert!(result.chunks_embedded <= 2);

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert!(stats.embedded_chunks < stats.total_chunks);
}

// T-386: run_incremental_embed_prioritizes_type_hints
#[test]
fn run_incremental_embed_prioritizes_type_hints() {
    let dir = tempdir().unwrap();
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
            source_kind: None,
            injection_flags: None,
        }],
        "h1",
        "",
        &[],
        None,
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
            source_kind: None,
            injection_flags: None,
        }],
        "h2",
        "",
        &[],
        None,
    )
    .unwrap();

    let conn = Arc::new(Mutex::new(conn));

    let result = run_incremental_embed(
        &conn,
        &MockEmbedder::default(),
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

// T-387: run_incremental_embed_none_hints_preserves_order
#[test]
fn run_incremental_embed_none_hints_preserves_order() {
    let dir = tempdir().unwrap();
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
            source_kind: None,
            injection_flags: None,
        }],
        "h1",
        "",
        &[],
        None,
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
            source_kind: None,
            injection_flags: None,
        }],
        "h2",
        "",
        &[],
        None,
    )
    .unwrap();

    let conn = Arc::new(Mutex::new(conn));

    let result = run_incremental_embed(&conn, &MockEmbedder::default(), 50, None).unwrap();

    assert_eq!(result.files_completed, 2);
    assert!(result.chunks_embedded >= 2);
}

// T-388: run_incremental_embed_recovers_after_intermittent_failure
#[test]
fn run_incremental_embed_recovers_after_intermittent_failure() {
    let dir = tempdir().unwrap();
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
                source_kind: None,
                injection_flags: None,
            }],
            &format!("r{i}"),
            "",
            &[],
            None,
        )
        .unwrap();
    }

    let conn = Arc::new(Mutex::new(conn));

    // AlternatingEmbedder: call 0 fails, 1 succeeds, 2 fails, 3 succeeds, ...
    // consecutive_errors resets to 0 on each success, so it never reaches MAX (5).
    let result = run_incremental_embed(&conn, &AlternatingEmbedder::new(), 50, None);

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

// T-389: run_incremental_embed_aborts_after_consecutive_failures
#[test]
fn run_incremental_embed_aborts_after_consecutive_failures() {
    let dir = tempdir().unwrap();
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
                source_kind: None,
                injection_flags: None,
            }],
            &format!("h{i}"),
            "",
            &[],
            None,
        )
        .unwrap();
    }

    let conn = Arc::new(Mutex::new(conn));

    let result = run_incremental_embed(&conn, &FailingEmbedder::all_fail("mock failure"), 50, None);

    assert!(
        result.is_err(),
        "should abort after {MAX_CONSECUTIVE_EMBED_ERRORS} consecutive failures"
    );
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("mock failure"), "got: {err_msg}");
}

// T-391: order_files_for_embedding_most_imported_first
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
            source_kind: None,
            injection_flags: None,
        }],
        "h1",
        "",
        &[storage::Reference {
            source_file: "src/A.tsx".to_owned(),
            target_file: "src/B.tsx".to_owned(),
            symbol_name: Some("B".to_owned()),
            ref_kind: storage::RefKind::Named,
        }],
        None,
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
            source_kind: None,
            injection_flags: None,
        }],
        "h2",
        "",
        &[],
        None,
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

// T-392: order_files_for_embedding_type_hints_prioritize
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
            source_kind: None,
            injection_flags: None,
        }],
        "h1",
        "",
        &[],
        None,
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
            source_kind: None,
            injection_flags: None,
        }],
        "h2",
        "",
        &[],
        None,
    )
    .unwrap();

    let ordered = order_files_for_embedding(&conn, Some(&[storage::ChunkType::TypeDef])).unwrap();
    assert_eq!(
        ordered[0], "src/types.tsx",
        "type hint file should be prioritized"
    );
}

// T-393: run_incremental_embed_with_multi_chunk_embedder_keeps_first_only
#[test]
fn run_incremental_embed_with_multi_chunk_embedder_keeps_first_only() {
    let (conn, _dir) = test_db();

    storage::replace_file_chunks_only(
        &conn,
        "src/App.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("App"),
            content: "function App() { return <main/>; }",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        }],
        "h1",
        "",
        &[],
        None,
    )
    .unwrap();

    let conn = Arc::new(Mutex::new(conn));

    // MockChunkedEmbedder returns 3 chunks per document; first_chunk keeps only the first.
    let result = run_incremental_embed(&conn, &MockChunkedEmbedder::new(3), 50, None).unwrap();

    assert_eq!(result.files_completed, 1);
    assert_eq!(result.chunks_embedded, 1);

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    // Embedding count must equal embeddable chunk count (1:1), NOT chunk_count * 3.
    assert_eq!(stats.embedded_chunks, stats.embeddable_chunks);
}

// T-394: order_files_for_embedding_empty_hints_no_reorder
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
            source_kind: None,
            injection_flags: None,
        }],
        "h1",
        "",
        &[],
        None,
    )
    .unwrap();

    let ordered = order_files_for_embedding(&conn, Some(&[])).unwrap();
    assert_eq!(ordered.len(), 1);
    assert_eq!(ordered[0], "src/A.tsx");
}

// T-396: run_incremental_embed_skips_count_mismatch
#[test]
fn run_incremental_embed_skips_count_mismatch() {
    let dir = tempdir().unwrap();
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
                source_kind: None,
                injection_flags: None,
            }],
            &format!("h{i}"),
            "",
            &[],
            None,
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
                    source_kind: None,
                    injection_flags: None,
                },
                storage::NewChunk {
                    chunk_type: &storage::ChunkType::Component,
                    name: Some("B"),
                    content: "function B() {}",
                    start_line: 2,
                    end_line: 2,
                    parent_index: None,
                    source_kind: None,
                    injection_flags: None,
                },
            ],
            "h_multi",
            "",
            &[],
            None,
        )
        .unwrap();
    }

    let result = run_incremental_embed(&conn, &MismatchEmbedder, 50, None).unwrap();

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

// T-397: run_chunk_only_index_handles_rust_files
#[test]
fn run_chunk_only_index_handles_rust_files() {
    let dir = tempdir().unwrap();
    let src_dir = dir.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    fs::write(
        src_dir.join("lib.rs"),
        "pub struct Config { name: String }\npub fn init() {}",
    )
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let result = run_chunk_only_index(&conn, dir.path(), false).unwrap();

    assert_eq!(result.files_processed, 1);
    assert_eq!(result.chunks_created, 2);
    assert_eq!(result.files_errored, 0);

    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats.total_files, 1);
    assert_eq!(stats.total_chunks, 2);

    let ref_count = storage::get_reference_count(&conn.lock().unwrap()).unwrap();
    assert_eq!(ref_count, 0, "Rust files should have no import references");
}

// T-374: run_chunk_only_index_skips_unchanged_files
#[test]
fn run_chunk_only_index_skips_unchanged_files() {
    let dir = tempdir().unwrap();
    let src_dir = dir.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    fs::write(src_dir.join("A.tsx"), "function A() { return 1; }").unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let r1 = run_chunk_only_index(&conn, dir.path(), false).unwrap();
    assert_eq!(r1.files_processed, 1);

    let r2 = run_chunk_only_index(&conn, dir.path(), false).unwrap();
    assert_eq!(r2.files_processed, 0);
    assert_eq!(r2.files_skipped, 1);
}

// T-375: run_chunk_only_index_force_reindexes
#[test]
fn run_chunk_only_index_force_reindexes() {
    let dir = tempdir().unwrap();
    let src_dir = dir.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    fs::write(src_dir.join("A.tsx"), "function A() { return 1; }").unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    run_chunk_only_index(&conn, dir.path(), false).unwrap();

    let r2 = run_chunk_only_index_force(&conn, dir.path(), false).unwrap();
    assert_eq!(r2.files_processed, 1);
    assert_eq!(r2.files_skipped, 0);
}

// T-376: run_chunk_only_index_removes_deleted_file_chunks
#[test]
fn run_chunk_only_index_removes_deleted_file_chunks() {
    let dir = tempdir().unwrap();
    let src_dir = dir.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    fs::write(src_dir.join("A.tsx"), "function A() { return 1; }").unwrap();
    fs::write(src_dir.join("B.tsx"), "function B() { return 2; }").unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let r1 = run_chunk_only_index(&conn, dir.path(), false).unwrap();
    assert_eq!(r1.files_processed, 2);
    assert_eq!(
        storage::get_stats(&conn.lock().unwrap())
            .unwrap()
            .total_files,
        2
    );

    fs::remove_file(src_dir.join("B.tsx")).unwrap();

    run_chunk_only_index(&conn, dir.path(), false).unwrap();
    let stats = storage::get_stats(&conn.lock().unwrap()).unwrap();
    assert_eq!(stats.total_files, 1, "orphaned file should be removed");
}

// T-381: run_chunk_only_index_stores_imports_in_file_context
#[test]
fn run_chunk_only_index_stores_imports_in_file_context() {
    let dir = tempdir().unwrap();
    let src_dir = dir.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();
    fs::write(
        src_dir.join("App.tsx"),
        "import { useState } from 'react';\nimport { useAuth } from './useAuth';\nfunction App() { return <div/>; }",
    ).unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    run_chunk_only_index(&conn, dir.path(), false).unwrap();

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

#[cfg(unix)]
// T-398: run_chunk_only_index_counts_unreadable_files_as_errors
#[test]
fn run_chunk_only_index_counts_unreadable_files_as_errors() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempdir().unwrap();
    let src_dir = dir.path().join("src");
    fs::create_dir_all(&src_dir).unwrap();

    fs::write(src_dir.join("Good.tsx"), "function Good() { return 1; }").unwrap();

    let bad_path = src_dir.join("Bad.tsx");
    fs::write(&bad_path, "function Bad() { return 2; }").unwrap();
    fs::set_permissions(&bad_path, fs::Permissions::from_mode(0o000)).unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let result = run_chunk_only_index(&conn, dir.path(), false).unwrap();

    fs::set_permissions(&bad_path, fs::Permissions::from_mode(0o644)).unwrap();

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

// T-207: index_error_from_corpus_error_matches_corpus_init_variant
// Spec FR-208 / FR-216 / FR-217: corpus init errors propagate as
// IndexError::CorpusInit(CorpusError) via #[from].
#[test]
fn index_error_from_corpus_error_matches_corpus_init_variant() {
    let corpus_err = injection::Corpus::load_from_str(
        r#"entries:
  - id: x
    pattern_type: bogus
    pattern: y
    severity: high
    category: c
    expected_flags: []
"#,
    )
    .unwrap_err();
    let index_err: IndexError = corpus_err.into();
    assert!(
        matches!(index_err, IndexError::CorpusInit(_)),
        "From<CorpusError> for IndexError must produce CorpusInit, got: {index_err:?}"
    );
}

// T-208: prepare_chunks_populates_source_kind_and_injection_flags
// Spec FR-209 / FR-212 / FR-213: prepare_chunks accepts &Corpus and
// populates PendingFile.source_kind / injection_flags; to_new_chunks
// borrows them into NewChunk.
#[test]
fn prepare_chunks_populates_source_kind_and_injection_flags() {
    let corpus = injection::Corpus::load_from_str("entries: []").unwrap();
    let checked = CheckedFile {
        rel_path: "src/foo.rs".to_owned(),
        source: "pub fn hello() {}".to_owned(),
        hash: "deadbeef".to_owned(),
    };
    let pf = prepare_chunks(checked, Path::new("src/foo.rs"), None, &corpus)
        .expect("rust source should yield at least one chunk");
    assert_eq!(pf.source_kind, Some(SourceKind::Src));
    assert_eq!(
        pf.injection_flags.len(),
        pf.raw_chunks.len(),
        "injection_flags.len() must equal raw_chunks.len()"
    );
    let new_chunks = pf.to_new_chunks();
    for nc in &new_chunks {
        assert_eq!(nc.source_kind, Some(SourceKind::Src));
        assert!(
            nc.injection_flags.is_some(),
            "NewChunk.injection_flags must be Some after matcher runs"
        );
    }
}

// T-209: prepare_chunks_clean_scan_yields_some_empty_array_not_none
// Spec FR-210 / BR-201 / NFR-203: silent-default-false regression gate.
// Matcher走行 + ヒットなし SHALL be Some("[]"), NOT None.
#[test]
fn prepare_chunks_clean_scan_yields_some_empty_array_not_none() {
    let corpus = injection::Corpus::load_from_str("entries: []").unwrap();
    let checked = CheckedFile {
        rel_path: "src/lib.rs".to_owned(),
        source: "pub fn add(a: i32, b: i32) -> i32 { a + b }".to_owned(),
        hash: "abc123".to_owned(),
    };
    let pf = prepare_chunks(checked, Path::new("src/lib.rs"), None, &corpus).unwrap();
    for (i, flags) in pf.injection_flags.iter().enumerate() {
        assert_eq!(
            flags, "[]",
            "chunk {i} must be \"[]\" (silent-default-false gate: clean-scan is the empty JSON array, never absent)"
        );
    }
}

// T-210: prepare_chunks_matched_yields_some_json_array
// Spec FR-211: corpus-matching content SHALL produce
// Some("[\"flag.id\", ...]") in JSON array form.
#[test]
fn prepare_chunks_matched_yields_some_json_array() {
    let corpus_yaml = r#"entries:
  - id: ignore-prev
    pattern_type: literal
    pattern: "Ignore previous instructions"
    severity: high
    category: instruction-override
    expected_flags: [injection.instruction-override]
"#;
    let corpus = injection::Corpus::load_from_str(corpus_yaml).unwrap();
    let checked = CheckedFile {
        rel_path: "src/lib.rs".to_owned(),
        source: r#"pub fn hello() { let _ = "Ignore previous instructions"; }"#.to_owned(),
        hash: "xyz789".to_owned(),
    };
    let pf = prepare_chunks(checked, Path::new("src/lib.rs"), None, &corpus).unwrap();
    let hit = pf
        .injection_flags
        .iter()
        .any(|f| f == "[\"injection.instruction-override\"]");
    assert!(
        hit,
        "at least one chunk must carry \"[\\\"injection.instruction-override\\\"]\", got: {:?}",
        pf.injection_flags
    );
}

// T-315b: run_chunk_only_index_inner_propagates_exclude_vendor
//
// Spec FR-315a / FR-315b: `run_chunk_only_index_inner` accepts
// `exclude_vendor: bool` and propagates to `walker::walk_source_files`.
//
// Two independent tempdirs (identical layout: `src/lib.rs` + `vendor/util.rs`)
// drive the propagation through to the chunks table. The
// `source_kind = 'vendor'` row count differentiates the two branches:
//
// | exclude_vendor | expected vendor chunk count |
// | -------------- | --------------------------- |
// | true           | == 0                        |
// | false          | > 0                         |
//
// Perspective: Combination (decision table above) + State (observable DB
// state corroborates the propagation).
#[test]
fn run_chunk_only_index_inner_propagates_exclude_vendor() {
    fn setup_files_with_vendor(root: &Path) {
        for (name, body) in [
            ("src/lib.rs", "pub fn keep() {}"),
            ("vendor/util.rs", "pub fn drop_me() {}"),
        ] {
            let path = root.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, body).unwrap();
        }
    }

    fn vendor_chunk_count(conn: &Arc<Mutex<storage::Db>>) -> i64 {
        conn.lock()
            .unwrap()
            .query_row(
                "SELECT COUNT(*) FROM chunks WHERE source_kind = 'vendor'",
                [],
                |row| row.get(0),
            )
            .unwrap()
    }

    // Case A: exclude_vendor=true → vendor row count must be 0.
    let dir_a = tempdir().unwrap();
    setup_files_with_vendor(dir_a.path());
    let db_a = dir_a.path().join(".yomu").join("index.db");
    let conn_a = Arc::new(Mutex::new(storage::open_db(&db_a).unwrap()));
    run_chunk_only_index_inner(&conn_a, dir_a.path(), false, true).unwrap();
    let count_a = vendor_chunk_count(&conn_a);
    assert_eq!(
        count_a, 0,
        "exclude_vendor=true must propagate so walker drops vendor/, got {count_a} vendor chunks"
    );

    // Case B: exclude_vendor=false → vendor row count must be > 0 (PR#2 baseline).
    let dir_b = tempdir().unwrap();
    setup_files_with_vendor(dir_b.path());
    let db_b = dir_b.path().join(".yomu").join("index.db");
    let conn_b = Arc::new(Mutex::new(storage::open_db(&db_b).unwrap()));
    run_chunk_only_index_inner(&conn_b, dir_b.path(), false, false).unwrap();
    let count_b = vendor_chunk_count(&conn_b);
    assert!(
        count_b > 0,
        "exclude_vendor=false must keep PR#2 baseline (walker includes vendor/), got {count_b} vendor chunks"
    );
}

// T-211: prepare_chunks_source_kind_classification_per_rel_path
// Spec FR-212: source_kind classification follows rel_path heuristic.
#[test]
fn prepare_chunks_source_kind_classification_per_rel_path() {
    let corpus = injection::Corpus::load_from_str("entries: []").unwrap();
    let cases = [
        ("src/lib.rs", SourceKind::Src),
        ("tests/foo.rs", SourceKind::Test),
        ("dist/bundle.rs", SourceKind::Vendor),
    ];
    for (rel_path, expected) in cases {
        let checked = CheckedFile {
            rel_path: rel_path.to_owned(),
            source: "pub fn x() {}".to_owned(),
            hash: "h".to_owned(),
        };
        let pf = prepare_chunks(checked, Path::new(rel_path), None, &corpus)
            .unwrap_or_else(|| panic!("rust source should chunk for {rel_path}"));
        assert_eq!(
            pf.source_kind,
            Some(expected),
            "source_kind for {rel_path} should be {expected:?}, got {:?}",
            pf.source_kind
        );
    }
}
