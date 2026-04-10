use super::embedder::{
    RECORDED_WARNINGS, get_recorded_warnings, parse_budget_value, record_embedder_warning,
};
use super::*;
use std::collections::HashMap;

use rurico::embed::{ChunkedEmbedding, FailingEmbedder};

fn parse_json(json: &str) -> serde_json::Value {
    serde_json::from_str(json).unwrap_or_else(|e| panic!("invalid JSON: {e}\n{json}"))
}

fn test_embedding() -> Vec<f32> {
    let mut emb = vec![0.0_f32; storage::EMBEDDING_DIMS];
    emb[0] = 1.0;
    emb
}

fn ce(v: Vec<f32>) -> ChunkedEmbedding {
    ChunkedEmbedding { chunks: vec![v] }
}

fn test_db() -> (storage::Db, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();
    (conn, dir)
}

fn test_yomu() -> (Yomu, tempfile::TempDir) {
    let (conn, dir) = test_db();
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), None);
    (y, dir)
}

fn setup_test_files(files: &[(&str, &str)]) -> (storage::Db, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    for (path, content) in files {
        let full_path = dir.path().join(path);
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&full_path, content).unwrap();
    }
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    (conn, dir)
}

fn test_yomu_with_files(files: &[(&str, &str)]) -> (Yomu, tempfile::TempDir) {
    let (conn, dir) = setup_test_files(files);
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), None);
    (y, dir)
}

fn test_yomu_with_files_and_embedder(
    files: &[(&str, &str)],
    embedder: Arc<dyn rurico::embed::Embed>,
) -> (Yomu, tempfile::TempDir) {
    let (conn, dir) = setup_test_files(files);
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), Some(embedder));
    (y, dir)
}

fn seed_index(conn: &storage::Db) {
    let embedding = test_embedding();
    storage::insert_chunk(
        conn,
        "src/dummy.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Other,
            name: None,
            content: "seed",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        },
        "seed",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();
}

#[test]
fn search_rejects_empty_query() {
    let (y, _dir) = test_yomu();
    let err = y.search("", 10, 0, false).unwrap_err();
    assert!(
        err.to_string().contains("empty"),
        "expected empty error, got: {}",
        err
    );
}

#[test]
fn search_rejects_long_query() {
    let (y, _dir) = test_yomu();
    let long_query = "a".repeat(MAX_QUERY_LENGTH + 1);
    let err = y.search(&long_query, 10, 0, false).unwrap_err();
    assert!(
        err.to_string().contains("maximum length"),
        "expected max length error, got: {}",
        err
    );
}

#[test]
fn search_without_embedder_degrades_gracefully() {
    let (y, _dir) = test_yomu();
    let text = y.search("test query", 10, 0, false).unwrap();
    assert!(
        text.contains("No results found"),
        "expected no results: {text}"
    );
}

#[test]
fn search_auto_indexes_empty_db() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/Button.tsx", "function Button() { return <div/>; }")],
        Arc::new(rurico::embed::MockEmbedder::default()),
    );

    let text = y.search("button component", 10, 0, false).unwrap();
    assert!(
        !text.contains("No results found"),
        "expected results after auto-index, got: {text}"
    );
    assert!(
        text.contains("Button"),
        "expected Button in results, got: {text}"
    );

    let stats = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap()
    };
    assert!(stats.total_chunks > 0, "expected chunks after auto-index");
    assert!(
        stats.embedded_chunks > 0,
        "expected embeddings after auto-index"
    );
}

#[test]
fn search_incremental_embeds_chunked_only() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/Form.tsx", "export function Form() { return <form/>; }")],
        Arc::new(rurico::embed::MockEmbedder::default()),
    );

    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    {
        let c = y.conn.lock().unwrap();
        let stats = storage::get_stats(&c).unwrap();
        assert!(stats.total_chunks > 0, "should have chunks");
        assert_eq!(stats.embedded_chunks, 0, "should have no embeddings yet");
    }

    let text = y.search("form component", 10, 0, false).unwrap();
    assert!(text.contains("Form"), "expected Form in results: {text}");

    {
        let c = y.conn.lock().unwrap();
        let stats = storage::get_stats(&c).unwrap();
        assert!(
            stats.embedded_chunks > 0,
            "expected embeddings after incremental embed"
        );
    }
}

#[test]
fn search_shows_coverage_on_no_results() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();

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
        "hash1",
        "",
        &[],
        None,
    )
    .unwrap();

    let stats = storage::get_stats(&conn).unwrap();
    assert!(stats.total_chunks > 0, "should have chunks");
    assert_eq!(stats.embedded_chunks, 0, "should have no embeddings");

    let msg = format_no_results_message(&stats);
    assert!(msg.contains("coverage"), "expected coverage info: {msg}");
    assert!(msg.contains("0/"), "expected 0 embedded: {msg}");
}

#[test]
fn search_degraded_empty_results_shows_note() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();

    storage::replace_file_chunks_only(
        &conn,
        "src/Button.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("Button"),
            content: "function Button() { return <div/>; }",
            start_line: 1,
            end_line: 3,
            parent_index: None,
        }],
        "h1",
        "",
        &[],
        None,
    )
    .unwrap();

    let y = Yomu::for_test(
        conn,
        dir.path().to_path_buf(),
        Some(Arc::new(FailingEmbedder::all_fail("service unavailable"))
            as Arc<dyn rurico::embed::Embed>),
    );

    let text = y.search("zzzznonexistent", 10, 0, false).unwrap();
    assert!(
        text.contains("No results found"),
        "expected no results: {text}"
    );
    assert!(
        text.contains("embedding"),
        "expected embedding note in empty results: {text}"
    );
}

#[test]
fn search_degraded_with_results_shows_note() {
    let (y, dir) = test_yomu_with_files_and_embedder(
        &[(
            "src/Button.tsx",
            "export function Button() { return <div/>; }",
        )],
        Arc::new(rurico::embed::MockEmbedder::default()),
    );

    indexer::run_index(
        Arc::clone(&y.conn),
        y.root.as_path(),
        &rurico::embed::MockEmbedder::default(),
        false,
    )
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn2 = storage::open_db(&db_path).unwrap();
    let y_failing = Yomu::for_test(
        conn2,
        dir.path().to_path_buf(),
        Some(Arc::new(FailingEmbedder::all_fail("service unavailable"))
            as Arc<dyn rurico::embed::Embed>),
    );

    let result = y_failing.search("Button", 10, 0, false).unwrap();
    assert!(result.contains("Button"), "should have search results");
    assert!(
        result.contains("embedding model not loaded"),
        "should show degraded note"
    );
}

#[test]
fn format_results_grouped_renders_file_header_and_context() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/Button.tsx".to_string(),
            chunk_type: storage::ChunkType::Component,
            name: Some("Button".to_string()),
            content: "function Button() { return <div/>; }".to_string(),
            start_line: 5,
            end_line: 7,
            parent_chunk_id: None,
        },
        chunk_id: None,
        distance: 0.15,
        match_source: storage::MatchSource::Semantic,
        score: 1.0 / (1.0 + 0.15),
    }];
    let imports_map = HashMap::from([(
        "src/Button.tsx".to_string(),
        "import React from 'react'".to_string(),
    )]);
    let siblings_map = HashMap::from([(
        "src/Button.tsx".to_string(),
        vec![storage::SiblingInfo {
            name: Some("ButtonProps".to_string()),
            chunk_type: storage::ChunkType::TypeDef,
            start_line: 1,
            end_line: 3,
        }],
    )]);
    let ctx = EnrichmentContext {
        imports: imports_map,
        siblings: siblings_map,
    };
    let text = format_results_grouped(&results, &ctx, &HashMap::new());
    assert!(
        text.contains("## src/Button.tsx"),
        "missing file header: {text}"
    );
    assert!(
        text.contains("Imports: import React from 'react'"),
        "missing imports: {text}"
    );
    assert!(
        text.contains("Siblings: ButtonProps [type_def]"),
        "missing siblings: {text}"
    );
    assert!(text.contains("Button"), "missing chunk name: {text}");
    assert!(text.contains("0.87"), "missing similarity: {text}");
}

#[test]
fn format_results_grouped_groups_same_file_chunks() {
    let results = vec![
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/Form.tsx".to_string(),
                chunk_type: storage::ChunkType::Component,
                name: Some("Form".to_string()),
                content: "function Form() {}".to_string(),
                start_line: 1,
                end_line: 5,
                parent_chunk_id: None,
            },
            chunk_id: None,
            distance: 0.1,
            match_source: storage::MatchSource::Semantic,
            score: 1.0 / (1.0 + 0.1),
        },
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/Form.tsx".to_string(),
                chunk_type: storage::ChunkType::Hook,
                name: Some("useForm".to_string()),
                content: "function useForm() {}".to_string(),
                start_line: 7,
                end_line: 10,
                parent_chunk_id: None,
            },
            chunk_id: None,
            distance: 0.2,
            match_source: storage::MatchSource::Semantic,
            score: 1.0 / (1.0 + 0.2),
        },
    ];
    let ctx = EnrichmentContext {
        imports: HashMap::new(),
        siblings: HashMap::new(),
    };
    let text = format_results_grouped(&results, &ctx, &HashMap::new());
    assert_eq!(
        text.matches("## src/Form.tsx").count(),
        1,
        "expected one file header: {text}"
    );
    assert!(text.contains("Form"), "missing Form: {text}");
    assert!(text.contains("useForm"), "missing useForm: {text}");
}

#[test]
fn format_results_grouped_deduplicates_siblings() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/A.tsx".to_string(),
            chunk_type: storage::ChunkType::Component,
            name: Some("A".to_string()),
            content: "function A() {}".to_string(),
            start_line: 5,
            end_line: 7,
            parent_chunk_id: None,
        },
        chunk_id: None,
        distance: 0.1,
        match_source: storage::MatchSource::Semantic,
        score: 1.0 / (1.0 + 0.1),
    }];
    let siblings_map = HashMap::from([(
        "src/A.tsx".to_string(),
        vec![
            storage::SiblingInfo {
                name: Some("A".to_string()),
                chunk_type: storage::ChunkType::Component,
                start_line: 5,
                end_line: 7,
            },
            storage::SiblingInfo {
                name: Some("AProps".to_string()),
                chunk_type: storage::ChunkType::TypeDef,
                start_line: 1,
                end_line: 3,
            },
        ],
    )]);
    let ctx = EnrichmentContext {
        imports: HashMap::new(),
        siblings: siblings_map,
    };
    let text = format_results_grouped(&results, &ctx, &HashMap::new());
    assert!(
        text.contains("AProps [type_def]"),
        "sibling should be included: {text}"
    );
    let siblings_line = text.lines().find(|l| l.starts_with("Siblings:")).unwrap();
    assert!(
        !siblings_line.contains("A [component]"),
        "search result should be excluded from siblings: {siblings_line}"
    );
}

#[test]
fn format_results_grouped_omits_empty_imports() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/A.tsx".to_string(),
            chunk_type: storage::ChunkType::Component,
            name: Some("A".to_string()),
            content: "code".to_string(),
            start_line: 1,
            end_line: 3,
            parent_chunk_id: None,
        },
        chunk_id: None,
        distance: 0.1,
        match_source: storage::MatchSource::Semantic,
        score: 1.0 / (1.0 + 0.1),
    }];
    let imports_map = HashMap::from([("src/A.tsx".to_string(), String::new())]);
    let ctx = EnrichmentContext {
        imports: imports_map,
        siblings: HashMap::new(),
    };
    let text = format_results_grouped(&results, &ctx, &HashMap::new());
    assert!(
        !text.contains("Imports:"),
        "empty imports should be omitted: {text}"
    );
}

#[test]
fn format_results_grouped_omits_empty_siblings() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/A.tsx".to_string(),
            chunk_type: storage::ChunkType::Component,
            name: Some("A".to_string()),
            content: "code".to_string(),
            start_line: 1,
            end_line: 3,
            parent_chunk_id: None,
        },
        chunk_id: None,
        distance: 0.1,
        match_source: storage::MatchSource::Semantic,
        score: 1.0 / (1.0 + 0.1),
    }];
    let siblings_map = HashMap::from([("src/A.tsx".to_string(), vec![])]);
    let ctx = EnrichmentContext {
        imports: HashMap::new(),
        siblings: siblings_map,
    };
    let text = format_results_grouped(&results, &ctx, &HashMap::new());
    assert!(
        !text.contains("Siblings:"),
        "empty siblings should be omitted: {text}"
    );
}

#[test]
fn format_results_grouped_sorts_files_by_best_similarity() {
    let results = vec![
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/B.tsx".to_string(),
                chunk_type: storage::ChunkType::Component,
                name: Some("B".to_string()),
                content: "code B".to_string(),
                start_line: 1,
                end_line: 3,
                parent_chunk_id: None,
            },
            chunk_id: None,
            distance: 0.5,
            match_source: storage::MatchSource::Semantic,
            score: 1.0 / (1.0 + 0.5),
        },
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/A.tsx".to_string(),
                chunk_type: storage::ChunkType::Component,
                name: Some("A".to_string()),
                content: "code A".to_string(),
                start_line: 1,
                end_line: 3,
                parent_chunk_id: None,
            },
            chunk_id: None,
            distance: 0.1,
            match_source: storage::MatchSource::Semantic,
            score: 1.0 / (1.0 + 0.1),
        },
    ];
    let ctx = EnrichmentContext {
        imports: HashMap::new(),
        siblings: HashMap::new(),
    };
    let text = format_results_grouped(&results, &ctx, &HashMap::new());
    let a_pos = text.find("## src/A.tsx").unwrap();
    let b_pos = text.find("## src/B.tsx").unwrap();
    assert!(
        a_pos < b_pos,
        "A (better similarity) should come before B: {text}"
    );
}

#[test]
fn format_results_grouped_shows_score_for_all() {
    let results = vec![
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/A.tsx".to_string(),
                chunk_type: storage::ChunkType::Component,
                name: Some("A".to_string()),
                content: "function A() {}".to_string(),
                start_line: 1,
                end_line: 3,
                parent_chunk_id: None,
            },
            chunk_id: None,
            distance: 0.5,
            match_source: storage::MatchSource::Semantic,
            score: 0.72,
        },
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/B.tsx".to_string(),
                chunk_type: storage::ChunkType::Hook,
                name: Some("useAuth".to_string()),
                content: "function useAuth() {}".to_string(),
                start_line: 1,
                end_line: 3,
                parent_chunk_id: None,
            },
            chunk_id: None,
            distance: f32::INFINITY,
            match_source: storage::MatchSource::Fts,
            score: 0.55,
        },
    ];
    let ctx = EnrichmentContext {
        imports: HashMap::new(),
        siblings: HashMap::new(),
    };
    let text = format_results_grouped(&results, &ctx, &HashMap::new());
    assert!(
        text.contains("(similarity: 0.72)"),
        "expected score for Semantic: {text}"
    );
    assert!(
        text.contains("(similarity: 0.55)"),
        "expected score for NameMatch: {text}"
    );
    assert!(
        !text.contains("(name match)"),
        "should not show (name match): {text}"
    );
}

#[test]
fn index_works_without_api_key() {
    let (y, _dir) = test_yomu_with_files(&[("src/A.tsx", "function A() {}")]);
    let text = y.index(false).unwrap();
    assert!(text.contains("complete"), "expected success: {text}");
}

#[test]
fn index_chunks_without_embedding() {
    let (y, _dir) = test_yomu_with_files(&[
        (
            "src/Header.tsx",
            "export function Header() { return <header/>; }",
        ),
        (
            "src/Footer.tsx",
            "export function Footer() { return <footer/>; }",
        ),
    ]);

    let text = y.index(false).unwrap();
    assert!(text.contains("complete"), "expected completion: {text}");
    assert!(
        text.contains("Embedding coverage:"),
        "should show coverage gap: {text}"
    );

    let stats = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap()
    };
    assert!(stats.total_chunks > 0, "should have chunks");
    assert_eq!(stats.embedded_chunks, 0, "should have no embeddings");
}

#[test]
fn rebuild_re_parses_all_files() {
    let (y, dir) = test_yomu_with_files(&[("src/A.tsx", "export function A() { return <div/>; }")]);
    y.index(false).unwrap();

    let chunks_before = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap().total_chunks
    };
    assert!(chunks_before > 0, "should have chunks after index");

    std::fs::write(
        dir.path().join("src/B.tsx"),
        "export function B() { return <span/>; }",
    )
    .unwrap();

    let text = y.rebuild(false).unwrap();
    assert!(text.contains("complete"), "expected completion: {text}");

    let chunks_after = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap().total_chunks
    };
    assert!(
        chunks_after > chunks_before,
        "rebuild should pick up new file: {chunks_before} -> {chunks_after}"
    );
}

#[test]
fn status_returns_empty_stats() {
    let (y, _dir) = test_yomu();
    let text = y.status(false).unwrap();
    assert!(text.contains("Files: 0"), "expected 0 files, got: {text}");
    assert!(text.contains("Chunks: 0"), "expected 0 chunks, got: {text}");
    assert!(
        text.contains("References: 0"),
        "expected 0 references, got: {text}"
    );
    assert!(text.contains("never"), "expected 'never', got: {text}");
}

#[test]
fn status_returns_counts_after_insert() {
    let (conn, _dir) = test_db();
    let embedding = test_embedding();
    storage::insert_chunk(
        &conn,
        "src/A.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
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

    let y = Yomu::for_test(conn, PathBuf::from("/tmp"), None);
    let text = y.status(false).unwrap();
    assert!(text.contains("Files: 1"), "expected 1 file, got: {text}");
    assert!(text.contains("Chunks: 1"), "expected 1 chunk, got: {text}");
}

#[test]
fn status_shows_embedded_total() {
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
        "hash1",
        "",
        &[],
        None,
    )
    .unwrap();

    let y = Yomu::for_test(conn, PathBuf::from("/tmp"), None);
    let text = y.status(false).unwrap();
    assert!(text.contains("0/1"), "expected 0/1 in status: {text}");
}

#[test]
fn impact_lists_dependents() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
        storage::replace_file_references(
            &conn,
            "src/A.tsx",
            &[storage::Reference {
                source_file: "src/A.tsx".into(),
                target_file: "src/hooks/useAuth.ts".into(),
                symbol_name: Some("useAuth".into()),
                ref_kind: storage::RefKind::Named,
            }],
        )
        .unwrap();
        storage::replace_file_references(
            &conn,
            "src/C.tsx",
            &[storage::Reference {
                source_file: "src/C.tsx".into(),
                target_file: "src/hooks/useAuth.ts".into(),
                symbol_name: Some("useAuth".into()),
                ref_kind: storage::RefKind::Named,
            }],
        )
        .unwrap();
    }

    let text = y.impact("src/hooks/useAuth.ts", None, 3, false).unwrap();
    assert!(text.contains("src/A.tsx"), "expected A.tsx: {text}");
    assert!(text.contains("src/C.tsx"), "expected C.tsx: {text}");
    assert!(
        text.contains("2 dependent"),
        "expected 2 dependents: {text}"
    );
    assert!(
        text.contains("Depth"),
        "expected depth heading in output: {text}"
    );
}

#[test]
fn impact_filters_by_symbol() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
        storage::replace_file_references(
            &conn,
            "src/A.tsx",
            &[storage::Reference {
                source_file: "src/A.tsx".into(),
                target_file: "src/hooks/useAuth.ts".into(),
                symbol_name: Some("useAuth".into()),
                ref_kind: storage::RefKind::Named,
            }],
        )
        .unwrap();
        storage::replace_file_references(
            &conn,
            "src/B.tsx",
            &[storage::Reference {
                source_file: "src/B.tsx".into(),
                target_file: "src/hooks/useAuth.ts".into(),
                symbol_name: Some("AuthProvider".into()),
                ref_kind: storage::RefKind::Named,
            }],
        )
        .unwrap();
    }

    let text = y
        .impact("src/hooks/useAuth.ts:useAuth", None, 3, false)
        .unwrap();
    assert!(
        text.contains("Direct symbol references"),
        "expected symbol section: {text}"
    );
    assert!(
        text.contains("src/A.tsx"),
        "expected A.tsx in symbol refs: {text}"
    );
}

#[test]
fn impact_rejects_empty_target() {
    let (y, _dir) = test_yomu();
    let err = y.impact("", None, 3, false).unwrap_err();
    assert!(
        err.to_string().contains("empty"),
        "expected empty error, got: {err}"
    );
}

#[test]
fn impact_errors_on_empty_index() {
    let (y, _dir) = test_yomu();
    let err = y.impact("src/A.tsx", None, 3, false).unwrap_err();
    assert!(
        err.to_string().contains("index is empty"),
        "expected empty index error, got: {err}"
    );
}

#[test]
fn impact_distinguishes_missing_file() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
    }
    let text = y.impact("src/nonexistent.tsx", None, 3, false).unwrap();
    assert!(
        text.contains("not found in index"),
        "expected file-not-found message: {text}"
    );
}

#[test]
fn impact_rejects_path_traversal() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
    }
    let err = y.impact("../etc/passwd", None, 3, false).unwrap_err();
    assert!(
        err.to_string().contains(".."),
        "expected path traversal error, got: {err}"
    );
}

#[test]
fn impact_rejects_absolute_path() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
    }
    let err = y.impact("/etc/passwd", None, 3, false).unwrap_err();
    assert!(
        err.to_string().contains("relative"),
        "expected rejection of absolute path, got: {err}"
    );
}

#[test]
fn impact_symbol_flag_overrides_colon_syntax() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
        storage::replace_file_references(
            &conn,
            "src/A.tsx",
            &[storage::Reference {
                source_file: "src/A.tsx".into(),
                target_file: "src/hooks/useAuth.ts".into(),
                symbol_name: Some("useAuth".into()),
                ref_kind: storage::RefKind::Named,
            }],
        )
        .unwrap();
        storage::replace_file_references(
            &conn,
            "src/B.tsx",
            &[storage::Reference {
                source_file: "src/B.tsx".into(),
                target_file: "src/hooks/useAuth.ts".into(),
                symbol_name: Some("AuthProvider".into()),
                ref_kind: storage::RefKind::Named,
            }],
        )
        .unwrap();
    }

    let text = y
        .impact(
            "src/hooks/useAuth.ts:useAuth",
            Some("AuthProvider"),
            3,
            false,
        )
        .unwrap();
    assert!(
        text.contains("src/B.tsx"),
        "expected B.tsx for AuthProvider: {text}"
    );
}

#[test]
fn integration_index_then_impact() {
    let (y, _dir) = test_yomu_with_files(&[
        (
            "src/A.tsx",
            "import { B } from './B';\nfunction A() { return <B/>; }",
        ),
        (
            "src/B.tsx",
            "import { C } from './C';\nexport function B() { return <C/>; }",
        ),
        ("src/C.tsx", "export function C() { return <div/>; }"),
    ]);

    indexer::run_index(
        Arc::clone(&y.conn),
        y.root.as_path(),
        &rurico::embed::MockEmbedder::default(),
        false,
    )
    .unwrap();

    let text = y.impact("src/C.tsx", None, 3, false).unwrap();
    assert!(
        text.contains("src/B.tsx"),
        "expected B.tsx as direct dependent: {text}"
    );
    assert!(
        text.contains("src/A.tsx"),
        "expected A.tsx as transitive dependent: {text}"
    );
}

#[test]
fn parse_impact_target_file_only() {
    let (file, symbol) = parse_impact_target("src/hooks/useAuth.ts");
    assert_eq!(file, "src/hooks/useAuth.ts");
    assert_eq!(symbol, None);
}

#[test]
fn parse_impact_target_with_symbol() {
    let (file, symbol) = parse_impact_target("src/hooks/useAuth.ts:useAuth");
    assert_eq!(file, "src/hooks/useAuth.ts");
    assert_eq!(symbol, Some("useAuth"));
}

#[test]
fn parse_impact_target_trailing_colon() {
    let (file, symbol) = parse_impact_target("src/A.tsx:");
    assert_eq!(file, "src/A.tsx:");
    assert_eq!(symbol, None);
}

#[test]
fn parse_impact_target_leading_colon() {
    let (file, symbol) = parse_impact_target(":symbol");
    assert_eq!(file, ":symbol");
    assert_eq!(symbol, None);
}

#[test]
fn parse_impact_target_symbol_with_slash_treated_as_file() {
    let (file, symbol) = parse_impact_target("src/A.tsx:path/like");
    assert_eq!(file, "src/A.tsx:path/like");
    assert_eq!(symbol, None);
}

#[test]
fn parse_impact_target_multiple_colons() {
    let (file, symbol) = parse_impact_target("src/A.tsx:B:C");
    assert_eq!(file, "src/A.tsx:B");
    assert_eq!(symbol, Some("C"));
}

#[test]
fn parse_budget_value_valid() {
    assert_eq!(parse_budget_value(Some("100")), 100);
    assert_eq!(parse_budget_value(Some("1")), 1);
    assert_eq!(parse_budget_value(Some("500")), 500);
}

#[test]
fn parse_budget_value_out_of_range() {
    assert_eq!(parse_budget_value(Some("0")), DEFAULT_EMBED_BUDGET);
    assert_eq!(parse_budget_value(Some("501")), DEFAULT_EMBED_BUDGET);
}

#[test]
fn parse_budget_value_invalid() {
    assert_eq!(parse_budget_value(Some("abc")), DEFAULT_EMBED_BUDGET);
    assert_eq!(parse_budget_value(Some("")), DEFAULT_EMBED_BUDGET);
}

#[test]
fn parse_budget_value_missing() {
    assert_eq!(parse_budget_value(None), DEFAULT_EMBED_BUDGET);
}

#[test]
fn determine_index_state_variants() {
    let empty = storage::IndexStatus {
        total_files: 0,
        total_chunks: 0,
        embeddable_chunks: 0,
        embedded_chunks: 0,
        last_indexed_at: None,
    };
    assert!(matches!(determine_index_state(&empty), IndexState::Empty));

    let chunked = storage::IndexStatus {
        total_files: 5,
        total_chunks: 20,
        embeddable_chunks: 20,
        embedded_chunks: 0,
        last_indexed_at: Some("2026-01-01".into()),
    };
    assert!(matches!(
        determine_index_state(&chunked),
        IndexState::ChunkedOnly
    ));

    let partial = storage::IndexStatus {
        total_files: 5,
        total_chunks: 20,
        embeddable_chunks: 20,
        embedded_chunks: 10,
        last_indexed_at: Some("2026-01-01".into()),
    };
    assert!(matches!(
        determine_index_state(&partial),
        IndexState::PartiallyEmbedded
    ));

    let full = storage::IndexStatus {
        total_files: 5,
        total_chunks: 20,
        embeddable_chunks: 20,
        embedded_chunks: 20,
        last_indexed_at: Some("2026-01-01".into()),
    };
    assert!(matches!(
        determine_index_state(&full),
        IndexState::FullyEmbedded
    ));
}

#[test]
fn ensure_indexed_partially_embedded_triggers_embed() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/App.tsx", "export function App() { return <div/>; }")],
        Arc::new(rurico::embed::MockEmbedder::default()),
    );

    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let stats_before = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap()
    };
    assert!(stats_before.total_chunks > 0, "should have chunks");
    assert_eq!(
        stats_before.embedded_chunks, 0,
        "should have no embeddings yet"
    );

    let result = y.search("App component", 10, 0, false).unwrap();
    assert!(
        result.contains("App"),
        "should find App after embedding: {result}"
    );

    let stats_after = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap()
    };
    assert!(
        stats_after.embedded_chunks > 0,
        "should have embeddings after search"
    );
}

#[test]
fn ensure_indexed_fully_embedded_skips_embed() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[(
            "src/Button.tsx",
            "export function Button() { return <button/>; }",
        )],
        Arc::new(rurico::embed::MockEmbedder::default()),
    );

    indexer::run_index(
        Arc::clone(&y.conn),
        y.root.as_path(),
        &rurico::embed::MockEmbedder::default(),
        false,
    )
    .unwrap();

    let stats = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap()
    };
    assert_eq!(
        stats.embedded_chunks, stats.total_chunks,
        "should be fully embedded"
    );

    let result = y.search("Button", 10, 0, false).unwrap();
    assert!(result.contains("Button"), "should find Button: {result}");
}

#[test]
fn ensure_indexed_fully_embedded_with_failing_embedder() {
    let (y, dir) = test_yomu_with_files_and_embedder(
        &[("src/Card.tsx", "export function Card() { return <div/>; }")],
        Arc::new(rurico::embed::MockEmbedder::default()),
    );

    indexer::run_index(
        Arc::clone(&y.conn),
        y.root.as_path(),
        &rurico::embed::MockEmbedder::default(),
        false,
    )
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn2 = storage::open_db(&db_path).unwrap();
    let y_failing = Yomu::for_test(
        conn2,
        dir.path().to_path_buf(),
        Some(Arc::new(FailingEmbedder::all_fail("service unavailable"))
            as Arc<dyn rurico::embed::Embed>),
    );

    let result = y_failing.search("Card", 10, 0, false).unwrap();
    assert!(
        result.contains("Card"),
        "should find Card with existing embeddings: {result}"
    );
}

#[test]
fn with_root_creates_db_and_returns_yomu() {
    let dir = tempfile::tempdir().unwrap();
    let result = Yomu::with_root(dir.path().to_path_buf());
    assert!(
        result.is_ok(),
        "with_root should succeed: {:?}",
        result.err()
    );
    let yomu = result.unwrap();
    assert_eq!(yomu.root, dir.path());
    assert!(dir.path().join(".yomu").join("index.db").exists());
}

#[test]
fn search_without_embedder_skips_embed_attempt() {
    let (y, _dir) =
        test_yomu_with_files(&[("src/Card.tsx", "export function Card() { return <div/>; }")]);

    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    {
        let c = y.conn.lock().unwrap();
        let stats = storage::get_stats(&c).unwrap();
        assert!(stats.total_chunks > 0, "should have chunks");
        assert_eq!(stats.embedded_chunks, 0, "should have no embeddings");
    }

    let text = y.search("card", 10, 0, false).unwrap();
    assert!(
        !text.contains("embedding failed"),
        "should not attempt embed when embedder unavailable: {text}"
    );
}

#[test]
fn search_json_format_returns_valid_json() {
    let (y, _dir) = test_yomu_with_files(&[(
        "src/Button.tsx",
        "export function Button() { return <button/>; }",
    )]);
    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let json = y.search("button", 10, 0, true).unwrap();
    let parsed = parse_json(&json);
    assert!(
        parsed["results"].is_array(),
        "should have results array: {json}"
    );
    assert!(
        parsed.get("degraded").is_some(),
        "should have degraded field: {json}"
    );
    assert!(
        parsed.get("notes").is_some(),
        "should have notes field: {json}"
    );
    assert!(
        json.contains("\"file\":\"src/Button.tsx\""),
        "should contain file path: {json}"
    );
    assert!(json.contains("\"score\":"), "should contain score: {json}");
}

#[test]
fn search_json_format_empty_results() {
    let (y, _dir) =
        test_yomu_with_files(&[("src/A.tsx", "export function A() { return <div/>; }")]);
    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let json = y.search("zzzznonexistent", 10, 0, true).unwrap();
    let parsed = parse_json(&json);
    assert!(
        parsed["results"].as_array().unwrap().is_empty(),
        "empty results: {json}"
    );
    assert!(
        parsed.get("degraded").is_some(),
        "should have degraded field: {json}"
    );
    assert!(
        parsed.get("notes").is_some(),
        "should have notes field: {json}"
    );
}

#[test]
fn dry_run_index_does_not_write_to_db() {
    let (y, _dir) = test_yomu_with_files(&[
        ("src/A.tsx", "export function A() {}"),
        ("src/B.tsx", "export function B() {}"),
    ]);

    let text = y.dry_run_index(false, false).unwrap();
    assert!(
        text.contains("2 files to process"),
        "should report files to process: {text}"
    );

    let stats = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap()
    };
    assert_eq!(
        stats.total_chunks, 0,
        "dry run should not create any chunks"
    );
}

#[test]
fn dry_run_index_shows_skip_for_unchanged() {
    let (y, _dir) = test_yomu_with_files(&[
        ("src/A.tsx", "export function A() {}"),
        ("src/B.tsx", "export function B() {}"),
    ]);

    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let text = y.dry_run_index(false, false).unwrap();
    assert!(
        text.contains("0 files to process"),
        "all files should be skipped: {text}"
    );
    assert!(
        text.contains("2 files unchanged"),
        "should show unchanged count: {text}"
    );
}

#[test]
fn search_json_format_degraded_includes_flag() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/Card.tsx", "export function Card() { return <div/>; }")],
        Arc::new(FailingEmbedder::all_fail("service unavailable")),
    );
    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let json = y.search("card", 10, 0, true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(
        parsed["degraded"], true,
        "should be degraded when embedder fails: {json}"
    );
    assert!(
        !parsed["results"].as_array().unwrap().is_empty(),
        "should still have results: {json}"
    );
}

#[test]
fn t011_innerfn_hit_shows_parent_context_in_text_output() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/UserForm.tsx".to_string(),
            chunk_type: storage::ChunkType::InnerFn,
            name: Some("handleSubmit".to_string()),
            content: "const handleSubmit = (e) => {\n  e.preventDefault();\n  submit(name);\n};"
                .to_string(),
            start_line: 10,
            end_line: 13,
            parent_chunk_id: Some(42),
        },
        chunk_id: Some(43),
        distance: 0.15,
        match_source: storage::MatchSource::Fts,
        score: 0.85,
    }];
    let ctx = EnrichmentContext {
        imports: HashMap::new(),
        siblings: HashMap::new(),
    };
    let mut parent_chunks = HashMap::new();
    parent_chunks.insert(
        42,
        storage::Chunk {
            file_path: "src/UserForm.tsx".to_string(),
            chunk_type: storage::ChunkType::Component,
            name: Some("UserForm".to_string()),
            content: "function UserForm() {\n  return <form/>;\n}".to_string(),
            start_line: 1,
            end_line: 20,
            parent_chunk_id: None,
        },
    );
    let text = format_results_grouped(&results, &ctx, &parent_chunks);
    assert!(
        text.contains("Parent context:"),
        "[T-011] InnerFn hit should display parent context section: {text}"
    );
}

#[test]
fn t012_parent_hit_no_duplicate_parent_display() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/UserForm.tsx".to_string(),
            chunk_type: storage::ChunkType::Component,
            name: Some("UserForm".to_string()),
            content: "function UserForm() {\n  return <form/>;\n}".to_string(),
            start_line: 1,
            end_line: 20,
            parent_chunk_id: None,
        },
        chunk_id: Some(42),
        distance: 0.1,
        match_source: storage::MatchSource::Semantic,
        score: 0.90,
    }];
    let ctx = EnrichmentContext {
        imports: HashMap::new(),
        siblings: HashMap::new(),
    };
    let text = format_results_grouped(&results, &ctx, &HashMap::new());

    assert!(
        !text.contains("Parent context:"),
        "[T-012] parent chunk hit should NOT show 'Parent context:' section: {text}"
    );
    assert!(
        text.contains("UserForm"),
        "[T-012] parent chunk content should still appear normally: {text}"
    );
}

#[test]
fn t015_json_output_includes_parent_chunk_id() {
    let results = vec![
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/UserForm.tsx".to_string(),
                chunk_type: storage::ChunkType::InnerFn,
                name: Some("handleSubmit".to_string()),
                content: "const handleSubmit = () => {}".to_string(),
                start_line: 10,
                end_line: 13,
                parent_chunk_id: Some(42),
            },
            chunk_id: Some(43),
            distance: 0.2,
            match_source: storage::MatchSource::Fts,
            score: 0.80,
        },
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/App.tsx".to_string(),
                chunk_type: storage::ChunkType::Component,
                name: Some("App".to_string()),
                content: "function App() {}".to_string(),
                start_line: 1,
                end_line: 5,
                parent_chunk_id: None,
            },
            chunk_id: Some(1),
            distance: 0.3,
            match_source: storage::MatchSource::Semantic,
            score: 0.75,
        },
    ];

    let json = format_results_json(&results, false, vec![]);
    let parsed = parse_json(&json);

    let items = parsed["results"].as_array().unwrap();
    assert_eq!(items.len(), 2, "should have 2 results");

    let innerfn_item = &items[0];
    assert_eq!(
        innerfn_item["parent_chunk_id"], 42,
        "[T-015] InnerFn result should have parent_chunk_id: {json}"
    );

    let component_item = &items[1];
    assert!(
        component_item.get("parent_chunk_id").is_some(),
        "[T-015] Component result should have parent_chunk_id field: {json}"
    );
    assert!(
        component_item["parent_chunk_id"].is_null(),
        "[T-015] Component result parent_chunk_id should be null: {json}"
    );
}

#[test]
fn t_ac5_subchunk_innerfn_is_hit_at_1_for_inner_function_query() {
    let mut lines = vec![
        "export function UserForm() {".to_string(),
        "  const [name, setName] = useState('');".to_string(),
        "  const handleSubmit = () => {".to_string(),
        "    submitFormData(name);".to_string(),
        "    resetForm();".to_string(),
        "  };".to_string(),
        "  const handleCancel = () => {".to_string(),
        "    resetForm();".to_string(),
        "  };".to_string(),
    ];
    for i in 0..48 {
        lines.push(format!("  const pad{i} = {i};"));
    }
    lines.push("  return <form onSubmit={handleSubmit}><input/></form>;".to_string());
    lines.push("}".to_string());
    let fixture = lines.join("\n");

    let (y, _dir) = test_yomu_with_files(&[("src/UserForm.tsx", &fixture)]);

    y.index(false).unwrap();

    let result = y.search("handleSubmit", 10, 0, false).unwrap();
    assert!(
        result.contains("handleSubmit"),
        "[AC-5] search should find handleSubmit: {result}"
    );
    assert!(
        result.contains("inner_fn"),
        "[AC-5] handleSubmit should be typed as inner_fn: {result}"
    );
}

#[test]
fn t_ac5_below_threshold_no_subchunks_in_index() {
    let fixture = r#"export function SmallCard() {
  const [count, setCount] = useState(0);
  const handleClick = () => { setCount(count + 1); };
  return <div><button onClick={handleClick}>{count}</button></div>;
}"#;
    let (y, _dir) = test_yomu_with_files(&[("src/SmallCard.tsx", fixture)]);
    y.index(false).unwrap();

    let stats = y.status(false).unwrap();
    assert!(
        !stats.contains("inner_fn"),
        "[AC-5] below-threshold component should not produce InnerFn: {stats}"
    );
}

#[test]
fn tc_003_parent_and_child_both_in_results_no_duplicate() {
    let parent_id = 42i64;
    let results = vec![
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/UserForm.tsx".to_string(),
                chunk_type: storage::ChunkType::Component,
                name: Some("UserForm".to_string()),
                content: "function UserForm() { return <form/>; }".to_string(),
                start_line: 1,
                end_line: 20,
                parent_chunk_id: None,
            },
            chunk_id: Some(parent_id),
            distance: 0.1,
            match_source: storage::MatchSource::Semantic,
            score: 0.90,
        },
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/UserForm.tsx".to_string(),
                chunk_type: storage::ChunkType::InnerFn,
                name: Some("handleSubmit".to_string()),
                content: "const handleSubmit = () => {}".to_string(),
                start_line: 10,
                end_line: 13,
                parent_chunk_id: Some(parent_id),
            },
            chunk_id: Some(43),
            distance: 0.15,
            match_source: storage::MatchSource::Fts,
            score: 0.85,
        },
    ];
    let ctx = EnrichmentContext {
        imports: HashMap::new(),
        siblings: HashMap::new(),
    };
    let mut parent_chunks = HashMap::new();
    parent_chunks.insert(
        parent_id,
        storage::Chunk {
            file_path: "src/UserForm.tsx".to_string(),
            chunk_type: storage::ChunkType::Component,
            name: Some("UserForm".to_string()),
            content: "function UserForm() { return <form/>; }".to_string(),
            start_line: 1,
            end_line: 20,
            parent_chunk_id: None,
        },
    );
    let text = format_results_grouped(&results, &ctx, &parent_chunks);

    assert!(
        !text.contains("Parent context:"),
        "[TC-003] when parent is in results, 'Parent context:' should be suppressed: {text}"
    );
    assert!(text.contains("UserForm"), "parent should appear: {text}");
    assert!(text.contains("handleSubmit"), "child should appear: {text}");
}

#[test]
fn tc_004_get_chunk_by_id_returns_parent_chunk_id() {
    let (conn, _dir) = test_db();

    let parent = storage::NewChunk {
        chunk_type: &storage::ChunkType::Component,
        name: Some("App"),
        content: "function App() {}",
        start_line: 1,
        end_line: 5,
        parent_index: None,
    };
    let child = storage::NewChunk {
        chunk_type: &storage::ChunkType::InnerFn,
        name: Some("handleClick"),
        content: "const handleClick = () => {}",
        start_line: 2,
        end_line: 4,
        parent_index: Some(0),
    };
    storage::replace_file_chunks_only(&conn, "src/App.tsx", &[parent, child], "h1", "", &[], None)
        .unwrap();

    let child_row: (i64, Option<i64>) = conn
        .query_row(
            "SELECT id, parent_chunk_id FROM chunks WHERE chunk_type = 'inner_fn' AND file_path = 'src/App.tsx'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();

    let chunk = storage::get_chunk_by_id(&conn, child_row.0)
        .unwrap()
        .expect("child chunk should exist");
    assert_eq!(chunk.chunk_type, storage::ChunkType::InnerFn);
    assert_eq!(chunk.name.as_deref(), Some("handleClick"));
    assert_eq!(chunk.parent_chunk_id, child_row.1);

    let missing = storage::get_chunk_by_id(&conn, 99999).unwrap();
    assert!(missing.is_none(), "nonexistent ID should return None");
}

#[test]
fn t001_search_with_not_installed_shows_note() {
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div/>; }",
    )]);
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), None);
    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let text = y.search("button", 10, 0, false).unwrap();
    assert!(
        text.contains("not installed"),
        "[T-001] expected NotInstalled note in results: {text}"
    );
}

#[test]
fn t002_search_with_ok_embedder_no_degraded_note() {
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div/>; }",
    )]);
    let y = Yomu::for_test(
        conn,
        dir.path().to_path_buf(),
        Some(Arc::new(rurico::embed::MockEmbedder::default()) as Arc<dyn rurico::embed::Embed>),
    );

    let text = y.search("button component", 10, 0, false).unwrap();
    assert!(
        !text.contains("not installed"),
        "[T-002] should have no 'not installed' note: {text}"
    );
    assert!(
        !text.contains("unavailable"),
        "[T-002] should have no 'unavailable' note: {text}"
    );
}

#[test]
fn t003_search_with_backend_unavailable_shows_note() {
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div/>; }",
    )]);
    let y = Yomu::for_test_raw(
        conn,
        dir.path().to_path_buf(),
        Err(DegradedReason::BackendUnavailable),
    );
    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let text = y.search("button", 10, 0, false).unwrap();
    assert!(
        text.contains("unavailable"),
        "[T-003] expected BackendUnavailable note in results: {text}"
    );
}

#[test]
fn t004_search_with_probe_failed_shows_note() {
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div/>; }",
    )]);
    let y = Yomu::for_test_raw(
        conn,
        dir.path().to_path_buf(),
        Err(DegradedReason::ProbeFailed),
    );
    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let text = y.search("button", 10, 0, false).unwrap();
    assert!(
        text.contains("unavailable"),
        "[T-004] expected ProbeFailed note in results: {text}"
    );
}

#[test]
fn t004_record_embedder_warning_observation_seam() {
    RECORDED_WARNINGS.with(|w| w.borrow_mut().clear());

    record_embedder_warning(DegradedReason::ProbeFailed, "model load failed");

    let warnings = get_recorded_warnings();
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].0, DegradedReason::ProbeFailed);
    assert_eq!(warnings[0].1, "model load failed");
}

#[test]
fn t009_disabled_no_user_note_in_search() {
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div/>; }",
    )]);
    let y = Yomu::for_test_raw(
        conn,
        dir.path().to_path_buf(),
        Err(DegradedReason::Disabled),
    );
    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let text = y.search("button", 10, 0, false).unwrap();
    assert!(
        !text.contains("not installed"),
        "[T-009] should not show 'not installed' when disabled: {text}"
    );
    assert!(
        !text.contains("unavailable"),
        "[T-009] should not show 'unavailable' when disabled: {text}"
    );
}

#[test]
fn user_note_exact_values_for_all_variants() {
    assert_eq!(DegradedReason::Disabled.user_note(), None);
    assert_eq!(
        DegradedReason::NotInstalled.user_note(),
        Some("embedding model not installed; results from text search only")
    );
    assert_eq!(
        DegradedReason::BackendUnavailable.user_note(),
        Some("embedding model unavailable; results from text search only")
    );
    assert_eq!(
        DegradedReason::ProbeFailed.user_note(),
        Some("embedding model unavailable; results from text search only")
    );
    assert_eq!(
        DegradedReason::DownloadFailed.user_note(),
        Some("embedding model download failed; results from text search only")
    );
}

#[test]
fn embed_disabled_yields_degraded_disabled() {
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div/>; }",
    )]);
    let y = Yomu::for_test_embed_disabled(conn, dir.path().to_path_buf());
    let _ = y.get_embedder();

    assert_eq!(y.degraded_reason(), Some(&DegradedReason::Disabled));
}

#[test]
fn json_notes_present_when_degraded() {
    let (conn, dir) =
        setup_test_files(&[("src/Card.tsx", "export function Card() { return <div/>; }")]);
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), None);
    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let json = y.search("card", 10, 0, true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["degraded"], true);
    let notes = parsed["notes"]
        .as_array()
        .expect("notes should be an array");
    assert!(
        !notes.is_empty(),
        "notes should contain degradation reason: {json}"
    );
    assert_eq!(
        notes[0], "embedding model not installed; results from text search only",
        "note should match NotInstalled variant: {json}"
    );
}

#[test]
fn json_notes_empty_via_search_with_ok_embedder() {
    let (conn, dir) =
        setup_test_files(&[("src/Nav.tsx", "export function Nav() { return <nav/>; }")]);
    let y = Yomu::for_test(
        conn,
        dir.path().to_path_buf(),
        Some(Arc::new(rurico::embed::MockEmbedder::default()) as Arc<dyn rurico::embed::Embed>),
    );

    let json = y.search("nav", 10, 0, true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["degraded"], false, "should not be degraded: {json}");
    let notes = parsed["notes"]
        .as_array()
        .expect("notes should be an array");
    assert!(
        notes.is_empty(),
        "notes should be empty with working embedder: {json}"
    );
}

#[test]
fn json_notes_outcome_degraded_fallback() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/Card.tsx", "export function Card() { return <div/>; }")],
        Arc::new(FailingEmbedder::all_fail("inference error")),
    );
    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let json = y.search("card", 10, 0, true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["degraded"], true);
    let notes = parsed["notes"]
        .as_array()
        .expect("notes should be an array");
    assert!(
        !notes.is_empty(),
        "notes should contain fallback reason: {json}"
    );
    assert_eq!(
        notes[0], "embedding model not loaded; results from text search only",
        "note should match outcome.degraded fallback: {json}"
    );
}

#[test]
fn json_notes_backend_unavailable() {
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div/>; }",
    )]);
    let y = Yomu::for_test_raw(
        conn,
        dir.path().to_path_buf(),
        Err(DegradedReason::BackendUnavailable),
    );
    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let json = y.search("button", 10, 0, true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["degraded"], true);
    let notes = parsed["notes"]
        .as_array()
        .expect("notes should be an array");
    assert_eq!(
        notes[0], "embedding model unavailable; results from text search only",
        "note should match BackendUnavailable variant: {json}"
    );
}

// --- JSON output tests for non-search commands ---

#[test]
fn index_json_returns_valid_json() {
    let (y, _dir) = test_yomu_with_files(&[
        ("src/A.tsx", "export function A() {}"),
        ("src/B.tsx", "export function B() {}"),
    ]);
    let json = y.index(true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["files_processed"], 2);
    assert!(parsed["chunks_created"].as_u64().unwrap() > 0);
    assert_eq!(parsed["files_skipped"], 0);
    assert_eq!(parsed["files_errored"], 0);
    assert!(
        parsed.get("coverage").is_some(),
        "should include coverage when not fully embedded: {json}"
    );
}

#[test]
fn rebuild_json_returns_valid_json() {
    let (y, _dir) = test_yomu_with_files(&[("src/A.tsx", "export function A() {}")]);
    y.index(false).unwrap();

    let json = y.rebuild(true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["files_processed"], 1);
    assert!(parsed["chunks_created"].as_u64().unwrap() > 0);
    assert_eq!(parsed["files_errored"], 0);
    assert!(
        parsed.get("files_skipped").is_none(),
        "rebuild JSON should not include files_skipped: {json}"
    );
}

#[test]
fn dry_run_index_json_returns_valid_json() {
    let (y, _dir) = test_yomu_with_files(&[
        ("src/A.tsx", "export function A() {}"),
        ("src/B.tsx", "export function B() {}"),
    ]);

    let json = y.dry_run_index(false, true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["files_to_process"], 2);
    assert_eq!(parsed["files_to_skip"], 0);
    assert_eq!(parsed["total_files"], 2);
    assert_eq!(parsed["files_errored"], 0);
    assert_eq!(parsed["orphans_to_remove"], 0);
}

#[test]
fn dry_run_index_json_shows_skip_for_unchanged() {
    let (y, _dir) = test_yomu_with_files(&[("src/A.tsx", "export function A() {}")]);
    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path()).unwrap();

    let json = y.dry_run_index(false, true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["files_to_process"], 0);
    assert_eq!(parsed["files_to_skip"], 1);
}

#[test]
fn status_json_empty_db() {
    let (y, _dir) = test_yomu();
    let json = y.status(true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["files"], 0);
    assert_eq!(parsed["chunks"], 0);
    assert_eq!(parsed["embedded_chunks"], 0);
    assert_eq!(parsed["embed_percentage"], 0);
    assert_eq!(parsed["references"], 0);
    assert!(parsed["last_indexed"].is_null(), "should be null: {json}");
}

#[test]
fn status_json_with_data() {
    let (conn, _dir) = test_db();
    let embedding = test_embedding();
    storage::insert_chunk(
        &conn,
        "src/A.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("A"),
            content: "function A() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
        },
        "h1",
        &ce(embedding.clone()),
        None,
    )
    .unwrap();

    let y = Yomu::for_test(conn, PathBuf::from("/tmp"), None);
    let json = y.status(true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["files"], 1);
    assert_eq!(parsed["chunks"], 1);
    assert_eq!(parsed["embedded_chunks"], 1);
    assert_eq!(parsed["embed_percentage"], 100);
}

#[test]
fn impact_json_with_dependents() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
        storage::replace_file_references(
            &conn,
            "src/A.tsx",
            &[storage::Reference {
                source_file: "src/A.tsx".into(),
                target_file: "src/hooks/useAuth.ts".into(),
                symbol_name: Some("useAuth".into()),
                ref_kind: storage::RefKind::Named,
            }],
        )
        .unwrap();
    }

    let json = y.impact("src/hooks/useAuth.ts", None, 3, true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["target"], "src/hooks/useAuth.ts");
    assert_eq!(
        parsed["in_index"], false,
        "useAuth.ts has no chunks in index"
    );
    assert_eq!(parsed["total"], 1);
    let deps = parsed["dependents"]
        .as_array()
        .expect("should have dependents");
    assert_eq!(deps[0]["file_path"], "src/A.tsx");
    assert_eq!(deps[0]["depth"], 1);
}

#[test]
fn impact_json_not_in_index() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
    }

    let json = y.impact("src/nonexistent.tsx", None, 3, true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["in_index"], false);
    assert_eq!(parsed["total"], 0);
    let deps = parsed["dependents"]
        .as_array()
        .expect("should have dependents");
    assert!(deps.is_empty());
}

#[test]
fn impact_json_with_symbol_refs() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
        storage::replace_file_references(
            &conn,
            "src/A.tsx",
            &[storage::Reference {
                source_file: "src/A.tsx".into(),
                target_file: "src/hooks/useAuth.ts".into(),
                symbol_name: Some("useAuth".into()),
                ref_kind: storage::RefKind::Named,
            }],
        )
        .unwrap();
        storage::replace_file_references(
            &conn,
            "src/C.tsx",
            &[storage::Reference {
                source_file: "src/C.tsx".into(),
                target_file: "src/hooks/useAuth.ts".into(),
                symbol_name: Some("useAuth".into()),
                ref_kind: storage::RefKind::Named,
            }],
        )
        .unwrap();
    }

    let json = y
        .impact("src/hooks/useAuth.ts:useAuth", None, 3, true)
        .unwrap();
    let parsed = parse_json(&json);
    let refs = parsed["symbol_refs"]
        .as_array()
        .expect("should have symbol_refs");
    assert!(
        refs.iter().any(|r| r == "src/A.tsx"),
        "should reference A.tsx: {json}"
    );
}
