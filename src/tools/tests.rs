use super::*;
use std::collections::HashMap;

use crate::indexer::embedder::FailingEmbedder;

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
    embedder: Arc<dyn crate::indexer::embedder::Embed>,
) -> (Yomu, tempfile::TempDir) {
    let (conn, dir) = setup_test_files(files);
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), Some(embedder));
    (y, dir)
}

fn seed_index(conn: &storage::Db) {
    let mut embedding = vec![0.0_f32; storage::EMBEDDING_DIMS as usize];
    embedding[0] = 1.0;
    storage::insert_chunk(
        conn,
        "src/dummy.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Other,
            name: None,
            content: "seed",
            start_line: 1,
            end_line: 1,
        },
        "seed",
        &embedding,
    )
    .unwrap();
}

#[tokio::test]
async fn search_rejects_empty_query() {
    let (y, _dir) = test_yomu();
    let err = y.search("", 10, 0).await.unwrap_err();
    assert!(
        err.to_string().contains("empty"),
        "expected empty error, got: {}",
        err
    );
}

#[tokio::test]
async fn search_rejects_long_query() {
    let (y, _dir) = test_yomu();
    let long_query = "a".repeat(MAX_QUERY_LENGTH + 1);
    let err = y.search(&long_query, 10, 0).await.unwrap_err();
    assert!(
        err.to_string().contains("maximum length"),
        "expected max length error, got: {}",
        err
    );
}

#[tokio::test]
async fn search_without_embedder_degrades_gracefully() {
    let (y, _dir) = test_yomu();
    let text = y.search("test query", 10, 0).await.unwrap();
    assert!(
        text.contains("No results found"),
        "expected no results: {text}"
    );
}

#[tokio::test]
async fn search_auto_indexes_empty_db() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/Button.tsx", "function Button() { return <div/>; }")],
        Arc::new(crate::indexer::embedder::MockEmbedder),
    );

    let text = y.search("button component", 10, 0).await.unwrap();
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

#[tokio::test]
async fn search_hybrid_flow_empty_db() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[(
            "src/Button.tsx",
            "export function Button() { return <div/>; }",
        )],
        Arc::new(crate::indexer::embedder::MockEmbedder),
    );

    let text = y.search("button component", 10, 0).await.unwrap();
    assert!(
        text.contains("Button"),
        "expected Button in results: {text}"
    );

    let stats = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap()
    };
    assert!(stats.total_chunks > 0, "expected chunks after hybrid index");
    assert!(
        stats.embedded_chunks > 0,
        "expected embeddings after hybrid index"
    );
}

#[tokio::test]
async fn search_incremental_embeds_chunked_only() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/Form.tsx", "export function Form() { return <form/>; }")],
        Arc::new(crate::indexer::embedder::MockEmbedder),
    );

    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path())
        .await
        .unwrap();

    {
        let c = y.conn.lock().unwrap();
        let stats = storage::get_stats(&c).unwrap();
        assert!(stats.total_chunks > 0, "should have chunks");
        assert_eq!(stats.embedded_chunks, 0, "should have no embeddings yet");
    }

    let text = y.search("form component", 10, 0).await.unwrap();
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

#[tokio::test]
async fn search_shows_coverage_on_no_results() {
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
        }],
        "hash1",
        "",
        &[],
    )
    .unwrap();

    let stats = storage::get_stats(&conn).unwrap();
    assert!(stats.total_chunks > 0, "should have chunks");
    assert_eq!(stats.embedded_chunks, 0, "should have no embeddings");

    let msg = format_no_results_message(&stats);
    assert!(msg.contains("coverage"), "expected coverage info: {msg}");
    assert!(msg.contains("0/"), "expected 0 embedded: {msg}");
}

#[tokio::test]
async fn search_degraded_empty_results_shows_note() {
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
        }],
        "h1",
        "",
        &[],
    )
    .unwrap();

    let y = Yomu::for_test(
        conn,
        dir.path().to_path_buf(),
        Some(Arc::new(FailingEmbedder::all_fail(
            500,
            "service unavailable",
        ))),
    );

    let text = y.search("zzzznonexistent", 10, 0).await.unwrap();
    assert!(
        text.contains("No results found"),
        "expected no results: {text}"
    );
    assert!(
        text.contains("embedding"),
        "expected embedding note in empty results: {text}"
    );
}

#[tokio::test]
async fn search_degraded_with_results_shows_note() {
    let (y, dir) = test_yomu_with_files_and_embedder(
        &[(
            "src/Button.tsx",
            "export function Button() { return <div/>; }",
        )],
        Arc::new(crate::indexer::embedder::MockEmbedder),
    );

    indexer::run_index(
        Arc::clone(&y.conn),
        y.root.as_path(),
        &crate::indexer::embedder::MockEmbedder,
        false,
    )
    .await
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn2 = storage::open_db(&db_path).unwrap();
    let y_failing = Yomu::for_test(
        conn2,
        dir.path().to_path_buf(),
        Some(Arc::new(FailingEmbedder::all_fail(
            500,
            "service unavailable",
        ))),
    );

    let result = y_failing.search("Button", 10, 0).await.unwrap();
    assert!(result.contains("Button"), "should have search results");
    assert!(
        result.contains("embedding API unavailable"),
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
    let text = format_results_grouped(&results, &ctx);
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
    let text = format_results_grouped(&results, &ctx);
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
    let text = format_results_grouped(&results, &ctx);
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
    let text = format_results_grouped(&results, &ctx);
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
    let text = format_results_grouped(&results, &ctx);
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
    let text = format_results_grouped(&results, &ctx);
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
            },
            chunk_id: None,
            distance: f32::INFINITY,
            match_source: storage::MatchSource::NameMatch,
            score: 0.55,
        },
    ];
    let ctx = EnrichmentContext {
        imports: HashMap::new(),
        siblings: HashMap::new(),
    };
    let text = format_results_grouped(&results, &ctx);
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
fn format_results_grouped_sorts_by_score() {
    let results = vec![
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/B.tsx".to_string(),
                chunk_type: storage::ChunkType::Component,
                name: Some("B".to_string()),
                content: "code B".to_string(),
                start_line: 1,
                end_line: 3,
            },
            chunk_id: None,
            distance: 0.5,
            match_source: storage::MatchSource::Semantic,
            score: 0.60,
        },
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/A.tsx".to_string(),
                chunk_type: storage::ChunkType::Component,
                name: Some("A".to_string()),
                content: "code A".to_string(),
                start_line: 1,
                end_line: 3,
            },
            chunk_id: None,
            distance: 0.1,
            match_source: storage::MatchSource::Semantic,
            score: 0.95,
        },
    ];
    let ctx = EnrichmentContext {
        imports: HashMap::new(),
        siblings: HashMap::new(),
    };
    let text = format_results_grouped(&results, &ctx);
    let a_pos = text.find("## src/A.tsx").unwrap();
    let b_pos = text.find("## src/B.tsx").unwrap();
    assert!(
        a_pos < b_pos,
        "A (score 0.95) should come before B (score 0.60): {text}"
    );
}

#[tokio::test]
async fn index_works_without_api_key() {
    let (y, _dir) = test_yomu_with_files(&[("src/A.tsx", "function A() {}")]);
    let text = y.index().await.unwrap();
    assert!(text.contains("complete"), "expected success: {text}");
}

#[tokio::test]
async fn index_chunks_without_embedding() {
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

    let text = y.index().await.unwrap();
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

#[tokio::test]
async fn rebuild_re_parses_all_files() {
    let (y, dir) = test_yomu_with_files(&[("src/A.tsx", "export function A() { return <div/>; }")]);
    y.index().await.unwrap();

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

    let text = y.rebuild().await.unwrap();
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

#[tokio::test]
async fn status_returns_empty_stats() {
    let (y, _dir) = test_yomu();
    let text = y.status().await.unwrap();
    assert!(text.contains("Files: 0"), "expected 0 files, got: {text}");
    assert!(text.contains("Chunks: 0"), "expected 0 chunks, got: {text}");
    assert!(
        text.contains("References: 0"),
        "expected 0 references, got: {text}"
    );
    assert!(text.contains("never"), "expected 'never', got: {text}");
}

#[tokio::test]
async fn status_returns_counts_after_insert() {
    let (conn, _dir) = test_db();
    let mut embedding = vec![0.0_f32; storage::EMBEDDING_DIMS as usize];
    embedding[0] = 1.0;
    storage::insert_chunk(
        &conn,
        "src/A.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("A"),
            content: "code",
            start_line: 1,
            end_line: 5,
        },
        "h1",
        &embedding,
    )
    .unwrap();

    let y = Yomu::for_test(conn, PathBuf::from("/tmp"), None);
    let text = y.status().await.unwrap();
    assert!(text.contains("Files: 1"), "expected 1 file, got: {text}");
    assert!(text.contains("Chunks: 1"), "expected 1 chunk, got: {text}");
}

#[tokio::test]
async fn status_shows_embedded_total() {
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
        }],
        "hash1",
        "",
        &[],
    )
    .unwrap();

    let y = Yomu::for_test(conn, PathBuf::from("/tmp"), None);
    let text = y.status().await.unwrap();
    assert!(text.contains("0/1"), "expected 0/1 in status: {text}");
}

#[tokio::test]
async fn impact_lists_dependents() {
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

    let text = y.impact("src/hooks/useAuth.ts", None, 3).await.unwrap();
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

#[tokio::test]
async fn impact_filters_by_symbol() {
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
        .impact("src/hooks/useAuth.ts:useAuth", None, 3)
        .await
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

#[tokio::test]
async fn impact_rejects_empty_target() {
    let (y, _dir) = test_yomu();
    let err = y.impact("", None, 3).await.unwrap_err();
    assert!(
        err.to_string().contains("empty"),
        "expected empty error, got: {err}"
    );
}

#[tokio::test]
async fn impact_errors_on_empty_index() {
    let (y, _dir) = test_yomu();
    let err = y.impact("src/A.tsx", None, 3).await.unwrap_err();
    assert!(
        err.to_string().contains("index is empty"),
        "expected empty index error, got: {err}"
    );
}

#[tokio::test]
async fn impact_distinguishes_missing_file() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
    }
    let text = y.impact("src/nonexistent.tsx", None, 3).await.unwrap();
    assert!(
        text.contains("not found in index"),
        "expected file-not-found message: {text}"
    );
}

#[tokio::test]
async fn impact_rejects_path_traversal() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
    }
    let err = y.impact("../etc/passwd", None, 3).await.unwrap_err();
    assert!(
        err.to_string().contains(".."),
        "expected path traversal error, got: {err}"
    );
}

#[tokio::test]
async fn impact_symbol_flag_overrides_colon_syntax() {
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
        .impact("src/hooks/useAuth.ts:useAuth", Some("AuthProvider"), 3)
        .await
        .unwrap();
    assert!(
        text.contains("src/B.tsx"),
        "expected B.tsx for AuthProvider: {text}"
    );
}

#[tokio::test]
async fn integration_index_then_impact() {
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
        &crate::indexer::embedder::MockEmbedder,
        false,
    )
    .await
    .unwrap();

    let text = y.impact("src/C.tsx", None, 3).await.unwrap();
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
        embedded_chunks: 0,
        last_indexed_at: None,
    };
    assert!(matches!(determine_index_state(&empty), IndexState::Empty));

    let chunked = storage::IndexStatus {
        total_files: 5,
        total_chunks: 20,
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
        embedded_chunks: 20,
        last_indexed_at: Some("2026-01-01".into()),
    };
    assert!(matches!(
        determine_index_state(&full),
        IndexState::FullyEmbedded
    ));
}

#[tokio::test]
async fn ensure_indexed_partially_embedded_triggers_embed() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/App.tsx", "export function App() { return <div/>; }")],
        Arc::new(crate::indexer::embedder::MockEmbedder),
    );

    indexer::run_chunk_only_index(Arc::clone(&y.conn), y.root.as_path())
        .await
        .unwrap();

    let stats_before = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap()
    };
    assert!(stats_before.total_chunks > 0, "should have chunks");
    assert_eq!(
        stats_before.embedded_chunks, 0,
        "should have no embeddings yet"
    );

    let result = y.search("App component", 10, 0).await.unwrap();
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

#[tokio::test]
async fn ensure_indexed_fully_embedded_skips_embed() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[(
            "src/Button.tsx",
            "export function Button() { return <button/>; }",
        )],
        Arc::new(crate::indexer::embedder::MockEmbedder),
    );

    indexer::run_index(
        Arc::clone(&y.conn),
        y.root.as_path(),
        &crate::indexer::embedder::MockEmbedder,
        false,
    )
    .await
    .unwrap();

    let stats = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap()
    };
    assert_eq!(
        stats.embedded_chunks, stats.total_chunks,
        "should be fully embedded"
    );

    let result = y.search("Button", 10, 0).await.unwrap();
    assert!(result.contains("Button"), "should find Button: {result}");
}

#[tokio::test]
async fn ensure_indexed_fully_embedded_with_failing_embedder() {
    let (y, dir) = test_yomu_with_files_and_embedder(
        &[("src/Card.tsx", "export function Card() { return <div/>; }")],
        Arc::new(crate::indexer::embedder::MockEmbedder),
    );

    indexer::run_index(
        Arc::clone(&y.conn),
        y.root.as_path(),
        &crate::indexer::embedder::MockEmbedder,
        false,
    )
    .await
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn2 = storage::open_db(&db_path).unwrap();
    let y_failing = Yomu::for_test(
        conn2,
        dir.path().to_path_buf(),
        Some(Arc::new(FailingEmbedder::all_fail(
            500,
            "service unavailable",
        ))),
    );

    let result = y_failing.search("Card", 10, 0).await.unwrap();
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
