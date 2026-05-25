use super::embedder::{
    DegradedReason, RECORDED_WARNINGS, get_recorded_warnings, parse_budget_value,
    record_embedder_warning,
};
use super::*;
use std::collections::HashMap;
use std::fs;

use rurico::embed::{Embed, FailingEmbedder, MockEmbedder};
use tempfile::{TempDir, tempdir};
use tracing_test::traced_test;

fn parse_json(json: &str) -> serde_json::Value {
    serde_json::from_str(json).unwrap_or_else(|e| panic!("invalid JSON: {e}\n{json}"))
}

fn test_embedding() -> Vec<f32> {
    let mut emb = vec![0.0_f32; storage::EMBEDDING_DIMS];
    emb[0] = 1.0;
    emb
}

fn test_db() -> (storage::Db, TempDir) {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();
    (conn, dir)
}

fn test_yomu() -> (Yomu, TempDir) {
    let (conn, dir) = test_db();
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), None);
    (y, dir)
}

fn setup_test_files(files: &[(&str, &str)]) -> (storage::Db, TempDir) {
    let dir = tempdir().unwrap();
    for (path, content) in files {
        let full_path = dir.path().join(path);
        if let Some(parent) = full_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(&full_path, content).unwrap();
    }
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    (conn, dir)
}

fn test_yomu_with_files(files: &[(&str, &str)]) -> (Yomu, TempDir) {
    let (conn, dir) = setup_test_files(files);
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), None);
    (y, dir)
}

fn test_yomu_with_files_and_embedder(
    files: &[(&str, &str)],
    embedder: Arc<dyn Embed>,
) -> (Yomu, TempDir) {
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
            source_kind: None,
            injection_flags: None,
        },
        "seed",
        &storage::ce(embedding.clone()),
        None,
    )
    .unwrap();
}

// T-175: search_rejects_empty_query
#[test]
fn search_rejects_empty_query() {
    let (y, _dir) = test_yomu();
    let err = y.search(Some(""), 10, 0, &[], false, None).unwrap_err();
    assert!(
        err.to_string().contains("empty"),
        "expected empty error, got: {}",
        err
    );
}

// T-176: search_rejects_long_query
#[test]
fn search_rejects_long_query() {
    let (y, _dir) = test_yomu();
    let long_query = "a".repeat(MAX_QUERY_LENGTH + 1);
    let err = y
        .search(Some(&long_query), 10, 0, &[], false, None)
        .unwrap_err();
    assert!(
        err.to_string().contains("max length"),
        "expected max length error, got: {}",
        err
    );
}

// T-177: search_rejects_path_traversal
#[test]
fn search_rejects_path_traversal() {
    let (y, _dir) = test_yomu();
    let err = y
        .search(Some("query"), 10, 0, &["../etc".to_owned()], false, None)
        .unwrap_err();
    assert!(
        err.to_string().contains("must be a relative path"),
        "expected traversal error, got: {err}"
    );
}

// T-178: search_rejects_absolute_path
#[test]
fn search_rejects_absolute_path() {
    let (y, _dir) = test_yomu();
    let err = y
        .search(Some("query"), 10, 0, &["/etc".to_owned()], false, None)
        .unwrap_err();
    assert!(
        err.to_string().contains("must be a relative path"),
        "expected relative path error, got: {err}"
    );
}

// T-179: search_path_filter_excludes_other_dirs
#[test]
fn search_path_filter_excludes_other_dirs() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[
            ("src/fetcher/index.ts", "export function fetchData() {}"),
            ("src/storage/db.ts", "export function openDb() {}"),
        ],
        Arc::new(MockEmbedder::default()),
    );

    let text = y
        .search(
            Some("open"),
            10,
            0,
            &["src/storage/".to_owned()],
            false,
            None,
        )
        .unwrap();
    assert!(
        text.contains("db.ts") || text.contains("openDb") || text.contains("No results"),
        "unexpected result: {text}"
    );
    assert!(
        !text.contains("fetchData"),
        "fetcher should be excluded: {text}"
    );
}

// T-180: search_without_embedder_degrades_gracefully
#[test]
fn search_without_embedder_degrades_gracefully() {
    let (y, _dir) = test_yomu();
    let text = y
        .search(Some("test query"), 10, 0, &[], false, None)
        .unwrap();
    assert!(
        text.contains("No results found"),
        "expected no results: {text}"
    );
}

// T-181: search_auto_indexes_empty_db
#[test]
fn search_auto_indexes_empty_db() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/Button.tsx", "function Button() { return <div/>; }")],
        Arc::new(MockEmbedder::default()),
    );

    let text = y
        .search(Some("button component"), 10, 0, &[], false, None)
        .unwrap();
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

// T-182: search_incremental_embeds_chunked_only
#[test]
fn search_incremental_embeds_chunked_only() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/Form.tsx", "export function Form() { return <form/>; }")],
        Arc::new(MockEmbedder::default()),
    );

    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    {
        let c = y.conn.lock().unwrap();
        let stats = storage::get_stats(&c).unwrap();
        assert!(stats.total_chunks > 0, "should have chunks");
        assert_eq!(stats.embedded_chunks, 0, "should have no embeddings yet");
    }

    let text = y
        .search(Some("form component"), 10, 0, &[], false, None)
        .unwrap();
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

// T-183: search_shows_coverage_on_no_results
#[test]
fn search_shows_coverage_on_no_results() {
    let dir = tempdir().unwrap();
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
            source_kind: None,
            injection_flags: None,
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

// T-184: search_degraded_empty_results_shows_note
#[test]
fn search_degraded_empty_results_shows_note() {
    let dir = tempdir().unwrap();
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
            source_kind: None,
            injection_flags: None,
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
        Some(Arc::new(FailingEmbedder::all_fail("service unavailable")) as Arc<dyn Embed>),
    );

    let text = y
        .search(Some("zzzznonexistent"), 10, 0, &[], false, None)
        .unwrap();
    assert!(
        text.contains("No results found"),
        "expected no results: {text}"
    );
    assert!(
        text.contains("embedding"),
        "expected embedding note in empty results: {text}"
    );
}

// T-185: search_degraded_with_results_shows_note
#[test]
fn search_degraded_with_results_shows_note() {
    let (y, dir) = test_yomu_with_files_and_embedder(
        &[(
            "src/Button.tsx",
            "export function Button() { return <div/>; }",
        )],
        Arc::new(MockEmbedder::default()),
    );

    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();
    indexer::run_incremental_embed(&y.conn, &MockEmbedder::default(), 3, None).unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn2 = storage::open_db(&db_path).unwrap();
    let y_failing = Yomu::for_test(
        conn2,
        dir.path().to_path_buf(),
        Some(Arc::new(FailingEmbedder::all_fail("service unavailable")) as Arc<dyn Embed>),
    );

    let result = y_failing
        .search(Some("Button"), 10, 0, &[], false, None)
        .unwrap();
    assert!(result.contains("Button"), "should have search results");
    assert!(
        result.contains("embedding model not loaded"),
        "should show degraded note"
    );
}

// T-186: format_results_grouped_renders_file_header_and_context
#[test]
fn format_results_grouped_renders_file_header_and_context() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/Button.tsx".to_owned(),
            chunk_type: storage::ChunkType::Component,
            name: Some("Button".to_owned()),
            content: "function Button() { return <div/>; }".to_owned(),
            start_line: 5,
            end_line: 7,
            parent_chunk_id: None,
            source_kind: None,
            injection_flags: None,
        },
        chunk_id: None,
        distance: 0.15,
        match_source: storage::MatchSource::Semantic,
        score: 1.0 / (1.0 + 0.15),
    }];
    let imports_map = HashMap::from([(
        "src/Button.tsx".to_owned(),
        "import React from 'react'".to_owned(),
    )]);
    let siblings_map = HashMap::from([(
        "src/Button.tsx".to_owned(),
        vec![storage::SiblingInfo {
            name: Some("ButtonProps".to_owned()),
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

// T-187: format_results_grouped_groups_same_file_chunks
#[test]
fn format_results_grouped_groups_same_file_chunks() {
    let results = vec![
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/Form.tsx".to_owned(),
                chunk_type: storage::ChunkType::Component,
                name: Some("Form".to_owned()),
                content: "function Form() {}".to_owned(),
                start_line: 1,
                end_line: 5,
                parent_chunk_id: None,
                source_kind: None,
                injection_flags: None,
            },
            chunk_id: None,
            distance: 0.1,
            match_source: storage::MatchSource::Semantic,
            score: 1.0 / (1.0 + 0.1),
        },
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/Form.tsx".to_owned(),
                chunk_type: storage::ChunkType::Hook,
                name: Some("useForm".to_owned()),
                content: "function useForm() {}".to_owned(),
                start_line: 7,
                end_line: 10,
                parent_chunk_id: None,
                source_kind: None,
                injection_flags: None,
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

// T-188: format_results_grouped_deduplicates_siblings
#[test]
fn format_results_grouped_deduplicates_siblings() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/A.tsx".to_owned(),
            chunk_type: storage::ChunkType::Component,
            name: Some("A".to_owned()),
            content: "function A() {}".to_owned(),
            start_line: 5,
            end_line: 7,
            parent_chunk_id: None,
            source_kind: None,
            injection_flags: None,
        },
        chunk_id: None,
        distance: 0.1,
        match_source: storage::MatchSource::Semantic,
        score: 1.0 / (1.0 + 0.1),
    }];
    let siblings_map = HashMap::from([(
        "src/A.tsx".to_owned(),
        vec![
            storage::SiblingInfo {
                name: Some("A".to_owned()),
                chunk_type: storage::ChunkType::Component,
                start_line: 5,
                end_line: 7,
            },
            storage::SiblingInfo {
                name: Some("AProps".to_owned()),
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

// T-189: format_results_grouped_omits_empty_imports
#[test]
fn format_results_grouped_omits_empty_imports() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/A.tsx".to_owned(),
            chunk_type: storage::ChunkType::Component,
            name: Some("A".to_owned()),
            content: "code".to_owned(),
            start_line: 1,
            end_line: 3,
            parent_chunk_id: None,
            source_kind: None,
            injection_flags: None,
        },
        chunk_id: None,
        distance: 0.1,
        match_source: storage::MatchSource::Semantic,
        score: 1.0 / (1.0 + 0.1),
    }];
    let imports_map = HashMap::from([("src/A.tsx".to_owned(), String::new())]);
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

// T-190: format_results_grouped_omits_empty_siblings
#[test]
fn format_results_grouped_omits_empty_siblings() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/A.tsx".to_owned(),
            chunk_type: storage::ChunkType::Component,
            name: Some("A".to_owned()),
            content: "code".to_owned(),
            start_line: 1,
            end_line: 3,
            parent_chunk_id: None,
            source_kind: None,
            injection_flags: None,
        },
        chunk_id: None,
        distance: 0.1,
        match_source: storage::MatchSource::Semantic,
        score: 1.0 / (1.0 + 0.1),
    }];
    let siblings_map = HashMap::from([("src/A.tsx".to_owned(), vec![])]);
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

// T-191: format_results_grouped_sorts_files_by_best_similarity
#[test]
fn format_results_grouped_sorts_files_by_best_similarity() {
    let results = vec![
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/B.tsx".to_owned(),
                chunk_type: storage::ChunkType::Component,
                name: Some("B".to_owned()),
                content: "code B".to_owned(),
                start_line: 1,
                end_line: 3,
                parent_chunk_id: None,
                source_kind: None,
                injection_flags: None,
            },
            chunk_id: None,
            distance: 0.5,
            match_source: storage::MatchSource::Semantic,
            score: 1.0 / (1.0 + 0.5),
        },
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/A.tsx".to_owned(),
                chunk_type: storage::ChunkType::Component,
                name: Some("A".to_owned()),
                content: "code A".to_owned(),
                start_line: 1,
                end_line: 3,
                parent_chunk_id: None,
                source_kind: None,
                injection_flags: None,
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

// T-192: format_results_grouped_shows_score_for_all
#[test]
fn format_results_grouped_shows_score_for_all() {
    let results = vec![
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/A.tsx".to_owned(),
                chunk_type: storage::ChunkType::Component,
                name: Some("A".to_owned()),
                content: "function A() {}".to_owned(),
                start_line: 1,
                end_line: 3,
                parent_chunk_id: None,
                source_kind: None,
                injection_flags: None,
            },
            chunk_id: None,
            distance: 0.5,
            match_source: storage::MatchSource::Semantic,
            score: 0.72,
        },
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/B.tsx".to_owned(),
                chunk_type: storage::ChunkType::Hook,
                name: Some("useAuth".to_owned()),
                content: "function useAuth() {}".to_owned(),
                start_line: 1,
                end_line: 3,
                parent_chunk_id: None,
                source_kind: None,
                injection_flags: None,
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

// T-193: index_works_without_api_key
#[test]
fn index_works_without_api_key() {
    let (y, _dir) = test_yomu_with_files(&[("src/A.tsx", "function A() {}")]);
    let text = y.index(IndexRunOptions::default(), false).unwrap();
    assert!(text.contains("complete"), "expected success: {text}");
}

// T-194: index_chunks_without_embedding
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

    let text = y.index(IndexRunOptions::default(), false).unwrap();
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

// T-195: rebuild_re_parses_all_files
#[test]
fn rebuild_re_parses_all_files() {
    let (y, dir) = test_yomu_with_files(&[("src/A.tsx", "export function A() { return <div/>; }")]);
    y.index(IndexRunOptions::default(), false).unwrap();

    let chunks_before = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap().total_chunks
    };
    assert!(chunks_before > 0, "should have chunks after index");

    fs::write(
        dir.path().join("src/B.tsx"),
        "export function B() { return <span/>; }",
    )
    .unwrap();

    let text = y.rebuild(IndexRunOptions::default(), false).unwrap();
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

// T-196: status_returns_empty_stats
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

// T-197: status_returns_counts_after_insert
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
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(embedding.clone()),
        None,
    )
    .unwrap();

    let y = Yomu::for_test(conn, PathBuf::from("/tmp"), None);
    let text = y.status(false).unwrap();
    assert!(text.contains("Files: 1"), "expected 1 file, got: {text}");
    assert!(text.contains("Chunks: 1"), "expected 1 chunk, got: {text}");
}

// T-198: status_shows_embedded_total
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
            source_kind: None,
            injection_flags: None,
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

// T-199: impact_lists_dependents
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

    let text = y
        .impact("src/hooks/useAuth.ts", None, 3, false, false)
        .unwrap();
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

// T-200: impact_filters_by_symbol
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
        .impact("src/hooks/useAuth.ts:useAuth", None, 3, false, false)
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

// T-201: impact_rejects_empty_target
#[test]
fn impact_rejects_empty_target() {
    let (y, _dir) = test_yomu();
    let err = y.impact("", None, 3, false, false).unwrap_err();
    assert!(
        err.to_string().contains("empty"),
        "expected empty error, got: {err}"
    );
}

// T-202: impact_errors_on_empty_index
#[test]
fn impact_errors_on_empty_index() {
    let (y, _dir) = test_yomu();
    let err = y.impact("src/A.tsx", None, 3, false, false).unwrap_err();
    assert!(
        err.to_string().contains("index is empty"),
        "expected empty index error, got: {err}"
    );
}

// T-203: impact_distinguishes_missing_file
#[test]
fn impact_distinguishes_missing_file() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
    }
    let text = y
        .impact("src/nonexistent.tsx", None, 3, false, false)
        .unwrap();
    assert!(
        text.contains("not found in index"),
        "expected file-not-found message: {text}"
    );
}

// T-204: impact_rejects_path_traversal
#[test]
fn impact_rejects_path_traversal() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
    }
    let err = y
        .impact("../etc/passwd", None, 3, false, false)
        .unwrap_err();
    assert!(
        err.to_string().contains(".."),
        "expected path traversal error, got: {err}"
    );
}

// T-205: impact_rejects_absolute_path
#[test]
fn impact_rejects_absolute_path() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
    }
    let err = y.impact("/etc/passwd", None, 3, false, false).unwrap_err();
    assert!(
        err.to_string().contains("relative"),
        "expected rejection of absolute path, got: {err}"
    );
}

// T-206: impact_symbol_flag_overrides_colon_syntax
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
            false,
        )
        .unwrap();
    assert!(
        text.contains("src/B.tsx"),
        "expected B.tsx for AuthProvider: {text}"
    );
}

// T-207: integration_index_then_impact
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

    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();
    indexer::run_incremental_embed(&y.conn, &MockEmbedder::default(), 6, None).unwrap();

    let text = y.impact("src/C.tsx", None, 3, false, false).unwrap();
    assert!(
        text.contains("src/B.tsx"),
        "expected B.tsx as direct dependent: {text}"
    );
    assert!(
        text.contains("src/A.tsx"),
        "expected A.tsx as transitive dependent: {text}"
    );
}

// T-208: parse_impact_target_file_only
#[test]
fn parse_impact_target_file_only() {
    let (file, symbol) = parse_impact_target("src/hooks/useAuth.ts");
    assert_eq!(file, "src/hooks/useAuth.ts");
    assert_eq!(symbol, None);
}

// T-209: parse_impact_target_with_symbol
#[test]
fn parse_impact_target_with_symbol() {
    let (file, symbol) = parse_impact_target("src/hooks/useAuth.ts:useAuth");
    assert_eq!(file, "src/hooks/useAuth.ts");
    assert_eq!(symbol, Some("useAuth"));
}

// T-210: parse_impact_target_trailing_colon
#[test]
fn parse_impact_target_trailing_colon() {
    let (file, symbol) = parse_impact_target("src/A.tsx:");
    assert_eq!(file, "src/A.tsx:");
    assert_eq!(symbol, None);
}

// T-211: parse_impact_target_leading_colon
#[test]
fn parse_impact_target_leading_colon() {
    let (file, symbol) = parse_impact_target(":symbol");
    assert_eq!(file, ":symbol");
    assert_eq!(symbol, None);
}

// T-212: parse_impact_target_symbol_with_slash_treated_as_file
#[test]
fn parse_impact_target_symbol_with_slash_treated_as_file() {
    let (file, symbol) = parse_impact_target("src/A.tsx:path/like");
    assert_eq!(file, "src/A.tsx:path/like");
    assert_eq!(symbol, None);
}

// T-213: parse_impact_target_multiple_colons
#[test]
fn parse_impact_target_multiple_colons() {
    let (file, symbol) = parse_impact_target("src/A.tsx:B:C");
    assert_eq!(file, "src/A.tsx:B");
    assert_eq!(symbol, Some("C"));
}

// T-214: parse_budget_value_valid
#[test]
fn parse_budget_value_valid() {
    assert_eq!(parse_budget_value(Some("100")), 100);
    assert_eq!(parse_budget_value(Some("1")), 1);
    assert_eq!(parse_budget_value(Some("500")), 500);
}

// T-215: parse_budget_value_out_of_range
#[test]
fn parse_budget_value_out_of_range() {
    assert_eq!(parse_budget_value(Some("0")), DEFAULT_EMBED_BUDGET);
    assert_eq!(parse_budget_value(Some("501")), DEFAULT_EMBED_BUDGET);
}

// T-216: parse_budget_value_invalid
#[test]
fn parse_budget_value_invalid() {
    assert_eq!(parse_budget_value(Some("abc")), DEFAULT_EMBED_BUDGET);
    assert_eq!(parse_budget_value(Some("")), DEFAULT_EMBED_BUDGET);
}

// T-217: parse_budget_value_missing
#[test]
fn parse_budget_value_missing() {
    assert_eq!(parse_budget_value(None), DEFAULT_EMBED_BUDGET);
}

// T-218: determine_index_state_variants
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

// T-219: ensure_indexed_partially_embedded_triggers_embed
#[test]
fn ensure_indexed_partially_embedded_triggers_embed() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/App.tsx", "export function App() { return <div/>; }")],
        Arc::new(MockEmbedder::default()),
    );

    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let stats_before = {
        let c = y.conn.lock().unwrap();
        storage::get_stats(&c).unwrap()
    };
    assert!(stats_before.total_chunks > 0, "should have chunks");
    assert_eq!(
        stats_before.embedded_chunks, 0,
        "should have no embeddings yet"
    );

    let result = y
        .search(Some("App component"), 10, 0, &[], false, None)
        .unwrap();
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

// T-222: with_root_creates_db_and_returns_yomu
#[test]
fn with_root_creates_db_and_returns_yomu() {
    let dir = tempdir().unwrap();
    let result = Yomu::with_root(
        dir.path().to_path_buf(),
        Default::default(),
        Default::default(),
    );
    assert!(
        result.is_ok(),
        "with_root should succeed: {:?}",
        result.err()
    );
    let yomu = result.unwrap();
    assert_eq!(yomu.root, dir.path());
    assert!(dir.path().join(".yomu").join("index.db").exists());
}

// T-223: search_without_embedder_skips_embed_attempt
#[test]
fn search_without_embedder_skips_embed_attempt() {
    let (y, _dir) =
        test_yomu_with_files(&[("src/Card.tsx", "export function Card() { return <div/>; }")]);

    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    {
        let c = y.conn.lock().unwrap();
        let stats = storage::get_stats(&c).unwrap();
        assert!(stats.total_chunks > 0, "should have chunks");
        assert_eq!(stats.embedded_chunks, 0, "should have no embeddings");
    }

    let text = y.search(Some("card"), 10, 0, &[], false, None).unwrap();
    assert!(
        !text.contains("embedding failed"),
        "should not attempt embed when embedder unavailable: {text}"
    );
}

// T-224: search_json_format_returns_valid_json
#[test]
fn search_json_format_returns_valid_json() {
    let (y, _dir) = test_yomu_with_files(&[(
        "src/Button.tsx",
        "export function Button() { return <button/>; }",
    )]);
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let json = y.search(Some("button"), 10, 0, &[], true, None).unwrap();
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

// T-225: search_json_format_empty_results
#[test]
fn search_json_format_empty_results() {
    let (y, _dir) =
        test_yomu_with_files(&[("src/A.tsx", "export function A() { return <div/>; }")]);
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let json = y
        .search(Some("zzzznonexistent"), 10, 0, &[], true, None)
        .unwrap();
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

// T-226: dry_run_index_does_not_write_to_db
#[test]
fn dry_run_index_does_not_write_to_db() {
    let (y, _dir) = test_yomu_with_files(&[
        ("src/A.tsx", "export function A() {}"),
        ("src/B.tsx", "export function B() {}"),
    ]);

    let text = y.dry_run_index(IndexRunOptions::default(), false).unwrap();
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

// T-227: dry_run_index_shows_skip_for_unchanged
#[test]
fn dry_run_index_shows_skip_for_unchanged() {
    let (y, _dir) = test_yomu_with_files(&[
        ("src/A.tsx", "export function A() {}"),
        ("src/B.tsx", "export function B() {}"),
    ]);

    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let text = y.dry_run_index(IndexRunOptions::default(), false).unwrap();
    assert!(
        text.contains("0 files to process"),
        "all files should be skipped: {text}"
    );
    assert!(
        text.contains("2 files unchanged"),
        "should show unchanged count: {text}"
    );
}

// T-228: search_json_format_degraded_includes_flag
#[test]
fn search_json_format_degraded_includes_flag() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/Card.tsx", "export function Card() { return <div/>; }")],
        Arc::new(FailingEmbedder::all_fail("service unavailable")),
    );
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let json = y.search(Some("card"), 10, 0, &[], true, None).unwrap();
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

// T-553: innerfn_hit_shows_parent_context_in_text_output
#[test]
fn innerfn_hit_shows_parent_context_in_text_output() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/UserForm.tsx".to_owned(),
            chunk_type: storage::ChunkType::InnerFn,
            name: Some("handleSubmit".to_owned()),
            content: "const handleSubmit = (e) => {\n  e.preventDefault();\n  submit(name);\n};"
                .to_owned(),
            start_line: 10,
            end_line: 13,
            parent_chunk_id: Some(42),
            source_kind: None,
            injection_flags: None,
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
            file_path: "src/UserForm.tsx".to_owned(),
            chunk_type: storage::ChunkType::Component,
            name: Some("UserForm".to_owned()),
            content: "function UserForm() {\n  return <form/>;\n}".to_owned(),
            start_line: 1,
            end_line: 20,
            parent_chunk_id: None,
            source_kind: None,
            injection_flags: None,
        },
    );
    let text = format_results_grouped(&results, &ctx, &parent_chunks);
    assert!(
        text.contains("Parent context:"),
        "InnerFn hit should display parent context section: {text}"
    );
}

// T-555: parent_hit_no_duplicate_parent_display
#[test]
fn parent_hit_no_duplicate_parent_display() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/UserForm.tsx".to_owned(),
            chunk_type: storage::ChunkType::Component,
            name: Some("UserForm".to_owned()),
            content: "function UserForm() {\n  return <form/>;\n}".to_owned(),
            start_line: 1,
            end_line: 20,
            parent_chunk_id: None,
            source_kind: None,
            injection_flags: None,
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
        "parent chunk hit should NOT show 'Parent context:' section: {text}"
    );
    assert!(
        text.contains("UserForm"),
        "parent chunk content should still appear normally: {text}"
    );
}

// T-229: json_output_includes_parent_chunk_id
#[test]
fn json_output_includes_parent_chunk_id() {
    let results = vec![
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/UserForm.tsx".to_owned(),
                chunk_type: storage::ChunkType::InnerFn,
                name: Some("handleSubmit".to_owned()),
                content: "const handleSubmit = () => {}".to_owned(),
                start_line: 10,
                end_line: 13,
                parent_chunk_id: Some(42),
                source_kind: None,
                injection_flags: None,
            },
            chunk_id: Some(43),
            distance: 0.2,
            match_source: storage::MatchSource::Fts,
            score: 0.80,
        },
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/App.tsx".to_owned(),
                chunk_type: storage::ChunkType::Component,
                name: Some("App".to_owned()),
                content: "function App() {}".to_owned(),
                start_line: 1,
                end_line: 5,
                parent_chunk_id: None,
                source_kind: None,
                injection_flags: None,
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
        "InnerFn result should have parent_chunk_id: {json}"
    );

    let component_item = &items[1];
    assert!(
        component_item.get("parent_chunk_id").is_some(),
        "Component result should have parent_chunk_id field: {json}"
    );
    assert!(
        component_item["parent_chunk_id"].is_null(),
        "Component result parent_chunk_id should be null: {json}"
    );
}

// T-230: subchunk_innerfn_is_hit_at_1_for_inner_function_query
#[test]
fn subchunk_innerfn_is_hit_at_1_for_inner_function_query() {
    let mut lines = vec![
        "export function UserForm() {".to_owned(),
        "  const [name, setName] = useState('');".to_owned(),
        "  const handleSubmit = () => {".to_owned(),
        "    submitFormData(name);".to_owned(),
        "    resetForm();".to_owned(),
        "  };".to_owned(),
        "  const handleCancel = () => {".to_owned(),
        "    resetForm();".to_owned(),
        "  };".to_owned(),
    ];
    for i in 0..48 {
        lines.push(format!("  const pad{i} = {i};"));
    }
    lines.push("  return <form onSubmit={handleSubmit}><input/></form>;".to_owned());
    lines.push("}".to_owned());
    let fixture = lines.join("\n");

    let (y, _dir) = test_yomu_with_files(&[("src/UserForm.tsx", &fixture)]);

    y.index(IndexRunOptions::default(), false).unwrap();

    let result = y
        .search(Some("handleSubmit"), 10, 0, &[], false, None)
        .unwrap();
    assert!(
        result.contains("handleSubmit"),
        "search should find handleSubmit: {result}"
    );
    assert!(
        result.contains("inner_fn"),
        "handleSubmit should be typed as inner_fn: {result}"
    );
}

// T-231: below_threshold_no_subchunks_in_index
#[test]
fn below_threshold_no_subchunks_in_index() {
    let fixture = r#"export function SmallCard() {
  const [count, setCount] = useState(0);
  const handleClick = () => { setCount(count + 1); };
  return <div><button onClick={handleClick}>{count}</button></div>;
}"#;
    let (y, _dir) = test_yomu_with_files(&[("src/SmallCard.tsx", fixture)]);
    y.index(IndexRunOptions::default(), false).unwrap();

    let stats = y.status(false).unwrap();
    assert!(
        !stats.contains("inner_fn"),
        "below-threshold component should not produce InnerFn: {stats}"
    );
}

// T-535: parent_and_child_both_in_results_no_duplicate
#[test]
fn parent_and_child_both_in_results_no_duplicate() {
    let parent_id = 42i64;
    let results = vec![
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/UserForm.tsx".to_owned(),
                chunk_type: storage::ChunkType::Component,
                name: Some("UserForm".to_owned()),
                content: "function UserForm() { return <form/>; }".to_owned(),
                start_line: 1,
                end_line: 20,
                parent_chunk_id: None,
                source_kind: None,
                injection_flags: None,
            },
            chunk_id: Some(parent_id),
            distance: 0.1,
            match_source: storage::MatchSource::Semantic,
            score: 0.90,
        },
        storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/UserForm.tsx".to_owned(),
                chunk_type: storage::ChunkType::InnerFn,
                name: Some("handleSubmit".to_owned()),
                content: "const handleSubmit = () => {}".to_owned(),
                start_line: 10,
                end_line: 13,
                parent_chunk_id: Some(parent_id),
                source_kind: None,
                injection_flags: None,
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
            file_path: "src/UserForm.tsx".to_owned(),
            chunk_type: storage::ChunkType::Component,
            name: Some("UserForm".to_owned()),
            content: "function UserForm() { return <form/>; }".to_owned(),
            start_line: 1,
            end_line: 20,
            parent_chunk_id: None,
            source_kind: None,
            injection_flags: None,
        },
    );
    let text = format_results_grouped(&results, &ctx, &parent_chunks);

    assert!(
        !text.contains("Parent context:"),
        "when parent is in results, 'Parent context:' should be suppressed: {text}"
    );
    assert!(text.contains("UserForm"), "parent should appear: {text}");
    assert!(text.contains("handleSubmit"), "child should appear: {text}");
}

// T-538: get_chunk_by_id_returns_parent_chunk_id
#[test]
fn get_chunk_by_id_returns_parent_chunk_id() {
    let (conn, _dir) = test_db();

    let parent = storage::NewChunk {
        chunk_type: &storage::ChunkType::Component,
        name: Some("App"),
        content: "function App() {}",
        start_line: 1,
        end_line: 5,
        parent_index: None,
        source_kind: None,
        injection_flags: None,
    };
    let child = storage::NewChunk {
        chunk_type: &storage::ChunkType::InnerFn,
        name: Some("handleClick"),
        content: "const handleClick = () => {}",
        start_line: 2,
        end_line: 4,
        parent_index: Some(0),
        source_kind: None,
        injection_flags: None,
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

// T-530: search_with_not_installed_shows_note
#[test]
fn search_with_not_installed_shows_note() {
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div/>; }",
    )]);
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), None);
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let text = y.search(Some("button"), 10, 0, &[], false, None).unwrap();
    assert!(
        text.contains("not installed"),
        "expected NotInstalled note in results: {text}"
    );
}

// T-533: search_with_ok_embedder_no_degraded_note
#[test]
fn search_with_ok_embedder_no_degraded_note() {
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div/>; }",
    )]);
    let y = Yomu::for_test(
        conn,
        dir.path().to_path_buf(),
        Some(Arc::new(MockEmbedder::default()) as Arc<dyn Embed>),
    );

    let text = y
        .search(Some("button component"), 10, 0, &[], false, None)
        .unwrap();
    assert!(
        !text.contains("not installed"),
        "should have no 'not installed' note: {text}"
    );
    assert!(
        !text.contains("unavailable"),
        "should have no 'unavailable' note: {text}"
    );
}

// T-536: search_with_backend_unavailable_shows_note
#[test]
fn search_with_backend_unavailable_shows_note() {
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div/>; }",
    )]);
    let y = Yomu::for_test_raw(
        conn,
        dir.path().to_path_buf(),
        Err(DegradedReason::BackendUnavailable),
    );
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let text = y.search(Some("button"), 10, 0, &[], false, None).unwrap();
    assert!(
        text.contains("unavailable"),
        "expected BackendUnavailable note in results: {text}"
    );
}

// T-539: search_with_probe_failed_shows_note
#[test]
fn search_with_probe_failed_shows_note() {
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div/>; }",
    )]);
    let y = Yomu::for_test_raw(
        conn,
        dir.path().to_path_buf(),
        Err(DegradedReason::ProbeFailed),
    );
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let text = y.search(Some("button"), 10, 0, &[], false, None).unwrap();
    assert!(
        text.contains("unavailable"),
        "expected ProbeFailed note in results: {text}"
    );
}

// T-540: record_embedder_warning_observation_seam
#[test]
fn record_embedder_warning_observation_seam() {
    RECORDED_WARNINGS.with(|w| w.borrow_mut().clear());

    record_embedder_warning(DegradedReason::ProbeFailed, "model load failed");

    let warnings = get_recorded_warnings();
    assert_eq!(warnings.len(), 1);
    assert_eq!(warnings[0].0, DegradedReason::ProbeFailed);
    assert_eq!(warnings[0].1, "model load failed");
}

// T-548: disabled_no_user_note_in_search
#[test]
fn disabled_no_user_note_in_search() {
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div/>; }",
    )]);
    let y = Yomu::for_test_raw(
        conn,
        dir.path().to_path_buf(),
        Err(DegradedReason::Disabled),
    );
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let text = y.search(Some("button"), 10, 0, &[], false, None).unwrap();
    assert!(
        !text.contains("not installed"),
        "should not show 'not installed' when disabled: {text}"
    );
    assert!(
        !text.contains("unavailable"),
        "should not show 'unavailable' when disabled: {text}"
    );
}

// T-116: degraded_reason() returns the amici-provided DegradedReason type
#[test]
fn degraded_reason_returns_amici_type() {
    let (conn, dir) = setup_test_files(&[]);
    let y = Yomu::for_test_raw(
        conn,
        dir.path().to_path_buf(),
        Err(DegradedReason::NotInstalled),
    );
    assert_eq!(y.degraded_reason(), Some(&DegradedReason::NotInstalled));
}

// T-233: embed_disabled_yields_degraded_disabled
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

// T-234: json_notes_present_when_degraded
#[test]
fn json_notes_present_when_degraded() {
    let (conn, dir) =
        setup_test_files(&[("src/Card.tsx", "export function Card() { return <div/>; }")]);
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), None);
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let json = y.search(Some("card"), 10, 0, &[], true, None).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["degraded"], true);
    let notes = parsed["notes"]
        .as_array()
        .expect("notes should be an array");
    assert!(
        !notes.is_empty(),
        "notes should contain degradation reason: {json}"
    );
    assert!(
        notes[0]
            .as_str()
            .is_some_and(|n| n.contains("yomu model download")),
        "note should include `yomu model download` hint: {json}"
    );
}

// T-109: un-embedded chunks are embedded, result reports count
#[test]
fn embed_pending_chunks_returns_count() {
    let embedder = Arc::new(MockEmbedder::default()) as Arc<dyn Embed>;
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div>button</div>; }",
    )]);
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), Some(embedder));
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let text = y.embed(false).unwrap();
    assert!(
        text.starts_with("Embedded"),
        "expected result starting with 'Embedded': {text}"
    );
    assert!(
        text.contains("chunks"),
        "expected 'chunks' in result: {text}"
    );
}

// T-110: no pending chunks → "nothing to embed"
#[test]
fn embed_nothing_to_embed() {
    let embedder = Arc::new(MockEmbedder::default()) as Arc<dyn Embed>;
    let (conn, dir) = setup_test_files(&[]);
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), Some(embedder));
    // No index run — no chunks to embed.

    let text = y.embed(false).unwrap();
    assert_eq!(
        text, "nothing to embed",
        "expected 'nothing to embed': {text}"
    );
}

// T-111: json=true produces {"embedded": N}
#[test]
fn embed_json_format() {
    let embedder = Arc::new(MockEmbedder::default()) as Arc<dyn Embed>;
    let (conn, dir) = setup_test_files(&[(
        "src/Button.tsx",
        "export function Button() { return <div>button</div>; }",
    )]);
    let y = Yomu::for_test(conn, dir.path().to_path_buf(), Some(embedder));
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let json = y.embed(true).unwrap();
    let parsed = parse_json(&json);
    assert!(
        parsed.get("embedded").is_some(),
        "expected 'embedded' key in JSON: {json}"
    );
}

// T-112: embedder not installed → YomuError::EmbedderUnavailable
#[test]
fn embed_embedder_unavailable_returns_error() {
    let (conn, dir) = setup_test_files(&[("src/lib.rs", "pub fn hello() {}")]);
    let y = Yomu::for_test_raw(
        conn,
        dir.path().to_path_buf(),
        Err(DegradedReason::NotInstalled),
    );
    y.index(IndexRunOptions::default(), false).unwrap();

    let result = y.embed(false);
    assert!(
        matches!(result, Err(YomuError::EmbedderUnavailable(_))),
        "expected EmbedderUnavailable error: {result:?}"
    );
}

// T-120: embed() with Disabled reason → YomuError::EmbedderUnavailable
#[test]
fn embed_disabled_returns_embedder_unavailable() {
    let (conn, dir) = setup_test_files(&[("src/lib.rs", "pub fn hello() {}")]);
    let y = Yomu::for_test_raw(
        conn,
        dir.path().to_path_buf(),
        Err(DegradedReason::Disabled),
    );
    y.index(IndexRunOptions::default(), false).unwrap();

    let result = y.embed(false);
    assert!(
        matches!(result, Err(YomuError::EmbedderUnavailable(_))),
        "expected EmbedderUnavailable for Disabled reason: {result:?}"
    );
}

// T-235: json_notes_empty_via_search_with_ok_embedder
#[test]
fn json_notes_empty_via_search_with_ok_embedder() {
    let (conn, dir) =
        setup_test_files(&[("src/Nav.tsx", "export function Nav() { return <nav/>; }")]);
    let y = Yomu::for_test(
        conn,
        dir.path().to_path_buf(),
        Some(Arc::new(MockEmbedder::default()) as Arc<dyn Embed>),
    );

    let json = y.search(Some("nav"), 10, 0, &[], true, None).unwrap();
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

// T-236: json_notes_outcome_degraded_fallback
#[test]
fn json_notes_outcome_degraded_fallback() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[("src/Card.tsx", "export function Card() { return <div/>; }")],
        Arc::new(FailingEmbedder::all_fail("inference error")),
    );
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let json = y.search(Some("card"), 10, 0, &[], true, None).unwrap();
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

// T-237: json_notes_backend_unavailable
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
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let json = y.search(Some("button"), 10, 0, &[], true, None).unwrap();
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

// T-238: index_json_returns_valid_json
#[test]
fn index_json_returns_valid_json() {
    let (y, _dir) = test_yomu_with_files(&[
        ("src/A.tsx", "export function A() {}"),
        ("src/B.tsx", "export function B() {}"),
    ]);
    let json = y.index(IndexRunOptions::default(), true).unwrap();
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

// T-239: rebuild_json_returns_valid_json
#[test]
fn rebuild_json_returns_valid_json() {
    let (y, _dir) = test_yomu_with_files(&[("src/A.tsx", "export function A() {}")]);
    y.index(IndexRunOptions::default(), false).unwrap();

    let json = y.rebuild(IndexRunOptions::default(), true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["files_processed"], 1);
    assert!(parsed["chunks_created"].as_u64().unwrap() > 0);
    assert_eq!(parsed["files_errored"], 0);
    assert!(
        parsed.get("files_skipped").is_none(),
        "rebuild JSON should not include files_skipped: {json}"
    );
}

// T-240: dry_run_index_json_returns_valid_json
#[test]
fn dry_run_index_json_returns_valid_json() {
    let (y, _dir) = test_yomu_with_files(&[
        ("src/A.tsx", "export function A() {}"),
        ("src/B.tsx", "export function B() {}"),
    ]);

    let json = y.dry_run_index(IndexRunOptions::default(), true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["files_to_process"], 2);
    assert_eq!(parsed["files_to_skip"], 0);
    assert_eq!(parsed["total_files"], 2);
    assert_eq!(parsed["files_errored"], 0);
    assert_eq!(parsed["orphans_to_remove"], 0);
}

// T-241: dry_run_index_json_shows_skip_for_unchanged
#[test]
fn dry_run_index_json_shows_skip_for_unchanged() {
    let (y, _dir) = test_yomu_with_files(&[("src/A.tsx", "export function A() {}")]);
    indexer::run_chunk_only_index(&y.conn, y.root.as_path(), false).unwrap();

    let json = y.dry_run_index(IndexRunOptions::default(), true).unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["files_to_process"], 0);
    assert_eq!(parsed["files_to_skip"], 1);
}

// T-242: status_json_empty_db
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

// T-243: status_json_with_data
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
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(embedding.clone()),
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

// T-244: impact_json_with_dependents
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

    let json = y
        .impact("src/hooks/useAuth.ts", None, 3, true, false)
        .unwrap();
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

// T-245: impact_json_not_in_index
#[test]
fn impact_json_not_in_index() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
    }

    let json = y
        .impact("src/nonexistent.tsx", None, 3, true, false)
        .unwrap();
    let parsed = parse_json(&json);
    assert_eq!(parsed["in_index"], false);
    assert_eq!(parsed["total"], 0);
    let deps = parsed["dependents"]
        .as_array()
        .expect("should have dependents");
    assert!(deps.is_empty());
}

// T-246: impact_json_with_symbol_refs
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
        .impact("src/hooks/useAuth.ts:useAuth", None, 3, true, false)
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

// T-565: --semantic with no stored embeddings returns empty semantic_related
#[test]
fn impact_semantic_no_embeddings_returns_empty() {
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

    // no stored embeddings for the target → semantic_related is empty, structural still works
    let text = y
        .impact("src/hooks/useAuth.ts", None, 3, false, true)
        .unwrap();
    assert!(
        text.contains("src/A.tsx"),
        "structural dependents should still appear: {text}"
    );
    assert!(
        !text.contains("Semantic related"),
        "no semantic section expected when embeddings are absent: {text}"
    );
}

// T-566: --semantic JSON output omits semantic_related when empty
#[test]
fn impact_semantic_json_field_absent_when_empty() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
    }

    let json = y
        .impact("src/hooks/useAuth.ts", None, 3, true, true)
        .unwrap();
    let parsed = parse_json(&json);
    assert!(
        parsed.get("semantic_related").is_none(),
        "semantic_related should be absent when empty: {json}"
    );
}

// T-567b: --semantic=false JSON output never has semantic_related
// T-247: impact_no_semantic_json_field_absent
#[test]
fn impact_no_semantic_json_field_absent() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_index(&conn);
    }

    let json = y
        .impact("src/hooks/useAuth.ts", None, 3, true, false)
        .unwrap();
    let parsed = parse_json(&json);
    assert!(
        parsed.get("semantic_related").is_none(),
        "semantic_related should be absent without --semantic: {json}"
    );
}

// T-013: --from target with no stored embeddings → Ok (exit code 0)
#[test]
fn search_from_no_embeddings_returns_ok() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        storage::replace_file_chunks_only(
            &conn,
            "src/foo.rs",
            &[storage::NewChunk {
                chunk_type: &storage::ChunkType::RustFn,
                name: Some("my_fn"),
                content: "fn my_fn() {}",
                start_line: 1,
                end_line: 1,
                parent_index: None,
                source_kind: None,
                injection_flags: None,
            }],
            "hash1",
            "",
            &[],
            None,
        )
        .unwrap();
    }
    let result = y.search(None, 10, 0, &[], false, Some("src/foo.rs"));
    assert!(
        result.is_ok(),
        "expected Ok for no-embeddings case, got: {:?}",
        result.unwrap_err()
    );
    let text = result.unwrap();
    assert!(
        text.contains("no stored embeddings"),
        "expected no-stored-embeddings note, got: {text}"
    );
}

// COV-1: search(query=None, from=None) → InvalidInput("query or --from is required")
// T-248: search_requires_query_or_from
#[test]
fn search_requires_query_or_from() {
    let (y, _dir) = test_yomu();
    let err = y.search(None, 10, 0, &[], false, None).unwrap_err();
    assert!(
        err.to_string().contains("query or --from"),
        "expected invalid input, got: {err}"
    );
}

// COV-2: --from rejects path traversal
// T-249: search_from_rejects_path_traversal
#[test]
fn search_from_rejects_path_traversal() {
    let (y, _dir) = test_yomu();
    let err = y
        .search(None, 10, 0, &[], false, Some("../../../etc/passwd"))
        .unwrap_err();
    assert!(
        err.to_string().contains("must be a relative path"),
        "expected traversal error, got: {err}"
    );
}

// T-562: integration — from-file search excludes source, returns ≤ limit
#[test]
fn search_from_file_integration() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[
            ("src/a.rs", "pub fn alpha() { println!(\"hello\"); }"),
            ("src/b.rs", "pub fn beta() { alpha(); }"),
            ("src/c.rs", "pub fn gamma() { beta(); }"),
        ],
        Arc::new(MockEmbedder::default()),
    );

    let text = y.search(None, 5, 0, &[], false, Some("src/a.rs")).unwrap();
    // Source file should not appear in results
    assert!(
        !text.contains("No results found"),
        "expected results from from-file search: {text}"
    );
    assert!(
        !text.contains("src/a.rs"),
        "source file should be excluded from results: {text}"
    );
}

// COV-5: --from + --path combination excludes files outside path prefix
// T-250: search_from_with_path_filter_excludes_other_dirs
#[test]
fn search_from_with_path_filter_excludes_other_dirs() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[
            ("src/a.rs", "pub fn alpha() {}"),
            ("src/b.rs", "pub fn beta() {}"),
            ("lib/c.rs", "pub fn gamma() {}"),
        ],
        Arc::new(MockEmbedder::default()),
    );
    let text = y
        .search(None, 10, 0, &["src/".to_owned()], false, Some("src/a.rs"))
        .unwrap();
    assert!(
        !text.contains("lib/c.rs") && !text.contains("gamma"),
        "lib/c.rs should be excluded by path filter: {text}"
    );
}

// COV-8: --from with json=true returns valid JSON with "results" key
// T-251: search_from_json_output_has_results_key
#[test]
fn search_from_json_output_has_results_key() {
    let (y, _dir) = test_yomu_with_files_and_embedder(
        &[
            ("src/a.rs", "pub fn alpha() {}"),
            ("src/b.rs", "pub fn beta() {}"),
        ],
        Arc::new(MockEmbedder::default()),
    );
    let json = y.search(None, 10, 0, &[], true, Some("src/a.rs")).unwrap();
    let parsed = parse_json(&json);
    assert!(
        parsed.get("results").is_some(),
        "expected 'results' key in JSON output, got: {json}"
    );
    assert!(
        parsed["results"].is_array(),
        "expected 'results' to be an array, got: {json}"
    );
}

fn seed_brief_chunks(conn: &storage::Db) {
    let emb = vec![0.0_f32; storage::EMBEDDING_DIMS];
    let new_chunk = |name: &'static str, start: u32| storage::NewChunk {
        chunk_type: &storage::ChunkType::RustFn,
        name: Some(name),
        content: "fn body() {}",
        start_line: start,
        end_line: start + 2,
        parent_index: None,
        source_kind: None,
        injection_flags: None,
    };
    storage::insert_chunk(
        conn,
        "src/a.rs",
        &new_chunk("a", 1),
        "h",
        &storage::ce(emb.clone()),
        None,
    )
    .unwrap();
    storage::insert_chunk(
        conn,
        "src/b.rs",
        &new_chunk("b", 1),
        "h",
        &storage::ce(emb),
        None,
    )
    .unwrap();
    storage::replace_file_references(
        conn,
        "src/a.rs",
        &[storage::Reference {
            source_file: "src/a.rs".into(),
            target_file: "src/b.rs".into(),
            symbol_name: None,
            ref_kind: storage::RefKind::Named,
        }],
    )
    .unwrap();
}

fn brief_task(seed_file: &str) -> brief::TaskBrief {
    brief::TaskBrief {
        task: "find body".to_owned(),
        seeds: vec![brief::Seed {
            kind: brief::SeedKind::File,
            value: seed_file.to_owned(),
        }],
        depth: 1,
        max_chunks: 80,
        max_bytes: 80_000,
    }
}

// T-568: yomu_brief_returns_plain_format
#[test]
fn yomu_brief_returns_plain_format() {
    let (conn, dir) = test_db();
    seed_brief_chunks(&conn);
    let yomu = Yomu::for_test(conn, dir.path().to_path_buf(), None);

    let output = yomu.brief(&brief_task("src/a.rs"), false).unwrap();

    assert!(
        output.contains("src/a.rs:1-3"),
        "expected header for src/a.rs, got: {output}"
    );
    assert!(
        output.contains("src/b.rs:1-3"),
        "expected header for src/b.rs, got: {output}"
    );
    assert!(
        output.contains("\n---\n"),
        "expected chunk separator, got: {output}"
    );
}

// T-569: yomu_brief_with_json_returns_valid_json
#[test]
fn yomu_brief_with_json_returns_valid_json() {
    let (conn, dir) = test_db();
    seed_brief_chunks(&conn);
    let yomu = Yomu::for_test(conn, dir.path().to_path_buf(), None);

    let output = yomu.brief(&brief_task("src/a.rs"), true).unwrap();
    let parsed = parse_json(&output);

    assert_eq!(parsed["degraded"], false);
    assert!(parsed["chunks"].is_array());
    assert!(
        parsed["chunks"].as_array().unwrap().len() >= 2,
        "expected at least seed + forward chunks, got: {output}"
    );
}

// T-014b [Spec FR-014b]: brief_fts_seeds_when_no_embedder_and_keyword_matches
#[test]
fn brief_fts_seeds_when_no_embedder_and_keyword_matches() {
    let (conn, dir) = test_db();
    let emb = vec![0.0_f32; storage::EMBEDDING_DIMS];
    storage::insert_chunk(
        &conn,
        "src/auth_handler.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("auth"),
            content: "fn auth() { /* auth logic */ }",
            start_line: 1,
            end_line: 3,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h",
        &storage::ce(emb),
        None,
    )
    .unwrap();
    let yomu = Yomu::for_test(conn, dir.path().to_path_buf(), None);

    let task = brief::TaskBrief {
        task: "auth".to_owned(),
        seeds: vec![],
        depth: 1,
        max_chunks: 80,
        max_bytes: 80_000,
    };

    let output = yomu.brief(&task, true).unwrap();
    let parsed = parse_json(&output);

    assert_eq!(
        parsed["degraded"], true,
        "no embedder must mark degraded, got: {output}"
    );
    let chunks = parsed["chunks"].as_array().unwrap();
    assert!(
        !chunks.is_empty(),
        "FTS hit on 'auth' must produce at least one seed chunk, got: {output}"
    );
}

// T-014c [Spec FR-014b] / T-571: brief_falls_back_to_degraded_when_no_embedder
// T-014c covers the FTS-0-hit branch: task keyword `infer something` finds no
// matching chunk in `fn body() {}`, so closure stays empty even after FTS
// fallback. Behavior identical to the original T-571 assertion.
#[test]
fn brief_falls_back_to_degraded_when_no_embedder() {
    let (conn, dir) = test_db();
    seed_brief_chunks(&conn);
    let yomu = Yomu::for_test(conn, dir.path().to_path_buf(), None);

    let task = brief::TaskBrief {
        task: "infer something".to_owned(),
        seeds: vec![],
        depth: 1,
        max_chunks: 80,
        max_bytes: 80_000,
    };

    let output = yomu.brief(&task, true).unwrap();
    let parsed = parse_json(&output);

    assert_eq!(
        parsed["degraded"], true,
        "no embedder + empty seeds must mark degraded, got: {output}"
    );
    assert_eq!(
        parsed["chunks"].as_array().unwrap().len(),
        0,
        "without seeds the closure is empty, got: {output}"
    );
}

// T-572: brief_emits_warn_on_seed_inference_embed_query_failure
#[traced_test]
#[test]
fn brief_emits_warn_on_seed_inference_embed_query_failure() {
    let (conn, dir) = test_db();
    seed_brief_chunks(&conn);
    let yomu = Yomu::for_test(
        conn,
        dir.path().to_path_buf(),
        Some(Arc::new(FailingEmbedder::all_fail("upstream embedder timeout")) as Arc<dyn Embed>),
    );
    let task = brief::TaskBrief {
        task: "infer something".to_owned(),
        seeds: vec![],
        depth: 1,
        max_chunks: 80,
        max_bytes: 80_000,
    };

    let output = yomu.brief(&task, true).unwrap();
    let parsed = parse_json(&output);
    assert_eq!(
        parsed["degraded"], true,
        "embed_query failure must mark degraded, got: {output}"
    );

    assert!(
        logs_contain("brief seed inference: embed_query"),
        "expected embed_query warn context"
    );
    assert!(
        logs_contain("brief: seed inference"),
        "expected seed inference degraded warn"
    );
}

// T-573: brief_emits_warn_and_degraded_on_zero_chunks [RC-014 #140]
#[traced_test]
#[test]
fn brief_emits_warn_and_degraded_on_zero_chunks() {
    let (yomu, _dir) = test_yomu();
    let task = brief_task("src/nonexistent_seed.rs");
    let output = yomu.brief(&task, true).unwrap();
    let parsed = parse_json(&output);
    assert_eq!(
        parsed["chunks"].as_array().unwrap().len(),
        0,
        "missing seed file must yield zero chunks, got: {output}"
    );
    assert_eq!(
        parsed["degraded"], true,
        "zero chunks must mark degraded, got: {output}"
    );
    assert!(
        logs_contain("brief produced zero chunks"),
        "expected warn for zero chunks"
    );
}

// T-570: yomu_brief_rejects_empty_task [Spec FR-010b prep]
#[test]
fn yomu_brief_rejects_empty_task() {
    let (yomu, _dir) = test_yomu();
    let mut task = brief_task("src/a.rs");
    task.task = "   ".to_owned();
    let err = yomu.brief(&task, false).unwrap_err();
    assert!(
        matches!(err, YomuError::InvalidInput(_)),
        "expected InvalidInput, got: {err:?}"
    );
}

// T-703: yomu_brief_rejects_symbol_seed [Issue #138 COV-004]
// FR-010b: --seed-symbol is not yet implemented, so a TaskBrief carrying any
// Symbol-kind seed must be rejected with SeedSymbolUnimplemented before any
// closure expansion runs.
#[test]
fn yomu_brief_rejects_symbol_seed() {
    let (yomu, _dir) = test_yomu();
    let mut task = brief_task("src/a.rs");
    task.seeds = vec![brief::Seed {
        kind: brief::SeedKind::Symbol,
        value: "Foo".to_owned(),
    }];
    let err = yomu.brief(&task, false).unwrap_err();
    assert!(
        matches!(
            err,
            YomuError::InvalidInput(InvalidInputKind::SeedSymbolUnimplemented)
        ),
        "Symbol seed must be rejected with SeedSymbolUnimplemented, got: {err:?}"
    );
}

// --- CliError / ErrorCode mapping (ADR-0066 Group 2) ---

fn io_err() -> io::Error {
    io::Error::other("boom")
}

// T-EC101: InvalidInput → UsageError (64)
#[test]
fn error_code_invalid_input_is_usage_error() {
    let err = YomuError::InvalidInput(InvalidInputKind::EmptyQuery);
    assert_eq!(err.error_code(), ErrorCode::UsageError);
    assert_eq!(err.exit_code(), ExitCode::from(64));
}

// T-EC102: Internal → Internal (70)
#[test]
fn error_code_internal_is_internal() {
    let err = YomuError::Internal("boom".into());
    assert_eq!(err.error_code(), ErrorCode::Internal);
    assert_eq!(err.exit_code(), ExitCode::from(70));
}

// T-EC103: Storage → CantCreat (73)
#[test]
fn error_code_storage_is_cant_creat() {
    let err = YomuError::Storage(storage::StorageError::Io(io_err()));
    assert_eq!(err.error_code(), ErrorCode::CantCreat);
    assert_eq!(err.exit_code(), ExitCode::from(73));
}

// T-EC104: Io → IoError (74)
#[test]
fn error_code_io_is_io_error() {
    let err = YomuError::Io(io_err());
    assert_eq!(err.error_code(), ErrorCode::IoError);
    assert_eq!(err.exit_code(), ExitCode::from(74));
}

// T-EC105: Index → CantCreat (73)
#[test]
fn error_code_index_is_cant_creat() {
    let err = YomuError::Index(indexer::IndexError::Internal("x".into()));
    assert_eq!(err.error_code(), ErrorCode::CantCreat);
    assert_eq!(err.exit_code(), ExitCode::from(73));
}

// T-EC106: Query → Internal (70)
#[test]
fn error_code_query_is_internal() {
    let err = YomuError::Query(QueryError::Internal("x".into()));
    assert_eq!(err.error_code(), ErrorCode::Internal);
    assert_eq!(err.exit_code(), ExitCode::from(70));
}

// T-EC107: EmbedderUnavailable → TempFailure (75)
#[test]
fn error_code_embedder_unavailable_is_temp_failure() {
    let err = YomuError::EmbedderUnavailable("offline".into());
    assert_eq!(err.error_code(), ErrorCode::TempFailure);
    assert_eq!(err.exit_code(), ExitCode::from(75));
}

// T-CFG001: from_env_with returns defaults when lookup yields nothing
#[test]
fn yomu_config_from_env_with_empty_lookup_uses_defaults() {
    let cfg = YomuConfig::from_env_with(|_| None);
    assert!(!cfg.embed_disabled);
    assert!(!cfg.rerank_enabled);
    assert_eq!(cfg.embed_budget, DEFAULT_EMBED_BUDGET);
}

// T-CFG002: YOMU_EMBED=0 sets embed_disabled
#[test]
fn yomu_config_yomu_embed_zero_disables_embedding() {
    let cfg = YomuConfig::from_env_with(|k| match k {
        "YOMU_EMBED" => Some("0".into()),
        _ => None,
    });
    assert!(cfg.embed_disabled);
}

// T-CFG003: YOMU_EMBED non-"0" leaves embed_disabled false
#[test]
fn yomu_config_yomu_embed_non_zero_keeps_embedding() {
    let cfg = YomuConfig::from_env_with(|k| match k {
        "YOMU_EMBED" => Some("1".into()),
        _ => None,
    });
    assert!(!cfg.embed_disabled);
}

// T-CFG004: YOMU_RERANK=1 enables reranker
#[test]
fn yomu_config_yomu_rerank_one_enables_reranker() {
    let cfg = YomuConfig::from_env_with(|k| match k {
        "YOMU_RERANK" => Some("1".into()),
        _ => None,
    });
    assert!(cfg.rerank_enabled);
}

// T-CFG005: YOMU_RERANK non-"1" leaves reranker disabled
#[test]
fn yomu_config_yomu_rerank_non_one_keeps_reranker_off() {
    let cfg = YomuConfig::from_env_with(|k| match k {
        "YOMU_RERANK" => Some("0".into()),
        _ => None,
    });
    assert!(!cfg.rerank_enabled);
}

// T-CFG006: YOMU_EMBED_BUDGET valid value flows through
#[test]
fn yomu_config_yomu_embed_budget_valid_value_applied() {
    let cfg = YomuConfig::from_env_with(|k| match k {
        "YOMU_EMBED_BUDGET" => Some("10".into()),
        _ => None,
    });
    assert_eq!(cfg.embed_budget, 10);
}

// T-CFG007: Default impl equals from_env_with empty lookup
#[test]
fn yomu_config_default_matches_empty_lookup() {
    let default = YomuConfig::default();
    let empty = YomuConfig::from_env_with(|_| None);
    assert_eq!(default.embed_disabled, empty.embed_disabled);
    assert_eq!(default.rerank_enabled, empty.rerank_enabled);
    assert_eq!(default.embed_budget, empty.embed_budget);
}

// --- Issue #192 Phase 2.2a: InvalidInputKind + YomuError methods ---
//
// Spec: .claude/workspace/planning/2026-05-20-192-phase-2-2a-error-payload/spec.md

// T-001: 8 InvalidInputKind variants are constructible and exhaustively matched.
// FR-001: classify each invalid-input case as one of 8 named variants.
// The `match` has no `_` arm so adding a 9th variant breaks compilation.
#[test]
fn invalid_input_kind_has_eight_variants_with_exhaustive_match() {
    let kinds = [
        InvalidInputKind::PathTraversal {
            path: "x".to_owned(),
        },
        InvalidInputKind::EmptyQuery,
        InvalidInputKind::QueryTooLong {
            max: 1000,
            actual: 1500,
        },
        InvalidInputKind::QueryOrFromRequired,
        InvalidInputKind::EmptyTarget { candidates: vec![] },
        InvalidInputKind::EmptyIndex,
        InvalidInputKind::EmptyTask,
        InvalidInputKind::SeedSymbolUnimplemented,
    ];
    let mut seen = 0_u8;
    for kind in &kinds {
        match kind {
            InvalidInputKind::PathTraversal { .. } => seen |= 1 << 0,
            InvalidInputKind::EmptyQuery => seen |= 1 << 1,
            InvalidInputKind::QueryTooLong { .. } => seen |= 1 << 2,
            InvalidInputKind::QueryOrFromRequired => seen |= 1 << 3,
            InvalidInputKind::EmptyTarget { .. } => seen |= 1 << 4,
            InvalidInputKind::EmptyIndex => seen |= 1 << 5,
            InvalidInputKind::EmptyTask => seen |= 1 << 6,
            InvalidInputKind::SeedSymbolUnimplemented => seen |= 1 << 7,
        }
    }
    assert_eq!(
        seen, 0xFF,
        "expected all 8 InvalidInputKind variants to be visited, got bitmask {seen:#010b}"
    );
}

// T-002: next_step for EmptyQuery returns the kind-specific guidance string.
// FR-002, BR-002.
#[test]
fn next_step_for_empty_query_returns_search_or_stdin_hint() {
    let e = YomuError::InvalidInput(InvalidInputKind::EmptyQuery);
    assert_eq!(
        e.next_step(),
        Some("run `yomu search \"<query>\"` or pipe a query via stdin".to_owned())
    );
}

// T-003: next_step for QueryTooLong embeds the `max` value verbatim.
// FR-002. Uses U+2264 (≤) per spec mapping table.
#[test]
fn next_step_for_query_too_long_includes_max_characters() {
    let e = YomuError::InvalidInput(InvalidInputKind::QueryTooLong {
        max: 1000,
        actual: 1500,
    });
    assert_eq!(
        e.next_step(),
        Some("shorten query to ≤ 1000 characters".to_owned())
    );
}

// T-004: next_step for EmbedderUnavailable suggests model download + env opt-out.
// FR-002, AS-002.
#[test]
fn next_step_for_embedder_unavailable_recommends_download_or_disable() {
    let e = YomuError::EmbedderUnavailable("test".into());
    assert_eq!(
        e.next_step(),
        Some("run `yomu model download`, or set `YOMU_EMBED=0` for FTS-only mode".to_owned())
    );
}

// T-005: next_step for Index errors recommends `yomu rebuild`.
// FR-002.
#[test]
fn next_step_for_index_error_recommends_rebuild() {
    let inner = indexer::IndexError::Internal("x".into());
    let e = YomuError::Index(inner);
    assert_eq!(
        e.next_step(),
        Some("run `yomu rebuild` to recreate the index".to_owned())
    );
}

// T-006: next_step for Storage errors recommends a permissions/disk check.
// FR-002.
#[test]
fn next_step_for_storage_error_recommends_permission_check() {
    let inner = storage::StorageError::LengthMismatch {
        chunks: 1,
        embeddings: 0,
    };
    let e = YomuError::Storage(inner);
    assert_eq!(
        e.next_step(),
        Some("check `.yomu/index.db` permissions and disk space".to_owned())
    );
}

// T-007: next_step for Internal errors recommends a bug report.
// FR-002.
#[test]
fn next_step_for_internal_error_recommends_bug_report() {
    let e = YomuError::Internal("bug".into());
    assert_eq!(
        e.next_step(),
        Some("file a bug report with the reproduction command and `RUST_BACKTRACE=1`".to_owned())
    );
}

// T-008: next_step for Io returns None (no actionable recommendation).
// FR-V01.
#[test]
fn next_step_for_io_error_returns_none() {
    let e = YomuError::Io(io::Error::other("x"));
    assert_eq!(e.next_step(), None);
}

// T-009: next_step for Query returns None until Query subtypes gain hints.
// FR-V01. Spec references QueryError::NoQuery, but YomuError::Query wraps
// crate::query::QueryError (Storage / Embed / Internal). Internal is the
// representative variant for FR-V01 coverage; see Suggestions for spec fix.
#[test]
fn next_step_for_query_error_returns_none() {
    let inner = query::QueryError::Internal("x".into());
    let e = YomuError::Query(inner);
    assert_eq!(e.next_step(), None);
}

// T-010: retryable=true for EmbedderUnavailable (recoverable via model download).
// FR-006, BR-001, AS-002.
#[test]
fn retryable_for_embedder_unavailable_is_true() {
    let e = YomuError::EmbedderUnavailable("x".into());
    assert!(e.retryable(), "EmbedderUnavailable must be retryable");
}

// T-011: retryable=false for Storage (disk/permission errors do not self-heal).
// FR-006, BR-001.
#[test]
fn retryable_for_storage_error_is_false() {
    let inner = storage::StorageError::LengthMismatch {
        chunks: 1,
        embeddings: 0,
    };
    let e = YomuError::Storage(inner);
    assert!(
        !e.retryable(),
        "Storage errors are not immediately retryable"
    );
}

// T-012: retryable=false for Io.
// FR-006, BR-001.
#[test]
fn retryable_for_io_error_is_false() {
    let e = YomuError::Io(io::Error::other("x"));
    assert!(!e.retryable(), "Io errors are not immediately retryable");
}

// T-013: retryable=false for Index.
// FR-006, BR-001.
#[test]
fn retryable_for_index_error_is_false() {
    let inner = indexer::IndexError::Internal("x".into());
    let e = YomuError::Index(inner);
    assert!(!e.retryable(), "Index errors are not immediately retryable");
}

// T-014: retryable=false for Internal (logic bugs do not recover on retry).
// FR-006, BR-001.
#[test]
fn retryable_for_internal_error_is_false() {
    let e = YomuError::Internal("x".into());
    assert!(!e.retryable(), "Internal errors are not retryable");
}

// T-015: retryable=false for Query.
// FR-006, BR-001.
#[test]
fn retryable_for_query_error_is_false() {
    let inner = query::QueryError::Internal("x".into());
    let e = YomuError::Query(inner);
    assert!(!e.retryable(), "Query errors are not immediately retryable");
}

// T-016: retryable=false for InvalidInput (caller must supply different input).
// FR-006, BR-001.
#[test]
fn retryable_for_invalid_input_is_false() {
    let e = YomuError::InvalidInput(InvalidInputKind::EmptyQuery);
    assert!(!e.retryable(), "InvalidInput errors are not retryable");
}

// T-017 (Issue #192 Phase 2.2a): superseded by T-005 (Issue #197).
// The original assertion "EmptyTarget returns empty candidates" inverts FR-001
// of #197 (EmptyTarget now carries candidates). T-005 below covers the
// remaining contract: non-EmptyTarget variants return Vec::new().

// Phase 2.2b — degraded / notes coverage on 6 routes (issue #192).
// Spec: .claude/workspace/planning/2026-05-20-192-phase-2-2b-degraded-notes/spec.md

fn make_index_result(files_errored: u32) -> indexer::IndexResult {
    indexer::IndexResult {
        files_processed: 3,
        chunks_created: 10,
        files_skipped: 0,
        files_errored,
    }
}

fn make_index_status() -> storage::IndexStatus {
    storage::IndexStatus {
        total_files: 3,
        total_chunks: 10,
        embeddable_chunks: 10,
        embedded_chunks: 10,
        last_indexed_at: Some("2026-05-20".to_owned()),
    }
}

fn make_partial_index_status() -> storage::IndexStatus {
    // embedded < embeddable → triggers JsonMutationResult.coverage Some(...).
    // Used by T-009 to assert notes never duplicates the coverage wording.
    storage::IndexStatus {
        total_files: 3,
        total_chunks: 10,
        embeddable_chunks: 10,
        embedded_chunks: 4,
        last_indexed_at: Some("2026-05-20".to_owned()),
    }
}

fn make_dry_run_result(files_errored: u32) -> indexer::DryRunResult {
    indexer::DryRunResult {
        total_files: 10,
        files_to_process: 5,
        files_to_skip: 4,
        files_errored,
        orphans_to_remove: 0,
    }
}

fn make_embed_result(chunks_embedded: u32, files_completed: u32) -> indexer::EmbedResult {
    indexer::EmbedResult {
        chunks_embedded,
        files_completed,
    }
}

// T-001: format_index_json with files_errored=2 marks degraded and explains why.
// FR-002.
#[test]
fn format_index_json_with_errored_files_is_degraded() {
    let result = make_index_result(2);
    let stats = make_index_status();
    let (degraded, notes) = super::degraded_for_chunk_errors(result.files_errored);

    let json = format_index_json(&result, &stats, degraded, notes);
    let parsed = parse_json(&json);

    assert_eq!(
        parsed["degraded"], true,
        "files_errored=2 should set degraded=true: {json}"
    );
    let notes_arr = parsed["notes"]
        .as_array()
        .unwrap_or_else(|| panic!("notes must be an array: {json}"));
    assert!(
        notes_arr
            .iter()
            .any(|n| n.as_str() == Some("failed to chunk 2 file(s)")),
        "notes must contain failure description, got: {json}"
    );
}

// T-002: format_index_json with no errors emits degraded=false and empty notes.
// FR-002 (false branch) + FR-001 (key always present).
#[test]
fn format_index_json_with_no_errors_is_not_degraded() {
    let result = make_index_result(0);
    let stats = make_index_status();
    let (degraded, notes) = super::degraded_for_chunk_errors(result.files_errored);

    let json = format_index_json(&result, &stats, degraded, notes);
    let parsed = parse_json(&json);

    assert_eq!(
        parsed["degraded"], false,
        "files_errored=0 should set degraded=false: {json}"
    );
    let notes_arr = parsed["notes"]
        .as_array()
        .unwrap_or_else(|| panic!("notes must be an array: {json}"));
    assert!(
        notes_arr.is_empty(),
        "notes must be empty when not degraded, got: {json}"
    );
}

// T-003: format_rebuild_json with files_errored=1 marks degraded.
// FR-003.
#[test]
fn format_rebuild_json_with_errored_files_is_degraded() {
    let result = indexer::IndexResult {
        files_processed: 2,
        chunks_created: 8,
        files_skipped: 0,
        files_errored: 1,
    };
    let stats = make_index_status();
    let (degraded, notes) = super::degraded_for_chunk_errors(result.files_errored);

    let json = format_rebuild_json(&result, &stats, degraded, notes);
    let parsed = parse_json(&json);

    assert_eq!(
        parsed["degraded"], true,
        "rebuild with files_errored=1 should set degraded=true: {json}"
    );
    let notes_arr = parsed["notes"]
        .as_array()
        .unwrap_or_else(|| panic!("notes must be an array: {json}"));
    assert!(
        notes_arr
            .iter()
            .any(|n| n.as_str() == Some("failed to chunk 1 file(s)")),
        "rebuild notes must contain failure description, got: {json}"
    );
}

// T-004: format_dry_run_json with files_errored=1 marks degraded with check wording.
// FR-004.
#[test]
fn format_dry_run_json_with_errored_files_is_degraded() {
    let result = make_dry_run_result(1);
    let (degraded, notes) = super::degraded_for_dry_run_errors(result.files_errored);

    let json = format_dry_run_json(&result, degraded, notes);
    let parsed = parse_json(&json);

    assert_eq!(
        parsed["degraded"], true,
        "dry_run with files_errored=1 should set degraded=true: {json}"
    );
    let notes_arr = parsed["notes"]
        .as_array()
        .unwrap_or_else(|| panic!("notes must be an array: {json}"));
    assert!(
        notes_arr
            .iter()
            .any(|n| n.as_str() == Some("check failed for 1 file(s)")),
        "dry_run notes must contain check-failure description, got: {json}"
    );
}

// T-005: format_embed_result with pending > chunks_embedded marks degraded
// and reports both counts plus the skip count.
// FR-005.
#[test]
fn format_embed_result_pending_exceeds_embedded_is_degraded() {
    let pending = 5;
    let result = make_embed_result(3, 2);
    let (degraded, notes) = super::degraded_for_embed_skips(pending, result.chunks_embedded);

    let json = format_embed_result(&result, true, degraded, notes);
    let parsed = parse_json(&json);

    assert_eq!(
        parsed["degraded"], true,
        "pending=5 > embedded=3 should set degraded=true: {json}"
    );
    let notes_arr = parsed["notes"]
        .as_array()
        .unwrap_or_else(|| panic!("notes must be an array: {json}"));
    assert!(
        notes_arr
            .iter()
            .any(|n| n.as_str() == Some("embedded 3 of 5 chunks (2 skipped)")),
        "embed notes must contain count breakdown, got: {json}"
    );
}

// T-006: format_status_json with (false, vec![]) emits constant non-degraded envelope.
// FR-006 (status is never degraded by this route's design).
#[test]
fn format_status_json_is_never_degraded() {
    let stats = make_index_status();
    let ref_count = 7;

    let json = format_status_json(&stats, ref_count, false, vec![]);
    let parsed = parse_json(&json);

    assert_eq!(
        parsed["degraded"], false,
        "status must emit degraded=false: {json}"
    );
    let notes_arr = parsed["notes"]
        .as_array()
        .unwrap_or_else(|| panic!("notes must be an array: {json}"));
    assert!(notes_arr.is_empty(), "status notes must be empty: {json}");
}

// T-007: format_impact_json with (false, vec![]) emits constant non-degraded envelope.
// FR-006 (impact is never degraded by this route's design).
#[test]
fn format_impact_json_is_never_degraded() {
    let dependents: Vec<storage::Dependent> = vec![];
    let direct_refs: HashMap<String, Vec<storage::DirectReference>> = HashMap::new();
    let symbol_refs: Vec<String> = vec![];
    let semantic_related: Vec<storage::SearchResult> = vec![];

    let json = format_impact_json(
        "src/Button.tsx",
        true,
        &dependents,
        &direct_refs,
        &symbol_refs,
        &semantic_related,
        false,
        vec![],
    );
    let parsed = parse_json(&json);

    assert_eq!(
        parsed["degraded"], false,
        "impact must emit degraded=false: {json}"
    );
    let notes_arr = parsed["notes"]
        .as_array()
        .unwrap_or_else(|| panic!("notes must be an array: {json}"));
    assert!(notes_arr.is_empty(), "impact notes must be empty: {json}");
}

// T-009: format_index_json notes never duplicate the coverage wording.
// FR-007 (negative assertion); paired with AS-002 (coverage field unchanged).
#[test]
fn format_index_json_notes_does_not_mention_coverage() {
    // files_errored=1 forces degraded=true so notes is non-empty.
    // partial embedding means JsonMutationResult.coverage is Some(...),
    // so the JSON does contain the word "coverage" — but only in that field,
    // never in the notes array.
    let result = make_index_result(1);
    let stats = make_partial_index_status();
    let (degraded, notes) = super::degraded_for_chunk_errors(result.files_errored);

    let json = format_index_json(&result, &stats, degraded, notes);
    let parsed = parse_json(&json);

    let notes_arr = parsed["notes"]
        .as_array()
        .unwrap_or_else(|| panic!("notes must be an array: {json}"));
    assert!(
        !notes_arr.is_empty(),
        "precondition: notes non-empty: {json}"
    );
    assert!(
        !notes_arr
            .iter()
            .any(|n| n.as_str().is_some_and(|s| s.contains("coverage"))),
        "notes must not mention coverage (FR-007), got: {json}"
    );
}

// T-010: format_embed_result with pending == chunks_embedded is not degraded.
// FR-005 (boundary at equality) + BR-001 (coverage partiality is not a degradation).
#[test]
fn format_embed_result_full_completion_is_not_degraded() {
    let pending = 5;
    let result = make_embed_result(5, 5);
    let (degraded, notes) = super::degraded_for_embed_skips(pending, result.chunks_embedded);

    let json = format_embed_result(&result, true, degraded, notes);
    let parsed = parse_json(&json);

    assert_eq!(
        parsed["degraded"], false,
        "pending==embedded should set degraded=false: {json}"
    );
    let notes_arr = parsed["notes"]
        .as_array()
        .unwrap_or_else(|| panic!("notes must be an array: {json}"));
    assert!(
        notes_arr.is_empty(),
        "notes must be empty when fully embedded: {json}"
    );
}

// --- Issue #197: EmptyTarget candidates dynamic supply ---
//
// Spec: .claude/workspace/planning/2026-05-20-197-empty-target-candidates/spec.md
//
// FR-001..FR-005 cover the `EmptyTarget` cause carrying an index-sourced
// `Vec<String>` of file paths. T-001 to T-003 exercise the unit boundary;
// T-005 fixes the negative half (other variants return empty Vec). T-004 is
// the integration counterpart and lives in `tests/cli_integration.rs`.

/// Unwraps an `EmptyTarget` cause and returns its `candidates` vector.
/// Panics for every other variant — used in T-001 / T-002 / T-003 to keep
/// each test's body assertion-focused. T-005 covers the negative dispatch.
fn unwrap_empty_target_candidates(err: YomuError) -> Vec<String> {
    match err {
        YomuError::InvalidInput(InvalidInputKind::EmptyTarget { candidates }) => candidates,
        other => panic!("expected EmptyTarget with candidates, got: {other:?}"),
    }
}

/// Inserts a placeholder chunk row at each given `file_path`. The only column
/// `get_all_file_paths` reads is `file_path`, so the chunk body is arbitrary;
/// using `insert_chunk_row` (vs. `insert_chunk`) skips vector embedding writes
/// and keeps the helper minimal.
fn seed_chunks_at_paths(conn: &storage::Db, paths: &[&str]) {
    for path in paths {
        storage::insert_chunk_row(
            conn,
            path,
            &storage::NewChunk {
                chunk_type: &storage::ChunkType::Other,
                name: None,
                content: "x",
                start_line: 1,
                end_line: 1,
                parent_index: None,
                source_kind: None,
                injection_flags: None,
            },
            "hash",
            None,
        )
        .unwrap();
    }
}

// T-001: impact_empty_target_returns_candidates_with_indexed_paths
// FR-001. With 3 indexed paths in already-sorted insertion order, the
// EmptyTarget cause carries those 3 paths verbatim.
#[test]
fn impact_empty_target_returns_candidates_with_indexed_paths() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_chunks_at_paths(&conn, &["a.rs", "b.rs", "c.rs"]);
    }
    let candidates =
        unwrap_empty_target_candidates(y.impact("", None, 0, false, false).unwrap_err());
    assert_eq!(
        candidates,
        vec!["a.rs".to_owned(), "b.rs".to_owned(), "c.rs".to_owned()],
        "FR-001: candidates must contain the 3 indexed paths"
    );
}

// T-002: impact_empty_target_sorts_candidates_alphabetically
// FR-002. Paths inserted in non-sorted order ("c", "a", "b") must emerge
// alphabetically. Distinct from T-001 because the input order differs from
// the asserted output — this is the test that actually discriminates sorting
// from insertion-order accidental ordering.
#[test]
fn impact_empty_target_sorts_candidates_alphabetically() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock().unwrap();
        seed_chunks_at_paths(&conn, &["c.rs", "a.rs", "b.rs"]);
    }
    let candidates =
        unwrap_empty_target_candidates(y.impact("", None, 0, false, false).unwrap_err());
    assert_eq!(
        candidates,
        vec!["a.rs".to_owned(), "b.rs".to_owned(), "c.rs".to_owned()],
        "FR-002: candidates must be sorted lexicographically regardless of insertion order"
    );
}

// T-003: impact_empty_target_caps_candidates_at_ten
// FR-003. With 15 paths "file00.rs" .. "file14.rs", the candidate list is
// truncated to the lexicographically-first 10 entries.
#[test]
fn impact_empty_target_caps_candidates_at_ten() {
    let (y, _dir) = test_yomu();
    let paths: Vec<String> = (0..15).map(|i| format!("file{i:02}.rs")).collect();
    let path_refs: Vec<&str> = paths.iter().map(String::as_str).collect();
    {
        let conn = y.conn.lock().unwrap();
        seed_chunks_at_paths(&conn, &path_refs);
    }
    let candidates =
        unwrap_empty_target_candidates(y.impact("", None, 0, false, false).unwrap_err());
    let expected: Vec<String> = (0..super::MAX_EMPTY_TARGET_CANDIDATES)
        .map(|i| format!("file{i:02}.rs"))
        .collect();
    assert_eq!(
        candidates,
        expected,
        "FR-003: candidates must be the lexicographically-first MAX_EMPTY_TARGET_CANDIDATES paths (capped at {})",
        super::MAX_EMPTY_TARGET_CANDIDATES
    );
}

// T-005: candidates_returns_empty_for_non_empty_target_variants
// FR-001 (negative). YomuError::candidates() returns Vec::new() for every
// InvalidInputKind variant except EmptyTarget. This guards the dispatch
// branch from regressing into a catch-all that leaks data through other
// variants.
#[test]
fn candidates_returns_empty_for_non_empty_target_variants() {
    let cases: Vec<YomuError> = vec![
        YomuError::InvalidInput(InvalidInputKind::PathTraversal {
            path: "x".to_owned(),
        }),
        YomuError::InvalidInput(InvalidInputKind::EmptyQuery),
        YomuError::InvalidInput(InvalidInputKind::QueryTooLong {
            max: 1000,
            actual: 1500,
        }),
        YomuError::InvalidInput(InvalidInputKind::QueryOrFromRequired),
        YomuError::InvalidInput(InvalidInputKind::EmptyIndex),
        YomuError::InvalidInput(InvalidInputKind::EmptyTask),
        YomuError::InvalidInput(InvalidInputKind::SeedSymbolUnimplemented),
    ];
    for e in &cases {
        assert_eq!(
            e.candidates(),
            Vec::<String>::new(),
            "FR-001 negative: non-EmptyTarget variant {e:?} must return Vec::new()"
        );
    }
}
