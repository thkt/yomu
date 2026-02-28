use super::*;
use std::collections::HashMap;

fn test_db() -> (storage::Db, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();
    (conn, dir)
}

fn test_yomu() -> (Yomu, tempfile::TempDir) {
    let (conn, dir) = test_db();
    let y = Yomu {
        conn: Arc::new(Mutex::new(conn)),
        embedder: None,
        root: dir.path().to_path_buf(),
        auto_indexed: Arc::new(AtomicBool::new(false)),
        auto_index_failures: Arc::new(AtomicU32::new(0)),
        tool_router: Yomu::tool_router(),
    };
    (y, dir)
}

#[test]
fn tool_router_has_four_tools() {
    let router = Yomu::tool_router();
    let names: Vec<&str> = router.map.keys().map(|k| k.as_ref()).collect();
    assert!(names.contains(&"explorer"), "missing explorer: {names:?}");
    assert!(names.contains(&"index"), "missing index: {names:?}");
    assert!(names.contains(&"impact"), "missing impact: {names:?}");
    assert!(names.contains(&"status"), "missing status: {names:?}");
    assert_eq!(names.len(), 4, "expected 4 tools, got {names:?}");
}

#[tokio::test]
async fn explorer_rejects_empty_query() {
    let (y, _dir) = test_yomu();
    let params = Parameters(ExplorerParams {
        query: String::new(),
        limit: None,
        offset: None,
    });
    let err = y.explorer(params).await.unwrap_err();
    assert!(
        err.message.contains("empty"),
        "expected empty error, got: {}",
        err.message
    );
}

#[tokio::test]
async fn status_returns_empty_stats() {
    let (y, _dir) = test_yomu();
    let result = y.status().await.unwrap();
    let text = &result.content[0].as_text().unwrap().text;
    assert!(text.contains("Files: 0"), "expected 0 files, got: {text}");
    assert!(
        text.contains("Chunks: 0"),
        "expected 0 chunks, got: {text}"
    );
    assert!(
        text.contains("References: 0"),
        "expected 0 references, got: {text}"
    );
    assert!(text.contains("never"), "expected 'never', got: {text}");
}

#[tokio::test]
async fn index_requires_api_key() {
    let (y, _dir) = test_yomu();
    let params = Parameters(IndexParams { force: None });
    let err = y.index(params).await.unwrap_err();
    assert!(
        err.message.contains("GEMINI_API_KEY"),
        "expected API key error, got: {}",
        err.message
    );
}

#[tokio::test]
async fn explorer_requires_api_key() {
    let (y, _dir) = test_yomu();
    let params = Parameters(ExplorerParams {
        query: "test query".to_string(),
        limit: None,
        offset: None,
    });
    let err = y.explorer(params).await.unwrap_err();
    assert!(
        err.message.contains("GEMINI_API_KEY"),
        "expected API key error, got: {}",
        err.message
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
    let text = format_results_grouped(&results, &imports_map, &siblings_map);
    assert!(text.contains("## src/Button.tsx"), "missing file header: {text}");
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
    let empty_imports: HashMap<String, String> = HashMap::new();
    let empty_siblings: HashMap<String, Vec<storage::SiblingInfo>> = HashMap::new();
    let text = format_results_grouped(&results, &empty_imports, &empty_siblings);
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
    let empty_imports = HashMap::new();
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
    let text = format_results_grouped(&results, &empty_imports, &siblings_map);
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
    let empty: HashMap<String, Vec<storage::SiblingInfo>> = HashMap::new();
    let text = format_results_grouped(&results, &imports_map, &empty);
    assert!(!text.contains("Imports:"), "empty imports should be omitted: {text}");
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
    let empty_imports: HashMap<String, String> = HashMap::new();
    let siblings_map = HashMap::from([("src/A.tsx".to_string(), vec![])]);
    let text = format_results_grouped(&results, &empty_imports, &siblings_map);
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
    let empty: HashMap<String, String> = HashMap::new();
    let empty_siblings: HashMap<String, Vec<storage::SiblingInfo>> = HashMap::new();
    let text = format_results_grouped(&results, &empty, &empty_siblings);
    let a_pos = text.find("## src/A.tsx").unwrap();
    let b_pos = text.find("## src/B.tsx").unwrap();
    assert!(
        a_pos < b_pos,
        "A (better similarity) should come before B: {text}"
    );
}

#[tokio::test]
async fn explorer_auto_indexes_empty_db() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("Button.tsx"),
        "function Button() { return <div/>; }",
    )
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let y = Yomu {
        conn: Arc::new(Mutex::new(conn)),
        embedder: Some(Arc::new(crate::indexer::embedder::MockEmbedder)),
        root: dir.path().to_path_buf(),
        auto_indexed: Arc::new(AtomicBool::new(false)),
        auto_index_failures: Arc::new(AtomicU32::new(0)),
        tool_router: Yomu::tool_router(),
    };

    let params = Parameters(ExplorerParams {
        query: "button component".to_string(),
        limit: None,
        offset: None,
    });
    let result = y.explorer(params).await.unwrap();
    let text = &result.content[0].as_text().unwrap().text;
    assert!(
        !text.contains("No results found"),
        "expected results after auto-index, got: {text}"
    );
    assert!(text.contains("Button"), "expected Button in results, got: {text}");
}

fn seed_index(conn: &storage::Db) {
    let embedding = vec![0.0_f32; 768];
    storage::insert_chunk(
        conn, "src/dummy.tsx",
        &storage::NewChunk { chunk_type: &storage::ChunkType::Other, name: None, content: "seed", start_line: 1, end_line: 1 },
        "seed", &embedding,
    ).unwrap();
}

#[tokio::test]
async fn impact_lists_dependents() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock();
        seed_index(&conn);
        storage::replace_file_references(&conn, "src/A.tsx", &[
            storage::Reference { source_file: "src/A.tsx".into(), target_file: "src/hooks/useAuth.ts".into(), symbol_name: Some("useAuth".into()), ref_kind: storage::RefKind::Named },
        ]).unwrap();
        storage::replace_file_references(&conn, "src/C.tsx", &[
            storage::Reference { source_file: "src/C.tsx".into(), target_file: "src/hooks/useAuth.ts".into(), symbol_name: Some("useAuth".into()), ref_kind: storage::RefKind::Named },
        ]).unwrap();
    }

    let params = Parameters(ImpactParams {
        target: "src/hooks/useAuth.ts".to_string(),
        symbol: None,
        depth: None,
    });
    let result = y.impact(params).await.unwrap();
    let text = &result.content[0].as_text().unwrap().text;
    assert!(text.contains("src/A.tsx"), "expected A.tsx: {text}");
    assert!(text.contains("src/C.tsx"), "expected C.tsx: {text}");
    assert!(text.contains("2 dependent"), "expected 2 dependents: {text}");
}

#[tokio::test]
async fn impact_filters_by_symbol() {
    let (y, _dir) = test_yomu();
    {
        let conn = y.conn.lock();
        seed_index(&conn);
        storage::replace_file_references(&conn, "src/A.tsx", &[
            storage::Reference { source_file: "src/A.tsx".into(), target_file: "src/hooks/useAuth.ts".into(), symbol_name: Some("useAuth".into()), ref_kind: storage::RefKind::Named },
        ]).unwrap();
        storage::replace_file_references(&conn, "src/B.tsx", &[
            storage::Reference { source_file: "src/B.tsx".into(), target_file: "src/hooks/useAuth.ts".into(), symbol_name: Some("AuthProvider".into()), ref_kind: storage::RefKind::Named },
        ]).unwrap();
    }

    let params = Parameters(ImpactParams {
        target: "src/hooks/useAuth.ts:useAuth".to_string(),
        symbol: None,
        depth: None,
    });
    let result = y.impact(params).await.unwrap();
    let text = &result.content[0].as_text().unwrap().text;
    assert!(text.contains("Direct symbol references"), "expected symbol section: {text}");
    assert!(text.contains("src/A.tsx"), "expected A.tsx in symbol refs: {text}");
}

#[tokio::test]
async fn impact_rejects_empty_target() {
    let (y, _dir) = test_yomu();
    let params = Parameters(ImpactParams {
        target: String::new(),
        symbol: None,
        depth: None,
    });
    let err = y.impact(params).await.unwrap_err();
    assert!(
        err.message.contains("empty"),
        "expected empty error, got: {}",
        err.message
    );
}

#[tokio::test]
async fn integration_index_then_impact() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();

    std::fs::write(
        src_dir.join("A.tsx"),
        "import { B } from './B';\nfunction A() { return <B/>; }",
    ).unwrap();
    std::fs::write(
        src_dir.join("B.tsx"),
        "import { C } from './C';\nexport function B() { return <C/>; }",
    ).unwrap();
    std::fs::write(
        src_dir.join("C.tsx"),
        "export function C() { return <div/>; }",
    ).unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    indexer::run_index(
        Arc::clone(&conn),
        dir.path(),
        &crate::indexer::embedder::MockEmbedder,
        false,
    ).await.unwrap();

    let y = Yomu {
        conn,
        embedder: None,
        root: dir.path().to_path_buf(),
        auto_indexed: Arc::new(AtomicBool::new(false)),
        auto_index_failures: Arc::new(AtomicU32::new(0)),
        tool_router: Yomu::tool_router(),
    };

    let params = Parameters(ImpactParams {
        target: "src/C.tsx".to_string(),
        symbol: None,
        depth: Some(3),
    });
    let result = y.impact(params).await.unwrap();
    let text = &result.content[0].as_text().unwrap().text;
    assert!(text.contains("src/B.tsx"), "expected B.tsx as direct dependent: {text}");
    assert!(text.contains("src/A.tsx"), "expected A.tsx as transitive dependent: {text}");
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

#[tokio::test]
async fn status_returns_counts_after_insert() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; 768];
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

    let y = Yomu {
        conn: Arc::new(Mutex::new(conn)),
        embedder: None,
        root: PathBuf::from("/tmp"),
        auto_index_failures: Arc::new(AtomicU32::new(0)),
        auto_indexed: Arc::new(AtomicBool::new(false)),
        tool_router: Yomu::tool_router(),
    };
    let result = y.status().await.unwrap();
    let text = &result.content[0].as_text().unwrap().text;
    assert!(text.contains("Files: 1"), "expected 1 file, got: {text}");
    assert!(
        text.contains("Chunks: 1"),
        "expected 1 chunk, got: {text}"
    );
}

#[tokio::test]
async fn explorer_hybrid_flow_empty_db() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("Button.tsx"),
        "export function Button() { return <div/>; }",
    )
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let y = Yomu {
        conn: Arc::clone(&conn),
        embedder: Some(Arc::new(crate::indexer::embedder::MockEmbedder)),
        root: dir.path().to_path_buf(),
        auto_indexed: Arc::new(AtomicBool::new(false)),
        auto_index_failures: Arc::new(AtomicU32::new(0)),
        tool_router: Yomu::tool_router(),
    };

    let params = Parameters(ExplorerParams {
        query: "button component".to_string(),
        limit: None,
        offset: None,
    });
    let result = y.explorer(params).await.unwrap();
    let text = &result.content[0].as_text().unwrap().text;
    assert!(text.contains("Button"), "expected Button in results: {text}");

    let stats = {
        let c = conn.lock();
        storage::get_stats(&c).unwrap()
    };
    assert!(stats.total_chunks > 0, "expected chunks after hybrid index");
    assert!(
        stats.embedded_chunks > 0,
        "expected embeddings after hybrid index"
    );
}

#[tokio::test]
async fn explorer_incremental_embeds_chunked_only() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("Form.tsx"),
        "export function Form() { return <form/>; }",
    )
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    indexer::run_chunk_only_index(Arc::clone(&conn), dir.path())
        .await
        .unwrap();

    {
        let c = conn.lock();
        let stats = storage::get_stats(&c).unwrap();
        assert!(stats.total_chunks > 0, "should have chunks");
        assert_eq!(stats.embedded_chunks, 0, "should have no embeddings yet");
    }

    let y = Yomu {
        conn: Arc::clone(&conn),
        embedder: Some(Arc::new(crate::indexer::embedder::MockEmbedder)),
        root: dir.path().to_path_buf(),
        auto_indexed: Arc::new(AtomicBool::new(true)), // already chunk-indexed
        auto_index_failures: Arc::new(AtomicU32::new(0)),
        tool_router: Yomu::tool_router(),
    };

    let params = Parameters(ExplorerParams {
        query: "form component".to_string(),
        limit: None,
        offset: None,
    });
    let result = y.explorer(params).await.unwrap();
    let text = &result.content[0].as_text().unwrap().text;
    assert!(text.contains("Form"), "expected Form in results: {text}");

    {
        let c = conn.lock();
        let stats = storage::get_stats(&c).unwrap();
        assert!(
            stats.embedded_chunks > 0,
            "expected embeddings after incremental embed"
        );
    }
}

#[tokio::test]
async fn explorer_shows_coverage_on_no_results() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("A.tsx"),
        "export function A() { return <div/>; }",
    )
    .unwrap();

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
async fn index_embeds_chunk_only_files() {
    let dir = tempfile::tempdir().unwrap();
    let src_dir = dir.path().join("src");
    std::fs::create_dir_all(&src_dir).unwrap();
    std::fs::write(
        src_dir.join("Header.tsx"),
        "export function Header() { return <header/>; }",
    )
    .unwrap();
    std::fs::write(
        src_dir.join("Footer.tsx"),
        "export function Footer() { return <footer/>; }",
    )
    .unwrap();

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    indexer::run_chunk_only_index(Arc::clone(&conn), dir.path())
        .await
        .unwrap();

    let y = Yomu {
        conn: Arc::clone(&conn),
        embedder: Some(Arc::new(crate::indexer::embedder::MockEmbedder)),
        root: dir.path().to_path_buf(),
        auto_indexed: Arc::new(AtomicBool::new(false)),
        auto_index_failures: Arc::new(AtomicU32::new(0)),
        tool_router: Yomu::tool_router(),
    };

    let params = Parameters(IndexParams { force: None });
    let result = y.index(params).await.unwrap();
    let text = &result.content[0].as_text().unwrap().text;
    assert!(
        text.contains("complete"),
        "expected completion message: {text}"
    );

    let stats = {
        let c = conn.lock();
        storage::get_stats(&c).unwrap()
    };
    assert!(stats.total_chunks > 0, "should have chunks");
    assert_eq!(
        stats.embedded_chunks, stats.total_chunks,
        "all chunks should be embedded"
    );
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

    let y = Yomu {
        conn: Arc::new(Mutex::new(conn)),
        embedder: None,
        root: PathBuf::from("/tmp"),
        auto_indexed: Arc::new(AtomicBool::new(false)),
        auto_index_failures: Arc::new(AtomicU32::new(0)),
        tool_router: Yomu::tool_router(),
    };

    let result = y.status().await.unwrap();
    let text = &result.content[0].as_text().unwrap().text;
    assert!(
        text.contains("Embedded: 0/"),
        "expected embedded count in status: {text}"
    );
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
    let empty_imports: HashMap<String, String> = HashMap::new();
    let empty_siblings: HashMap<String, Vec<storage::SiblingInfo>> = HashMap::new();
    let text = format_results_grouped(&results, &empty_imports, &empty_siblings);
    assert!(text.contains("(similarity: 0.72)"), "expected score for Semantic: {text}");
    assert!(text.contains("(similarity: 0.55)"), "expected score for NameMatch: {text}");
    assert!(!text.contains("(name match)"), "should not show (name match): {text}");
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
    let empty: HashMap<String, String> = HashMap::new();
    let empty_siblings: HashMap<String, Vec<storage::SiblingInfo>> = HashMap::new();
    let text = format_results_grouped(&results, &empty, &empty_siblings);
    let a_pos = text.find("## src/A.tsx").unwrap();
    let b_pos = text.find("## src/B.tsx").unwrap();
    assert!(a_pos < b_pos, "A (score 0.95) should come before B (score 0.60): {text}");
}

#[test]
fn with_root_creates_db_and_returns_yomu() {
    let dir = tempfile::tempdir().unwrap();
    let result = Yomu::with_root(dir.path().to_path_buf());
    assert!(result.is_ok(), "with_root should succeed: {:?}", result.err());
    let yomu = result.unwrap();
    assert_eq!(yomu.root, dir.path());
    assert!(dir.path().join(".yomu").join("index.db").exists());
}

#[test]
fn with_root_initializes_all_fields() {
    let dir = tempfile::tempdir().unwrap();
    let yomu = Yomu::with_root(dir.path().to_path_buf()).unwrap();
    assert!(!yomu.auto_indexed.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(yomu.auto_index_failures.load(std::sync::atomic::Ordering::SeqCst), 0);
    assert_eq!(yomu.tool_router.map.len(), 4);
}
