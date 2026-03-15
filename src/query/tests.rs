use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use super::*;
use crate::indexer::embedder::{Embed, EmbedError, Embedder, MockEmbedder, EMBEDDING_DIMS};
use crate::storage;

#[test]
fn query_error_from_embed_error() {
    let ee = EmbedError::ApiKeyNotSet;
    let qe: QueryError = ee.into();
    assert!(qe.to_string().contains("GEMINI_API_KEY"));
}

#[tokio::test]
async fn search_with_mock_embedder() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let mut emb = vec![0.0_f32; EMBEDDING_DIMS as usize];
    emb[0] = 1.0;
    storage::insert_chunk(
        &conn,
        "src/Button.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("Button"),
            content: "function Button() { return <div/>; }",
            start_line: 1, end_line: 3,
        },
        "hash1",
        &emb,
    ).unwrap();

    let conn = Arc::new(Mutex::new(conn));
    let outcome = search(conn, &MockEmbedder, "button", 10, 0).await.unwrap();
    assert!(!outcome.degraded);
    assert_eq!(outcome.results.len(), 1);
    assert_eq!(outcome.results[0].chunk.name.as_deref(), Some("Button"));
}

#[test]
fn extract_keywords_basic() {
    let kw = extract_keywords("streaming chat hooks");
    assert_eq!(kw, vec!["streaming", "chat", "hooks", "stream", "hook"]);
}

#[test]
fn extract_keywords_filters_stopwords() {
    let kw = extract_keywords("the form for validation");
    assert_eq!(kw, vec!["form", "validation"]);
}

#[test]
fn extract_keywords_filters_short_tokens() {
    let kw = extract_keywords("a UI in React");
    assert_eq!(kw, vec!["ui", "react"]);
}

#[test]
fn extract_keywords_lowercases() {
    let kw = extract_keywords("UseAuth Component");
    assert_eq!(kw, vec!["useauth", "use", "auth", "component"]);
}

#[test]
fn extract_keywords_no_stem_for_non_gerunds() {
    let kw = extract_keywords("string parsing");
    assert!(kw.contains(&"string".to_string()), "string should remain");
    assert!(!kw.contains(&"str".to_string()), "str should not appear");
}

#[test]
fn extract_keywords_no_stem_for_non_plurals() {
    let kw = extract_keywords("class definition");
    assert!(kw.contains(&"class".to_string()), "class should remain");
    assert!(!kw.contains(&"clas".to_string()), "clas should not appear");
}

#[test]
fn extract_keywords_short_ing_not_stemmed() {
    let kw = extract_keywords("thing setup");
    assert_eq!(kw, vec!["thing", "setup"]);
}

#[test]
fn split_identifier_camel_case() {
    assert_eq!(split_identifier("useChat"), vec!["use", "Chat"]);
    assert_eq!(split_identifier("DataTable"), vec!["Data", "Table"]);
}

#[test]
fn split_identifier_acronym() {
    assert_eq!(split_identifier("HTMLElement"), vec!["HTML", "Element"]);
    assert_eq!(split_identifier("getXMLParser"), vec!["get", "XML", "Parser"]);
    assert_eq!(split_identifier("UIMessage"), vec!["UI", "Message"]);
}

#[test]
fn split_identifier_kebab_and_snake() {
    assert_eq!(split_identifier("data-table"), vec!["data", "table"]);
    assert_eq!(split_identifier("get_value"), vec!["get", "value"]);
    assert_eq!(split_identifier("use-chat"), vec!["use", "chat"]);
}

#[test]
fn split_identifier_single_word() {
    assert_eq!(split_identifier("stream"), vec!["stream"]);
    assert_eq!(split_identifier("chat"), vec!["chat"]);
}

#[test]
fn extract_keywords_expands_camel_case() {
    let kw = extract_keywords("useChat");
    assert!(kw.contains(&"usechat".to_string()), "whole token: {kw:?}");
    assert!(kw.contains(&"use".to_string()), "part 'use': {kw:?}");
    assert!(kw.contains(&"chat".to_string()), "part 'chat': {kw:?}");
}

#[test]
fn extract_keywords_expands_kebab_case() {
    let kw = extract_keywords("data-table component");
    assert!(kw.contains(&"data-table".to_string()));
    assert!(kw.contains(&"data".to_string()));
    assert!(kw.contains(&"table".to_string()));
    assert!(kw.contains(&"component".to_string()));
}

#[test]
fn extract_keywords_no_duplicate_parts() {
    let kw = extract_keywords("DataTable");
    let data_count = kw.iter().filter(|k| *k == "data").count();
    assert_eq!(data_count, 1, "no duplicates: {kw:?}");
}

#[test]
fn extract_keywords_normal_query_unchanged() {
    let kw = extract_keywords("streaming chat hooks");
    assert_eq!(kw, vec!["streaming", "chat", "hooks", "stream", "hook"]);
}

#[test]
fn extract_type_hints_hooks() {
    let hints = extract_type_hints("streaming hooks");
    assert_eq!(hints, vec![storage::ChunkType::Hook]);
}

#[test]
fn extract_type_hints_multiple() {
    let hints = extract_type_hints("component styles");
    assert!(hints.contains(&storage::ChunkType::Component));
    assert!(hints.contains(&storage::ChunkType::CssRule));
    assert_eq!(hints.len(), 2);
}

#[test]
fn extract_type_hints_no_match() {
    let hints = extract_type_hints("streaming chat completion");
    assert!(hints.is_empty());
}

#[test]
fn extract_type_hints_singular_and_plural() {
    assert_eq!(extract_type_hints("hook"), vec![storage::ChunkType::Hook]);
    assert_eq!(extract_type_hints("hooks"), vec![storage::ChunkType::Hook]);
    assert_eq!(extract_type_hints("test"), vec![storage::ChunkType::TestCase]);
    assert_eq!(extract_type_hints("spec"), vec![storage::ChunkType::TestCase]);
}

#[tokio::test]
async fn search_fallback_merges_vector_and_name_results() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let mut emb = vec![0.0_f32; EMBEDDING_DIMS as usize];
    emb[0] = 1.0;
    storage::insert_chunk(
        &conn, "src/A.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("AuthForm"),
            content: "function AuthForm() {}",
            start_line: 1, end_line: 3,
        },
        "h1", &emb,
    ).unwrap();

    storage::replace_file_chunks_only(
        &conn, "src/B.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Hook,
            name: Some("useAuth"),
            content: "function useAuth() {}",
            start_line: 1, end_line: 3,
        }],
        "h2", "", &[],
    ).unwrap();

    let conn = Arc::new(Mutex::new(conn));
    let results = search(conn, &MockEmbedder, "auth hook", 5, 0).await.unwrap().results;

    assert!(results.len() >= 2, "expected at least 2 results (vector + fallback), got {}", results.len());

    let names: Vec<&str> = results.iter().filter_map(|r| r.chunk.name.as_deref()).collect();
    assert!(names.contains(&"AuthForm"), "expected AuthForm from vector: {names:?}");
    assert!(names.contains(&"useAuth"), "expected useAuth from fallback: {names:?}");
    assert_eq!(results[0].match_source, storage::MatchSource::Semantic);
}

#[tokio::test]
async fn search_deduplicates_vector_and_name_results() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let mut emb = vec![0.0_f32; EMBEDDING_DIMS as usize];
    emb[0] = 1.0;
    storage::insert_chunk(
        &conn, "src/A.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Hook,
            name: Some("useAuth"),
            content: "function useAuth() {}",
            start_line: 1, end_line: 3,
        },
        "h1", &emb,
    ).unwrap();

    let conn = Arc::new(Mutex::new(conn));
    let results = search(conn, &MockEmbedder, "auth", 5, 0).await.unwrap().results;

    let auth_count = results.iter().filter(|r| r.chunk.name.as_deref() == Some("useAuth")).count();
    assert_eq!(auth_count, 1, "useAuth should appear exactly once");
    assert_eq!(results[0].match_source, storage::MatchSource::Semantic);
}

fn make_result(
    file_path: &str,
    name: &str,
    chunk_type: storage::ChunkType,
    distance: f32,
    match_source: storage::MatchSource,
) -> storage::SearchResult {
    let score = match match_source {
        storage::MatchSource::Semantic => 1.0 / (1.0 + distance),
        storage::MatchSource::NameMatch => 0.5,
        storage::MatchSource::ContentMatch => 0.45,
    };
    storage::SearchResult {
        chunk: storage::Chunk {
            file_path: file_path.to_string(),
            chunk_type,
            name: Some(name.to_string()),
            content: format!("function {name}() {{}}"),
            start_line: 1,
            end_line: 3,
        },
        chunk_id: None,
        distance,
        match_source,
        score,
    }
}

#[test]
fn rerank_semantic_base_score() {
    let mut results = vec![
        make_result("src/A.tsx", "A", storage::ChunkType::Component, 0.5, storage::MatchSource::Semantic),
    ];
    rerank(&mut results, &RerankContext::default(), &HashMap::new());
    let expected = 1.0 / (1.0 + 0.5);
    assert!((results[0].score - expected).abs() < 1e-6,
        "expected {expected}, got {}", results[0].score);
}

#[test]
fn rerank_name_match_base_score() {
    let kw = vec!["auth".to_string()];
    let mut results = vec![
        make_result("src/A.tsx", "useAuth", storage::ChunkType::Hook, f32::INFINITY, storage::MatchSource::NameMatch),
    ];
    // 1/1 keywords match → base = 0.40 + 0.20 = 0.60, + NAME_MATCH_BONUS = 0.65
    rerank(&mut results, &RerankContext { keywords: &kw, ..Default::default() }, &HashMap::new());
    assert!((results[0].score - 0.65).abs() < 1e-6,
        "expected 0.65, got {}", results[0].score);
}

#[test]
fn rerank_sorts_by_score_descending() {
    let mut results = vec![
        make_result("src/B.tsx", "B", storage::ChunkType::Component, 0.5, storage::MatchSource::Semantic),
        make_result("src/A.tsx", "A", storage::ChunkType::Component, 0.1, storage::MatchSource::Semantic),
    ];
    rerank(&mut results, &RerankContext::default(), &HashMap::new());
    assert!(results[0].score >= results[1].score,
        "expected descending: {} >= {}", results[0].score, results[1].score);
    assert_eq!(results[0].chunk.name.as_deref(), Some("A"));
}

#[test]
fn rerank_type_hint_bonus() {
    let mut with_hint = vec![
        make_result("src/A.tsx", "useAuth", storage::ChunkType::Hook, 0.5, storage::MatchSource::Semantic),
    ];
    let mut without_hint = with_hint.clone();

    rerank(&mut with_hint, &RerankContext { type_hints: &[storage::ChunkType::Hook], ..Default::default() }, &HashMap::new());
    rerank(&mut without_hint, &RerankContext::default(), &HashMap::new());

    let diff = with_hint[0].score - without_hint[0].score;
    assert!((diff - 0.03).abs() < 1e-6,
        "type_hint bonus should be +0.03, got diff {diff}");
}

#[test]
fn rerank_import_rank_bonus() {
    let mut results = vec![
        make_result("src/popular.tsx", "Popular", storage::ChunkType::Component, 0.3, storage::MatchSource::Semantic),
        make_result("src/unpopular.tsx", "Unpopular", storage::ChunkType::Component, 0.3, storage::MatchSource::Semantic),
    ];
    let import_counts = HashMap::from([
        ("src/popular.tsx".to_string(), 10u32),
        ("src/unpopular.tsx".to_string(), 1u32),
    ]);
    rerank(&mut results, &RerankContext::default(), &import_counts);

    let popular = results.iter().find(|r| r.chunk.name.as_deref() == Some("Popular")).unwrap();
    let unpopular = results.iter().find(|r| r.chunk.name.as_deref() == Some("Unpopular")).unwrap();
    assert!(popular.score > unpopular.score,
        "popular should rank higher: {} > {}", popular.score, unpopular.score);
}

#[test]
fn rerank_test_path_penalty() {
    let mut test_result = vec![
        make_result("src/__tests__/A.test.tsx", "A", storage::ChunkType::Component, 0.3, storage::MatchSource::Semantic),
    ];
    let mut normal_result = vec![
        make_result("src/A.tsx", "A", storage::ChunkType::Component, 0.3, storage::MatchSource::Semantic),
    ];
    let ctx = RerankContext::default();
    rerank(&mut test_result, &ctx, &HashMap::new());
    rerank(&mut normal_result, &ctx, &HashMap::new());

    let diff = normal_result[0].score - test_result[0].score;
    assert!((diff - 0.05).abs() < 1e-6,
        "test penalty should be -0.05, got diff {diff}");
}

#[test]
fn rerank_test_query_exempts_penalty() {
    let mut results = vec![
        make_result("src/__tests__/A.test.tsx", "testA", storage::ChunkType::TestCase, 0.3, storage::MatchSource::Semantic),
    ];
    let base = 1.0 / (1.0 + 0.3_f32) + 0.03; // base + type_hint bonus (TestCase matches)
    rerank(&mut results, &RerankContext { type_hints: &[storage::ChunkType::TestCase], ..Default::default() }, &HashMap::new());
    assert!((results[0].score - base).abs() < 1e-6,
        "TestCase query should exempt test penalty: expected {base}, got {}", results[0].score);
}

#[test]
fn rerank_name_match_all_bonuses() {
    let kw = vec!["auth".to_string()];
    let mut results = vec![
        make_result("src/useAuth.tsx", "useAuth", storage::ChunkType::Hook, f32::INFINITY, storage::MatchSource::NameMatch),
    ];
    let import_counts = HashMap::from([("src/useAuth.tsx".to_string(), 10u32)]);
    // 1/1 match → base=0.60, + NAME_MATCH=0.05, + TYPE_HINT=0.03, + IMPORT=0.03 = 0.71
    rerank(&mut results, &RerankContext { type_hints: &[storage::ChunkType::Hook], keywords: &kw, ..Default::default() }, &import_counts);
    assert!((results[0].score - 0.71).abs() < 1e-6,
        "expected 0.71, got {}", results[0].score);
}

#[test]
fn rerank_combined_bonuses_semantic() {
    let mut results = vec![
        make_result("src/useAuth.tsx", "useAuth", storage::ChunkType::Hook, 0.1, storage::MatchSource::Semantic),
    ];
    let import_counts = HashMap::from([("src/useAuth.tsx".to_string(), 10u32)]);
    rerank(&mut results, &RerankContext { type_hints: &[storage::ChunkType::Hook], ..Default::default() }, &import_counts);

    let base = 1.0 / (1.0 + 0.1_f32);
    let expected = base + TYPE_HINT_BONUS + IMPORT_RANK_BONUS;
    assert!((results[0].score - expected).abs() < 1e-6,
        "combined semantic: expected {expected}, got {}", results[0].score);
}

#[test]
fn rerank_combined_score_can_exceed_one() {
    let mut results = vec![
        make_result("src/useAuth.tsx", "useAuth", storage::ChunkType::Hook, 0.0, storage::MatchSource::Semantic),
    ];
    let import_counts = HashMap::from([("src/useAuth.tsx".to_string(), 10u32)]);
    rerank(&mut results, &RerankContext { type_hints: &[storage::ChunkType::Hook], ..Default::default() }, &import_counts);

    assert!(results[0].score > 1.0,
        "score can exceed 1.0 with all bonuses: {}", results[0].score);
    let expected = 1.0 + TYPE_HINT_BONUS + IMPORT_RANK_BONUS;
    assert!((results[0].score - expected).abs() < 1e-6,
        "expected {expected}, got {}", results[0].score);
}

#[tokio::test]
async fn search_returns_results_sorted_by_score() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let mut emb = vec![0.0_f32; EMBEDDING_DIMS as usize];
    emb[0] = 1.0;
    storage::insert_chunk(
        &conn, "src/A.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("AuthForm"),
            content: "function AuthForm() {}",
            start_line: 1, end_line: 3,
        },
        "h1", &emb,
    ).unwrap();

    storage::replace_file_chunks_only(
        &conn, "src/B.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Hook,
            name: Some("useAuth"),
            content: "function useAuth() {}",
            start_line: 1, end_line: 3,
        }],
        "h2", "", &[],
    ).unwrap();

    let conn = Arc::new(Mutex::new(conn));
    let results = search(conn, &MockEmbedder, "auth hook", 5, 0).await.unwrap().results;

    for w in results.windows(2) {
        assert!(w[0].score >= w[1].score,
            "results should be sorted by score: {} >= {}", w[0].score, w[1].score);
    }
}

#[test]
fn semantic_confidence_at_full_coverage() {
    let c = semantic_confidence(1.0);
    assert!((c - 1.0).abs() < 1e-6, "full coverage → 1.0, got {c}");
}

#[test]
fn semantic_confidence_at_zero_coverage() {
    let c = semantic_confidence(0.0);
    assert!((c - 0.4).abs() < 1e-6, "zero coverage → 0.4, got {c}");
}

#[test]
fn semantic_confidence_at_low_coverage() {
    // 2% coverage: sqrt(0.02) * 2.0 + 0.4 ≈ 0.683
    let c = semantic_confidence(0.02);
    assert!(c > 0.6 && c < 0.75, "2% coverage → ~0.68, got {c}");
}

#[test]
fn semantic_confidence_reaches_one_by_9_percent() {
    // sqrt(0.09) * 2.0 + 0.4 = 1.0
    let c = semantic_confidence(0.09);
    assert!((c - 1.0).abs() < 0.01, "9% coverage → ~1.0, got {c}");
}

#[test]
fn rerank_low_coverage_dampens_semantic() {
    let mut results = vec![
        make_result("src/A.tsx", "A", storage::ChunkType::Component, 0.5, storage::MatchSource::Semantic),
    ];
    rerank(&mut results, &RerankContext { embed_coverage: 0.02, ..Default::default() }, &HashMap::new());
    let confidence = semantic_confidence(0.02);
    let expected = 1.0 / (1.0 + 0.5) * confidence;
    assert!((results[0].score - expected).abs() < 1e-6,
        "expected {expected}, got {}", results[0].score);
}

#[test]
fn rerank_low_coverage_name_match_beats_semantic() {
    let kw = vec!["chat".to_string()];
    let mut results = vec![
        // At 2% coverage: 1/(1+0.45) * 0.68 ≈ 0.47
        make_result("src/A.tsx", "Unrelated", storage::ChunkType::Component, 0.45, storage::MatchSource::Semantic),
        // Name match: 0.40 + 0.20*(1/1) + 0.05 = 0.65 (unaffected by coverage)
        make_result("src/B.tsx", "useChat", storage::ChunkType::Hook, f32::INFINITY, storage::MatchSource::NameMatch),
    ];
    rerank(&mut results, &RerankContext { keywords: &kw, embed_coverage: 0.02, ..Default::default() }, &HashMap::new());
    assert_eq!(results[0].chunk.name.as_deref(), Some("useChat"),
        "name match should rank first at low coverage");
    assert!(results[0].score > results[1].score,
        "name ({}) should beat semantic ({}) at 2% coverage",
        results[0].score, results[1].score);
}

#[test]
fn content_match_scores_from_content_body() {
    let kw = vec!["validation".to_string(), "form".to_string()];
    let result = storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/submit.ts".to_string(),
            chunk_type: storage::ChunkType::Other,
            name: Some("submitHandler".to_string()),
            content: "function submitHandler(form: Form) { if (!validation(form)) return; }".to_string(),
            start_line: 1,
            end_line: 3,
        },
        chunk_id: None,
        distance: f32::INFINITY,
        match_source: storage::MatchSource::ContentMatch,
        score: 0.0,
    };
    let ratio = keyword_hit_ratio(&result, &kw, &[], true);
    assert!((ratio - 1.0).abs() < 1e-6, "both keywords match → ratio=1.0, got {ratio}");

    let ratio_no_content = keyword_hit_ratio(&result, &kw, &[], false);
    assert!((ratio_no_content).abs() < 1e-6, "name/path don't match → ratio=0.0, got {ratio_no_content}");
}

use crate::indexer::embedder::FailingEmbedder;

#[tokio::test]
async fn search_degrades_on_embed_failure() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    storage::replace_file_chunks_only(
        &conn, "src/Auth.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("AuthForm"),
            content: "function AuthForm() { return <form/>; }",
            start_line: 1, end_line: 3,
        }],
        "h1", "", &[],
    ).unwrap();

    let conn = Arc::new(Mutex::new(conn));
    let outcome = search(conn, &FailingEmbedder::query_only(429, "rate limited"), "auth", 10, 0).await.unwrap();
    assert!(outcome.degraded, "should be degraded when embed fails");
    assert!(!outcome.results.is_empty(), "should still return text-matched results");
    assert_eq!(outcome.results[0].chunk.name.as_deref(), Some("AuthForm"));
    assert_ne!(outcome.results[0].match_source, storage::MatchSource::Semantic);
}

#[tokio::test]
async fn search_degrades_on_api_key_not_set() {
    struct NoKeyEmbedder;
    impl Embed for NoKeyEmbedder {
        fn embed_query<'a>(
            &'a self,
            _text: &'a str,
        ) -> Pin<Box<dyn Future<Output = Result<Vec<f32>, EmbedError>> + Send + 'a>> {
            Box::pin(async { Err(EmbedError::ApiKeyNotSet) })
        }
        fn embed_documents<'a>(
            &'a self,
            _texts: &'a [String],
        ) -> Pin<Box<dyn Future<Output = Result<Vec<Vec<f32>>, EmbedError>> + Send + 'a>> {
            Box::pin(async { Ok(vec![]) })
        }
    }

    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let outcome = search(conn, &NoKeyEmbedder, "test", 10, 0).await.unwrap();
    assert!(outcome.degraded, "should degrade to FTS5 when API key not set");
}

#[tokio::test]
#[ignore = "requires GEMINI_API_KEY"]
async fn search_returns_results() {
    let embedder = Embedder::from_env(reqwest::Client::new()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let embedding = embedder.embed_query("Button component").await.unwrap();
    storage::insert_chunk(
        &conn,
        "src/Button.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("Button"),
            content: "function Button() { return <div/>; }",
            start_line: 1, end_line: 3,
        },
        "hash1",
        &embedding,
    ).unwrap();

    let conn = Arc::new(Mutex::new(conn));
    let outcome = search(conn, &embedder, "button", 10, 0).await.unwrap();
    assert!(!outcome.results.is_empty(), "expected at least one result");
}

#[test]
fn search_pipeline_text_only_returns_name_matches() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    storage::replace_file_chunks_only(
        &conn, "src/Auth.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("AuthForm"),
            content: "function AuthForm() { return <form/>; }",
            start_line: 1, end_line: 3,
        }],
        "h1", "", &[],
    ).unwrap();

    let results = search_pipeline(&conn, "auth", None, 10, 0).unwrap();
    assert!(!results.is_empty(), "text-only pipeline should find name matches");
    assert_eq!(results[0].chunk.name.as_deref(), Some("AuthForm"));
    assert_ne!(results[0].match_source, storage::MatchSource::Semantic);
}

#[test]
fn search_pipeline_with_embedding_returns_semantic() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let mut emb = vec![0.0_f32; EMBEDDING_DIMS as usize];
    emb[0] = 1.0;
    storage::insert_chunk(
        &conn, "src/Button.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("Button"),
            content: "function Button() { return <div/>; }",
            start_line: 1, end_line: 3,
        },
        "h1", &emb,
    ).unwrap();

    let results = search_pipeline(&conn, "button", Some(&emb), 10, 0).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].match_source, storage::MatchSource::Semantic);
}

#[test]
fn search_pipeline_caps_per_file() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let chunks: Vec<storage::NewChunk> = (1..=5).map(|i| storage::NewChunk {
        chunk_type: &storage::ChunkType::Other,
        name: Some("fn"),
        content: "function fn() {}",
        start_line: i,
        end_line: i + 2,
    }).collect();
    storage::replace_file_chunks_only(
        &conn, "src/big.ts", &chunks, "h1", "", &[],
    ).unwrap();

    let results = search_pipeline(&conn, "fn", None, 10, 0).unwrap();
    let file_count = results.iter().filter(|r| r.chunk.file_path == "src/big.ts").count();
    assert!(file_count <= MAX_RESULTS_PER_FILE,
        "expected at most {MAX_RESULTS_PER_FILE} per file, got {file_count}");
}
