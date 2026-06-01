use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use bytemuck::cast_slice;
use rurico::embed::{Embedder, FailingEmbedder, MockEmbedder, ModelId};
use rurico::reranker::{MockReranker, RankedResult, Rerank, RerankerError};
use tempfile::tempdir;

use super::rank::{
    IMPORT_RANK_BONUS, MAX_RESULTS_PER_FILE, TYPE_HINT_BONUS, is_test_path, keyword_hit_ratio,
    semantic_confidence,
};
use super::*;
use crate::storage;

fn test_embedding() -> Vec<f32> {
    let mut emb = vec![0.0_f32; storage::EMBEDDING_DIMS];
    emb[0] = 1.0;
    emb
}

// T-252: search_with_mock_embedder
#[test]
fn search_with_mock_embedder() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let emb = test_embedding();
    storage::insert_chunk(
        &conn,
        "src/Button.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("Button"),
            content: "function Button() { return <div/>; }",
            start_line: 1,
            end_line: 3,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "hash1",
        &storage::ce(emb.clone()),
        None,
    )
    .unwrap();

    let conn = Arc::new(Mutex::new(conn));
    let outcome = search(
        &conn,
        &MockEmbedder::default(),
        "button",
        10,
        0,
        None,
        &[],
        false,
    )
    .unwrap();
    assert!(!outcome.degraded);
    assert_eq!(outcome.results.len(), 1);
    assert_eq!(outcome.results[0].chunk.name.as_deref(), Some("Button"));
}

// T-253: extract_keywords_basic
#[test]
fn extract_keywords_basic() {
    let kw = extract_keywords("streaming chat hooks");
    assert_eq!(kw, vec!["streaming", "chat", "hooks", "stream", "hook"]);
}

// T-254: extract_keywords_filters_stopwords
#[test]
fn extract_keywords_filters_stopwords() {
    let kw = extract_keywords("the form for validation");
    assert_eq!(kw, vec!["form", "validation"]);
}

// T-255: extract_keywords_filters_short_tokens
#[test]
fn extract_keywords_filters_short_tokens() {
    let kw = extract_keywords("a UI in React");
    assert_eq!(kw, vec!["ui", "react"]);
}

// T-256: extract_keywords_lowercases
#[test]
fn extract_keywords_lowercases() {
    let kw = extract_keywords("UseAuth Component");
    assert_eq!(kw, vec!["useauth", "use", "auth", "component"]);
}

// T-257: extract_keywords_no_stem_for_non_gerunds
#[test]
fn extract_keywords_no_stem_for_non_gerunds() {
    let kw = extract_keywords("string parsing");
    assert!(kw.contains(&"string".to_owned()), "string should remain");
    assert!(!kw.contains(&"str".to_owned()), "str should not appear");
}

// T-258: extract_keywords_no_stem_for_non_plurals
#[test]
fn extract_keywords_no_stem_for_non_plurals() {
    let kw = extract_keywords("class definition");
    assert!(kw.contains(&"class".to_owned()), "class should remain");
    assert!(!kw.contains(&"clas".to_owned()), "clas should not appear");
}

// T-259: extract_keywords_short_ing_not_stemmed
#[test]
fn extract_keywords_short_ing_not_stemmed() {
    let kw = extract_keywords("thing setup");
    assert_eq!(kw, vec!["thing", "setup"]);
}

// T-260: extract_keywords_expands_camel_case
#[test]
fn extract_keywords_expands_camel_case() {
    let kw = extract_keywords("useChat");
    assert!(kw.contains(&"usechat".to_owned()), "whole token: {kw:?}");
    assert!(kw.contains(&"use".to_owned()), "part 'use': {kw:?}");
    assert!(kw.contains(&"chat".to_owned()), "part 'chat': {kw:?}");
}

// T-261: extract_keywords_expands_kebab_case
#[test]
fn extract_keywords_expands_kebab_case() {
    let kw = extract_keywords("data-table component");
    assert!(kw.contains(&"data-table".to_owned()));
    assert!(kw.contains(&"data".to_owned()));
    assert!(kw.contains(&"table".to_owned()));
    assert!(kw.contains(&"component".to_owned()));
}

// T-262: extract_keywords_no_duplicate_parts
#[test]
fn extract_keywords_no_duplicate_parts() {
    let kw = extract_keywords("DataTable");
    let data_count = kw.iter().filter(|k| *k == "data").count();
    assert_eq!(data_count, 1, "no duplicates: {kw:?}");
}

// T-263: extract_keywords_cjk_chars_counted_not_bytes
#[test]
fn extract_keywords_cjk_chars_counted_not_bytes() {
    let kw = extract_keywords("認証");
    assert!(
        kw.contains(&"認証".to_owned()),
        "2-char CJK token should be kept: {kw:?}"
    );

    // Single CJK char should be filtered (chars().count() == 1 < 2)
    let kw_single = extract_keywords("認 認証フロー");
    assert!(
        !kw_single.contains(&"認".to_owned()),
        "single CJK char should be filtered by chars().count(): {kw_single:?}"
    );
    assert!(
        kw_single.contains(&"認証フロー".to_owned()),
        "multi-char CJK token should be kept: {kw_single:?}"
    );
}

// T-264: extract_type_hints_hooks
#[test]
fn extract_type_hints_hooks() {
    let hints = extract_type_hints("streaming hooks");
    assert_eq!(hints, vec![storage::ChunkType::Hook]);
}

// T-265: extract_type_hints_multiple
#[test]
fn extract_type_hints_multiple() {
    let hints = extract_type_hints("component styles");
    assert!(hints.contains(&storage::ChunkType::Component));
    assert!(hints.contains(&storage::ChunkType::CssRule));
    assert_eq!(hints.len(), 2);
}

// T-266: extract_type_hints_no_match
#[test]
fn extract_type_hints_no_match() {
    let hints = extract_type_hints("streaming chat completion");
    assert!(hints.is_empty());
}

// T-267: extract_type_hints_singular_and_plural
#[test]
fn extract_type_hints_singular_and_plural() {
    assert_eq!(extract_type_hints("hook"), vec![storage::ChunkType::Hook]);
    assert_eq!(extract_type_hints("hooks"), vec![storage::ChunkType::Hook]);
    assert_eq!(
        extract_type_hints("test"),
        vec![storage::ChunkType::TestCase]
    );
    assert_eq!(
        extract_type_hints("spec"),
        vec![storage::ChunkType::TestCase]
    );
}

// T-268: search_fallback_merges_vector_and_name_results
#[test]
fn search_fallback_merges_vector_and_name_results() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let emb = test_embedding();
    storage::insert_chunk(
        &conn,
        "src/A.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("AuthForm"),
            content: "function AuthForm() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb.clone()),
        None,
    )
    .unwrap();

    storage::replace_file_chunks_only(
        &conn,
        "src/B.tsx",
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
    let results = search(
        &conn,
        &MockEmbedder::default(),
        "auth",
        5,
        0,
        None,
        &[],
        false,
    )
    .unwrap()
    .results;

    assert!(
        results.len() >= 2,
        "expected at least 2 results (vector + fallback), got {}",
        results.len()
    );

    let names: Vec<&str> = results
        .iter()
        .filter_map(|r| r.chunk.name.as_deref())
        .collect();
    assert!(
        names.contains(&"AuthForm"),
        "expected AuthForm from vector: {names:?}"
    );
    assert!(
        names.contains(&"useAuth"),
        "expected useAuth from fallback: {names:?}"
    );
    let sources: Vec<_> = results.iter().map(|r| r.match_source).collect();
    assert!(
        sources.contains(&storage::MatchSource::Semantic),
        "expected Semantic result in set: {sources:?}"
    );
}

// T-269: search_deduplicates_vector_and_name_results
#[test]
fn search_deduplicates_vector_and_name_results() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let emb = test_embedding();
    storage::insert_chunk(
        &conn,
        "src/A.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Hook,
            name: Some("useAuth"),
            content: "function useAuth() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb.clone()),
        None,
    )
    .unwrap();

    let conn = Arc::new(Mutex::new(conn));
    let results = search(
        &conn,
        &MockEmbedder::default(),
        "auth",
        5,
        0,
        None,
        &[],
        false,
    )
    .unwrap()
    .results;

    let auth_count = results
        .iter()
        .filter(|r| r.chunk.name.as_deref() == Some("useAuth"))
        .count();
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
        storage::MatchSource::Fts => 0.45,
    };
    storage::SearchResult {
        chunk: storage::Chunk {
            file_path: file_path.to_owned(),
            chunk_type,
            name: Some(name.to_owned()),
            content: format!("function {name}() {{}}"),
            start_line: 1,
            end_line: 3,
            parent_chunk_id: None,
            source_kind: None,
            injection_flags: None,
        },
        chunk_id: None,
        distance,
        match_source,
        score,
    }
}

// T-270: rerank_semantic_base_score
#[test]
fn rerank_semantic_base_score() {
    let mut results = vec![make_result(
        "src/A.tsx",
        "A",
        storage::ChunkType::Component,
        0.5,
        storage::MatchSource::Semantic,
    )];
    rerank(&mut results, &RerankContext::default(), &HashMap::new());
    let expected = 1.0 / (1.0 + 0.5);
    assert!(
        (results[0].score - expected).abs() < 1e-6,
        "expected {expected}, got {}",
        results[0].score
    );
}

// T-271: rerank_fts_base_score_passthrough
#[test]
fn rerank_fts_base_score_passthrough() {
    let kw = vec!["auth".to_owned()];
    let mut results = vec![make_result(
        "src/A.tsx",
        "useAuth",
        storage::ChunkType::Hook,
        f32::INFINITY,
        storage::MatchSource::Fts,
    )];
    rerank(
        &mut results,
        &RerankContext {
            keywords: &kw,
            ..Default::default()
        },
        &HashMap::new(),
    );
    assert!(
        (results[0].score - 0.45).abs() < 1e-6,
        "expected 0.45, got {}",
        results[0].score
    );
}

// T-272: rerank_sorts_by_score_descending
#[test]
fn rerank_sorts_by_score_descending() {
    let mut results = vec![
        make_result(
            "src/B.tsx",
            "B",
            storage::ChunkType::Component,
            0.5,
            storage::MatchSource::Semantic,
        ),
        make_result(
            "src/A.tsx",
            "A",
            storage::ChunkType::Component,
            0.1,
            storage::MatchSource::Semantic,
        ),
    ];
    rerank(&mut results, &RerankContext::default(), &HashMap::new());
    assert!(
        results[0].score >= results[1].score,
        "expected descending: {} >= {}",
        results[0].score,
        results[1].score
    );
    assert_eq!(results[0].chunk.name.as_deref(), Some("A"));
}

// T-273: rerank_type_hint_bonus
#[test]
fn rerank_type_hint_bonus() {
    let mut with_hint = vec![make_result(
        "src/A.tsx",
        "useAuth",
        storage::ChunkType::Hook,
        0.5,
        storage::MatchSource::Semantic,
    )];
    let mut without_hint = with_hint.clone();

    rerank(
        &mut with_hint,
        &RerankContext {
            type_hints: &[storage::ChunkType::Hook],
            ..Default::default()
        },
        &HashMap::new(),
    );
    rerank(
        &mut without_hint,
        &RerankContext::default(),
        &HashMap::new(),
    );

    let diff = with_hint[0].score - without_hint[0].score;
    assert!(
        (diff - 0.03).abs() < 1e-6,
        "type_hint bonus should be +0.03, got diff {diff}"
    );
}

// T-274: rerank_import_rank_bonus
#[test]
fn rerank_import_rank_bonus() {
    let mut results = vec![
        make_result(
            "src/popular.tsx",
            "Popular",
            storage::ChunkType::Component,
            0.3,
            storage::MatchSource::Semantic,
        ),
        make_result(
            "src/unpopular.tsx",
            "Unpopular",
            storage::ChunkType::Component,
            0.3,
            storage::MatchSource::Semantic,
        ),
    ];
    let import_counts = HashMap::from([
        ("src/popular.tsx".to_owned(), 10u32),
        ("src/unpopular.tsx".to_owned(), 1u32),
    ]);
    rerank(&mut results, &RerankContext::default(), &import_counts);

    let popular = results
        .iter()
        .find(|r| r.chunk.name.as_deref() == Some("Popular"))
        .unwrap();
    let unpopular = results
        .iter()
        .find(|r| r.chunk.name.as_deref() == Some("Unpopular"))
        .unwrap();
    assert!(
        popular.score > unpopular.score,
        "popular should rank higher: {} > {}",
        popular.score,
        unpopular.score
    );
}

// T-275: rerank_test_path_penalty
#[test]
fn rerank_test_path_penalty() {
    let mut test_result = vec![make_result(
        "src/__tests__/A.test.tsx",
        "A",
        storage::ChunkType::Component,
        0.3,
        storage::MatchSource::Semantic,
    )];
    let mut normal_result = vec![make_result(
        "src/A.tsx",
        "A",
        storage::ChunkType::Component,
        0.3,
        storage::MatchSource::Semantic,
    )];
    let ctx = RerankContext::default();
    rerank(&mut test_result, &ctx, &HashMap::new());
    rerank(&mut normal_result, &ctx, &HashMap::new());

    let diff = normal_result[0].score - test_result[0].score;
    assert!(
        (diff - 0.05).abs() < 1e-6,
        "test penalty should be -0.05, got diff {diff}"
    );
}

// T-276: rerank_test_query_exempts_penalty
#[test]
fn rerank_test_query_exempts_penalty() {
    let mut results = vec![make_result(
        "src/__tests__/A.test.tsx",
        "testA",
        storage::ChunkType::TestCase,
        0.3,
        storage::MatchSource::Semantic,
    )];
    let base = 1.0 / (1.0 + 0.3_f32) + 0.03; // base + type_hint bonus (TestCase matches)
    rerank(
        &mut results,
        &RerankContext {
            type_hints: &[storage::ChunkType::TestCase],
            ..Default::default()
        },
        &HashMap::new(),
    );
    assert!(
        (results[0].score - base).abs() < 1e-6,
        "TestCase query should exempt test penalty: expected {base}, got {}",
        results[0].score
    );
}

// T-277: rerank_fts_all_bonuses
#[test]
fn rerank_fts_all_bonuses() {
    let kw = vec!["auth".to_owned()];
    let mut results = vec![make_result(
        "src/useAuth.tsx",
        "useAuth",
        storage::ChunkType::Hook,
        f32::INFINITY,
        storage::MatchSource::Fts,
    )];
    let import_counts = HashMap::from([("src/useAuth.tsx".to_owned(), 10u32)]);
    rerank(
        &mut results,
        &RerankContext {
            type_hints: &[storage::ChunkType::Hook],
            keywords: &kw,
            ..Default::default()
        },
        &import_counts,
    );
    assert!(
        (results[0].score - 0.51).abs() < 1e-6,
        "expected 0.51, got {}",
        results[0].score
    );
}

// T-278: rerank_combined_bonuses_semantic
#[test]
fn rerank_combined_bonuses_semantic() {
    let mut results = vec![make_result(
        "src/useAuth.tsx",
        "useAuth",
        storage::ChunkType::Hook,
        0.1,
        storage::MatchSource::Semantic,
    )];
    let import_counts = HashMap::from([("src/useAuth.tsx".to_owned(), 10u32)]);
    rerank(
        &mut results,
        &RerankContext {
            type_hints: &[storage::ChunkType::Hook],
            ..Default::default()
        },
        &import_counts,
    );

    let base = 1.0 / (1.0 + 0.1_f32);
    let expected = base + TYPE_HINT_BONUS + IMPORT_RANK_BONUS;
    assert!(
        (results[0].score - expected).abs() < 1e-6,
        "combined semantic: expected {expected}, got {}",
        results[0].score
    );
}

// T-279: rerank_combined_score_can_exceed_one
#[test]
fn rerank_combined_score_can_exceed_one() {
    let mut results = vec![make_result(
        "src/useAuth.tsx",
        "useAuth",
        storage::ChunkType::Hook,
        0.0,
        storage::MatchSource::Semantic,
    )];
    let import_counts = HashMap::from([("src/useAuth.tsx".to_owned(), 10u32)]);
    rerank(
        &mut results,
        &RerankContext {
            type_hints: &[storage::ChunkType::Hook],
            ..Default::default()
        },
        &import_counts,
    );

    assert!(
        results[0].score > 1.0,
        "score can exceed 1.0 with all bonuses: {}",
        results[0].score
    );
    let expected = 1.0 + TYPE_HINT_BONUS + IMPORT_RANK_BONUS;
    assert!(
        (results[0].score - expected).abs() < 1e-6,
        "expected {expected}, got {}",
        results[0].score
    );
}

// T-280: search_returns_results_sorted_by_score
#[test]
fn search_returns_results_sorted_by_score() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let emb = test_embedding();
    storage::insert_chunk(
        &conn,
        "src/A.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("AuthForm"),
            content: "function AuthForm() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb.clone()),
        None,
    )
    .unwrap();

    storage::replace_file_chunks_only(
        &conn,
        "src/B.tsx",
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
    let results = search(
        &conn,
        &MockEmbedder::default(),
        "auth hook",
        5,
        0,
        None,
        &[],
        false,
    )
    .unwrap()
    .results;

    for w in results.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "results should be sorted by score: {} >= {}",
            w[0].score,
            w[1].score
        );
    }
}

// T-281: semantic_confidence_at_full_coverage
#[test]
fn semantic_confidence_at_full_coverage() {
    let c = semantic_confidence(1.0);
    assert!((c - 1.0).abs() < 1e-6, "full coverage → 1.0, got {c}");
}

// T-282: semantic_confidence_at_zero_coverage
#[test]
fn semantic_confidence_at_zero_coverage() {
    let c = semantic_confidence(0.0);
    assert!((c - 0.4).abs() < 1e-6, "zero coverage → 0.4, got {c}");
}

// T-283: semantic_confidence_at_low_coverage
#[test]
fn semantic_confidence_at_low_coverage() {
    // 2% coverage: sqrt(0.02) * 2.0 + 0.4 ≈ 0.683
    let c = semantic_confidence(0.02);
    assert!(c > 0.6 && c < 0.75, "2% coverage → ~0.68, got {c}");
}

// T-284: semantic_confidence_reaches_one_by_9_percent
#[test]
fn semantic_confidence_reaches_one_by_9_percent() {
    // sqrt(0.09) * 2.0 + 0.4 = 1.0
    let c = semantic_confidence(0.09);
    assert!((c - 1.0).abs() < 0.01, "9% coverage → ~1.0, got {c}");
}

// T-285: rerank_low_coverage_dampens_semantic
#[test]
fn rerank_low_coverage_dampens_semantic() {
    let mut results = vec![make_result(
        "src/A.tsx",
        "A",
        storage::ChunkType::Component,
        0.5,
        storage::MatchSource::Semantic,
    )];
    rerank(
        &mut results,
        &RerankContext {
            embed_coverage: 0.02,
            ..Default::default()
        },
        &HashMap::new(),
    );
    let confidence = semantic_confidence(0.02);
    let expected = 1.0 / (1.0 + 0.5) * confidence;
    assert!(
        (results[0].score - expected).abs() < 1e-6,
        "expected {expected}, got {}",
        results[0].score
    );
}

// T-286: rerank_low_coverage_fts_beats_semantic
#[test]
fn rerank_low_coverage_fts_beats_semantic() {
    let kw = vec!["chat".to_owned()];
    let mut results = vec![
        // At 2% coverage: 1/(1+0.6) * 0.68 ≈ 0.43
        make_result(
            "src/A.tsx",
            "Unrelated",
            storage::ChunkType::Component,
            0.6,
            storage::MatchSource::Semantic,
        ),
        // Fts passthrough: 0.45 (unaffected by coverage)
        make_result(
            "src/B.tsx",
            "useChat",
            storage::ChunkType::Hook,
            f32::INFINITY,
            storage::MatchSource::Fts,
        ),
    ];
    rerank(
        &mut results,
        &RerankContext {
            keywords: &kw,
            embed_coverage: 0.02,
            ..Default::default()
        },
        &HashMap::new(),
    );
    assert_eq!(
        results[0].chunk.name.as_deref(),
        Some("useChat"),
        "FTS should rank first at low coverage"
    );
    assert!(
        results[0].score > results[1].score,
        "FTS ({}) should beat semantic ({}) at 2% coverage",
        results[0].score,
        results[1].score
    );
}

// T-287: content_match_scores_from_content_body
#[test]
fn content_match_scores_from_content_body() {
    let kw = vec!["validation".to_owned(), "form".to_owned()];
    let result = storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/submit.ts".to_owned(),
            chunk_type: storage::ChunkType::Other,
            name: Some("submitHandler".to_owned()),
            content: "function submitHandler(form: Form) { if (!validation(form)) return; }"
                .to_owned(),
            start_line: 1,
            end_line: 3,
            parent_chunk_id: None,
            source_kind: None,
            injection_flags: None,
        },
        chunk_id: None,
        distance: f32::INFINITY,
        match_source: storage::MatchSource::Fts,
        score: 0.0,
    };
    let ratio = keyword_hit_ratio(&result, &kw, &[], true);
    assert!(
        (ratio - 1.0).abs() < 1e-6,
        "both keywords match → ratio=1.0, got {ratio}"
    );

    let ratio_no_content = keyword_hit_ratio(&result, &kw, &[], false);
    assert!(
        (ratio_no_content).abs() < 1e-6,
        "name/path don't match → ratio=0.0, got {ratio_no_content}"
    );
}

// T-288: rerank_semantic_keyword_overlap_bonus
#[test]
fn rerank_semantic_keyword_overlap_bonus() {
    let kw = vec!["chat".to_owned()];
    let mut with_overlap = vec![make_result(
        "src/useChat.tsx",
        "useChat",
        storage::ChunkType::Hook,
        0.2,
        storage::MatchSource::Semantic,
    )];
    let mut without_overlap = vec![make_result(
        "src/stream.tsx",
        "processStream",
        storage::ChunkType::Other,
        0.2,
        storage::MatchSource::Semantic,
    )];
    let ctx = RerankContext {
        keywords: &kw,
        ..Default::default()
    };
    rerank(&mut with_overlap, &ctx, &HashMap::new());
    rerank(&mut without_overlap, &ctx, &HashMap::new());

    assert!(
        with_overlap[0].score > without_overlap[0].score,
        "semantic result matching keywords should score higher: {} > {}",
        with_overlap[0].score,
        without_overlap[0].score,
    );
    let base = 1.0 / (1.0 + 0.2_f32);
    assert!(
        (without_overlap[0].score - base).abs() < 1e-6,
        "no overlap → no bonus: expected {base}, got {}",
        without_overlap[0].score,
    );
}

// T-289: search_degrades_on_embed_failure
#[test]
fn search_degrades_on_embed_failure() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    storage::replace_file_chunks_only(
        &conn,
        "src/Auth.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("AuthForm"),
            content: "function AuthForm() { return <form/>; }",
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

    let conn = Arc::new(Mutex::new(conn));
    let outcome = search(
        &conn,
        &FailingEmbedder::query_only("embedding unavailable"),
        "auth",
        10,
        0,
        None,
        &[],
        false,
    )
    .unwrap();
    assert!(outcome.degraded, "should be degraded when embed fails");
    assert!(
        !outcome.results.is_empty(),
        "should still return text-matched results"
    );
    assert_eq!(outcome.results[0].chunk.name.as_deref(), Some("AuthForm"));
    assert_ne!(
        outcome.results[0].match_source,
        storage::MatchSource::Semantic
    );
}

// T-290: search_degrades_on_model_not_available
#[test]
fn search_degrades_on_model_not_available() {
    let embedder = FailingEmbedder::all_fail("embedder not available");

    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();
    let conn = Arc::new(Mutex::new(conn));

    let outcome = search(&conn, &embedder, "test", 10, 0, None, &[], false).unwrap();
    assert!(
        outcome.degraded,
        "should degrade to FTS5 when model not available"
    );
}

// ── Reranker integration tests (Spec #55 Phase 1) ──────────────────────────

/// Mock reranker that assigns scores based on document content.
///
/// Scores each document by looking up its content in a predefined map.
/// Documents not in the map receive 0.0. This lets tests control
/// cross-encoder ranking order without a live model.
struct ScriptedReranker {
    /// Map from document substring -> score. First matching entry wins.
    scores: Vec<(&'static str, f32)>,
}

impl ScriptedReranker {
    fn new(scores: Vec<(&'static str, f32)>) -> Self {
        Self { scores }
    }

    fn score_doc(&self, doc: &str) -> f32 {
        self.scores
            .iter()
            .find(|(substr, _)| doc.contains(substr))
            .map(|(_, s)| *s)
            .unwrap_or(0.0)
    }
}

impl Rerank for ScriptedReranker {
    fn score(&self, _query: &str, document: &str) -> Result<f32, RerankerError> {
        Ok(self.score_doc(document))
    }

    fn score_batch(&self, pairs: &[(&str, &str)]) -> Result<Vec<f32>, RerankerError> {
        Ok(pairs.iter().map(|(_, doc)| self.score_doc(doc)).collect())
    }

    fn rerank(&self, _query: &str, documents: &[&str]) -> Result<Vec<RankedResult>, RerankerError> {
        let mut results: Vec<RankedResult> = documents
            .iter()
            .enumerate()
            .map(|(index, doc)| RankedResult {
                index,
                score: self.score_doc(doc),
            })
            .collect();
        results.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        Ok(results)
    }
}

// T-001: YOMU_RERANK=1 + MockReranker -> results reordered by reranker score.
#[test]
fn reranker_reorders_results_by_cross_encoder_score() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let emb = test_embedding();
    // Chunk A: has embedding -> high RRF base score
    storage::insert_chunk(
        &conn,
        "src/A.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("AuthForm"),
            content: "function AuthForm() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb.clone()),
        None,
    )
    .unwrap();

    // Chunk B: FTS only -> lower RRF base score
    storage::replace_file_chunks_only(
        &conn,
        "src/B.tsx",
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

    let reranker = ScriptedReranker::new(vec![("useAuth", 0.9), ("AuthForm", 0.1)]);

    let conn = Arc::new(Mutex::new(conn));
    let outcome = search(
        &conn,
        &MockEmbedder::default(),
        "auth",
        5,
        0,
        Some(&reranker),
        &[],
        false,
    )
    .unwrap();

    assert!(
        outcome.results.len() >= 2,
        "expected at least 2 results, got {}",
        outcome.results.len()
    );
    assert_eq!(
        outcome.results[0].chunk.name.as_deref(),
        Some("useAuth"),
        "reranker should place useAuth first, got: {:?}",
        outcome.results[0].chunk.name
    );
}

// T-002: YOMU_RERANK unset -> identical to existing RRF results.
#[test]
fn no_reranker_produces_rrf_results() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let emb = test_embedding();
    storage::insert_chunk(
        &conn,
        "src/A.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("AuthForm"),
            content: "function AuthForm() {}",
            start_line: 1,
            end_line: 3,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb.clone()),
        None,
    )
    .unwrap();

    let conn = Arc::new(Mutex::new(conn));
    let outcome = search(
        &conn,
        &MockEmbedder::default(),
        "auth",
        5,
        0,
        None,
        &[],
        false,
    )
    .unwrap();

    let outcome_existing = search(
        &conn,
        &MockEmbedder::default(),
        "auth",
        5,
        0,
        None,
        &[],
        false,
    )
    .unwrap();

    assert_eq!(
        outcome.results.len(),
        outcome_existing.results.len(),
        "result count should match existing behavior"
    );
    for (a, b) in outcome.results.iter().zip(outcome_existing.results.iter()) {
        assert_eq!(
            a.chunk.name, b.chunk.name,
            "result order should match existing behavior"
        );
        assert!(
            (a.score - b.score).abs() < 1e-6,
            "scores should match: {} vs {}",
            a.score,
            b.score
        );
    }
}

// T-006: YOMU_RERANK=1, limit=10, offset=5 -> fetch_limit = 10 * 4 + 5 = 45.
#[test]
fn reranker_increases_fetch_limit() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let emb = test_embedding();
    // Insert 50 chunks so we have enough to observe the difference
    for i in 0..50 {
        let file_path = format!("src/File{i}.tsx");
        let name = format!("Component{i}");
        let content = format!("function Component{i}() {{ /* auth related code */ }}");
        storage::insert_chunk(
            &conn,
            &file_path,
            &storage::NewChunk {
                chunk_type: &storage::ChunkType::Component,
                name: Some(&name),
                content: &content,
                start_line: 1,
                end_line: 3,
                parent_index: None,
                source_kind: None,
                injection_flags: None,
            },
            &format!("hash{i}"),
            &storage::ce(emb.clone()),
            None,
        )
        .unwrap();
    }

    let limit: u32 = 10;
    let offset: u32 = 5;
    let reranker = MockReranker::default();

    let conn = Arc::new(Mutex::new(conn));
    let outcome = search(
        &conn,
        &MockEmbedder::default(),
        "component",
        limit,
        offset,
        Some(&reranker),
        &[],
        false,
    )
    .unwrap();

    // With reranker, the pipeline fetches limit*4+offset = 45 candidates
    // then reranks and applies offset+limit truncation.
    // Without reranker, fetch_limit = limit+offset = 15.
    // The final result count should be at most `limit` (10).
    assert!(
        outcome.results.len() <= limit as usize,
        "results should be capped at limit={limit}, got {}",
        outcome.results.len()
    );
    // With 50 chunks and fetch_limit=45, we should get the full `limit` results
    assert_eq!(
        outcome.results.len(),
        limit as usize,
        "with 50 chunks and fetch_limit=45, should return {limit} results"
    );
}

// T-007: YOMU_RERANK=1, cap_per_file=2, file A has 3 candidates
#[test]
fn reranker_scores_applied_before_cap_per_file() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let emb = test_embedding();
    // File A: 3 chunks, with RRF ordering = chunk1 > chunk2 > chunk3
    // Reranker will score: chunk3 (0.9) > chunk1 (0.5) > chunk2 (0.1)
    // After cap_per_file=2, expected survivors: chunk3, chunk1
    let chunks = [
        ("Chunk1", "function Chunk1() { /* auth handler */ }"),
        ("Chunk2", "function Chunk2() { /* auth validator */ }"),
        ("Chunk3", "function Chunk3() { /* auth middleware */ }"),
    ];

    for (name, content) in &chunks {
        storage::insert_chunk(
            &conn,
            "src/auth.tsx",
            &storage::NewChunk {
                chunk_type: &storage::ChunkType::Other,
                name: Some(name),
                content,
                start_line: 1,
                end_line: 3,
                parent_index: None,
                source_kind: None,
                injection_flags: None,
            },
            &format!("hash_{name}"),
            &storage::ce(emb.clone()),
            None,
        )
        .unwrap();
    }

    // Also add a chunk from a different file to verify cap_per_file is per-file
    storage::insert_chunk(
        &conn,
        "src/other.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Other,
            name: Some("OtherComponent"),
            content: "function OtherComponent() { /* auth related */ }",
            start_line: 1,
            end_line: 3,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "hash_other",
        &storage::ce(emb.clone()),
        None,
    )
    .unwrap();

    let reranker = ScriptedReranker::new(vec![
        ("middleware", 0.9),
        ("handler", 0.5),
        ("validator", 0.1),
    ]);

    let conn = Arc::new(Mutex::new(conn));
    let outcome = search(
        &conn,
        &MockEmbedder::default(),
        "auth",
        10,
        0,
        Some(&reranker),
        &[],
        false,
    )
    .unwrap();

    // Count how many results come from src/auth.tsx
    let auth_results: Vec<_> = outcome
        .results
        .iter()
        .filter(|r| r.chunk.file_path == "src/auth.tsx")
        .collect();

    assert!(
        auth_results.len() <= MAX_RESULTS_PER_FILE,
        "cap_per_file should limit auth.tsx results to {MAX_RESULTS_PER_FILE}, got {}",
        auth_results.len()
    );

    // The survivors should be the ones with the highest reranker scores:
    // Chunk3 (middleware, 0.9) and Chunk1 (handler, 0.5)
    let surviving_names: Vec<&str> = auth_results
        .iter()
        .filter_map(|r| r.chunk.name.as_deref())
        .collect();

    assert!(
        surviving_names.contains(&"Chunk3"),
        "Chunk3 (highest reranker score) should survive cap_per_file, got: {surviving_names:?}"
    );
    assert!(
        surviving_names.contains(&"Chunk1"),
        "Chunk1 (second highest reranker score) should survive cap_per_file, got: {surviving_names:?}"
    );
    assert!(
        !surviving_names.contains(&"Chunk2"),
        "Chunk2 (lowest reranker score) should be capped, got: {surviving_names:?}"
    );
}

// T-291: search_returns_results
#[test]
#[ignore = "requires model download"]
fn search_returns_results() {
    use rurico::embed::download_model;
    let paths = download_model(ModelId::default()).expect("download model");
    let embedder = Embedder::new(&paths).expect("load model");
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let embedding = embedder.embed_query("Button component").unwrap();
    storage::insert_chunk(
        &conn,
        "src/Button.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("Button"),
            content: "function Button() { return <div/>; }",
            start_line: 1,
            end_line: 3,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "hash1",
        &storage::ce(embedding),
        None,
    )
    .unwrap();

    let conn = Arc::new(Mutex::new(conn));
    let outcome = search(&conn, &embedder, "button", 10, 0, None, &[], false).unwrap();
    assert!(!outcome.results.is_empty(), "expected at least one result");
}

// T-292: search_pipeline_text_only_returns_name_matches
#[test]
fn search_pipeline_text_only_returns_name_matches() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    storage::replace_file_chunks_only(
        &conn,
        "src/Auth.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("AuthForm"),
            content: "function AuthForm() { return <form/>; }",
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

    let (results, _stages) = search_pipeline(&conn, "auth", None, 10, 0, None, &[], false).unwrap();
    assert!(
        !results.is_empty(),
        "text-only pipeline should find name matches"
    );
    assert_eq!(results[0].chunk.name.as_deref(), Some("AuthForm"));
    assert_ne!(results[0].match_source, storage::MatchSource::Semantic);
}

// T-293: search_pipeline_with_embedding_returns_semantic
#[test]
fn search_pipeline_with_embedding_returns_semantic() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let emb = test_embedding();
    storage::insert_chunk(
        &conn,
        "src/Button.tsx",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("Button"),
            content: "function Button() { return <div/>; }",
            start_line: 1,
            end_line: 3,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb.clone()),
        None,
    )
    .unwrap();

    let (results, _stages) =
        search_pipeline(&conn, "button", Some(&emb), 10, 0, None, &[], false).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].match_source, storage::MatchSource::Semantic);
}

// T-294: search_pipeline_caps_per_file
#[test]
fn search_pipeline_caps_per_file() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let chunks: Vec<storage::NewChunk> = (1..=5)
        .map(|i| storage::NewChunk {
            chunk_type: &storage::ChunkType::Other,
            name: Some("fn"),
            content: "function fn() {}",
            start_line: i,
            end_line: i + 2,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        })
        .collect();
    storage::replace_file_chunks_only(&conn, "src/big.ts", &chunks, "h1", "", &[], None).unwrap();

    let (results, _stages) = search_pipeline(&conn, "fn", None, 10, 0, None, &[], false).unwrap();
    let file_count = results
        .iter()
        .filter(|r| r.chunk.file_path == "src/big.ts")
        .count();
    assert!(
        file_count <= MAX_RESULTS_PER_FILE,
        "expected at most {MAX_RESULTS_PER_FILE} per file, got {file_count}"
    );
}

// T-295: text_only_search_with_offset_returns_results
#[test]
fn text_only_search_with_offset_returns_results() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    let names = [
        "WidgetAlpha",
        "WidgetBeta",
        "WidgetGamma",
        "WidgetDelta",
        "WidgetEpsilon",
        "WidgetZeta",
        "WidgetEta",
        "WidgetTheta",
        "WidgetIota",
        "WidgetKappa",
        "WidgetLambda",
        "WidgetMu",
        "WidgetNu",
        "WidgetXi",
        "WidgetOmicron",
    ];
    for (i, name) in names.iter().enumerate() {
        storage::replace_file_chunks_only(
            &conn,
            &format!("src/{name}.tsx"),
            &[storage::NewChunk {
                chunk_type: &storage::ChunkType::Component,
                name: Some(name),
                content: &format!("function {name}() {{ return <div/>; }}"),
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
    let outcome = search(
        &conn,
        &FailingEmbedder::query_only("unavailable"),
        "widget",
        3,
        10,
        None,
        &[],
        false,
    )
    .unwrap();
    assert!(
        !outcome.results.is_empty(),
        "text-only search with offset=10 should still return results"
    );
}

// T-296: search_chunk_only_index_with_embedder_falls_back_to_text
#[test]
fn search_chunk_only_index_with_embedder_falls_back_to_text() {
    let dir = tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = storage::open_db(&db_path).unwrap();

    storage::replace_file_chunks_only(
        &conn,
        "src/Button.tsx",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Component,
            name: Some("Button"),
            content: "function Button() { return <button>Click</button>; }",
            start_line: 1,
            end_line: 3,
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
    assert_eq!(stats.embedded_chunks, 0, "no embeddings should exist");

    let conn = Arc::new(Mutex::new(conn));
    let outcome = search(
        &conn,
        &MockEmbedder::default(),
        "button",
        10,
        0,
        None,
        &[],
        false,
    )
    .unwrap();
    assert!(
        !outcome.results.is_empty(),
        "should return text/name fallback results even with zero embeddings"
    );
    assert_eq!(outcome.results[0].chunk.name.as_deref(), Some("Button"));
}

fn from_test_db() -> (storage::Db, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let conn = storage::open_db(&dir.path().join("test.db")).unwrap();
    (conn, dir)
}

// RC-002: measure N KNN round-trips latency (N=1 vs N=20) with 500 indexed chunks
// Run with: cargo test --lib bench_knn_round_trips -- --nocapture --ignored
// T-297: bench_knn_round_trips
#[test]
#[ignore]
fn bench_knn_round_trips() {
    use std::time::Instant;

    let (conn, _dir) = from_test_db();

    // Seed 500 chunks with random-ish embeddings across 32 axes
    for i in 0u32..500 {
        let axis = (i % 32) as usize;
        let mut emb = vec![0.0_f32; storage::EMBEDDING_DIMS];
        emb[axis] = 1.0;
        emb[(axis + 1) % storage::EMBEDDING_DIMS] = 0.3;
        storage::insert_chunk(
            &conn,
            &format!("src/file_{i}.rs"),
            &storage::NewChunk {
                chunk_type: &storage::ChunkType::RustFn,
                name: Some("fn_name"),
                content: &format!("fn fn_{i}() {{}}"),
                start_line: 1,
                end_line: 1,
                parent_index: None,
                source_kind: None,
                injection_flags: None,
            },
            &format!("h{i}"),
            &storage::ce(emb),
            None,
        )
        .unwrap();
    }

    let source_ids = HashSet::new();

    let run = |n: usize, label: &str| {
        let emb_bytes: Vec<Vec<u8>> = (0..n).map(|i| emb_axis_bytes(i % 32)).collect();
        let iters = 50;
        let t = Instant::now();
        for _ in 0..iters {
            let _ = search_from_file(&conn, &emb_bytes, &source_ids, None, 10, &[]).unwrap();
        }
        let avg_us = t.elapsed().as_micros() / iters;
        println!("N={n:2} ({label}): {avg_us} µs/call");
    };

    run(1, "single fn");
    run(5, "small file");
    run(10, "medium file");
    run(20, "cap");
}

fn emb_axis(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; storage::EMBEDDING_DIMS];
    v[axis] = 1.0;
    v
}

fn emb_axis_bytes(axis: usize) -> Vec<u8> {
    cast_slice::<f32, u8>(&emb_axis(axis)).to_vec()
}

// T-009: search_from_file excludes source chunk_ids
#[test]
fn search_from_file_excludes_source_chunks() {
    let (conn, _dir) = from_test_db();

    let id_src1 = storage::insert_chunk(
        &conn,
        "src/source.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("src_fn1"),
            content: "fn src_fn1() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb_axis(0)),
        None,
    )
    .unwrap();
    let id_src2 = storage::insert_chunk(
        &conn,
        "src/source.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("src_fn2"),
            content: "fn src_fn2() {}",
            start_line: 3,
            end_line: 3,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb_axis(1)),
        None,
    )
    .unwrap();
    let id_other = storage::insert_chunk(
        &conn,
        "src/other.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("other_fn"),
            content: "fn other_fn() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h2",
        &storage::ce(emb_axis(2)),
        None,
    )
    .unwrap();

    let emb_bytes = vec![emb_axis_bytes(0)];
    let source_ids = HashSet::from([id_src1, id_src2]);

    let results = search_from_file(&conn, &emb_bytes, &source_ids, None, 10, &[]).unwrap();
    let result_ids: Vec<i64> = results.iter().filter_map(|r| r.chunk_id).collect();
    assert!(!result_ids.contains(&id_src1));
    assert!(!result_ids.contains(&id_src2));
    assert!(result_ids.contains(&id_other));
}

// T-010: search_from_file with query applies FTS filter
#[test]
fn search_from_file_with_query_applies_fts_filter() {
    let (conn, _dir) = from_test_db();

    let id_source = storage::insert_chunk(
        &conn,
        "src/source.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("source"),
            content: "fn source() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb_axis(0)),
        None,
    )
    .unwrap();
    let id_error = storage::insert_chunk(
        &conn,
        "src/error.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("handle_error"),
            content: "fn handle_error() { error handling logic }",
            start_line: 1,
            end_line: 3,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h2",
        &storage::ce(emb_axis(1)),
        None,
    )
    .unwrap();
    let _id_unrelated = storage::insert_chunk(
        &conn,
        "src/unrelated.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("unrelated"),
            content: "fn unrelated() { nothing here }",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h3",
        &storage::ce(emb_axis(2)),
        None,
    )
    .unwrap();

    let emb_bytes = vec![emb_axis_bytes(0)];
    let source_ids = HashSet::from([id_source]);

    let results = search_from_file(&conn, &emb_bytes, &source_ids, Some("error"), 10, &[]).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk_id, Some(id_error));
}

// T-011: search_from_file FTS zero matches returns empty
#[test]
fn search_from_file_fts_zero_matches_returns_empty() {
    let (conn, _dir) = from_test_db();

    let id_source = storage::insert_chunk(
        &conn,
        "src/source.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("source"),
            content: "fn source() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb_axis(0)),
        None,
    )
    .unwrap();
    storage::insert_chunk(
        &conn,
        "src/other.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("other"),
            content: "fn other() { some logic }",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h2",
        &storage::ce(emb_axis(1)),
        None,
    )
    .unwrap();

    let emb_bytes = vec![emb_axis_bytes(0)];
    let source_ids = HashSet::from([id_source]);

    let results =
        search_from_file(&conn, &emb_bytes, &source_ids, Some("xyzzyx"), 10, &[]).unwrap();
    assert!(results.is_empty());
}

// T-012: search_from_file with no stored embeddings returns empty
#[test]
fn search_from_file_no_embeddings_returns_empty() {
    let (conn, _dir) = from_test_db();
    let source_ids = HashSet::from([1]);
    let results = search_from_file(&conn, &[], &source_ids, None, 10, &[]).unwrap();
    assert!(results.is_empty());
}

// T-556: search_from_file caps at 20 sub-embeddings
#[test]
fn search_from_file_caps_sub_embeddings_at_20() {
    let (conn, _dir) = from_test_db();

    storage::insert_chunk(
        &conn,
        "src/target.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("target"),
            content: "fn target() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb_axis(0)),
        None,
    )
    .unwrap();

    // Generate 25 sub-embeddings (all same vector for simplicity)
    let emb_bytes = emb_axis_bytes(0);
    let entries: Vec<Vec<u8>> = (0..25).map(|_| emb_bytes.clone()).collect();
    let source_ids = HashSet::new();

    // Should not error; internally caps at 20
    let results = search_from_file(&conn, &entries, &source_ids, None, 10, &[]).unwrap();
    assert!(!results.is_empty());
}

// T-558: FTS filter uses fetch_limit (not limit) so rerank picks best semantic match
#[test]
fn search_from_file_fts_uses_fetch_limit_not_limit() {
    let (conn, _dir) = from_test_db();

    // Source chunk — we search "from" this
    let id_source = storage::insert_chunk(
        &conn,
        "src/source.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("source"),
            content: "fn source() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb_axis(0)),
        None,
    )
    .unwrap();

    // A: closest to source (same axis), low keyword frequency → low BM25
    let id_a = storage::insert_chunk(
        &conn,
        "src/a.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("close_handler"),
            content: "fn close_handler() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h2",
        &storage::ce(emb_axis(0)),
        None,
    )
    .unwrap();

    // B: far from source, high keyword frequency → high BM25
    let _id_b = storage::insert_chunk(
        &conn,
        "src/b.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("far_handler"),
            content: "fn far_handler() { handler handler handler handler handler }",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h3",
        &storage::ce(emb_axis(1)),
        None,
    )
    .unwrap();

    // C: far from source, medium keyword frequency
    let _id_c = storage::insert_chunk(
        &conn,
        "src/c.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("mid_handler"),
            content: "fn mid_handler() { handler handler handler }",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h4",
        &storage::ce(emb_axis(2)),
        None,
    )
    .unwrap();

    let emb_bytes = vec![emb_axis_bytes(0)];
    let source_ids = HashSet::from([id_source]);

    // limit=1: with bug, FTS picks top BM25 (far_handler); with fix, all pass through to rerank
    let results =
        search_from_file(&conn, &emb_bytes, &source_ids, Some("handler"), 1, &[]).unwrap();
    assert_eq!(results.len(), 1, "should return exactly 1 result");
    // Semantically closest FTS match should win after rerank, not highest BM25
    assert_eq!(
        results[0].chunk_id,
        Some(id_a),
        "closest semantic match (close_handler) should beat high-BM25 (far_handler)"
    );
}

// COV-4: fetch_limit = limit * 3 (3× overfetch) — the 3rd-closest candidate must reach FTS pool
// T-298: search_from_file_3x_overfetch_reaches_third_candidate
#[test]
fn search_from_file_3x_overfetch_reaches_third_candidate() {
    let (conn, _dir) = from_test_db();

    // KNN distances from emb_axis(0): A=0, B=0.1, C=0.5, source≈1.41 (won't appear in top 3)
    let emb_b = {
        let mut v = vec![0.0_f32; storage::EMBEDDING_DIMS];
        v[0] = 1.0;
        v[1] = 0.1;
        v
    };
    let emb_c = {
        let mut v = vec![0.0_f32; storage::EMBEDDING_DIMS];
        v[0] = 1.0;
        v[1] = 0.5;
        v
    };

    // Source stored far from query (emb_axis(5)), so it won't consume a top-3 KNN slot
    let id_source = storage::insert_chunk(
        &conn,
        "src/source.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("src_fn"),
            content: "fn src_fn() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "hsource",
        &storage::ce(emb_axis(5)),
        None,
    )
    .unwrap();

    // A: KNN rank 1 (closest), NO keyword
    storage::insert_chunk(
        &conn,
        "src/a.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("a_fn"),
            content: "fn a_fn() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "ha",
        &storage::ce(emb_axis(0)),
        None,
    )
    .unwrap();

    // B: KNN rank 2, NO keyword
    storage::insert_chunk(
        &conn,
        "src/b.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("b_fn"),
            content: "fn b_fn() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "hb",
        &storage::ce(emb_b),
        None,
    )
    .unwrap();

    // C: KNN rank 3, HAS keyword "magic"
    let id_c = storage::insert_chunk(
        &conn,
        "src/c.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("c_fn"),
            content: "fn c_fn() { magic magic magic }",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "hc",
        &storage::ce(emb_c),
        None,
    )
    .unwrap();

    // Search from emb_axis(0) with query "magic", limit=1 → fetch_limit=3
    let emb_bytes = vec![emb_axis_bytes(0)];
    let source_ids = HashSet::from([id_source]);

    let results = search_from_file(&conn, &emb_bytes, &source_ids, Some("magic"), 1, &[]).unwrap();
    assert_eq!(results.len(), 1, "should return exactly 1 result");
    assert_eq!(
        results[0].chunk_id,
        Some(id_c),
        "3rd-closest candidate (c_fn, only keyword match) should be reached by 3× overfetch"
    );
}

// P1: source exclusion must not deplete results when limit=1 and source is nearest neighbour
// T-299: search_from_file_source_exclusion_does_not_empty_results_at_limit_1
#[test]
fn search_from_file_source_exclusion_does_not_empty_results_at_limit_1() {
    let (conn, _dir) = from_test_db();

    // Source at emb_axis(0); other is a close neighbour (slightly off-axis)
    let id_source = storage::insert_chunk(
        &conn,
        "src/source.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("source"),
            content: "fn source() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb_axis(0)),
        None,
    )
    .unwrap();
    let id_other = storage::insert_chunk(
        &conn,
        "src/other.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("other"),
            content: "fn other() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h2",
        &storage::ce({
            let mut v = emb_axis(0);
            v[1] = 0.1;
            v
        }),
        None,
    )
    .unwrap();

    let emb_bytes = vec![emb_axis_bytes(0)];
    let source_ids = HashSet::from([id_source]);

    let results = search_from_file(&conn, &emb_bytes, &source_ids, None, 1, &[]).unwrap();
    assert_eq!(
        results.len(),
        1,
        "should return 1 result even when source fills KNN slot"
    );
    assert_eq!(results[0].chunk_id, Some(id_other));
}

// P2: stopword-only query must not empty semantic results
// T-300: search_from_file_stopword_query_falls_back_to_semantic
#[test]
fn search_from_file_stopword_query_falls_back_to_semantic() {
    let (conn, _dir) = from_test_db();

    let id_source = storage::insert_chunk(
        &conn,
        "src/source.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("source"),
            content: "fn source() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h1",
        &storage::ce(emb_axis(0)),
        None,
    )
    .unwrap();
    storage::insert_chunk(
        &conn,
        "src/other.rs",
        &storage::NewChunk {
            chunk_type: &storage::ChunkType::RustFn,
            name: Some("other"),
            content: "fn other() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h2",
        &storage::ce(emb_axis(0)),
        None,
    )
    .unwrap();

    let emb_bytes = vec![emb_axis_bytes(0)];
    let source_ids = HashSet::from([id_source]);

    // "a" is a stopword → extract_keywords returns [] → must behave like query=None
    let results = search_from_file(&conn, &emb_bytes, &source_ids, Some("a"), 10, &[]).unwrap();
    assert!(
        !results.is_empty(),
        "stopword-only query should return semantic results, not []"
    );
}

// === TC-1: keyword_hit_ratio IDF-weighted branch ===

// T-301: keyword_hit_ratio_uses_idf_weights
#[test]
fn keyword_hit_ratio_uses_idf_weights() {
    let result = storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/auth.ts".to_owned(),
            chunk_type: storage::ChunkType::Other,
            name: Some("authenticate".to_owned()),
            content: "function authenticate() {}".to_owned(),
            start_line: 1,
            end_line: 1,
            parent_chunk_id: None,
            source_kind: None,
            injection_flags: None,
        },
        chunk_id: None,
        distance: f32::INFINITY,
        match_source: storage::MatchSource::Fts,
        score: 0.0,
    };
    let keywords = vec!["auth".to_owned(), "form".to_owned()];
    let idfs = [1.0_f32, 2.0_f32];
    // "auth" matches name "authenticate" (contains "auth"); "form" does not match.
    // total_idf = 3.0 >= 1e-6 → IDF-weighted branch executes.
    // hit_idf = 1.0, total_idf = 3.0 → expected ratio = 1.0/3.0.
    let ratio = keyword_hit_ratio(&result, &keywords, &idfs, false);
    let expected = 1.0_f32 / 3.0;
    assert!(
        (ratio - expected).abs() < 1e-5,
        "IDF-weighted branch: expected {expected:.6}, got {ratio:.6}"
    );
}

// === TC-5: search_pipeline offset boundary ===

// T-302: search_pipeline_offset_beyond_results_returns_empty
#[test]
fn search_pipeline_offset_beyond_results_returns_empty() {
    let (conn, _dir) = from_test_db();

    for (i, name) in ["WidgetA", "WidgetB"].iter().enumerate() {
        storage::replace_file_chunks_only(
            &conn,
            &format!("src/{name}.tsx"),
            &[storage::NewChunk {
                chunk_type: &storage::ChunkType::Component,
                name: Some(name),
                content: &format!("function {name}() {{ return <div/>; }}"),
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

    // offset=10 exceeds the 2-result set → empty Vec expected.
    let (results, _stages) =
        search_pipeline(&conn, "widget", None, 10, 10, None, &[], false).unwrap();
    assert!(
        results.is_empty(),
        "offset >= results.len() should return empty Vec, got: {results:?}"
    );
}

// === TC-6: stage capture (Issue #182 Phase 2) ===

// T-QL-014: capture_stages=false leaves stages as None (NFR-002 zero-cost flag-off).
#[test]
fn search_pipeline_capture_stages_false_returns_none() {
    let (conn, _dir) = from_test_db();
    storage::replace_file_chunks_only(
        &conn,
        "src/auth.ts",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Other,
            name: Some("authHandler"),
            content: "function authHandler() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        }],
        "h0",
        "",
        &[],
        None,
    )
    .unwrap();

    let (_results, stages) = search_pipeline(&conn, "auth", None, 10, 0, None, &[], false).unwrap();
    assert!(
        stages.is_none(),
        "capture_stages=false must return None, got: {stages:?}"
    );
}

// T-QL-015: capture_stages=true populates fts_results / rrf_results / reranked_results
// on the FTS-only path (no embedding) with source="fts".
#[test]
fn search_pipeline_capture_stages_fts_path_populates_stages() {
    let (conn, _dir) = from_test_db();
    storage::replace_file_chunks_only(
        &conn,
        "src/auth.ts",
        &[storage::NewChunk {
            chunk_type: &storage::ChunkType::Other,
            name: Some("authHandler"),
            content: "function authHandler() {}",
            start_line: 1,
            end_line: 1,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        }],
        "h0",
        "",
        &[],
        None,
    )
    .unwrap();

    let (results, stages) = search_pipeline(&conn, "auth", None, 10, 0, None, &[], true).unwrap();
    assert!(!results.is_empty(), "expected at least 1 result");
    let s = stages.expect("capture_stages=true must return Some");
    assert!(
        s.vec_results.is_empty(),
        "no embedding => vec_results must be empty, got: {:?}",
        s.vec_results
    );
    assert!(
        !s.fts_results.is_empty(),
        "FTS path must populate fts_results"
    );
    assert_eq!(
        s.fts_results[0].source, "fts",
        "fts stage source must be 'fts'"
    );
    assert!(
        !s.rrf_results.is_empty(),
        "post-merge rrf_results must include FTS hits"
    );
    assert!(
        !s.reranked_results.is_empty(),
        "post-rerank reranked_results must be populated"
    );
}

// === is_test_path pattern coverage ===

// T-303: is_test_path_returns_true_for_all_patterns
#[test]
fn is_test_path_returns_true_for_all_patterns() {
    let cases = [
        "src/__tests__/foo.ts",
        "src/__mocks__/foo.ts",
        "src/__fixtures__/foo.ts",
        "src/foo.test.ts",
        "src/foo.spec.ts",
        "src/foo.stories.ts",
        "src/test/foo.ts",
        "test/foo.ts",
        "src/examples/foo.ts",
        "examples/foo.ts",
        "src/fixtures/foo.ts",
        "src/e2e/foo.ts",
    ];
    for path in cases {
        assert!(is_test_path(path), "expected is_test_path true for: {path}");
    }
}

// T-304: is_test_path_returns_false_for_source_files
#[test]
fn is_test_path_returns_false_for_source_files() {
    let cases = [
        "src/components/Button.tsx",
        "src/utils/helpers.ts",
        "src/query/mod.rs",
    ];
    for path in cases {
        assert!(
            !is_test_path(path),
            "expected is_test_path false for: {path}"
        );
    }
}
