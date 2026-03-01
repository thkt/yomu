//! Semantic search over indexed code chunks.

use std::collections::HashSet;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::indexer::embedder::{Embed, EmbedError};
use crate::storage::{self, ChunkType, Db, SearchResult, StorageError};

#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    #[error("search query failed: {0}")]
    Storage(#[from] StorageError),
    #[error("search embedding failed: {0}")]
    Embed(#[from] EmbedError),
    #[error("internal task failed: {0}")]
    Internal(String),
}

impl From<tokio::task::JoinError> for QueryError {
    fn from(e: tokio::task::JoinError) -> Self {
        Self::Internal(e.to_string())
    }
}

const STOP_WORDS: &[&str] = &["the", "a", "an", "in", "for", "of", "with", "and", "or"];

/// Split identifiers by camelCase, PascalCase, kebab-case, and snake_case.
/// "useChat" → ["use", "Chat"], "HTMLElement" → ["HTML", "Element"],
/// "data-table" → ["data", "table"], "get_value" → ["get", "value"]
fn split_identifier(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = s.chars().collect();

    for i in 0..chars.len() {
        let c = chars[i];
        if c == '-' || c == '_' {
            if !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
        } else if c.is_uppercase() {
            let prev_lower = i > 0 && chars[i - 1].is_lowercase();
            let prev_upper = i > 0 && chars[i - 1].is_uppercase();
            let next_lower = i + 1 < chars.len() && chars[i + 1].is_lowercase();
            // Split before: lowercase→Uppercase or ACRONYM→Titlecase
            if (prev_lower || (prev_upper && next_lower)) && !current.is_empty() {
                parts.push(std::mem::take(&mut current));
            }
            current.push(c);
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

pub fn extract_keywords(query: &str) -> Vec<String> {
    let tokens: Vec<&str> = query.split_whitespace().collect();

    // Collect base keywords: whole tokens + identifier parts
    let mut base: Vec<String> = Vec::new();
    for token in &tokens {
        let lower = token.to_lowercase();
        if lower.len() >= 2 && !STOP_WORDS.contains(&lower.as_str()) && !base.contains(&lower) {
            base.push(lower);
        }
        // Expand compound identifiers: "useChat" → "use" + "chat"
        for part in split_identifier(token) {
            let part_lower = part.to_lowercase();
            if part_lower.len() >= 2
                && !STOP_WORDS.contains(&part_lower.as_str())
                && !base.contains(&part_lower)
            {
                base.push(part_lower);
            }
        }
    }

    // Words ending in -ing that are NOT gerunds (should not be stemmed)
    const ING_DENY: &[&str] = &[
        "string", "bring", "thing", "nothing", "something", "everything",
        "ring", "king", "spring", "swing", "sing", "sting", "wing",
    ];
    // Words ending in -s that are NOT plurals
    const S_DENY: &[&str] = &[
        "class", "this", "alias", "canvas", "focus", "status", "bus",
        "process", "address", "access", "express", "progress",
    ];

    // Apply stemming on all base keywords
    let mut all = base.clone();
    for kw in &base {
        let stems: Vec<String> = if kw.len() > 5
            && kw.ends_with("ing")
            && !ING_DENY.contains(&kw.as_str())
        {
            vec![kw[..kw.len() - 3].to_string()]
        } else if kw.len() > 3
            && kw.ends_with('s')
            && !kw.ends_with("ss")
            && !S_DENY.contains(&kw.as_str())
        {
            vec![kw[..kw.len() - 1].to_string()]
        } else {
            vec![]
        };
        for s in stems {
            if s.len() >= 2 && !all.contains(&s) {
                all.push(s);
            }
        }
    }
    all
}

pub fn extract_type_hints(query: &str) -> Vec<ChunkType> {
    let tokens: Vec<String> = query.split_whitespace().map(|t| t.to_lowercase()).collect();
    let mut hints = Vec::new();
    for token in &tokens {
        let hint = match token.as_str() {
            "hook" | "hooks" => Some(ChunkType::Hook),
            "component" | "components" => Some(ChunkType::Component),
            "type" | "types" | "interface" => Some(ChunkType::TypeDef),
            "css" | "style" | "styles" => Some(ChunkType::CssRule),
            "test" | "tests" | "spec" => Some(ChunkType::TestCase),
            _ => None,
        };
        if let Some(h) = hint
            && !hints.contains(&h)
        {
            hints.push(h);
        }
    }
    hints
}

/// IDF-weighted hit ratio: how well do keywords match this result?
/// Returns a value between 0.0 (no match) and 1.0 (all keywords match, weighted by IDF).
fn keyword_hit_ratio(
    result: &storage::SearchResult,
    keywords: &[String],
    idfs: &[f32],
    check_content: bool,
) -> f32 {
    if keywords.is_empty() {
        return 0.0;
    }
    let name_lower = result
        .chunk
        .name
        .as_deref()
        .unwrap_or("")
        .to_lowercase();
    let path_lower = result.chunk.file_path.to_lowercase();
    let content_lower;
    let content_ref = if check_content {
        content_lower = result.chunk.content.to_lowercase();
        Some(content_lower.as_str())
    } else {
        None
    };

    let total_idf: f32 = idfs.iter().sum();
    if total_idf < 1e-6 {
        // Fallback to uniform weight if no IDF data
        let hits = keywords
            .iter()
            .filter(|kw| {
                name_lower.contains(kw.as_str())
                    || path_lower.contains(kw.as_str())
                    || content_ref.is_some_and(|c| c.contains(kw.as_str()))
            })
            .count();
        return hits as f32 / keywords.len() as f32;
    }

    let hit_idf: f32 = keywords
        .iter()
        .zip(idfs)
        .filter(|(kw, _)| {
            name_lower.contains(kw.as_str())
                || path_lower.contains(kw.as_str())
                || content_ref.is_some_and(|c| c.contains(kw.as_str()))
        })
        .map(|(_, idf)| idf)
        .sum();
    hit_idf / total_idf
}

const MAX_RESULTS_PER_FILE: usize = 2;

/// Cap results to at most `max` entries per file to prevent one file from dominating.
fn cap_per_file(results: &mut Vec<storage::SearchResult>, max: usize) {
    let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    results.retain(|r| {
        let count = counts.entry(r.chunk.file_path.clone()).or_insert(0);
        *count += 1;
        *count <= max
    });
}

const NAME_MATCH_BASE: f32 = 0.40;
const NAME_MATCH_RATIO_WEIGHT: f32 = 0.20;
const CONTENT_MATCH_BASE: f32 = 0.35;
const CONTENT_MATCH_RATIO_WEIGHT: f32 = 0.15;
const NAME_MATCH_BONUS: f32 = 0.05;
const TYPE_HINT_BONUS: f32 = 0.03;
const IMPORT_RANK_BONUS: f32 = 0.03;
const TEST_PATH_PENALTY: f32 = 0.05;

fn is_test_path(path: &str) -> bool {
    path.contains("__tests__")
        || path.contains("__mocks__")
        || path.contains("__fixtures__")
        || path.contains(".test.")
        || path.contains(".spec.")
        || path.contains(".stories.")
        || path.contains("/test/")
        || path.starts_with("test/")
        || path.contains("/examples/")
        || path.starts_with("examples/")
        || path.contains("/fixtures/")
        || path.contains("/e2e/")
}

/// Dampen semantic scores when embedding coverage is low.
/// Returns 1.0 at ≥9% coverage, scaling down to 0.4 at 0%.
fn semantic_confidence(embed_coverage: f32) -> f32 {
    (embed_coverage.sqrt() * 2.0 + 0.4).min(1.0)
}

/// Rerank results by score (name match, type hint, import rank, test penalty).
/// `embed_coverage` is embedded_chunks / total_chunks (0.0–1.0).
/// `keyword_idfs` contains IDF weight per keyword (same length as `keywords`).
pub fn rerank(
    results: &mut [storage::SearchResult],
    type_hints: &[storage::ChunkType],
    import_counts: &std::collections::HashMap<String, u32>,
    keywords: &[String],
    keyword_idfs: &[f32],
    embed_coverage: f32,
) {
    if results.is_empty() {
        return;
    }

    let confidence = semantic_confidence(embed_coverage);

    let mut counts: Vec<u32> = import_counts.values().copied().filter(|&c| c > 0).collect();
    counts.sort_unstable();
    let top_25_threshold = if counts.is_empty() {
        0
    } else {
        counts[counts.len().saturating_sub(counts.len() / 4 + 1)]
    };

    let query_wants_tests = type_hints.contains(&storage::ChunkType::TestCase);

    for result in results.iter_mut() {
        let base = match result.match_source {
            storage::MatchSource::Semantic => 1.0 / (1.0 + result.distance) * confidence,
            storage::MatchSource::NameMatch => {
                let ratio = keyword_hit_ratio(result, keywords, keyword_idfs, false);
                NAME_MATCH_BASE + NAME_MATCH_RATIO_WEIGHT * ratio
            }
            storage::MatchSource::ContentMatch => {
                let ratio = keyword_hit_ratio(result, keywords, keyword_idfs, true);
                CONTENT_MATCH_BASE + CONTENT_MATCH_RATIO_WEIGHT * ratio
            }
        };

        let name_bonus = if result.match_source == storage::MatchSource::NameMatch {
            NAME_MATCH_BONUS
        } else {
            0.0
        };

        let type_bonus = if !type_hints.is_empty()
            && type_hints.contains(&result.chunk.chunk_type)
        {
            TYPE_HINT_BONUS
        } else {
            0.0
        };

        let ic = import_counts
            .get(&result.chunk.file_path)
            .copied()
            .unwrap_or(0);
        let import_bonus = if top_25_threshold > 0 && ic >= top_25_threshold {
            IMPORT_RANK_BONUS
        } else {
            0.0
        };

        let test_penalty = if is_test_path(&result.chunk.file_path) && !query_wants_tests {
            TEST_PATH_PENALTY
        } else {
            0.0
        };

        result.score = base + name_bonus + type_bonus + import_bonus - test_penalty;
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Three-tier search pipeline (vector → name LIKE → FTS5 content) with reranking.
fn search_pipeline(
    conn: &Db,
    query: &str,
    query_embedding: &[f32],
    limit: u32,
    offset: u32,
) -> Result<Vec<SearchResult>, StorageError> {
    let keywords = extract_keywords(query);
    let type_hints = extract_type_hints(query);

    let mut results = storage::search_similar(conn, query_embedding, limit, offset)?;

    let fallback_limit = limit * 3;

    if !keywords.is_empty() {
        let type_filter = if type_hints.is_empty() { None } else { Some(type_hints.as_slice()) };

        let exclude_ids: HashSet<i64> = results.iter().filter_map(|r| r.chunk_id).collect();
        let keyword_refs: Vec<&str> = keywords.iter().map(|s| s.as_str()).collect();
        let name_results = storage::search_by_name(conn, &keyword_refs, type_filter, &exclude_ids, fallback_limit)?;
        results.extend(name_results);

        let exclude_ids: HashSet<i64> = results.iter().filter_map(|r| r.chunk_id).collect();
        let content_results = storage::search_by_content(conn, &keyword_refs, type_filter, &exclude_ids, fallback_limit)?;
        results.extend(content_results);
    }

    let stats = storage::get_stats(conn)?;
    let embed_coverage = if stats.total_chunks > 0 {
        stats.embedded_chunks as f32 / stats.total_chunks as f32
    } else {
        0.0
    };

    let keyword_refs: Vec<&str> = keywords.iter().map(|s| s.as_str()).collect();
    let dfs = storage::get_keyword_doc_frequencies(conn, &keyword_refs)?;
    let total = stats.total_chunks.max(1) as f32;
    let keyword_idfs: Vec<f32> = dfs.iter().map(|&df| (total / (df.max(1) as f32)).ln()).collect();

    let file_paths: Vec<&str> = results.iter().map(|r| r.chunk.file_path.as_str()).collect();
    let import_counts = storage::get_import_counts(conn, &file_paths)?;
    rerank(&mut results, &type_hints, &import_counts, &keywords, &keyword_idfs, embed_coverage);
    cap_per_file(&mut results, MAX_RESULTS_PER_FILE);
    results.truncate(limit as usize);

    Ok(results)
}

/// Vector search with name-based fallback when results < limit.
pub async fn search(
    conn: Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    query: &str,
    limit: u32,
    offset: u32,
) -> Result<Vec<SearchResult>, QueryError> {
    let query_embedding = embedder.embed_query(query).await?;
    let query_owned = query.to_string();

    let results = tokio::task::spawn_blocking(move || {
        let conn = conn.lock();
        search_pipeline(&conn, &query_owned, &query_embedding, limit, offset)
    })
    .await?;
    Ok(results?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::embedder::{Embedder, MockEmbedder, EMBEDDING_DIMS};

    #[test]
    fn query_error_from_storage_error() {
        let se = StorageError::Sqlite(rusqlite::Error::QueryReturnedNoRows);
        let qe: QueryError = se.into();
        assert!(qe.to_string().contains("Query returned no rows"));
    }

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
        let results = search(conn, &MockEmbedder, "button", 10, 0).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.name.as_deref(), Some("Button"));
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
        // "a" (stopword), "UI" → "ui" (2 chars, keep), "in" (stopword), "React" → "react"
        assert_eq!(kw, vec!["ui", "react"]);
    }

    #[test]
    fn extract_keywords_lowercases() {
        let kw = extract_keywords("UseAuth Component");
        // "UseAuth" → whole "useauth" + parts "use", "auth"; "Component" → "component"
        assert_eq!(kw, vec!["useauth", "use", "auth", "component"]);
    }

    #[test]
    fn extract_keywords_no_stem_for_non_gerunds() {
        // "string" is in ING_DENY — should NOT be stemmed
        let kw = extract_keywords("string parsing");
        assert!(kw.contains(&"string".to_string()), "string should remain");
        assert!(!kw.contains(&"str".to_string()), "str should not appear");
    }

    #[test]
    fn extract_keywords_no_stem_for_non_plurals() {
        // "class" is in S_DENY — should NOT be stemmed
        let kw = extract_keywords("class definition");
        assert!(kw.contains(&"class".to_string()), "class should remain");
        assert!(!kw.contains(&"clas".to_string()), "clas should not appear");
    }

    #[test]
    fn extract_keywords_short_ing_not_stemmed() {
        // "thing" is len=5 (not > 5) — should NOT be stemmed
        let kw = extract_keywords("thing setup");
        assert_eq!(kw, vec!["thing", "setup"]);
    }

    // --- CamelCase splitting tests ---

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
        // "DataTable" splits to "data" + "table", same as token lowercased parts
        let kw = extract_keywords("DataTable");
        let data_count = kw.iter().filter(|k| *k == "data").count();
        assert_eq!(data_count, 1, "no duplicates: {kw:?}");
    }

    #[test]
    fn extract_keywords_normal_query_unchanged() {
        // Normal space-separated queries produce the same result as before
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
        let results = search(conn, &MockEmbedder, "auth hook", 5, 0).await.unwrap();

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
        let results = search(conn, &MockEmbedder, "auth", 5, 0).await.unwrap();

        let auth_count = results.iter().filter(|r| r.chunk.name.as_deref() == Some("useAuth")).count();
        assert_eq!(auth_count, 1, "useAuth should appear exactly once");
        assert_eq!(results[0].match_source, storage::MatchSource::Semantic);
    }

    use std::collections::HashMap;

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
        rerank(&mut results, &[], &HashMap::new(), &[], &[], 1.0);
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
        rerank(&mut results, &[], &HashMap::new(), &kw, &[], 1.0);
        assert!((results[0].score - 0.65).abs() < 1e-6,
            "expected 0.65, got {}", results[0].score);
    }

    #[test]
    fn rerank_sorts_by_score_descending() {
        let mut results = vec![
            make_result("src/B.tsx", "B", storage::ChunkType::Component, 0.5, storage::MatchSource::Semantic),
            make_result("src/A.tsx", "A", storage::ChunkType::Component, 0.1, storage::MatchSource::Semantic),
        ];
        rerank(&mut results, &[], &HashMap::new(), &[], &[], 1.0);
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

        rerank(&mut with_hint, &[storage::ChunkType::Hook], &HashMap::new(), &[], &[], 1.0);
        rerank(&mut without_hint, &[], &HashMap::new(), &[], &[], 1.0);

        let diff = with_hint[0].score - without_hint[0].score;
        assert!((diff - 0.03).abs() < 1e-6,
            "type_hint bonus should be +0.03, got diff {diff}");
    }

    #[test]
    fn rerank_no_type_hint_bonus_when_empty() {
        let mut results = vec![
            make_result("src/A.tsx", "useAuth", storage::ChunkType::Hook, 0.5, storage::MatchSource::Semantic),
        ];
        let base = 1.0 / (1.0 + 0.5_f32);
        rerank(&mut results, &[], &HashMap::new(), &[], &[], 1.0);
        assert!((results[0].score - base).abs() < 1e-6,
            "no bonus expected: {base} vs {}", results[0].score);
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
        rerank(&mut results, &[], &import_counts, &[], &[], 1.0);

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
        rerank(&mut test_result, &[], &HashMap::new(), &[], &[], 1.0);
        rerank(&mut normal_result, &[], &HashMap::new(), &[], &[], 1.0);

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
        rerank(&mut results, &[storage::ChunkType::TestCase], &HashMap::new(), &[], &[], 1.0);
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
        rerank(&mut results, &[storage::ChunkType::Hook], &import_counts, &kw, &[], 1.0);
        assert!((results[0].score - 0.71).abs() < 1e-6,
            "expected 0.71, got {}", results[0].score);
    }

    #[test]
    fn rerank_combined_bonuses_semantic() {
        let mut results = vec![
            make_result("src/useAuth.tsx", "useAuth", storage::ChunkType::Hook, 0.1, storage::MatchSource::Semantic),
        ];
        let import_counts = HashMap::from([("src/useAuth.tsx".to_string(), 10u32)]);
        rerank(&mut results, &[storage::ChunkType::Hook], &import_counts, &[], &[], 1.0);

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
        rerank(&mut results, &[storage::ChunkType::Hook], &import_counts, &[], &[], 1.0);

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
        let results = search(conn, &MockEmbedder, "auth hook", 5, 0).await.unwrap();

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
        // At 2% coverage (confidence ≈ 0.68), semantic score should be dampened
        rerank(&mut results, &[], &HashMap::new(), &[], &[], 0.02);
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
        // At 2% coverage, name match should beat semantic
        rerank(&mut results, &[], &HashMap::new(), &kw, &[], 0.02);
        assert_eq!(results[0].chunk.name.as_deref(), Some("useChat"),
            "name match should rank first at low coverage");
        assert!(results[0].score > results[1].score,
            "name ({}) should beat semantic ({}) at 2% coverage",
            results[0].score, results[1].score);
    }

    #[test]
    fn content_match_scores_from_content_body() {
        let kw = vec!["validation".to_string(), "form".to_string()];
        // Name/path don't match keywords, but content does
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
        // check_content=true should find both "validation" and "form" in content
        let ratio = keyword_hit_ratio(&result, &kw, &[], true);
        assert!((ratio - 1.0).abs() < 1e-6, "both keywords match → ratio=1.0, got {ratio}");

        // check_content=false should find 0 (name/path don't contain keywords)
        let ratio_no_content = keyword_hit_ratio(&result, &kw, &[], false);
        assert!((ratio_no_content).abs() < 1e-6, "name/path don't match → ratio=0.0, got {ratio_no_content}");
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
        let results = search(conn, &embedder, "button", 10, 0).await.unwrap();
        assert!(!results.is_empty(), "expected at least one result");
    }
}
