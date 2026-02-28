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

pub fn extract_keywords(query: &str) -> Vec<String> {
    query
        .split_whitespace()
        .map(|t| t.to_lowercase())
        .filter(|t| t.len() >= 2 && !STOP_WORDS.contains(&t.as_str()))
        .collect()
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

const NAME_MATCH_BONUS: f32 = 0.05;
const TYPE_HINT_BONUS: f32 = 0.03;
const IMPORT_RANK_BONUS: f32 = 0.03;
const TEST_PATH_PENALTY: f32 = 0.05;

fn is_test_path(path: &str) -> bool {
    path.contains("__tests__")
        || path.contains(".test.")
        || path.contains(".spec.")
        || path.contains("/test/")
        || path.contains("/examples/")
}

/// Rerank results by score (name match, type hint, import rank, test penalty).
pub fn rerank(
    results: &mut [storage::SearchResult],
    type_hints: &[storage::ChunkType],
    import_counts: &std::collections::HashMap<String, u32>,
) {
    if results.is_empty() {
        return;
    }

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
            storage::MatchSource::Semantic => 1.0 / (1.0 + result.distance),
            storage::MatchSource::NameMatch => 0.5,
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
        let keywords = extract_keywords(&query_owned);
        let type_hints = extract_type_hints(&query_owned);

        let mut results = storage::search_similar(&conn, &query_embedding, limit, offset)?;

        if (results.len() as u32) < limit && !keywords.is_empty() {
            let type_filter = if type_hints.is_empty() { None } else { Some(type_hints.as_slice()) };

            let exclude_ids: HashSet<i64> = results
                .iter()
                .filter_map(|r| r.chunk_id)
                .collect();

            let remaining = limit - results.len() as u32;
            let keyword_refs: Vec<&str> = keywords.iter().map(|s| s.as_str()).collect();
            let fallback = storage::search_by_name(&conn, &keyword_refs, type_filter, &exclude_ids, remaining)?;
            results.extend(fallback);
        }

        // Rerank with multiple signals
        let file_paths: Vec<&str> = results.iter().map(|r| r.chunk.file_path.as_str()).collect();
        let import_counts = storage::get_import_counts(&conn, &file_paths)?;
        rerank(&mut results, &type_hints, &import_counts);

        Ok::<_, StorageError>(results)
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
        assert_eq!(kw, vec!["streaming", "chat", "hooks"]);
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
        assert_eq!(kw, vec!["useauth", "component"]);
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
        rerank(&mut results, &[], &HashMap::new());
        let expected = 1.0 / (1.0 + 0.5);
        assert!((results[0].score - expected).abs() < 1e-6,
            "expected {expected}, got {}", results[0].score);
    }

    #[test]
    fn rerank_name_match_base_score() {
        let mut results = vec![
            make_result("src/A.tsx", "useAuth", storage::ChunkType::Hook, f32::INFINITY, storage::MatchSource::NameMatch),
        ];
        rerank(&mut results, &[], &HashMap::new());
        assert!((results[0].score - 0.55).abs() < 1e-6,
            "expected 0.55, got {}", results[0].score);
    }

    #[test]
    fn rerank_sorts_by_score_descending() {
        let mut results = vec![
            make_result("src/B.tsx", "B", storage::ChunkType::Component, 0.5, storage::MatchSource::Semantic),
            make_result("src/A.tsx", "A", storage::ChunkType::Component, 0.1, storage::MatchSource::Semantic),
        ];
        rerank(&mut results, &[], &HashMap::new());
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

        rerank(&mut with_hint, &[storage::ChunkType::Hook], &HashMap::new());
        rerank(&mut without_hint, &[], &HashMap::new());

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
        rerank(&mut results, &[], &HashMap::new());
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
        rerank(&mut results, &[], &import_counts);

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
        rerank(&mut test_result, &[], &HashMap::new());
        rerank(&mut normal_result, &[], &HashMap::new());

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
        rerank(&mut results, &[storage::ChunkType::TestCase], &HashMap::new());
        assert!((results[0].score - base).abs() < 1e-6,
            "TestCase query should exempt test penalty: expected {base}, got {}", results[0].score);
    }

    #[test]
    fn rerank_name_match_all_bonuses() {
        let mut results = vec![
            make_result("src/useAuth.tsx", "useAuth", storage::ChunkType::Hook, f32::INFINITY, storage::MatchSource::NameMatch),
        ];
        let import_counts = HashMap::from([("src/useAuth.tsx".to_string(), 10u32)]);
        rerank(&mut results, &[storage::ChunkType::Hook], &import_counts);
        assert!((results[0].score - 0.61).abs() < 1e-6,
            "expected 0.61, got {}", results[0].score);
    }

    #[test]
    fn rerank_combined_bonuses_semantic() {
        let mut results = vec![
            make_result("src/useAuth.tsx", "useAuth", storage::ChunkType::Hook, 0.1, storage::MatchSource::Semantic),
        ];
        let import_counts = HashMap::from([("src/useAuth.tsx".to_string(), 10u32)]);
        rerank(&mut results, &[storage::ChunkType::Hook], &import_counts);

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
        rerank(&mut results, &[storage::ChunkType::Hook], &import_counts);

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
