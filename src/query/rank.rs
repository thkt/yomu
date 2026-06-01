use std::collections::HashMap;

use crate::storage::{self, SearchResult};
use rurico::reranker::Rerank;

pub(super) fn keyword_hit_ratio(
    result: &storage::SearchResult,
    keywords: &[String],
    idfs: &[f32],
    check_content: bool,
) -> f32 {
    if keywords.is_empty() {
        return 0.0;
    }
    let name_lower = result.chunk.name.as_deref().unwrap_or("").to_lowercase();
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

pub(super) const MAX_RESULTS_PER_FILE: usize = 2;

pub(super) fn cross_encoder_rerank(
    results: &mut [SearchResult],
    query: &str,
    reranker: &dyn Rerank,
) {
    if results.is_empty() {
        return;
    }
    let pairs: Vec<(&str, &str)> = results
        .iter()
        .map(|r| (query, r.chunk.content.as_str()))
        .collect();
    match reranker.score_batch(&pairs) {
        Ok(scores) => {
            for (result, score) in results.iter_mut().zip(scores) {
                result.score = score;
            }
            results.sort_by(|a, b| b.score.total_cmp(&a.score));
        }
        Err(e) => {
            tracing::warn!(error = %e, "cross-encoder reranking failed, keeping heuristic order");
        }
    }
}

pub(super) fn cap_per_file(results: &mut Vec<storage::SearchResult>, max: usize) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    results.retain(|r| {
        let count = counts.entry(r.chunk.file_path.clone()).or_insert(0);
        *count += 1;
        *count <= max
    });
}

pub(super) const TYPE_HINT_BONUS: f32 = 0.03;
pub(super) const IMPORT_RANK_BONUS: f32 = 0.03;
const TEST_PATH_PENALTY: f32 = 0.05;
const SEMANTIC_KEYWORD_OVERLAP_BONUS: f32 = 0.05;

pub(super) fn is_test_path(path: &str) -> bool {
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

/// Confidence weight for semantic search scores.
/// Reaches 1.0 at 9% embedding coverage — below this, text-match scores
/// are boosted relative to vector scores to compensate for sparse embeddings.
/// At 0% coverage: 0.4 (semantic results still shown but heavily discounted).
pub(super) fn semantic_confidence(embed_coverage: f32) -> f32 {
    (embed_coverage.sqrt() * 2.0 + 0.4).min(1.0)
}

pub struct RerankContext<'a> {
    pub type_hints: &'a [storage::ChunkType],
    pub keywords: &'a [String],
    pub keyword_idfs: &'a [f32],
    pub embed_coverage: f32,
}

impl Default for RerankContext<'_> {
    fn default() -> Self {
        Self {
            type_hints: &[],
            keywords: &[],
            keyword_idfs: &[],
            embed_coverage: 1.0,
        }
    }
}

pub fn rerank(
    results: &mut [storage::SearchResult],
    ctx: &RerankContext<'_>,
    import_counts: &HashMap<String, u32>,
) {
    if results.is_empty() {
        return;
    }

    let confidence = semantic_confidence(ctx.embed_coverage);

    let mut counts: Vec<u32> = import_counts.values().copied().filter(|&c| c > 0).collect();
    counts.sort_unstable();
    let top_25_threshold = if counts.is_empty() {
        0
    } else {
        counts[counts.len().saturating_sub(counts.len() / 4 + 1)]
    };

    let query_wants_tests = ctx.type_hints.contains(&storage::ChunkType::TestCase);

    for result in results.iter_mut() {
        let base = match result.match_source {
            storage::MatchSource::Semantic => 1.0 / (1.0 + result.distance) * confidence,
            storage::MatchSource::Fts => result.score,
        };

        let overlap_bonus =
            if result.match_source == storage::MatchSource::Semantic && !ctx.keywords.is_empty() {
                SEMANTIC_KEYWORD_OVERLAP_BONUS
                    * keyword_hit_ratio(result, ctx.keywords, ctx.keyword_idfs, true)
            } else {
                0.0
            };

        let type_bonus =
            if !ctx.type_hints.is_empty() && ctx.type_hints.contains(&result.chunk.chunk_type) {
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

        result.score = base + overlap_bonus + type_bonus + import_bonus - test_penalty;
    }

    results.sort_by(|a, b| b.score.total_cmp(&a.score));
}
