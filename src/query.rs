use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use crate::storage::{self, ChunkType, Db, SearchResult, StorageError};
use crate::text::split_identifier;
use rurico::embed::{Embed, EmbedError};
use rurico::reranker::Rerank;

#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    #[error("search query failed: {0}")]
    Storage(#[from] StorageError),
    #[error("search embedding failed: {0}")]
    Embed(#[from] EmbedError),
    #[error("internal task failed: {0}")]
    Internal(String),
}

#[derive(Debug)]
pub struct SearchOutcome {
    pub results: Vec<SearchResult>,
    pub degraded: bool,
}

const STOP_WORDS: &[&str] = &["the", "a", "an", "in", "for", "of", "with", "and", "or"];

// Words ending in -ing that are not gerunds (stripping -ing produces a non-word).
const ING_DENY: &[&str] = &[
    "string",
    "bring",
    "thing",
    "nothing",
    "something",
    "everything",
    "ring",
    "king",
    "spring",
    "swing",
    "sing",
    "sting",
    "wing",
];

// Words ending in -s that are not plurals (stripping -s produces a non-word).
const S_DENY: &[&str] = &[
    "class", "this", "alias", "canvas", "focus", "status", "bus", "process", "address", "access",
    "express", "progress",
];

fn stem_keyword(kw: &str) -> Option<&str> {
    let stem = if kw.len() > 5 && kw.ends_with("ing") && !ING_DENY.contains(&kw) {
        Some(&kw[..kw.len() - 3])
    } else if kw.len() > 3 && kw.ends_with('s') && !kw.ends_with("ss") && !S_DENY.contains(&kw) {
        Some(&kw[..kw.len() - 1])
    } else {
        None
    };
    stem.filter(|s| s.len() >= 2)
}

pub fn extract_keywords(query: &str) -> Vec<String> {
    let mut base: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for token in query.split_whitespace() {
        let lower = token.to_lowercase();
        if lower.chars().count() >= 2
            && !STOP_WORDS.contains(&lower.as_str())
            && seen.insert(lower.clone())
        {
            base.push(lower);
        }
        for part in split_identifier(token) {
            let part_lower = part.to_lowercase();
            if part_lower.chars().count() >= 2
                && !STOP_WORDS.contains(&part_lower.as_str())
                && seen.insert(part_lower.clone())
            {
                base.push(part_lower);
            }
        }
    }

    let stems: Vec<String> = base
        .iter()
        .filter_map(|kw| stem_keyword(kw))
        .filter(|s| !seen.contains(*s))
        .map(str::to_owned)
        .collect();
    let mut all = base;
    for s in stems {
        seen.insert(s.clone());
        all.push(s);
    }
    all
}

pub fn extract_type_hints(query: &str) -> Vec<ChunkType> {
    let mut hints = Vec::new();
    for token in query.split_whitespace() {
        let token = token.to_lowercase();
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

fn keyword_hit_ratio(
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

const MAX_RESULTS_PER_FILE: usize = 2;

fn cross_encoder_rerank(results: &mut [SearchResult], query: &str, reranker: &dyn Rerank) {
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

fn cap_per_file(results: &mut Vec<storage::SearchResult>, max: usize) {
    let mut counts: HashMap<String, usize> = HashMap::new();
    results.retain(|r| {
        let count = counts.entry(r.chunk.file_path.clone()).or_insert(0);
        *count += 1;
        *count <= max
    });
}

const TYPE_HINT_BONUS: f32 = 0.03;
const IMPORT_RANK_BONUS: f32 = 0.03;
const TEST_PATH_PENALTY: f32 = 0.05;
const SEMANTIC_KEYWORD_OVERLAP_BONUS: f32 = 0.05;

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

/// Confidence weight for semantic search scores.
/// Reaches 1.0 at 9% embedding coverage — below this, text-match scores
/// are boosted relative to vector scores to compensate for sparse embeddings.
/// At 0% coverage: 0.4 (semantic results still shown but heavily discounted).
fn semantic_confidence(embed_coverage: f32) -> f32 {
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

const MAX_FROM_SUB_EMBEDDINGS: usize = 20;

#[allow(clippy::cast_possible_truncation)]
pub fn search_from_file(
    conn: &Db,
    embedding_bytes: &[Vec<u8>],
    source_chunk_ids: &HashSet<i64>,
    query: Option<&str>,
    limit: u32,
    path_filter: &[String],
) -> Result<Vec<SearchResult>, StorageError> {
    if embedding_bytes.is_empty() {
        return Ok(Vec::new());
    }

    // FR-010 / BR-005
    let capped = if embedding_bytes.len() > MAX_FROM_SUB_EMBEDDINGS {
        tracing::warn!(
            count = embedding_bytes.len(),
            cap = MAX_FROM_SUB_EMBEDDINGS,
            "Truncating sub-embeddings to cap"
        );
        &embedding_bytes[..MAX_FROM_SUB_EMBEDDINGS]
    } else {
        embedding_bytes
    };

    let f32_vecs: Vec<Vec<f32>> = capped
        .iter()
        .map(|bytes| {
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        })
        .collect();
    let emb_refs: Vec<&[f32]> = f32_vecs.iter().map(Vec::as_slice).collect();

    // FR-004 / BR-002: min-distance merge across sub-embeddings.
    // Overfetch by (a) 3× when FTS will narrow candidates, and (b) source_ids.len() to survive
    // the post-KNN source exclusion filter (P1: without this, limit=1 can return 0 results when
    // the source chunk occupies the only KNN slot).
    let source_buf = source_chunk_ids.len() as u32;
    let fetch_limit = if query.is_some() {
        limit.saturating_mul(3).saturating_add(source_buf)
    } else {
        limit.saturating_add(source_buf)
    };
    let mut results = storage::vec_search_multi(conn, &emb_refs, fetch_limit, path_filter)?;

    // FR-005 / BR-001
    results.retain(|r| !r.chunk_id.is_some_and(|id| source_chunk_ids.contains(&id)));

    // FR-008 / BR-006: FTS intersection — only when keywords can be extracted.
    // Skip when extract_keywords returns [] (stopwords-only input) to avoid silently clearing
    // valid semantic results (P2).
    if let Some(q) = query {
        let keywords = extract_keywords(q);
        if !keywords.is_empty() {
            let keyword_refs: Vec<&str> = keywords.iter().map(String::as_str).collect();
            let knn_ids: HashSet<i64> = results.iter().filter_map(|r| r.chunk_id).collect();
            let fts_hits = storage::search_by_fts(
                conn,
                &keyword_refs,
                None,
                &HashSet::new(),
                Some(&knn_ids),
                fetch_limit,
                path_filter,
            )?;
            let fts_ids: HashSet<i64> = fts_hits.iter().filter_map(|r| r.chunk_id).collect();
            results.retain(|r| r.chunk_id.is_some_and(|id| fts_ids.contains(&id)));
        }
    }

    // BR-003 / BR-004: rerank with default context (no keyword/type hints)
    let import_counts = if results.is_empty() {
        HashMap::new()
    } else {
        let file_paths: Vec<&str> = results.iter().map(|r| r.chunk.file_path.as_str()).collect();
        storage::get_import_counts(conn, &file_paths)?
    };
    rerank(&mut results, &RerankContext::default(), &import_counts);

    results.truncate(limit as usize);
    Ok(results)
}

fn search_pipeline(
    conn: &Db,
    query: &str,
    query_embedding: Option<&[f32]>,
    limit: u32,
    offset: u32,
    reranker: Option<&dyn Rerank>,
    path_filter: &[String],
) -> Result<Vec<SearchResult>, StorageError> {
    let keywords = extract_keywords(query);
    let type_hints = extract_type_hints(query);
    let keyword_refs: Vec<&str> = keywords.iter().map(String::as_str).collect();

    let stats = storage::get_stats(conn)?;
    let fetch_limit = if reranker.is_some() {
        limit.saturating_mul(4).saturating_add(offset)
    } else {
        limit.saturating_add(offset)
    };
    let use_semantic = query_embedding.is_some() && stats.embedded_chunks > 0;
    let mut results = match query_embedding {
        Some(emb) if use_semantic => storage::vec_search(conn, emb, fetch_limit, path_filter)?,
        _ => Vec::new(),
    };

    // FTS fallback is independent of reranker's semantic overfetch to avoid
    // handing an excessively large batch to the cross-encoder.
    let fallback_limit = limit.saturating_add(offset).saturating_mul(3);

    if !keywords.is_empty() {
        let type_filter = if type_hints.is_empty() {
            None
        } else {
            Some(type_hints.as_slice())
        };

        let exclude_ids: HashSet<i64> = results.iter().filter_map(|r| r.chunk_id).collect();
        let fts_results = storage::search_by_fts(
            conn,
            &keyword_refs,
            type_filter,
            &exclude_ids,
            None,
            fallback_limit,
            path_filter,
        )?;
        results.extend(fts_results);
    }
    let embed_coverage = stats.embed_coverage();

    let (keyword_idfs, import_counts) = if results.is_empty() {
        (Vec::new(), HashMap::new())
    } else {
        let dfs = storage::get_keyword_doc_frequencies(conn, &keyword_refs, stats.total_chunks)?;
        let total = stats.total_chunks.max(1) as f32;
        let idfs: Vec<f32> = dfs
            .iter()
            .map(|&df| (total / (df.max(1) as f32)).ln())
            .collect();
        let file_paths: Vec<&str> = results.iter().map(|r| r.chunk.file_path.as_str()).collect();
        let ic = storage::get_import_counts(conn, &file_paths)?;
        (idfs, ic)
    };
    let ctx = RerankContext {
        type_hints: &type_hints,
        keywords: &keywords,
        keyword_idfs: &keyword_idfs,
        embed_coverage,
    };
    rerank(&mut results, &ctx, &import_counts);
    if let Some(ranker) = reranker {
        cross_encoder_rerank(&mut results, query, ranker);
    }
    cap_per_file(&mut results, MAX_RESULTS_PER_FILE);
    if offset > 0 {
        let skip = (offset as usize).min(results.len());
        results.drain(..skip);
    }
    results.truncate(limit as usize);

    Ok(results)
}

pub fn search(
    conn: &Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    query: &str,
    limit: u32,
    offset: u32,
    reranker: Option<&dyn Rerank>,
    path_filter: &[String],
) -> Result<SearchOutcome, QueryError> {
    let (query_embedding, degraded) = match embedder.embed_query(query) {
        Ok(emb) => (Some(emb), false),
        Err(e) => {
            tracing::warn!(error = %e, "Query embedding failed, falling back to text search");
            (None, true)
        }
    };

    let conn = conn.lock().unwrap();
    let results = search_pipeline(
        &conn,
        query,
        query_embedding.as_deref(),
        limit,
        offset,
        reranker,
        path_filter,
    )?;
    Ok(SearchOutcome { results, degraded })
}

#[cfg(test)]
mod tests;
