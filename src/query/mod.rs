use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::storage::{self, ChunkType, Db, SearchResult, StorageError};
use rurico::embed::{Embed, EmbedError};

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
    let mut base: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
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
    const S_DENY: &[&str] = &[
        "class", "this", "alias", "canvas", "focus", "status", "bus", "process", "address",
        "access", "express", "progress",
    ];

    let stems: Vec<String> = base
        .iter()
        .filter_map(|kw| {
            let stem = if kw.len() > 5 && kw.ends_with("ing") && !ING_DENY.contains(&kw.as_str()) {
                Some(&kw[..kw.len() - 3])
            } else if kw.len() > 3
                && kw.ends_with('s')
                && !kw.ends_with("ss")
                && !S_DENY.contains(&kw.as_str())
            {
                Some(&kw[..kw.len() - 1])
            } else {
                None
            };
            stem.filter(|s| s.len() >= 2 && !seen.contains(*s))
                .map(|s| s.to_string())
        })
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
    import_counts: &std::collections::HashMap<String, u32>,
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
            storage::MatchSource::NameMatch => {
                let ratio = keyword_hit_ratio(result, ctx.keywords, ctx.keyword_idfs, false);
                NAME_MATCH_BASE + NAME_MATCH_RATIO_WEIGHT * ratio
            }
            storage::MatchSource::ContentMatch => {
                let ratio = keyword_hit_ratio(result, ctx.keywords, ctx.keyword_idfs, true);
                CONTENT_MATCH_BASE + CONTENT_MATCH_RATIO_WEIGHT * ratio
            }
        };

        let name_bonus = if result.match_source == storage::MatchSource::NameMatch {
            NAME_MATCH_BONUS
        } else {
            0.0
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

        result.score = base + name_bonus + overlap_bonus + type_bonus + import_bonus - test_penalty;
    }

    results.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn search_pipeline(
    conn: &Db,
    query: &str,
    query_embedding: Option<&[f32]>,
    limit: u32,
    offset: u32,
) -> Result<Vec<SearchResult>, StorageError> {
    let keywords = extract_keywords(query);
    let type_hints = extract_type_hints(query);
    let keyword_refs: Vec<&str> = keywords.iter().map(|s| s.as_str()).collect();

    let stats = storage::get_stats(conn)?;
    let fetch_limit = limit.saturating_add(offset);
    let use_semantic = query_embedding.is_some() && stats.embedded_chunks > 0;
    let mut results = match query_embedding {
        Some(emb) if use_semantic => storage::search_similar(conn, emb, fetch_limit, 0)?,
        _ => Vec::new(),
    };

    let fallback_limit = fetch_limit * 3;

    if !keywords.is_empty() {
        let type_filter = if type_hints.is_empty() {
            None
        } else {
            Some(type_hints.as_slice())
        };

        let mut exclude_ids: HashSet<i64> = results.iter().filter_map(|r| r.chunk_id).collect();
        let name_results = storage::search_by_name(
            conn,
            &keyword_refs,
            type_filter,
            &exclude_ids,
            fallback_limit,
        )?;
        exclude_ids.extend(name_results.iter().filter_map(|r| r.chunk_id));
        results.extend(name_results);
        let content_results = storage::search_by_content(
            conn,
            &keyword_refs,
            type_filter,
            &exclude_ids,
            fallback_limit,
        )?;
        results.extend(content_results);
    }
    let embed_coverage = if stats.embeddable_chunks > 0 {
        stats.embedded_chunks as f32 / stats.embeddable_chunks as f32
    } else {
        0.0
    };

    let dfs = storage::get_keyword_doc_frequencies(conn, &keyword_refs, stats.total_chunks)?;
    let total = stats.total_chunks.max(1) as f32;
    let keyword_idfs: Vec<f32> = dfs
        .iter()
        .map(|&df| (total / (df.max(1) as f32)).ln())
        .collect();

    let file_paths: Vec<&str> = results.iter().map(|r| r.chunk.file_path.as_str()).collect();
    let import_counts = storage::get_import_counts(conn, &file_paths)?;
    let ctx = RerankContext {
        type_hints: &type_hints,
        keywords: &keywords,
        keyword_idfs: &keyword_idfs,
        embed_coverage,
    };
    rerank(&mut results, &ctx, &import_counts);
    cap_per_file(&mut results, MAX_RESULTS_PER_FILE);
    if offset > 0 {
        let skip = std::cmp::min(offset as usize, results.len());
        results.drain(..skip);
    }
    results.truncate(limit as usize);

    Ok(results)
}

pub fn search(
    conn: Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    query: &str,
    limit: u32,
    offset: u32,
) -> Result<SearchOutcome, QueryError> {
    let (query_embedding, degraded) = match embedder.embed_query(query) {
        Ok(emb) => (Some(emb), false),
        Err(e) => {
            tracing::warn!(error = %e, "Query embedding failed, falling back to text search");
            (None, true)
        }
    };

    let conn = conn.lock().unwrap();
    let results = search_pipeline(&conn, query, query_embedding.as_deref(), limit, offset)?;
    Ok(SearchOutcome { results, degraded })
}

#[cfg(test)]
mod tests;
