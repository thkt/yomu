use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use crate::query_log::StageHit;
use crate::storage::{self, Db, MatchSource, SearchResult, StorageError};
use rurico::embed::{Embed, EmbedError};
use rurico::reranker::Rerank;

mod keywords;
mod rank;

pub use keywords::{extract_keywords, extract_type_hints};
pub use rank::{RerankContext, rerank};

#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    #[error("search query failed: {0}")]
    Storage(#[from] StorageError),
    #[error("search embedding failed: {0}")]
    Embed(#[from] EmbedError),
    #[error("internal task failed: {0}")]
    Internal(String),
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct SearchStages {
    pub fts_results: Vec<StageHit>,
    pub vec_results: Vec<StageHit>,
    pub rrf_results: Vec<StageHit>,
    pub reranked_results: Vec<StageHit>,
}

#[derive(Debug)]
pub struct SearchOutcome {
    pub results: Vec<SearchResult>,
    pub degraded: bool,
    pub stages: Option<SearchStages>,
}

fn capture_stage(results: &[SearchResult]) -> Vec<StageHit> {
    results
        .iter()
        .filter_map(|r| {
            r.chunk_id.map(|id| StageHit {
                chunk_id: id,
                score: r.score,
                source: match r.match_source {
                    MatchSource::Semantic => "semantic".to_owned(),
                    MatchSource::Fts => "fts".to_owned(),
                },
            })
        })
        .collect()
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
    let mut results = storage::vec_search_multi(conn, &emb_refs, fetch_limit, None, path_filter)?;

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

#[allow(clippy::too_many_arguments)]
fn search_pipeline(
    conn: &Db,
    query: &str,
    query_embedding: Option<&[f32]>,
    limit: u32,
    offset: u32,
    reranker: Option<&dyn Rerank>,
    path_filter: &[String],
    capture_stages: bool,
) -> Result<(Vec<SearchResult>, Option<SearchStages>), StorageError> {
    let keywords = extract_keywords(query);
    let type_hints = extract_type_hints(query);
    let keyword_refs: Vec<&str> = keywords.iter().map(String::as_str).collect();

    let stats = storage::get_stats(conn)?;
    let fetch_limit = if reranker.is_some() {
        limit.saturating_mul(4).saturating_add(offset)
    } else {
        limit.saturating_add(offset)
    };
    let type_filter = if type_hints.is_empty() {
        None
    } else {
        Some(type_hints.as_slice())
    };

    let mut stages = capture_stages.then(SearchStages::default);

    let use_semantic = query_embedding.is_some() && stats.embedded_chunks > 0;
    let mut results = match query_embedding {
        Some(emb) if use_semantic => {
            storage::vec_search(conn, emb, fetch_limit, type_filter, path_filter)?
        }
        _ => Vec::new(),
    };
    if let Some(s) = stages.as_mut() {
        s.vec_results = capture_stage(&results);
    }

    // FTS fallback is independent of reranker's semantic overfetch to avoid
    // handing an excessively large batch to the cross-encoder.
    let fallback_limit = limit.saturating_add(offset).saturating_mul(3);

    if !keywords.is_empty() {
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
        if let Some(s) = stages.as_mut() {
            s.fts_results = capture_stage(&fts_results);
        }
        results.extend(fts_results);
    }
    if let Some(s) = stages.as_mut() {
        s.rrf_results = capture_stage(&results);
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
        rank::cross_encoder_rerank(&mut results, query, ranker);
    }
    if let Some(s) = stages.as_mut() {
        s.reranked_results = capture_stage(&results);
    }
    rank::cap_per_file(&mut results, rank::MAX_RESULTS_PER_FILE);
    if offset > 0 {
        let skip = (offset as usize).min(results.len());
        results.drain(..skip);
    }
    results.truncate(limit as usize);

    Ok((results, stages))
}

#[allow(clippy::too_many_arguments)]
pub fn search(
    conn: &Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    query: &str,
    limit: u32,
    offset: u32,
    reranker: Option<&dyn Rerank>,
    path_filter: &[String],
    capture_stages: bool,
) -> Result<SearchOutcome, QueryError> {
    let (query_embedding, degraded) = match embedder.embed_query(query) {
        Ok(emb) => (Some(emb), false),
        Err(e) => {
            tracing::warn!(error = %e, "Query embedding failed, falling back to text search");
            (None, true)
        }
    };

    let conn = conn.lock().expect("DB lock poisoned (query::search)");
    let (results, stages) = search_pipeline(
        &conn,
        query,
        query_embedding.as_deref(),
        limit,
        offset,
        reranker,
        path_filter,
        capture_stages,
    )?;
    Ok(SearchOutcome {
        results,
        degraded,
        stages,
    })
}

#[cfg(test)]
mod tests;
