mod embedder;
mod format;
mod reranker;

use std::collections::{HashMap, HashSet};
use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use amici::cli::embed_with_spinners;
use amici::model::{ModelLoad, download_and_verify_model};
use rurico::embed::Embed;
use rurico::reranker::Rerank;

use crate::config;
use crate::indexer;
use crate::query::{self, QueryError};
use crate::storage;

#[cfg(any(test, feature = "test-support"))]
use embedder::DEFAULT_EMBED_BUDGET;
use embedder::{DegradedReason, degraded_reason_user_note, parse_embed_budget};
use format::{
    EnrichmentContext, format_coverage, format_coverage_note, format_dry_run_json,
    format_embed_result, format_impact_all, format_impact_json, format_impact_results,
    format_index_json, format_no_results_message, format_rebuild_json, format_results_grouped,
    format_results_json, format_status_json,
};

const SEMANTIC_THRESHOLD: f32 = 0.7;

const INDEX_FRESHNESS_SECS: u32 = 60;
const MAX_QUERY_LENGTH: usize = 2000;

pub const MAX_SEARCH_LIMIT: u32 = 100;
pub const MAX_SEARCH_OFFSET: u32 = 500;
pub const MAX_IMPACT_DEPTH: u32 = 10;

#[derive(Debug, Clone, Copy, PartialEq)]
enum IndexState {
    Empty,
    ChunkedOnly,
    PartiallyEmbedded,
    FullyEmbedded,
}

fn determine_index_state(stats: &storage::IndexStatus) -> IndexState {
    if stats.total_chunks == 0 {
        IndexState::Empty
    } else if stats.embedded_chunks == 0 {
        IndexState::ChunkedOnly
    } else if stats.embedded_chunks < stats.embeddable_chunks {
        IndexState::PartiallyEmbedded
    } else {
        IndexState::FullyEmbedded
    }
}

fn validate_path(path: &str) -> Result<(), YomuError> {
    if path.contains("..") || Path::new(path).is_absolute() {
        return Err(YomuError::InvalidInput(format!(
            "'{path}' must be a relative path and must not contain '..'"
        )));
    }
    Ok(())
}

pub(crate) fn parse_impact_target(target: &str) -> (&str, Option<&str>) {
    if let Some(colon_pos) = target.rfind(':')
        && colon_pos > 0
        && colon_pos < target.len() - 1
    {
        let file = &target[..colon_pos];
        let symbol = &target[colon_pos + 1..];
        if !symbol.contains('/') && !symbol.contains('\\') {
            return (file, Some(symbol));
        }
    }
    (target, None)
}

#[derive(Debug, thiserror::Error)]
pub enum YomuError {
    #[error("storage error: {0}")]
    Storage(#[from] storage::StorageError),
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Index(#[from] indexer::IndexError),
    #[error("internal error: {0}")]
    Internal(String),
    #[error("query error: {0}")]
    Query(#[from] QueryError),
    #[error("{0}")]
    EmbedderUnavailable(String),
}

pub struct Yomu {
    conn: Arc<Mutex<storage::Db>>,
    embedder: OnceLock<Result<Arc<dyn Embed>, DegradedReason>>,
    root: PathBuf,
    embed_budget: u32,
    embed_disabled: bool,
    rerank_enabled: bool,
    reranker: OnceLock<ModelLoad<Box<dyn Rerank>>>,
}

impl Yomu {
    pub fn new() -> Result<Self, YomuError> {
        let cwd = env::current_dir()?;
        let root = config::detect_root(&cwd);
        Self::with_root(root)
    }

    pub fn with_root(root: PathBuf) -> Result<Self, YomuError> {
        tracing::info!(root = %root.display(), "Detected project root");
        let db_path = root.join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path)?;

        let embed_disabled = env::var("YOMU_EMBED").as_deref() == Ok("0");
        let rerank_enabled = env::var("YOMU_RERANK").as_deref() == Ok("1");
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            embedder: OnceLock::new(),
            root,
            embed_budget: parse_embed_budget(),
            embed_disabled,
            rerank_enabled,
            reranker: OnceLock::new(),
        })
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn for_test(conn: storage::Db, root: PathBuf, embedder: Option<Arc<dyn Embed>>) -> Self {
        let state = match embedder {
            Some(e) => Ok(e),
            None => Err(DegradedReason::NotInstalled),
        };
        let embedder_lock = OnceLock::new();
        let _ = embedder_lock.set(state);
        Self {
            conn: Arc::new(Mutex::new(conn)),
            embedder: embedder_lock,
            root,
            embed_budget: DEFAULT_EMBED_BUDGET,
            embed_disabled: false,
            rerank_enabled: false,
            reranker: OnceLock::new(),
        }
    }

    pub fn search(
        &self,
        query: Option<&str>,
        limit: u32,
        offset: u32,
        paths: &[String],
        json: bool,
        from_target: Option<&str>,
    ) -> Result<String, YomuError> {
        if let Some(q) = query {
            if q.is_empty() {
                return Err(YomuError::InvalidInput("query must not be empty".into()));
            }
            if q.len() > MAX_QUERY_LENGTH {
                return Err(YomuError::InvalidInput(format!(
                    "query exceeds maximum length of {MAX_QUERY_LENGTH} characters"
                )));
            }
        }

        for path in paths {
            validate_path(path)?;
        }

        let limit = limit.min(MAX_SEARCH_LIMIT);

        if let Some(from) = from_target {
            // FR-006: --offset is intentionally ignored in from-file mode
            return self.search_from(from, query, limit, paths, json);
        }

        let query =
            query.ok_or_else(|| YomuError::InvalidInput("query or --from is required".into()))?;

        let embedder = self.get_embedder();
        let offset = offset.min(MAX_SEARCH_OFFSET);

        tracing::debug!(query, limit, offset, ?paths, "search request");

        let type_hints = query::extract_type_hints(query);
        let hints_ref = if type_hints.is_empty() {
            None
        } else {
            Some(type_hints.as_slice())
        };

        let stats = self.with_db(storage::get_stats)?;
        let state = determine_index_state(&stats);
        let index_notes = self.ensure_indexed(embedder, state, hints_ref)?;

        let outcome = query::search(
            &self.conn,
            embedder,
            query,
            limit,
            offset,
            self.get_reranker(),
            paths,
        )?;

        let mut notes: Vec<String> = Vec::new();
        if let Some(msg) = index_notes {
            notes.push(msg);
        }
        if let Some(note) = self.reranker_note() {
            notes.push(note);
        }
        if let Some(reason) = self.degraded_reason() {
            if let Some(note) = degraded_reason_user_note(*reason) {
                notes.push(note.to_owned());
            }
        } else if outcome.degraded {
            notes.push("embedding model not loaded; results from text search only".into());
        }

        self.format_search_results(&outcome.results, &stats, notes, json, outcome.degraded)
    }

    fn search_from(
        &self,
        from: &str,
        query: Option<&str>,
        limit: u32,
        paths: &[String],
        json: bool,
    ) -> Result<String, YomuError> {
        let (file, symbol) = parse_impact_target(from);
        validate_path(file)?;

        let embedder = self.get_embedder();
        let stats = self.with_db(storage::get_stats)?;
        let state = determine_index_state(&stats);
        let index_notes = self.ensure_indexed(embedder, state, None)?;

        let (chunk_ids, embedding_bytes) = self.with_db(|c| {
            let chunk_ids = storage::get_chunks_for_from_target(c, file, symbol)?;
            let raw = storage::get_sub_embeddings_for_chunks(c, &chunk_ids)?;
            let embedding_bytes: Vec<Vec<u8>> = raw.into_iter().map(|(_, b)| b).collect();
            Ok((chunk_ids, embedding_bytes))
        })?;

        let mut notes: Vec<String> = Vec::new();
        if let Some(msg) = index_notes {
            notes.push(msg);
        }

        let results = if embedding_bytes.is_empty() {
            tracing::warn!(from, "no stored embeddings for from-target");
            notes.push(format!(
                "no stored embeddings for '{from}'; try running `yomu index`"
            ));
            Vec::new()
        } else {
            let source_ids: HashSet<i64> = chunk_ids.into_iter().collect();
            self.with_db(|conn| {
                query::search_from_file(conn, &embedding_bytes, &source_ids, query, limit, paths)
            })?
        };

        self.format_search_results(&results, &stats, notes, json, false)
    }

    fn format_search_results(
        &self,
        results: &[storage::SearchResult],
        stats: &storage::IndexStatus,
        notes: Vec<String>,
        json: bool,
        degraded: bool,
    ) -> Result<String, YomuError> {
        if json {
            return Ok(format_results_json(results, degraded, notes));
        }
        if results.is_empty() {
            let mut msg = format_no_results_message(stats);
            for note in &notes {
                msg.push_str(&format!("\n\nNote: {note}"));
            }
            return Ok(msg);
        }
        let ctx = self.fetch_enrichment_context(results)?;
        let parent_chunks = self.fetch_parent_chunks(results)?;
        let mut text = format_results_grouped(results, &ctx, &parent_chunks);
        for note in &notes {
            text.push_str(&format!("\n---\nNote: {note}\n"));
        }
        Ok(text)
    }

    pub fn index(&self, json: bool) -> Result<String, YomuError> {
        let chunk_result = indexer::run_chunk_only_index(&self.conn, &self.root)?;
        let stats = self.with_db(storage::get_stats)?;

        if json {
            return Ok(format_index_json(&chunk_result, &stats));
        }

        let mut text = format!(
            "Indexing complete: {} files chunked, {} chunks created, {} files skipped (unchanged), {} errors",
            chunk_result.files_processed,
            chunk_result.chunks_created,
            chunk_result.files_skipped,
            chunk_result.files_errored,
        );
        if let Some(note) = format_coverage_note(&stats) {
            text.push_str(&note);
        }
        Ok(text)
    }

    pub fn dry_run_index(&self, force: bool, json: bool) -> Result<String, YomuError> {
        let preview = indexer::dry_run_index(&self.conn, &self.root, force)?;

        if json {
            return Ok(format_dry_run_json(&preview));
        }

        let mut text = format!(
            "Dry run: {} files to process, {} files unchanged (skip), {} total files",
            preview.files_to_process, preview.files_to_skip, preview.total_files,
        );
        if preview.files_errored > 0 {
            text.push_str(&format!(", {} errors", preview.files_errored));
        }
        if preview.orphans_to_remove > 0 {
            text.push_str(&format!(
                ", {} orphaned files to remove",
                preview.orphans_to_remove
            ));
        }
        Ok(text)
    }

    pub fn rebuild(&self, json: bool) -> Result<String, YomuError> {
        let chunk_result = indexer::run_chunk_only_index_force(&self.conn, &self.root)?;
        let stats = self.with_db(storage::get_stats)?;

        if json {
            return Ok(format_rebuild_json(&chunk_result, &stats));
        }

        let mut text = format!(
            "Rebuild complete: {} files chunked, {} chunks created, {} errors",
            chunk_result.files_processed, chunk_result.chunks_created, chunk_result.files_errored,
        );
        if let Some(note) = format_coverage_note(&stats) {
            text.push_str(&note);
        }
        Ok(text)
    }

    pub fn impact(
        &self,
        target: &str,
        symbol: Option<&str>,
        depth: u32,
        json: bool,
        semantic: bool,
    ) -> Result<String, YomuError> {
        if target.is_empty() {
            return Err(YomuError::InvalidInput("target must not be empty".into()));
        }

        let stats = self.with_db(storage::get_stats)?;
        if stats.total_chunks == 0 {
            return Err(YomuError::InvalidInput(
                "index is empty — run `yomu index` first, or use `yomu search` which auto-indexes"
                    .into(),
            ));
        }

        let (file_path, parsed_symbol) = parse_impact_target(target);

        validate_path(file_path)?;

        let symbol_filter = symbol.or(parsed_symbol);
        let max_depth = depth.min(MAX_IMPACT_DEPTH);
        let fp = file_path.to_owned();
        let sym_owned = symbol_filter.map(str::to_owned);

        let (file_in_index, dependents, symbol_refs) = self.with_db(move |conn| {
            let exists = storage::file_exists_in_index(conn, &fp)?;
            let dependents = storage::get_transitive_dependents(conn, &fp, max_depth)?;
            let refs = match &sym_owned {
                Some(sym) => storage::get_symbol_dependents(conn, &fp, sym)?,
                None => vec![],
            };
            Ok((exists, dependents, refs))
        })?;

        let semantic_related = if semantic {
            let state = determine_index_state(&stats);
            self.semantic_search(file_path, symbol_filter, state)?
        } else {
            vec![]
        };

        if json {
            return Ok(format_impact_json(
                target,
                file_in_index,
                &dependents,
                &symbol_refs,
                &semantic_related,
            ));
        }

        if dependents.is_empty() && semantic_related.is_empty() {
            return Ok(if file_in_index {
                format!("No dependents found for `{}`.", target)
            } else {
                format!(
                    "`{}` not found in index. Run `yomu index` to update.",
                    file_path
                )
            });
        }

        let text = if symbol_filter.is_some() {
            format_impact_results(target, &symbol_refs, &dependents, &semantic_related)
        } else {
            format_impact_all(target, &dependents, &semantic_related)
        };

        Ok(text)
    }

    pub fn status(&self, json: bool) -> Result<String, YomuError> {
        let (stats, ref_count) = self.with_db(|conn| {
            let stats = storage::get_stats(conn)?;
            let ref_count = storage::get_reference_count(conn)?;
            Ok((stats, ref_count))
        })?;

        if json {
            return Ok(format_status_json(&stats, ref_count));
        }

        Ok(format!(
            "Index status:\n  Files: {}\n  Chunks: {}\n  Embedded: {}\n  References: {}\n  Last indexed: {}",
            stats.total_files,
            stats.total_chunks,
            format_coverage(&stats),
            ref_count,
            stats.last_indexed_at.as_deref().unwrap_or("never")
        ))
    }

    pub fn embed(&self, json: bool) -> Result<String, YomuError> {
        let pending = self.with_db(|conn| {
            let stats = storage::get_stats(conn)?;
            Ok(stats
                .embeddable_chunks
                .saturating_sub(stats.embedded_chunks))
        })?;

        let result = embed_with_spinners(
            pending,
            |_| {
                self.try_embedder_arc().map_err(|reason| {
                    let msg = match reason {
                        DegradedReason::Disabled => "embedding is disabled (YOMU_EMBED=0)",
                        DegradedReason::NotInstalled => "embedding model not installed",
                        DegradedReason::BackendUnavailable | DegradedReason::ProbeFailed => {
                            "embedding model unavailable"
                        }
                    };
                    YomuError::EmbedderUnavailable(msg.to_owned())
                })
            },
            |r: &indexer::EmbedResult| format!("Embedded {} chunks", r.chunks_embedded),
            |model: Arc<dyn Embed>, update| {
                indexer::run_incremental_embed_with_progress(
                    &self.conn,
                    model.as_ref(),
                    u32::MAX,
                    None,
                    |n| update(&format!("Embedding... {n}/{pending} chunks")),
                )
                .map_err(YomuError::from)
            },
        )?;

        match result {
            Some(r) => Ok(format_embed_result(&r, json)),
            None => Ok(format_embed_result(&indexer::EmbedResult::default(), json)),
        }
    }
}

impl Yomu {
    fn with_db<T, F>(&self, f: F) -> Result<T, YomuError>
    where
        F: FnOnce(&storage::Db) -> Result<T, storage::StorageError>,
    {
        let conn = self.conn.lock().unwrap();
        f(&conn).map_err(YomuError::from)
    }

    fn check_index_fresh(&self) -> bool {
        match self.with_db(|conn| storage::is_index_fresh(conn, INDEX_FRESHNESS_SECS)) {
            Ok(fresh) => fresh,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to check index freshness, assuming stale");
                false
            }
        }
    }

    fn try_rechunk(&self) -> Option<String> {
        match indexer::run_chunk_only_index(&self.conn, &self.root) {
            Ok(r) if r.files_errored > 0 => Some(format!(
                "{} files had errors during re-indexing",
                r.files_errored
            )),
            Ok(_) => None,
            Err(e) => Some(format!("re-chunking failed: {e}")),
        }
    }

    fn handle_empty_index(&self) -> Result<bool, YomuError> {
        tracing::info!("Index is empty, running chunk-only index");
        indexer::run_chunk_only_index(&self.conn, &self.root)?;
        Ok(true)
    }

    fn rechunk_if_stale(&self) -> Option<String> {
        if self.check_index_fresh() {
            return None;
        }
        self.try_rechunk()
    }

    fn handle_fully_embedded(&self) -> Result<(bool, Option<String>), YomuError> {
        if self.check_index_fresh() {
            return Ok((false, None));
        }
        let note = self.try_rechunk();
        let stats = self.with_db(storage::get_stats)?;
        Ok((stats.embedded_chunks < stats.embeddable_chunks, note))
    }

    fn ensure_indexed(
        &self,
        embedder: &dyn Embed,
        state: IndexState,
        type_hints: Option<&[storage::ChunkType]>,
    ) -> Result<Option<String>, YomuError> {
        let (needs_embed, rechunk_note) = match state {
            IndexState::Empty => (self.handle_empty_index()?, None),
            IndexState::ChunkedOnly | IndexState::PartiallyEmbedded => {
                let note = self.rechunk_if_stale();
                (true, note)
            }
            IndexState::FullyEmbedded => self.handle_fully_embedded()?,
        };

        let embed_note = if needs_embed && self.embedding_available() {
            match indexer::run_incremental_embed(
                &self.conn,
                embedder,
                self.embed_budget,
                type_hints,
            ) {
                Ok(_) => None,
                Err(e @ indexer::IndexError::Storage(_)) => return Err(e.into()),
                Err(e) => {
                    tracing::warn!(error = %e, "Incremental embed failed, searching existing embeddings");
                    Some(format!("embedding failed: {e}"))
                }
            }
        } else {
            None
        };

        let notes: Vec<&str> = [rechunk_note.as_deref(), embed_note.as_deref()]
            .into_iter()
            .flatten()
            .collect();

        Ok(if notes.is_empty() {
            None
        } else {
            Some(notes.join("; "))
        })
    }

    fn fetch_enrichment_context(
        &self,
        results: &[storage::SearchResult],
    ) -> Result<EnrichmentContext, YomuError> {
        let unique_paths: Vec<String> = {
            let mut seen = HashSet::new();
            results
                .iter()
                .filter(|r| seen.insert(&r.chunk.file_path))
                .map(|r| r.chunk.file_path.clone())
                .collect()
        };
        self.with_db(move |conn| {
            let path_refs: Vec<&str> = unique_paths.iter().map(String::as_str).collect();
            let imports = storage::get_file_contexts(conn, &path_refs)?;
            let siblings = storage::get_file_siblings(conn, &path_refs)?;
            Ok(EnrichmentContext { imports, siblings })
        })
    }

    fn fetch_parent_chunks(
        &self,
        results: &[storage::SearchResult],
    ) -> Result<HashMap<i64, storage::Chunk>, YomuError> {
        let parent_ids: Vec<i64> = results
            .iter()
            .filter_map(|r| r.chunk.parent_chunk_id)
            .collect();
        if parent_ids.is_empty() {
            return Ok(HashMap::new());
        }
        self.with_db(move |conn| storage::get_chunks_by_ids(conn, &parent_ids))
    }

    fn semantic_search(
        &self,
        file_path: &str,
        symbol: Option<&str>,
        state: IndexState,
    ) -> Result<Vec<storage::SearchResult>, YomuError> {
        let embedder = self.get_embedder();
        self.ensure_indexed(embedder, state, None)?;

        let fp = file_path.to_owned();
        let sym = symbol.map(str::to_owned);
        let mut results = self.with_db(move |c| {
            let ids = storage::get_chunks_for_from_target(c, &fp, sym.as_deref())?;
            let bytes: Vec<Vec<u8>> = storage::get_sub_embeddings_for_chunks(c, &ids)?
                .into_iter()
                .map(|(_, b)| b)
                .collect();
            if bytes.is_empty() {
                return Ok(vec![]);
            }
            let source_ids: HashSet<i64> = ids.into_iter().collect();
            query::search_from_file(c, &bytes, &source_ids, None, 20, &[])
        })?;
        results.retain(|r| r.score >= SEMANTIC_THRESHOLD);
        let mut seen: HashSet<String> = HashSet::new();
        results.retain(|r| seen.insert(r.chunk.file_path.clone()));
        Ok(results)
    }

    pub fn model_download(json: bool) -> Result<String, YomuError> {
        download_and_verify_model().map_err(|e| YomuError::Internal(e.to_string()))?;
        if json {
            Ok(serde_json::json!({"status": "ok"}).to_string())
        } else {
            Ok("Model downloaded and verified".to_owned())
        }
    }
}

#[cfg(test)]
mod tests;
