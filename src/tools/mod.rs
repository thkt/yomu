mod format;

use std::collections::HashSet;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;
use reqwest::Client;

use crate::config;
use crate::indexer::{self, embedder::Embed, embedder::EmbedError, embedder::Embedder};
use crate::query;
use crate::storage;

use format::{
    format_coverage, format_coverage_note, format_impact_all, format_impact_results,
    format_no_results_message, format_results_grouped, EnrichmentContext,
};

const DEFAULT_EMBED_BUDGET: u32 = 50;
const INDEX_FRESHNESS_SECS: u32 = 60;
const MIN_EMBED_BUDGET: u32 = 1;
const MAX_EMBED_BUDGET: u32 = 500;
const MAX_QUERY_LENGTH: usize = 2000;

pub const MAX_SEARCH_LIMIT: u32 = 100;
pub const MAX_SEARCH_OFFSET: u32 = 500;
pub const MAX_IMPACT_DEPTH: u32 = 10;

fn parse_embed_budget() -> u32 {
    parse_budget_value(std::env::var("YOMU_EMBED_BUDGET").ok().as_deref())
}

fn parse_budget_value(value: Option<&str>) -> u32 {
    match value {
        Some(v) => match v.parse::<u32>() {
            Ok(n) if (MIN_EMBED_BUDGET..=MAX_EMBED_BUDGET).contains(&n) => n,
            Ok(n) => {
                tracing::warn!(
                    value = n,
                    "YOMU_EMBED_BUDGET out of range ({MIN_EMBED_BUDGET}..={MAX_EMBED_BUDGET}), using default"
                );
                DEFAULT_EMBED_BUDGET
            }
            Err(_) => {
                tracing::warn!(value = %v, "Invalid YOMU_EMBED_BUDGET, using default");
                DEFAULT_EMBED_BUDGET
            }
        },
        None => DEFAULT_EMBED_BUDGET,
    }
}

#[derive(Debug, PartialEq)]
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
    } else if stats.embedded_chunks < stats.total_chunks {
        IndexState::PartiallyEmbedded
    } else {
        IndexState::FullyEmbedded
    }
}

struct NoOpEmbedder;

impl Embed for NoOpEmbedder {
    fn embed_query<'a>(
        &'a self,
        _text: &'a str,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<f32>, EmbedError>> + Send + 'a>> {
        Box::pin(async { Err(EmbedError::ApiKeyNotSet) })
    }
    fn embed_documents<'a>(
        &'a self,
        _texts: &'a [String],
    ) -> Pin<Box<dyn std::future::Future<Output = Result<Vec<Vec<f32>>, EmbedError>> + Send + 'a>>
    {
        Box::pin(async { Err(EmbedError::ApiKeyNotSet) })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum YomuError {
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("storage error: {0}")]
    Storage(#[from] storage::StorageError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    InvalidInput(String),
    #[error("{0}")]
    Index(#[from] indexer::IndexError),
    #[error("internal error: {0}")]
    Internal(String),
    #[error("query error: {0}")]
    Query(#[from] crate::query::QueryError),
}

pub struct Yomu {
    conn: Arc<Mutex<storage::Db>>,
    embedder: OnceLock<Option<Arc<dyn Embed>>>,
    root: PathBuf,
    embed_budget: u32,
}

impl Yomu {
    pub fn new() -> Result<Self, YomuError> {
        let cwd = std::env::current_dir()?;
        let root = config::detect_root(&cwd);
        Self::with_root(root)
    }

    pub fn with_root(root: PathBuf) -> Result<Self, YomuError> {
        tracing::info!(root = %root.display(), "Detected project root");
        let db_path = root.join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            embedder: OnceLock::new(),
            root,
            embed_budget: parse_embed_budget(),
        })
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn for_test(conn: storage::Db, root: PathBuf, embedder: Option<Arc<dyn Embed>>) -> Self {
        let embedder_lock = OnceLock::new();
        let _ = embedder_lock.set(embedder);
        Self {
            conn: Arc::new(Mutex::new(conn)),
            embedder: embedder_lock,
            root,
            embed_budget: parse_embed_budget(),
        }
    }

    pub async fn search(
        &self,
        query: &str,
        limit: u32,
        offset: u32,
    ) -> Result<String, YomuError> {
        if query.is_empty() {
            return Err(YomuError::InvalidInput("query must not be empty".into()));
        }
        if query.len() > MAX_QUERY_LENGTH {
            return Err(YomuError::InvalidInput(format!(
                "query exceeds maximum length of {MAX_QUERY_LENGTH} characters"
            )));
        }

        let embedder = self.get_embedder();
        let limit = limit.min(MAX_SEARCH_LIMIT);
        let offset = offset.min(MAX_SEARCH_OFFSET);

        tracing::debug!(query, limit, offset, "search request");

        let type_hints = query::extract_type_hints(query);
        let hints_ref = if type_hints.is_empty() {
            None
        } else {
            Some(type_hints.as_slice())
        };

        let stats = self.with_db(storage::get_stats).await?;
        let state = determine_index_state(&stats);
        let index_notes = self.ensure_indexed(embedder, state, hints_ref).await?;

        let outcome =
            query::search(Arc::clone(&self.conn), embedder, query, limit, offset).await?;

        let mut notes: Vec<String> = Vec::new();
        if let Some(msg) = index_notes {
            notes.push(msg);
        }
        if outcome.degraded {
            notes.push(
                "embedding API unavailable, results from text search only".into(),
            );
        }

        if outcome.results.is_empty() {
            let stats = self.with_db(storage::get_stats).await?;
            let mut msg = format_no_results_message(&stats);
            for note in &notes {
                msg.push_str(&format!("\n\nNote: {note}"));
            }
            return Ok(msg);
        }

        let ctx = self.fetch_enrichment_context(&outcome.results).await?;
        let mut text = format_results_grouped(&outcome.results, &ctx);
        for note in &notes {
            text.push_str(&format!("\n---\nNote: {note}\n"));
        }
        Ok(text)
    }

    pub async fn index(&self) -> Result<String, YomuError> {
        let chunk_result =
            indexer::run_chunk_only_index(Arc::clone(&self.conn), &self.root).await?;

        let stats = self.with_db(storage::get_stats).await?;
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

    pub async fn rebuild(&self) -> Result<String, YomuError> {
        let chunk_result =
            indexer::run_chunk_only_index_force(Arc::clone(&self.conn), &self.root).await?;

        let stats = self.with_db(storage::get_stats).await?;
        let mut text = format!(
            "Rebuild complete: {} files chunked, {} chunks created, {} errors",
            chunk_result.files_processed,
            chunk_result.chunks_created,
            chunk_result.files_errored,
        );
        if let Some(note) = format_coverage_note(&stats) {
            text.push_str(&note);
        }
        Ok(text)
    }

    pub async fn impact(
        &self,
        target: &str,
        symbol: Option<&str>,
        depth: u32,
    ) -> Result<String, YomuError> {
        if target.is_empty() {
            return Err(YomuError::InvalidInput("target must not be empty".into()));
        }

        let stats = self.with_db(storage::get_stats).await?;
        if stats.total_chunks == 0 {
            return Err(YomuError::InvalidInput(
                "index is empty — run `yomu index` first, or use `yomu search` which auto-indexes"
                    .into(),
            ));
        }

        let (file_path, parsed_symbol) = parse_impact_target(target);

        if file_path.contains("..") || std::path::Path::new(file_path).is_absolute() {
            return Err(YomuError::InvalidInput(
                "target path must be relative and must not contain '..'".into(),
            ));
        }

        let symbol_filter = symbol.or(parsed_symbol);
        let max_depth = depth.min(MAX_IMPACT_DEPTH);
        let fp = file_path.to_string();
        let sym_owned = symbol_filter.map(|s| s.to_string());

        let (file_in_index, dependents, symbol_refs) = self
            .with_db(move |conn| {
                let exists = storage::file_exists_in_index(conn, &fp)?;
                let dependents =
                    storage::get_transitive_dependents(conn, &fp, max_depth)?;
                let refs = match &sym_owned {
                    Some(sym) => storage::get_symbol_dependents(conn, &fp, sym)?,
                    None => vec![],
                };
                Ok((exists, dependents, refs))
            })
            .await?;

        if dependents.is_empty() {
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
            format_impact_results(target, &symbol_refs, &dependents)
        } else {
            format_impact_all(target, &dependents)
        };

        Ok(text)
    }

    pub async fn status(&self) -> Result<String, YomuError> {
        let (stats, ref_count) = self
            .with_db(|conn| {
                let stats = storage::get_stats(conn)?;
                let ref_count = storage::get_reference_count(conn)?;
                Ok((stats, ref_count))
            })
            .await?;

        Ok(format!(
            "Index status:\n  Files: {}\n  Chunks: {}\n  Embedded: {}\n  References: {}\n  Last indexed: {}",
            stats.total_files,
            stats.total_chunks,
            format_coverage(&stats),
            ref_count,
            stats.last_indexed_at.as_deref().unwrap_or("never")
        ))
    }
}

impl Yomu {
    async fn ensure_indexed(
        &self,
        embedder: &dyn Embed,
        state: IndexState,
        type_hints: Option<&[storage::ChunkType]>,
    ) -> Result<Option<String>, YomuError> {
        let (needs_embed, rechunk_note) = match state {
            IndexState::Empty => (self.handle_empty_index().await?, None),
            IndexState::ChunkedOnly | IndexState::PartiallyEmbedded => {
                let note = self.rechunk_if_stale().await;
                (true, note)
            }
            IndexState::FullyEmbedded => self.handle_fully_embedded().await?,
        };

        let embed_note = if needs_embed {
            match indexer::run_incremental_embed(
                Arc::clone(&self.conn),
                embedder,
                self.embed_budget,
                type_hints,
            )
            .await
            {
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

    async fn handle_empty_index(&self) -> Result<bool, YomuError> {
        tracing::info!("Index is empty, running chunk-only index");
        indexer::run_chunk_only_index(Arc::clone(&self.conn), &self.root).await?;
        Ok(true)
    }

    async fn try_rechunk(&self) -> Option<String> {
        match indexer::run_chunk_only_index(Arc::clone(&self.conn), &self.root).await {
            Ok(_) => None,
            Err(e) => Some(format!("re-chunking failed: {e}")),
        }
    }

    async fn rechunk_if_stale(&self) -> Option<String> {
        if self.check_index_fresh().await {
            return None;
        }
        self.try_rechunk().await
    }

    async fn handle_fully_embedded(&self) -> Result<(bool, Option<String>), YomuError> {
        if self.check_index_fresh().await {
            return Ok((false, None));
        }
        let note = self.try_rechunk().await;
        let stats = self.with_db(storage::get_stats).await?;
        Ok((stats.embedded_chunks < stats.total_chunks, note))
    }

    async fn check_index_fresh(&self) -> bool {
        match self
            .with_db(|conn| storage::is_index_fresh(conn, INDEX_FRESHNESS_SECS))
            .await
        {
            Ok(fresh) => fresh,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to check index freshness, assuming stale");
                false
            }
        }
    }

    fn get_embedder(&self) -> &dyn Embed {
        static NOOP: NoOpEmbedder = NoOpEmbedder;
        self.embedder
            .get_or_init(|| {
                let http = match Client::builder().build() {
                    Ok(h) => h,
                    Err(e) => {
                        tracing::warn!(error = %e, "HTTP client init failed, embedder unavailable");
                        return None;
                    }
                };
                Embedder::from_env(http)
                    .ok()
                    .map(|e| Arc::new(e) as Arc<dyn Embed>)
            })
            .as_deref()
            .unwrap_or(&NOOP)
    }

    async fn fetch_enrichment_context(
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
            let path_refs: Vec<&str> = unique_paths.iter().map(|s| s.as_str()).collect();
            let imports = storage::get_file_contexts(conn, &path_refs)?;
            let siblings = storage::get_file_siblings(conn, &path_refs)?;
            Ok(EnrichmentContext { imports, siblings })
        })
        .await
    }

    async fn with_db<T, F>(&self, f: F) -> Result<T, YomuError>
    where
        F: FnOnce(&storage::Db) -> Result<T, storage::StorageError> + Send + 'static,
        T: Send + 'static,
    {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            f(&conn)
        })
        .await
        .map_err(|e| YomuError::Internal(format!("task join failed: {e}")))?
        .map_err(YomuError::from)
    }
}

fn parse_impact_target(target: &str) -> (&str, Option<&str>) {
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

#[cfg(test)]
mod tests;
