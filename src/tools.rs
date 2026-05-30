mod embedder;
mod format;
mod reranker;

use std::collections::{HashMap, HashSet};
use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use amici::cli::embed_with_spinners;
use amici::cli::env_lookup;
use amici::cli::exit_code::CliError;
use amici::model::{ModelLoad, degrade_with_warn, download_and_verify_model, record_degraded};
use rurico::embed::Embed;
use rurico::reranker::Rerank;

use crate::brief;
use crate::config;
use crate::error::ErrorCode;
use crate::indexer;
use crate::query::{self, QueryError};
use crate::query_log::{self, QueryLogRecord};
use crate::recall::{self, corpus};
use crate::storage;

use embedder::{DegradedReason, degraded_reason_user_note};
use format::{
    EnrichmentContext, format_coverage, format_coverage_note, format_dry_run_json,
    format_impact_all, format_impact_json, format_impact_results, format_index_json,
    format_no_results_message, format_rebuild_json, format_results_grouped, format_results_json,
    format_status_json,
};

const SEMANTIC_THRESHOLD: f32 = 0.7;

const MAX_QUERY_LENGTH: usize = 2000;

pub const MAX_SEARCH_LIMIT: u32 = 100;
pub const MAX_SEARCH_OFFSET: u32 = 500;
pub const MAX_IMPACT_DEPTH: u32 = 10;
const BRIEF_MAX_INFERRED_SEEDS: u32 = 5;
// Production `brief` defaults (mirrors main.rs clap defaults); `recall` measures
// seed-less recall at the values agents actually run with.
const RECALL_DEPTH: u32 = 3;
const RECALL_MAX_CHUNKS: u32 = 80;
const RECALL_MAX_BYTES: u32 = 80_000;

/// Upper bound for the EmptyTarget `candidates` retry list emitted to agents.
/// A short, scannable list is the actionable shape per ADR-0060.
pub const MAX_EMPTY_TARGET_CANDIDATES: usize = 10;

/// Note guiding the user to build the index, or `None` when fully embedded.
/// Search routes are read-only, so this surfaces what the user must run manually.
fn index_hint(stats: &storage::IndexStatus) -> Option<String> {
    if stats.total_chunks == 0 {
        Some("index is empty; run `yomu index`".to_owned())
    } else if stats.embedded_chunks == 0 {
        Some("embeddings missing; run `yomu index`".to_owned())
    } else if stats.embedded_chunks < stats.embeddable_chunks {
        Some("embeddings incomplete; run `yomu index`".to_owned())
    } else {
        None
    }
}

fn validate_path(path: &str) -> Result<(), YomuError> {
    if path.contains("..") || Path::new(path).is_absolute() {
        return Err(YomuError::InvalidInput(InvalidInputKind::PathTraversal {
            path: path.to_owned(),
        }));
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

/// Routes with no current degraded condition (status, impact) per FR-006:
/// the constants are emitted so the envelope shape stays uniform across all
/// six JSON success routes.
fn never_degraded() -> (bool, Vec<String>) {
    (false, Vec::new())
}

/// degraded signal for chunking-mutation routes (index, rebuild) per FR-002 / FR-003.
/// Single source of truth for the note wording so production and tests stay in sync.
pub(super) fn degraded_for_chunk_errors(files_errored: u32) -> (bool, Vec<String>) {
    if files_errored > 0 {
        (
            true,
            vec![format!("failed to chunk {} file(s)", files_errored)],
        )
    } else {
        (false, Vec::new())
    }
}

/// degraded signal for index / rebuild: chunk errors (FR-002 / FR-003) plus an
/// embedding shortfall. `embed_pending` returns `Ok` even when recoverable
/// per-file embed failures skip some chunks, leaving `embedded_chunks <
/// embeddable_chunks`. Surface that as degraded so a JSON consumer keying on
/// the flag does not treat a half-embedded index as complete.
pub(super) fn degraded_for_index(
    files_errored: u32,
    stats: &storage::IndexStatus,
) -> (bool, Vec<String>) {
    let (mut degraded, mut notes) = degraded_for_chunk_errors(files_errored);
    if stats.embedded_chunks < stats.embeddable_chunks {
        degraded = true;
        notes.push(format!(
            "embedded {} of {} chunks; run `yomu index` again to finish",
            stats.embedded_chunks, stats.embeddable_chunks
        ));
    }
    (degraded, notes)
}

/// degraded signal for dry-run preview per FR-004.
pub(super) fn degraded_for_dry_run_errors(files_errored: u32) -> (bool, Vec<String>) {
    if files_errored > 0 {
        (
            true,
            vec![format!("check failed for {} file(s)", files_errored)],
        )
    } else {
        (false, Vec::new())
    }
}

/// Runtime options for [`Yomu::new`] / [`Yomu::with_root`].
///
/// Each field is OR-merged with the corresponding env var, so either source
/// can opt in.
#[derive(Debug, Default, Clone, Copy)]
pub struct YomuOptions {
    /// Opt in to query log JSONL output (`--log-query`). Default off per #182.
    pub log_query: bool,
}

/// Behavioral options for index / rebuild / dry-run paths.
///
/// `force` is consumed only by [`Yomu::dry_run_index`] (where it predicts
/// rebuild vs index semantics). [`Yomu::index`] and [`Yomu::rebuild`]
/// **ignore** `force` entirely — they are hard-wired to `false` and `true`
/// respectively. The struct is shared across all three for call-site parity
/// (Issue #206 / PR#3 Spec AS-307); callers of `index` / `rebuild` should
/// leave `force` at its default.
///
/// `json` stays a separate parameter because it controls presentation, not
/// indexing behavior.
#[derive(Debug, Default, Clone, Copy)]
pub struct IndexRunOptions {
    /// Predict rebuild semantics in [`Yomu::dry_run_index`]. Ignored by
    /// [`Yomu::index`] and [`Yomu::rebuild`].
    pub force: bool,
    /// Skip vendor directories (`node_modules`, `vendor`, etc.).
    pub exclude_vendor: bool,
}

/// Env-derived runtime configuration for [`Yomu::with_root`].
///
/// Production callers obtain this via [`YomuConfig::from_env`]; tests use
/// [`YomuConfig::from_env_with`] to inject a deterministic lookup closure.
#[derive(Debug, Clone, Copy)]
pub struct YomuConfig {
    pub rerank_enabled: bool,
}

impl YomuConfig {
    /// Read configuration from the process environment.
    pub fn from_env() -> Self {
        Self::from_env_with(env_lookup())
    }

    /// Read configuration through an injected lookup closure.
    pub fn from_env_with<F: Fn(&str) -> Option<String>>(get: F) -> Self {
        Self {
            rerank_enabled: get("YOMU_RERANK").as_deref() == Some("1"),
        }
    }
}

impl Default for YomuConfig {
    fn default() -> Self {
        Self::from_env_with(|_| None)
    }
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InvalidInputKind {
    #[error("'{path}' must be a relative path and must not contain '..'")]
    PathTraversal { path: String },
    #[error("query must not be empty")]
    EmptyQuery,
    #[error("query exceeds max length ({actual} > {max})")]
    QueryTooLong { max: usize, actual: usize },
    #[error("query or --from is required")]
    QueryOrFromRequired,
    #[error("target must not be empty")]
    EmptyTarget { candidates: Vec<String> },
    #[error("index is empty — run `yomu index` first, or use `yomu search` which auto-indexes")]
    EmptyIndex,
    #[error("task must not be empty")]
    EmptyTask,
    #[error("--seed-symbol is not yet implemented; use --seed-file")]
    SeedSymbolUnimplemented,
}

#[derive(Debug, thiserror::Error)]
pub enum YomuError {
    #[error("storage error: {0}")]
    Storage(#[from] storage::StorageError),
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error(transparent)]
    InvalidInput(#[from] InvalidInputKind),
    #[error("{0}")]
    Index(#[from] indexer::IndexError),
    #[error("internal error: {0}")]
    Internal(String),
    #[error("query error: {0}")]
    Query(#[from] QueryError),
    #[error("{0}")]
    EmbedderUnavailable(String),
}

impl InvalidInputKind {
    /// Kind-specific recommendation an AI agent can follow without parsing
    /// the message (FR-002, BR-002). Every variant carries a concrete next
    /// step; the optional layer lives at [`YomuError::next_step`].
    pub fn next_step(&self) -> String {
        match self {
            Self::PathTraversal { .. } => "use a relative path inside the project root".to_owned(),
            Self::EmptyQuery => {
                "run `yomu search \"<query>\"` or pipe a query via stdin".to_owned()
            }
            Self::QueryTooLong { max, .. } => format!("shorten query to ≤ {max} characters"),
            Self::QueryOrFromRequired => {
                "run `yomu search \"<query>\"` or `yomu search --from <file>`".to_owned()
            }
            Self::EmptyTarget { .. } => {
                "provide a target path, e.g. `yomu impact src/foo.rs`".to_owned()
            }
            Self::EmptyIndex => {
                "run `yomu index` first, or use `yomu search` which auto-indexes".to_owned()
            }
            Self::EmptyTask => {
                "provide a task description, e.g. `yomu brief \"add OAuth login\"`".to_owned()
            }
            Self::SeedSymbolUnimplemented => {
                "use `--seed-file` instead of `--seed-symbol`".to_owned()
            }
        }
    }
}

impl YomuError {
    /// Classifies this error for both exit-code routing and the
    /// `--json` envelope. Single source of truth per ADR-0066 Group 2.
    pub fn error_code(&self) -> ErrorCode {
        match self {
            Self::InvalidInput(_) => ErrorCode::UsageError,
            Self::Internal(_) | Self::Query(_) => ErrorCode::Internal,
            Self::Storage(_) | Self::Index(_) => ErrorCode::CantCreat,
            Self::Io(_) => ErrorCode::IoError,
            Self::EmbedderUnavailable(_) => ErrorCode::TempFailure,
        }
    }

    /// ADR-0060 next_step: kind-specific recommendation for the agent.
    /// `None` for variants without an actionable agent recommendation
    /// (`Io`, `Query`).
    pub fn next_step(&self) -> Option<String> {
        match self {
            Self::InvalidInput(kind) => Some(kind.next_step()),
            Self::Index(_) => Some("run `yomu rebuild` to recreate the index".to_owned()),
            Self::Storage(_) => {
                Some("check `.yomu/index.db` permissions and disk space".to_owned())
            }
            Self::Io(_) | Self::Query(_) => None,
            Self::EmbedderUnavailable(_) => {
                Some("run `yomu model download` to install the embedding model".to_owned())
            }
            Self::Internal(_) => Some(
                "file a bug report with the reproduction command and `RUST_BACKTRACE=1`".to_owned(),
            ),
        }
    }

    /// ADR-0060 candidates: file paths the agent can try next. Currently
    /// only `EmptyTarget` carries them (lexicographically-first
    /// `MAX_EMPTY_TARGET_CANDIDATES` indexed paths); all other variants
    /// return an empty vector (FR-001 / BR-004 / #197).
    pub fn candidates(&self) -> Vec<String> {
        match self {
            Self::InvalidInput(InvalidInputKind::EmptyTarget { candidates }) => candidates.clone(),
            _ => Vec::new(),
        }
    }

    /// ADR-0060 retryable: `true` only when immediate retry can recover
    /// (`EmbedderUnavailable` after a model download). All other variants
    /// require a different input or operator action (BR-001).
    pub fn retryable(&self) -> bool {
        matches!(self, Self::EmbedderUnavailable(_))
    }
}

impl CliError for YomuError {
    fn exit_code(&self) -> ExitCode {
        ExitCode::from(self.error_code().exit_code())
    }
}

pub struct Yomu {
    conn: Arc<Mutex<storage::Db>>,
    embedder: OnceLock<Result<Arc<dyn Embed>, DegradedReason>>,
    root: PathBuf,
    rerank_enabled: bool,
    reranker: OnceLock<ModelLoad<Box<dyn Rerank>>>,
    log_query: bool,
}

impl Yomu {
    pub fn new(options: YomuOptions) -> Result<Self, YomuError> {
        let cwd = env::current_dir()?;
        let root = config::detect_root(&cwd);
        Self::with_root(root, options, YomuConfig::from_env())
    }

    pub fn with_root(
        root: PathBuf,
        options: YomuOptions,
        config: YomuConfig,
    ) -> Result<Self, YomuError> {
        tracing::info!(root = %root.display(), "Detected project root");
        let db_path = root.join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            embedder: OnceLock::new(),
            root,
            rerank_enabled: config.rerank_enabled,
            reranker: OnceLock::new(),
            log_query: options.log_query,
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
            rerank_enabled: false,
            reranker: OnceLock::new(),
            log_query: false,
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
                return Err(YomuError::InvalidInput(InvalidInputKind::EmptyQuery));
            }
            if q.len() > MAX_QUERY_LENGTH {
                return Err(YomuError::InvalidInput(InvalidInputKind::QueryTooLong {
                    max: MAX_QUERY_LENGTH,
                    actual: q.len(),
                }));
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
            query.ok_or_else(|| YomuError::InvalidInput(InvalidInputKind::QueryOrFromRequired))?;

        let embedder = self.get_embedder();
        let offset = offset.min(MAX_SEARCH_OFFSET);

        tracing::debug!(query, limit, offset, ?paths, "search request");

        let stats = self.with_db(storage::get_stats)?;

        let start = Instant::now();
        let outcome = query::search(
            &self.conn,
            embedder,
            query,
            limit,
            offset,
            self.get_reranker(),
            paths,
            self.log_query,
        )?;
        if self.log_query {
            let latency_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
            self.emit_query_log(query, &outcome, latency_ms);
        }

        let mut notes: Vec<String> = Vec::new();
        if let Some(msg) = index_hint(&stats) {
            notes.push(msg);
        }
        if let Some(note) = self.reranker_note() {
            notes.push(note);
        }
        if let Some(reason) = self.degraded_reason() {
            if let Some(note) = degraded_reason_user_note(*reason, "yomu model download") {
                notes.push(note);
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

        let stats = self.with_db(storage::get_stats)?;

        let (chunk_ids, embedding_bytes) = self.with_db(|c| {
            let chunk_ids = storage::get_chunks_for_from_target(c, file, symbol)?;
            let raw = storage::get_sub_embeddings_for_chunks(c, &chunk_ids)?;
            let embedding_bytes: Vec<Vec<u8>> = raw.into_iter().map(|(_, b)| b).collect();
            Ok((chunk_ids, embedding_bytes))
        })?;

        let mut notes: Vec<String> = Vec::new();
        if let Some(msg) = index_hint(&stats) {
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

    pub fn index(&self, opts: IndexRunOptions, json: bool) -> Result<String, YomuError> {
        let chunk_result =
            indexer::run_chunk_only_index(&self.conn, &self.root, opts.exclude_vendor)?;
        self.embed_pending()?;
        let stats = self.with_db(storage::get_stats)?;

        if json {
            let (degraded, notes) = degraded_for_index(chunk_result.files_errored, &stats);
            return Ok(format_index_json(&chunk_result, &stats, degraded, notes));
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

    pub fn dry_run_index(&self, opts: IndexRunOptions, json: bool) -> Result<String, YomuError> {
        let preview =
            indexer::dry_run_index(&self.conn, &self.root, opts.force, opts.exclude_vendor)?;

        if json {
            let (degraded, notes) = degraded_for_dry_run_errors(preview.files_errored);
            return Ok(format_dry_run_json(&preview, degraded, notes));
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

    pub fn rebuild(&self, opts: IndexRunOptions, json: bool) -> Result<String, YomuError> {
        let chunk_result =
            indexer::run_chunk_only_index_force(&self.conn, &self.root, opts.exclude_vendor)?;
        self.embed_pending()?;
        let stats = self.with_db(storage::get_stats)?;

        if json {
            let (degraded, notes) = degraded_for_index(chunk_result.files_errored, &stats);
            return Ok(format_rebuild_json(&chunk_result, &stats, degraded, notes));
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

    /// Lexicographically-first `max` indexed file paths. Used to populate
    /// `EmptyTarget.candidates` when impact is invoked with an empty target
    /// (#197). Storage failures degrade to an empty vector so the primary
    /// `UsageError` code is preserved (FR-004 / BR-002). Ordering is
    /// alphabetical (not ranked) to keep results deterministic across runs.
    fn first_indexed_paths(&self, max: usize) -> Vec<String> {
        self.with_db(storage::get_all_file_paths)
            .map(|set| {
                let mut v: Vec<String> = set.into_iter().collect();
                v.sort();
                v.truncate(max);
                v
            })
            .unwrap_or_default()
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
            let candidates = self.first_indexed_paths(MAX_EMPTY_TARGET_CANDIDATES);
            return Err(YomuError::InvalidInput(InvalidInputKind::EmptyTarget {
                candidates,
            }));
        }

        let stats = self.with_db(storage::get_stats)?;
        if stats.total_chunks == 0 {
            return Err(YomuError::InvalidInput(InvalidInputKind::EmptyIndex));
        }

        let (file_path, parsed_symbol) = parse_impact_target(target);

        validate_path(file_path)?;

        let symbol_filter = symbol.or(parsed_symbol);
        let max_depth = depth.min(MAX_IMPACT_DEPTH);
        let fp = file_path.to_owned();
        let sym_owned = symbol_filter.map(str::to_owned);

        let (file_in_index, dependents, symbol_refs, direct_refs) = self.with_db(move |conn| {
            let exists = storage::file_exists_in_index(conn, &fp)?;
            let dependents = storage::get_transitive_dependents(conn, &fp, max_depth)?;
            let refs = match &sym_owned {
                Some(sym) => storage::get_symbol_dependents(conn, &fp, sym)?,
                None => vec![],
            };
            let direct = storage::get_direct_references(conn, &fp)?;
            let mut grouped: HashMap<String, Vec<storage::DirectReference>> = HashMap::new();
            for r in direct {
                grouped.entry(r.source_file.clone()).or_default().push(r);
            }
            Ok((exists, dependents, refs, grouped))
        })?;

        let semantic_related = if semantic {
            self.semantic_search(file_path, symbol_filter)?
        } else {
            vec![]
        };

        if json {
            let (degraded, notes) = never_degraded();
            return Ok(format_impact_json(
                target,
                file_in_index,
                &dependents,
                &direct_refs,
                &symbol_refs,
                &semantic_related,
                degraded,
                notes,
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
            let (degraded, notes) = never_degraded();
            return Ok(format_status_json(&stats, ref_count, degraded, notes));
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

    /// Embeds all pending chunks with progress spinners. Errors out when the
    /// model is unavailable so callers never silently leave a chunk-only index.
    fn embed_pending(&self) -> Result<(), YomuError> {
        let pending = self.with_db(|conn| {
            let stats = storage::get_stats(conn)?;
            Ok(stats
                .embeddable_chunks
                .saturating_sub(stats.embedded_chunks))
        })?;

        embed_with_spinners(
            pending,
            |_| {
                self.try_embedder_arc().map_err(|reason| {
                    // `Disabled` can no longer occur (yomu never disables
                    // embedding); it folds into the generic arm rather than an
                    // unreachable! that diff-coverage would flag as untested.
                    let msg = match reason {
                        DegradedReason::NotInstalled => {
                            "embedding model not installed; run `yomu model download` to enable semantic search"
                        }
                        _ => "embedding model unavailable",
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
        Ok(())
    }

    fn infer_seed_paths(&self, task: &str, max_seeds: u32) -> (Vec<String>, bool) {
        match self.embedder_seed_paths(task, max_seeds) {
            Ok(paths) => (paths, false),
            Err(reason) => {
                record_degraded(reason, "brief: seed inference");
                (self.fts_fallback_seed_paths(task, max_seeds), true)
            }
        }
    }

    fn embedder_seed_paths(
        &self,
        task: &str,
        max_seeds: u32,
    ) -> Result<Vec<String>, DegradedReason> {
        let embedder = self.try_embedder_arc()?;
        let task_emb = embedder.embed_query(task).map_err(degrade_with_warn(
            "brief seed inference: embed_query",
            DegradedReason::ProbeFailed,
        ))?;
        let conn = self
            .conn
            .lock()
            .expect("DB lock poisoned (embedder_seed_paths)");
        let results = storage::vec_search(&conn, &task_emb, max_seeds, None, &[]).map_err(
            degrade_with_warn(
                "brief seed inference: vec_search",
                DegradedReason::ProbeFailed,
            ),
        )?;
        drop(conn);

        Ok(dedupe_seed_paths(results, max_seeds as usize))
    }

    fn fts_fallback_seed_paths(&self, task: &str, max_seeds: u32) -> Vec<String> {
        let keywords = query::extract_keywords(task);
        if keywords.is_empty() {
            return Vec::new();
        }
        let keyword_refs: Vec<&str> = keywords.iter().map(String::as_str).collect();
        let oversample = max_seeds.saturating_mul(3);
        let conn = self
            .conn
            .lock()
            .expect("DB lock poisoned (fts_fallback_seed_paths)");
        let results = storage::search_by_fts(
            &conn,
            &keyword_refs,
            None,
            &HashSet::new(),
            None,
            oversample,
            &[],
        )
        .map_err(degrade_with_warn(
            "brief seed inference: fts fallback",
            DegradedReason::ProbeFailed,
        ))
        .unwrap_or_default();
        drop(conn);

        dedupe_seed_paths(results, max_seeds as usize)
    }

    /// Runs `brief` over `task`, inferring file seeds from `task.task` when none
    /// are given (seed-less), and returns the closure output with `degraded` set
    /// when seed inference fell back or the closure was empty. Shared by `brief`
    /// (renders) and `recall` (measures); callers validate `task` first.
    fn brief_output(&self, task: &brief::TaskBrief) -> Result<brief::BriefOutput, YomuError> {
        let mut effective = task.clone();
        let mut degraded = false;
        if effective.seeds.is_empty() {
            let (paths, seed_degraded) =
                self.infer_seed_paths(&effective.task, BRIEF_MAX_INFERRED_SEEDS);
            effective.seeds = paths
                .into_iter()
                .map(|value| brief::Seed {
                    kind: brief::SeedKind::File,
                    value,
                })
                .collect();
            degraded |= seed_degraded;
        }

        let mut output = self.with_db(|conn| brief::expand_plan(conn, &effective))?;
        output.degraded |= degraded;

        if output.chunks.is_empty() {
            tracing::warn!(
                seeds = effective.seeds.len(),
                degraded = output.degraded,
                "brief produced zero chunks"
            );
            output.degraded = true;
        }
        Ok(output)
    }

    pub fn brief(&self, task: &brief::TaskBrief, json: bool) -> Result<String, YomuError> {
        if task.task.trim().is_empty() {
            return Err(YomuError::InvalidInput(InvalidInputKind::EmptyTask));
        }
        if task
            .seeds
            .iter()
            .any(|s| matches!(s.kind, brief::SeedKind::Symbol))
        {
            return Err(YomuError::InvalidInput(
                InvalidInputKind::SeedSymbolUnimplemented,
            ));
        }

        let output = self.brief_output(task)?;
        Ok(if json {
            brief::render_json(&output)
        } else {
            brief::render_plain(&output)
        })
    }

    /// Measures seed-less recall and weighted cap-fit for every bundled GT entry
    /// whose repo matches `repo`, against the current index, and renders a
    /// per-entry plus aggregate report (FR-011). Returns the rendered text and the
    /// aggregate degraded flag. The caller exits non-zero when degraded (FR-012):
    /// an unavailable embedding model makes seed inference fall back and flag
    /// degraded, so a model-less run never reports a silent pass.
    pub fn recall(&self, repo: &str, json: bool) -> Result<(String, bool), YomuError> {
        let gt = corpus::load_bundled()
            .map_err(|e| YomuError::Internal(format!("bundled GT corpus: {e}")))?;
        let mut entries = Vec::new();
        for entry in gt.entries.iter().filter(|e| e.repo == repo) {
            let task = brief::TaskBrief {
                task: entry.task.clone(),
                seeds: Vec::new(),
                depth: RECALL_DEPTH,
                max_chunks: RECALL_MAX_CHUNKS,
                max_bytes: RECALL_MAX_BYTES,
                include_tests: false,
            };
            let output = self.brief_output(&task)?;
            let out_files: HashSet<String> =
                output.chunks.iter().map(|c| c.file_path.clone()).collect();
            let reachable: HashSet<String> = output.reachable_files.iter().cloned().collect();
            let mut report = recall::measure(&entry.must_include, &out_files, &reachable);
            report.degraded |= output.degraded;
            entries.push(recall::EntryReport {
                id: entry.id.clone(),
                report,
            });
        }
        let report = recall::CorpusReport::new(repo.to_owned(), entries);
        let text = if json {
            recall::render_recall_json(&report)
        } else {
            recall::render_recall_plain(&report)
        };
        Ok((text, report.aggregate.degraded))
    }
}

impl Yomu {
    fn with_db<T, F>(&self, f: F) -> Result<T, YomuError>
    where
        F: FnOnce(&storage::Db) -> Result<T, storage::StorageError>,
    {
        let conn = self.conn.lock().expect("DB lock poisoned (Yomu::with_db)");
        f(&conn).map_err(YomuError::from)
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
        self.with_db(move |conn| storage::get_chunks_by_ids(conn, &parent_ids, None, &[]))
    }

    fn semantic_search(
        &self,
        file_path: &str,
        symbol: Option<&str>,
    ) -> Result<Vec<storage::SearchResult>, YomuError> {
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

fn dedupe_seed_paths(results: Vec<storage::SearchResult>, cap: usize) -> Vec<String> {
    let mut paths = Vec::with_capacity(cap);
    let mut seen = HashSet::new();
    for r in results {
        if !seen.insert(r.chunk.file_path.clone()) {
            continue;
        }
        paths.push(r.chunk.file_path);
        if paths.len() >= cap {
            break;
        }
    }
    paths
}

impl Yomu {
    fn emit_query_log(&self, query: &str, outcome: &query::SearchOutcome, latency_ms: u64) {
        let Some(path) = query_log::resolve_log_path_from_env() else {
            tracing::warn!("query log path unresolved (HOME unset); skipping emit");
            return;
        };
        let timestamp = OffsetDateTime::now_utc()
            .format(&Rfc3339)
            .unwrap_or_default();
        let stages = outcome.stages.clone().unwrap_or_default();
        let record = QueryLogRecord {
            timestamp,
            yomu_version: env!("CARGO_PKG_VERSION").to_owned(),
            original_query: query.to_owned(),
            fts_results: stages.fts_results,
            vec_results: stages.vec_results,
            rrf_results: stages.rrf_results,
            reranked_results: stages.reranked_results,
            final_context_ids: outcome.results.iter().filter_map(|r| r.chunk_id).collect(),
            latency_ms,
        };
        match query_log::open_append_writer(&path) {
            Ok(mut writer) => {
                if let Err(e) = writer.write_record(&record) {
                    tracing::warn!(error = %e, "query log write failed");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, path = %path.display(), "query log open failed");
            }
        }
    }
}

#[cfg(test)]
mod tests;
