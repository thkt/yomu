mod briefing;
mod embedder;
mod format;
mod impact;
mod indexing;
mod model;
mod reranker;
mod search;
mod status;

use std::env;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::{Arc, Mutex, OnceLock};

use amici::cli::env_lookup;
use amici::cli::exit_code::CliError;
use amici::model::ModelLoad;
use rurico::embed::Embed;
use rurico::reranker::Rerank;

use crate::config;
use crate::error::ErrorCode;
use crate::indexer;
use crate::query::QueryError;
use crate::storage;

use embedder::DegradedReason;

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
    #[error("index is empty — run `yomu index` first")]
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
            Self::EmptyIndex => "run `yomu index` first".to_owned(),
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

    fn with_db<T, F>(&self, f: F) -> Result<T, YomuError>
    where
        F: FnOnce(&storage::Db) -> Result<T, storage::StorageError>,
    {
        let conn = self.conn.lock().expect("DB lock poisoned (Yomu::with_db)");
        f(&conn).map_err(YomuError::from)
    }
}

#[cfg(test)]
mod tests;
