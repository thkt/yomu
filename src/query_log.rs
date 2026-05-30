//! Append-only JSONL query log for search pipeline observability.
//!
//! Opt-in via the `--log-query` CLI flag (default off). Each `Yomu::search`
//! invocation appends 1 record to `$XDG_DATA_HOME/yomu/query_log.jsonl`
//! (or `$HOME/.local/share/yomu/query_log.jsonl` when the env var is unset).
//!
//! Schema is intentionally a superset of amici `QueryResult` so future
//! amici extraction can read yomu logs after a documented conversion
//! (`i64::to_string()` + `Some(_)` for chunk ids). Stage-wise fields
//! (`fts_results` etc.) are yomu-observability-only.

use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

const LOG_DIR_NAME: &str = "yomu";
const LOG_FILE_NAME: &str = "query_log.jsonl";

/// Resolves the log file path. Reads `$XDG_DATA_HOME` first, falling back
/// to `$HOME/.local/share` per the XDG Base Directory Specification.
///
/// Parameters are taken via injection so unit tests can drive resolution
/// without touching process environment.
pub fn resolve_log_path(xdg_data_home: Option<&str>, home: &str) -> PathBuf {
    let base = match xdg_data_home {
        Some(s) if !s.is_empty() => PathBuf::from(s),
        _ => PathBuf::from(home).join(".local").join("share"),
    };
    base.join(LOG_DIR_NAME).join(LOG_FILE_NAME)
}

/// Convenience wrapper that reads the live process environment.
pub fn resolve_log_path_from_env() -> Option<PathBuf> {
    let home = env::var("HOME").ok()?;
    let xdg = env::var("XDG_DATA_HOME").ok();
    Some(resolve_log_path(xdg.as_deref(), &home))
}

/// Append-only writer over a `Write` sink. Production callers wrap a
/// `std::fs::File`; tests inject `Vec<u8>` to assert serialized content.
pub struct QueryLogWriter<W: Write> {
    writer: W,
}

impl<W: Write> QueryLogWriter<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    /// Serializes `record` as a single JSONL line. Newlines within string
    /// fields are escaped as `\n` by `serde_json`, preserving the 1
    /// record = 1 line invariant.
    pub fn write_record(&mut self, record: &QueryLogRecord) -> io::Result<()> {
        serde_json::to_writer(&mut self.writer, record).map_err(io::Error::other)?;
        self.writer.write_all(b"\n")
    }
}

/// Opens the log file at `path` in append-only mode, creating the parent
/// directory if missing. Errors propagate so callers can degrade to
/// `tracing::warn` per FR-012 without failing the search.
pub fn open_append_writer(path: &Path) -> io::Result<QueryLogWriter<File>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let file = OpenOptions::new().append(true).create(true).open(path)?;
    Ok(QueryLogWriter::new(file))
}

/// Per-stage hit shape, repeated across `fts_results`, `vec_results`,
/// `rrf_results`, `reranked_results`. Keeping a single shape lets consumers
/// write one parser for all stages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StageHit {
    pub chunk_id: i64,
    pub score: f32,
    pub source: String,
}

/// One JSONL record per `Yomu::search` invocation. Field order is stable
/// and field types are pinned for forward-compat with amici (see
/// `approaches.md` mapping table).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct QueryLogRecord {
    pub timestamp: String,
    pub yomu_version: String,
    pub original_query: String,
    pub fts_results: Vec<StageHit>,
    pub vec_results: Vec<StageHit>,
    pub rrf_results: Vec<StageHit>,
    pub reranked_results: Vec<StageHit>,
    pub final_context_ids: Vec<i64>,
    pub latency_ms: u64,
}

#[cfg(test)]
mod tests;
