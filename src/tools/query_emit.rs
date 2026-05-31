use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::query;
use crate::query_log::{self, QueryLogRecord};

use super::Yomu;

impl Yomu {
    /// Appends one JSONL record for a search invocation when `--log-query` is on.
    ///
    /// This is real-I/O glue over the separately tested `query_log` module:
    /// resolve the path from the environment, serialize, append, and degrade to
    /// `tracing::warn` on any failure (FR-012) so logging never fails a search.
    ///
    /// Excluded from coverage like `model_download`: the logic it delegates to
    /// (path resolution, serialization, append) is unit-tested in `query_log` via
    /// injection seams, and what remains here is env-coupled glue that yomu's
    /// `unsafe_code = "forbid"` makes un-unit-testable (no `set_var` to redirect
    /// `$HOME`/`$XDG_DATA_HOME`). Kept in its own file so the path-exclusion in
    /// CI does not also blind the rest of the search routes.
    pub(super) fn emit_query_log(
        &self,
        query: &str,
        outcome: &query::SearchOutcome,
        latency_ms: u64,
    ) {
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
