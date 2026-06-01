use std::collections::{HashMap, HashSet};
use std::time::Instant;

use crate::{query, storage};

use super::embedder::degraded_reason_user_note;
use super::format::{
    EnrichmentContext, format_no_results_message, format_results_grouped, format_results_json,
};
use super::{
    InvalidInputKind, MAX_QUERY_LENGTH, MAX_SEARCH_LIMIT, MAX_SEARCH_OFFSET, Yomu, YomuError,
    index_hint, parse_impact_target, validate_path,
};

impl Yomu {
    pub fn search(
        &self,
        query: Option<&str>,
        limit: u32,
        offset: u32,
        paths: &[String],
        json: bool,
        from_target: Option<&str>,
    ) -> Result<String, YomuError> {
        validate_search_query(query)?;
        for path in paths {
            validate_path(path)?;
        }
        let limit = limit.min(MAX_SEARCH_LIMIT);

        if let Some(from) = from_target {
            // FR-006: --offset is intentionally ignored in from-file mode.
            return self.search_from(from, query, limit, paths, json);
        }

        let query =
            query.ok_or_else(|| YomuError::InvalidInput(InvalidInputKind::QueryOrFromRequired))?;
        let offset = offset.min(MAX_SEARCH_OFFSET);

        let stats = self.with_db(storage::get_stats)?;
        let outcome = self.run_search_query(query, limit, offset, paths)?;
        let notes = self.build_search_notes(&stats, outcome.degraded);

        self.format_search_results(&outcome.results, &stats, notes, json, outcome.degraded)
    }

    /// Runs the embedding-backed query and emits the query log when enabled,
    /// isolating the DB + timing path from `search`'s validation and note
    /// assembly so each part is testable on its own (RC-009 test seam).
    fn run_search_query(
        &self,
        query: &str,
        limit: u32,
        offset: u32,
        paths: &[String],
    ) -> Result<query::SearchOutcome, YomuError> {
        let embedder = self.get_embedder();
        tracing::debug!(query, limit, offset, ?paths, "search request");

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
        Ok(outcome)
    }

    /// Assembles the user-facing advisory notes (index-coverage hint, reranker
    /// status, degraded-embedding warning) appended to search output.
    fn build_search_notes(&self, stats: &storage::IndexStatus, degraded: bool) -> Vec<String> {
        let mut notes: Vec<String> = Vec::new();
        if let Some(msg) = index_hint(stats) {
            notes.push(msg);
        }
        if let Some(note) = self.reranker_note() {
            notes.push(note);
        }
        if let Some(reason) = self.degraded_reason() {
            if let Some(note) = degraded_reason_user_note(*reason, "yomu model download") {
                notes.push(note);
            }
        } else if degraded {
            notes.push("embedding model not loaded; results from text search only".into());
        }
        notes
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
}

/// Rejects an empty or over-length query before any DB work. Returns `Ok(())`
/// when no query is supplied; the `--from` path validates its target separately.
pub(super) fn validate_search_query(query: Option<&str>) -> Result<(), YomuError> {
    let Some(q) = query else {
        return Ok(());
    };
    if q.is_empty() {
        return Err(YomuError::InvalidInput(InvalidInputKind::EmptyQuery));
    }
    if q.len() > MAX_QUERY_LENGTH {
        return Err(YomuError::InvalidInput(InvalidInputKind::QueryTooLong {
            max: MAX_QUERY_LENGTH,
            actual: q.len(),
        }));
    }
    Ok(())
}
