use std::collections::HashSet;

use amici::model::{degrade_with_warn, record_degraded};

use crate::{brief, query, storage};

use super::embedder::DegradedReason;
use super::{BRIEF_MAX_INFERRED_SEEDS, InvalidInputKind, Yomu, YomuError};

// `recall` is a maintainer diagnostic used only by the `recall-bench` crate
// (ADR-0005). It is cfg'd out of the `test-support` (coverage) build, so its
// exclusive imports follow it out to avoid unused-import warnings there.
#[cfg(not(feature = "test-support"))]
use super::{RECALL_DEPTH, RECALL_MAX_BYTES, RECALL_MAX_CHUNKS};
#[cfg(not(feature = "test-support"))]
use crate::recall::{self, corpus};

impl Yomu {
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
    #[cfg(not(feature = "test-support"))]
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
