#[cfg(not(feature = "test-support"))]
use std::collections::HashMap;
use std::collections::HashSet;

use amici::model::{degrade_with_warn, record_degraded};
use rurico::reranker::Rerank;

use crate::{brief, query, storage};

use super::embedder::DegradedReason;
use super::{BRIEF_MAX_INFERRED_SEEDS, InvalidInputKind, Yomu, YomuError};

// `recall` is a maintainer diagnostic used only by the `recall-bench` crate
// (ADR-0005). It is cfg'd out of the `test-support` (coverage) build, so its
// exclusive imports follow it out to avoid unused-import warnings there.
#[cfg(not(feature = "test-support"))]
use super::{ARM_CANDIDATE_DEPTH, RECALL_DEPTH, RECALL_MAX_BYTES, RECALL_MAX_CHUNKS};
#[cfg(not(feature = "test-support"))]
use crate::recall::{self, corpus};

impl Yomu {
    fn infer_seed_paths(&self, task: &str, max_seeds: u32) -> (Vec<String>, bool) {
        match self.embedder_seed_paths(task, max_seeds, self.get_reranker()) {
            Ok(paths) => (paths, false),
            Err(reason) => {
                record_degraded(reason, "brief: seed inference");
                (self.fts_fallback_seed_paths(task, max_seeds), true)
            }
        }
    }

    pub(super) fn embedder_seed_paths(
        &self,
        task: &str,
        max_seeds: u32,
        reranker: Option<&dyn Rerank>,
    ) -> Result<Vec<String>, DegradedReason> {
        let embedder = self.try_embedder_arc()?;
        let task_emb = embedder.embed_query(task).map_err(degrade_with_warn(
            "brief seed inference: embed_query",
            DegradedReason::ProbeFailed,
        ))?;
        // Oversample when a cross-encoder is available so it can promote
        // candidates the vec top-N would cut off; the RRF blend keeps the vec
        // rank as a prior and rerank failure keeps the vec order outright
        // (warn, not degraded) (#290).
        let fetch = if reranker.is_some() {
            max_seeds.saturating_mul(3)
        } else {
            max_seeds
        };
        let conn = self
            .conn
            .lock()
            .expect("DB lock poisoned (embedder_seed_paths)");
        let mut results =
            storage::vec_search(&conn, &task_emb, fetch, None, &[]).map_err(degrade_with_warn(
                "brief seed inference: vec_search",
                DegradedReason::ProbeFailed,
            ))?;
        drop(conn);
        if let Some(ranker) = reranker {
            query::cross_encoder_rrf_rerank(&mut results, task, ranker);
        }

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
    /// are given (seed-less), and returns the closure output with a degradation
    /// cause recorded when seed inference fell back or the closure was empty.
    /// Shared by `brief` (renders) and `recall` (measures); callers validate
    /// `task` first.
    fn brief_output(&self, task: &brief::TaskBrief) -> Result<brief::BriefOutput, YomuError> {
        let mut effective = task.clone();
        let mut seed_fallback = false;
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
            seed_fallback |= seed_degraded;
        }

        let mut output = self.with_db(|conn| brief::expand_plan(conn, &effective))?;
        if seed_fallback {
            // Front insert, not push: expand_plan may already have recorded
            // EmptySeeds, but seed inference is the earlier stage, so its
            // cause must lead the note (DegradedCause stage-order contract).
            output
                .degraded_causes
                .insert(0, brief::DegradedCause::SeedFtsFallback);
        }

        if output.chunks.is_empty() {
            tracing::warn!(
                seeds = effective.seeds.len(),
                degraded = output.degraded(),
                "brief produced zero chunks"
            );
            // EmptySeeds already explains a zero-chunk closure; recording
            // EmptyClosure on top would just restate it.
            if !output
                .degraded_causes
                .contains(&brief::DegradedCause::EmptySeeds)
            {
                output
                    .degraded_causes
                    .push(brief::DegradedCause::EmptyClosure);
            }
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

    /// Unembedded embeddable-chunk count for the embed-completeness check
    /// (#288). A query failure propagates instead of folding to 0: a gap that
    /// cannot be measured must not read as "no gap" (audit RC-1 — the gap
    /// query touches `embedded_chunk_ids`, which the FTS/vec measurement
    /// queries do not, so they can succeed while this fails).
    #[cfg(not(feature = "test-support"))]
    fn embed_gap(&self) -> Result<u32, YomuError> {
        let conn = self.conn.lock().expect("DB lock poisoned (embed_gap)");
        Ok(storage::embed_gap_count(&conn)?)
    }

    /// Measures seed-less recall and weighted cap-fit for every bundled GT entry
    /// whose repo matches `repo`, against the current index, and renders a
    /// per-entry plus aggregate report (FR-011). Returns the rendered text and the
    /// aggregate degraded flag. The caller exits non-zero when degraded (FR-012):
    /// an unavailable embedding model makes seed inference fall back and flag
    /// degraded, so a model-less run never reports a silent pass. An incomplete
    /// embed (index died mid-embed, #288) likewise degrades, with the gap count
    /// carried in the report.
    ///
    /// Not concurrent-safe: the gap is sampled once before the measurement
    /// loop, so running `yomu index` against the same DB during a bench run
    /// can desynchronize the two (audit RC-4). recall-bench is a one-shot
    /// maintainer diagnostic; do not run it while indexing.
    #[cfg(not(feature = "test-support"))]
    pub fn recall(&self, repo: &str, json: bool) -> Result<(String, bool), YomuError> {
        let gt = corpus::load_bundled()
            .map_err(|e| YomuError::Internal(format!("bundled GT corpus: {e}")))?;
        let embed_gap = self.embed_gap()?;
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
            report.degraded |= output.degraded();
            entries.push(recall::EntryReport {
                id: entry.id.clone(),
                report,
            });
        }
        let report = recall::CorpusReport::new(repo.to_owned(), entries, embed_gap);
        let text = if json {
            recall::render_recall_json(&report)
        } else {
            recall::render_recall_plain(&report)
        };
        Ok((text, report.aggregate.degraded))
    }

    /// No-embed arm seed candidates (#250): a competent keyword search, not the
    /// production AND-fallback ([`Self::fts_fallback_seed_paths`], which only
    /// fires when the embedder is down and ANDs every task term). Each extracted
    /// keyword runs as its own single-term `search_by_fts` (AND-of-one, leaving
    /// the shared query builder untouched); results merge by best score so a file
    /// matching any strong term surfaces. Phase-1 fair FTS baseline, not the
    /// deferred Phase-2 ripgrep design.
    #[cfg(not(feature = "test-support"))]
    fn arm_fts_seed_paths(&self, task: &str, depth: u32) -> Vec<String> {
        let keywords = query::extract_keywords(task);
        if keywords.is_empty() {
            return Vec::new();
        }
        let conn = self
            .conn
            .lock()
            .expect("DB lock poisoned (arm_fts_seed_paths)");
        let mut best: HashMap<String, f32> = HashMap::new();
        for kw in &keywords {
            let results = storage::search_by_fts(
                &conn,
                &[kw.as_str()],
                None,
                &HashSet::new(),
                None,
                depth,
                &[],
            )
            .unwrap_or_default();
            for r in results {
                best.entry(r.chunk.file_path)
                    .and_modify(|s| {
                        if r.score > *s {
                            *s = r.score;
                        }
                    })
                    .or_insert(r.score);
            }
        }
        drop(conn);
        let mut ranked: Vec<(String, f32)> = best.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked
            .into_iter()
            .take(depth as usize)
            .map(|(p, _)| p)
            .collect()
    }

    /// Measures seed-stage retrieval for both arms (embedding vs FTS5-only)
    /// against every bundled GT entry whose repo matches `repo`, and renders the
    /// arm-comparison report (#250 Phase 1). Returns the rendered text and a
    /// degraded flag set when the embedding arm could not run (model
    /// unavailable); the caller exits non-zero when degraded, mirroring
    /// [`Yomu::recall`].
    ///
    /// The no-embed arm is the deletion test: the FTS5 keyword path the
    /// production seed inference uses only as a fallback, run here as a
    /// standalone arm. The embedding-skip stays inside this maintainer
    /// diagnostic (ADR-0005 / OUTCOME.md:39), never a product CLI flag. A
    /// degraded embedding arm records an empty candidate list rather than
    /// falling back to FTS, so the two arms stay isolated.
    ///
    /// Not concurrent-safe, same contract as [`Yomu::recall`] (audit RC-4):
    /// do not run while `yomu index` writes the same DB.
    #[cfg(not(feature = "test-support"))]
    pub fn recall_arms(&self, repo: &str, json: bool) -> Result<(String, bool), YomuError> {
        let gt = corpus::load_bundled()
            .map_err(|e| YomuError::Internal(format!("bundled GT corpus: {e}")))?;
        let depth = ARM_CANDIDATE_DEPTH;
        let k = depth as usize;
        // An incomplete embed (#288) invalidates the embedding arm: FTS rows
        // are complete, so only one arm degrades — exactly the asymmetry the
        // arm comparison must not silently absorb. The constructor folds the
        // gap into `degraded`.
        let embed_gap = self.embed_gap()?;
        let mut entries = Vec::new();
        let mut degraded = false;
        for entry in gt.entries.iter().filter(|e| e.repo == repo) {
            let class = recall::classify_query(&entry.task, &entry.seed);
            let seed_set: HashSet<String> = entry.seed.iter().cloned().collect();
            let must_set: HashSet<String> =
                entry.must_include.iter().map(|f| f.path.clone()).collect();

            // Direct call (not infer_seed_paths) because this arm controls
            // `depth` and isolates the embedding stage; it still passes the
            // reranker so the bench measures the production inference path (#290).
            let emb_ranked = match self.embedder_seed_paths(&entry.task, depth, self.get_reranker())
            {
                Ok(paths) => paths,
                Err(reason) => {
                    record_degraded(reason, "recall arms: embedding");
                    degraded = true;
                    Vec::new()
                }
            };
            let fts_ranked = self.arm_fts_seed_paths(&entry.task, depth);

            entries.push(recall::ArmEntryReport {
                id: entry.id.clone(),
                arm: recall::Arm::Embedding,
                class,
                seed_rank: recall::hit_rank(&emb_ranked, &seed_set, k),
                must_recall: recall::recall_at_k(&emb_ranked, &must_set, k),
            });
            entries.push(recall::ArmEntryReport {
                id: entry.id.clone(),
                arm: recall::Arm::NoEmbed,
                class,
                seed_rank: recall::hit_rank(&fts_ranked, &seed_set, k),
                must_recall: recall::recall_at_k(&fts_ranked, &must_set, k),
            });
        }
        let report = recall::ArmComparisonReport::new(
            repo.to_owned(),
            entries,
            recall::ARM_K_VALUES,
            degraded,
            embed_gap,
        );
        let text = if json {
            recall::render_arm_json(&report)
        } else {
            recall::render_arm_plain(&report)
        };
        Ok((text, report.degraded))
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
