//! Recall / cap-fit measurement for `brief` output against a ground-truth corpus.
//!
//! Pure (no I/O). Mirrors the degraded-on-vacuous contract of [`crate::verify`]
//! (`verify.rs` `compute_rates`): when a denominator is structurally zero the
//! rate is vacuous (1.0) and `degraded` is set so a gate fails loud instead of
//! passing on a meaningless metric.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

pub mod corpus;

mod arms;
pub use arms::{
    ARM_K_VALUES, Arm, ArmClassSummary, ArmComparisonReport, ArmEntryReport, QueryClass,
    classify_query, hit_rank, recall_at_k, render_arm_json, render_arm_plain, summarize_arms,
};

/// A must-include file paired with its domain-assigned importance weight.
///
/// Weight ranks how much dropping the file would hurt the brief's completeness
/// (1 = supporting, higher = central). Used only by the weighted cap-fit metric;
/// plain recall ignores it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct WeightedFile {
    pub path: String,
    pub weight: u32,
}

/// Recall and cap-fit for one ground-truth entry.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RecallReport {
    /// Unweighted must-include hit rate: present / total.
    pub recall: f64,
    /// Weighted ratio of must-include kept by the cap, among those the closure
    /// reached pre-cap. Isolates the cap's effect from the closure's coverage.
    pub cap_fit: f64,
    /// True when a denominator was zero (vacuous metric, see module docs).
    pub degraded: bool,
}

pub(super) fn count_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Computes [`RecallReport`] for one GT entry.
///
/// - `recall` = `|must_include ∩ output| / |must_include|`.
/// - `cap_fit` = `Σ weight(must_include ∩ output ∩ reachable) / Σ weight(must_include ∩ reachable)`,
///   where `reachable` is the pre-cap closure set. Both sides require membership
///   in `reachable`, so cap-fit stays in `[0, 1]` even if a caller passes an
///   `output` not contained in `reachable`; it measures the cap alone, not the
///   closure's recall.
/// - `degraded` = true when either denominator is zero.
pub fn measure(
    must_include: &[WeightedFile],
    output: &HashSet<String>,
    reachable: &HashSet<String>,
) -> RecallReport {
    let total = count_u32(must_include.len());
    let hit = count_u32(
        must_include
            .iter()
            .filter(|f| output.contains(&f.path))
            .count(),
    );
    let reachable_weight: u32 = must_include
        .iter()
        .filter(|f| reachable.contains(&f.path))
        .map(|f| f.weight)
        .sum();
    // Numerator is constrained to `reachable` too, so a file in `output` but not
    // in `reachable` cannot inflate cap-fit past 1.0. In practice `output` is a
    // post-cap subset of `reachable`, but `measure` is pure and must not trust
    // its caller to preserve that.
    let surviving_weight: u32 = must_include
        .iter()
        .filter(|f| output.contains(&f.path) && reachable.contains(&f.path))
        .map(|f| f.weight)
        .sum();

    // Vacuous denominators report 1.0 and flag degraded, so a gate fails loud
    // instead of passing on an empty corpus (verify.rs `compute_rates` parity).
    let recall_degraded = total == 0;
    let cap_fit_degraded = reachable_weight == 0;
    let recall = if recall_degraded {
        1.0
    } else {
        f64::from(hit) / f64::from(total)
    };
    let cap_fit = if cap_fit_degraded {
        1.0
    } else {
        f64::from(surviving_weight) / f64::from(reachable_weight)
    };

    RecallReport {
        recall,
        cap_fit,
        degraded: recall_degraded || cap_fit_degraded,
    }
}

/// Returns `true` when the mean seeded recall meets the committed `floor`.
///
/// FR-009: the seeded gate (Gate1) fails when `mean` is below `floor`. The floor
/// is an inclusive minimum, so `mean == floor` passes. Keeping the comparison in
/// one pure function lets the threshold be unit-tested without a live index, and
/// the integration gate wires a measured mean to the committed constant.
pub fn gate_passes(mean: f64, floor: f64) -> bool {
    mean >= floor
}

/// One ground-truth entry's seed-less recall, tagged with its id, for the
/// `yomu recall` report (FR-011). Flattens [`RecallReport`] so each entry object
/// carries `recall` / `cap_fit` / `degraded` directly.
#[derive(Debug, Clone, Serialize)]
pub struct EntryReport {
    pub id: String,
    #[serde(flatten)]
    pub report: RecallReport,
}

/// Seed-less recall report for one repo's GT entries (FR-011). `aggregate` holds
/// the mean recall / cap-fit and a degraded flag set when any entry degraded or
/// no entry matched. Emitted per-repo; the recall workflow merges repos.
#[derive(Debug, Clone, Serialize)]
pub struct CorpusReport {
    pub repo: String,
    pub aggregate: RecallReport,
    pub entries: Vec<EntryReport>,
    /// Embeddable chunks with no embedding at measurement time (#288). Non-zero
    /// means indexing died mid-embed, so seed inference ran against a partial
    /// vec candidate space and the numbers are invalid until `yomu index` is
    /// re-run. The measuring caller sets this from
    /// `storage::embed_gap_count` and flags `degraded`; the report itself must
    /// carry the reason because recall-bench has no tracing subscriber.
    pub embed_gap: u32,
}

impl CorpusReport {
    /// Builds a report from per-entry measurements, with the aggregate as the
    /// unweighted mean. An empty entry set (no GT entry matched `repo`) is
    /// degraded with a vacuous 1.0 mean, mirroring [`measure`]'s degraded-on-
    /// vacuous contract so a `--repo` typo never reads as a silent pass.
    /// `embed_gap` starts at 0; the measuring caller overwrites it (#288).
    pub fn new(repo: String, entries: Vec<EntryReport>) -> Self {
        let aggregate = if entries.is_empty() {
            RecallReport {
                recall: 1.0,
                cap_fit: 1.0,
                degraded: true,
            }
        } else {
            let n = entries.len() as f64;
            RecallReport {
                recall: entries.iter().map(|e| e.report.recall).sum::<f64>() / n,
                cap_fit: entries.iter().map(|e| e.report.cap_fit).sum::<f64>() / n,
                degraded: entries.iter().any(|e| e.report.degraded),
            }
        };
        Self {
            repo,
            aggregate,
            entries,
            embed_gap: 0,
        }
    }
}

/// Renders a [`CorpusReport`] as a single-line JSON object (FR-011).
pub fn render_recall_json(report: &CorpusReport) -> String {
    serde_json::to_string(report).unwrap_or_else(|_| "{}".to_owned())
}

/// Renders a [`CorpusReport`] as a human-readable plain-text table (FR-011).
pub fn render_recall_plain(report: &CorpusReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "recall report: {} (degraded: {})",
        report.repo, report.aggregate.degraded
    );
    if report.embed_gap > 0 {
        let _ = writeln!(
            out,
            "  warning: {} embeddable chunks lack embeddings (indexing died mid-embed?) — numbers are invalid; re-run `yomu index` (#288)",
            report.embed_gap
        );
    }
    let _ = writeln!(
        out,
        "  aggregate: recall={:.3} cap_fit={:.3} over {} entries",
        report.aggregate.recall,
        report.aggregate.cap_fit,
        report.entries.len()
    );
    for entry in &report.entries {
        let _ = writeln!(
            out,
            "  {:<40} recall={:.3} cap_fit={:.3}{}",
            entry.id,
            entry.report.recall,
            entry.report.cap_fit,
            if entry.report.degraded {
                " (degraded)"
            } else {
                ""
            }
        );
    }
    out
}

#[cfg(test)]
mod tests;
