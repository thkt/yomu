//! Recall / cap-fit measurement for `brief` output against a ground-truth corpus.
//!
//! Pure (no I/O). Mirrors the degraded-on-vacuous contract of [`crate::verify`]
//! (`verify.rs` `compute_rates`): when a denominator is structurally zero the
//! rate is vacuous (1.0) and `degraded` is set so a gate fails loud instead of
//! passing on a meaningless metric.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::query::extract_keywords;

pub mod corpus;

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

fn count_u32(n: usize) -> u32 {
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

/// Deterministic identifier-distance bucket for a recall query (Phase 1, #250).
///
/// Splits queries by whether the task text literally names its seed, so a
/// per-class breakdown can show where embedding earns its keep over a literal
/// FTS5 / grep match. Derived from the corpus alone (no hand annotation), so the
/// bucket stays reproducible: same corpus → same class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum QueryClass {
    /// The task shares an identifier token with the seed paths' stems or dirs, so
    /// a literal keyword match can reach the seed without semantic inference.
    IdentifierNear,
    /// The task shares no identifier token with the seed paths; reaching the seed
    /// needs a semantic jump (the class where embedding should beat grep).
    SemanticFar,
}

impl QueryClass {
    /// Stable label for plain-text rows.
    pub fn label(self) -> &'static str {
        match self {
            QueryClass::IdentifierNear => "identifier-near",
            QueryClass::SemanticFar => "semantic-far",
        }
    }
}

/// 0-indexed rank of the first `targets` member within the top-`k` of `ranked`,
/// or `None` when no target appears in the top-`k`. Hit@k is `.is_some()`; the
/// rank itself exposes how deep in the candidate list the seed sat.
pub fn hit_rank(ranked: &[String], targets: &HashSet<String>, k: usize) -> Option<usize> {
    ranked.iter().take(k).position(|p| targets.contains(p))
}

/// Fraction of `targets` present in the top-`k` of `ranked` (Recall@k).
///
/// Empty `targets` is a vacuous denominator and reports 1.0, mirroring
/// [`measure`]'s degraded-on-vacuous contract.
pub fn recall_at_k(ranked: &[String], targets: &HashSet<String>, k: usize) -> f64 {
    if targets.is_empty() {
        return 1.0;
    }
    let top: HashSet<&str> = ranked.iter().take(k).map(String::as_str).collect();
    let hit = targets.iter().filter(|t| top.contains(t.as_str())).count();
    f64::from(count_u32(hit)) / f64::from(count_u32(targets.len()))
}

/// Path components that carry no identifying signal and must not by themselves
/// pull a query into [`QueryClass::IdentifierNear`].
const GENERIC_DIRS: &[&str] = &["src", "lib", "bin", "tests", "crates"];

/// Terms a seed path contributes to query classification: each component except
/// a generic dir, extension stripped, then tokenized and stemmed by
/// [`extract_keywords`] (camel / snake / kebab split, lowercased, stopword- and
/// short-token-filtered, gerund/plural-stemmed) so the seed side matches the
/// task side under the same normalization (e.g. `artifacts` -> `artifact`).
fn seed_terms(path: &str) -> Vec<String> {
    path.split('/')
        .filter(|c| !GENERIC_DIRS.contains(c))
        .map(|c| c.rsplit_once('.').map_or(c, |(stem, _)| stem))
        .flat_map(extract_keywords)
        .collect()
}

/// Classifies `task` by identifier-distance to `seeds` (deterministic, #250).
///
/// Returns [`QueryClass::IdentifierNear`] when the task and a seed path share a
/// term after both sides pass through [`extract_keywords`] (same split + stem +
/// stopword filter), else [`QueryClass::SemanticFar`]. Normalizing both sides
/// fixes single/plural and camel-split mismatches (`artifacts` seed vs
/// `artifact` task). With no seeds (nothing to match) the query is `SemanticFar`.
pub fn classify_query(task: &str, seeds: &[String]) -> QueryClass {
    let task_terms: HashSet<String> = extract_keywords(task).into_iter().collect();
    if seeds
        .iter()
        .flat_map(|path| seed_terms(path))
        .any(|term| task_terms.contains(&term))
    {
        QueryClass::IdentifierNear
    } else {
        QueryClass::SemanticFar
    }
}

/// Seed-acquisition arm under comparison (#250 Phase 1).
///
/// The closure stage is shared across arms; only how seed candidates are
/// retrieved differs, isolating the embedding layer's contribution over a
/// literal FTS5 baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Arm {
    /// Embedding vector search — the production seed-inference path.
    Embedding,
    /// FTS5 keyword search — the deletion-test baseline (no embedding).
    NoEmbed,
}

impl Arm {
    /// Stable lowercase label for plain-text rows and `--json`.
    pub fn label(self) -> &'static str {
        match self {
            Arm::Embedding => "embedding",
            Arm::NoEmbed => "no-embed",
        }
    }
}

/// Seed-stage retrieval result for one GT entry under one arm (#250 Phase 1).
///
/// `seed_rank` is the 0-indexed position of the GT seed in the arm's candidate
/// list (`None` = not retrieved within the measured depth). `must_recall` is
/// Recall@depth of the must_include set at the seed stage, before closure — the
/// discriminating read, since a shared closure would otherwise fill must_include
/// from any decent seed and compress the arm difference. Hit@k is derived from
/// `seed_rank` at summary time, so the per-entry record stays minimal.
#[derive(Debug, Clone, Serialize)]
pub struct ArmEntryReport {
    pub id: String,
    pub arm: Arm,
    pub class: QueryClass,
    pub seed_rank: Option<usize>,
    pub must_recall: f64,
}

/// Per `(arm, class)` aggregate over [`ArmEntryReport`]s (#250 Phase 1).
#[derive(Debug, Clone, Serialize)]
pub struct ArmClassSummary {
    pub arm: Arm,
    pub class: QueryClass,
    pub n: usize,
    /// Hit@k for each k in the report's k-values, in the same order.
    pub hit_at_k: Vec<f64>,
    pub mean_must_recall: f64,
}

/// k values at which Hit@k is reported (#250 Phase 1). Small because the seed
/// count cap is 5 and the GT candidate lists are short.
pub const ARM_K_VALUES: &[usize] = &[1, 3, 5];

/// Aggregates per-entry arm results into `(arm, class)` summaries (#250 Phase 1).
///
/// Groups in a fixed arm-major, class order so the output is deterministic, and
/// skips a group with no entries (e.g. a corpus with no semantic-far queries) so
/// an absent bucket never shows as a misleading zero row. Hit@k for each `k` is
/// the fraction of the group whose seed rank is below `k`.
pub fn summarize_arms(entries: &[ArmEntryReport], k_values: &[usize]) -> Vec<ArmClassSummary> {
    let mut out = Vec::new();
    for arm in [Arm::Embedding, Arm::NoEmbed] {
        for class in [QueryClass::IdentifierNear, QueryClass::SemanticFar] {
            let group: Vec<&ArmEntryReport> = entries
                .iter()
                .filter(|e| e.arm == arm && e.class == class)
                .collect();
            if group.is_empty() {
                continue;
            }
            let denom = f64::from(count_u32(group.len()));
            let hit_at_k = k_values
                .iter()
                .map(|&k| {
                    let hits = group
                        .iter()
                        .filter(|e| e.seed_rank.is_some_and(|r| r < k))
                        .count();
                    f64::from(count_u32(hits)) / denom
                })
                .collect();
            let mean_must_recall = group.iter().map(|e| e.must_recall).sum::<f64>() / denom;
            out.push(ArmClassSummary {
                arm,
                class,
                n: group.len(),
                hit_at_k,
                mean_must_recall,
            });
        }
    }
    out
}

/// Full arm-comparison report for one repo (#250 Phase 1): the per-`(arm,class)`
/// summaries plus the per-entry rows (kept because N is small enough to read
/// directly), and a `degraded` flag set when an arm could not run (e.g. the
/// embedding model was unavailable).
#[derive(Debug, Clone, Serialize)]
pub struct ArmComparisonReport {
    pub repo: String,
    pub k_values: Vec<usize>,
    pub summaries: Vec<ArmClassSummary>,
    pub entries: Vec<ArmEntryReport>,
    pub degraded: bool,
}

impl ArmComparisonReport {
    /// Builds a report from per-entry rows, computing the `(arm, class)`
    /// summaries via [`summarize_arms`].
    pub fn new(
        repo: String,
        entries: Vec<ArmEntryReport>,
        k_values: &[usize],
        degraded: bool,
    ) -> Self {
        let summaries = summarize_arms(&entries, k_values);
        Self {
            repo,
            k_values: k_values.to_vec(),
            summaries,
            entries,
            degraded,
        }
    }
}

/// Renders an [`ArmComparisonReport`] as single-line JSON (#250 Phase 1).
pub fn render_arm_json(report: &ArmComparisonReport) -> String {
    serde_json::to_string(report).unwrap_or_else(|_| "{}".to_owned())
}

/// Renders an [`ArmComparisonReport`] as a human-readable table (#250 Phase 1).
///
/// States the N up front so the per-class numbers read as directional, not as a
/// statistically powered comparison (the GT corpus has ~2-4 entries per bucket).
pub fn render_arm_plain(report: &ArmComparisonReport) -> String {
    use std::fmt::Write as _;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "arm comparison: {} (degraded: {})",
        report.repo, report.degraded
    );
    let _ = writeln!(
        out,
        "  N={} per-entry rows; per-class read is directional, not statistically powered",
        report.entries.len()
    );
    let ks = report
        .k_values
        .iter()
        .map(|k| format!("Hit@{k}"))
        .collect::<Vec<_>>()
        .join("/");
    for s in &report.summaries {
        let hits = s
            .hit_at_k
            .iter()
            .map(|h| format!("{h:.2}"))
            .collect::<Vec<_>>()
            .join("/");
        let _ = writeln!(
            out,
            "  {:<10} {:<16} n={} {ks}={hits} must_recall@depth={:.3}",
            s.arm.label(),
            s.class.label(),
            s.n,
            s.mean_must_recall
        );
    }
    let _ = writeln!(out, "  --- per entry ---");
    for e in &report.entries {
        let rank = e
            .seed_rank
            .map_or_else(|| "miss".to_owned(), |r| r.to_string());
        let _ = writeln!(
            out,
            "  {:<10} {:<32} {:<16} seed_rank={rank} must_recall={:.3}",
            e.arm.label(),
            e.id,
            e.class.label(),
            e.must_recall
        );
    }
    out
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
}

impl CorpusReport {
    /// Builds a report from per-entry measurements, with the aggregate as the
    /// unweighted mean. An empty entry set (no GT entry matched `repo`) is
    /// degraded with a vacuous 1.0 mean, mirroring [`measure`]'s degraded-on-
    /// vacuous contract so a `--repo` typo never reads as a silent pass.
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
