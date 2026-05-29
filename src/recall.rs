//! Recall / cap-fit measurement for `brief` output against a ground-truth corpus.
//!
//! Pure (no I/O). Mirrors the degraded-on-vacuous contract of [`crate::verify`]
//! (`verify.rs` `compute_rates`): when a denominator is structurally zero the
//! rate is vacuous (1.0) and `degraded` is set so a gate fails loud instead of
//! passing on a meaningless metric.

use std::collections::HashSet;

use serde::Deserialize;

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
#[derive(Debug, Clone, PartialEq)]
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

#[cfg(test)]
mod tests {
    use super::*;

    fn wf(path: &str, weight: u32) -> WeightedFile {
        WeightedFile {
            path: path.to_owned(),
            weight,
        }
    }

    fn set(paths: &[&str]) -> HashSet<String> {
        paths.iter().map(|s| (*s).to_owned()).collect()
    }

    // T-001: measure_recall_is_hit_over_total
    #[test]
    fn measure_recall_is_hit_over_total() {
        let must = [wf("a", 1), wf("b", 1), wf("c", 1), wf("d", 1)];
        let output = set(&["a", "b", "c"]);
        let reachable = set(&["a", "b", "c", "d"]);
        let r = measure(&must, &output, &reachable);
        assert!(
            (r.recall - 0.75).abs() < f64::EPSILON,
            "3 of 4 must-include present → recall 0.75, got {}",
            r.recall
        );
        assert!(!r.degraded, "non-zero denominators must not degrade");
    }

    // T-003: measure_cap_fit_is_weighted_survival_over_reachable
    #[test]
    fn measure_cap_fit_is_weighted_survival_over_reachable() {
        // Pre-cap reachable must-include weight = 4+3+2+1 = 10; cap keeps a+b = 7.
        let must = [wf("a", 4), wf("b", 3), wf("c", 2), wf("d", 1)];
        let output = set(&["a", "b"]);
        let reachable = set(&["a", "b", "c", "d"]);
        let r = measure(&must, &output, &reachable);
        assert!(
            (r.cap_fit - 0.7).abs() < f64::EPSILON,
            "surviving weight 7 / reachable weight 10 → cap_fit 0.7, got {}",
            r.cap_fit
        );
        // recall is unweighted and independent: 2 of 4 present = 0.5.
        assert!(
            (r.recall - 0.5).abs() < f64::EPSILON,
            "recall must stay unweighted (0.5), independent of cap_fit, got {}",
            r.recall
        );
    }

    // T-004: measure_zero_reachable_weight_degrades
    #[test]
    fn measure_zero_reachable_weight_degrades() {
        // No must-include file is reachable pre-cap → cap-fit denominator 0.
        let must = [wf("a", 1)];
        let output = set(&[]);
        let reachable = set(&[]);
        let r = measure(&must, &output, &reachable);
        assert!(
            r.degraded,
            "zero reachable must-include weight must set degraded (vacuous cap-fit)"
        );
        assert!(
            (r.cap_fit - 1.0).abs() < f64::EPSILON,
            "vacuous cap-fit reports 1.0 (verify.rs parity), got {}",
            r.cap_fit
        );
    }

    // T-009: gate_passes_is_false_below_floor_true_at_or_above
    #[test]
    fn gate_passes_is_false_below_floor_true_at_or_above() {
        // FR-009: the seeded gate fails when mean recall is below the committed floor.
        assert!(
            !gate_passes(0.74, 0.80),
            "mean below floor must fail the gate"
        );
        // The floor is an inclusive minimum: equal or above passes.
        assert!(gate_passes(0.80, 0.80), "mean equal to floor must pass");
        assert!(gate_passes(0.95, 0.80), "mean above floor must pass");
    }

    // cap-fit invariant: a must-include file present in `output` but absent from
    // `reachable` must not inflate cap-fit past 1.0 (guards FR-003 against a
    // caller whose output is not a subset of reachable).
    #[test]
    fn measure_cap_fit_stays_within_one_when_output_escapes_reachable() {
        let must = [wf("a", 1), wf("b", 1)];
        // `b` is in output but was never reached pre-cap: it must not be credited.
        let output = set(&["a", "b"]);
        let reachable = set(&["a"]);
        let r = measure(&must, &output, &reachable);
        assert!(
            (r.cap_fit - 1.0).abs() < f64::EPSILON,
            "numerator constrained to reachable → cap_fit 1.0 (not 2.0), got {}",
            r.cap_fit
        );
    }
}
