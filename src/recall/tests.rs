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

// T-020: corpus_report_aggregates_mean_and_renders_keys_per_entry_and_aggregate
#[test]
fn corpus_report_aggregates_mean_and_renders_keys() {
    let entries = vec![
        EntryReport {
            id: "e1".to_owned(),
            report: RecallReport {
                recall: 1.0,
                cap_fit: 1.0,
                degraded: false,
            },
        },
        EntryReport {
            id: "e2".to_owned(),
            report: RecallReport {
                recall: 0.5,
                cap_fit: 0.5,
                degraded: false,
            },
        },
    ];
    let report = CorpusReport::new("rurico".to_owned(), entries);
    assert!(
        (report.aggregate.recall - 0.75).abs() < f64::EPSILON,
        "aggregate recall is the mean (0.75), got {}",
        report.aggregate.recall
    );
    assert!(
        (report.aggregate.cap_fit - 0.75).abs() < f64::EPSILON,
        "aggregate cap_fit is the mean (0.75), got {}",
        report.aggregate.cap_fit
    );
    assert!(
        !report.aggregate.degraded,
        "no entry degraded → aggregate not degraded"
    );

    let json = render_recall_json(&report);
    assert!(
        json.contains("\"repo\":\"rurico\""),
        "repo present, got: {json}"
    );
    assert!(
        json.contains("\"id\":\"e1\""),
        "entry id present, got: {json}"
    );
    // recall/cap_fit keys appear per-entry (2) plus once in the aggregate (3).
    assert_eq!(
        json.matches("\"recall\":").count(),
        3,
        "recall key present per-entry and in aggregate, got: {json}"
    );
    assert_eq!(
        json.matches("\"cap_fit\":").count(),
        3,
        "cap_fit key present per-entry and in aggregate, got: {json}"
    );
}

// T-022: corpus_report_with_no_entries_is_degraded
#[test]
fn corpus_report_with_no_entries_is_degraded() {
    let report = CorpusReport::new("ghost".to_owned(), Vec::new());
    assert!(
        report.aggregate.degraded,
        "no entries (e.g. --repo mismatch) → degraded, never a silent pass"
    );
}
