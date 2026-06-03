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

fn vec_s(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
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

// --- #250 Phase 1: seed-stage Hit@k / Recall@k and query classification ---

// T-023: hit_rank_returns_first_target_position_within_k
#[test]
fn hit_rank_returns_first_target_within_k() {
    let ranked = vec_s(&["a", "b", "c", "d"]);
    assert_eq!(
        hit_rank(&ranked, &set(&["c"]), 5),
        Some(2),
        "c sits at 0-indexed rank 2"
    );
    assert_eq!(
        hit_rank(&ranked, &set(&["c"]), 2),
        None,
        "c beyond top-2 is a miss at k=2"
    );
    assert_eq!(
        hit_rank(&ranked, &set(&["x"]), 5),
        None,
        "absent target is a miss"
    );
    assert_eq!(
        hit_rank(&ranked, &set(&["a"]), 0),
        None,
        "k=0 admits no candidate, so even rank-0 is a miss"
    );
}

// T-024: hit_rank_picks_earliest_when_multiple_targets_present
#[test]
fn hit_rank_picks_earliest_target() {
    let ranked = vec_s(&["a", "b", "c"]);
    assert_eq!(
        hit_rank(&ranked, &set(&["b", "c"]), 3),
        Some(1),
        "earliest of b,c is b at rank 1"
    );
}

// T-025: recall_at_k_is_fraction_of_targets_in_top_k
#[test]
fn recall_at_k_is_fraction_present_in_top_k() {
    let ranked = vec_s(&["a", "b", "c", "d"]);
    // top-3 = a,b,c; of targets a,c,x only a,c are present = 2/3.
    assert!(
        (recall_at_k(&ranked, &set(&["a", "c", "x"]), 3) - 2.0 / 3.0).abs() < f64::EPSILON,
        "2 of 3 targets within top-3"
    );
}

// T-026: recall_at_k_counts_only_within_top_k
#[test]
fn recall_at_k_counts_only_top_k() {
    let ranked = vec_s(&["a", "b", "c", "d"]);
    // d is rank 3, outside top-3, so only a counts.
    assert!(
        (recall_at_k(&ranked, &set(&["a", "d"]), 3) - 0.5).abs() < f64::EPSILON,
        "only a is within top-3 → 1/2"
    );
}

// T-027: recall_at_k_empty_targets_is_vacuous_one
#[test]
fn recall_at_k_empty_targets_reports_one() {
    assert!(
        (recall_at_k(&vec_s(&["a"]), &set(&[]), 3) - 1.0).abs() < f64::EPSILON,
        "empty targets is a vacuous 1.0 (measure parity)"
    );
}

// T-028: recall_at_k_zero_k_is_zero_when_targets_present
#[test]
fn recall_at_k_zero_k_is_zero() {
    assert!(
        recall_at_k(&vec_s(&["a", "b"]), &set(&["a"]), 0).abs() < f64::EPSILON,
        "k=0 admits no candidate → 0 of 1 present"
    );
}

// T-029: classify_query_identifier_near_when_task_names_seed_stem
#[test]
fn classify_query_near_when_task_names_seed() {
    let class = classify_query(
        "Add a new ModernBERT config field and thread it through the forward pass",
        &vec_s(&["src/modernbert/config.rs"]),
    );
    assert_eq!(
        class,
        QueryClass::IdentifierNear,
        "task literally names modernbert and config"
    );
}

// T-030: classify_query_near_via_substring_stem_match
#[test]
fn classify_query_near_when_seed_stem_is_substring_of_task_word() {
    // The seed dir `embed` is a substring of the task word `embedding`.
    let class = classify_query(
        "Modify the batched MLX forward pass in the embedding backend",
        &vec_s(&["src/embed/mlx.rs"]),
    );
    assert_eq!(
        class,
        QueryClass::IdentifierNear,
        "embed is a substring of embedding (and mlx names mlx) → near"
    );
}

// T-031: classify_query_semantic_far_when_no_identifier_overlap
#[test]
fn classify_query_far_when_no_identifier_overlap() {
    // Semantic-jump query: describes intent, names no seed identifier.
    let class = classify_query(
        "Verify the credential check when a session expires",
        &vec_s(&["src/auth/login.rs"]),
    );
    assert_eq!(
        class,
        QueryClass::SemanticFar,
        "neither auth nor login appears in the task → far"
    );
}

// T-032: classify_query_excludes_generic_src_dir
#[test]
fn classify_query_ignores_generic_src_dir() {
    // `src` is a generic dir and must not by itself make a query near.
    let class = classify_query(
        "Refactor the source tree layout",
        &vec_s(&["src/auth/login.rs"]),
    );
    assert_eq!(
        class,
        QueryClass::SemanticFar,
        "src is excluded and auth/login are absent → far"
    );
}

// T-033: summarize_arms_computes_hit_at_k_and_mean_recall_per_group
#[test]
fn summarize_arms_hit_at_k_per_arm_and_class() {
    let entries = vec![
        ArmEntryReport {
            id: "e1".to_owned(),
            arm: Arm::Embedding,
            class: QueryClass::IdentifierNear,
            seed_rank: Some(0),
            must_recall: 1.0,
        },
        ArmEntryReport {
            id: "e2".to_owned(),
            arm: Arm::Embedding,
            class: QueryClass::IdentifierNear,
            seed_rank: Some(4),
            must_recall: 0.5,
        },
        ArmEntryReport {
            id: "e3".to_owned(),
            arm: Arm::NoEmbed,
            class: QueryClass::IdentifierNear,
            seed_rank: None,
            must_recall: 0.0,
        },
    ];
    let summaries = summarize_arms(&entries, &[1, 5]);

    let emb = summaries
        .iter()
        .find(|s| s.arm == Arm::Embedding && s.class == QueryClass::IdentifierNear)
        .expect("embedding/near group present");
    assert_eq!(emb.n, 2, "two embedding/near entries");
    assert!(
        (emb.hit_at_k[0] - 0.5).abs() < f64::EPSILON,
        "Hit@1: only e1 (rank 0) qualifies → 1/2, got {}",
        emb.hit_at_k[0]
    );
    assert!(
        (emb.hit_at_k[1] - 1.0).abs() < f64::EPSILON,
        "Hit@5: e1 (rank 0) and e2 (rank 4) both qualify → 2/2, got {}",
        emb.hit_at_k[1]
    );
    assert!(
        (emb.mean_must_recall - 0.75).abs() < f64::EPSILON,
        "mean must_recall = (1.0 + 0.5) / 2, got {}",
        emb.mean_must_recall
    );

    let noe = summaries
        .iter()
        .find(|s| s.arm == Arm::NoEmbed && s.class == QueryClass::IdentifierNear)
        .expect("no-embed/near group present");
    assert!(
        (noe.hit_at_k[1] - 0.0).abs() < f64::EPSILON,
        "missed seed → Hit@5 = 0, got {}",
        noe.hit_at_k[1]
    );

    assert!(
        !summaries.iter().any(|s| s.class == QueryClass::SemanticFar),
        "no semantic-far entries → those groups are skipped, not emitted as zero rows"
    );
}

// T-034: arm_comparison_report_renders_arm_labels_in_json_and_plain
#[test]
fn arm_comparison_report_json_and_plain() {
    let entries = vec![
        ArmEntryReport {
            id: "entry-one".to_owned(),
            arm: Arm::Embedding,
            class: QueryClass::IdentifierNear,
            seed_rank: Some(0),
            must_recall: 1.0,
        },
        ArmEntryReport {
            id: "entry-one".to_owned(),
            arm: Arm::NoEmbed,
            class: QueryClass::IdentifierNear,
            seed_rank: None,
            must_recall: 0.25,
        },
    ];
    let report = ArmComparisonReport::new("rurico".to_owned(), entries, &[1, 5], false);

    let json = render_arm_json(&report);
    assert!(
        json.contains("\"repo\":\"rurico\""),
        "repo present, got: {json}"
    );
    assert!(
        json.contains("\"arm\":\"embedding\""),
        "embedding arm label (kebab), got: {json}"
    );
    assert!(
        json.contains("\"arm\":\"no-embed\""),
        "no-embed arm label (kebab), got: {json}"
    );

    let plain = render_arm_plain(&report);
    assert!(plain.contains("embedding"), "plain names the embedding arm");
    assert!(plain.contains("no-embed"), "plain names the no-embed arm");
    assert!(
        plain.contains("seed_rank=miss"),
        "a missed seed renders as miss, got: {plain}"
    );
    assert!(
        plain.contains("seed_rank=0"),
        "a hit seed renders its rank, got: {plain}"
    );
    assert!(
        plain.contains("directional"),
        "plain states the N limitation, got: {plain}"
    );
}
