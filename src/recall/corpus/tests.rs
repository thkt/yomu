use super::*;
use std::collections::{HashMap, HashSet};

fn wf(path: &str, weight: u32) -> WeightedFile {
    WeightedFile {
        path: path.to_owned(),
        weight,
    }
}

fn entry(id: &str, seed: &[&str], must_include: Vec<WeightedFile>) -> GtEntry {
    GtEntry {
        id: id.to_owned(),
        repo: "rurico".to_owned(),
        rev: "deadbeef".to_owned(),
        domain: "test".to_owned(),
        task: "test task".to_owned(),
        seed: seed.iter().map(|s| (*s).to_owned()).collect(),
        rationale: "test rationale".to_owned(),
        must_include,
    }
}

// T-018: load_from_str_parses_valid_entries
#[test]
fn load_from_str_parses_valid_entries() {
    let yaml = r#"
entries:
  - id: e1
    repo: rurico
    rev: 1d6650a
    domain: scoring
    task: adjust the reranker scoring
    seed:
      - src/reranker/score.rs
    rationale: scoring spans the trait and its caller
    must_include:
      - path: src/reranker/score.rs
        weight: 5
      - path: src/reranker.rs
        weight: 3
  - id: e2
    repo: amici
    rev: ae8c068
    domain: storage
    task: change persistence schema
    seed:
      - src/storage.rs
    rationale: storage spans schema and migration
    must_include:
      - path: src/storage.rs
        weight: 4
"#;
    let corpus = load_from_str(yaml).expect("valid 2-entry yaml loads");
    assert_eq!(corpus.entries.len(), 2, "two entries parsed");
    let e1 = &corpus.entries[0];
    assert_eq!(e1.id, "e1");
    assert_eq!(e1.seed, vec!["src/reranker/score.rs".to_owned()]);
    assert_eq!(e1.must_include.len(), 2, "both must-include files retained");
    assert_eq!(e1.must_include[0].weight, 5, "per-file weight preserved");
}

// T-007: load_from_str_rejects_zero_weight_naming_entry
#[test]
fn load_from_str_rejects_zero_weight_naming_entry() {
    let yaml = r#"
entries:
  - id: bad-weight
    repo: rurico
    rev: 1d6650a
    domain: d
    task: t
    seed:
      - src/a.rs
    rationale: r
    must_include:
      - path: src/a.rs
        weight: 0
"#;
    let err = load_from_str(yaml).expect_err("zero weight must error");
    assert!(
        matches!(&err, GtLoadError::ZeroWeight { id, .. } if id == "bad-weight"),
        "error names the offending entry id, got: {err:?}"
    );
}

// T-019: load_from_str_rejects_empty_must_include
#[test]
fn load_from_str_rejects_empty_must_include() {
    let yaml = r#"
entries:
  - id: empty-mi
    repo: amici
    rev: ae8c068
    domain: d
    task: t
    seed:
      - src/a.rs
    rationale: r
    must_include: []
"#;
    let err = load_from_str(yaml).expect_err("empty must_include must error");
    assert!(
        matches!(&err, GtLoadError::EmptyMustInclude { id } if id == "empty-mi"),
        "error names the offending entry id, got: {err:?}"
    );
}

// T-006: canary_flags_entry_with_all_must_include_inside_forward_closure
#[test]
fn canary_flags_entry_with_all_must_include_inside_forward_closure() {
    // `taut`: every must-include sits within the 1-hop forward closure → tautological.
    let taut = entry(
        "taut",
        &["src/seed.rs"],
        vec![wf("src/seed.rs", 3), wf("src/near.rs", 2)],
    );
    // `good`: src/far.rs is outside the forward closure (reverse/multi-hop) → exercises closure.
    let good = entry(
        "good",
        &["src/seed2.rs"],
        vec![wf("src/seed2.rs", 3), wf("src/far.rs", 2)],
    );
    let entries = vec![taut, good];

    let mut forward_1hop: HashMap<String, HashSet<String>> = HashMap::new();
    forward_1hop.insert(
        "taut".to_owned(),
        HashSet::from(["src/seed.rs".to_owned(), "src/near.rs".to_owned()]),
    );
    forward_1hop.insert(
        "good".to_owned(),
        HashSet::from(["src/seed2.rs".to_owned()]),
    );

    let violations = canary_violations(&entries, &forward_1hop);
    assert_eq!(
        violations,
        vec!["taut".to_owned()],
        "only the tautological entry is flagged, named by id"
    );
}

// T-005: bundled_gt_corpus_loads_with_ten_weighted_entries
#[test]
fn bundled_gt_corpus_loads_with_ten_weighted_entries() {
    let corpus = load_bundled().expect("bundled gt.yaml loads and passes weight validation");
    assert!(
        corpus.entries.len() >= 10,
        "FR-005: corpus must carry >= 10 domain-diverse entries, got {}",
        corpus.entries.len()
    );
    for entry in &corpus.entries {
        assert!(
            !entry.seed.is_empty(),
            "entry {:?} must declare at least one seed",
            entry.id
        );
        // load_from_str already rejects empty must-include and zero weights;
        // assert the positive-weight invariant explicitly as a standing guard.
        assert!(
            entry.must_include.iter().all(|file| file.weight > 0),
            "entry {:?} must-include weights must all be positive",
            entry.id
        );
    }
}

// T-021: canary_counts_seed_as_trivially_reachable
#[test]
fn canary_counts_seed_as_trivially_reachable() {
    // The closure source here omits the seed (src/s.rs) and lists only its
    // forward dep. Without seed-awareness the seed would read as "outside the
    // forward closure" and the entry would pass the canary via its own seed —
    // toothless. Seed-awareness keeps it flagged.
    let entry = entry(
        "seed-only",
        &["src/s.rs"],
        vec![wf("src/s.rs", 5), wf("src/dep.rs", 2)],
    );
    let mut forward_1hop: HashMap<String, HashSet<String>> = HashMap::new();
    forward_1hop.insert(
        "seed-only".to_owned(),
        HashSet::from(["src/dep.rs".to_owned()]),
    );

    let violations = canary_violations(&[entry], &forward_1hop);
    assert_eq!(
        violations,
        vec!["seed-only".to_owned()],
        "an entry whose only out-of-closure must-include is its own seed is tautological"
    );
}
