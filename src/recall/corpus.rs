//! Ground-truth (GT) corpus loading and the anti-tautology canary for `brief`
//! recall measurement.
//!
//! A GT entry pairs an investigation task with the file set a complete answer
//! must span (`must_include`, per-file weighted). The loader validates weights
//! (FR-016); the canary (FR-006 / BR-002) rejects entries whose every
//! must-include lies inside the seed's 1-hop forward closure, which would make
//! recall vacuously high.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use super::WeightedFile;

/// One ground-truth entry: an investigation task plus the file set a complete
/// answer must span. Deserialized from `tests/fixtures/brief/gt.yaml`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GtEntry {
    /// Stable identifier, unique within the corpus (named in load errors and the canary).
    pub id: String,
    /// Target repository (`rurico` | `amici`).
    pub repo: String,
    /// Pinned submodule rev the seed and must-include paths resolve against.
    pub rev: String,
    /// Domain area the task exercises (e.g. `embedding`, `storage`).
    pub domain: String,
    /// One-sentence investigation or change task the brief must answer.
    pub task: String,
    /// Explicit seed paths (repo-relative) passed to `brief --seed-file`.
    pub seed: Vec<String>,
    /// Why these files form a complete answer: domain reasoning, not derived from
    /// brief output (AC-2 / AS-004).
    pub rationale: String,
    /// Files a complete answer must span, each with a positive importance weight.
    pub must_include: Vec<WeightedFile>,
}

/// A loaded ground-truth corpus.
#[derive(Debug, Clone, Deserialize)]
pub struct GtCorpus {
    pub entries: Vec<GtEntry>,
}

/// Errors returned by [`load_from_str`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GtLoadError {
    /// The yaml did not parse into a [`GtCorpus`].
    #[error("gt corpus parse error: {0}")]
    Parse(#[from] serde_yaml::Error),
    /// An entry's must-include set is empty (FR-016 / entity invariant).
    #[error("gt entry {id:?} has an empty must-include set")]
    EmptyMustInclude { id: String },
    /// A must-include file carries a zero (non-positive) weight (FR-016).
    #[error("gt entry {id:?} must-include {path:?} has zero weight; weight must be positive")]
    ZeroWeight { id: String, path: String },
}

/// Loads and validates a GT corpus from a yaml string.
///
/// # Errors
/// [`GtLoadError::Parse`] on malformed yaml, [`GtLoadError::EmptyMustInclude`] for an
/// entry with no must-include files, and [`GtLoadError::ZeroWeight`] when a must-include
/// weight is zero (FR-016: weights must be positive). Each non-parse error names the
/// offending entry id.
pub fn load_from_str(text: &str) -> Result<GtCorpus, GtLoadError> {
    let corpus: GtCorpus = serde_yaml::from_str(text)?;
    for entry in &corpus.entries {
        if entry.must_include.is_empty() {
            return Err(GtLoadError::EmptyMustInclude {
                id: entry.id.clone(),
            });
        }
        for file in &entry.must_include {
            if file.weight == 0 {
                return Err(GtLoadError::ZeroWeight {
                    id: entry.id.clone(),
                    path: file.path.clone(),
                });
            }
        }
    }
    Ok(corpus)
}

/// Returns the ids of entries that fail the anti-tautology canary (FR-006 / BR-002).
///
/// An entry fails when every must-include file is trivially reachable: it is either
/// the seed itself or lies in the seed's 1-hop forward closure. Such an entry needs
/// no bidirectional or multi-hop closure work, so `recall` would be vacuously high.
/// `forward_1hop` maps each entry id to the file set reachable by 1-hop forward edges
/// from its seed (the `forward_paths` element of `brief::collect_closure(seed, 1)`,
/// which includes the seeds at depth 0). The seed is also checked explicitly so the
/// canary stays strict even if a closure source omits seeds from `forward_1hop`. An
/// entry absent from the map fails the canary because its closure is unproven.
///
/// Pure set logic: the caller wires `collect_closure` (Gate1) or a synthetic set
/// (unit test).
pub fn canary_violations(
    entries: &[GtEntry],
    forward_1hop: &HashMap<String, HashSet<String>>,
) -> Vec<String> {
    entries
        .iter()
        .filter(|entry| match forward_1hop.get(&entry.id) {
            None => true,
            Some(reachable) => entry
                .must_include
                .iter()
                .all(|file| entry.seed.contains(&file.path) || reachable.contains(&file.path)),
        })
        .map(|entry| entry.id.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    const BUNDLED_GT_YAML: &str = include_str!("../../tests/fixtures/brief/gt.yaml");

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
        let corpus = load_from_str(BUNDLED_GT_YAML)
            .expect("bundled gt.yaml loads and passes weight validation");
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
}
