//! Precision/recall measurement for the injection matcher.
//!
//! Pure (no I/O). The caller (`tools::Yomu::verify_standalone` for CLI,
//! `tests/verification.rs` for the gate) is responsible for loading the
//! bundled corpora and assembling `PositiveCase` / `NegativeCase` lists.
//!
//! ADR-0069 contract: `precision >= 0.90 && recall >= 0.95 && !degraded`.
//! Both denominators are kept structurally non-zero (BR-401):
//! - precision denominator: ≥30 independent negative cases.
//! - recall denominator: ≥1 corpus entry with near-miss `test_content`
//!   (FR-403f), regression-canaried by `tests/verification.rs::T-425`.

use serde::Serialize;

use crate::indexer::injection::{Corpus, CorpusEntry, NegativeEntry};
use crate::injection_check::InjectionCheck;

/// Single source for the corpus fixture path. Imported by `Yomu::verify_standalone`,
/// `tests/verification.rs`, and `src/indexer/injection/tests.rs` so a future
/// fixture move requires a single edit.
pub const BUNDLED_CORPUS_YAML: &str = include_str!("../tests/fixtures/injection/corpus.yaml");
pub const BUNDLED_NEGATIVE_YAML: &str =
    include_str!("../tests/fixtures/injection/corpus.negative.yaml");

pub struct PositiveCase {
    pub id: String,
    pub content: String,
    pub expected_flags: Vec<String>,
}

pub struct NegativeCase {
    pub id: String,
    pub corresponds_to: String,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EntryKind {
    Positive,
    Negative,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntryResult {
    pub id: String,
    pub kind: EntryKind,
    pub matched: bool,
    pub expected_flags: Vec<String>,
    pub actual_flags: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct VerificationReport {
    pub precision: f64,
    pub recall: f64,
    pub tp: u32,
    pub fp: u32,
    pub fn_count: u32,
    pub degraded: bool,
    pub details: Vec<EntryResult>,
    pub injection_check: InjectionCheck,
}

/// FR-403e/FR-412 validator: every entry SHALL have a non-empty `test_content`.
/// Returns Err with the offending id if validation fails.
pub fn positives_from_entries(entries: &[CorpusEntry]) -> Result<Vec<PositiveCase>, String> {
    entries
        .iter()
        .map(|e| {
            let content = e
                .test_content
                .as_ref()
                .filter(|s| !s.trim().is_empty())
                .ok_or_else(|| {
                    format!(
                        "corpus entry {} is missing non-empty `test_content` (FR-403e)",
                        e.id
                    )
                })?;
            Ok(PositiveCase {
                id: e.id.clone(),
                content: content.clone(),
                expected_flags: e.expected_flags.clone(),
            })
        })
        .collect()
}

pub fn negatives_from_entries(entries: &[NegativeEntry]) -> Vec<NegativeCase> {
    entries
        .iter()
        .map(|n| NegativeCase {
            id: n.id.clone(),
            corresponds_to: n.corresponds_to.clone(),
            content: n.content.clone(),
        })
        .collect()
}

pub fn measure(
    corpus: &Corpus,
    positives: &[PositiveCase],
    negatives: &[NegativeCase],
) -> VerificationReport {
    let mut tp = 0u32;
    let mut fp = 0u32;
    let mut fn_count = 0u32;
    let mut details = Vec::with_capacity(positives.len() + negatives.len());

    for p in positives {
        let actual_flags = corpus.check_chunk(&p.content);
        let matched = p.expected_flags.iter().any(|ef| actual_flags.contains(ef));
        if matched {
            tp += 1;
        } else {
            fn_count += 1;
        }
        details.push(EntryResult {
            id: p.id.clone(),
            kind: EntryKind::Positive,
            matched,
            expected_flags: p.expected_flags.clone(),
            actual_flags,
        });
    }

    for n in negatives {
        let actual_flags = corpus.check_chunk(&n.content);
        let matched = !actual_flags.is_empty();
        if matched {
            fp += 1;
        }
        details.push(EntryResult {
            id: n.id.clone(),
            kind: EntryKind::Negative,
            matched,
            expected_flags: Vec::new(),
            actual_flags,
        });
    }

    let (precision, recall, degraded) = compute_rates(tp, fp, fn_count);

    VerificationReport {
        precision,
        recall,
        tp,
        fp,
        fn_count,
        degraded,
        details,
        injection_check: InjectionCheck::Ran,
    }
}

/// Returns `(precision, recall, degraded)`. When a denominator is zero the
/// rate is vacuously 1.0 and `degraded` is set so the gate (T-401) fails
/// loudly instead of silently passing on a vacuous metric.
fn compute_rates(tp: u32, fp: u32, fn_count: u32) -> (f64, f64, bool) {
    let precision_denom = tp + fp;
    let recall_denom = tp + fn_count;
    let precision_degraded = precision_denom == 0;
    let recall_degraded = recall_denom == 0;
    let precision = if precision_degraded {
        1.0
    } else {
        f64::from(tp) / f64::from(precision_denom)
    };
    let recall = if recall_degraded {
        1.0
    } else {
        f64::from(tp) / f64::from(recall_denom)
    };
    (precision, recall, precision_degraded || recall_degraded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_literal_corpus(pattern: &str, flag: &str) -> Corpus {
        let yaml = format!(
            r#"entries:
  - id: only
    pattern_type: literal
    pattern: "{pattern}"
    severity: high
    category: c
    expected_flags: ["{flag}"]
"#
        );
        Corpus::load_from_str(&yaml).expect("test corpus loads")
    }

    fn pos(id: &str, content: &str, flags: &[&str]) -> PositiveCase {
        PositiveCase {
            id: id.to_owned(),
            content: content.to_owned(),
            expected_flags: flags.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    fn neg(id: &str, content: &str) -> NegativeCase {
        NegativeCase {
            id: id.to_owned(),
            corresponds_to: "only".to_owned(),
            content: content.to_owned(),
        }
    }

    // T-411: measure_one_positive_match_zero_negatives
    #[test]
    fn measure_one_positive_match_zero_negatives() {
        let corpus = one_literal_corpus("X", "flag.x");
        let positives = vec![pos("p1", "X here", &["flag.x"])];
        let negatives: Vec<NegativeCase> = vec![];
        let r = measure(&corpus, &positives, &negatives);
        assert_eq!(r.tp, 1);
        assert_eq!(r.fp, 0);
        assert_eq!(r.fn_count, 0);
        assert!((r.precision - 1.0).abs() < f64::EPSILON);
        assert!((r.recall - 1.0).abs() < f64::EPSILON);
        // recall denominator (tp+fn=1) is non-zero; precision denominator
        // (tp+fp=1) is non-zero. Both axes valid. degraded=false.
        assert!(!r.degraded);
    }

    // T-412: measure_one_positive_miss_marks_recall_valid_precision_degraded
    #[test]
    fn measure_one_positive_miss_marks_recall_valid_precision_degraded() {
        let corpus = one_literal_corpus("X", "flag.x");
        // Y doesn't contain X → FN
        let positives = vec![pos("p1", "Y here", &["flag.x"])];
        let negatives: Vec<NegativeCase> = vec![];
        let r = measure(&corpus, &positives, &negatives);
        assert_eq!(r.tp, 0);
        assert_eq!(r.fp, 0);
        assert_eq!(r.fn_count, 1);
        assert!((r.recall - 0.0).abs() < f64::EPSILON);
        // precision denominator (tp+fp=0) is vacuous → precision=1.0, degraded=true
        assert!((r.precision - 1.0).abs() < f64::EPSILON);
        assert!(
            r.degraded,
            "precision denominator zero must set degraded=true"
        );
    }

    // T-413: measure_one_negative_fp_recall_degraded
    #[test]
    fn measure_one_negative_fp_recall_degraded() {
        let corpus = one_literal_corpus("X", "flag.x");
        let positives: Vec<PositiveCase> = vec![];
        // negative content contains X → FP
        let negatives = vec![neg("n1", "X is matched")];
        let r = measure(&corpus, &positives, &negatives);
        assert_eq!(r.tp, 0);
        assert_eq!(r.fp, 1);
        assert_eq!(r.fn_count, 0);
        assert!((r.precision - 0.0).abs() < f64::EPSILON);
        // recall denominator (tp+fn=0) is vacuous → recall=1.0, degraded=true
        assert!((r.recall - 1.0).abs() < f64::EPSILON);
        assert!(r.degraded, "recall denominator zero must set degraded=true");
    }

    // T-414: measure_one_positive_one_negative_both_match
    #[test]
    fn measure_one_positive_one_negative_both_match() {
        let corpus = one_literal_corpus("X", "flag.x");
        let positives = vec![pos("p1", "X here", &["flag.x"])];
        let negatives = vec![neg("n1", "X is matched")];
        let r = measure(&corpus, &positives, &negatives);
        assert_eq!(r.tp, 1);
        assert_eq!(r.fp, 1);
        assert_eq!(r.fn_count, 0);
        // precision = 1/2 = 0.5, recall = 1/1 = 1.0
        assert!((r.precision - 0.5).abs() < f64::EPSILON);
        assert!((r.recall - 1.0).abs() < f64::EPSILON);
        assert!(!r.degraded);
    }

    // T-415: measure_two_each_mixed_outcomes
    #[test]
    fn measure_two_each_mixed_outcomes() {
        let corpus = one_literal_corpus("X", "flag.x");
        let positives = vec![
            pos("p1", "X here", &["flag.x"]),
            pos("p2", "Y here", &["flag.x"]),
        ];
        let negatives = vec![neg("n1", "X is matched"), neg("n2", "Z is clean")];
        let r = measure(&corpus, &positives, &negatives);
        assert_eq!(r.tp, 1);
        assert_eq!(r.fp, 1);
        assert_eq!(r.fn_count, 1);
        // precision = 1/2, recall = 1/2
        assert!((r.precision - 0.5).abs() < f64::EPSILON);
        assert!((r.recall - 0.5).abs() < f64::EPSILON);
        assert!(!r.degraded);
    }
}
