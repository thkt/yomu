//! Verification gate: precision >= 0.90, recall >= 0.95, !degraded.
//!
//! Spec FR-408, FR-411, BR-403, BR-401, BR-407.

use yomu::indexer::injection::{Corpus, NegativeFile};
use yomu::verify::{
    BUNDLED_CORPUS_YAML, BUNDLED_NEGATIVE_YAML, measure, negatives_from_entries,
    positives_from_entries,
};

// T-401: verification_gate_precision_recall_above_threshold
// Spec FR-408 / BR-403: precision >= 0.90 && recall >= 0.95 && !degraded.
// Spec FR-411: MUST NOT be `#[ignore]` or skipped under any cfg.
#[test]
fn verification_gate_precision_recall_above_threshold() {
    let (corpus, entries) =
        Corpus::load_with_entries(BUNDLED_CORPUS_YAML).expect("production corpus loads");
    let neg_file =
        NegativeFile::load_from_str(BUNDLED_NEGATIVE_YAML).expect("negative corpus loads");

    let positives = positives_from_entries(&entries)
        .expect("every positive entry has non-empty test_content (FR-403e/FR-412)");
    let negatives = negatives_from_entries(&neg_file.entries);

    let report = measure(&corpus, &positives, &negatives);

    assert!(
        !report.degraded,
        "FR-405d / BR-404: degraded=true means a denominator was vacuous. \
         tp={} fp={} fn={} positives={} negatives={}",
        report.tp,
        report.fp,
        report.fn_count,
        positives.len(),
        negatives.len()
    );
    assert!(
        report.precision >= 0.90,
        "BR-403: precision must be >= 0.90, got {}. tp={} fp={}",
        report.precision,
        report.tp,
        report.fp
    );
    assert!(
        report.recall >= 0.95,
        "BR-403: recall must be >= 0.95, got {}. tp={} fn={}",
        report.recall,
        report.tp,
        report.fn_count
    );
}

// T-425: recall_denominator_non_zero_canary
// Spec FR-403f / BR-407: at least 1 positive entry must produce FN under
// `verify::measure` so the recall denominator is not vacuous. Empty negatives
// list isolates the recall side (FP=0 by construction).
#[test]
fn recall_denominator_non_zero_canary() {
    let (corpus, entries) =
        Corpus::load_with_entries(BUNDLED_CORPUS_YAML).expect("production corpus loads");
    let positives =
        positives_from_entries(&entries).expect("every positive entry has non-empty test_content");

    let report = measure(&corpus, &positives, &[]);

    assert!(
        report.fn_count >= 1,
        "FR-403f / BR-407: at least 1 entry must produce FN to keep the recall \
         denominator non-zero (trivial-100 trap protection). Got fn_count={}. \
         verification-spec sets the 30-entry budget at max FN=1 (recall 0.95 = 28.5/30), \
         so the canary lower bound is 1.",
        report.fn_count
    );
}
