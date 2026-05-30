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
