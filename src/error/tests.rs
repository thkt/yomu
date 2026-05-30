use super::*;

// T-EC001: each ErrorCode maps to the documented sysexits number.
#[test]
fn exit_code_numbers_match_sysexits_baseline() {
    assert_eq!(ErrorCode::UsageError.exit_code(), 64);
    assert_eq!(ErrorCode::Internal.exit_code(), 70);
    assert_eq!(ErrorCode::CantCreat.exit_code(), 73);
    assert_eq!(ErrorCode::IoError.exit_code(), 74);
    assert_eq!(ErrorCode::TempFailure.exit_code(), 75);
    assert_eq!(ErrorCode::Unknown.exit_code(), 104);
}

// T-EC002: JSON envelope serializes the variant in SCREAMING_SNAKE_CASE.
// Issue #192 Phase 2.2a: envelope now always includes `retryable` (FR-005).
#[test]
fn json_envelope_uses_screaming_snake_case() {
    let json = render_json_error(ErrorCode::UsageError, "bad arg");
    assert_eq!(
        json,
        r#"{"error":{"code":"USAGE_ERROR","message":"bad arg","retryable":false}}"#
    );
}

// T-EC003: Unknown variant serializes as the "UNKNOWN" label.
#[test]
fn json_envelope_unknown_serializes_to_unknown_label() {
    let json = render_json_error(ErrorCode::Unknown, "x");
    assert!(json.contains(r#""code":"UNKNOWN""#), "got: {json}");
}

// --- Issue #192 Phase 2.2a: render_json_error_with serialization ---
//
// Spec: .claude/workspace/planning/2026-05-20-192-phase-2-2a-error-payload/spec.md
// T-018 ~ T-020 validate the extended envelope shape (next_step /
// candidates / retryable) follows the omit-when-empty contract
// (FR-002 ~ FR-005).

// T-018: render_json_error_with serializes the full envelope when
// next_step is Some, candidates is empty, and retryable is false.
// FR-002, FR-005. Candidates key is omitted on empty (FR-004).
#[test]
fn render_json_error_with_emits_next_step_and_retryable_without_candidates() {
    let json = render_json_error_with(
        ErrorCode::UsageError,
        "msg",
        Some("step".to_owned()),
        &[],
        false,
    );
    let parsed: serde_json::Value =
        serde_json::from_str(&json).unwrap_or_else(|e| panic!("invalid JSON: {e}: {json}"));
    assert_eq!(parsed["error"]["code"], "USAGE_ERROR", "json: {json}");
    assert_eq!(parsed["error"]["message"], "msg", "json: {json}");
    assert_eq!(parsed["error"]["next_step"], "step", "json: {json}");
    assert_eq!(parsed["error"]["retryable"], false, "json: {json}");
    assert!(
        parsed["error"].get("candidates").is_none(),
        "candidates key must be omitted when empty: {json}"
    );
}

// T-019: render_json_error_with omits next_step entirely when None.
// FR-003. retryable remains always-present per FR-005.
#[test]
fn render_json_error_with_omits_next_step_when_none() {
    let json = render_json_error_with(ErrorCode::UsageError, "msg", None, &[], false);
    let parsed: serde_json::Value =
        serde_json::from_str(&json).unwrap_or_else(|e| panic!("invalid JSON: {e}: {json}"));
    assert!(
        parsed["error"].get("next_step").is_none(),
        "next_step key must be omitted when None: {json}"
    );
    assert_eq!(parsed["error"]["retryable"], false, "json: {json}");
}

// T-020: render_json_error_with serializes candidates when the slice is
// non-empty. FR-004.
#[test]
fn render_json_error_with_includes_candidates_when_non_empty() {
    let json = render_json_error_with(ErrorCode::UsageError, "msg", None, &["a".to_owned()], false);
    let parsed: serde_json::Value =
        serde_json::from_str(&json).unwrap_or_else(|e| panic!("invalid JSON: {e}: {json}"));
    assert_eq!(
        parsed["error"]["candidates"],
        serde_json::json!(["a"]),
        "json: {json}"
    );
}
