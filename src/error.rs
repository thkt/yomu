//! Error classification per ADR-0066 Group 2 baseline.
//!
//! `ErrorCode` is the single source of truth for both exit code numbers
//! (sysexits.h via [`amici::cli::exit_code::codes`]) and the `error.code`
//! field emitted in `--json` mode. Variants serialize as `SCREAMING_SNAKE_CASE`
//! so JSON consumers branch on the concept name, not the sysexits label.
//!
//! yomu uses a 6-variant subset (no `DATA_ERROR`) because current
//! [`YomuError`](crate::tools::YomuError) variants do not distinguish
//! schema-level data faults from generic usage errors. Add a variant if a
//! future error path needs it.

use amici::cli::exit_code::codes;
use serde::Serialize;

/// JSON-serializable error classification per ADR-0066 Group 2.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorCode {
    UsageError,
    Internal,
    CantCreat,
    IoError,
    TempFailure,
    Unknown,
}

impl ErrorCode {
    /// Returns the sysexits-derived `u8` exit code. Wrap with
    /// `ExitCode::from(...)` at the CLI boundary.
    pub fn exit_code(self) -> u8 {
        match self {
            Self::UsageError => codes::USAGE,
            Self::Internal => codes::INTERNAL,
            Self::CantCreat => codes::CANT_CREAT,
            Self::IoError => codes::IO_ERR,
            Self::TempFailure => codes::TEMP_FAIL,
            Self::Unknown => codes::UNKNOWN,
        }
    }
}

/// Wire envelope emitted to stderr when `--json` is set and the command failed.
#[derive(Debug, Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorPayload<'a>,
}

#[derive(Debug, Serialize)]
struct ErrorPayload<'a> {
    code: ErrorCode,
    message: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    next_step: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    candidates: Vec<String>,
    retryable: bool,
}

/// Serializes the JSON error envelope as a single line, suitable for
/// `eprintln!`. Convenience wrapper for legacy call sites without ADR-0060
/// fields; new sites should call [`render_json_error_with`].
pub fn render_json_error(code: ErrorCode, message: &str) -> String {
    render_json_error_with(code, message, None, &[], false)
}

/// Renders the JSON error envelope with ADR-0060 agent-friendly fields.
/// `next_step` is omitted when `None`, `candidates` is omitted when empty,
/// `retryable` is always present.
pub fn render_json_error_with(
    code: ErrorCode,
    message: &str,
    next_step: Option<String>,
    candidates: &[String],
    retryable: bool,
) -> String {
    serde_json::to_string(&ErrorEnvelope {
        error: ErrorPayload {
            code,
            message,
            next_step,
            candidates: candidates.to_vec(),
            retryable,
        },
    })
    .expect("ErrorEnvelope is always serializable")
}

#[cfg(test)]
mod tests {
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
        let json =
            render_json_error_with(ErrorCode::UsageError, "msg", None, &["a".to_owned()], false);
        let parsed: serde_json::Value =
            serde_json::from_str(&json).unwrap_or_else(|e| panic!("invalid JSON: {e}: {json}"));
        assert_eq!(
            parsed["error"]["candidates"],
            serde_json::json!(["a"]),
            "json: {json}"
        );
    }
}
