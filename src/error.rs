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
pub struct ErrorEnvelope<'a> {
    pub error: ErrorPayload<'a>,
}

#[derive(Debug, Serialize)]
pub struct ErrorPayload<'a> {
    pub code: ErrorCode,
    pub message: &'a str,
}

/// Serializes the JSON error envelope as a single line, suitable for
/// `eprintln!`. Never returns an `Err` because the payload only carries
/// owned/borrowed primitives.
pub fn render_json_error(code: ErrorCode, message: &str) -> String {
    serde_json::to_string(&ErrorEnvelope {
        error: ErrorPayload { code, message },
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

    // T-EC002: JSON envelope round-trips with SCREAMING_SNAKE_CASE code.
    #[test]
    fn json_envelope_uses_screaming_snake_case() {
        let json = render_json_error(ErrorCode::UsageError, "bad arg");
        assert_eq!(
            json,
            r#"{"error":{"code":"USAGE_ERROR","message":"bad arg"}}"#
        );
    }

    // T-EC003: Unknown variant surfaces as the project extension code.
    #[test]
    fn json_envelope_unknown_serializes_to_unknown_label() {
        let json = render_json_error(ErrorCode::Unknown, "x");
        assert!(json.contains(r#""code":"UNKNOWN""#), "got: {json}");
    }
}
