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
mod tests;
