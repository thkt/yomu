//! Filesystem reads that distinguish "expected absence" from "unexpected I/O failure".
//!
//! Each helper returns `None` on the `NotFound` absence path silently, but
//! emits `tracing::warn!` for other I/O errors (permissions, corrupted FS,
//! platform mtime unsupported, pre-1970 mtime, etc.) so operators can
//! investigate anomalies that previously hid behind plain `.ok()` calls
//! (Issue #134 SIL-003, SIL-006, SIL-007, RES-002).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Read a UTF-8 file. `NotFound` returns `None` silently; other errors warn
/// and return `None`.
pub fn read_to_string_optional(path: &Path) -> Option<String> {
    match fs::read_to_string(path) {
        Ok(c) => Some(c),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "unexpected I/O error reading file"
            );
            None
        }
    }
}

/// Canonicalize a path. `NotFound` returns `None` silently; other errors warn
/// and return `None`.
pub fn canonicalize_optional(path: &Path) -> Option<PathBuf> {
    match path.canonicalize() {
        Ok(p) => Some(p),
        Err(e) if e.kind() == io::ErrorKind::NotFound => None,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "canonicalize failed unexpectedly"
            );
            None
        }
    }
}

/// Read a file's mtime as Unix epoch seconds (saturating at `i64::MAX` for
/// year-2262+ wraparound). `NotFound` returns `None` silently; other errors
/// (permissions, platform without mtime, pre-1970 mtime) warn and return `None`.
pub fn read_mtime_epoch(path: &Path) -> Option<i64> {
    let metadata = match fs::metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return None,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "metadata read failed"
            );
            return None;
        }
    };
    let modified = match metadata.modified() {
        Ok(t) => t,
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "mtime unsupported on this platform"
            );
            return None;
        }
    };
    match modified.duration_since(UNIX_EPOCH) {
        Ok(d) => Some(i64::try_from(d.as_secs()).unwrap_or(i64::MAX)),
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "mtime predates UNIX epoch"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests;
