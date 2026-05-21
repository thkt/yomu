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
mod tests {
    use super::*;
    use std::fs as stdfs;
    use tempfile::tempdir;
    use tracing_test::traced_test;

    // T-FSO-001: read_to_string_optional returns Some(content) for an existing file.
    #[test]
    fn read_to_string_optional_returns_content() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("file.txt");
        stdfs::write(&path, "hello").unwrap();
        assert_eq!(read_to_string_optional(&path), Some("hello".to_owned()));
    }

    // T-FSO-002: read_to_string_optional returns None silently for NotFound.
    #[test]
    #[traced_test]
    fn read_to_string_optional_silent_on_notfound() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("nope.txt");
        assert_eq!(read_to_string_optional(&missing), None);
        assert!(
            !logs_contain("unexpected I/O error"),
            "NotFound must not emit warn"
        );
    }

    // T-FSO-003: read_to_string_optional warns on non-NotFound errors.
    #[test]
    #[traced_test]
    fn read_to_string_optional_warns_on_unexpected_error() {
        let dir = tempdir().unwrap();
        // Read a directory as a file — yields a non-NotFound I/O error.
        assert_eq!(read_to_string_optional(dir.path()), None);
        assert!(
            logs_contain("unexpected I/O error"),
            "non-NotFound error must emit warn"
        );
    }

    // T-FSO-004: canonicalize_optional returns Some(canonical) for an existing path.
    #[test]
    fn canonicalize_optional_returns_canonical() {
        let dir = tempdir().unwrap();
        let resolved = canonicalize_optional(dir.path()).expect("tempdir must canonicalize");
        assert!(resolved.is_absolute());
    }

    // T-FSO-005: canonicalize_optional returns None silently for NotFound.
    #[test]
    #[traced_test]
    fn canonicalize_optional_silent_on_notfound() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("nope");
        assert_eq!(canonicalize_optional(&missing), None);
        assert!(
            !logs_contain("canonicalize failed unexpectedly"),
            "NotFound must not emit warn"
        );
    }

    // T-FSO-005b: canonicalize_optional warns on non-NotFound errors.
    #[cfg(unix)]
    #[test]
    #[traced_test]
    fn canonicalize_optional_warns_on_unexpected_error() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir().unwrap();
        let restricted = dir.path().join("locked");
        stdfs::create_dir(&restricted).unwrap();
        let inside = restricted.join("target");
        stdfs::write(&inside, "x").unwrap();
        // Remove search permission so canonicalize traversal yields PermissionDenied.
        stdfs::set_permissions(&restricted, stdfs::Permissions::from_mode(0o000)).unwrap();
        let result = canonicalize_optional(&inside);
        // Restore permission so tempdir cleanup succeeds.
        stdfs::set_permissions(&restricted, stdfs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(result, None);
        assert!(
            logs_contain("canonicalize failed unexpectedly"),
            "non-NotFound error must emit warn"
        );
    }

    // T-FSO-006: read_mtime_epoch returns Some(epoch) for an existing file.
    #[test]
    fn read_mtime_epoch_returns_value() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("file.txt");
        stdfs::write(&path, "x").unwrap();
        let epoch = read_mtime_epoch(&path).expect("just-written file must have mtime");
        assert!(epoch > 0, "epoch must be positive, got {epoch}");
    }

    // T-FSO-007: read_mtime_epoch returns None silently for NotFound.
    #[test]
    #[traced_test]
    fn read_mtime_epoch_silent_on_notfound() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("nope.txt");
        assert_eq!(read_mtime_epoch(&missing), None);
        assert!(
            !logs_contain("metadata read failed"),
            "NotFound must not emit warn"
        );
    }

    // T-FSO-008: read_mtime_epoch warns on non-NotFound metadata errors.
    #[cfg(unix)]
    #[test]
    #[traced_test]
    fn read_mtime_epoch_warns_on_unexpected_error() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempdir().unwrap();
        let restricted = dir.path().join("locked");
        stdfs::create_dir(&restricted).unwrap();
        let inside = restricted.join("target.txt");
        stdfs::write(&inside, "x").unwrap();
        // Remove search permission so fs::metadata traversal yields PermissionDenied.
        stdfs::set_permissions(&restricted, stdfs::Permissions::from_mode(0o000)).unwrap();
        let result = read_mtime_epoch(&inside);
        // Restore permission so tempdir cleanup succeeds.
        stdfs::set_permissions(&restricted, stdfs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(result, None);
        assert!(
            logs_contain("metadata read failed"),
            "non-NotFound error must emit warn"
        );
    }
}
