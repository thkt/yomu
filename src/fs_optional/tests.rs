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
