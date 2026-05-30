//! Path-based classification of source files into [`SourceKind::Vendor`] /
//! [`SourceKind::Test`] / [`SourceKind::Src`].
//!
//! Precedence per FR-202: vendor > test > src.

use std::path::Path;

use crate::storage::SourceKind;

const VENDOR_DIRS: &[&str] = &[
    "node_modules",
    "vendor",
    "third_party",
    "bower_components",
    "dist",
    "build",
    "target",
    ".git",
];

const TEST_DIRS: &[&str] = &["tests", "test", "__tests__", "specs", "spec"];

const JS_EXTS: &[&str] = &["ts", "tsx", "js", "jsx", "mjs", "cjs"];

const RS_GO_PY_EXTS: &[&str] = &["rs", "go", "py"];

/// Classify a relative path into a [`SourceKind`].
///
/// Precedence: vendor > test > src (FR-202).
pub fn classify(rel_path: &str) -> SourceKind {
    let components: Vec<&str> = Path::new(rel_path)
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    let ancestor_count = components.len().saturating_sub(1);
    let ancestors = &components[..ancestor_count];

    if ancestors.iter().any(|c| VENDOR_DIRS.contains(c)) {
        return SourceKind::Vendor;
    }
    if ancestors.iter().any(|c| TEST_DIRS.contains(c)) {
        return SourceKind::Test;
    }
    if components.last().is_some_and(|f| is_test_filename(f)) {
        return SourceKind::Test;
    }
    SourceKind::Src
}

fn is_test_filename(name: &str) -> bool {
    let Some((stem, ext)) = name.rsplit_once('.') else {
        return false;
    };
    if JS_EXTS.contains(&ext) && (stem.ends_with(".test") || stem.ends_with(".spec")) {
        return true;
    }
    if RS_GO_PY_EXTS.contains(&ext) && stem.ends_with("_test") {
        return true;
    }
    // Rust inline test module: `#[cfg(test)] mod tests;` split into `tests.rs`.
    if ext == "rs" && stem == "tests" {
        return true;
    }
    false
}

#[cfg(test)]
mod tests;
