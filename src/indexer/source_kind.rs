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
mod tests {
    use super::*;

    // T-201: vendor_ancestor_directory_names_return_vendor
    //
    // Perspective: Equivalence + Boundary. The 8 vendor directory names from
    // FR-203 form one equivalence class; covering each member is the boundary
    // of the set.
    //
    // FR: FR-201, FR-203
    #[test]
    fn vendor_ancestor_directory_names_return_vendor() {
        let cases = [
            "node_modules/x.ts",
            "vendor/y.rs",
            "third_party/z.js",
            "bower_components/a.css",
            "dist/b.html",
            "build/c.md",
            "target/d.rs",
            ".git/e.txt",
        ];
        for input in cases {
            assert_eq!(
                classify(input),
                SourceKind::Vendor,
                "expected SourceKind::Vendor for input {input:?}"
            );
        }
    }

    // T-202: test_extension_or_test_directory_returns_test
    //
    // Perspective: Branch + Equivalence. FR-204 defines two independent
    // branches (extension match OR ancestor-directory match). Cover both.
    //
    // Decision table:
    //
    // | row | extension match | dir match | expected |
    // | --- | --------------- | --------- | -------- |
    // | a   | T               | F         | Test     |
    // | b   | F               | T         | Test     |
    //
    // FR: FR-201, FR-204
    #[test]
    fn test_extension_or_test_directory_returns_test() {
        let extension_cases = ["src/foo.test.ts", "src/bar.spec.tsx", "src/baz_test.rs"];
        for input in extension_cases {
            assert_eq!(
                classify(input),
                SourceKind::Test,
                "expected SourceKind::Test (extension branch) for input {input:?}"
            );
        }

        let directory_cases = ["tests/foo.rs", "__tests__/bar.ts", "specs/baz.js"];
        for input in directory_cases {
            assert_eq!(
                classify(input),
                SourceKind::Test,
                "expected SourceKind::Test (directory branch) for input {input:?}"
            );
        }
    }

    // T-203: non_vendor_non_test_paths_return_src
    //
    // Perspective: Branch (fallback). Both vendor and test predicates are
    // false; the function must fall through to `SourceKind::Src`.
    //
    // FR: FR-201
    #[test]
    fn non_vendor_non_test_paths_return_src() {
        let cases = ["src/lib.rs", "lib/foo.ts", "app/handlers.rs", "README.md"];
        for input in cases {
            assert_eq!(
                classify(input),
                SourceKind::Src,
                "expected SourceKind::Src for input {input:?}"
            );
        }
    }

    // T-204: vendor_precedence_wins_over_test
    //
    // Perspective: Combination + Hazard. Decision table for the precedence
    // rule (BR-202): when both vendor and test predicates are true, the
    // result must be `SourceKind::Vendor`.
    //
    // | row | vendor match | test match | expected |
    // | --- | ------------ | ---------- | -------- |
    // |  1  | T            | T          | Vendor   |
    //
    // Inputs exercise the row from multiple angles (extension-based test
    // signal nested under vendor; directory-based test signal nested under
    // vendor) to harden the precedence assertion.
    //
    // FR: FR-202
    #[test]
    fn vendor_precedence_wins_over_test() {
        let cases = [
            "node_modules/foo/bar.test.js",
            "vendor/specs/x.rs",
            "dist/__tests__/y.ts",
        ];
        for input in cases {
            assert_eq!(
                classify(input),
                SourceKind::Vendor,
                "expected SourceKind::Vendor (precedence over test) for input {input:?}"
            );
        }
    }

    // T-205: inline_test_module_filename_returns_test
    //
    // Perspective: Equivalence. Rust's `#[cfg(test)] mod tests;` split into a
    // `tests.rs` file is the most common inline-test convention. The classifier
    // must tag it Test so brief and search can exclude it.
    //
    // FR: FR-A1-1
    #[test]
    fn inline_test_module_filename_returns_test() {
        let cases = ["src/tools/tests.rs", "src/storage/tests.rs", "tests.rs"];
        for input in cases {
            assert_eq!(
                classify(input),
                SourceKind::Test,
                "expected SourceKind::Test for Rust inline test module {input:?}"
            );
        }
    }

    // T-206/T-207: tests_stem_is_rust_specific
    //
    // Perspective: Branch (negative). The new `tests.rs` branch is gated on the
    // `rs` extension. Go (`_test.go`) and Python (`tests/` dir or `test_*.py`)
    // must not be caught by a `tests` stem, or the classifier would over-trigger
    // on legitimate non-test module names.
    //
    // FR: FR-A1-2
    #[test]
    fn tests_stem_is_rust_specific() {
        let cases = ["src/foo/tests.go", "src/foo/tests.py"];
        for input in cases {
            assert_eq!(
                classify(input),
                SourceKind::Src,
                "expected SourceKind::Src (tests stem is rs-only) for input {input:?}"
            );
        }
    }

    // T-208: non_tests_rust_filename_returns_src
    //
    // Perspective: Boundary. A `.rs` file whose stem is not exactly `tests`
    // stays Src even when a sibling `tests.rs` exists. Guards the branch against
    // widening beyond the exact stem (e.g. `_tests` plural must not match).
    //
    // FR: FR-A1-1
    #[test]
    fn non_tests_rust_filename_returns_src() {
        let cases = [
            "src/tools.rs",
            "src/tools/format.rs",
            "src/integration_tests.rs",
        ];
        for input in cases {
            assert_eq!(
                classify(input),
                SourceKind::Src,
                "expected SourceKind::Src for non-tests rust file {input:?}"
            );
        }
    }
}
