//! Path-based classification of source files into `"vendor"` / `"test"` / `"src"`.
//!
//! Precedence per FR-202: vendor > test > src.

use std::path::Path;

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

/// Classify a relative path into one of `"vendor"`, `"test"`, or `"src"`.
///
/// Precedence: vendor > test > src (FR-202).
pub fn classify(rel_path: &str) -> &'static str {
    let components: Vec<&str> = Path::new(rel_path)
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    let ancestor_count = components.len().saturating_sub(1);
    let ancestors = &components[..ancestor_count];

    if ancestors.iter().any(|c| VENDOR_DIRS.contains(c)) {
        return "vendor";
    }
    if ancestors.iter().any(|c| TEST_DIRS.contains(c)) {
        return "test";
    }
    if components.last().is_some_and(|f| is_test_filename(f)) {
        return "test";
    }
    "src"
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
                "vendor",
                "expected \"vendor\" for input {input:?}"
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
    // | a   | T               | F         | "test"   |
    // | b   | F               | T         | "test"   |
    //
    // FR: FR-201, FR-204
    #[test]
    fn test_extension_or_test_directory_returns_test() {
        let extension_cases = ["src/foo.test.ts", "src/bar.spec.tsx", "src/baz_test.rs"];
        for input in extension_cases {
            assert_eq!(
                classify(input),
                "test",
                "expected \"test\" (extension branch) for input {input:?}"
            );
        }

        let directory_cases = ["tests/foo.rs", "__tests__/bar.ts", "specs/baz.js"];
        for input in directory_cases {
            assert_eq!(
                classify(input),
                "test",
                "expected \"test\" (directory branch) for input {input:?}"
            );
        }
    }

    // T-203: non_vendor_non_test_paths_return_src
    //
    // Perspective: Branch (fallback). Both vendor and test predicates are
    // false; the function must fall through to "src".
    //
    // FR: FR-201
    #[test]
    fn non_vendor_non_test_paths_return_src() {
        let cases = ["src/lib.rs", "lib/foo.ts", "app/handlers.rs", "README.md"];
        for input in cases {
            assert_eq!(
                classify(input),
                "src",
                "expected \"src\" for input {input:?}"
            );
        }
    }

    // T-204: vendor_precedence_wins_over_test
    //
    // Perspective: Combination + Hazard. Decision table for the precedence
    // rule (BR-202): when both vendor and test predicates are true, the
    // result must be "vendor".
    //
    // | row | vendor match | test match | expected |
    // | --- | ------------ | ---------- | -------- |
    // |  1  | T            | T          | "vendor" |
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
                "vendor",
                "expected \"vendor\" (precedence over test) for input {input:?}"
            );
        }
    }
}
