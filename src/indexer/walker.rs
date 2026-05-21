use ignore::{DirEntry, WalkBuilder};
use std::path::{Path, PathBuf};

use super::source_kind;
use crate::storage::SourceKind;

const SUPPORTED_EXTENSIONS: &[&str] = &["ts", "tsx", "js", "jsx", "mjs", "css", "html", "rs", "md"];

/// Walks source files under `root`. When `exclude_vendor` is `true`, any
/// entry whose path classifies as [`SourceKind::Vendor`] per
/// [`source_kind::classify`] is filtered out — this is additive to the
/// walker's existing `.gitignore` exclusion (ADR-0069 pillar 1, walker-level
/// vendor filter).
pub fn walk_source_files(root: &Path, exclude_vendor: bool) -> Vec<PathBuf> {
    WalkBuilder::new(root)
        .hidden(true)
        .follow_links(false)
        .require_git(false)
        .git_global(false)
        .git_exclude(false)
        .build()
        .filter_map(|entry| match entry {
            Ok(e) => Some(e),
            Err(e) => {
                tracing::warn!(error = %e, "Walk entry error, skipping");
                None
            }
        })
        .filter(|entry| {
            entry.file_type().is_some_and(|ft| ft.is_file())
                && entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| SUPPORTED_EXTENSIONS.contains(&ext))
        })
        .filter(|entry| {
            if !exclude_vendor {
                return true;
            }
            let rel = entry.path().strip_prefix(root).unwrap_or(entry.path());
            let rel_str = rel.to_string_lossy();
            source_kind::classify(&rel_str) != SourceKind::Vendor
        })
        .map(DirEntry::into_path)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::process::Command;

    use tempfile::tempdir;

    use super::*;

    fn setup_project(dir: &Path) {
        for name in [
            "src/App.tsx",
            "src/Button.ts",
            "src/utils.js",
            "src/Card.jsx",
            "src/entry.mjs",
            "src/styles.css",
            "public/index.html",
            "src/lib.rs",
            "docs/guide.md",
        ] {
            let path = dir.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, "// content").unwrap();
        }

        for name in ["package.json", "README.md", "tsconfig.json"] {
            fs::write(dir.join(name), "{}").unwrap();
        }

        let nm = dir.join("node_modules/react/index.js");
        fs::create_dir_all(nm.parent().unwrap()).unwrap();
        fs::write(&nm, "// react").unwrap();

        let git = dir.join(".git/config");
        fs::create_dir_all(git.parent().unwrap()).unwrap();
        fs::write(&git, "").unwrap();

        let yomu = dir.join(".yomu/index.db");
        fs::create_dir_all(yomu.parent().unwrap()).unwrap();
        fs::write(&yomu, "").unwrap();

        let claude = dir.join(".claude/worktrees/agent-abc/src/App.tsx");
        fs::create_dir_all(claude.parent().unwrap()).unwrap();
        fs::write(&claude, "// worktree copy").unwrap();
    }

    fn setup_project_with_gitignore(dir: &Path) {
        setup_project(dir);
        Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir)
            .output()
            .unwrap();
        fs::write(dir.join(".gitignore"), "node_modules/\n").unwrap();
    }

    // T-357: walk_collects_source_files
    #[test]
    fn walk_collects_source_files() {
        let tmp = tempdir().unwrap();
        setup_project_with_gitignore(tmp.path());

        let files = walk_source_files(tmp.path(), false);
        let mut names: Vec<&str> = files
            .iter()
            .map(|p| p.strip_prefix(tmp.path()).unwrap().to_str().unwrap())
            .collect();
        names.sort();

        assert_eq!(
            names,
            [
                "README.md",
                "docs/guide.md",
                "public/index.html",
                "src/App.tsx",
                "src/Button.ts",
                "src/Card.jsx",
                "src/entry.mjs",
                "src/lib.rs",
                "src/styles.css",
                "src/utils.js",
            ]
        );
    }

    // T-358: walk_excludes_hidden_and_gitignored_dirs
    #[test]
    fn walk_excludes_hidden_and_gitignored_dirs() {
        let tmp = tempdir().unwrap();
        setup_project_with_gitignore(tmp.path());

        let files = walk_source_files(tmp.path(), false);
        for f in &files {
            let s = f.to_str().unwrap();
            assert!(!s.contains(".git"), "included .git: {s}");
            assert!(!s.contains(".yomu"), "included .yomu: {s}");
            assert!(!s.contains(".claude"), "included .claude: {s}");
            assert!(!s.contains("node_modules"), "included node_modules: {s}");
        }
    }

    // T-359: walk_respects_custom_gitignore_patterns
    #[test]
    fn walk_respects_custom_gitignore_patterns() {
        let tmp = tempdir().unwrap();
        setup_project_with_gitignore(tmp.path());

        let plugin_file = tmp.path().join("plugins/cache/big-plugin/README.md");
        fs::create_dir_all(plugin_file.parent().unwrap()).unwrap();
        fs::write(&plugin_file, "# Plugin").unwrap();

        fs::write(tmp.path().join(".gitignore"), "node_modules/\nplugins/*\n").unwrap();

        let files = walk_source_files(tmp.path(), false);
        for f in &files {
            let s = f.to_str().unwrap();
            assert!(!s.contains("plugins"), "included gitignored plugins: {s}");
        }
    }

    // Minimal fixture for vendor-filter tests (T-316 / T-317). No `.gitignore`
    // on `vendor/` so that the only thing excluding it is the new
    // `exclude_vendor` flag — not the existing `.gitignore` walker behavior.
    fn setup_vendor_project(dir: &Path) {
        for name in ["src/foo.ts", "vendor/lib.ts"] {
            let path = dir.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, "export function f() { return 1; }").unwrap();
        }
        Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir)
            .output()
            .unwrap();
    }

    // T-316: walk_excludes_vendor_when_flag_true
    //
    // Perspective: Branch (FR-314b true-branch) + Equivalence. The
    // `exclude_vendor=true` branch filters out paths whose
    // `source_kind::classify` returns `SourceKind::Vendor`. `vendor/lib.ts`
    // is the representative for the vendor equivalence class; `src/foo.ts`
    // is the representative for the non-vendor equivalence class that must
    // remain.
    //
    // FR: FR-314b
    #[test]
    fn walk_excludes_vendor_when_flag_true() {
        let tmp = tempdir().unwrap();
        setup_vendor_project(tmp.path());

        let files = walk_source_files(tmp.path(), true);
        let mut names: Vec<&str> = files
            .iter()
            .map(|p| p.strip_prefix(tmp.path()).unwrap().to_str().unwrap())
            .collect();
        names.sort();

        assert!(
            names.contains(&"src/foo.ts"),
            "src/foo.ts must be retained when exclude_vendor=true, got: {names:?}"
        );
        assert!(
            !names.contains(&"vendor/lib.ts"),
            "vendor/lib.ts must be filtered when exclude_vendor=true, got: {names:?}"
        );
    }

    // T-317: walk_includes_vendor_when_flag_false
    //
    // Perspective: Branch (FR-314c false-branch / regression). When
    // `exclude_vendor=false` the walker behavior matches the PR#2 baseline:
    // no additional filter beyond `.gitignore`. Both `src/foo.ts` and
    // `vendor/lib.ts` must be returned because no `.gitignore` excludes
    // `vendor/` in this fixture.
    //
    // FR: FR-314c
    #[test]
    fn walk_includes_vendor_when_flag_false() {
        let tmp = tempdir().unwrap();
        setup_vendor_project(tmp.path());

        let files = walk_source_files(tmp.path(), false);
        let mut names: Vec<&str> = files
            .iter()
            .map(|p| p.strip_prefix(tmp.path()).unwrap().to_str().unwrap())
            .collect();
        names.sort();

        assert!(
            names.contains(&"src/foo.ts"),
            "src/foo.ts must be present when exclude_vendor=false, got: {names:?}"
        );
        assert!(
            names.contains(&"vendor/lib.ts"),
            "vendor/lib.ts must be present when exclude_vendor=false (PR#2 baseline), got: {names:?}"
        );
    }
}
