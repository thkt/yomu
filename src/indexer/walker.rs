use ignore::WalkBuilder;
use std::path::{Path, PathBuf};

const SUPPORTED_EXTENSIONS: &[&str] = &["ts", "tsx", "js", "jsx", "mjs", "css", "html", "rs", "md"];

pub fn walk_source_files(root: &Path) -> Vec<PathBuf> {
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
        .map(|entry| entry.into_path())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

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
        std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(dir)
            .output()
            .unwrap();
        fs::write(dir.join(".gitignore"), "node_modules/\n").unwrap();
    }

    // T-357: walk_collects_source_files
    #[test]
    fn walk_collects_source_files() {
        let tmp = tempfile::tempdir().unwrap();
        setup_project_with_gitignore(tmp.path());

        let files = walk_source_files(tmp.path());
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
        let tmp = tempfile::tempdir().unwrap();
        setup_project_with_gitignore(tmp.path());

        let files = walk_source_files(tmp.path());
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
        let tmp = tempfile::tempdir().unwrap();
        setup_project_with_gitignore(tmp.path());

        let plugin_file = tmp.path().join("plugins/cache/big-plugin/README.md");
        fs::create_dir_all(plugin_file.parent().unwrap()).unwrap();
        fs::write(&plugin_file, "# Plugin").unwrap();

        fs::write(tmp.path().join(".gitignore"), "node_modules/\nplugins/*\n").unwrap();

        let files = walk_source_files(tmp.path());
        for f in &files {
            let s = f.to_str().unwrap();
            assert!(!s.contains("plugins"), "included gitignored plugins: {s}");
        }
    }
}
