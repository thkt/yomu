use std::path::{Path, PathBuf};
use walkdir::WalkDir;

const FRONTEND_EXTENSIONS: &[&str] = &[
    "ts", "tsx", "js", "jsx", "mjs", "css", "html",
];

const EXCLUDED_DIRS: &[&str] = &[
    "node_modules", ".git", ".yomu", "dist", "build", ".next", "target",
    "storybook-static", "coverage", "out", ".turbo", ".cache",
];

/// Walk `root` recursively, collecting files with frontend extensions.
pub fn walk_frontend_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            if entry.file_type().is_dir() {
                let name = entry.file_name().to_str().unwrap_or("");
                return !EXCLUDED_DIRS.contains(&name);
            }
            true
        })
        .filter_map(|entry| match entry {
            Ok(e) => Some(e),
            Err(e) => {
                tracing::warn!(error = %e, "Walk entry error, skipping");
                None
            }
        })
        .filter(|entry| entry.file_type().is_file())
        .filter(|entry| {
            entry
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| FRONTEND_EXTENSIONS.contains(&ext))
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
    }

    #[test]
    fn walk_collects_frontend_files() {
        let tmp = tempfile::tempdir().unwrap();
        setup_project(tmp.path());

        let files = walk_frontend_files(tmp.path());
        let names: Vec<&str> = files
            .iter()
            .map(|p| p.strip_prefix(tmp.path()).unwrap().to_str().unwrap())
            .collect();

        assert!(names.contains(&"src/App.tsx"), "missing tsx: {names:?}");
        assert!(names.contains(&"src/Button.ts"), "missing ts: {names:?}");
        assert!(names.contains(&"src/utils.js"), "missing js: {names:?}");
        assert!(names.contains(&"src/Card.jsx"), "missing jsx: {names:?}");
        assert!(names.contains(&"src/entry.mjs"), "missing mjs: {names:?}");
        assert!(names.contains(&"src/styles.css"), "missing css: {names:?}");
        assert!(
            names.contains(&"public/index.html"),
            "missing html: {names:?}"
        );
        assert_eq!(files.len(), 7, "expected 7 frontend files, got: {names:?}");
    }

    #[test]
    fn walk_excludes_node_modules_and_dot_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        setup_project(tmp.path());

        let files = walk_frontend_files(tmp.path());
        for f in &files {
            let s = f.to_str().unwrap();
            assert!(!s.contains("node_modules"), "included node_modules: {s}");
            assert!(!s.contains(".git"), "included .git: {s}");
            assert!(!s.contains(".yomu"), "included .yomu: {s}");
        }
    }
}
