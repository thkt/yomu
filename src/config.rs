//! Project root detection by walking up to find `.yomu/` or `.git/` markers.

use std::path::{Path, PathBuf};

/// Walk up from `start` looking for project root markers.
///
/// Checks `.yomu/` first (project-specific), then `.git/`.
/// Returns `start` unchanged if neither marker is found.
/// Expects `start` to be a directory (e.g. from `std::env::current_dir()`).
pub fn detect_root(start: &Path) -> PathBuf {
    for marker in &[".yomu", ".git"] {
        let mut current = Some(start);
        while let Some(dir) = current {
            if dir.join(marker).is_dir() {
                return dir.to_path_buf();
            }
            current = dir.parent();
        }
    }
    start.to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn detect_root_finds_git_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let git_dir = tmp.path().join(".git");
        fs::create_dir(&git_dir).unwrap();

        let sub = tmp.path().join("src").join("components");
        fs::create_dir_all(&sub).unwrap();

        let root = detect_root(&sub);
        assert_eq!(root, tmp.path());
    }

    #[test]
    fn detect_root_prefers_yomu_over_git() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir(tmp.path().join(".git")).unwrap();
        fs::create_dir(tmp.path().join(".yomu")).unwrap();

        let root = detect_root(tmp.path());
        assert_eq!(root, tmp.path());
    }

    #[test]
    fn detect_root_falls_back_to_start() {
        let tmp = tempfile::tempdir().unwrap();
        let root = detect_root(tmp.path());
        assert_eq!(root, tmp.path());
    }
}
