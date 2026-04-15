use std::path::{Path, PathBuf};

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
    use tempfile::tempdir;

    use super::*;
    use std::fs;

    // T-312: detect_root_finds_git_dir
    #[test]
    fn detect_root_finds_git_dir() {
        let tmp = tempdir().unwrap();
        let git_dir = tmp.path().join(".git");
        fs::create_dir(&git_dir).unwrap();

        let sub = tmp.path().join("src").join("components");
        fs::create_dir_all(&sub).unwrap();

        let root = detect_root(&sub);
        assert_eq!(root, tmp.path());
    }

    // T-313: detect_root_prefers_yomu_over_git
    #[test]
    fn detect_root_prefers_yomu_over_git() {
        let tmp = tempdir().unwrap();
        fs::create_dir(tmp.path().join(".git")).unwrap();
        fs::create_dir(tmp.path().join(".yomu")).unwrap();

        let root = detect_root(tmp.path());
        assert_eq!(root, tmp.path());
    }

    // T-314: detect_root_falls_back_to_start
    #[test]
    fn detect_root_falls_back_to_start() {
        let tmp = tempdir().unwrap();
        let root = detect_root(tmp.path());
        assert_eq!(root, tmp.path());
    }
}
