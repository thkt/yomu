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
mod tests;
