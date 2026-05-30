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
mod tests;
