use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::fs_optional;

const PROBE_EXTENSIONS: &[&str] = &["tsx", "ts", "jsx", "js"];
const SUPPORTED_EXTENSIONS: &[&str] = &["tsx", "ts", "jsx", "js", "css", "html"];
const INDEX_FILES: &[&str] = &["index.tsx", "index.ts", "index.jsx", "index.js"];

#[derive(Debug, Clone, PartialEq)]
pub struct PathAlias {
    pub prefix: String,
    pub target: String,
}

pub struct Resolver {
    root: PathBuf,
    canonical_root: Option<PathBuf>,
    aliases: Vec<PathAlias>,
}

impl Resolver {
    fn apply_alias(&self, source: &str) -> Option<String> {
        for alias in &self.aliases {
            if let Some(rest) = source.strip_prefix(&alias.prefix) {
                return Some(format!("./{}{}", alias.target, rest));
            }
        }
        None
    }

    fn to_relative(&self, abs: &Path) -> Option<String> {
        to_relative_path(abs, &self.root, self.canonical_root.as_deref())
    }

    fn probe_path(&self, candidate: &Path) -> Option<String> {
        if let Some(ext) = candidate.extension().and_then(|e| e.to_str())
            && SUPPORTED_EXTENSIONS.contains(&ext)
            && let Some(rel) = self.to_relative(candidate)
        {
            return Some(rel);
        }

        for ext in PROBE_EXTENSIONS {
            let with_ext = candidate.with_extension(ext);
            if with_ext.exists() {
                return self.to_relative(&with_ext);
            }
        }

        if candidate.is_dir() {
            for index in INDEX_FILES {
                let index_path = candidate.join(index);
                if index_path.exists() {
                    return self.to_relative(&index_path);
                }
            }
        }

        None
    }

    pub fn new(root: &Path) -> Self {
        let canonical_root = fs_optional::canonicalize_optional(root);
        Self {
            root: root.to_path_buf(),
            canonical_root,
            aliases: load_aliases(root),
        }
    }

    /// Returns None for bare specifiers (npm packages) or unresolvable paths.
    pub fn resolve(&self, source: &str, from_file: &str) -> Option<String> {
        let resolved_source = self.apply_alias(source);
        let source = resolved_source.as_deref().unwrap_or(source);

        if !source.starts_with('.') && !source.starts_with('/') && resolved_source.is_none() {
            return None;
        }

        let base_dir = if resolved_source.is_some() {
            self.root.clone()
        } else {
            let from_abs = self.root.join(from_file);
            from_abs.parent()?.to_path_buf()
        };

        let candidate = base_dir.join(source);
        self.probe_path(&candidate)
    }
}

pub trait Resolve {
    fn resolve(&self, source: &str, from_file: &str) -> Option<String>;
    fn resolve_mod_decl(&self, _name: &str, _from_file: &str) -> Option<String> {
        None
    }
}

impl Resolve for Resolver {
    fn resolve(&self, source: &str, from_file: &str) -> Option<String> {
        self.resolve(source, from_file)
    }
}

/// Strip canonical root prefix from an already-canonical path.
pub fn strip_canonical_prefix(abs: &Path, canonical_root: Option<&Path>) -> Option<String> {
    let root = canonical_root?;
    abs.strip_prefix(root)
        .ok()
        .map(|p| p.to_string_lossy().to_string())
}

/// Convert absolute path to project-relative path. Returns None if path escapes root.
pub fn to_relative_path(abs: &Path, root: &Path, canonical_root: Option<&Path>) -> Option<String> {
    let abs = match abs.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            tracing::warn!(path = %abs.display(), error = %e, "canonicalize failed for existing path");
            return None;
        }
    };
    strip_canonical_prefix(&abs, canonical_root).or_else(|| {
        tracing::warn!(path = %abs.display(), root = %root.display(), "Resolved path escapes project root or canonical root unavailable");
        None
    })
}

pub fn load_aliases(root: &Path) -> Vec<PathAlias> {
    let tsconfig_path = root.join("tsconfig.json");
    let content = match fs::read_to_string(&tsconfig_path) {
        Ok(c) => c,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return vec![],
        Err(e) => {
            tracing::warn!(path = %tsconfig_path.display(), error = %e, "Failed to read tsconfig.json");
            return vec![];
        }
    };

    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(path = %tsconfig_path.display(), error = %e, "Failed to parse tsconfig.json");
            return vec![];
        }
    };

    let compiler_options = match json.get("compilerOptions") {
        Some(co) => co,
        None => return vec![],
    };

    // TypeScript resolves `paths` targets relative to `baseUrl` (which defaults
    // to "." when omitted). Without folding `baseUrl` in, an alias like
    // `{ "baseUrl": "src", "paths": { "@/*": ["*"] } }` resolves to the repo root
    // instead of `src/`, dropping the target from the forward closure.
    let base_url = compiler_options
        .get("baseUrl")
        .and_then(|b| b.as_str())
        .unwrap_or(".");

    let paths = match compiler_options.get("paths").and_then(|p| p.as_object()) {
        Some(p) => p,
        None => return vec![],
    };

    paths
        .iter()
        .filter_map(|(key, value)| {
            // key: "@/*", value: ["*"] (relative to baseUrl) or ["src/*"]
            let prefix = key.strip_suffix('*')?;
            let target_arr = value.as_array()?;
            let target_str = target_arr.first()?.as_str()?;
            let raw_target = target_str.strip_suffix('*')?;
            Some(PathAlias {
                prefix: prefix.to_owned(),
                target: compose_alias_target(base_url, raw_target),
            })
        })
        .collect()
}

/// Prefix a tsconfig `paths` target with `base_url` (TypeScript resolves `paths`
/// relative to `baseUrl`, which defaults to "."). `path_target` is the portion
/// of a `paths` value before the `*` wildcard and is appended verbatim so its
/// tail is preserved: `"src/"` for a path-segment wildcard (`src/*`),
/// `"generated/lib-"` for a filename-prefix wildcard (`generated/lib-*`).
/// Re-normalizing the tail would break the latter.
fn compose_alias_target(base_url: &str, path_target: &str) -> String {
    let base = base_url.trim_start_matches("./").trim_end_matches('/');
    let prefix = if base.is_empty() || base == "." {
        String::new()
    } else {
        format!("{base}/")
    };
    format!("{prefix}{path_target}")
}

#[cfg(test)]
mod tests;
