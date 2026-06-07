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

/// tsconfig path-resolution inputs: `paths` aliases plus the explicit
/// `baseUrl` (None when tsconfig omits it — bare specifiers then stay npm
/// packages; `paths` target composition separately defaults to ".").
#[derive(Debug, Default, PartialEq)]
pub struct TsPathConfig {
    pub aliases: Vec<PathAlias>,
    pub base_url: Option<String>,
}

pub struct Resolver {
    root: PathBuf,
    canonical_root: Option<PathBuf>,
    aliases: Vec<PathAlias>,
    base_url: Option<String>,
}

impl Resolver {
    /// All alias rewrites whose prefix matches `source`, in declaration order.
    /// Multiple candidates arise from multi-target `paths` values (#233) and
    /// from distinct alias keys sharing a matching prefix.
    fn alias_candidates(&self, source: &str) -> Vec<String> {
        self.aliases
            .iter()
            .filter_map(|alias| {
                source
                    .strip_prefix(&alias.prefix)
                    .map(|rest| format!("./{}{}", alias.target, rest))
            })
            .collect()
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
        let config = load_path_config(root);
        Self {
            root: root.to_path_buf(),
            canonical_root,
            aliases: config.aliases,
            base_url: config.base_url,
        }
    }

    /// Returns None for bare specifiers (npm packages) or unresolvable paths.
    pub fn resolve(&self, source: &str, from_file: &str) -> Option<String> {
        // tsconfig `paths` first: probe each matching alias target in
        // declaration order, first existing file wins (TypeScript tries the
        // substitutions in order). A matched prefix whose targets all miss
        // resolves to None without falling through to baseUrl below.
        let alias_candidates = self.alias_candidates(source);
        if !alias_candidates.is_empty() {
            return alias_candidates
                .iter()
                .find_map(|candidate| self.probe_path(&self.root.join(candidate)));
        }

        if source.starts_with('.') || source.starts_with('/') {
            let from_abs = self.root.join(from_file);
            let base_dir = from_abs.parent()?;
            return self.probe_path(&base_dir.join(source));
        }

        // Bare specifier: with an explicit `baseUrl`, TypeScript resolves
        // non-relative imports against it before node_modules (#233). No
        // explicit baseUrl (or no file there) → npm package, not ours.
        let base_url = self.base_url.as_deref()?;
        self.probe_path(&self.root.join(base_url).join(source))
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

pub fn load_path_config(root: &Path) -> TsPathConfig {
    let tsconfig_path = root.join("tsconfig.json");
    let content = match fs::read_to_string(&tsconfig_path) {
        Ok(c) => c,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return TsPathConfig::default(),
        Err(e) => {
            tracing::warn!(path = %tsconfig_path.display(), error = %e, "Failed to read tsconfig.json");
            return TsPathConfig::default();
        }
    };

    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(path = %tsconfig_path.display(), error = %e, "Failed to parse tsconfig.json");
            return TsPathConfig::default();
        }
    };

    let compiler_options = match json.get("compilerOptions") {
        Some(co) => co,
        None => return TsPathConfig::default(),
    };

    let explicit_base_url = compiler_options.get("baseUrl").and_then(|b| b.as_str());

    // An absolute baseUrl either escapes the project root (never indexed) or
    // would fold into a bogus root-relative path via compose_alias_target
    // (`/abs` → `.//abs`). Disable tsconfig path resolution entirely (#233).
    if let Some(base) = explicit_base_url
        && base.starts_with('/')
    {
        tracing::warn!(base_url = base, path = %tsconfig_path.display(), "absolute baseUrl is unsupported; ignoring tsconfig paths/baseUrl");
        return TsPathConfig::default();
    }

    let base_url_owned = explicit_base_url.map(str::to_owned);

    let Some(paths) = compiler_options.get("paths").and_then(|p| p.as_object()) else {
        return TsPathConfig {
            aliases: Vec::new(),
            base_url: base_url_owned,
        };
    };

    // TypeScript resolves `paths` targets relative to `baseUrl` (which defaults
    // to "." when omitted). Without folding `baseUrl` in, an alias like
    // `{ "baseUrl": "src", "paths": { "@/*": ["*"] } }` resolves to the repo root
    // instead of `src/`, dropping the target from the forward closure.
    let base_url = explicit_base_url.unwrap_or(".");

    let aliases = paths
        .iter()
        .filter_map(|(key, value)| {
            // key: "@/*", value: ["*"] (relative to baseUrl) or ["src/*", "lib/*"]
            Some((key.strip_suffix('*')?, value.as_array()?))
        })
        .flat_map(|(prefix, targets)| {
            // Keep every wildcard target: TypeScript probes the substitutions
            // in order until one exists (#233).
            targets.iter().filter_map(move |target| {
                let raw_target = target.as_str()?.strip_suffix('*')?;
                Some(PathAlias {
                    prefix: prefix.to_owned(),
                    target: compose_alias_target(base_url, raw_target),
                })
            })
        })
        .collect();

    TsPathConfig {
        aliases,
        base_url: base_url_owned,
    }
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
