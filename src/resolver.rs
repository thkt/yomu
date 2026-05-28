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
mod tests {
    use std::collections::HashSet;
    use std::fs;

    use tempfile::tempdir;

    use super::*;
    use crate::indexer::chunker::parse_reexports;

    /// Test-only helper: follow re-export chains through barrel files, returning
    /// resolved file paths in traversal order (excluding the start). Kept inside
    /// the test module so the cfg(test) symbol does not leak into the Resolver
    /// API surface (Issue #141 TEST-005 / RC-016).
    fn resolve_reexport_chain(resolver: &Resolver, start_file: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(start_file.to_owned());
        let mut current = start_file.to_owned();

        loop {
            let abs_path = resolver.root.join(&current);
            let content = match fs::read_to_string(&abs_path) {
                Ok(c) => c,
                Err(e) => {
                    tracing::warn!(file = %current, error = %e, "Failed to read file in re-export chain");
                    break;
                }
            };

            let ext = current.rsplit('.').next().unwrap_or("ts");
            let reexports = parse_reexports(&content, ext);

            let mut found_next = false;
            for re in &reexports {
                if let Some(resolved) = resolver.resolve(&re.source, &current) {
                    if visited.contains(&resolved) {
                        continue;
                    }
                    visited.insert(resolved.clone());
                    result.push(resolved.clone());
                    current = resolved;
                    found_next = true;
                    break;
                }
            }
            if !found_next {
                break;
            }
        }
        result
    }

    // T-327: resolve_relative_tsx
    #[test]
    fn resolve_relative_tsx() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("Button.tsx"), "").unwrap();
        fs::write(src.join("App.tsx"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./Button", "src/App.tsx");
        assert_eq!(result, Some("src/Button.tsx".to_owned()));
    }

    // T-328: resolve_index_file_tsx
    #[test]
    fn resolve_index_file_tsx() {
        let tmp = tempdir().unwrap();
        let components = tmp.path().join("src").join("components");
        fs::create_dir_all(&components).unwrap();
        fs::write(components.join("index.tsx"), "").unwrap();
        fs::write(tmp.path().join("src").join("App.tsx"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./components", "src/App.tsx");
        assert_eq!(result, Some("src/components/index.tsx".to_owned()));
    }

    // T-329: resolve_missing_returns_none
    #[test]
    fn resolve_missing_returns_none() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("App.tsx"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./Missing", "src/App.tsx");
        assert_eq!(result, None);
    }

    // T-330: resolve_bare_specifier_returns_none
    #[test]
    fn resolve_bare_specifier_returns_none() {
        let tmp = tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("react", "src/App.tsx");
        assert_eq!(result, None);
    }

    // T-331: load_aliases_from_tsconfig
    #[test]
    fn load_aliases_from_tsconfig() {
        let tmp = tempdir().unwrap();
        let tsconfig = r#"{
            "compilerOptions": {
                "paths": {
                    "@/*": ["src/*"]
                }
            }
        }"#;
        fs::write(tmp.path().join("tsconfig.json"), tsconfig).unwrap();

        let aliases = load_aliases(tmp.path());
        assert_eq!(
            aliases,
            vec![PathAlias {
                prefix: "@/".to_owned(),
                target: "src/".to_owned(),
            }]
        );
    }

    // T-347: load_aliases_with_baseurl
    // tsconfig `paths` targets are relative to `baseUrl`; `{ baseUrl: "src",
    // paths: { "@/*": ["*"] } }` must yield the same `src/` target as the
    // root-relative `["src/*"]` form.
    #[test]
    fn load_aliases_with_baseurl() {
        let tmp = tempdir().unwrap();
        let tsconfig = r#"{
            "compilerOptions": {
                "baseUrl": "src",
                "paths": {
                    "@/*": ["*"]
                }
            }
        }"#;
        fs::write(tmp.path().join("tsconfig.json"), tsconfig).unwrap();

        let aliases = load_aliases(tmp.path());
        assert_eq!(
            aliases,
            vec![PathAlias {
                prefix: "@/".to_owned(),
                target: "src/".to_owned(),
            }]
        );
    }

    // T-348: compose_alias_target_variants
    #[test]
    fn compose_alias_target_variants() {
        // baseUrl pushed, empty path target
        assert_eq!(compose_alias_target("src", ""), "src/");
        // leading "./" on baseUrl is normalized away
        assert_eq!(compose_alias_target("./src", ""), "src/");
        // baseUrl "." is skipped; the path target stands alone
        assert_eq!(compose_alias_target(".", "src/"), "src/");
        // both empty → repo root (empty prefix)
        assert_eq!(compose_alias_target(".", ""), "");
        // filename-prefix wildcard (e.g. `generated/lib-*`): the target tail is
        // preserved verbatim with no forced trailing "/"
        assert_eq!(
            compose_alias_target(".", "generated/lib-"),
            "generated/lib-"
        );
        assert_eq!(
            compose_alias_target("src", "generated/lib-"),
            "src/generated/lib-"
        );
    }

    // T-349: load_aliases_no_compiler_options
    // A tsconfig without `compilerOptions` yields no aliases (no panic).
    #[test]
    fn load_aliases_no_compiler_options() {
        let tmp = tempdir().unwrap();
        fs::write(tmp.path().join("tsconfig.json"), "{}").unwrap();
        assert!(load_aliases(tmp.path()).is_empty());
    }

    // T-332: load_aliases_no_tsconfig
    #[test]
    fn load_aliases_no_tsconfig() {
        let tmp = tempdir().unwrap();

        let aliases = load_aliases(tmp.path());
        assert!(aliases.is_empty());
    }

    // T-333: resolve_alias_path
    #[test]
    fn resolve_alias_path() {
        let tmp = tempdir().unwrap();
        let lib = tmp.path().join("src").join("lib");
        fs::create_dir_all(&lib).unwrap();
        fs::write(lib.join("auth.ts"), "").unwrap();

        let tsconfig = r#"{
            "compilerOptions": {
                "paths": {
                    "@/*": ["src/*"]
                }
            }
        }"#;
        fs::write(tmp.path().join("tsconfig.json"), tsconfig).unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("@/lib/auth", "src/App.tsx");
        assert_eq!(result, Some("src/lib/auth.ts".to_owned()));
    }

    // T-334: resolve_relative_ts
    #[test]
    fn resolve_relative_ts() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("utils.ts"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./utils", "src/App.tsx");
        assert_eq!(result, Some("src/utils.ts".to_owned()));
    }

    // T-335: resolve_parent_directory
    #[test]
    fn resolve_parent_directory() {
        let tmp = tempdir().unwrap();
        let utils = tmp.path().join("src").join("utils");
        let components = tmp.path().join("src").join("components");
        fs::create_dir_all(&utils).unwrap();
        fs::create_dir_all(&components).unwrap();
        fs::write(utils.join("format.ts"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("../utils/format", "src/components/Button.tsx");
        assert_eq!(result, Some("src/utils/format.ts".to_owned()));
    }

    // T-336: resolve_css_file
    #[test]
    fn resolve_css_file() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("styles.css"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./styles.css", "src/App.tsx");
        assert_eq!(result, Some("src/styles.css".to_owned()));
    }

    // T-337: resolve_prefers_tsx_over_ts
    #[test]
    fn resolve_prefers_tsx_over_ts() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("Button.tsx"), "").unwrap();
        fs::write(src.join("Button.ts"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./Button", "src/App.tsx");
        assert_eq!(result, Some("src/Button.tsx".to_owned()));
    }

    // T-338: resolve_index_file_ts
    #[test]
    fn resolve_index_file_ts() {
        let tmp = tempdir().unwrap();
        let hooks = tmp.path().join("src").join("hooks");
        fs::create_dir_all(&hooks).unwrap();
        fs::write(hooks.join("index.ts"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./hooks", "src/App.tsx");
        assert_eq!(result, Some("src/hooks/index.ts".to_owned()));
    }

    // T-339: load_aliases_multiple
    #[test]
    fn load_aliases_multiple() {
        let tmp = tempdir().unwrap();
        let tsconfig = r#"{
            "compilerOptions": {
                "paths": {
                    "@/*": ["src/*"],
                    "~/*": ["lib/*"]
                }
            }
        }"#;
        fs::write(tmp.path().join("tsconfig.json"), tsconfig).unwrap();

        let aliases = load_aliases(tmp.path());
        assert_eq!(aliases.len(), 2);
        assert!(aliases.contains(&PathAlias {
            prefix: "@/".to_owned(),
            target: "src/".to_owned(),
        }));
        assert!(aliases.contains(&PathAlias {
            prefix: "~/".to_owned(),
            target: "lib/".to_owned(),
        }));
    }

    // T-340: resolve_non_frontend_extension_returns_none
    #[test]
    fn resolve_non_frontend_extension_returns_none() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("data.json"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./data", "src/App.tsx");
        assert_eq!(result, None);
    }

    // T-341: resolve_scoped_package_returns_none
    #[test]
    fn resolve_scoped_package_returns_none() {
        let tmp = tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("@tanstack/react-query", "src/App.tsx");
        assert_eq!(result, None);
    }

    // T-342: resolve_tilde_alias
    #[test]
    fn resolve_tilde_alias() {
        let tmp = tempdir().unwrap();
        let lib = tmp.path().join("lib").join("core");
        fs::create_dir_all(&lib).unwrap();
        fs::write(lib.join("engine.ts"), "").unwrap();

        let tsconfig = r#"{
            "compilerOptions": {
                "paths": {
                    "~/*": ["lib/*"]
                }
            }
        }"#;
        fs::write(tmp.path().join("tsconfig.json"), tsconfig).unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("~/core/engine", "src/App.tsx");
        assert_eq!(result, Some("lib/core/engine.ts".to_owned()));
    }

    // T-343: resolve_reexport_chain_circular
    #[test]
    fn resolve_reexport_chain_circular() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("a.ts"), "export { X } from './b';").unwrap();
        fs::write(src.join("b.ts"), "export { X } from './a';").unwrap();

        let resolver = Resolver::new(tmp.path());
        let chain = resolve_reexport_chain(&resolver, "src/a.ts");
        assert_eq!(chain, vec!["src/b.ts".to_owned()]);
    }

    // T-344: resolve_reexport_chain_linear
    #[test]
    fn resolve_reexport_chain_linear() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("a.ts"), "export { X } from './b';").unwrap();
        fs::write(src.join("b.ts"), "export { X } from './c';").unwrap();
        fs::write(src.join("c.ts"), "export const X = 1;").unwrap();

        let resolver = Resolver::new(tmp.path());
        let chain = resolve_reexport_chain(&resolver, "src/a.ts");
        assert_eq!(chain, vec!["src/b.ts".to_owned(), "src/c.ts".to_owned()]);
    }

    // T-345: resolve_reexport_chain_no_reexports
    #[test]
    fn resolve_reexport_chain_no_reexports() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("a.ts"), "export const X = 1;").unwrap();

        let resolver = Resolver::new(tmp.path());
        let chain = resolve_reexport_chain(&resolver, "src/a.ts");
        assert!(chain.is_empty());
    }

    // T-346: resolve_html_file
    #[test]
    fn resolve_html_file() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("template.html"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./template.html", "src/App.tsx");
        assert_eq!(result, Some("src/template.html".to_owned()));
    }
}
