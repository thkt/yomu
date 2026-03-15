use std::path::{Path, PathBuf};

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
    aliases: Vec<PathAlias>,
}

impl Resolver {
    pub fn new(root: &Path) -> Self {
        Self {
            root: root.to_path_buf(),
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

    fn apply_alias(&self, source: &str) -> Option<String> {
        for alias in &self.aliases {
            if let Some(rest) = source.strip_prefix(&alias.prefix) {
                return Some(format!("./{}{}", alias.target, rest));
            }
        }
        None
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

    /// Returns None if the path escapes the project root.
    fn to_relative(&self, abs: &Path) -> Option<String> {
        let abs = match abs.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(path = %abs.display(), error = %e, "canonicalize failed for existing path");
                return None;
            }
        };
        let root = match self.root.canonicalize() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(path = %self.root.display(), error = %e, "canonicalize failed for project root");
                return None;
            }
        };
        match abs.strip_prefix(&root) {
            Ok(p) => Some(p.to_string_lossy().to_string()),
            Err(_) => {
                tracing::warn!(path = %abs.display(), root = %root.display(), "Resolved path escapes project root");
                None
            }
        }
    }

    /// Follow re-export chain through barrel files. Returns files excluding start.
    #[cfg(test)]
    pub fn resolve_reexport_chain(&self, start_file: &str) -> Vec<String> {
        use crate::indexer::chunker::parse_reexports;
        use std::collections::HashSet;

        let mut result = Vec::new();
        let mut visited = HashSet::new();
        visited.insert(start_file.to_string());
        let mut current = start_file.to_string();

        loop {
            let abs_path = self.root.join(&current);
            let content = match std::fs::read_to_string(&abs_path) {
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
                if let Some(resolved) = self.resolve(&re.source, &current) {
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
}

pub fn load_aliases(root: &Path) -> Vec<PathAlias> {
    let tsconfig_path = root.join("tsconfig.json");
    let content = match std::fs::read_to_string(&tsconfig_path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return vec![],
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

    let paths = match json
        .get("compilerOptions")
        .and_then(|co| co.get("paths"))
        .and_then(|p| p.as_object())
    {
        Some(p) => p,
        None => return vec![],
    };

    paths
        .iter()
        .filter_map(|(key, value)| {
            // key: "@/*", value: ["src/*"]
            let prefix = key.strip_suffix('*')?;
            let target_arr = value.as_array()?;
            let target_str = target_arr.first()?.as_str()?;
            let target = target_str.strip_suffix('*')?;
            Some(PathAlias {
                prefix: prefix.to_string(),
                target: target.to_string(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn resolve_relative_tsx() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("Button.tsx"), "").unwrap();
        fs::write(src.join("App.tsx"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./Button", "src/App.tsx");
        assert_eq!(result, Some("src/Button.tsx".to_string()));
    }

    #[test]
    fn resolve_index_file_tsx() {
        let tmp = tempfile::tempdir().unwrap();
        let components = tmp.path().join("src").join("components");
        fs::create_dir_all(&components).unwrap();
        fs::write(components.join("index.tsx"), "").unwrap();
        fs::write(tmp.path().join("src").join("App.tsx"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./components", "src/App.tsx");
        assert_eq!(result, Some("src/components/index.tsx".to_string()));
    }

    #[test]
    fn resolve_missing_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("App.tsx"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./Missing", "src/App.tsx");
        assert_eq!(result, None);
    }

    #[test]
    fn resolve_bare_specifier_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("react", "src/App.tsx");
        assert_eq!(result, None);
    }

    #[test]
    fn load_aliases_from_tsconfig() {
        let tmp = tempfile::tempdir().unwrap();
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
                prefix: "@/".to_string(),
                target: "src/".to_string(),
            }]
        );
    }

    #[test]
    fn load_aliases_no_tsconfig() {
        let tmp = tempfile::tempdir().unwrap();

        let aliases = load_aliases(tmp.path());
        assert!(aliases.is_empty());
    }

    #[test]
    fn resolve_alias_path() {
        let tmp = tempfile::tempdir().unwrap();
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
        assert_eq!(result, Some("src/lib/auth.ts".to_string()));
    }

    #[test]
    fn resolve_relative_ts() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("utils.ts"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./utils", "src/App.tsx");
        assert_eq!(result, Some("src/utils.ts".to_string()));
    }

    #[test]
    fn resolve_parent_directory() {
        let tmp = tempfile::tempdir().unwrap();
        let utils = tmp.path().join("src").join("utils");
        let components = tmp.path().join("src").join("components");
        fs::create_dir_all(&utils).unwrap();
        fs::create_dir_all(&components).unwrap();
        fs::write(utils.join("format.ts"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("../utils/format", "src/components/Button.tsx");
        assert_eq!(result, Some("src/utils/format.ts".to_string()));
    }

    #[test]
    fn resolve_css_file() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("styles.css"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./styles.css", "src/App.tsx");
        assert_eq!(result, Some("src/styles.css".to_string()));
    }

    #[test]
    fn resolve_prefers_tsx_over_ts() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("Button.tsx"), "").unwrap();
        fs::write(src.join("Button.ts"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./Button", "src/App.tsx");
        assert_eq!(result, Some("src/Button.tsx".to_string()));
    }

    #[test]
    fn resolve_index_file_ts() {
        let tmp = tempfile::tempdir().unwrap();
        let hooks = tmp.path().join("src").join("hooks");
        fs::create_dir_all(&hooks).unwrap();
        fs::write(hooks.join("index.ts"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./hooks", "src/App.tsx");
        assert_eq!(result, Some("src/hooks/index.ts".to_string()));
    }

    #[test]
    fn load_aliases_multiple() {
        let tmp = tempfile::tempdir().unwrap();
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
            prefix: "@/".to_string(),
            target: "src/".to_string(),
        }));
        assert!(aliases.contains(&PathAlias {
            prefix: "~/".to_string(),
            target: "lib/".to_string(),
        }));
    }

    #[test]
    fn resolve_exact_extension_match() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("styles.css"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./styles.css", "src/App.tsx");
        assert_eq!(result, Some("src/styles.css".to_string()));
    }

    #[test]
    fn resolve_non_frontend_extension_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("data.json"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./data", "src/App.tsx");
        assert_eq!(result, None);
    }

    #[test]
    fn resolve_scoped_package_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("@tanstack/react-query", "src/App.tsx");
        assert_eq!(result, None);
    }

    #[test]
    fn resolver_new_loads_aliases() {
        let tmp = tempfile::tempdir().unwrap();
        let tsconfig = r#"{
            "compilerOptions": {
                "paths": {
                    "@/*": ["src/*"]
                }
            }
        }"#;
        fs::write(tmp.path().join("tsconfig.json"), tsconfig).unwrap();

        let resolver = Resolver::new(tmp.path());
        let src = tmp.path().join("src").join("lib");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("utils.ts"), "").unwrap();

        let result = resolver.resolve("@/lib/utils", "src/App.tsx");
        assert_eq!(result, Some("src/lib/utils.ts".to_string()));
    }

    #[test]
    fn resolve_tilde_alias() {
        let tmp = tempfile::tempdir().unwrap();
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
        assert_eq!(result, Some("lib/core/engine.ts".to_string()));
    }

    #[test]
    fn resolve_reexport_chain_circular() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("a.ts"), "export { X } from './b';").unwrap();
        fs::write(src.join("b.ts"), "export { X } from './a';").unwrap();

        let resolver = Resolver::new(tmp.path());
        let chain = resolver.resolve_reexport_chain("src/a.ts");
        assert_eq!(chain, vec!["src/b.ts".to_string()]);
    }

    #[test]
    fn resolve_reexport_chain_linear() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("a.ts"), "export { X } from './b';").unwrap();
        fs::write(src.join("b.ts"), "export { X } from './c';").unwrap();
        fs::write(src.join("c.ts"), "export const X = 1;").unwrap();

        let resolver = Resolver::new(tmp.path());
        let chain = resolver.resolve_reexport_chain("src/a.ts");
        assert_eq!(chain, vec!["src/b.ts".to_string(), "src/c.ts".to_string()]);
    }

    #[test]
    fn resolve_reexport_chain_no_reexports() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("a.ts"), "export const X = 1;").unwrap();

        let resolver = Resolver::new(tmp.path());
        let chain = resolver.resolve_reexport_chain("src/a.ts");
        assert!(chain.is_empty());
    }

    #[test]
    fn resolve_html_file() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("template.html"), "").unwrap();

        let resolver = Resolver::new(tmp.path());
        let result = resolver.resolve("./template.html", "src/App.tsx");
        assert_eq!(result, Some("src/template.html".to_string()));
    }
}
