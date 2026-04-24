use std::path::{Path, PathBuf};

use crate::resolver::{Resolve, strip_canonical_prefix};

pub struct RustResolver {
    root: PathBuf,
    canonical_root: Option<PathBuf>,
}

fn module_path_from_file(from_file: &str) -> Vec<String> {
    let path = from_file.strip_prefix("src/").unwrap_or(from_file);
    let stem = Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    let parent = Path::new(path).parent().unwrap_or(Path::new(""));

    if stem == "lib" || stem == "main" {
        return vec![];
    }

    let mut segments: Vec<String> = parent
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();

    if stem != "mod" {
        segments.push(stem.to_owned());
    }

    segments
}

impl RustResolver {
    fn probe_module(&self, segments: &[&str]) -> Option<String> {
        let path = segments.join("/");
        let canonical_root = self.canonical_root.as_deref();
        let rs_candidate = self.root.join("src").join(format!("{path}.rs"));
        if let Ok(abs) = rs_candidate.canonicalize() {
            return strip_canonical_prefix(&abs, canonical_root);
        }
        let mod_candidate = self.root.join("src").join(&path).join("mod.rs");
        if let Ok(abs) = mod_candidate.canonicalize() {
            return strip_canonical_prefix(&abs, canonical_root);
        }
        None
    }

    fn probe_crate_root(&self) -> Option<String> {
        let canonical_root = self.canonical_root.as_deref();
        for name in ["lib.rs", "main.rs"] {
            let candidate = self.root.join("src").join(name);
            if let Ok(abs) = candidate.canonicalize() {
                return strip_canonical_prefix(&abs, canonical_root);
            }
        }
        None
    }

    fn probe_module_or_crate_root(&self, module_path: &[String]) -> Option<String> {
        if module_path.is_empty() {
            return self.probe_crate_root();
        }
        let segments: Vec<&str> = module_path.iter().map(String::as_str).collect();
        self.probe_module(&segments)
    }

    fn probe_with_symbol_fallback(&self, segments: &[&str], min_len: usize) -> Option<String> {
        self.probe_module(segments).or_else(|| {
            (segments.len() > min_len)
                .then(|| self.probe_module(&segments[..segments.len() - 1]))
                .flatten()
        })
    }

    fn resolve_crate(&self, rest: &str) -> Option<String> {
        if rest.is_empty() {
            return self.probe_crate_root();
        }
        let segments: Vec<&str> = rest.split("::").collect();
        self.probe_with_symbol_fallback(&segments, 1)
    }

    fn resolve_super(&self, source: &str, from_file: &str) -> Option<String> {
        let mut module_path = module_path_from_file(from_file);
        let mut rest = source;
        loop {
            let next = if let Some(after) = rest.strip_prefix("super::") {
                after
            } else if rest == "super" {
                ""
            } else {
                break;
            };
            if module_path.is_empty() {
                return None;
            }
            module_path.pop();
            rest = next;
        }
        if rest.is_empty() {
            return self.probe_module_or_crate_root(&module_path);
        }
        let min_len = module_path.len();
        let segments: Vec<&str> = module_path
            .iter()
            .map(String::as_str)
            .chain(rest.split("::"))
            .collect();
        self.probe_with_symbol_fallback(&segments, min_len)
    }

    fn resolve_self(&self, rest: &str, from_file: &str) -> Option<String> {
        let module_path = module_path_from_file(from_file);
        if rest.is_empty() {
            return self.probe_module_or_crate_root(&module_path);
        }
        let min_len = module_path.len();
        let segments: Vec<&str> = module_path
            .iter()
            .map(String::as_str)
            .chain(rest.split("::"))
            .collect();
        self.probe_with_symbol_fallback(&segments, min_len)
    }

    pub fn new(root: &Path) -> Self {
        let canonical_root = root.canonicalize().ok();
        Self {
            root: root.to_path_buf(),
            canonical_root,
        }
    }

    pub fn resolve(&self, source: &str, from_file: &str) -> Option<String> {
        let (head, rest) = source.split_once("::").unwrap_or((source, ""));
        match head {
            "crate" => self.resolve_crate(rest),
            "super" => self.resolve_super(source, from_file),
            "self" => self.resolve_self(rest, from_file),
            _ => None,
        }
    }
}

impl Resolve for RustResolver {
    fn resolve(&self, source: &str, from_file: &str) -> Option<String> {
        self.resolve(source, from_file)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::*;

    // T-315: resolve_crate_basic
    #[test]
    fn resolve_crate_basic() {
        let tmp = tempdir().unwrap();
        let foo = tmp.path().join("src").join("foo");
        fs::create_dir_all(&foo).unwrap();
        fs::write(foo.join("bar.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("crate::foo::bar", "src/lib.rs");
        assert_eq!(result, Some("src/foo/bar.rs".to_owned()));
    }

    // T-316: resolve_crate_nested
    #[test]
    fn resolve_crate_nested() {
        let tmp = tempdir().unwrap();
        let abc = tmp.path().join("src").join("a").join("b");
        fs::create_dir_all(&abc).unwrap();
        fs::write(abc.join("c.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("crate::a::b::c", "src/lib.rs");
        assert_eq!(result, Some("src/a/b/c.rs".to_owned()));
    }

    // T-317: resolve_crate_mod_rs_fallback
    #[test]
    fn resolve_crate_mod_rs_fallback() {
        let tmp = tempdir().unwrap();
        let foo = tmp.path().join("src").join("foo");
        fs::create_dir_all(&foo).unwrap();
        fs::write(foo.join("mod.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("crate::foo", "src/lib.rs");
        assert_eq!(result, Some("src/foo/mod.rs".to_owned()));
    }

    // T-318: resolve_super_sibling
    #[test]
    fn resolve_super_sibling() {
        let tmp = tempdir().unwrap();
        let foo = tmp.path().join("src").join("foo");
        fs::create_dir_all(&foo).unwrap();
        fs::write(foo.join("bar.rs"), "").unwrap();
        fs::write(foo.join("baz.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("super::baz", "src/foo/bar.rs");
        assert_eq!(result, Some("src/foo/baz.rs".to_owned()));
    }

    // T-319: resolve_super_at_root_returns_none
    #[test]
    fn resolve_super_at_root_returns_none() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("foo.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("super::foo", "src/lib.rs");
        assert_eq!(result, None);
    }

    // T-320: resolve_self_submodule
    #[test]
    fn resolve_self_submodule() {
        let tmp = tempdir().unwrap();
        let foo = tmp.path().join("src").join("foo");
        fs::create_dir_all(&foo).unwrap();
        fs::write(foo.join("mod.rs"), "").unwrap();
        fs::write(foo.join("sub.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("self::sub", "src/foo/mod.rs");
        assert_eq!(result, Some("src/foo/sub.rs".to_owned()));
    }

    // T-321: resolve_prefers_rs_over_mod_rs
    #[test]
    fn resolve_prefers_rs_over_mod_rs() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        let foo_dir = src.join("foo");
        fs::create_dir_all(&foo_dir).unwrap();
        fs::write(src.join("foo.rs"), "").unwrap();
        fs::write(foo_dir.join("mod.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("crate::foo", "src/lib.rs");
        assert_eq!(result, Some("src/foo.rs".to_owned()));
    }

    // T-322: resolve_missing_returns_none
    #[test]
    fn resolve_missing_returns_none() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("crate::missing", "src/lib.rs");
        assert_eq!(result, None);
    }

    // T-323: resolve_super_from_mod_rs
    #[test]
    fn resolve_super_from_mod_rs() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        let foo = src.join("foo");
        fs::create_dir_all(&foo).unwrap();
        fs::write(foo.join("mod.rs"), "").unwrap();
        fs::write(src.join("bar.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("super::bar", "src/foo/mod.rs");
        assert_eq!(result, Some("src/bar.rs".to_owned()));
    }

    // T-324: resolve_self_from_file
    #[test]
    fn resolve_self_from_file() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        let foo = src.join("foo");
        fs::create_dir_all(&foo).unwrap();
        fs::write(src.join("foo.rs"), "").unwrap();
        fs::write(foo.join("bar.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("self::bar", "src/foo.rs");
        assert_eq!(result, Some("src/foo/bar.rs".to_owned()));
    }

    // T-325: resolve_external_crate_returns_none
    #[test]
    fn resolve_external_crate_returns_none() {
        let tmp = tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("serde::Deserialize", "src/lib.rs");
        assert_eq!(result, None);
    }

    // T-326: resolve_super_super_multi_level
    #[test]
    fn resolve_super_super_multi_level() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        let a = src.join("a");
        let deep = a.join("b");
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("c.rs"), "").unwrap();
        fs::write(a.join("util.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        // a::b::c → super → a::b → super → a → util → a::util
        let result = resolver.resolve("super::super::util", "src/a/b/c.rs");
        assert_eq!(result, Some("src/a/util.rs".to_owned()));
    }

    // T-327: resolve_super_symbol_fallback
    #[test]
    fn resolve_super_symbol_fallback() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        let tools = src.join("tools");
        fs::create_dir_all(&tools).unwrap();
        fs::write(src.join("tools.rs"), "").unwrap();
        fs::write(tools.join("reranker.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        // `use super::Yomu` from src/tools/reranker.rs → parent module src/tools.rs
        let result = resolver.resolve("super::Yomu", "src/tools/reranker.rs");
        assert_eq!(result, Some("src/tools.rs".to_owned()));
    }

    // T-328: resolve_self_symbol_fallback
    #[test]
    fn resolve_self_symbol_fallback() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("foo.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        // `use self::Symbol` from src/foo.rs → current module src/foo.rs
        let result = resolver.resolve("self::Symbol", "src/foo.rs");
        assert_eq!(result, Some("src/foo.rs".to_owned()));
    }

    // T-329: resolve_bare_super
    #[test]
    fn resolve_bare_super() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        let tools = src.join("tools");
        fs::create_dir_all(&tools).unwrap();
        fs::write(src.join("tools.rs"), "").unwrap();
        fs::write(tools.join("reranker.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        // parser emits source="super" for `use super::Yomu;`
        let result = resolver.resolve("super", "src/tools/reranker.rs");
        assert_eq!(result, Some("src/tools.rs".to_owned()));
    }

    // T-330: resolve_bare_self
    #[test]
    fn resolve_bare_self() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("foo.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        // parser emits source="self" for `use self::Symbol;`
        let result = resolver.resolve("self", "src/foo.rs");
        assert_eq!(result, Some("src/foo.rs".to_owned()));
    }

    // T-331: resolve_bare_crate_to_lib_rs
    #[test]
    fn resolve_bare_crate_to_lib_rs() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("lib.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        // parser emits source="crate" for `use crate::Yomu;`
        let result = resolver.resolve("crate", "src/lib.rs");
        assert_eq!(result, Some("src/lib.rs".to_owned()));
    }

    // T-332: resolve_bare_crate_to_main_rs
    #[test]
    fn resolve_bare_crate_to_main_rs() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("main.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("crate", "src/main.rs");
        assert_eq!(result, Some("src/main.rs".to_owned()));
    }

    // T-333: resolve_bare_super_at_root_returns_none
    #[test]
    fn resolve_bare_super_at_root_returns_none() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("lib.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("super", "src/lib.rs");
        assert_eq!(result, None);
    }
}
