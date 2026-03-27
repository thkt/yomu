use std::path::{Path, PathBuf};

use crate::resolver::{Resolve, strip_canonical_prefix};

pub struct RustResolver {
    root: PathBuf,
    canonical_root: Option<PathBuf>,
}

impl RustResolver {
    pub fn new(root: &Path) -> Self {
        let canonical_root = root.canonicalize().ok();
        Self {
            root: root.to_path_buf(),
            canonical_root,
        }
    }

    pub fn resolve(&self, source: &str, from_file: &str) -> Option<String> {
        if let Some(rest) = source.strip_prefix("crate::") {
            self.resolve_crate(rest)
        } else if source.starts_with("super::") {
            self.resolve_super(source, from_file)
        } else if let Some(rest) = source.strip_prefix("self::") {
            self.resolve_self(rest, from_file)
        } else {
            None
        }
    }

    fn resolve_crate(&self, rest: &str) -> Option<String> {
        let segments: Vec<&str> = rest.split("::").collect();
        self.probe_with_symbol_fallback(&segments, 1)
    }

    fn resolve_super(&self, source: &str, from_file: &str) -> Option<String> {
        let mut module_path = module_path_from_file(from_file);
        let mut rest = source;
        while let Some(after) = rest.strip_prefix("super::") {
            if module_path.is_empty() {
                return None;
            }
            module_path.pop();
            rest = after;
        }
        let min_len = module_path.len() + 1;
        let segments: Vec<&str> = module_path
            .iter()
            .map(|s| s.as_str())
            .chain(rest.split("::"))
            .collect();
        self.probe_with_symbol_fallback(&segments, min_len)
    }

    fn resolve_self(&self, rest: &str, from_file: &str) -> Option<String> {
        let module_path = module_path_from_file(from_file);
        let min_len = module_path.len() + 1;
        let segments: Vec<&str> = module_path
            .iter()
            .map(|s| s.as_str())
            .chain(rest.split("::"))
            .collect();
        self.probe_with_symbol_fallback(&segments, min_len)
    }

    fn probe_with_symbol_fallback(&self, segments: &[&str], min_len: usize) -> Option<String> {
        self.probe_module(segments).or_else(|| {
            (segments.len() > min_len)
                .then(|| self.probe_module(&segments[..segments.len() - 1]))
                .flatten()
        })
    }

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
}

impl Resolve for RustResolver {
    fn resolve(&self, source: &str, from_file: &str) -> Option<String> {
        self.resolve(source, from_file)
    }
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
        segments.push(stem.to_string());
    }

    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn resolve_crate_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let foo = tmp.path().join("src").join("foo");
        fs::create_dir_all(&foo).unwrap();
        fs::write(foo.join("bar.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("crate::foo::bar", "src/lib.rs");
        assert_eq!(result, Some("src/foo/bar.rs".to_string()));
    }

    #[test]
    fn resolve_crate_nested() {
        let tmp = tempfile::tempdir().unwrap();
        let abc = tmp.path().join("src").join("a").join("b");
        fs::create_dir_all(&abc).unwrap();
        fs::write(abc.join("c.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("crate::a::b::c", "src/lib.rs");
        assert_eq!(result, Some("src/a/b/c.rs".to_string()));
    }

    #[test]
    fn resolve_crate_mod_rs_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let foo = tmp.path().join("src").join("foo");
        fs::create_dir_all(&foo).unwrap();
        fs::write(foo.join("mod.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("crate::foo", "src/lib.rs");
        assert_eq!(result, Some("src/foo/mod.rs".to_string()));
    }

    #[test]
    fn resolve_super_sibling() {
        let tmp = tempfile::tempdir().unwrap();
        let foo = tmp.path().join("src").join("foo");
        fs::create_dir_all(&foo).unwrap();
        fs::write(foo.join("bar.rs"), "").unwrap();
        fs::write(foo.join("baz.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("super::baz", "src/foo/bar.rs");
        assert_eq!(result, Some("src/foo/baz.rs".to_string()));
    }

    #[test]
    fn resolve_super_at_root_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("foo.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("super::foo", "src/lib.rs");
        assert_eq!(result, None);
    }

    #[test]
    fn resolve_self_submodule() {
        let tmp = tempfile::tempdir().unwrap();
        let foo = tmp.path().join("src").join("foo");
        fs::create_dir_all(&foo).unwrap();
        fs::write(foo.join("mod.rs"), "").unwrap();
        fs::write(foo.join("sub.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("self::sub", "src/foo/mod.rs");
        assert_eq!(result, Some("src/foo/sub.rs".to_string()));
    }

    #[test]
    fn resolve_prefers_rs_over_mod_rs() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let foo_dir = src.join("foo");
        fs::create_dir_all(&foo_dir).unwrap();
        fs::write(src.join("foo.rs"), "").unwrap();
        fs::write(foo_dir.join("mod.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("crate::foo", "src/lib.rs");
        assert_eq!(result, Some("src/foo.rs".to_string()));
    }

    #[test]
    fn resolve_missing_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        fs::create_dir_all(&src).unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("crate::missing", "src/lib.rs");
        assert_eq!(result, None);
    }

    #[test]
    fn resolve_super_from_mod_rs() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let foo = src.join("foo");
        fs::create_dir_all(&foo).unwrap();
        fs::write(foo.join("mod.rs"), "").unwrap();
        fs::write(src.join("bar.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("super::bar", "src/foo/mod.rs");
        assert_eq!(result, Some("src/bar.rs".to_string()));
    }

    #[test]
    fn resolve_self_from_file() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let foo = src.join("foo");
        fs::create_dir_all(&foo).unwrap();
        fs::write(src.join("foo.rs"), "").unwrap();
        fs::write(foo.join("bar.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("self::bar", "src/foo.rs");
        assert_eq!(result, Some("src/foo/bar.rs".to_string()));
    }

    #[test]
    fn resolve_external_crate_returns_none() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();

        let resolver = RustResolver::new(tmp.path());
        let result = resolver.resolve("serde::Deserialize", "src/lib.rs");
        assert_eq!(result, None);
    }

    #[test]
    fn resolve_super_super_multi_level() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        let a = src.join("a");
        let deep = a.join("b");
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("c.rs"), "").unwrap();
        fs::write(a.join("util.rs"), "").unwrap();

        let resolver = RustResolver::new(tmp.path());
        // a::b::c → super → a::b → super → a → util → a::util
        let result = resolver.resolve("super::super::util", "src/a/b/c.rs");
        assert_eq!(result, Some("src/a/util.rs".to_string()));
    }
}
