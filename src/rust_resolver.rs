use std::path::{Path, PathBuf};

use crate::fs_optional;
use crate::resolver::{Resolve, strip_canonical_prefix};

pub struct RustResolver {
    root: PathBuf,
    canonical_root: Option<PathBuf>,
    crate_name: Option<String>,
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
        let canonical_root = fs_optional::canonicalize_optional(root);
        let crate_name = read_crate_name(root);
        Self {
            root: root.to_path_buf(),
            canonical_root,
            crate_name,
        }
    }

    pub(crate) fn crate_name(&self) -> Option<&str> {
        self.crate_name.as_deref()
    }

    pub fn resolve(&self, source: &str, from_file: &str) -> Option<String> {
        let (head, rest) = source.split_once("::").unwrap_or((source, ""));
        match head {
            "crate" => self.resolve_crate(rest),
            "super" => self.resolve_super(source, from_file),
            "self" => self.resolve_self(rest, from_file),
            name if Some(name) == self.crate_name.as_deref() => self.resolve_crate(rest),
            _ => None,
        }
    }

    pub fn resolve_mod_decl(&self, name: &str, from_file: &str) -> Option<String> {
        self.resolve(&format!("self::{name}"), from_file)
    }
}

fn read_crate_name(root: &Path) -> Option<String> {
    let content = fs_optional::read_to_string_optional(&root.join("Cargo.toml"))?;
    let mut in_package = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(section) = trimmed.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            in_package = section.trim() == "package";
            continue;
        }
        if !in_package {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("name") {
            let after_eq = rest.trim_start().strip_prefix('=')?.trim();
            let stripped = after_eq.strip_prefix('"')?;
            let end = stripped.find('"')?;
            return Some(stripped[..end].replace('-', "_"));
        }
    }
    None
}

impl Resolve for RustResolver {
    // Both methods delegate to the inherent impls — Rust's inherent-method
    // resolution takes precedence over trait methods on `self`. Removing
    // either inherent definition would cause infinite recursion here.
    fn resolve(&self, source: &str, from_file: &str) -> Option<String> {
        self.resolve(source, from_file)
    }

    fn resolve_mod_decl(&self, name: &str, from_file: &str) -> Option<String> {
        self.resolve_mod_decl(name, from_file)
    }
}

#[cfg(test)]
mod tests;
