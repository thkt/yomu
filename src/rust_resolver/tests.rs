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

// T-334 [Spec T-002]: resolve_mod_decl_basic
//
// FR-002: `mod foo;` declared in src/lib.rs resolves to src/foo.rs
// when that sibling file exists. Mirrors the standard Rust module
// resolution rule for crate-root-level mod declarations.
#[test]
fn resolve_mod_decl_basic() {
    let tmp = tempdir().unwrap();
    let src = tmp.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("foo.rs"), "").unwrap();

    let resolver = RustResolver::new(tmp.path());
    let result = resolver.resolve_mod_decl("foo", "src/lib.rs");
    assert_eq!(result, Some("src/foo.rs".to_owned()));
}

// T-335 [Spec T-002]: resolve_mod_decl_mod_rs_fallback
//
// FR-002: when src/foo.rs is absent but src/foo/mod.rs exists,
// `mod foo;` resolves to the directory's mod.rs (legacy module layout).
#[test]
fn resolve_mod_decl_mod_rs_fallback() {
    let tmp = tempdir().unwrap();
    let foo = tmp.path().join("src").join("foo");
    fs::create_dir_all(&foo).unwrap();
    fs::write(foo.join("mod.rs"), "").unwrap();

    let resolver = RustResolver::new(tmp.path());
    let result = resolver.resolve_mod_decl("foo", "src/lib.rs");
    assert_eq!(result, Some("src/foo/mod.rs".to_owned()));
}

// T-336 [Spec T-002]: resolve_mod_decl_in_submodule
//
// FR-002: `mod graph;` declared inside src/storage.rs resolves to
// src/storage/graph.rs (sibling-module probing under the declaring
// file's module path, not crate root).
#[test]
fn resolve_mod_decl_in_submodule() {
    let tmp = tempdir().unwrap();
    let storage = tmp.path().join("src").join("storage");
    fs::create_dir_all(&storage).unwrap();
    fs::write(storage.join("graph.rs"), "").unwrap();

    let resolver = RustResolver::new(tmp.path());
    let result = resolver.resolve_mod_decl("graph", "src/storage.rs");
    assert_eq!(result, Some("src/storage/graph.rs".to_owned()));
}
