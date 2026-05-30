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
