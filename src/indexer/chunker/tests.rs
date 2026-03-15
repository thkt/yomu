use super::*;

#[test]
fn chunk_tsx_component_function() {
    let source = "function Button() { return <div/>; }";
    let result = chunk_file(source, "tsx");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::Component);
    assert_eq!(result.chunks[0].name.as_deref(), Some("Button"));
}

#[test]
fn chunk_tsx_hook() {
    let source = "function useAuth() { return { user: null }; }";
    let result = chunk_file(source, "tsx");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::Hook);
    assert_eq!(result.chunks[0].name.as_deref(), Some("useAuth"));
}

#[test]
fn chunk_tsx_interface() {
    let source = "interface Props { label: string; }";
    let result = chunk_file(source, "tsx");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::TypeDef);
    assert_eq!(result.chunks[0].name.as_deref(), Some("Props"));
}

#[test]
fn chunk_tsx_exported_arrow_component() {
    let source = "export const Card = () => { return <div/>; };";
    let result = chunk_file(source, "tsx");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::Component);
    assert_eq!(result.chunks[0].name.as_deref(), Some("Card"));
}

#[test]
fn chunk_css_rule_set() {
    let source = ".container { color: red; }";
    let result = chunk_file(source, "css");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::CssRule);
    assert!(result.imports.is_empty());
}

#[test]
fn chunk_html_element() {
    let source = "<html><body>Hello</body></html>";
    let result = chunk_file(source, "html");
    assert!(!result.chunks.is_empty());
    assert_eq!(result.chunks[0].chunk_type, ChunkType::HtmlElement);
    assert!(result.imports.is_empty());
}

#[test]
fn chunk_tsx_test_case() {
    let source = r#"describe('Auth', () => {
    it('should render', () => {
        expect(true).toBe(true);
    });
});"#;
    let result = chunk_file(source, "tsx");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::TestCase);
    assert_eq!(result.chunks[0].name.as_deref(), Some("Auth"));
}

#[test]
fn chunk_fallback_for_unknown_extension() {
    let source = "line1\nline2\nline3\nline4\nline5";
    let result = chunk_file(source, "toml");
    assert!(!result.chunks.is_empty());
    assert_eq!(result.chunks[0].chunk_type, ChunkType::Other);
    assert!(result.imports.is_empty());
}

#[test]
fn chunk_tsx_type_alias() {
    let source = "type Theme = 'light' | 'dark';";
    let result = chunk_file(source, "tsx");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::TypeDef);
    assert_eq!(result.chunks[0].name.as_deref(), Some("Theme"));
}

#[test]
fn chunk_js_file() {
    let source = "function App() { return 'hello'; }";
    let result = chunk_file(source, "js");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::Component);
    assert_eq!(result.chunks[0].name.as_deref(), Some("App"));
}

#[test]
fn chunk_line_numbers() {
    let source = "\nfunction Foo() {\n  return 42;\n}\n";
    let result = chunk_file(source, "tsx");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].start_line, 2);
    assert_eq!(result.chunks[0].end_line, 4);
}

#[test]
fn chunk_tsx_it_test_case() {
    let source = r#"it('should work', () => {
    expect(1).toBe(1);
});"#;
    let result = chunk_file(source, "tsx");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::TestCase);
    assert_eq!(result.chunks[0].name.as_deref(), Some("should work"));
}

#[test]
fn chunk_tsx_test_fn_case() {
    let source = r#"test('adds numbers', () => {
    expect(1 + 2).toBe(3);
});"#;
    let result = chunk_file(source, "tsx");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::TestCase);
    assert_eq!(result.chunks[0].name.as_deref(), Some("adds numbers"));
}

#[test]
fn chunk_css_media_statement() {
    let source = "@media (max-width: 768px) { .container { display: none; } }";
    let result = chunk_file(source, "css");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::CssRule);
}

#[test]
fn chunk_css_keyframes() {
    let source = "@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }";
    let result = chunk_file(source, "css");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::CssRule);
}

#[test]
fn classify_non_hook_use_function() {
    let source = "function username() { return 'alice'; }";
    let result = chunk_file(source, "tsx");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::Other);
}

#[test]
fn classify_utility_function_as_other() {
    let source = "function formatDate() { return '2024-01-01'; }";
    let result = chunk_file(source, "tsx");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::Other);
    assert_eq!(result.chunks[0].name.as_deref(), Some("formatDate"));
}

#[test]
fn chunk_fallback_produces_overlapping_chunks() {
    let line = "x".repeat(60);
    let source = std::iter::repeat_n(line.as_str(), 30)
        .collect::<Vec<_>>()
        .join("\n");
    let chunks = chunk_fallback(&source);
    assert!(
        chunks.len() >= 2,
        "expected overlapping chunks, got {}",
        chunks.len()
    );
    assert!(
        chunks[1].start_line < chunks[0].end_line,
        "expected overlap: chunk[1].start={} < chunk[0].end={}",
        chunks[1].start_line,
        chunks[0].end_line
    );
}

#[test]
fn extract_single_import() {
    let source = "import { useState } from 'react';\nfunction App() { return <div/>; }";
    let result = chunk_file(source, "tsx");
    assert_eq!(result.imports.len(), 1);
    assert_eq!(result.imports[0], "import { useState } from 'react';");
    assert_eq!(result.chunks.len(), 1);
    assert_eq!(result.chunks[0].chunk_type, ChunkType::Component);
}

#[test]
fn extract_multiple_imports() {
    let source = "import { useState } from 'react';\nimport { useAuth } from './useAuth';\nimport type { Props } from './types';\nfunction App() { return <div/>; }";
    let result = chunk_file(source, "tsx");
    assert_eq!(result.imports.len(), 3);
    assert!(result.imports[0].contains("useState"));
    assert!(result.imports[1].contains("useAuth"));
    assert!(result.imports[2].contains("Props"));
}

#[test]
fn no_imports_returns_empty_vec() {
    let source = "function App() { return <div/>; }";
    let result = chunk_file(source, "tsx");
    assert!(result.imports.is_empty());
}

// ── parse_structured_imports tests (FR-001 / T-001..T-006 + edge cases) ──

#[test]
fn parse_named_imports() {
    let source = "import { a, b } from './mod';";
    let result = parse_structured_imports(source, "tsx");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].source, "./mod");
    assert_eq!(result[0].specifiers.len(), 2);
    assert_eq!(result[0].specifiers[0].name, "a");
    assert_eq!(result[0].specifiers[0].kind, ImportKind::Named);
    assert_eq!(result[0].specifiers[0].alias, None);
    assert_eq!(result[0].specifiers[1].name, "b");
    assert_eq!(result[0].specifiers[1].kind, ImportKind::Named);
}

#[test]
fn parse_default_import() {
    let source = "import X from './mod';";
    let result = parse_structured_imports(source, "tsx");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].source, "./mod");
    assert_eq!(result[0].specifiers.len(), 1);
    assert_eq!(result[0].specifiers[0].name, "X");
    assert_eq!(result[0].specifiers[0].kind, ImportKind::Default);
}

#[test]
fn parse_namespace_import() {
    let source = "import * as X from './mod';";
    let result = parse_structured_imports(source, "tsx");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].source, "./mod");
    assert_eq!(result[0].specifiers.len(), 1);
    assert_eq!(result[0].specifiers[0].name, "*");
    assert_eq!(result[0].specifiers[0].alias, Some("X".to_string()));
    assert_eq!(result[0].specifiers[0].kind, ImportKind::Namespace);
}

#[test]
fn parse_type_only_import() {
    let source = "import type { T } from './types';";
    let result = parse_structured_imports(source, "tsx");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].source, "./types");
    assert_eq!(result[0].specifiers.len(), 1);
    assert_eq!(result[0].specifiers[0].name, "T");
    assert_eq!(result[0].specifiers[0].kind, ImportKind::TypeOnly);
}

#[test]
fn parse_side_effect_import() {
    let source = "import './side-effect';";
    let result = parse_structured_imports(source, "tsx");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].source, "./side-effect");
    assert!(result[0].specifiers.is_empty());
}

#[test]
fn parse_no_imports_returns_empty() {
    let source = "function App() { return <div/>; }";
    let result = parse_structured_imports(source, "tsx");
    assert!(result.is_empty());
}

#[test]
fn parse_multiple_imports() {
    let source = "\
import { useState } from 'react';
import { useAuth } from './useAuth';
import type { Props } from './types';
function App() { return <div/>; }";
    let result = parse_structured_imports(source, "tsx");
    assert_eq!(result.len(), 3);
    assert_eq!(result[0].source, "react");
    assert_eq!(result[0].specifiers[0].name, "useState");
    assert_eq!(result[1].source, "./useAuth");
    assert_eq!(result[1].specifiers[0].name, "useAuth");
    assert_eq!(result[2].source, "./types");
    assert_eq!(result[2].specifiers[0].kind, ImportKind::TypeOnly);
}

#[test]
fn parse_default_type_import() {
    let source = "import type X from './mod';";
    let result = parse_structured_imports(source, "tsx");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].specifiers.len(), 1);
    assert_eq!(result[0].specifiers[0].name, "X");
    assert_eq!(result[0].specifiers[0].kind, ImportKind::TypeOnly);
}

#[test]
fn parse_inline_type_specifier() {
    let source = "import { type T, useState } from 'react';";
    let result = parse_structured_imports(source, "tsx");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].source, "react");
    assert_eq!(result[0].specifiers.len(), 2);
    let type_spec = result[0]
        .specifiers
        .iter()
        .find(|s| s.name == "T")
        .expect("should have specifier T");
    assert_eq!(type_spec.kind, ImportKind::TypeOnly);
    let named_spec = result[0]
        .specifiers
        .iter()
        .find(|s| s.name == "useState")
        .expect("should have specifier useState");
    assert_eq!(named_spec.kind, ImportKind::Named);
}

// ── parse_reexports tests (FR-004 / T-014..T-015 + edge cases) ──

#[test]
fn parse_reexport_named() {
    let source = "export { Button } from './Button';";
    let result = parse_reexports(source, "tsx");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].symbol_name, Some("Button".to_string()));
    assert_eq!(result[0].source, "./Button");
}

#[test]
fn parse_reexport_star() {
    let source = "export * from './module';";
    let result = parse_reexports(source, "tsx");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].symbol_name, None);
    assert_eq!(result[0].source, "./module");
}

#[test]
fn parse_reexport_multiple_named() {
    let source = "export { Button, Card } from './components';";
    let result = parse_reexports(source, "tsx");
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].symbol_name, Some("Button".to_string()));
    assert_eq!(result[1].symbol_name, Some("Card".to_string()));
    assert_eq!(result[0].source, "./components");
}

#[test]
fn parse_reexport_ignores_local_export() {
    let source = "export const X = 1;";
    let result = parse_reexports(source, "tsx");
    assert!(result.is_empty());
}

#[test]
fn parse_reexport_mixed_with_imports() {
    let source = "import { useState } from 'react';\nexport { Button } from './Button';\nexport * from './utils';";
    let result = parse_reexports(source, "tsx");
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].symbol_name, Some("Button".to_string()));
    assert_eq!(result[1].symbol_name, None);
    assert_eq!(result[1].source, "./utils");
}

#[test]
fn parse_reexport_empty_source() {
    let source = "function App() { return <div/>; }";
    let result = parse_reexports(source, "tsx");
    assert!(result.is_empty());
}

#[test]
fn parse_aliased_named_import() {
    let source = "import { useState as useS } from 'react';";
    let result = parse_structured_imports(source, "tsx");
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].specifiers.len(), 1);
    assert_eq!(result[0].specifiers[0].name, "useState");
    assert_eq!(result[0].specifiers[0].alias, Some("useS".to_string()));
    assert_eq!(result[0].specifiers[0].kind, ImportKind::Named);
}
