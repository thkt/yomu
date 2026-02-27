use crate::storage::ChunkType;

/// Intermediate chunk representation produced by the chunker.
///
/// Data flow: source code → `RawChunk` (chunker) → `NewChunk` (storage insert) → `Chunk` (storage read).
#[derive(Debug, Clone)]
pub struct RawChunk {
    pub chunk_type: ChunkType,
    pub name: Option<String>,
    pub content: String,
    pub start_line: u32,
    pub end_line: u32,
}

/// Chunker output with file-level context.
///
/// Data flow: source code → `FileChunks` (chunker) → `PendingFile` (indexer) → DB.
#[derive(Debug, Clone)]
pub struct FileChunks {
    /// Raw import statement text (JS/TS only; empty for CSS/HTML/fallback).
    pub imports: Vec<String>,
    pub chunks: Vec<RawChunk>,
}

/// Split source code into semantic chunks using AST parsing.
///
/// Supported extensions: `tsx`, `jsx`, `ts`, `js`, `mjs`, `css`, `html`.
/// Unknown extensions fall back to character-based splitting.
/// Returns empty chunks if `source` is empty/whitespace-only.
/// For JS/TS files, also extracts import statements as file-level context.
pub fn chunk_file(source: &str, extension: &str) -> FileChunks {
    match extension {
        "tsx" | "jsx" => chunk_tsx(source),
        "ts" | "js" | "mjs" => chunk_ts(source),
        "css" => FileChunks { imports: vec![], chunks: chunk_css(source) },
        "html" => FileChunks { imports: vec![], chunks: chunk_html(source) },
        _ => FileChunks { imports: vec![], chunks: chunk_fallback(source) },
    }
}

fn make_parser(lang: &tree_sitter::Language) -> Option<tree_sitter::Parser> {
    let mut parser = tree_sitter::Parser::new();
    if let Err(e) = parser.set_language(lang) {
        tracing::warn!(error = %e, "Failed to set tree-sitter language, using fallback chunker");
        return None;
    }
    Some(parser)
}

fn chunk_tsx(source: &str) -> FileChunks {
    let Some(mut parser) = make_parser(&tree_sitter_typescript::LANGUAGE_TSX.into()) else {
        return FileChunks { imports: vec![], chunks: chunk_fallback(source) };
    };
    chunk_js_like_with_imports(source, &mut parser)
}

fn chunk_ts(source: &str) -> FileChunks {
    let Some(mut parser) = make_parser(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()) else {
        return FileChunks { imports: vec![], chunks: chunk_fallback(source) };
    };
    chunk_js_like_with_imports(source, &mut parser)
}

fn chunk_with_ast(
    source: &str,
    parser: &mut tree_sitter::Parser,
    classify: impl Fn(&tree_sitter::Node, &str) -> Option<RawChunk>,
) -> Vec<RawChunk> {
    let Some(tree) = parser.parse(source, None) else {
        tracing::warn!("AST parse failed, using fallback chunker");
        return chunk_fallback(source);
    };
    let root = tree.root_node();
    let mut cursor = root.walk();
    let chunks: Vec<RawChunk> = root
        .children(&mut cursor)
        .filter_map(|node| classify(&node, source))
        .collect();
    if chunks.is_empty() {
        return chunk_fallback(source);
    }
    chunks
}

fn chunk_js_like_with_imports(source: &str, parser: &mut tree_sitter::Parser) -> FileChunks {
    let Some(tree) = parser.parse(source, None) else {
        tracing::warn!("AST parse failed, using fallback chunker");
        return FileChunks { imports: vec![], chunks: chunk_fallback(source) };
    };
    let root = tree.root_node();
    let imports = extract_imports_from_ast(&root, source);
    let mut cursor = root.walk();
    let chunks: Vec<RawChunk> = root
        .children(&mut cursor)
        .filter_map(|node| classify_js_node(&node, source))
        .collect();
    if chunks.is_empty() {
        return FileChunks { imports, chunks: chunk_fallback(source) };
    }
    FileChunks { imports, chunks }
}

fn extract_imports_from_ast(root: &tree_sitter::Node, source: &str) -> Vec<String> {
    let mut cursor = root.walk();
    root.children(&mut cursor)
        .filter(|node| node.kind() == "import_statement")
        .map(|node| source[node.byte_range()].to_string())
        .collect()
}

fn classify_js_node(node: &tree_sitter::Node, source: &str) -> Option<RawChunk> {
    match node.kind() {
        "function_declaration" => {
            let name = extract_name(node, source);
            let chunk_type = classify_function(name.as_deref());
            Some(make_chunk(source, node, chunk_type, name))
        }
        "export_statement" => {
            let (chunk_type, name) = classify_export(node, source);
            Some(make_chunk(source, node, chunk_type, name))
        }
        "interface_declaration" | "type_alias_declaration" => {
            let name = extract_name(node, source);
            Some(make_chunk(source, node, ChunkType::TypeDef, name))
        }
        "expression_statement" => {
            let (chunk_type, name) = classify_expression(node, source);
            Some(make_chunk(source, node, chunk_type, name))
        }
        "import_statement" | "comment" => None,
        _ => {
            let text = &source[node.byte_range()];
            if text.trim().is_empty() {
                None
            } else {
                Some(make_chunk(source, node, ChunkType::Other, None))
            }
        }
    }
}

fn chunk_css(source: &str) -> Vec<RawChunk> {
    let Some(mut parser) = make_parser(&tree_sitter_css::LANGUAGE.into()) else {
        return chunk_fallback(source);
    };
    chunk_with_ast(source, &mut parser, classify_css_node)
}

fn classify_css_node(node: &tree_sitter::Node, source: &str) -> Option<RawChunk> {
    match node.kind() {
        "rule_set" | "media_statement" | "at_rule" | "keyframes_statement" => {
            Some(make_chunk(source, node, ChunkType::CssRule, None))
        }
        "comment" => None,
        _ => {
            let text = &source[node.byte_range()];
            if text.trim().is_empty() {
                None
            } else {
                Some(make_chunk(source, node, ChunkType::Other, None))
            }
        }
    }
}

fn chunk_html(source: &str) -> Vec<RawChunk> {
    let Some(mut parser) = make_parser(&tree_sitter_html::LANGUAGE.into()) else {
        return chunk_fallback(source);
    };
    chunk_with_ast(source, &mut parser, classify_html_node)
}

fn classify_html_node(node: &tree_sitter::Node, source: &str) -> Option<RawChunk> {
    match node.kind() {
        "element" | "doctype" => Some(make_chunk(source, node, ChunkType::HtmlElement, None)),
        _ => None,
    }
}

fn chunk_fallback(source: &str) -> Vec<RawChunk> {
    const MAX_CHUNK_SIZE: usize = 1000;

    if source.trim().is_empty() {
        return Vec::new();
    }

    let lines: Vec<&str> = source.lines().collect();
    let mut chunks = Vec::new();
    let mut start_idx = 0;

    while start_idx < lines.len() {
        let mut end_idx = start_idx;
        let mut char_count = 0;

        while end_idx < lines.len() && char_count < MAX_CHUNK_SIZE {
            char_count += lines[end_idx].len() + 1; // +1 for newline
            end_idx += 1;
        }

        let content: String = lines[start_idx..end_idx].join("\n");
        if !content.trim().is_empty() {
            chunks.push(RawChunk {
                chunk_type: ChunkType::Other,
                name: None,
                content,
                start_line: (start_idx + 1) as u32,
                end_line: end_idx as u32,
            });
        }

        // ~4 lines of overlap assuming ~50 chars/line for frontend code
        const OVERLAP_LINES: usize = 4;
        if end_idx >= lines.len() {
            break;
        }
        start_idx = end_idx.saturating_sub(OVERLAP_LINES);
    }

    chunks
}

fn make_chunk(
    source: &str,
    node: &tree_sitter::Node,
    chunk_type: ChunkType,
    name: Option<String>,
) -> RawChunk {
    RawChunk {
        chunk_type,
        name,
        content: source[node.byte_range()].to_string(),
        start_line: node.start_position().row as u32 + 1,
        end_line: node.end_position().row as u32 + 1,
    }
}

fn extract_name(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" || child.kind() == "type_identifier" {
            return Some(source[child.byte_range()].to_string());
        }
    }
    None
}

fn classify_function(name: Option<&str>) -> ChunkType {
    match name {
        Some(n) if n.starts_with("use")
            && n.len() > 3
            && n.as_bytes()[3].is_ascii_uppercase() => ChunkType::Hook,
        Some(n) if n.as_bytes().first().is_some_and(|b| b.is_ascii_uppercase()) => {
            ChunkType::Component
        }
        _ => ChunkType::Other,
    }
}

fn classify_export(node: &tree_sitter::Node, source: &str) -> (ChunkType, Option<String>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        match child.kind() {
            "function_declaration" => {
                let name = extract_name(&child, source);
                return (classify_function(name.as_deref()), name);
            }
            "lexical_declaration" => {
                return classify_lexical_declaration(&child, source);
            }
            "interface_declaration" | "type_alias_declaration" => {
                let name = extract_name(&child, source);
                return (ChunkType::TypeDef, name);
            }
            _ => {}
        }
    }
    (ChunkType::Other, None)
}

fn classify_lexical_declaration(
    node: &tree_sitter::Node,
    source: &str,
) -> (ChunkType, Option<String>) {
    let mut inner = node.walk();
    for var_decl in node.children(&mut inner) {
        if var_decl.kind() != "variable_declarator" {
            continue;
        }
        let name = extract_name(&var_decl, source);
        let has_arrow = var_decl
            .children(&mut var_decl.walk())
            .any(|c| c.kind() == "arrow_function");
        if has_arrow {
            return (classify_function(name.as_deref()), name);
        }
        return (ChunkType::Other, name);
    }
    (ChunkType::Other, None)
}

fn classify_expression(node: &tree_sitter::Node, source: &str) -> (ChunkType, Option<String>) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "call_expression"
            && let Some(result) = classify_test_call(&child, source)
        {
            return result;
        }
    }
    (ChunkType::Other, None)
}

fn classify_test_call(
    call: &tree_sitter::Node,
    source: &str,
) -> Option<(ChunkType, Option<String>)> {
    let mut inner = call.walk();
    let fn_node = call.children(&mut inner).find(|c| c.kind() == "identifier")?;
    let fn_name = &source[fn_node.byte_range()];
    if !matches!(fn_name, "describe" | "it" | "test") {
        return None;
    }
    let args = call.child_by_field_name("arguments")?;
    let test_name = args
        .children(&mut args.walk())
        .find(|c| c.kind() == "string")
        .map(|s| {
            let raw = &source[s.byte_range()];
            raw.trim_matches('\'').trim_matches('"').to_string()
        });
    Some((ChunkType::TestCase, test_name))
}

#[cfg(test)]
mod tests {
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
        let source = std::iter::repeat(line.as_str())
            .take(30)
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_fallback(&source);
        assert!(chunks.len() >= 2, "expected overlapping chunks, got {}", chunks.len());
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
}
