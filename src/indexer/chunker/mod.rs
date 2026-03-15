use crate::storage::ChunkType;

// ── Import structure types (FR-001) ──
// Used by resolver (Phase 2) and indexer integration (Phase 4).

#[derive(Debug, Clone, PartialEq)]
pub enum ImportKind {
    Named,
    Default,
    Namespace,
    TypeOnly,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportSpecifier {
    pub name: String,
    pub alias: Option<String>,
    pub kind: ImportKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedImport {
    pub specifiers: Vec<ImportSpecifier>,
    pub source: String,
}

// ── Re-export types (FR-004) ──

/// Re-export entry in a barrel file.
#[derive(Debug, Clone, PartialEq)]
pub struct ReExport {
    pub symbol_name: Option<String>,
    pub source: String,
}

/// Parse re-export statements from source code.
/// Handles `export { X } from './Y'` and `export * from './Y'`.
pub fn parse_reexports(source: &str, extension: &str) -> Vec<ReExport> {
    let lang = match extension {
        "tsx" | "ts" | "jsx" | "js" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        _ => return vec![],
    };
    let Some(mut parser) = make_parser(&lang) else {
        return vec![];
    };
    let tree = match parser.parse(source, None) {
        Some(t) => t,
        None => return vec![],
    };
    extract_reexports(&tree.root_node(), source)
}

fn extract_specifier_name(spec: &tree_sitter::Node, source: &str) -> Option<String> {
    spec.child_by_field_name("name")
        .or_else(|| find_child_by_kind(spec, "identifier"))
        .map(|n| source[n.byte_range()].to_string())
}

fn extract_reexports(root: &tree_sitter::Node, source: &str) -> Vec<ReExport> {
    let mut reexports = Vec::new();
    let mut cursor = root.walk();
    for child in root.children(&mut cursor) {
        if child.kind() != "export_statement" {
            continue;
        }
        // Must have a `from` source (re-export, not local export)
        let source_str = match extract_export_source(&child, source) {
            Some(s) => s,
            None => continue,
        };
        // Check for `export *` (namespace re-export)
        if has_star_export(&child) {
            reexports.push(ReExport {
                symbol_name: None,
                source: source_str,
            });
            continue;
        }
        // Check for `export { X, Y }` (named re-exports)
        if let Some(clause) = find_child_by_kind(&child, "export_clause") {
            let mut clause_cursor = clause.walk();
            for spec in clause.children(&mut clause_cursor) {
                if spec.kind() == "export_specifier"
                    && let Some(name) = extract_specifier_name(&spec, source)
                {
                    reexports.push(ReExport {
                        symbol_name: Some(name),
                        source: source_str.clone(),
                    });
                }
            }
        }
    }
    reexports
}

fn extract_export_source(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let string_node = find_child_by_kind(node, "string")?;
    find_child_by_kind(&string_node, "string_fragment")
        .map(|f| source[f.byte_range()].to_string())
        .or_else(|| {
            let text = &source[string_node.byte_range()];
            Some(text.trim_matches('\'').trim_matches('"').to_string())
        })
}

fn has_star_export(node: &tree_sitter::Node) -> bool {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .any(|c| c.kind() == "*" || (c.kind() == "namespace_export"))
}

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
    /// Structured import data for reference graph construction (JS/TS only).
    pub parsed_imports: Vec<ParsedImport>,
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
        "css" => FileChunks {
            imports: vec![],
            parsed_imports: vec![],
            chunks: chunk_css(source),
        },
        "html" => FileChunks {
            imports: vec![],
            parsed_imports: vec![],
            chunks: chunk_html(source),
        },
        _ => FileChunks {
            imports: vec![],
            parsed_imports: vec![],
            chunks: chunk_fallback(source),
        },
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
        return FileChunks {
            imports: vec![],
            parsed_imports: vec![],
            chunks: chunk_fallback(source),
        };
    };
    chunk_js_like_with_imports(source, &mut parser)
}

fn chunk_ts(source: &str) -> FileChunks {
    let Some(mut parser) = make_parser(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()) else {
        return FileChunks {
            imports: vec![],
            parsed_imports: vec![],
            chunks: chunk_fallback(source),
        };
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
        return FileChunks {
            imports: vec![],
            parsed_imports: vec![],
            chunks: chunk_fallback(source),
        };
    };
    let root = tree.root_node();
    let imports = extract_imports_from_ast(&root, source);
    let parsed_imports = extract_parsed_imports(&root, source);
    let mut cursor = root.walk();
    let chunks: Vec<RawChunk> = root
        .children(&mut cursor)
        .filter_map(|node| classify_js_node(&node, source))
        .collect();
    if chunks.is_empty() {
        return FileChunks {
            imports,
            parsed_imports,
            chunks: chunk_fallback(source),
        };
    }
    FileChunks {
        imports,
        parsed_imports,
        chunks,
    }
}

fn extract_parsed_imports(root: &tree_sitter::Node, source: &str) -> Vec<ParsedImport> {
    let mut cursor = root.walk();
    root.children(&mut cursor)
        .filter(|node| node.kind() == "import_statement")
        .filter_map(|node| parse_single_import(&node, source))
        .collect()
}

/// Parse structured import information from source code using tree-sitter.
///
/// Returns a list of parsed imports with their specifiers and source paths.
/// Returns empty Vec if no imports found or if the file type doesn't support imports.
#[cfg(test)]
pub fn parse_structured_imports(source: &str, extension: &str) -> Vec<ParsedImport> {
    match extension {
        "tsx" | "jsx" => {
            let Some(mut parser) = make_parser(&tree_sitter_typescript::LANGUAGE_TSX.into()) else {
                return Vec::new();
            };
            parse_imports_from_ast(source, &mut parser)
        }
        "ts" | "js" | "mjs" => {
            let Some(mut parser) = make_parser(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
            else {
                return Vec::new();
            };
            parse_imports_from_ast(source, &mut parser)
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
fn parse_imports_from_ast(source: &str, parser: &mut tree_sitter::Parser) -> Vec<ParsedImport> {
    let Some(tree) = parser.parse(source, None) else {
        return Vec::new();
    };
    let root = tree.root_node();
    let mut cursor = root.walk();
    root.children(&mut cursor)
        .filter(|node| node.kind() == "import_statement")
        .filter_map(|node| parse_single_import(&node, source))
        .collect()
}

fn parse_single_import(node: &tree_sitter::Node, source: &str) -> Option<ParsedImport> {
    let source_path = extract_import_source(node, source)?;
    let is_type_import = has_type_keyword(node);
    let specifiers = match find_child_by_kind(node, "import_clause") {
        Some(clause) => parse_import_clause(&clause, source, is_type_import),
        None => Vec::new(), // side-effect import
    };
    Some(ParsedImport {
        specifiers,
        source: source_path,
    })
}

fn extract_import_source(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let string_node = find_child_by_kind(node, "string")?;
    let fragment = find_child_by_kind(&string_node, "string_fragment")?;
    Some(source[fragment.byte_range()].to_string())
}

fn has_type_keyword(node: &tree_sitter::Node) -> bool {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .any(|child| !child.is_named() && child.kind() == "type")
}

fn parse_import_clause(
    clause: &tree_sitter::Node,
    source: &str,
    is_type_import: bool,
) -> Vec<ImportSpecifier> {
    let mut specifiers = Vec::new();
    let mut cursor = clause.walk();
    for child in clause.children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                // default import: `import X from ...`
                let name = source[child.byte_range()].to_string();
                let kind = if is_type_import {
                    ImportKind::TypeOnly
                } else {
                    ImportKind::Default
                };
                specifiers.push(ImportSpecifier {
                    name,
                    alias: None,
                    kind,
                });
            }
            "named_imports" => {
                parse_named_imports_node(&child, source, is_type_import, &mut specifiers);
            }
            "namespace_import" => {
                let alias = find_child_by_kind(&child, "identifier")
                    .map(|id| source[id.byte_range()].to_string());
                specifiers.push(ImportSpecifier {
                    name: "*".to_string(),
                    alias,
                    kind: ImportKind::Namespace,
                });
            }
            _ => {}
        }
    }
    specifiers
}

fn parse_named_imports_node(
    node: &tree_sitter::Node,
    source: &str,
    is_type_import: bool,
    specifiers: &mut Vec<ImportSpecifier>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "import_specifier"
            && let Some(spec) = parse_import_specifier(&child, source, is_type_import)
        {
            specifiers.push(spec);
        }
    }
}

fn parse_import_specifier(
    node: &tree_sitter::Node,
    source: &str,
    is_type_import: bool,
) -> Option<ImportSpecifier> {
    let has_inline_type = has_type_keyword(node);
    let mut identifiers: Vec<String> = Vec::new();
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "identifier" {
            identifiers.push(source[child.byte_range()].to_string());
        }
    }
    let (name, alias) = match identifiers.len() {
        0 => return None,
        1 => (identifiers.remove(0), None),
        _ => {
            let name = identifiers.remove(0);
            let alias = Some(identifiers.remove(0));
            (name, alias)
        }
    };
    let kind = if is_type_import || has_inline_type {
        ImportKind::TypeOnly
    } else {
        ImportKind::Named
    };
    Some(ImportSpecifier { name, alias, kind })
}

fn find_child_by_kind<'a>(
    node: &'a tree_sitter::Node<'a>,
    kind: &str,
) -> Option<tree_sitter::Node<'a>> {
    let mut cursor = node.walk();
    node.children(&mut cursor).find(|c| c.kind() == kind)
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
        Some(n) if n.starts_with("use") && n.len() > 3 && n.as_bytes()[3].is_ascii_uppercase() => {
            ChunkType::Hook
        }
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
    let fn_node = call
        .children(&mut inner)
        .find(|c| c.kind() == "identifier")?;
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
mod tests;
