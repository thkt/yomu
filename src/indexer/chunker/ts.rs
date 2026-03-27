use crate::storage::ChunkType;

use super::{
    FileChunks, ImportKind, ImportSpecifier, ParsedImport, RawChunk, ReExport,
    attach_pending_comments, chunk_fallback, classify_function, extract_name, find_child_by_kind,
    make_chunk, make_parser, other_or_skip,
};

pub(super) fn chunk_tsx(source: &str) -> FileChunks {
    let Some(mut parser) = make_parser(&tree_sitter_typescript::LANGUAGE_TSX.into()) else {
        return FileChunks::chunks_only(chunk_fallback(source));
    };
    chunk_js_like_with_imports(source, &mut parser)
}

pub(super) fn chunk_ts(source: &str) -> FileChunks {
    let Some(mut parser) = make_parser(&tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()) else {
        return FileChunks::chunks_only(chunk_fallback(source));
    };
    chunk_js_like_with_imports(source, &mut parser)
}

fn chunk_js_like_with_imports(source: &str, parser: &mut tree_sitter::Parser) -> FileChunks {
    let Some(tree) = parser.parse(source, None) else {
        tracing::warn!("AST parse failed, using fallback chunker");
        return FileChunks::chunks_only(chunk_fallback(source));
    };
    let root = tree.root_node();
    let mut imports = Vec::new();
    let mut parsed_imports = Vec::new();
    let mut chunks = Vec::new();
    let mut pending_comments: Vec<tree_sitter::Node> = Vec::new();
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        if node.kind() == "import_statement" {
            imports.push(source[node.byte_range()].to_string());
            if let Some(pi) = parse_single_import(&node, source) {
                parsed_imports.push(pi);
            }
            pending_comments.clear();
        } else if node.kind() == "comment" {
            pending_comments.push(node);
        } else if let Some(mut chunk) = classify_js_node(&node, source) {
            attach_pending_comments(&mut chunk, &mut pending_comments, source);
            chunks.push(chunk);
        } else {
            pending_comments.clear();
        }
    }
    if chunks.is_empty() {
        chunks = chunk_fallback(source);
    }
    FileChunks {
        imports,
        parsed_imports,
        chunks,
    }
}

#[cfg(test)]
pub(crate) fn parse_structured_imports(source: &str, extension: &str) -> Vec<ParsedImport> {
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
        _ => other_or_skip(source, node),
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
        let source_str = match extract_export_source(&child, source) {
            Some(s) => s,
            None => continue,
        };
        if has_star_export(&child) {
            reexports.push(ReExport {
                symbol_name: None,
                source: source_str,
            });
            continue;
        }
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
