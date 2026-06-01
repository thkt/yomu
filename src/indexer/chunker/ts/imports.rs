//! TypeScript/JavaScript import-statement parsing: walks an `import_statement`
//! AST node into a [`ParsedImport`] (source path + [`ImportSpecifier`] list),
//! handling default, named, namespace, and inline/statement `type` imports.

use tree_sitter::Node;

use super::super::{ImportKind, ImportSpecifier, ParsedImport, find_child_by_kind};

fn has_type_keyword(node: &Node) -> bool {
    let mut cursor = node.walk();
    node.children(&mut cursor)
        .any(|child| !child.is_named() && child.kind() == "type")
}

fn extract_import_source(node: &Node, source: &str) -> Option<String> {
    let string_node = find_child_by_kind(node, "string")?;
    let fragment = find_child_by_kind(&string_node, "string_fragment")?;
    Some(source[fragment.byte_range()].to_owned())
}

fn parse_import_specifier(
    node: &Node,
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

fn parse_named_imports_node(
    node: &Node,
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

fn parse_import_clause(clause: &Node, source: &str, is_type_import: bool) -> Vec<ImportSpecifier> {
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
                    name: "*".to_owned(),
                    alias,
                    kind: ImportKind::Namespace,
                });
            }
            _ => {}
        }
    }
    specifiers
}

pub(super) fn parse_single_import(node: &Node, source: &str) -> Option<ParsedImport> {
    let source_path = extract_import_source(node, source)?;
    let is_type_import = has_type_keyword(node);
    let specifiers = find_child_by_kind(node, "import_clause").map_or_else(Vec::new, |clause| {
        parse_import_clause(&clause, source, is_type_import)
    });
    Some(ParsedImport {
        specifiers,
        source: source_path,
    })
}
