use crate::storage::ChunkType;

use super::{
    FileChunks, ImportKind, ImportSpecifier, ParsedImport, RawChunk, attach_pending_comments,
    chunk_fallback, extract_name, make_chunk, make_parser, other_or_skip,
};

pub(super) fn chunk_rust(source: &str) -> FileChunks {
    let Some(mut parser) = make_parser(&tree_sitter_rust::LANGUAGE.into()) else {
        return FileChunks::chunks_only(chunk_fallback(source));
    };
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
        match node.kind() {
            "use_declaration" => {
                imports.push(source[node.byte_range()].to_string());
                parsed_imports.extend(parse_rust_use(&node, source));
                pending_comments.clear();
            }
            "line_comment" | "block_comment" => {
                pending_comments.push(node);
            }
            "impl_item" => {
                let impl_index = chunks.len();
                let name = extract_rust_impl_name(&node, source);
                let mut impl_chunk = make_chunk(source, &node, ChunkType::RustImpl, name);
                attach_pending_comments(&mut impl_chunk, &mut pending_comments, source);
                chunks.push(impl_chunk);
                chunks.extend(extract_impl_methods(&node, source, impl_index));
            }
            _ => {
                if let Some(mut chunk) = classify_rust_node(&node, source) {
                    attach_pending_comments(&mut chunk, &mut pending_comments, source);
                    chunks.push(chunk);
                } else {
                    pending_comments.clear();
                }
            }
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

fn classify_rust_node(node: &tree_sitter::Node, source: &str) -> Option<RawChunk> {
    let chunk_type = match node.kind() {
        "function_item" => ChunkType::RustFn,
        "struct_item" => ChunkType::RustStruct,
        "enum_item" => ChunkType::RustEnum,
        "trait_item" => ChunkType::RustTrait,
        _ => return other_or_skip(source, node),
    };
    let name = extract_name(node, source);
    Some(make_chunk(source, node, chunk_type, name))
}

fn extract_impl_methods(
    impl_node: &tree_sitter::Node,
    source: &str,
    impl_index: usize,
) -> Vec<RawChunk> {
    let Some(body) = impl_node.child_by_field_name("body") else {
        return vec![];
    };
    let mut result = Vec::new();
    let mut cursor = body.walk();
    for child in body.children(&mut cursor) {
        if child.kind() == "function_item" {
            let name = extract_name(&child, source);
            let mut chunk = make_chunk(source, &child, ChunkType::RustFn, name);
            chunk.parent_index = Some(impl_index);
            result.push(chunk);
        }
    }
    result
}

fn parse_rust_use(node: &tree_sitter::Node, source: &str) -> Vec<ParsedImport> {
    let Some(argument) = node.child_by_field_name("argument") else {
        return vec![];
    };
    match argument.kind() {
        "scoped_identifier" => parse_scoped_identifier(&argument, source)
            .into_iter()
            .collect(),
        "scoped_use_list" => parse_scoped_use_list(&argument, source),
        "use_wildcard" => parse_use_wildcard(&argument, source).into_iter().collect(),
        "use_as_clause" => parse_use_as_clause(&argument, source).into_iter().collect(),
        _ => vec![],
    }
}

fn parse_scoped_identifier(node: &tree_sitter::Node, source: &str) -> Option<ParsedImport> {
    let path_node = node.child_by_field_name("path")?;
    let name_node = node.child_by_field_name("name")?;
    let path = &source[path_node.byte_range()];
    if !is_internal_path(path) {
        return None;
    }
    Some(ParsedImport {
        source: path.to_string(),
        specifiers: vec![ImportSpecifier {
            name: source[name_node.byte_range()].to_string(),
            alias: None,
            kind: ImportKind::Named,
        }],
    })
}

fn parse_scoped_use_list(node: &tree_sitter::Node, source: &str) -> Vec<ParsedImport> {
    let (Some(path_node), Some(list_node)) = (
        node.child_by_field_name("path"),
        node.child_by_field_name("list"),
    ) else {
        return vec![];
    };
    let base_path = &source[path_node.byte_range()];
    if !is_internal_path(base_path) {
        return vec![];
    }
    collect_use_list_imports(base_path, &list_node, source)
}

fn parse_use_wildcard(node: &tree_sitter::Node, source: &str) -> Option<ParsedImport> {
    let mut cursor = node.walk();
    let path_node = node.children(&mut cursor).find(|c| c.is_named())?;
    let path = &source[path_node.byte_range()];
    if !is_internal_path(path) {
        return None;
    }
    Some(ParsedImport {
        source: path.to_string(),
        specifiers: vec![ImportSpecifier {
            name: "*".to_string(),
            alias: None,
            kind: ImportKind::Namespace,
        }],
    })
}

fn parse_use_as_clause(node: &tree_sitter::Node, source: &str) -> Option<ParsedImport> {
    let path_node = node.child_by_field_name("path")?;
    let alias_node = node.child_by_field_name("alias")?;
    if path_node.kind() == "scoped_identifier" {
        let inner_path = path_node.child_by_field_name("path")?;
        let inner_name = path_node.child_by_field_name("name")?;
        let path = &source[inner_path.byte_range()];
        if !is_internal_path(path) {
            return None;
        }
        Some(ParsedImport {
            source: path.to_string(),
            specifiers: vec![ImportSpecifier {
                name: source[inner_name.byte_range()].to_string(),
                alias: Some(source[alias_node.byte_range()].to_string()),
                kind: ImportKind::Named,
            }],
        })
    } else {
        None
    }
}

fn collect_use_list_imports(
    base_path: &str,
    list: &tree_sitter::Node,
    source: &str,
) -> Vec<ParsedImport> {
    let mut base_specifiers = Vec::new();
    let mut extra_imports = Vec::new();
    let mut cursor = list.walk();
    for child in list.children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                base_specifiers.push(ImportSpecifier {
                    name: source[child.byte_range()].to_string(),
                    alias: None,
                    kind: ImportKind::Named,
                });
            }
            "scoped_identifier" => {
                if let (Some(path_node), Some(name_node)) = (
                    child.child_by_field_name("path"),
                    child.child_by_field_name("name"),
                ) {
                    let nested_path = format!("{}::{}", base_path, &source[path_node.byte_range()]);
                    extra_imports.push(ParsedImport {
                        source: nested_path,
                        specifiers: vec![ImportSpecifier {
                            name: source[name_node.byte_range()].to_string(),
                            alias: None,
                            kind: ImportKind::Named,
                        }],
                    });
                }
            }
            "scoped_use_list" => {
                if let (Some(inner_path), Some(inner_list)) = (
                    child.child_by_field_name("path"),
                    child.child_by_field_name("list"),
                ) {
                    let nested_path =
                        format!("{}::{}", base_path, &source[inner_path.byte_range()]);
                    extra_imports.extend(collect_use_list_imports(
                        &nested_path,
                        &inner_list,
                        source,
                    ));
                }
            }
            "use_wildcard" => {
                base_specifiers.push(ImportSpecifier {
                    name: "*".to_string(),
                    alias: None,
                    kind: ImportKind::Namespace,
                });
            }
            _ => {}
        }
    }
    if !base_specifiers.is_empty() {
        extra_imports.insert(
            0,
            ParsedImport {
                source: base_path.to_string(),
                specifiers: base_specifiers,
            },
        );
    }
    extra_imports
}

fn is_internal_path(path: &str) -> bool {
    let prefix = path.split("::").next().unwrap_or("");
    matches!(prefix, "crate" | "super" | "self")
}

fn extract_rust_impl_name(node: &tree_sitter::Node, source: &str) -> Option<String> {
    let mut cursor = node.walk();
    let mut iter = node
        .children(&mut cursor)
        .filter(|c| c.kind() == "type_identifier" || c.kind() == "generic_type");
    match (iter.next(), iter.next()) {
        (None, _) => None,
        (Some(a), None) => Some(source[a.byte_range()].to_string()),
        (Some(a), Some(b)) => Some(format!(
            "{} for {}",
            &source[a.byte_range()],
            &source[b.byte_range()]
        )),
    }
}
