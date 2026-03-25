use crate::storage::ChunkType;

use super::{chunk_fallback, chunk_with_ast, extract_name, make_chunk, make_parser, other_or_skip, RawChunk};

pub(super) fn chunk_rust(source: &str) -> Vec<RawChunk> {
    let Some(mut parser) = make_parser(&tree_sitter_rust::LANGUAGE.into()) else {
        return chunk_fallback(source);
    };
    chunk_with_ast(source, &mut parser, classify_rust_node)
}

fn classify_rust_node(node: &tree_sitter::Node, source: &str) -> Option<RawChunk> {
    let chunk_type = match node.kind() {
        "function_item" => ChunkType::RustFn,
        "struct_item" => ChunkType::RustStruct,
        "enum_item" => ChunkType::RustEnum,
        "trait_item" => ChunkType::RustTrait,
        "impl_item" => {
            let name = extract_rust_impl_name(node, source);
            return Some(make_chunk(source, node, ChunkType::RustImpl, name));
        }
        "use_declaration" | "line_comment" | "block_comment" => return None,
        _ => return other_or_skip(source, node),
    };
    let name = extract_name(node, source);
    Some(make_chunk(source, node, chunk_type, name))
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