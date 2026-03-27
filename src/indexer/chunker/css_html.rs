use crate::storage::ChunkType;

use super::{RawChunk, chunk_fallback, chunk_with_ast, make_chunk, make_parser, other_or_skip};

pub(super) fn chunk_css(source: &str) -> Vec<RawChunk> {
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
        _ => other_or_skip(source, node),
    }
}

pub(super) fn chunk_html(source: &str) -> Vec<RawChunk> {
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
