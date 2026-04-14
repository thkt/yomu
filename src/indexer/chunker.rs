mod css_html;
mod markdown;
mod rust;
mod ts;

use crate::storage::ChunkType;

pub use ts::parse_reexports;

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

#[derive(Debug, Clone, PartialEq)]
pub struct ReExport {
    pub symbol_name: Option<String>,
    pub source: String,
}

#[derive(Debug, Clone)]
pub struct RawChunk {
    pub chunk_type: ChunkType,
    pub name: Option<String>,
    pub content: String,
    pub start_line: u32,
    pub end_line: u32,
    pub parent_index: Option<usize>,
    /// AST node's original start line before comment attachment shifts start_line.
    /// Used by pass-2 subchunk extraction to match chunks back to AST nodes.
    pub ast_start_line: u32,
}

#[derive(Debug, Clone)]
pub struct FileChunks {
    pub imports: Vec<String>,
    pub parsed_imports: Vec<ParsedImport>,
    pub chunks: Vec<RawChunk>,
}

impl FileChunks {
    pub(super) fn chunks_only(chunks: Vec<RawChunk>) -> Self {
        Self {
            imports: vec![],
            parsed_imports: vec![],
            chunks,
        }
    }
}

pub fn chunk_file(source: &str, extension: &str) -> FileChunks {
    match extension {
        "tsx" | "jsx" => ts::chunk_tsx(source),
        "ts" | "js" | "mjs" => ts::chunk_ts(source),
        "rs" => rust::chunk_rust(source),
        "css" => FileChunks::chunks_only(css_html::chunk_css(source)),
        "html" => FileChunks::chunks_only(css_html::chunk_html(source)),
        "md" => FileChunks::chunks_only(markdown::chunk_markdown(source)),
        _ => FileChunks::chunks_only(chunk_fallback(source)),
    }
}

#[cfg(test)]
pub(crate) use ts::parse_structured_imports;

fn make_parser(lang: &tree_sitter::Language) -> Option<tree_sitter::Parser> {
    let mut parser = tree_sitter::Parser::new();
    if let Err(e) = parser.set_language(lang) {
        tracing::warn!(error = %e, "Failed to set tree-sitter language, using fallback chunker");
        return None;
    }
    Some(parser)
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
            let s = (start_idx + 1) as u32;
            chunks.push(RawChunk {
                chunk_type: ChunkType::Other,
                name: None,
                content,
                parent_index: None,
                start_line: s,
                end_line: end_idx as u32,
                ast_start_line: s,
            });
        }

        const OVERLAP_LINES: usize = 4;
        if end_idx >= lines.len() {
            break;
        }
        start_idx = end_idx.saturating_sub(OVERLAP_LINES);
    }

    chunks
}

pub(super) fn attach_pending_comments(
    chunk: &mut RawChunk,
    pending_comments: &mut Vec<tree_sitter::Node>,
    source: &str,
) {
    if pending_comments.is_empty() {
        return;
    }
    let comment_text: String = pending_comments
        .iter()
        .map(|c| &source[c.byte_range()])
        .collect::<Vec<_>>()
        .join("\n");
    chunk.start_line = pending_comments[0].start_position().row as u32 + 1;
    chunk.content = format!("{comment_text}\n{}", chunk.content);
    pending_comments.clear();
}

fn other_or_skip(source: &str, node: &tree_sitter::Node) -> Option<RawChunk> {
    let text = &source[node.byte_range()];
    if text.trim().is_empty() {
        None
    } else {
        Some(make_chunk(source, node, ChunkType::Other, None))
    }
}

fn make_chunk(
    source: &str,
    node: &tree_sitter::Node,
    chunk_type: ChunkType,
    name: Option<String>,
) -> RawChunk {
    let start = node.start_position().row as u32 + 1;
    RawChunk {
        chunk_type,
        name,
        content: source[node.byte_range()].to_string(),
        start_line: start,
        end_line: node.end_position().row as u32 + 1,
        parent_index: None,
        ast_start_line: start,
    }
}

/// Minimum line count for a Component chunk to trigger subchunk extraction.
const SUBCHUNK_THRESHOLD: u32 = 50;

/// Hook names whose callback arguments should be extracted as subchunks.
const HOOK_CALLBACK_NAMES: &[&str] = &["useEffect", "useMemo", "useCallback"];

/// Extract direct inner functions from a Component chunk's AST node.
/// Returns subchunks with `parent_index` pointing to the parent's position.
pub(super) fn extract_inner_functions(
    source: &str,
    body_node: &tree_sitter::Node,
    parent_index: usize,
) -> Vec<RawChunk> {
    let mut subchunks = Vec::new();
    let mut cursor = body_node.walk();

    for stmt in body_node.children(&mut cursor) {
        match stmt.kind() {
            // const handleClick = () => { ... };
            // const cached = useMemo(() => { ... }, [dep]);
            "lexical_declaration" => {
                if let Some(sub) = extract_arrow_from_lexical(source, &stmt, parent_index) {
                    subchunks.push(sub);
                } else if let Some(sub) = extract_hook_from_lexical(source, &stmt, parent_index) {
                    subchunks.push(sub);
                }
            }
            // function handleSubmit() { ... }
            "function_declaration" => {
                let name = extract_name(&stmt, source);
                let s = stmt.start_position().row as u32 + 1;
                subchunks.push(RawChunk {
                    chunk_type: ChunkType::InnerFn,
                    name,
                    content: source[stmt.byte_range()].to_string(),
                    start_line: s,
                    end_line: stmt.end_position().row as u32 + 1,
                    parent_index: Some(parent_index),
                    ast_start_line: s,
                });
            }
            // useEffect(() => { ... }, []);
            "expression_statement" => {
                if let Some(sub) = extract_hook_callback(source, &stmt, parent_index) {
                    subchunks.push(sub);
                }
            }
            _ => {}
        }
    }
    subchunks
}

fn extract_arrow_from_lexical(
    source: &str,
    node: &tree_sitter::Node,
    parent_index: usize,
) -> Option<RawChunk> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() != "variable_declarator" {
            continue;
        }
        let has_arrow = child
            .children(&mut child.walk())
            .any(|c| c.kind() == "arrow_function");
        if has_arrow {
            let name = extract_name(&child, source);
            let s = node.start_position().row as u32 + 1;
            return Some(RawChunk {
                chunk_type: ChunkType::InnerFn,
                name,
                content: source[node.byte_range()].to_string(),
                start_line: s,
                end_line: node.end_position().row as u32 + 1,
                parent_index: Some(parent_index),
                ast_start_line: s,
            });
        }
    }
    None
}

fn extract_hook_from_lexical(
    source: &str,
    node: &tree_sitter::Node,
    parent_index: usize,
) -> Option<RawChunk> {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() != "variable_declarator" {
            continue;
        }
        // Look for call_expression as the value: const x = useMemo(...)
        let mut c2 = child.walk();
        for grandchild in child.children(&mut c2) {
            if grandchild.kind() == "call_expression" {
                let callee = grandchild.children(&mut grandchild.walk()).next()?;
                let callee_name = &source[callee.byte_range()];
                if HOOK_CALLBACK_NAMES.contains(&callee_name) {
                    let s = node.start_position().row as u32 + 1;
                    return Some(RawChunk {
                        chunk_type: ChunkType::InnerFn,
                        name: Some(callee_name.to_string()),
                        content: source[node.byte_range()].to_string(),
                        start_line: s,
                        end_line: node.end_position().row as u32 + 1,
                        ast_start_line: s,
                        parent_index: Some(parent_index),
                    });
                }
            }
        }
    }
    None
}

fn extract_hook_callback(
    source: &str,
    stmt: &tree_sitter::Node,
    parent_index: usize,
) -> Option<RawChunk> {
    let mut cursor = stmt.walk();
    for child in stmt.children(&mut cursor) {
        if child.kind() != "call_expression" {
            continue;
        }
        let callee = child.children(&mut child.walk()).next()?;
        let callee_name = &source[callee.byte_range()];
        if HOOK_CALLBACK_NAMES.contains(&callee_name) {
            let s = stmt.start_position().row as u32 + 1;
            return Some(RawChunk {
                chunk_type: ChunkType::InnerFn,
                name: Some(callee_name.to_string()),
                content: source[stmt.byte_range()].to_string(),
                start_line: s,
                end_line: stmt.end_position().row as u32 + 1,
                parent_index: Some(parent_index),
                ast_start_line: s,
            });
        }
    }
    None
}

pub(super) fn should_extract_subchunks(chunk: &RawChunk) -> bool {
    chunk.chunk_type == ChunkType::Component
        && (chunk.end_line - chunk.start_line + 1) > SUBCHUNK_THRESHOLD
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

fn find_child_by_kind<'a>(
    node: &'a tree_sitter::Node<'a>,
    kind: &str,
) -> Option<tree_sitter::Node<'a>> {
    let mut cursor = node.walk();
    node.children(&mut cursor).find(|c| c.kind() == kind)
}

#[cfg(test)]
mod tests;
