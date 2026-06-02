use tree_sitter::Node;

use crate::storage::ChunkType;

use super::{
    FileChunks, ImportKind, ImportSpecifier, ParsedImport, RawChunk, attach_pending_comments,
    chunk_fallback, extract_name, make_chunk, make_parser, other_or_skip,
};

fn is_internal_path(path: &str, crate_name: Option<&str>) -> bool {
    let prefix = path.split("::").next().unwrap_or("");
    matches!(prefix, "crate" | "super" | "self") || Some(prefix) == crate_name
}

fn classify_rust_node(node: &Node, source: &str) -> Option<RawChunk> {
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

fn extract_impl_methods(impl_node: &Node, source: &str, impl_index: usize) -> Vec<RawChunk> {
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

/// An [`ImportSpecifier`] with no alias — the common shape for a `use` leaf, a
/// `*` glob, or a `mod` declaration. `use x as y` builds its specifier inline
/// because it carries an alias.
fn simple_specifier(name: String, kind: ImportKind) -> ImportSpecifier {
    ImportSpecifier {
        name,
        alias: None,
        kind,
    }
}

/// A `a::b` member *inside* a `use` list (e.g. `bar::Baz` in
/// `use crate::foo::{bar::Baz, Quux}`). Unlike [`parse_scoped_identifier`], the
/// enclosing `use` was already `is_internal_path`-filtered, so this re-roots the
/// member under `base_path` without re-checking internality.
fn scoped_identifier_import(base_path: &str, node: &Node, source: &str) -> Option<ParsedImport> {
    let path_node = node.child_by_field_name("path")?;
    let name_node = node.child_by_field_name("name")?;
    let nested_path = format!("{}::{}", base_path, &source[path_node.byte_range()]);
    Some(ParsedImport {
        source: nested_path,
        specifiers: vec![simple_specifier(
            source[name_node.byte_range()].to_owned(),
            ImportKind::Named,
        )],
    })
}

/// A nested `a::{...}` group *inside* a `use` list (e.g. `bar::{Baz, Qux}` in
/// `use crate::foo::{bar::{Baz, Qux}}`). Re-roots under `base_path` and recurses
/// into [`collect_use_list_imports`]; like [`scoped_identifier_import`] it skips
/// the `is_internal_path` check that top-level [`parse_scoped_use_list`] applies.
fn scoped_use_list_imports(base_path: &str, node: &Node, source: &str) -> Vec<ParsedImport> {
    node.child_by_field_name("path")
        .zip(node.child_by_field_name("list"))
        .map(|(inner_path, inner_list)| {
            let nested_path = format!("{}::{}", base_path, &source[inner_path.byte_range()]);
            collect_use_list_imports(&nested_path, &inner_list, source)
        })
        .unwrap_or_default()
}

/// Flattens a `use` list's members into [`ParsedImport`]s. Bare names and globs
/// collapse into a single import at `base_path`; scoped members
/// ([`scoped_identifier_import`], [`scoped_use_list_imports`]) each get their own
/// re-rooted entry, ordered after the `base_path` group.
fn collect_use_list_imports(base_path: &str, list: &Node, source: &str) -> Vec<ParsedImport> {
    let mut base_specifiers = Vec::new();
    let mut extra_imports = Vec::new();
    let mut cursor = list.walk();
    for child in list.children(&mut cursor) {
        match child.kind() {
            "identifier" => base_specifiers.push(simple_specifier(
                source[child.byte_range()].to_owned(),
                ImportKind::Named,
            )),
            "use_wildcard" => {
                base_specifiers.push(simple_specifier("*".to_owned(), ImportKind::Namespace));
            }
            "scoped_identifier" => {
                extra_imports.extend(scoped_identifier_import(base_path, &child, source));
            }
            "scoped_use_list" => {
                extra_imports.extend(scoped_use_list_imports(base_path, &child, source));
            }
            _ => {}
        }
    }
    let base_group = (!base_specifiers.is_empty()).then(|| ParsedImport {
        source: base_path.to_owned(),
        specifiers: base_specifiers,
    });
    // `Option::into_iter` yields the base group (0 or 1) before the scoped extras.
    base_group.into_iter().chain(extra_imports).collect()
}

fn parse_scoped_identifier(
    node: &Node,
    source: &str,
    crate_name: Option<&str>,
) -> Option<ParsedImport> {
    let path_node = node.child_by_field_name("path")?;
    let name_node = node.child_by_field_name("name")?;
    let path = &source[path_node.byte_range()];
    if !is_internal_path(path, crate_name) {
        return None;
    }
    Some(ParsedImport {
        source: path.to_owned(),
        specifiers: vec![simple_specifier(
            source[name_node.byte_range()].to_owned(),
            ImportKind::Named,
        )],
    })
}

fn parse_scoped_use_list(node: &Node, source: &str, crate_name: Option<&str>) -> Vec<ParsedImport> {
    let (Some(path_node), Some(list_node)) = (
        node.child_by_field_name("path"),
        node.child_by_field_name("list"),
    ) else {
        return vec![];
    };
    let base_path = &source[path_node.byte_range()];
    if !is_internal_path(base_path, crate_name) {
        return vec![];
    }
    collect_use_list_imports(base_path, &list_node, source)
}

fn parse_use_wildcard(node: &Node, source: &str, crate_name: Option<&str>) -> Option<ParsedImport> {
    let mut cursor = node.walk();
    let path_node = node
        .children(&mut cursor)
        .find(tree_sitter::Node::is_named)?;
    let path = &source[path_node.byte_range()];
    if !is_internal_path(path, crate_name) {
        return None;
    }
    Some(ParsedImport {
        source: path.to_owned(),
        specifiers: vec![simple_specifier("*".to_owned(), ImportKind::Namespace)],
    })
}

fn parse_use_as_clause(
    node: &Node,
    source: &str,
    crate_name: Option<&str>,
) -> Option<ParsedImport> {
    let path_node = node.child_by_field_name("path")?;
    let alias_node = node.child_by_field_name("alias")?;
    if path_node.kind() == "scoped_identifier" {
        let inner_path = path_node.child_by_field_name("path")?;
        let inner_name = path_node.child_by_field_name("name")?;
        let path = &source[inner_path.byte_range()];
        if !is_internal_path(path, crate_name) {
            return None;
        }
        Some(ParsedImport {
            source: path.to_owned(),
            specifiers: vec![ImportSpecifier {
                name: source[inner_name.byte_range()].to_owned(),
                alias: Some(source[alias_node.byte_range()].to_owned()),
                kind: ImportKind::Named,
            }],
        })
    } else {
        None
    }
}

/// `mod foo;` (body-less) を ParsedImport として返す。
/// `mod foo { ... }` (body 有) や名前不在の場合は None を返し、
/// 呼び出し側で通常の chunk 経路に委譲する。
fn parse_mod_decl(node: &Node, source: &str) -> Option<ParsedImport> {
    if node.child_by_field_name("body").is_some() {
        return None;
    }
    let name_node = node.child_by_field_name("name")?;
    let name = source[name_node.byte_range()].to_owned();
    Some(ParsedImport {
        source: name.clone(),
        specifiers: vec![simple_specifier(name, ImportKind::ModDecl)],
    })
}

fn parse_rust_use(node: &Node, source: &str, crate_name: Option<&str>) -> Vec<ParsedImport> {
    let Some(argument) = node.child_by_field_name("argument") else {
        return vec![];
    };
    match argument.kind() {
        "scoped_identifier" => parse_scoped_identifier(&argument, source, crate_name)
            .into_iter()
            .collect(),
        "scoped_use_list" => parse_scoped_use_list(&argument, source, crate_name),
        "use_wildcard" => parse_use_wildcard(&argument, source, crate_name)
            .into_iter()
            .collect(),
        "use_as_clause" => parse_use_as_clause(&argument, source, crate_name)
            .into_iter()
            .collect(),
        _ => vec![],
    }
}

fn extract_rust_impl_name(node: &Node, source: &str) -> Option<String> {
    let mut cursor = node.walk();
    let mut iter = node
        .children(&mut cursor)
        .filter(|c| c.kind() == "type_identifier" || c.kind() == "generic_type");
    match (iter.next(), iter.next()) {
        (None, _) => None,
        (Some(a), None) => Some(source[a.byte_range()].to_owned()),
        (Some(a), Some(b)) => Some(format!(
            "{} for {}",
            &source[a.byte_range()],
            &source[b.byte_range()]
        )),
    }
}

/// Accumulates the imports and chunks discovered while walking a Rust file's
/// top-level nodes. `pending` holds doc comments awaiting the next chunk; its
/// `'a` lifetime ties the borrowed [`Node`]s to the parse tree they index.
#[derive(Debug, Default)]
struct RustChunkState<'a> {
    imports: Vec<String>,
    parsed_imports: Vec<ParsedImport>,
    chunks: Vec<RawChunk>,
    pending: Vec<Node<'a>>,
}

impl<'a> RustChunkState<'a> {
    /// Routes one top-level node to its accumulator. Comments queue in `pending`
    /// for the next chunk; every other kind clears `pending`, so a `use`/`mod`
    /// between a doc comment and its item drops the comment. Takes `&Node<'a>`
    /// (not the elided `&Node` of the sibling methods) because the comment arm
    /// stores `*node` into `pending: Vec<Node<'a>>`.
    fn add_node(&mut self, node: &Node<'a>, source: &str, crate_name: Option<&str>) {
        match node.kind() {
            "use_declaration" => {
                self.imports.push(source[node.byte_range()].to_owned());
                self.parsed_imports
                    .extend(parse_rust_use(node, source, crate_name));
                self.pending.clear();
            }
            "mod_item" => self.add_mod(node, source),
            "line_comment" | "block_comment" => self.pending.push(*node),
            "impl_item" => self.add_impl(node, source),
            _ => self.classify_or_clear(node, source),
        }
    }

    /// `mod foo;` becomes an import; `mod foo { ... }` falls through to a chunk.
    fn add_mod(&mut self, node: &Node, source: &str) {
        if let Some(import) = parse_mod_decl(node, source) {
            self.parsed_imports.push(import);
            self.pending.clear();
        } else {
            self.classify_or_clear(node, source);
        }
    }

    /// Emits the impl header chunk followed by one chunk per method, linking each
    /// method back to the header via `parent_index`.
    fn add_impl(&mut self, node: &Node, source: &str) {
        let impl_index = self.chunks.len();
        let name = extract_rust_impl_name(node, source);
        let mut impl_chunk = make_chunk(source, node, ChunkType::RustImpl, name);
        attach_pending_comments(&mut impl_chunk, &mut self.pending, source);
        self.chunks.push(impl_chunk);
        self.chunks
            .extend(extract_impl_methods(node, source, impl_index));
    }

    /// Pushes a classifiable node as a chunk (carrying any pending comments); a
    /// node that is not a chunk boundary instead discards the pending comments.
    fn classify_or_clear(&mut self, node: &Node, source: &str) {
        if let Some(mut chunk) = classify_rust_node(node, source) {
            attach_pending_comments(&mut chunk, &mut self.pending, source);
            self.chunks.push(chunk);
        } else {
            self.pending.clear();
        }
    }

    /// Consumes the accumulated state into [`FileChunks`], substituting the
    /// line-based fallback when no structural chunk was found.
    fn into_file_chunks(mut self, source: &str) -> FileChunks {
        if self.chunks.is_empty() {
            self.chunks = chunk_fallback(source);
        }
        FileChunks {
            imports: self.imports,
            parsed_imports: self.parsed_imports,
            chunks: self.chunks,
        }
    }
}

pub(super) fn chunk_rust(source: &str, crate_name: Option<&str>) -> FileChunks {
    let Some(mut parser) = make_parser(&tree_sitter_rust::LANGUAGE.into()) else {
        return FileChunks::chunks_only(chunk_fallback(source));
    };
    let Some(tree) = parser.parse(source, None) else {
        tracing::warn!("AST parse failed, using fallback chunker");
        return FileChunks::chunks_only(chunk_fallback(source));
    };
    let root = tree.root_node();
    let mut state = RustChunkState::default();
    let mut cursor = root.walk();
    for node in root.children(&mut cursor) {
        state.add_node(&node, source, crate_name);
    }
    state.into_file_chunks(source)
}
