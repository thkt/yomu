use crate::resolver::Resolve;
use crate::storage::{RefKind, Reference};

use super::chunker;

fn import_kind_to_ref_kind(kind: &chunker::ImportKind) -> RefKind {
    match kind {
        chunker::ImportKind::Named => RefKind::Named,
        chunker::ImportKind::Default => RefKind::Default,
        chunker::ImportKind::Namespace => RefKind::Namespace,
        chunker::ImportKind::TypeOnly => RefKind::TypeOnly,
        chunker::ImportKind::ModDecl => RefKind::ModDecl,
    }
}

fn resolve_target(
    import: &chunker::ParsedImport,
    source_path: &str,
    resolver: &impl Resolve,
) -> Option<String> {
    let is_mod_decl = import
        .specifiers
        .first()
        .is_some_and(|s| s.kind == chunker::ImportKind::ModDecl);
    if is_mod_decl {
        resolver.resolve_mod_decl(&import.source, source_path)
    } else {
        resolver.resolve(&import.source, source_path)
    }
}

fn import_to_references(
    import: &chunker::ParsedImport,
    source_path: &str,
    target: String,
) -> Vec<Reference> {
    if import.specifiers.is_empty() {
        return vec![Reference {
            source_file: source_path.to_owned(),
            target_file: target,
            symbol_name: None,
            ref_kind: RefKind::SideEffect,
        }];
    }
    import
        .specifiers
        .iter()
        .map(|s| Reference {
            source_file: source_path.to_owned(),
            target_file: target.clone(),
            symbol_name: Some(s.name.clone()),
            ref_kind: import_kind_to_ref_kind(&s.kind),
        })
        .collect()
}

pub(in crate::indexer) fn build_references(
    imports: &[chunker::ParsedImport],
    source_path: &str,
    resolver: &impl Resolve,
) -> Vec<Reference> {
    imports
        .iter()
        .filter_map(|import| {
            let target = resolve_target(import, source_path, resolver)?;
            Some(import_to_references(import, source_path, target))
        })
        .flatten()
        .collect()
}
