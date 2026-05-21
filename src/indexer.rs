pub mod chunker;
mod embed;
pub mod injection;
mod source_kind;
pub mod walker;

pub use crate::storage::SourceKind;

use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::{Arc, Mutex};

use rurico::embed::EmbedError;

use crate::fs_optional;
use crate::resolver::{Resolve, Resolver};
use crate::rust_resolver::RustResolver;
use crate::storage::{self, Db, RefKind, Reference, StorageError};
use chunker::ParsedImport;

pub use embed::{EmbedResult, run_incremental_embed, run_incremental_embed_with_progress};
#[cfg(test)]
use embed::{MAX_CONSECUTIVE_EMBED_ERRORS, enrich_for_embedding, order_files_for_embedding};

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("storage error during indexing: {0}")]
    Storage(#[from] StorageError),
    #[error("embedding error: {0}")]
    Embed(#[from] EmbedError),
    #[error("IO error during indexing: {0}")]
    Io(#[from] io::Error),
    #[error("internal task failed: {0}")]
    Internal(String),
    #[error("corpus init failed: {0}")]
    CorpusInit(#[from] injection::CorpusError),
}

#[derive(Debug)]
pub struct IndexResult {
    pub files_processed: u32,
    pub chunks_created: u32,
    pub files_skipped: u32,
    pub files_errored: u32,
}

#[derive(Debug)]
pub struct DryRunResult {
    pub total_files: u32,
    pub files_to_process: u32,
    pub files_to_skip: u32,
    pub files_errored: u32,
    pub orphans_to_remove: u32,
}

struct PendingFile {
    rel_path: String,
    raw_chunks: Vec<chunker::RawChunk>,
    imports_text: String,
    parsed_imports: Vec<chunker::ParsedImport>,
    hash: String,
    mtime_epoch: Option<i64>,
    source_kind: Option<SourceKind>,
    /// One JSON array string per `raw_chunks` element. Invariant:
    /// `injection_flags.len() == raw_chunks.len()`. Clean-scan is `"[]"`,
    /// hit is `"[\"flag.id\", ...]"`. PR#2 always produces `Some` at the
    /// schema level (NewChunk); the matcher-未走行 sentinel `None` is reserved
    /// for post-PR#2 sampling scenarios.
    injection_flags: Vec<String>,
}

impl PendingFile {
    fn to_new_chunks(&self) -> Vec<storage::NewChunk<'_>> {
        self.raw_chunks
            .iter()
            .zip(self.injection_flags.iter())
            .map(|(c, flags)| storage::NewChunk {
                chunk_type: &c.chunk_type,
                name: c.name.as_deref(),
                content: &c.content,
                start_line: c.start_line,
                end_line: c.end_line,
                parent_index: c.parent_index,
                source_kind: self.source_kind,
                injection_flags: Some(flags.as_str()),
            })
            .collect()
    }
}

const MAX_FILE_SIZE: u64 = 1_000_000;

enum FileAction {
    Skip,
    Error,
}

fn to_rel_path(root: &Path, file_path: &Path) -> String {
    let rel = match file_path.strip_prefix(root) {
        Ok(r) => r,
        Err(_) => {
            tracing::warn!(
                root = %root.display(),
                path = %file_path.display(),
                "Path is not under project root, using absolute path"
            );
            file_path
        }
    };
    match rel.to_str() {
        Some(s) => s.to_owned(),
        None => {
            tracing::warn!(path = %rel.display(), "Non-UTF-8 path, using lossy conversion");
            rel.to_string_lossy().into_owned()
        }
    }
}

fn read_source(file_path: &Path) -> Result<String, FileAction> {
    let metadata = fs::metadata(file_path).map_err(|e| {
        tracing::warn!(file = %file_path.display(), error = %e, "IO error, skipping file");
        FileAction::Error
    })?;
    if metadata.len() > MAX_FILE_SIZE {
        tracing::warn!(file = %file_path.display(), size = metadata.len(), "Skipped (too large)");
        return Err(FileAction::Skip);
    }
    fs::read_to_string(file_path).map_err(|e| {
        tracing::warn!(file = %file_path.display(), error = %e, "Read error, skipping file");
        FileAction::Error
    })
}

struct CheckedFile {
    rel_path: String,
    source: String,
    hash: String,
}

enum CheckResult {
    Changed(CheckedFile),
    Skip,
    Error,
}

fn file_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn check_file(
    conn: &Db,
    rel_path: String,
    file_path: &Path,
    force: bool,
) -> Result<CheckResult, IndexError> {
    let source = match read_source(file_path) {
        Ok(s) => s,
        Err(FileAction::Skip) => return Ok(CheckResult::Skip),
        Err(_) => return Ok(CheckResult::Error),
    };
    let hash = file_hash(&source);

    if !force && !storage::should_reindex(conn, &rel_path, &hash)? {
        tracing::debug!(file = %rel_path, "Skipped (unchanged)");
        return Ok(CheckResult::Skip);
    }

    Ok(CheckResult::Changed(CheckedFile {
        rel_path,
        source,
        hash,
    }))
}

fn prepare_chunks(
    checked: CheckedFile,
    file_path: &Path,
    crate_name: Option<&str>,
    corpus: &injection::Corpus,
) -> Option<PendingFile> {
    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let file_chunks = chunker::chunk_file_with_crate_name(&checked.source, ext, crate_name);
    if file_chunks.chunks.is_empty() {
        tracing::debug!(file = %checked.rel_path, "Skipped (no chunks)");
        return None;
    }

    let mtime_epoch = fs_optional::read_mtime_epoch(file_path);

    let imports_text = file_chunks.imports.join("\n");
    let source_kind = Some(source_kind::classify(&checked.rel_path));
    let injection_flags: Vec<String> = file_chunks
        .chunks
        .iter()
        .map(|c| {
            let flags = corpus.check_chunk(&c.content);
            if flags.is_empty() {
                "[]".to_owned()
            } else {
                serde_json::to_string(&flags)
                    .expect("serde_json::to_string on Vec<String> is infallible")
            }
        })
        .collect();

    Some(PendingFile {
        rel_path: checked.rel_path,
        raw_chunks: file_chunks.chunks,
        imports_text,
        parsed_imports: file_chunks.parsed_imports,
        hash: checked.hash,
        mtime_epoch,
        source_kind,
        injection_flags,
    })
}

enum FileOutcome {
    Processed(u32),
    Skipped,
    Errored,
}

#[allow(clippy::cast_possible_truncation)]
fn process_file(
    conn: &Db,
    rel_path: String,
    file_path: &Path,
    force: bool,
    resolver: &Resolver,
    rust_resolver: &RustResolver,
    corpus: &injection::Corpus,
) -> Result<FileOutcome, IndexError> {
    let checked = match check_file(conn, rel_path, file_path, force)? {
        CheckResult::Changed(c) => c,
        CheckResult::Skip => return Ok(FileOutcome::Skipped),
        CheckResult::Error => return Ok(FileOutcome::Errored),
    };
    let Some(pf) = prepare_chunks(checked, file_path, rust_resolver.crate_name(), corpus) else {
        return Ok(FileOutcome::Skipped);
    };
    let n = pf.raw_chunks.len() as u32;
    let new_chunks = pf.to_new_chunks();
    let refs = if pf.rel_path.ends_with(".rs") {
        build_references(&pf.parsed_imports, &pf.rel_path, rust_resolver)
    } else {
        build_references(&pf.parsed_imports, &pf.rel_path, resolver)
    };
    let data = storage::FileData {
        file_path: &pf.rel_path,
        chunks: &new_chunks,
        file_hash: &pf.hash,
        imports_text: &pf.imports_text,
        refs: &refs,
        mtime_epoch: pf.mtime_epoch,
    };
    storage::replace_file_data(conn, &data, None)?;
    Ok(FileOutcome::Processed(n))
}

fn remove_orphans(
    conn: &Arc<Mutex<Db>>,
    current_rel_paths: &HashSet<String>,
) -> Result<(), IndexError> {
    let conn = conn.lock().unwrap();
    let indexed_paths = storage::get_all_file_paths(&conn)?;

    let orphans: Vec<_> = indexed_paths
        .difference(current_rel_paths)
        .cloned()
        .collect();
    if orphans.is_empty() {
        return Ok(());
    }

    let tx = conn.unchecked_transaction().map_err(StorageError::from)?;
    for orphan in &orphans {
        storage::delete_file_chunks_in(&tx, orphan)?;
    }
    tx.commit().map_err(StorageError::from)?;
    tracing::info!(removed = orphans.len(), "Removed orphaned file chunks");
    Ok(())
}

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
    import: &ParsedImport,
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
    import: &ParsedImport,
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

fn build_references(
    imports: &[ParsedImport],
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

#[allow(clippy::cast_possible_truncation)]
pub fn dry_run_index(
    conn: &Arc<Mutex<Db>>,
    root: &Path,
    force: bool,
    exclude_vendor: bool,
) -> Result<DryRunResult, IndexError> {
    let files = walker::walk_source_files(root, exclude_vendor);
    let total_files = files.len() as u32;
    let mut files_to_process = 0u32;
    let mut files_to_skip = 0u32;
    let mut files_errored = 0u32;
    let mut current_rel_paths = HashSet::with_capacity(files.len());

    let conn_guard = conn.lock().unwrap();
    for file_path in &files {
        let rel_path = to_rel_path(root, file_path);
        current_rel_paths.insert(rel_path.clone());
        match check_file(&conn_guard, rel_path, file_path, force) {
            Ok(CheckResult::Changed(_)) => files_to_process += 1,
            Ok(CheckResult::Skip) => files_to_skip += 1,
            Ok(CheckResult::Error) => files_errored += 1,
            Err(e) => {
                tracing::warn!(file = %file_path.display(), error = %e, "check_file failed during dry run");
                files_errored += 1;
            }
        }
    }

    let indexed_paths = storage::get_all_file_paths(&conn_guard)?;
    let orphans_to_remove = indexed_paths.difference(&current_rel_paths).count() as u32;

    Ok(DryRunResult {
        total_files,
        files_to_process,
        files_to_skip,
        files_errored,
        orphans_to_remove,
    })
}

fn run_chunk_only_index_inner(
    conn: &Arc<Mutex<Db>>,
    root: &Path,
    force: bool,
    exclude_vendor: bool,
) -> Result<IndexResult, IndexError> {
    const CORPUS_YAML: &str = include_str!("../tests/fixtures/injection/corpus.yaml");
    let corpus = injection::Corpus::load_from_str(CORPUS_YAML)?;

    let files = walker::walk_source_files(root, exclude_vendor);
    tracing::info!(
        file_count = files.len(),
        force,
        "Starting chunk-only indexing"
    );

    let resolver = Resolver::new(root);
    let rust_resolver = RustResolver::new(root);

    let (files_processed, chunks_created, files_skipped, files_errored, current_rel_paths) = {
        let mut files_processed = 0u32;
        let mut chunks_created = 0u32;
        let mut files_skipped = 0u32;
        let mut files_errored = 0u32;
        let mut current_rel_paths = HashSet::with_capacity(files.len());

        let conn_guard = conn.lock().unwrap();
        let _automerge = storage::FtsAutomergeGuard::new(&conn_guard)?;

        for file_path in &files {
            let rel_path = to_rel_path(root, file_path);
            current_rel_paths.insert(rel_path.clone());
            match process_file(
                &conn_guard,
                rel_path,
                file_path,
                force,
                &resolver,
                &rust_resolver,
                &corpus,
            )? {
                FileOutcome::Processed(n) => {
                    chunks_created += n;
                    files_processed += 1;
                }
                FileOutcome::Skipped => files_skipped += 1,
                FileOutcome::Errored => files_errored += 1,
            }
        }

        if files_processed > 0 {
            storage::fts_optimize(&conn_guard)?;
        }

        (
            files_processed,
            chunks_created,
            files_skipped,
            files_errored,
            current_rel_paths,
        )
    };

    remove_orphans(conn, &current_rel_paths)?;

    tracing::info!(
        files_processed,
        chunks_created,
        files_skipped,
        files_errored,
        "Chunk-only indexing complete"
    );

    Ok(IndexResult {
        files_processed,
        chunks_created,
        files_skipped,
        files_errored,
    })
}

pub fn run_chunk_only_index(
    conn: &Arc<Mutex<Db>>,
    root: &Path,
    exclude_vendor: bool,
) -> Result<IndexResult, IndexError> {
    run_chunk_only_index_inner(conn, root, false, exclude_vendor)
}

pub fn run_chunk_only_index_force(
    conn: &Arc<Mutex<Db>>,
    root: &Path,
    exclude_vendor: bool,
) -> Result<IndexResult, IndexError> {
    run_chunk_only_index_inner(conn, root, true, exclude_vendor)
}

#[cfg(test)]
mod tests;
