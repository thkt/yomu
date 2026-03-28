pub mod chunker;
mod embed;
pub mod walker;

use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::{Arc, Mutex};

use rurico::embed::{Embed, EmbedError};

use crate::resolver::{Resolve, Resolver};
use crate::rust_resolver::RustResolver;
use crate::storage::{self, Db, RefKind, Reference, StorageError};
use chunker::ParsedImport;

use embed::embed_and_store;
pub use embed::{EmbedResult, run_incremental_embed};
#[cfg(test)]
use embed::{MAX_CONSECUTIVE_EMBED_ERRORS, enrich_for_embedding, order_files_for_embedding};

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("storage error during indexing: {0}")]
    Storage(#[from] StorageError),
    #[error("embedding error: {0}")]
    Embed(#[from] EmbedError),
    #[error("IO error during indexing: {0}")]
    Io(#[from] std::io::Error),
    #[error("internal task failed: {0}")]
    Internal(String),
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
}

impl PendingFile {
    fn to_new_chunks(&self) -> Vec<storage::NewChunk<'_>> {
        self.raw_chunks
            .iter()
            .map(|c| storage::NewChunk {
                chunk_type: &c.chunk_type,
                name: c.name.as_deref(),
                content: &c.content,
                start_line: c.start_line,
                end_line: c.end_line,
                parent_index: c.parent_index,
            })
            .collect()
    }
}

const MAX_FILE_SIZE: u64 = 1_000_000;
const LARGE_PROJECT_THRESHOLD: usize = 5_000;

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
    let metadata = std::fs::metadata(file_path).map_err(|e| {
        tracing::warn!(file = %file_path.display(), error = %e, "IO error, skipping file");
        FileAction::Error
    })?;
    if metadata.len() > MAX_FILE_SIZE {
        tracing::warn!(file = %file_path.display(), size = metadata.len(), "Skipped (too large)");
        return Err(FileAction::Skip);
    }
    std::fs::read_to_string(file_path).map_err(|e| {
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

fn prepare_chunks(checked: CheckedFile, file_path: &Path) -> Option<PendingFile> {
    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let file_chunks = chunker::chunk_file(&checked.source, ext);
    if file_chunks.chunks.is_empty() {
        tracing::debug!(file = %checked.rel_path, "Skipped (no chunks)");
        return None;
    }

    let mtime_epoch = std::fs::metadata(file_path)
        .ok()
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as i64);

    let imports_text = file_chunks.imports.join("\n");
    Some(PendingFile {
        rel_path: checked.rel_path,
        raw_chunks: file_chunks.chunks,
        imports_text,
        parsed_imports: file_chunks.parsed_imports,
        hash: checked.hash,
        mtime_epoch,
    })
}

enum FileOutcome {
    Processed(u32),
    Skipped,
    Errored,
}

fn process_file(
    conn: &Db,
    rel_path: String,
    file_path: &Path,
    force: bool,
    resolver: &Resolver,
    rust_resolver: &RustResolver,
) -> Result<FileOutcome, IndexError> {
    let checked = match check_file(conn, rel_path, file_path, force)? {
        CheckResult::Changed(c) => c,
        CheckResult::Skip => return Ok(FileOutcome::Skipped),
        CheckResult::Error => return Ok(FileOutcome::Errored),
    };
    let pf = match prepare_chunks(checked, file_path) {
        Some(pf) => pf,
        None => return Ok(FileOutcome::Skipped),
    };
    let n = pf.raw_chunks.len() as u32;
    let new_chunks = pf.to_new_chunks();
    let refs = if pf.rel_path.ends_with(".rs") {
        build_references(&pf.parsed_imports, &pf.rel_path, rust_resolver)
    } else {
        build_references(&pf.parsed_imports, &pf.rel_path, resolver)
    };
    storage::replace_file_chunks_only(
        conn,
        &pf.rel_path,
        &new_chunks,
        &pf.hash,
        &pf.imports_text,
        &refs,
        pf.mtime_epoch,
    )?;
    Ok(FileOutcome::Processed(n))
}

struct CollectResult {
    pending: Vec<PendingFile>,
    files_skipped: u32,
    files_errored: u32,
    rel_paths: std::collections::HashSet<String>,
}

fn collect_pending_files(
    conn: &Arc<Mutex<Db>>,
    root: &Path,
    files: &[std::path::PathBuf],
    force: bool,
) -> Result<CollectResult, IndexError> {
    let mut pending: Vec<PendingFile> = Vec::new();
    let mut files_skipped = 0u32;
    let mut files_errored = 0u32;
    let mut rel_paths = std::collections::HashSet::with_capacity(files.len());

    for file_path in files {
        let rel_path = to_rel_path(root, file_path);
        rel_paths.insert(rel_path.clone());
        let checked = {
            let conn_guard = conn.lock().unwrap();
            match check_file(&conn_guard, rel_path, file_path, force)? {
                CheckResult::Changed(c) => c,
                CheckResult::Skip => {
                    files_skipped += 1;
                    continue;
                }
                CheckResult::Error => {
                    files_errored += 1;
                    continue;
                }
            }
        };
        match prepare_chunks(checked, file_path) {
            Some(pf) => pending.push(pf),
            None => {
                files_skipped += 1;
            }
        }
    }

    Ok(CollectResult {
        pending,
        files_skipped,
        files_errored,
        rel_paths,
    })
}

fn remove_orphans(
    conn: &Arc<Mutex<Db>>,
    current_rel_paths: std::collections::HashSet<String>,
) -> Result<(), IndexError> {
    let conn = conn.lock().unwrap();
    let indexed_paths = storage::get_all_file_paths(&conn)?;

    let orphans: Vec<_> = indexed_paths
        .difference(&current_rel_paths)
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
    }
}

fn build_references(
    imports: &[ParsedImport],
    source_path: &str,
    resolver: &impl Resolve,
) -> Vec<Reference> {
    imports
        .iter()
        .filter_map(|import| {
            let target = resolver.resolve(&import.source, source_path)?;
            if import.specifiers.is_empty() {
                Some(vec![Reference {
                    source_file: source_path.to_string(),
                    target_file: target,
                    symbol_name: None,
                    ref_kind: RefKind::SideEffect,
                }])
            } else {
                Some(
                    import
                        .specifiers
                        .iter()
                        .map(|s| Reference {
                            source_file: source_path.to_string(),
                            target_file: target.clone(),
                            symbol_name: Some(s.name.clone()),
                            ref_kind: import_kind_to_ref_kind(&s.kind),
                        })
                        .collect(),
                )
            }
        })
        .flatten()
        .collect()
}

pub fn dry_run_index(
    conn: Arc<Mutex<Db>>,
    root: &Path,
    force: bool,
) -> Result<DryRunResult, IndexError> {
    let files = walker::walk_source_files(root);
    let total_files = files.len() as u32;
    let mut files_to_process = 0u32;
    let mut files_to_skip = 0u32;
    let mut files_errored = 0u32;
    let mut current_rel_paths = std::collections::HashSet::with_capacity(files.len());

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

pub fn run_chunk_only_index(conn: Arc<Mutex<Db>>, root: &Path) -> Result<IndexResult, IndexError> {
    run_chunk_only_index_inner(conn, root, false)
}

pub fn run_chunk_only_index_force(
    conn: Arc<Mutex<Db>>,
    root: &Path,
) -> Result<IndexResult, IndexError> {
    run_chunk_only_index_inner(conn, root, true)
}

fn run_chunk_only_index_inner(
    conn: Arc<Mutex<Db>>,
    root: &Path,
    force: bool,
) -> Result<IndexResult, IndexError> {
    let files = walker::walk_source_files(root);
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
        let mut current_rel_paths = std::collections::HashSet::with_capacity(files.len());

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

    remove_orphans(&conn, current_rel_paths)?;

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

/// Per-file embed+store ensures partial progress survives embedding failures.
pub fn run_index(
    conn: Arc<Mutex<Db>>,
    root: &Path,
    embedder: &(impl Embed + ?Sized),
    force: bool,
) -> Result<IndexResult, IndexError> {
    let files = walker::walk_source_files(root);
    if files.len() > LARGE_PROJECT_THRESHOLD {
        tracing::warn!(
            count = files.len(),
            "Large number of files detected — indexing may be slow"
        );
    }
    tracing::info!(file_count = files.len(), force, "Starting indexing");

    let collected = collect_pending_files(&conn, root, &files, force)?;
    let files_skipped = collected.files_skipped;
    let mut files_errored = collected.files_errored;

    remove_orphans(&conn, collected.rel_paths)?;

    let resolver = Resolver::new(root);
    let rust_resolver = RustResolver::new(root);
    let embed_result = embed_and_store(
        &conn,
        embedder,
        collected.pending,
        &resolver,
        &rust_resolver,
    )?;
    let files_processed = embed_result.files_processed;
    let chunks_created = embed_result.chunks_created;
    files_errored += embed_result.files_errored;

    tracing::info!(
        files_processed,
        chunks_created,
        files_skipped,
        files_errored,
        "Indexing complete"
    );

    Ok(IndexResult {
        files_processed,
        chunks_created,
        files_skipped,
        files_errored,
    })
}

fn file_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests;
