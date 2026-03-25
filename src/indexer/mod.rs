pub mod chunker;
mod embed;
pub mod walker;

use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::{Arc, Mutex};

use rurico::embed::{Embed, EmbedError};

use crate::resolver::Resolver;
use crate::storage::{self, Db, RefKind, Reference, StorageError};
use chunker::ParsedImport;

pub use embed::{run_incremental_embed, EmbedResult};
use embed::embed_and_store;
#[cfg(test)]
use embed::{enrich_for_embedding, order_files_for_embedding, MAX_CONSECUTIVE_EMBED_ERRORS};

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

struct PendingFile {
    rel_path: String,
    raw_chunks: Vec<chunker::RawChunk>,
    imports_text: String,
    parsed_imports: Vec<chunker::ParsedImport>,
    hash: String,
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
            })
            .collect()
    }
}

const MAX_FILE_SIZE: u64 = 1_000_000;
const LARGE_PROJECT_THRESHOLD: usize = 5_000;

enum FileAction {
    Process(PendingFile),
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

fn prepare_file(rel_path: String, file_path: &Path) -> Result<PendingFile, FileAction> {
    let source = read_source(file_path)?;
    let hash = file_hash(&source);

    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let file_chunks = chunker::chunk_file(&source, ext);
    if file_chunks.chunks.is_empty() {
        tracing::debug!(file = %rel_path, "Skipped (no chunks)");
        return Err(FileAction::Skip);
    }

    let imports_text = file_chunks.imports.join("\n");
    Ok(PendingFile {
        rel_path,
        raw_chunks: file_chunks.chunks,
        imports_text,
        parsed_imports: file_chunks.parsed_imports,
        hash,
    })
}

fn process_file(
    conn: &Db,
    rel_path: String,
    file_path: &Path,
    force: bool,
) -> Result<FileAction, IndexError> {
    let pf = match prepare_file(rel_path, file_path) {
        Ok(pf) => pf,
        Err(action) => return Ok(action),
    };

    if !force && !storage::should_reindex(conn, &pf.rel_path, &pf.hash)? {
        tracing::debug!(file = %pf.rel_path, "Skipped (unchanged)");
        return Ok(FileAction::Skip);
    }

    Ok(FileAction::Process(pf))
}

fn collect_pending_files(
    conn: &Arc<Mutex<Db>>,
    root: &Path,
    files: &[std::path::PathBuf],
    force: bool,
) -> Result<(Vec<PendingFile>, u32, u32, std::collections::HashSet<String>), IndexError> {
    let mut pending: Vec<PendingFile> = Vec::new();
    let mut files_skipped = 0u32;
    let mut files_errored = 0u32;
    let mut all_rel_paths = std::collections::HashSet::with_capacity(files.len());

    for file_path in files {
        let rel_path = to_rel_path(root, file_path);
        all_rel_paths.insert(rel_path.clone());
        let pf = match prepare_file(rel_path, file_path) {
            Ok(pf) => pf,
            Err(FileAction::Skip) => {
                files_skipped += 1;
                continue;
            }
            Err(_) => {
                files_errored += 1;
                continue;
            }
        };

        if !force {
            let conn_guard = conn.lock().unwrap();
            if !storage::should_reindex(&conn_guard, &pf.rel_path, &pf.hash)? {
                tracing::debug!(file = %pf.rel_path, "Skipped (unchanged)");
                files_skipped += 1;
                continue;
            }
        }

        pending.push(pf);
    }

    Ok((pending, files_skipped, files_errored, all_rel_paths))
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
    resolver: &Resolver,
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

pub fn run_chunk_only_index(
    conn: Arc<Mutex<Db>>,
    root: &Path,
) -> Result<IndexResult, IndexError> {
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

    let (files_processed, chunks_created, files_skipped, files_errored, current_rel_paths) = {
        let mut files_processed = 0u32;
        let mut chunks_created = 0u32;
        let mut files_skipped = 0u32;
        let mut files_errored = 0u32;
        let mut current_rel_paths =
            std::collections::HashSet::with_capacity(files.len());

        let conn_guard = conn.lock().unwrap();
        storage::fts_set_automerge(&conn_guard, false)?;

        for file_path in &files {
            let rel_path = to_rel_path(root, file_path);
            current_rel_paths.insert(rel_path.clone());
            match process_file(&conn_guard, rel_path, file_path, force)? {
                FileAction::Process(pf) => {
                    let n = pf.raw_chunks.len() as u32;
                    let new_chunks = pf.to_new_chunks();
                    let refs = build_references(&pf.parsed_imports, &pf.rel_path, &resolver);
                    storage::replace_file_chunks_only(
                        &conn_guard,
                        &pf.rel_path,
                        &new_chunks,
                        &pf.hash,
                        &pf.imports_text,
                        &refs,
                    )?;
                    chunks_created += n;
                    files_processed += 1;
                }
                FileAction::Skip => files_skipped += 1,
                FileAction::Error => files_errored += 1,
            }
        }

        if files_processed > 0 {
            storage::fts_optimize(&conn_guard)?;
        }
        storage::fts_set_automerge(&conn_guard, true)?;

        (files_processed, chunks_created, files_skipped, files_errored, current_rel_paths)
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

    let (pending, files_skipped, mut files_errored, current_rel_paths) =
        collect_pending_files(&conn, root, &files, force)?;

    remove_orphans(&conn, current_rel_paths)?;

    let resolver = Resolver::new(root);
    let (files_processed, chunks_created, embed_errors) =
        embed_and_store(&conn, embedder, pending, &resolver)?;
    files_errored += embed_errors;

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
