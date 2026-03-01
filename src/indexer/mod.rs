//! Walks, chunks, embeds, and stores frontend source files.

pub mod chunker;
pub mod embedder;
pub mod walker;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use sha2::{Digest, Sha256};

use crate::resolver::Resolver;
use crate::storage::{self, Db, RefKind, Reference, StorageError};
use chunker::ParsedImport;
use embedder::{Embed, EmbedError};

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

impl From<tokio::task::JoinError> for IndexError {
    fn from(e: tokio::task::JoinError) -> Self {
        Self::Internal(e.to_string())
    }
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

fn process_file(
    conn: &Db,
    root: &Path,
    file_path: &Path,
    force: bool,
) -> Result<FileAction, IndexError> {
    let source = match read_source(file_path) {
        Ok(s) => s,
        Err(action) => return Ok(action),
    };
    let hash = file_hash(&source);
    let rel_path = to_rel_path(root, file_path);

    if !force && !storage::should_reindex(conn, &rel_path, &hash)? {
        tracing::debug!(file = %rel_path, "Skipped (unchanged)");
        return Ok(FileAction::Skip);
    }

    let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");
    let file_chunks = chunker::chunk_file(&source, ext);
    if file_chunks.chunks.is_empty() {
        tracing::debug!(file = %rel_path, "Skipped (no chunks)");
        return Ok(FileAction::Skip);
    }

    let imports_text = file_chunks.imports.join("\n");
    Ok(FileAction::Process(PendingFile {
        rel_path,
        raw_chunks: file_chunks.chunks,
        imports_text,
        parsed_imports: file_chunks.parsed_imports,
        hash,
    }))
}

fn collect_pending_files(
    conn: &Arc<Mutex<Db>>,
    root: &Path,
    files: &[std::path::PathBuf],
    force: bool,
) -> Result<(Vec<PendingFile>, u32, u32), IndexError> {
    let mut pending: Vec<PendingFile> = Vec::new();
    let mut files_skipped = 0u32;
    let mut files_errored = 0u32;

    for file_path in files {
        let conn_guard = conn.lock();
        match process_file(&conn_guard, root, file_path, force)? {
            FileAction::Process(pf) => {
                drop(conn_guard);
                pending.push(pf);
            }
            FileAction::Skip => files_skipped += 1,
            FileAction::Error => files_errored += 1,
        }
    }

    Ok((pending, files_skipped, files_errored))
}

async fn remove_orphans(
    conn: &Arc<Mutex<Db>>,
    current_rel_paths: std::collections::HashSet<String>,
) -> Result<(), IndexError> {
    let indexed_paths = {
        let conn = Arc::clone(conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            storage::get_all_file_paths(&conn)
        })
        .await
        ?
    }?;

    let orphans: Vec<_> = indexed_paths
        .difference(&current_rel_paths)
        .cloned()
        .collect();
    if orphans.is_empty() {
        return Ok(());
    }

    let conn = Arc::clone(conn);
    let result = tokio::task::spawn_blocking(move || {
        let conn = conn.lock();
        let tx = conn.unchecked_transaction()?;
        for orphan in &orphans {
            storage::delete_file_chunks_in(&tx, orphan)?;
        }
        tx.commit()?;
        tracing::info!(removed = orphans.len(), "Removed orphaned file chunks");
        Ok::<_, StorageError>(())
    })
    .await
    ?;
    Ok(result?)
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

const MAX_CONSECUTIVE_EMBED_ERRORS: u32 = 5;
const RATE_LIMIT_INTERVAL: Duration = Duration::from_millis(700);

enum EmbedFailure {
    RateLimited(EmbedError),
    Abort(EmbedError),
    Skip,
}

fn classify_embed_error(
    e: EmbedError,
    consecutive_errors: &mut u32,
    file_path: &str,
) -> EmbedFailure {
    if matches!(&e, EmbedError::Api { status: 429, .. }) {
        tracing::warn!("Rate limit exhausted (429 after retries)");
        return EmbedFailure::RateLimited(e);
    }
    *consecutive_errors += 1;
    tracing::warn!(
        file = %file_path, error = %e, consecutive = *consecutive_errors,
        "Embedding failed, skipping file",
    );
    if *consecutive_errors >= MAX_CONSECUTIVE_EMBED_ERRORS {
        tracing::error!(consecutive_errors = *consecutive_errors, "Too many consecutive embedding failures, aborting");
        EmbedFailure::Abort(e)
    } else {
        EmbedFailure::Skip
    }
}

fn store_file_data(
    conn: &Arc<Mutex<Db>>,
    pf: PendingFile,
    embeddings: Vec<Vec<f32>>,
    refs: Vec<Reference>,
) -> Result<(), StorageError> {
    let new_chunks = pf.to_new_chunks();
    let conn = conn.lock();
    storage::replace_file_chunks(&conn, &pf.rel_path, &new_chunks, &embeddings, &pf.hash, &pf.imports_text, &refs)
}

async fn embed_and_store(
    conn: &Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    pending: Vec<PendingFile>,
    resolver: &Resolver,
) -> Result<(u32, u32, u32), IndexError> {
    let pending_total = pending.len();
    let mut files_processed = 0u32;
    let mut chunks_created = 0u32;
    let mut files_errored = 0u32;
    let mut consecutive_errors = 0u32;

    for pf in pending {
        let texts: Vec<String> = pf.raw_chunks.iter().map(|c| c.content.clone()).collect();

        let embeddings = match embedder.embed_documents(&texts).await {
            Ok(embs) => {
                consecutive_errors = 0;
                embs
            }
            Err(e) => match classify_embed_error(e, &mut consecutive_errors, &pf.rel_path) {
                EmbedFailure::RateLimited(e) | EmbedFailure::Abort(e) => {
                    return Err(IndexError::Embed(e));
                }
                EmbedFailure::Skip => {
                    files_errored += 1;
                    continue;
                }
            },
        };

        tokio::time::sleep(RATE_LIMIT_INTERVAL).await;

        let n = pf.raw_chunks.len() as u32;
        let rel_path = pf.rel_path.clone();
        let refs = build_references(&pf.parsed_imports, &pf.rel_path, resolver);

        let conn_clone = Arc::clone(conn);
        tokio::task::spawn_blocking(move || store_file_data(&conn_clone, pf, embeddings, refs))
            .await??;

        chunks_created += n;
        files_processed += 1;
        if files_processed.is_multiple_of(10) {
            tracing::info!(files_processed, total = pending_total, "Indexing progress");
        }
        tracing::debug!(file = %rel_path, chunks = n, "Indexed");
    }

    Ok((files_processed, chunks_created, files_errored))
}

#[derive(Debug)]
pub struct EmbedResult {
    pub chunks_embedded: u32,
    pub files_completed: u32,
    pub budget_exhausted: bool,
}

/// Walk, chunk, and store files without calling the embedding API.
pub async fn run_chunk_only_index(
    conn: Arc<Mutex<Db>>,
    root: &Path,
) -> Result<IndexResult, IndexError> {
    run_chunk_only_index_inner(conn, root, false).await
}

pub async fn run_chunk_only_index_force(
    conn: Arc<Mutex<Db>>,
    root: &Path,
) -> Result<IndexResult, IndexError> {
    run_chunk_only_index_inner(conn, root, true).await
}

async fn run_chunk_only_index_inner(
    conn: Arc<Mutex<Db>>,
    root: &Path,
    force: bool,
) -> Result<IndexResult, IndexError> {
    let files = walker::walk_frontend_files(root);
    tracing::info!(file_count = files.len(), force, "Starting chunk-only indexing");

    let current_rel_paths: std::collections::HashSet<String> =
        files.iter().map(|f| to_rel_path(root, f)).collect();

    let mut files_processed = 0u32;
    let mut chunks_created = 0u32;
    let mut files_skipped = 0u32;
    let mut files_errored = 0u32;

    let resolver = Resolver::new(root);

    for file_path in &files {
        let conn_guard = conn.lock();
        match process_file(&conn_guard, root, file_path, force)? {
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
                drop(conn_guard);
                chunks_created += n;
                files_processed += 1;
            }
            FileAction::Skip => files_skipped += 1,
            FileAction::Error => files_errored += 1,
        }
    }

    remove_orphans(&conn, current_rel_paths).await?;

    tracing::info!(files_processed, chunks_created, files_skipped, files_errored, "Chunk-only indexing complete");

    Ok(IndexResult {
        files_processed,
        chunks_created,
        files_skipped,
        files_errored,
    })
}

fn order_files_for_embedding(
    conn: &Db,
    type_hints: Option<&[storage::ChunkType]>,
) -> Result<Vec<String>, StorageError> {
    let mut files = storage::get_files_by_import_count(conn)?;

    if let Some(hints) = type_hints
        && !hints.is_empty()
    {
        let hint_files = storage::get_files_with_chunk_types(conn, &files, hints)?;
        if !hint_files.is_empty() {
            let mut prioritized: Vec<String> = Vec::new();
            let mut rest: Vec<String> = Vec::new();
            for f in files {
                if hint_files.contains(&f) {
                    prioritized.push(f);
                } else {
                    rest.push(f);
                }
            }
            prioritized.extend(rest);
            files = prioritized;
        }
    }

    Ok(files)
}

/// Embed un-embedded chunks up to `max_chunks` budget, most-imported-first.
pub async fn run_incremental_embed(
    conn: Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    max_chunks: u32,
    type_hints: Option<&[storage::ChunkType]>,
) -> Result<EmbedResult, IndexError> {
    let ordered_files = {
        let conn_guard = conn.lock();
        order_files_for_embedding(&conn_guard, type_hints)?
    };

    if ordered_files.is_empty() {
        return Ok(EmbedResult {
            chunks_embedded: 0,
            files_completed: 0,
            budget_exhausted: false,
        });
    }

    let mut chunks_embedded = 0u32;
    let mut files_completed = 0u32;
    let mut budget_exhausted = false;
    let mut consecutive_errors = 0u32;

    for file_path in &ordered_files {
        let (chunk_ids, texts) = {
            let conn_guard = conn.lock();
            let pairs =
                storage::get_unembedded_chunks_for_file(&conn_guard, file_path)?;
            let ids: Vec<i64> = pairs.iter().map(|(id, _)| *id).collect();
            let texts: Vec<String> = pairs.into_iter().map(|(_, t)| t).collect();
            (ids, texts)
        };

        if texts.is_empty() {
            continue;
        }

        if chunks_embedded.saturating_add(texts.len() as u32) > max_chunks && chunks_embedded > 0 {
            budget_exhausted = true;
            break;
        }

        let embeddings = match embedder.embed_documents(&texts).await {
            Ok(embs) => {
                consecutive_errors = 0;
                embs
            }
            Err(e) => match classify_embed_error(e, &mut consecutive_errors, file_path) {
                EmbedFailure::RateLimited(_) => {
                    budget_exhausted = true;
                    break;
                }
                EmbedFailure::Abort(e) => return Err(IndexError::Embed(e)),
                EmbedFailure::Skip => continue,
            },
        };

        if embeddings.len() != chunk_ids.len() {
            consecutive_errors += 1;
            tracing::warn!(
                file = %file_path,
                expected = chunk_ids.len(),
                actual = embeddings.len(),
                consecutive = consecutive_errors,
                "Embedding count mismatch, skipping file"
            );
            if consecutive_errors >= MAX_CONSECUTIVE_EMBED_ERRORS {
                tracing::error!(consecutive_errors, "Too many consecutive failures in incremental embed, aborting");
                break;
            }
            continue;
        }

        tokio::time::sleep(RATE_LIMIT_INTERVAL).await;

        let pairs: Vec<(i64, Vec<f32>)> = chunk_ids
            .into_iter()
            .zip(embeddings)
            .collect();

        let n = pairs.len() as u32;
        let conn_clone = Arc::clone(&conn);
        tokio::task::spawn_blocking(move || {
            let conn_guard = conn_clone.lock();
            storage::add_embeddings(&conn_guard, &pairs)
        })
        .await??;

        chunks_embedded += n;
        files_completed += 1;

        if chunks_embedded >= max_chunks {
            budget_exhausted = true;
            break;
        }
    }

    tracing::info!(chunks_embedded, files_completed, budget_exhausted, "Incremental embedding complete");

    Ok(EmbedResult {
        chunks_embedded,
        files_completed,
        budget_exhausted,
    })
}

/// Per-file embed+store ensures partial progress survives API failures.
pub async fn run_index(
    conn: Arc<Mutex<Db>>,
    root: &Path,
    embedder: &(impl Embed + ?Sized),
    force: bool,
) -> Result<IndexResult, IndexError> {
    let files = walker::walk_frontend_files(root);
    if files.len() > LARGE_PROJECT_THRESHOLD {
        tracing::warn!(count = files.len(), "Large number of files detected — indexing may be slow");
    }
    tracing::info!(file_count = files.len(), force, "Starting indexing");

    let current_rel_paths: std::collections::HashSet<String> =
        files.iter().map(|f| to_rel_path(root, f)).collect();

    let (pending, files_skipped, mut files_errored) = {
        let conn = Arc::clone(&conn);
        let root = root.to_owned();
        let result = tokio::task::spawn_blocking(move || {
            collect_pending_files(&conn, &root, &files, force)
        })
        .await
        ?;
        result?
    };

    remove_orphans(&conn, current_rel_paths).await?;

    let resolver = Resolver::new(root);
    let (files_processed, chunks_created, embed_errors) =
        embed_and_store(&conn, embedder, pending, &resolver).await?;
    files_errored += embed_errors;

    tracing::info!(files_processed, chunks_created, files_skipped, files_errored, "Indexing complete");

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
