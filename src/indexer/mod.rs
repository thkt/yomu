//! Walks, chunks, embeds, and stores frontend source files.

pub mod chunker;
pub mod embedder;
pub mod walker;

use std::path::Path;
use std::sync::Arc;

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

/// Summary of an indexing run returned by [`run_index`].
pub struct IndexResult {
    /// Files that were chunked, embedded, and stored.
    pub files_processed: u32,
    /// Total chunks inserted across all processed files.
    pub chunks_created: u32,
    /// Files skipped because content hash was unchanged.
    pub files_skipped: u32,
    /// Files that failed to read or embed (skipped, not fatal).
    pub files_errored: u32,
}

struct PendingFile {
    rel_path: String,
    raw_chunks: Vec<chunker::RawChunk>,
    imports_text: String,
    parsed_imports: Vec<chunker::ParsedImport>,
    hash: String,
}

const MAX_FILE_SIZE: u64 = 1_000_000;

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
    let conn = conn.lock();
    let mut pending: Vec<PendingFile> = Vec::new();
    let mut files_skipped = 0u32;
    let mut files_errored = 0u32;

    for file_path in files {
        match process_file(&conn, root, file_path, force)? {
            FileAction::Process(pf) => pending.push(pf),
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

fn store_file_data(
    conn: &Arc<Mutex<Db>>,
    pf: PendingFile,
    embeddings: Vec<Vec<f32>>,
    refs: Vec<Reference>,
) -> Result<(), StorageError> {
    let new_chunks: Vec<storage::NewChunk> = pf
        .raw_chunks
        .iter()
        .map(|c| storage::NewChunk {
            chunk_type: &c.chunk_type,
            name: c.name.as_deref(),
            content: &c.content,
            start_line: c.start_line,
            end_line: c.end_line,
        })
        .collect();
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
            Err(e) => {
                consecutive_errors += 1;
                tracing::warn!(file = %pf.rel_path, error = %e, consecutive = consecutive_errors, "Embedding failed, skipping file");
                files_errored += 1;
                if consecutive_errors >= MAX_CONSECUTIVE_EMBED_ERRORS {
                    tracing::error!(consecutive_errors, "Too many consecutive embedding failures, aborting");
                    return Err(IndexError::Embed(e));
                }
                continue;
            }
        };

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

/// Summary of a chunk-only indexing run (no embedding API calls).
pub struct ChunkOnlyResult {
    pub files_processed: u32,
    pub chunks_created: u32,
    pub files_skipped: u32,
    pub files_errored: u32,
}

/// Summary of an incremental embedding run.
pub struct EmbedResult {
    pub chunks_embedded: u32,
    pub files_completed: u32,
    pub budget_exhausted: bool,
}

/// Walk, chunk, and store files without calling the embedding API.
/// Populates chunks, file_context, and file_references.
pub async fn run_chunk_only_index(
    conn: Arc<Mutex<Db>>,
    root: &Path,
) -> Result<ChunkOnlyResult, IndexError> {
    let files = walker::walk_frontend_files(root);
    tracing::info!(file_count = files.len(), "Starting chunk-only indexing");

    let current_rel_paths: std::collections::HashSet<String> =
        files.iter().map(|f| to_rel_path(root, f)).collect();

    let mut files_processed = 0u32;
    let mut chunks_created = 0u32;
    let mut files_skipped = 0u32;
    let mut files_errored = 0u32;

    let resolver = Resolver::new(root);

    {
        let conn_guard = conn.lock();
        for file_path in &files {
            match process_file(&conn_guard, root, file_path, false)? {
                FileAction::Process(pf) => {
                    let n = pf.raw_chunks.len() as u32;
                    let new_chunks: Vec<storage::NewChunk> = pf
                        .raw_chunks
                        .iter()
                        .map(|c| storage::NewChunk {
                            chunk_type: &c.chunk_type,
                            name: c.name.as_deref(),
                            content: &c.content,
                            start_line: c.start_line,
                            end_line: c.end_line,
                        })
                        .collect();
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
    }

    remove_orphans(&conn, current_rel_paths).await?;

    tracing::info!(files_processed, chunks_created, files_skipped, files_errored, "Chunk-only indexing complete");

    Ok(ChunkOnlyResult {
        files_processed,
        chunks_created,
        files_skipped,
        files_errored,
    })
}

/// Embed un-embedded chunks up to `max_chunks` budget.
/// Processes files in most-imported-first order.
/// When `type_hints` is provided, files containing matching chunk types are prioritized.
pub async fn run_incremental_embed(
    conn: Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    max_chunks: u32,
    type_hints: Option<&[storage::ChunkType]>,
) -> Result<EmbedResult, IndexError> {
    let ordered_files = {
        let conn_guard = conn.lock();
        let mut files = storage::get_files_by_import_count(&conn_guard)?;

        if let Some(hints) = type_hints {
            if !hints.is_empty() {
                let type_list: Vec<String> = hints.iter().map(|t| format!("'{}'", t.as_str())).collect();
                let sql = format!(
                    "SELECT DISTINCT file_path FROM chunks WHERE chunk_type IN ({}) AND file_path IN ({})",
                    type_list.join(","),
                    files.iter().map(|f| format!("'{f}'")).collect::<Vec<_>>().join(","),
                );
                let mut stmt = conn_guard.prepare(&sql).map_err(storage::StorageError::from)?;
                let hint_files: std::collections::HashSet<String> = stmt
                    .query_map([], |row| row.get::<_, String>(0))
                    .map_err(storage::StorageError::from)?
                    .filter_map(|r| r.ok())
                    .collect();

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
        }

        files
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

    for file_path in &ordered_files {
        // Get chunk IDs and content for this file
        let (chunk_ids, texts) = {
            let conn_guard = conn.lock();
            let mut stmt = conn_guard.prepare(
                "SELECT c.id, c.content FROM chunks c
                 LEFT JOIN vec_chunks v ON c.id = v.chunk_id
                 WHERE c.file_path = ?1 AND v.chunk_id IS NULL",
            ).map_err(storage::StorageError::from)?;
            let rows = stmt.query_map([file_path.as_str()], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            }).map_err(storage::StorageError::from)?;
            let pairs: Vec<(i64, String)> = rows
                .collect::<Result<Vec<_>, _>>()
                .map_err(storage::StorageError::from)?;
            let ids: Vec<i64> = pairs.iter().map(|(id, _)| *id).collect();
            let texts: Vec<String> = pairs.into_iter().map(|(_, t)| t).collect();
            (ids, texts)
        };

        if texts.is_empty() {
            continue;
        }

        // Check budget
        if chunks_embedded + texts.len() as u32 > max_chunks && chunks_embedded > 0 {
            budget_exhausted = true;
            break;
        }

        let embeddings = match embedder.embed_documents(&texts).await {
            Ok(embs) => embs,
            Err(e) => {
                tracing::warn!(file = %file_path, error = %e, "Embedding failed, skipping file");
                continue;
            }
        };

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
    if files.len() > 5000 {
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
mod tests {
    use super::*;
    #[test]
    fn file_hash_is_deterministic() {
        let h1 = file_hash("hello world");
        let h2 = file_hash("hello world");
        assert_eq!(h1, h2);
    }

    #[test]
    fn file_hash_changes_with_content() {
        let h1 = file_hash("hello");
        let h2 = file_hash("world");
        assert_ne!(h1, h2);
    }

    use crate::indexer::embedder::MockEmbedder;

    #[tokio::test]
    async fn run_index_with_mock_embedder() {
        let dir = tempfile::tempdir().unwrap();

        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(
            src_dir.join("Button.tsx"),
            "function Button() { return <div/>; }",
        ).unwrap();
        std::fs::write(
            src_dir.join("App.tsx"),
            "function App() { return <main/>; }",
        ).unwrap();

        let db_path = dir.path().join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path).unwrap();
        let conn = Arc::new(Mutex::new(conn));

        let result = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false)
            .await
            .unwrap();

        assert_eq!(result.files_processed, 2);
        assert!(result.chunks_created >= 2);
        assert_eq!(result.files_errored, 0);

        let stats = storage::get_stats(&conn.lock()).unwrap();
        assert_eq!(stats.total_files, 2);
        assert!(stats.total_chunks >= 2);
    }

    #[tokio::test]
    async fn run_index_skips_unchanged_files() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(src_dir.join("A.tsx"), "function A() { return 1; }").unwrap();

        let db_path = dir.path().join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path).unwrap();
        let conn = Arc::new(Mutex::new(conn));

        // First run: processes file
        let r1 = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).await.unwrap();
        assert_eq!(r1.files_processed, 1);

        // Second run: skips unchanged file
        let r2 = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).await.unwrap();
        assert_eq!(r2.files_processed, 0);
        assert_eq!(r2.files_skipped, 1);
    }

    #[tokio::test]
    async fn run_index_force_reindexes() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(src_dir.join("A.tsx"), "function A() { return 1; }").unwrap();

        let db_path = dir.path().join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path).unwrap();
        let conn = Arc::new(Mutex::new(conn));

        run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).await.unwrap();

        // Force reindex
        let r2 = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, true).await.unwrap();
        assert_eq!(r2.files_processed, 1);
        assert_eq!(r2.files_skipped, 0);
    }

    #[tokio::test]
    async fn run_index_removes_deleted_file_chunks() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(src_dir.join("A.tsx"), "function A() { return 1; }").unwrap();
        std::fs::write(src_dir.join("B.tsx"), "function B() { return 2; }").unwrap();

        let db_path = dir.path().join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path).unwrap();
        let conn = Arc::new(Mutex::new(conn));

        // Index both files
        let r1 = run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).await.unwrap();
        assert_eq!(r1.files_processed, 2);
        assert_eq!(storage::get_stats(&conn.lock()).unwrap().total_files, 2);

        // Delete B.tsx from disk
        std::fs::remove_file(src_dir.join("B.tsx")).unwrap();

        // Re-index: should remove orphaned B.tsx chunks
        run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).await.unwrap();
        let stats = storage::get_stats(&conn.lock()).unwrap();
        assert_eq!(stats.total_files, 1, "orphaned file should be removed");
    }

    #[test]
    fn build_references_named_import() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("B.tsx"), "").unwrap();

        let resolver = crate::resolver::Resolver::new(tmp.path());
        let imports = vec![chunker::ParsedImport {
            specifiers: vec![chunker::ImportSpecifier {
                name: "B".to_string(),
                alias: None,
                kind: chunker::ImportKind::Named,
            }],
            source: "./B".to_string(),
        }];
        let refs = build_references(&imports, "src/A.tsx", &resolver);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].source_file, "src/A.tsx");
        assert_eq!(refs[0].target_file, "src/B.tsx");
        assert_eq!(refs[0].symbol_name, Some("B".to_string()));
        assert_eq!(refs[0].ref_kind, RefKind::Named);
    }

    #[test]
    fn build_references_side_effect_import() {
        let tmp = tempfile::tempdir().unwrap();
        let src = tmp.path().join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("styles.css"), "").unwrap();

        let resolver = crate::resolver::Resolver::new(tmp.path());
        let imports = vec![chunker::ParsedImport {
            specifiers: vec![],
            source: "./styles.css".to_string(),
        }];
        let refs = build_references(&imports, "src/A.tsx", &resolver);
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].symbol_name, None);
        assert_eq!(refs[0].ref_kind, RefKind::SideEffect);
    }

    #[test]
    fn build_references_unresolvable_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();

        let resolver = crate::resolver::Resolver::new(tmp.path());
        let imports = vec![chunker::ParsedImport {
            specifiers: vec![chunker::ImportSpecifier {
                name: "useState".to_string(),
                alias: None,
                kind: chunker::ImportKind::Named,
            }],
            source: "react".to_string(),
        }];
        let refs = build_references(&imports, "src/A.tsx", &resolver);
        assert!(refs.is_empty());
    }

    #[tokio::test]
    async fn run_index_stores_imports_in_file_context() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(
            src_dir.join("App.tsx"),
            "import { useState } from 'react';\nimport { useAuth } from './useAuth';\nfunction App() { return <div/>; }",
        ).unwrap();

        let db_path = dir.path().join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path).unwrap();
        let conn = Arc::new(Mutex::new(conn));

        run_index(Arc::clone(&conn), dir.path(), &MockEmbedder, false).await.unwrap();

        let contexts = storage::get_file_contexts(&conn.lock(), &["src/App.tsx"]).unwrap();
        assert_eq!(contexts.len(), 1);
        let imports = &contexts["src/App.tsx"];
        assert!(imports.contains("import { useState } from 'react'"), "expected useState import, got: {imports}");
        assert!(imports.contains("import { useAuth } from './useAuth'"), "expected useAuth import, got: {imports}");
    }

    // ══════════════════════════════════════════════════════════════
    // Incremental Indexing – Phase 2 Indexer (T-011 〜 T-013)
    // ══════════════════════════════════════════════════════════════

    // ── T-011: run_chunk_only_index → chunks あり, vec_chunks なし, refs あり ──

    #[tokio::test]
    async fn run_chunk_only_index_stores_chunks_without_embeddings() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(
            src_dir.join("App.tsx"),
            "import { Button } from './Button';\nfunction App() { return <Button/>; }",
        ).unwrap();
        std::fs::write(
            src_dir.join("Button.tsx"),
            "export function Button() { return <div/>; }",
        ).unwrap();

        let db_path = dir.path().join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path).unwrap();
        let conn = Arc::new(Mutex::new(conn));

        let result = run_chunk_only_index(Arc::clone(&conn), dir.path()).await.unwrap();

        assert_eq!(result.files_processed, 2);
        assert!(result.chunks_created >= 2);
        assert_eq!(result.files_errored, 0);

        let stats = storage::get_stats(&conn.lock()).unwrap();
        assert_eq!(stats.total_files, 2);
        assert!(stats.total_chunks >= 2);
        // No embeddings
        assert_eq!(stats.embedded_chunks, 0);

        // file_references should be populated
        let ref_count = storage::get_reference_count(&conn.lock()).unwrap();
        assert!(ref_count >= 1, "expected at least 1 reference from App→Button, got {ref_count}");
    }

    // ── T-012: run_incremental_embed with budget → embed within budget ──

    #[tokio::test]
    async fn run_incremental_embed_within_budget() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        // Create 3 small files (1 chunk each ≈ 3 chunks total)
        for name in ["A.tsx", "B.tsx", "C.tsx"] {
            std::fs::write(
                src_dir.join(name),
                &format!("function {}() {{ return 1; }}", &name[..1]),
            ).unwrap();
        }

        let db_path = dir.path().join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path).unwrap();
        let conn = Arc::new(Mutex::new(conn));

        // First: chunk-only index
        run_chunk_only_index(Arc::clone(&conn), dir.path()).await.unwrap();
        let stats_before = storage::get_stats(&conn.lock()).unwrap();
        assert_eq!(stats_before.embedded_chunks, 0);

        // Incremental embed with budget=50 (enough for all 3 files)
        let result = run_incremental_embed(
            Arc::clone(&conn), &MockEmbedder, 50, None,
        ).await.unwrap();

        assert!(result.chunks_embedded >= 3);
        assert_eq!(result.files_completed, 3);
        assert!(!result.budget_exhausted);

        let stats_after = storage::get_stats(&conn.lock()).unwrap();
        assert_eq!(stats_after.embedded_chunks, stats_after.total_chunks);
    }

    // ── T-013: run_incremental_embed with small budget → budget_exhausted ──

    #[tokio::test]
    async fn run_incremental_embed_exhausts_budget() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        // Create 5 files, each with 1 chunk
        for i in 0..5 {
            std::fs::write(
                src_dir.join(format!("F{i}.tsx")),
                &format!("function F{i}() {{ return {i}; }}"),
            ).unwrap();
        }

        let db_path = dir.path().join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path).unwrap();
        let conn = Arc::new(Mutex::new(conn));

        run_chunk_only_index(Arc::clone(&conn), dir.path()).await.unwrap();

        // Budget of 2 chunks — should embed only 2 files worth
        let result = run_incremental_embed(
            Arc::clone(&conn), &MockEmbedder, 2, None,
        ).await.unwrap();

        assert!(result.budget_exhausted);
        assert!(result.chunks_embedded <= 2);

        // Some chunks should remain un-embedded
        let stats = storage::get_stats(&conn.lock()).unwrap();
        assert!(stats.embedded_chunks < stats.total_chunks);
    }

    // ══════════════════════════════════════════════════════════════
    // Search Quality – Phase 3 Indexer (T-007, T-008)
    // ══════════════════════════════════════════════════════════════

    // ── T-007: type_hints=[Hook] → hook files embedded first ──

    #[tokio::test]
    async fn run_incremental_embed_prioritizes_type_hints() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path).unwrap();

        // TypeDef file (no hook/component)
        storage::replace_file_chunks_only(
            &conn, "src/types.tsx",
            &[storage::NewChunk {
                chunk_type: &storage::ChunkType::TypeDef,
                name: Some("AuthConfig"),
                content: "interface AuthConfig {}",
                start_line: 1, end_line: 3,
            }],
            "h1", "", &[],
        ).unwrap();

        // Hook file
        storage::replace_file_chunks_only(
            &conn, "src/useAuth.tsx",
            &[storage::NewChunk {
                chunk_type: &storage::ChunkType::Hook,
                name: Some("useAuth"),
                content: "function useAuth() {}",
                start_line: 1, end_line: 3,
            }],
            "h2", "", &[],
        ).unwrap();

        let conn = Arc::new(Mutex::new(conn));

        // Budget of 1 file → only 1 file gets embedded
        let result = run_incremental_embed(
            Arc::clone(&conn), &MockEmbedder, 1,
            Some(&[storage::ChunkType::Hook]),
        ).await.unwrap();

        assert_eq!(result.files_completed, 1);
        assert!(result.chunks_embedded >= 1);

        // The hook file should have been embedded (prioritized by type_hints)
        let stats = storage::get_stats(&conn.lock()).unwrap();
        assert!(stats.embedded_chunks >= 1);

        // Verify hook file is embedded, typedef is not
        let hook_embedded: bool = conn.lock().query_row(
            "SELECT EXISTS(
                SELECT 1 FROM vec_chunks v
                INNER JOIN chunks c ON c.id = v.chunk_id
                WHERE c.file_path = 'src/useAuth.tsx'
            )", [], |row| row.get(0),
        ).unwrap();
        assert!(hook_embedded, "hook file should be embedded first");
    }

    // ── T-008: type_hints=None → existing behavior (import count order) ──

    #[tokio::test]
    async fn run_incremental_embed_none_hints_preserves_order() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path).unwrap();

        // Two files, no references → alphabetical order as tiebreaker
        storage::replace_file_chunks_only(
            &conn, "src/B.tsx",
            &[storage::NewChunk {
                chunk_type: &storage::ChunkType::Component,
                name: Some("B"),
                content: "function B() {}",
                start_line: 1, end_line: 3,
            }],
            "h1", "", &[],
        ).unwrap();
        storage::replace_file_chunks_only(
            &conn, "src/A.tsx",
            &[storage::NewChunk {
                chunk_type: &storage::ChunkType::Component,
                name: Some("A"),
                content: "function A() {}",
                start_line: 1, end_line: 3,
            }],
            "h2", "", &[],
        ).unwrap();

        let conn = Arc::new(Mutex::new(conn));

        // hints=None: should use existing import count order
        let result = run_incremental_embed(
            Arc::clone(&conn), &MockEmbedder, 50, None,
        ).await.unwrap();

        assert_eq!(result.files_completed, 2);
        assert!(result.chunks_embedded >= 2);
        assert!(!result.budget_exhausted);
    }
}
