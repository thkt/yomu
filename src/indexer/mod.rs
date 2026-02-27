//! Walks, chunks, embeds, and stores frontend source files.

pub mod chunker;
pub mod embedder;
pub mod walker;

use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use sha2::{Digest, Sha256};

use crate::storage::{self, Db, StorageError};
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

async fn embed_and_store(
    conn: &Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    pending: Vec<PendingFile>,
) -> Result<(u32, u32, u32), IndexError> {
    let pending_total = pending.len();
    let mut files_processed = 0u32;
    let mut chunks_created = 0u32;
    let mut files_errored = 0u32;

    for pf in pending {
        let texts: Vec<String> = pf.raw_chunks.iter().map(|c| c.content.clone()).collect();

        let embeddings = match embedder.embed_documents(&texts).await {
            Ok(embs) => embs,
            Err(e) => {
                tracing::warn!(file = %pf.rel_path, error = %e, "Embedding failed, skipping file");
                files_errored += 1;
                continue;
            }
        };

        let n = pf.raw_chunks.len() as u32;
        let rel_path = pf.rel_path.clone();
        let conn_clone = Arc::clone(conn);
        let result = tokio::task::spawn_blocking(move || {
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
            let conn = conn_clone.lock();
            storage::replace_file_chunks(&conn, &pf.rel_path, &new_chunks, &embeddings, &pf.hash, &pf.imports_text)
        })
        .await
        ?;
        result?;

        chunks_created += n;
        files_processed += 1;
        if files_processed.is_multiple_of(10) {
            tracing::info!(files_processed, total = pending_total, "Indexing progress");
        }
        tracing::debug!(file = %rel_path, chunks = n, "Indexed");
    }

    Ok((files_processed, chunks_created, files_errored))
}

/// Index frontend source files under `root` by chunking, embedding, and storing them.
///
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

    let (files_processed, chunks_created, embed_errors) =
        embed_and_store(&conn, embedder, pending).await?;
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
}
