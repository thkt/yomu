use std::sync::{Arc, Mutex};

use rurico::embed::{ChunkedEmbedding, Embed, EmbedError};

use crate::resolver::Resolver;
use crate::rust_resolver::RustResolver;
use crate::storage::{self, Db, Reference, StorageError};

use super::{IndexError, PendingFile, build_references};

pub(super) const MAX_CONSECUTIVE_EMBED_ERRORS: u32 = 5;

#[derive(Debug, Default)]
pub struct EmbedResult {
    pub chunks_embedded: u32,
    pub files_completed: u32,
    pub budget_exhausted: bool,
}

pub(super) struct EmbedStoreResult {
    pub files_processed: u32,
    pub chunks_created: u32,
    pub files_errored: u32,
}

enum EmbedFailure {
    Abort(EmbedError),
    /// rurico returned an empty `ChunkedEmbedding.chunks`, violating its documented contract.
    Contract,
    Skip,
}

fn validate_chunked_embeddings(embs: Vec<ChunkedEmbedding>) -> Result<Vec<ChunkedEmbedding>, ()> {
    if embs.iter().any(|e| e.chunks.is_empty()) {
        return Err(());
    }
    Ok(embs)
}

pub(super) fn enrich_for_embedding(
    file_path: &str,
    chunk_type: &str,
    name: Option<&str>,
    parent_name: Option<&str>,
    imports: &str,
    content: &str,
) -> String {
    let mut result = format!("// File: {file_path}\n// Type: {chunk_type}\n");
    if let Some(n) = name {
        result.push_str("// Name: ");
        result.push_str(n);
        result.push('\n');
    }
    if let Some(p) = parent_name {
        result.push_str("// Parent: ");
        result.push_str(p);
        result.push('\n');
    }
    if !imports.is_empty() {
        for line in imports.lines() {
            result.push_str("// ");
            result.push_str(line);
            result.push('\n');
        }
    }
    result.push_str(content);
    result
}

fn classify_embed_error(
    e: EmbedError,
    consecutive_errors: &mut u32,
    file_path: &str,
) -> EmbedFailure {
    *consecutive_errors += 1;
    tracing::warn!(
        file = %file_path, error = %e, consecutive = *consecutive_errors,
        "Embedding failed, skipping file",
    );
    if *consecutive_errors >= MAX_CONSECUTIVE_EMBED_ERRORS {
        tracing::error!(
            consecutive_errors = *consecutive_errors,
            "Too many consecutive embedding failures, aborting"
        );
        EmbedFailure::Abort(e)
    } else {
        EmbedFailure::Skip
    }
}

fn run_embed_batch(
    embedder: &(impl Embed + ?Sized),
    texts: &[String],
    consecutive_errors: &mut u32,
    file_path: &str,
) -> Result<Vec<ChunkedEmbedding>, EmbedFailure> {
    let texts_ref: Vec<&str> = texts.iter().map(String::as_str).collect();
    match embedder.embed_documents_batch(&texts_ref) {
        Ok(embs) => {
            *consecutive_errors = 0;
            validate_chunked_embeddings(embs).map_err(|()| EmbedFailure::Contract)
        }
        Err(e) => Err(classify_embed_error(e, consecutive_errors, file_path)),
    }
}

fn store_file_data(
    conn: &Db,
    pf: &PendingFile,
    embeddings: &[ChunkedEmbedding],
    refs: &[Reference],
) -> Result<(), StorageError> {
    let new_chunks = pf.to_new_chunks();
    let data = storage::FileData {
        file_path: &pf.rel_path,
        chunks: &new_chunks,
        file_hash: &pf.hash,
        imports_text: &pf.imports_text,
        refs,
        mtime_epoch: pf.mtime_epoch,
    };
    storage::replace_file_chunks_with(conn, &data, embeddings)
}

#[allow(clippy::cast_possible_truncation)]
pub(super) fn embed_and_store(
    conn: &Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    pending: Vec<PendingFile>,
    resolver: &Resolver,
    rust_resolver: &RustResolver,
) -> Result<EmbedStoreResult, IndexError> {
    let pending_total = pending.len();
    let mut files_processed = 0u32;
    let mut chunks_created = 0u32;
    let mut files_errored = 0u32;
    let mut consecutive_errors = 0u32;

    for pf in pending {
        let texts: Vec<String> = pf
            .raw_chunks
            .iter()
            .filter(|c| c.chunk_type != storage::ChunkType::InnerFn)
            .map(|c| {
                let parent_name = c
                    .parent_index
                    .and_then(|i| pf.raw_chunks[i].name.as_deref());
                enrich_for_embedding(
                    &pf.rel_path,
                    c.chunk_type.as_str(),
                    c.name.as_deref(),
                    parent_name,
                    &pf.imports_text,
                    &c.content,
                )
            })
            .collect();

        let embeddings =
            match run_embed_batch(embedder, &texts, &mut consecutive_errors, &pf.rel_path) {
                Ok(embs) => embs,
                Err(EmbedFailure::Abort(e)) => return Err(IndexError::Embed(e)),
                Err(EmbedFailure::Contract) => {
                    return Err(IndexError::Internal(
                        "rurico returned empty ChunkedEmbedding.chunks (contract violation)".into(),
                    ));
                }
                Err(EmbedFailure::Skip) => {
                    files_errored += 1;
                    continue;
                }
            };

        let n = embeddings.len() as u32;
        let refs = if pf.rel_path.ends_with(".rs") {
            build_references(&pf.parsed_imports, &pf.rel_path, rust_resolver)
        } else {
            build_references(&pf.parsed_imports, &pf.rel_path, resolver)
        };
        tracing::debug!(file = %pf.rel_path, chunks = n, "Indexing file");

        {
            let conn = conn.lock().unwrap();
            store_file_data(&conn, &pf, &embeddings, &refs)?;
        }

        chunks_created += n;
        files_processed += 1;
        if files_processed.is_multiple_of(10) {
            tracing::info!(files_processed, total = pending_total, "Indexing progress");
        }
    }

    Ok(EmbedStoreResult {
        files_processed,
        chunks_created,
        files_errored,
    })
}

pub(super) fn order_files_for_embedding(
    conn: &Db,
    type_hints: Option<&[storage::ChunkType]>,
) -> Result<Vec<String>, StorageError> {
    let mut files = storage::get_files_by_import_count(conn)?;

    if let Some(hints) = type_hints
        && !hints.is_empty()
    {
        let hint_files = storage::get_files_with_chunk_types(conn, &files, hints)?;
        if !hint_files.is_empty() {
            let (mut prioritized, rest): (Vec<_>, Vec<_>) =
                files.into_iter().partition(|f| hint_files.contains(f));
            prioritized.extend(rest);
            files = prioritized;
        }
    }

    Ok(files)
}

fn fetch_unembedded_file(
    conn: &Arc<Mutex<Db>>,
    file_path: &str,
) -> Result<(Vec<i64>, Vec<String>), IndexError> {
    let conn_guard = conn.lock().unwrap();
    let rows = storage::get_unembedded_chunks_for_file(&conn_guard, file_path)?;
    let imports = storage::get_imports_for_file(&conn_guard, file_path)?;
    let ids: Vec<i64> = rows.iter().map(|r| r.id).collect();
    let texts: Vec<String> = rows
        .into_iter()
        .map(|r| {
            enrich_for_embedding(
                file_path,
                &r.chunk_type,
                r.name.as_deref(),
                r.parent_name.as_deref(),
                &imports,
                &r.content,
            )
        })
        .collect();
    Ok((ids, texts))
}

/// Embeds and stores chunks for a single file. Returns `Some(count)` on success, `None` on skip.
#[allow(clippy::cast_possible_truncation)]
fn embed_file_chunks(
    embedder: &(impl Embed + ?Sized),
    conn: &Arc<Mutex<Db>>,
    file_path: &str,
    chunk_ids: Vec<i64>,
    texts: &[String],
    consecutive_errors: &mut u32,
) -> Result<Option<u32>, IndexError> {
    let embeddings: Vec<ChunkedEmbedding> =
        match run_embed_batch(embedder, texts, consecutive_errors, file_path) {
            Ok(embs) => embs,
            Err(EmbedFailure::Abort(e)) => return Err(IndexError::Embed(e)),
            Err(EmbedFailure::Contract) => {
                return Err(IndexError::Internal(
                    "rurico returned empty ChunkedEmbedding.chunks (contract violation)".into(),
                ));
            }
            Err(EmbedFailure::Skip) => return Ok(None),
        };

    if embeddings.len() != chunk_ids.len() {
        *consecutive_errors += 1;
        tracing::warn!(
            file = %file_path,
            expected = chunk_ids.len(),
            actual = embeddings.len(),
            consecutive = *consecutive_errors,
            "Embedding count mismatch, skipping file"
        );
        if *consecutive_errors >= MAX_CONSECUTIVE_EMBED_ERRORS {
            tracing::error!(
                consecutive_errors = *consecutive_errors,
                "Too many consecutive failures in incremental embed, aborting"
            );
            return Err(IndexError::Internal(format!(
                "Aborting incremental embed after {} consecutive failures",
                consecutive_errors
            )));
        }
        return Ok(None);
    }

    let pairs: Vec<(i64, ChunkedEmbedding)> = chunk_ids.into_iter().zip(embeddings).collect();
    let n = pairs.len() as u32;
    {
        let conn_guard = conn.lock().unwrap();
        storage::add_chunked_embeddings(&conn_guard, &pairs)?;
    }
    Ok(Some(n))
}

#[allow(clippy::cast_possible_truncation)]
pub fn run_incremental_embed(
    conn: &Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    max_chunks: u32,
    type_hints: Option<&[storage::ChunkType]>,
) -> Result<EmbedResult, IndexError> {
    run_incremental_embed_with_progress(conn, embedder, max_chunks, type_hints, |_| {})
}

/// Like [`run_incremental_embed`] but calls `on_progress(chunks_embedded)` after each file.
#[allow(clippy::cast_possible_truncation)]
pub fn run_incremental_embed_with_progress(
    conn: &Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    max_chunks: u32,
    type_hints: Option<&[storage::ChunkType]>,
    mut on_progress: impl FnMut(u32),
) -> Result<EmbedResult, IndexError> {
    let ordered_files = {
        let conn_guard = conn.lock().unwrap();
        order_files_for_embedding(&conn_guard, type_hints)?
    };

    if ordered_files.is_empty() {
        return Ok(EmbedResult::default());
    }

    let mut chunks_embedded = 0u32;
    let mut files_completed = 0u32;
    let mut budget_exhausted = false;
    let mut consecutive_errors = 0u32;

    for file_path in &ordered_files {
        let (chunk_ids, texts) = fetch_unembedded_file(conn, file_path)?;
        if texts.is_empty() {
            continue;
        }

        if chunks_embedded.saturating_add(texts.len() as u32) > max_chunks && chunks_embedded > 0 {
            budget_exhausted = true;
            break;
        }

        let Some(n) = embed_file_chunks(
            embedder,
            conn,
            file_path,
            chunk_ids,
            &texts,
            &mut consecutive_errors,
        )?
        else {
            continue;
        };

        chunks_embedded += n;
        files_completed += 1;
        on_progress(chunks_embedded);

        if chunks_embedded >= max_chunks {
            budget_exhausted = true;
            break;
        }
    }

    tracing::info!(
        chunks_embedded,
        files_completed,
        budget_exhausted,
        "Incremental embedding complete"
    );

    Ok(EmbedResult {
        chunks_embedded,
        files_completed,
        budget_exhausted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-360: validate_rejects_empty_chunks
    #[test]
    fn validate_rejects_empty_chunks() {
        let embs = vec![ChunkedEmbedding { chunks: vec![] }];
        assert!(validate_chunked_embeddings(embs).is_err());
    }

    // T-361: validate_passes_non_empty_chunks
    #[test]
    fn validate_passes_non_empty_chunks() {
        let embs = vec![
            ChunkedEmbedding {
                chunks: vec![vec![1.0_f32; 3]],
            },
            ChunkedEmbedding {
                chunks: vec![vec![2.0_f32; 3], vec![3.0_f32; 3]],
            },
        ];
        assert!(validate_chunked_embeddings(embs).is_ok());
    }
}
