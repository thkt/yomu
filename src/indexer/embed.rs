use std::sync::{Arc, Mutex};

use rurico::embed::{Embed, EmbedError};

use crate::resolver::Resolver;
use crate::rust_resolver::RustResolver;
use crate::storage::{self, Db, Reference, StorageError};

use super::{IndexError, PendingFile, build_references};

pub(super) const MAX_CONSECUTIVE_EMBED_ERRORS: u32 = 5;

#[derive(Debug)]
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
    Skip,
}

pub(super) fn enrich_for_embedding(
    file_path: &str,
    chunk_type: &str,
    imports: &str,
    content: &str,
) -> String {
    let mut result = format!("// File: {file_path}\n// Type: {chunk_type}\n");
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

fn store_file_data(
    conn: &Db,
    pf: PendingFile,
    embeddings: Vec<Vec<f32>>,
    refs: Vec<Reference>,
) -> Result<(), StorageError> {
    let new_chunks = pf.to_new_chunks();
    let data = storage::FileData {
        file_path: &pf.rel_path,
        chunks: &new_chunks,
        file_hash: &pf.hash,
        imports_text: &pf.imports_text,
        refs: &refs,
    };
    storage::replace_file_chunks_with(conn, &data, &embeddings)
}

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
                enrich_for_embedding(
                    &pf.rel_path,
                    c.chunk_type.as_str(),
                    &pf.imports_text,
                    &c.content,
                )
            })
            .collect();

        let texts_ref: Vec<&str> = texts.iter().map(String::as_str).collect();
        let embeddings = match embedder.embed_documents_batch(&texts_ref) {
            Ok(embs) => {
                consecutive_errors = 0;
                embs
            }
            Err(e) => match classify_embed_error(e, &mut consecutive_errors, &pf.rel_path) {
                EmbedFailure::Abort(e) => {
                    return Err(IndexError::Embed(e));
                }
                EmbedFailure::Skip => {
                    files_errored += 1;
                    continue;
                }
            },
        };

        let n = pf.raw_chunks.len() as u32;
        let refs = if pf.rel_path.ends_with(".rs") {
            build_references(&pf.parsed_imports, &pf.rel_path, rust_resolver)
        } else {
            build_references(&pf.parsed_imports, &pf.rel_path, resolver)
        };
        tracing::debug!(file = %pf.rel_path, chunks = n, "Indexing file");

        {
            let conn = conn.lock().unwrap();
            store_file_data(&conn, pf, embeddings, refs)?;
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

fn fetch_unembedded_file(
    conn: &Arc<Mutex<Db>>,
    file_path: &str,
) -> Result<(Vec<i64>, Vec<String>), IndexError> {
    let conn_guard = conn.lock().unwrap();
    let triples = storage::get_unembedded_chunks_for_file(&conn_guard, file_path)?;
    let imports = storage::get_imports_for_file(&conn_guard, file_path)?;
    let ids: Vec<i64> = triples.iter().map(|(id, _, _)| *id).collect();
    let texts: Vec<String> = triples
        .into_iter()
        .map(|(_, content, chunk_type)| {
            enrich_for_embedding(file_path, &chunk_type, &imports, &content)
        })
        .collect();
    Ok((ids, texts))
}

/// Embeds and stores chunks for a single file. Returns `Some(count)` on success, `None` on skip.
fn embed_file_chunks(
    embedder: &(impl Embed + ?Sized),
    conn: &Arc<Mutex<Db>>,
    file_path: &str,
    chunk_ids: Vec<i64>,
    texts: Vec<String>,
    consecutive_errors: &mut u32,
) -> Result<Option<u32>, IndexError> {
    let texts_ref: Vec<&str> = texts.iter().map(String::as_str).collect();
    let embeddings = match embedder.embed_documents_batch(&texts_ref) {
        Ok(embs) => {
            *consecutive_errors = 0;
            embs
        }
        Err(e) => match classify_embed_error(e, consecutive_errors, file_path) {
            EmbedFailure::Abort(e) => return Err(IndexError::Embed(e)),
            EmbedFailure::Skip => return Ok(None),
        },
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

    let pairs: Vec<(i64, Vec<f32>)> = chunk_ids.into_iter().zip(embeddings).collect();
    let n = pairs.len() as u32;
    {
        let conn_guard = conn.lock().unwrap();
        storage::add_embeddings(&conn_guard, &pairs)?;
    }
    Ok(Some(n))
}

pub fn run_incremental_embed(
    conn: Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    max_chunks: u32,
    type_hints: Option<&[storage::ChunkType]>,
) -> Result<EmbedResult, IndexError> {
    let ordered_files = {
        let conn_guard = conn.lock().unwrap();
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
        let (chunk_ids, texts) = fetch_unembedded_file(&conn, file_path)?;
        if texts.is_empty() {
            continue;
        }

        if chunks_embedded.saturating_add(texts.len() as u32) > max_chunks && chunks_embedded > 0 {
            budget_exhausted = true;
            break;
        }

        let n = match embed_file_chunks(
            embedder,
            &conn,
            file_path,
            chunk_ids,
            texts,
            &mut consecutive_errors,
        )? {
            Some(n) => n,
            None => continue,
        };

        chunks_embedded += n;
        files_completed += 1;

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
