use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rurico::embed::{ChunkedEmbedding, Embed, EmbedError};

use crate::storage::{self, Db, StorageError};

use super::IndexError;

pub(super) const MAX_CONSECUTIVE_EMBED_ERRORS: u32 = 5;

#[derive(Debug, Default)]
pub struct EmbedResult {
    pub chunks_embedded: u32,
    pub files_completed: u32,
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
) -> Result<(Vec<ChunkedEmbedding>, Duration), EmbedFailure> {
    let texts_ref: Vec<&str> = texts.iter().map(String::as_str).collect();
    let started = Instant::now();
    let result = embedder.embed_documents_batch(&texts_ref);
    let elapsed = started.elapsed();
    match result {
        Ok(embs) => {
            tracing::debug!(
                file = %file_path,
                batch_size = texts.len(),
                elapsed_ms = elapsed.as_millis(),
                "embed batch"
            );
            *consecutive_errors = 0;
            let validated =
                validate_chunked_embeddings(embs).map_err(|()| EmbedFailure::Contract)?;
            Ok((validated, elapsed))
        }
        Err(e) => Err(classify_embed_error(e, consecutive_errors, file_path)),
    }
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

/// DB-read phase for one file: fetches unembedded chunk rows and the file's
/// imports. Enrichment (text assembly) is left to the caller so the read and
/// enrich phases can be timed independently.
fn fetch_chunks_db(
    conn: &Arc<Mutex<Db>>,
    file_path: &str,
) -> Result<(Vec<storage::UnembeddedChunk>, String), IndexError> {
    let conn_guard = conn.lock().expect("DB lock poisoned (fetch_chunks_db)");
    let rows = storage::get_unembedded_chunks_for_file(&conn_guard, file_path)?;
    let imports = storage::get_imports_for_file(&conn_guard, file_path)?;
    Ok((rows, imports))
}

/// Enrich phase for one file: assembles the embedding input text for each chunk
/// row, returning the chunk ids alongside their enriched texts.
fn enrich_rows(
    file_path: &str,
    rows: &[storage::UnembeddedChunk],
    imports: &str,
) -> (Vec<i64>, Vec<String>) {
    let chunk_ids = rows.iter().map(|r| r.id).collect();
    let texts = rows
        .iter()
        .map(|r| {
            enrich_for_embedding(
                file_path,
                &r.chunk_type,
                r.name.as_deref(),
                r.parent_name.as_deref(),
                imports,
                &r.content,
            )
        })
        .collect();
    (chunk_ids, texts)
}

/// Embeds and stores chunks for a single file. Returns `Some((count, embed, store))`
/// with the forward-pass and storage-write durations on success, `None` on skip.
#[allow(clippy::cast_possible_truncation)]
fn embed_file_chunks(
    embedder: &(impl Embed + ?Sized),
    conn: &Arc<Mutex<Db>>,
    file_path: &str,
    chunk_ids: Vec<i64>,
    texts: &[String],
    consecutive_errors: &mut u32,
) -> Result<Option<(u32, Duration, Duration)>, IndexError> {
    let (embeddings, embed_dur) =
        match run_embed_batch(embedder, texts, consecutive_errors, file_path) {
            Ok((embs, dur)) => (embs, dur),
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
    let store_started = Instant::now();
    {
        let conn_guard = conn.lock().expect("DB lock poisoned (embed_file_chunks)");
        storage::add_chunked_embeddings(&conn_guard, &pairs)?;
    }
    let store_dur = store_started.elapsed();
    Ok(Some((n, embed_dur, store_dur)))
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
    let started = Instant::now();
    let ordered_files = {
        let conn_guard = conn
            .lock()
            .expect("DB lock poisoned (run_incremental_embed_with_progress)");
        order_files_for_embedding(&conn_guard, type_hints)?
    };
    let t_order = started.elapsed();

    if ordered_files.is_empty() {
        return Ok(EmbedResult::default());
    }

    let mut chunks_embedded = 0u32;
    let mut files_completed = 0u32;
    let mut consecutive_errors = 0u32;
    let mut t_fetch = Duration::ZERO;
    let mut t_enrich = Duration::ZERO;
    let mut t_embed = Duration::ZERO;
    let mut t_store = Duration::ZERO;

    for file_path in &ordered_files {
        let fetch_started = Instant::now();
        let (rows, imports) = fetch_chunks_db(conn, file_path)?;
        t_fetch += fetch_started.elapsed();

        let enrich_started = Instant::now();
        let (chunk_ids, texts) = enrich_rows(file_path, &rows, &imports);
        t_enrich += enrich_started.elapsed();

        if texts.is_empty() {
            continue;
        }

        if chunks_embedded.saturating_add(texts.len() as u32) > max_chunks && chunks_embedded > 0 {
            break;
        }

        let Some((n, embed_dur, store_dur)) = embed_file_chunks(
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
        t_embed += embed_dur;
        t_store += store_dur;

        chunks_embedded += n;
        files_completed += 1;
        on_progress(chunks_embedded);

        if chunks_embedded >= max_chunks {
            break;
        }
    }

    let elapsed_ms = started.elapsed().as_millis();
    let chunks_per_sec = chunks_per_sec(chunks_embedded, elapsed_ms);
    tracing::info!(
        chunks_embedded,
        files_completed,
        elapsed_ms,
        chunks_per_sec,
        order_ms = t_order.as_millis(),
        fetch_ms = t_fetch.as_millis(),
        enrich_ms = t_enrich.as_millis(),
        embed_ms = t_embed.as_millis(),
        store_ms = t_store.as_millis(),
        "Incremental embedding complete"
    );

    Ok(EmbedResult {
        chunks_embedded,
        files_completed,
    })
}

/// `chunks_embedded / elapsed_seconds`, truncated to an integer.
/// Returns `0` when `elapsed_ms == 0` (sub-millisecond runs from empty / mock paths).
fn chunks_per_sec(chunks_embedded: u32, elapsed_ms: u128) -> u128 {
    u128::from(chunks_embedded)
        .saturating_mul(1000)
        .checked_div(elapsed_ms)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-360: validate_rejects_empty_chunks
    #[test]
    fn validate_rejects_empty_chunks() {
        let embs = vec![ChunkedEmbedding::new(vec![])];
        assert!(validate_chunked_embeddings(embs).is_err());
    }

    // T-361: validate_passes_non_empty_chunks
    #[test]
    fn validate_passes_non_empty_chunks() {
        let embs = vec![
            ChunkedEmbedding::new(vec![vec![1.0_f32; 3]]),
            ChunkedEmbedding::new(vec![vec![2.0_f32; 3], vec![3.0_f32; 3]]),
        ];
        assert!(validate_chunked_embeddings(embs).is_ok());
    }
}
