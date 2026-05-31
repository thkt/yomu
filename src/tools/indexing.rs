use std::sync::Arc;

use amici::cli::embed_with_spinners;
use rurico::embed::Embed;

use crate::{indexer, storage};

use super::embedder::DegradedReason;
use super::format::{
    format_coverage_note, format_dry_run_json, format_index_json, format_rebuild_json,
};
use super::{IndexRunOptions, Yomu, YomuError, degraded_for_dry_run_errors, degraded_for_index};

impl Yomu {
    pub fn index(&self, opts: IndexRunOptions, json: bool) -> Result<String, YomuError> {
        let chunk_result =
            indexer::run_chunk_only_index(&self.conn, &self.root, opts.exclude_vendor)?;
        self.embed_pending()?;
        let stats = self.with_db(storage::get_stats)?;

        if json {
            let (degraded, notes) = degraded_for_index(chunk_result.files_errored, &stats);
            return Ok(format_index_json(&chunk_result, &stats, degraded, notes));
        }

        let mut text = format!(
            "Indexing complete: {} files chunked, {} chunks created, {} files skipped (unchanged), {} errors",
            chunk_result.files_processed,
            chunk_result.chunks_created,
            chunk_result.files_skipped,
            chunk_result.files_errored,
        );
        if let Some(note) = format_coverage_note(&stats) {
            text.push_str(&note);
        }
        Ok(text)
    }

    pub fn dry_run_index(&self, opts: IndexRunOptions, json: bool) -> Result<String, YomuError> {
        let preview =
            indexer::dry_run_index(&self.conn, &self.root, opts.force, opts.exclude_vendor)?;

        if json {
            let (degraded, notes) = degraded_for_dry_run_errors(preview.files_errored);
            return Ok(format_dry_run_json(&preview, degraded, notes));
        }

        let mut text = format!(
            "Dry run: {} files to process, {} files unchanged (skip), {} total files",
            preview.files_to_process, preview.files_to_skip, preview.total_files,
        );
        if preview.files_errored > 0 {
            text.push_str(&format!(", {} errors", preview.files_errored));
        }
        if preview.orphans_to_remove > 0 {
            text.push_str(&format!(
                ", {} orphaned files to remove",
                preview.orphans_to_remove
            ));
        }
        Ok(text)
    }

    pub fn rebuild(&self, opts: IndexRunOptions, json: bool) -> Result<String, YomuError> {
        let chunk_result =
            indexer::run_chunk_only_index_force(&self.conn, &self.root, opts.exclude_vendor)?;
        self.embed_pending()?;
        let stats = self.with_db(storage::get_stats)?;

        if json {
            let (degraded, notes) = degraded_for_index(chunk_result.files_errored, &stats);
            return Ok(format_rebuild_json(&chunk_result, &stats, degraded, notes));
        }

        let mut text = format!(
            "Rebuild complete: {} files chunked, {} chunks created, {} errors",
            chunk_result.files_processed, chunk_result.chunks_created, chunk_result.files_errored,
        );
        if let Some(note) = format_coverage_note(&stats) {
            text.push_str(&note);
        }
        Ok(text)
    }

    /// Embeds all pending chunks with progress spinners. Errors out when the
    /// model is unavailable so callers never silently leave a chunk-only index.
    fn embed_pending(&self) -> Result<(), YomuError> {
        let pending = self.with_db(|conn| {
            let stats = storage::get_stats(conn)?;
            Ok(stats
                .embeddable_chunks
                .saturating_sub(stats.embedded_chunks))
        })?;

        embed_with_spinners(
            pending,
            |_| {
                self.try_embedder_arc().map_err(|reason| {
                    // `Disabled` can no longer occur (yomu never disables
                    // embedding); it folds into the generic arm rather than an
                    // unreachable! that diff-coverage would flag as untested.
                    let msg = match reason {
                        DegradedReason::NotInstalled => {
                            "embedding model not installed; run `yomu model download` to enable semantic search"
                        }
                        _ => "embedding model unavailable",
                    };
                    YomuError::EmbedderUnavailable(msg.to_owned())
                })
            },
            |r: &indexer::EmbedResult| format!("Embedded {} chunks", r.chunks_embedded),
            |model: Arc<dyn Embed>, update| {
                indexer::run_incremental_embed_with_progress(
                    &self.conn,
                    model.as_ref(),
                    u32::MAX,
                    None,
                    |n| update(&format!("Embedding... {n}/{pending} chunks")),
                )
                .map_err(YomuError::from)
            },
        )?;
        Ok(())
    }
}
