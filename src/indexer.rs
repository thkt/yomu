mod chunk_only;
pub mod chunker;
mod embed;
pub mod injection;
mod source_kind;
pub mod walker;

pub use crate::storage::SourceKind;

use std::io;

use rurico::embed::EmbedError;

use crate::storage::StorageError;

pub use chunk_only::{dry_run_index, run_chunk_only_index, run_chunk_only_index_force};
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

#[cfg(test)]
mod tests;
