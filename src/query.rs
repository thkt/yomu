//! Semantic search over indexed code chunks.

use std::sync::Arc;

use parking_lot::Mutex;

use crate::indexer::embedder::{Embed, EmbedError};
use crate::storage::{self, Db, SearchResult, StorageError};

#[derive(Debug, thiserror::Error)]
pub enum QueryError {
    #[error("search query failed: {0}")]
    Storage(#[from] StorageError),
    #[error("search embedding failed: {0}")]
    Embed(#[from] EmbedError),
    #[error("internal task failed: {0}")]
    Internal(String),
}

impl From<tokio::task::JoinError> for QueryError {
    fn from(e: tokio::task::JoinError) -> Self {
        Self::Internal(e.to_string())
    }
}

/// Search for code chunks semantically similar to `query`.
///
/// Embeds the query string, then finds the nearest neighbors in the vector index.
/// Returns up to `limit` results starting from `offset`.
pub async fn search(
    conn: Arc<Mutex<Db>>,
    embedder: &(impl Embed + ?Sized),
    query: &str,
    limit: u32,
    offset: u32,
) -> Result<Vec<SearchResult>, QueryError> {
    let query_embedding = embedder.embed_query(query).await?;
    let results = tokio::task::spawn_blocking(move || {
        let conn = conn.lock();
        storage::search_similar(&conn, &query_embedding, limit, offset)
    })
    .await?;
    Ok(results?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexer::embedder::{Embedder, MockEmbedder, EMBEDDING_DIMS};

    #[test]
    fn query_error_from_storage_error() {
        let se = StorageError::Sqlite(rusqlite::Error::QueryReturnedNoRows);
        let qe: QueryError = se.into();
        assert!(qe.to_string().contains("Query returned no rows"));
    }

    #[test]
    fn query_error_from_embed_error() {
        let ee = EmbedError::ApiKeyNotSet;
        let qe: QueryError = ee.into();
        assert!(qe.to_string().contains("GEMINI_API_KEY"));
    }

    #[tokio::test]
    async fn search_with_mock_embedder() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = storage::open_db(&db_path).unwrap();

        let mut emb = vec![0.0_f32; EMBEDDING_DIMS as usize];
        emb[0] = 1.0;
        storage::insert_chunk(
            &conn,
            "src/Button.tsx",
            &storage::NewChunk {
                chunk_type: &storage::ChunkType::Component,
                name: Some("Button"),
                content: "function Button() { return <div/>; }",
                start_line: 1, end_line: 3,
            },
            "hash1",
            &emb,
        ).unwrap();

        let conn = Arc::new(Mutex::new(conn));
        let results = search(conn, &MockEmbedder, "button", 10, 0).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk.name.as_deref(), Some("Button"));
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY"]
    async fn search_returns_results() {
        let embedder = Embedder::from_env(reqwest::Client::new()).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = storage::open_db(&db_path).unwrap();

        let embedding = embedder.embed_query("Button component").await.unwrap();
        storage::insert_chunk(
            &conn,
            "src/Button.tsx",
            &storage::NewChunk {
                chunk_type: &storage::ChunkType::Component,
                name: Some("Button"),
                content: "function Button() { return <div/>; }",
                start_line: 1, end_line: 3,
            },
            "hash1",
            &embedding,
        ).unwrap();

        let conn = Arc::new(Mutex::new(conn));
        let results = search(conn, &embedder, "button", 10, 0).await.unwrap();
        assert!(!results.is_empty(), "expected at least one result");
    }
}
