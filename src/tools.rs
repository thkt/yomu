//! MCP tool handlers for explorer, index, and status.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use parking_lot::Mutex;
use reqwest::Client;
use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::*,
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::config;
use crate::indexer::{self, embedder::{Embed, Embedder}};
use crate::query;
use crate::storage;

#[derive(Debug, thiserror::Error)]
pub(crate) enum YomuError {
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("storage error: {0}")]
    Storage(#[from] storage::StorageError),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Clone)]
pub struct Yomu {
    conn: Arc<Mutex<storage::Db>>,
    embedder: Option<Arc<dyn Embed>>,
    root: PathBuf,
    auto_indexed: Arc<AtomicBool>,
    tool_router: ToolRouter<Self>,
}

#[derive(Deserialize, JsonSchema)]
pub struct ExplorerParams {
    /// Natural language query describing what you're looking for. Examples: "form validation logic", "authentication hooks", "button component with loading state"
    pub query: String,
    /// Maximum number of results to return (default: 10)
    pub limit: Option<u32>,
    /// Number of results to skip (default: 0)
    pub offset: Option<u32>,
}

#[derive(Deserialize, JsonSchema)]
pub struct IndexParams {
    /// Force full rebuild (default: false, uses incremental update)
    pub force: Option<bool>,
}

#[tool_router]
impl Yomu {
    /// Create a new Yomu server, detecting the project root from cwd.
    ///
    /// Prefer [`with_root`](Self::with_root) when the root path is known.
    pub fn new() -> Result<Self, YomuError> {
        let cwd = std::env::current_dir().map_err(|e| {
            std::io::Error::new(e.kind(), format!("cannot determine current directory: {e}"))
        })?;
        let root = config::detect_root(&cwd);
        Self::with_root(root)
    }

    pub fn with_root(root: PathBuf) -> Result<Self, YomuError> {
        tracing::info!(root = %root.display(), "Detected project root");
        let db_path = root.join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path)?;

        let http = Client::builder()
            .timeout(Duration::from_secs(60))
            .build()?;

        // Intentional: .ok() so the server starts even without GEMINI_API_KEY.
        // Tools that need embeddings will return a clear error via require_embedder().
        let embedder: Option<Arc<dyn Embed>> = match Embedder::from_env(http) {
            Ok(e) => Some(Arc::new(e) as _),
            Err(e) => {
                tracing::warn!("Embedder unavailable: {e}. explorer and index tools will not work.");
                None
            }
        };

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            embedder,
            root,
            auto_indexed: Arc::new(AtomicBool::new(false)),
            tool_router: Self::tool_router(),
        })
    }

    #[tool(
        name = "explorer",
        description = "Semantic code search for frontend projects. Finds components, hooks, types, and patterns by meaning, not just text matching. Automatically builds the index on first use. Returns ranked results with file paths, line ranges, and code snippets."
    )]
    async fn explorer(
        &self,
        Parameters(params): Parameters<ExplorerParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.query.is_empty() {
            return Err(McpError::invalid_params("query must not be empty", None));
        }

        let embedder = self.require_embedder()?;
        let limit = params.limit.unwrap_or(10).min(100);
        let offset = params.offset.unwrap_or(0).min(500);
        if params.limit.is_some_and(|l| l > 100) || params.offset.is_some_and(|o| o > 500) {
            tracing::debug!(limit, offset, "Parameters clamped to maximum");
        }

        tracing::debug!(query = %params.query, limit, offset, "explorer search request");

        let stats = self.with_db(storage::get_stats).await?;
        if stats.total_chunks == 0
            && self
                .auto_indexed
                .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
        {
            tracing::info!("Index is empty, auto-indexing before search");
            if let Err(e) =
                indexer::run_index(Arc::clone(&self.conn), &self.root, embedder, false).await
            {
                self.auto_indexed.store(false, Ordering::SeqCst);
                return Err(McpError::internal_error(e.to_string(), None));
            }
        }

        let results =
            query::search(Arc::clone(&self.conn), embedder, &params.query, limit, offset)
                .await
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        if results.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(
                "No results found.",
            )]));
        }

        let (imports_map, siblings_map) = self.fetch_enrichment_context(&results).await?;
        let text = format_results_grouped(&results, &imports_map, &siblings_map);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        name = "index",
        description = "Build or update the search index for the current project. Scans frontend files (TS, TSX, JS, JSX, CSS, HTML), splits them into semantic chunks using AST parsing, and generates embeddings."
    )]
    async fn index(
        &self,
        Parameters(params): Parameters<IndexParams>,
    ) -> Result<CallToolResult, McpError> {
        let embedder = self.require_embedder()?;
        let force = params.force.unwrap_or(false);

        let result = indexer::run_index(Arc::clone(&self.conn), &self.root, embedder, force)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let text = format!(
            "Indexing complete: {} files processed, {} chunks created, {} files skipped (unchanged), {} files errored",
            result.files_processed, result.chunks_created, result.files_skipped, result.files_errored
        );
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        name = "status",
        description = "Show index statistics: number of indexed files, chunks, and last update time."
    )]
    async fn status(&self) -> Result<CallToolResult, McpError> {
        let stats = self.with_db(storage::get_stats).await?;

        let text = format!(
            "Index status:\n  Files: {}\n  Chunks: {}\n  Last indexed: {}",
            stats.total_files,
            stats.total_chunks,
            stats.last_indexed_at.as_deref().unwrap_or("never")
        );
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

impl Yomu {
    fn require_embedder(&self) -> Result<&dyn Embed, McpError> {
        self.embedder.as_deref().ok_or_else(|| {
            McpError::internal_error(
                "GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey",
                None,
            )
        })
    }

    async fn fetch_enrichment_context(
        &self,
        results: &[storage::SearchResult],
    ) -> Result<
        (
            std::collections::HashMap<String, String>,
            std::collections::HashMap<String, Vec<storage::SiblingInfo>>,
        ),
        McpError,
    > {
        let conn = Arc::clone(&self.conn);
        let unique_paths: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            results
                .iter()
                .filter(|r| seen.insert(&r.chunk.file_path))
                .map(|r| r.chunk.file_path.clone())
                .collect()
        };
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            let path_refs: Vec<&str> = unique_paths.iter().map(|s| s.as_str()).collect();
            let imports = storage::get_file_contexts(&conn, &path_refs)?;
            let siblings = storage::get_file_siblings(&conn, &path_refs)?;
            Ok::<_, storage::StorageError>((imports, siblings))
        })
        .await
        .map_err(|e| McpError::internal_error(format!("internal task failed: {e}"), None))?
        .map_err(|e| McpError::internal_error(e.to_string(), None))
    }

    /// Run a blocking storage operation on the connection.
    async fn with_db<T, F>(&self, f: F) -> Result<T, McpError>
    where
        F: FnOnce(&storage::Db) -> Result<T, storage::StorageError> + Send + 'static,
        T: Send + 'static,
    {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock();
            f(&conn)
        })
        .await
        .map_err(|e| McpError::internal_error(format!("internal task failed: {e}"), None))?
        .map_err(|e| McpError::internal_error(e.to_string(), None))
    }
}

fn format_results_grouped(
    results: &[storage::SearchResult],
    imports_map: &std::collections::HashMap<String, String>,
    siblings_map: &std::collections::HashMap<String, Vec<storage::SiblingInfo>>,
) -> String {
    let mut groups: std::collections::HashMap<&str, Vec<(usize, &storage::SearchResult)>> =
        std::collections::HashMap::new();
    for (i, result) in results.iter().enumerate() {
        groups
            .entry(&result.chunk.file_path)
            .or_default()
            .push((i, result));
    }

    let mut sorted: Vec<_> = groups.into_iter().collect();
    sorted.sort_by(|a, b| {
        let best = |items: &[(usize, &storage::SearchResult)]| {
            items
                .iter()
                .map(|(_, r)| r.distance)
                .fold(f32::INFINITY, f32::min)
        };
        best(&a.1)
            .partial_cmp(&best(&b.1))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut output = String::new();
    for (file_path, chunks) in &sorted {
        format_file_group(&mut output, file_path, chunks, imports_map, siblings_map);
    }
    output
}

fn format_file_group(
    output: &mut String,
    file_path: &str,
    chunks: &[(usize, &storage::SearchResult)],
    imports_map: &std::collections::HashMap<String, String>,
    siblings_map: &std::collections::HashMap<String, Vec<storage::SiblingInfo>>,
) {
    output.push_str(&format!("## {}\n", file_path));

    if let Some(imports_text) = imports_map.get(file_path) {
        let items: Vec<&str> = imports_text.split('\n').filter(|s| !s.is_empty()).collect();
        if !items.is_empty() {
            output.push_str(&format!("Imports: {}\n", items.join(", ")));
        }
    }

    if let Some(siblings) = siblings_map.get(file_path) {
        let result_ranges: std::collections::HashSet<(u32, u32)> = chunks
            .iter()
            .map(|(_, r)| (r.chunk.start_line, r.chunk.end_line))
            .collect();
        let filtered: Vec<String> = siblings
            .iter()
            .filter(|s| !result_ranges.contains(&(s.start_line, s.end_line)))
            .map(|s| {
                let name = s.name.as_deref().unwrap_or("(unnamed)");
                format!("{} [{}]", name, s.chunk_type.as_str())
            })
            .collect();
        if !filtered.is_empty() {
            output.push_str(&format!("Siblings: {}\n", filtered.join(", ")));
        }
    }

    output.push('\n');

    for (rank, result) in chunks {
        let chunk = &result.chunk;
        let name = chunk.name.as_deref().unwrap_or("(unnamed)");
        let similarity = 1.0 / (1.0 + result.distance);
        output.push_str(&format!(
            "{}. {} [{}] — {}:{} (similarity: {:.2})\n",
            rank + 1,
            name,
            chunk.chunk_type.as_str(),
            chunk.start_line,
            chunk.end_line,
            similarity,
        ));
        output.push_str(&chunk.content);
        output.push_str("\n\n");
    }
}

#[tool_handler]
impl ServerHandler for Yomu {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(
                "yomu: Semantic code search for frontend projects (TS, TSX, JS, CSS, HTML). \
                 Prefer 'explorer' over grep/glob when: (1) you don't know exact names or keywords, \
                 (2) you want to find code by concept (e.g. \"form validation\", \"auth flow\"), \
                 (3) you need to discover related components/hooks/types across the codebase. \
                 Use grep/glob instead when you need exact string matching or known file paths. \
                 'index' rebuilds the search index (usually not needed — explorer auto-indexes on first use). \
                 'status' shows index statistics."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn test_db() -> (storage::Db, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = storage::open_db(&db_path).unwrap();
        (conn, dir)
    }

    fn test_yomu() -> (Yomu, tempfile::TempDir) {
        let (conn, dir) = test_db();
        let y = Yomu {
            conn: Arc::new(Mutex::new(conn)),
            embedder: None,
            root: dir.path().to_path_buf(),
            auto_indexed: Arc::new(AtomicBool::new(false)),
            tool_router: Yomu::tool_router(),
        };
        (y, dir)
    }

    #[test]
    fn tool_router_has_three_tools() {
        let router = Yomu::tool_router();
        let names: Vec<&str> = router.map.keys().map(|k| k.as_ref()).collect();
        assert!(names.contains(&"explorer"), "missing explorer: {names:?}");
        assert!(names.contains(&"index"), "missing index: {names:?}");
        assert!(names.contains(&"status"), "missing status: {names:?}");
        assert_eq!(names.len(), 3, "expected 3 tools, got {names:?}");
    }

    #[tokio::test]
    async fn explorer_rejects_empty_query() {
        let (y, _dir) = test_yomu();
        let params = Parameters(ExplorerParams {
            query: String::new(),
            limit: None,
            offset: None,
        });
        let err = y.explorer(params).await.unwrap_err();
        assert!(
            err.message.contains("empty"),
            "expected empty error, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn status_returns_empty_stats() {
        let (y, _dir) = test_yomu();
        let result = y.status().await.unwrap();
        let text = &result.content[0].as_text().unwrap().text;
        assert!(text.contains("Files: 0"), "expected 0 files, got: {text}");
        assert!(
            text.contains("Chunks: 0"),
            "expected 0 chunks, got: {text}"
        );
        assert!(text.contains("never"), "expected 'never', got: {text}");
    }

    #[tokio::test]
    async fn index_requires_api_key() {
        let (y, _dir) = test_yomu();
        let params = Parameters(IndexParams { force: None });
        let err = y.index(params).await.unwrap_err();
        assert!(
            err.message.contains("GEMINI_API_KEY"),
            "expected API key error, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn explorer_requires_api_key() {
        let (y, _dir) = test_yomu();
        let params = Parameters(ExplorerParams {
            query: "test query".to_string(),
            limit: None,
            offset: None,
        });
        let err = y.explorer(params).await.unwrap_err();
        assert!(
            err.message.contains("GEMINI_API_KEY"),
            "expected API key error, got: {}",
            err.message
        );
    }

    #[test]
    fn format_results_grouped_renders_file_header_and_context() {
        let results = vec![storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/Button.tsx".to_string(),
                chunk_type: storage::ChunkType::Component,
                name: Some("Button".to_string()),
                content: "function Button() { return <div/>; }".to_string(),
                start_line: 5,
                end_line: 7,
            },
            distance: 0.15,
        }];
        let imports_map = HashMap::from([(
            "src/Button.tsx".to_string(),
            "import React from 'react'".to_string(),
        )]);
        let siblings_map = HashMap::from([(
            "src/Button.tsx".to_string(),
            vec![storage::SiblingInfo {
                name: Some("ButtonProps".to_string()),
                chunk_type: storage::ChunkType::TypeDef,
                start_line: 1,
                end_line: 3,
            }],
        )]);
        let text = format_results_grouped(&results, &imports_map, &siblings_map);
        assert!(text.contains("## src/Button.tsx"), "missing file header: {text}");
        assert!(
            text.contains("Imports: import React from 'react'"),
            "missing imports: {text}"
        );
        assert!(
            text.contains("Siblings: ButtonProps [type_def]"),
            "missing siblings: {text}"
        );
        assert!(text.contains("Button"), "missing chunk name: {text}");
        assert!(text.contains("0.87"), "missing similarity: {text}");
    }

    #[test]
    fn format_results_grouped_groups_same_file_chunks() {
        let results = vec![
            storage::SearchResult {
                chunk: storage::Chunk {
                    file_path: "src/Form.tsx".to_string(),
                    chunk_type: storage::ChunkType::Component,
                    name: Some("Form".to_string()),
                    content: "function Form() {}".to_string(),
                    start_line: 1,
                    end_line: 5,
                },
                distance: 0.1,
            },
            storage::SearchResult {
                chunk: storage::Chunk {
                    file_path: "src/Form.tsx".to_string(),
                    chunk_type: storage::ChunkType::Hook,
                    name: Some("useForm".to_string()),
                    content: "function useForm() {}".to_string(),
                    start_line: 7,
                    end_line: 10,
                },
                distance: 0.2,
            },
        ];
        let empty_imports: HashMap<String, String> = HashMap::new();
        let empty_siblings: HashMap<String, Vec<storage::SiblingInfo>> = HashMap::new();
        let text = format_results_grouped(&results, &empty_imports, &empty_siblings);
        assert_eq!(
            text.matches("## src/Form.tsx").count(),
            1,
            "expected one file header: {text}"
        );
        assert!(text.contains("Form"), "missing Form: {text}");
        assert!(text.contains("useForm"), "missing useForm: {text}");
    }

    #[test]
    fn format_results_grouped_deduplicates_siblings() {
        let results = vec![storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/A.tsx".to_string(),
                chunk_type: storage::ChunkType::Component,
                name: Some("A".to_string()),
                content: "function A() {}".to_string(),
                start_line: 5,
                end_line: 7,
            },
            distance: 0.1,
        }];
        let empty_imports = HashMap::new();
        let siblings_map = HashMap::from([(
            "src/A.tsx".to_string(),
            vec![
                storage::SiblingInfo {
                    name: Some("A".to_string()),
                    chunk_type: storage::ChunkType::Component,
                    start_line: 5,
                    end_line: 7,
                },
                storage::SiblingInfo {
                    name: Some("AProps".to_string()),
                    chunk_type: storage::ChunkType::TypeDef,
                    start_line: 1,
                    end_line: 3,
                },
            ],
        )]);
        let text = format_results_grouped(&results, &empty_imports, &siblings_map);
        assert!(
            text.contains("AProps [type_def]"),
            "sibling should be included: {text}"
        );
        let siblings_line = text.lines().find(|l| l.starts_with("Siblings:")).unwrap();
        assert!(
            !siblings_line.contains("A [component]"),
            "search result should be excluded from siblings: {siblings_line}"
        );
    }

    #[test]
    fn format_results_grouped_omits_empty_imports() {
        let results = vec![storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/A.tsx".to_string(),
                chunk_type: storage::ChunkType::Component,
                name: Some("A".to_string()),
                content: "code".to_string(),
                start_line: 1,
                end_line: 3,
            },
            distance: 0.1,
        }];
        let imports_map = HashMap::from([("src/A.tsx".to_string(), String::new())]);
        let empty: HashMap<String, Vec<storage::SiblingInfo>> = HashMap::new();
        let text = format_results_grouped(&results, &imports_map, &empty);
        assert!(!text.contains("Imports:"), "empty imports should be omitted: {text}");
    }

    #[test]
    fn format_results_grouped_omits_empty_siblings() {
        let results = vec![storage::SearchResult {
            chunk: storage::Chunk {
                file_path: "src/A.tsx".to_string(),
                chunk_type: storage::ChunkType::Component,
                name: Some("A".to_string()),
                content: "code".to_string(),
                start_line: 1,
                end_line: 3,
            },
            distance: 0.1,
        }];
        let empty_imports: HashMap<String, String> = HashMap::new();
        let siblings_map = HashMap::from([("src/A.tsx".to_string(), vec![])]);
        let text = format_results_grouped(&results, &empty_imports, &siblings_map);
        assert!(
            !text.contains("Siblings:"),
            "empty siblings should be omitted: {text}"
        );
    }

    #[test]
    fn format_results_grouped_sorts_files_by_best_similarity() {
        let results = vec![
            storage::SearchResult {
                chunk: storage::Chunk {
                    file_path: "src/B.tsx".to_string(),
                    chunk_type: storage::ChunkType::Component,
                    name: Some("B".to_string()),
                    content: "code B".to_string(),
                    start_line: 1,
                    end_line: 3,
                },
                distance: 0.5,
            },
            storage::SearchResult {
                chunk: storage::Chunk {
                    file_path: "src/A.tsx".to_string(),
                    chunk_type: storage::ChunkType::Component,
                    name: Some("A".to_string()),
                    content: "code A".to_string(),
                    start_line: 1,
                    end_line: 3,
                },
                distance: 0.1,
            },
        ];
        let empty: HashMap<String, String> = HashMap::new();
        let empty_siblings: HashMap<String, Vec<storage::SiblingInfo>> = HashMap::new();
        let text = format_results_grouped(&results, &empty, &empty_siblings);
        let a_pos = text.find("## src/A.tsx").unwrap();
        let b_pos = text.find("## src/B.tsx").unwrap();
        assert!(
            a_pos < b_pos,
            "A (better similarity) should come before B: {text}"
        );
    }

    #[tokio::test]
    async fn explorer_auto_indexes_empty_db() {
        let dir = tempfile::tempdir().unwrap();
        let src_dir = dir.path().join("src");
        std::fs::create_dir_all(&src_dir).unwrap();
        std::fs::write(
            src_dir.join("Button.tsx"),
            "function Button() { return <div/>; }",
        )
        .unwrap();

        let db_path = dir.path().join(".yomu").join("index.db");
        let conn = storage::open_db(&db_path).unwrap();
        let y = Yomu {
            conn: Arc::new(Mutex::new(conn)),
            embedder: Some(Arc::new(crate::indexer::embedder::MockEmbedder)),
            root: dir.path().to_path_buf(),
            auto_indexed: Arc::new(AtomicBool::new(false)),
            tool_router: Yomu::tool_router(),
        };

        let params = Parameters(ExplorerParams {
            query: "button component".to_string(),
            limit: None,
            offset: None,
        });
        let result = y.explorer(params).await.unwrap();
        let text = &result.content[0].as_text().unwrap().text;
        assert!(
            !text.contains("No results found"),
            "expected results after auto-index, got: {text}"
        );
        assert!(text.contains("Button"), "expected Button in results, got: {text}");
    }

    #[tokio::test]
    async fn status_returns_counts_after_insert() {
        let (conn, _dir) = test_db();
        let embedding = vec![0.0_f32; 768];
        storage::insert_chunk(
            &conn,
            "src/A.tsx",
            &storage::NewChunk {
                chunk_type: &storage::ChunkType::Component,
                name: Some("A"),
                content: "code",
                start_line: 1,
                end_line: 5,
            },
            "h1",
            &embedding,
        )
        .unwrap();

        let y = Yomu {
            conn: Arc::new(Mutex::new(conn)),
            embedder: None,
            root: PathBuf::from("/tmp"),
            auto_indexed: Arc::new(AtomicBool::new(false)),
            tool_router: Yomu::tool_router(),
        };
        let result = y.status().await.unwrap();
        let text = &result.content[0].as_text().unwrap().text;
        assert!(text.contains("Files: 1"), "expected 1 file, got: {text}");
        assert!(
            text.contains("Chunks: 1"),
            "expected 1 chunk, got: {text}"
        );
    }
}
