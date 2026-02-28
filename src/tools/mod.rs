//! MCP tool handlers for explorer, index, and status.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

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

const DEFAULT_EMBED_BUDGET: u32 = 50;

fn max_chunks_per_budget() -> u32 {
    static VALUE: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        match std::env::var("YOMU_EMBED_BUDGET") {
            Ok(v) => match v.parse() {
                Ok(n) => n,
                Err(_) => {
                    tracing::warn!(value = %v, "Invalid YOMU_EMBED_BUDGET, using default {DEFAULT_EMBED_BUDGET}");
                    DEFAULT_EMBED_BUDGET
                }
            },
            Err(_) => DEFAULT_EMBED_BUDGET,
        }
    })
}

#[derive(Debug, PartialEq)]
enum IndexState {
    Empty,
    ChunkedOnly,
    PartiallyEmbedded,
    FullyEmbedded,
}

fn determine_index_state(stats: &storage::IndexStatus) -> IndexState {
    if stats.total_chunks == 0 {
        IndexState::Empty
    } else if stats.embedded_chunks == 0 {
        IndexState::ChunkedOnly
    } else if stats.embedded_chunks < stats.total_chunks {
        IndexState::PartiallyEmbedded
    } else {
        IndexState::FullyEmbedded
    }
}

fn format_no_results_message(stats: &storage::IndexStatus) -> String {
    format!(
        "No results found. Index coverage: {}/{} chunks ({}%). Use 'index' for full coverage or repeat search to expand.",
        stats.embedded_chunks, stats.total_chunks, stats.embed_percentage()
    )
}

#[derive(Debug, thiserror::Error)]
pub enum YomuError {
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
    auto_index_failures: Arc<AtomicU32>,
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

#[derive(Deserialize, JsonSchema)]
pub struct ImpactParams {
    /// File path to analyze, relative to project root. Example: "src/hooks/useAuth.ts"
    pub target: String,
    /// Filter to dependents that reference this specific symbol. Example: "useAuth"
    pub symbol: Option<String>,
    /// Maximum depth for transitive dependency traversal (default: 3, max: 10)
    pub depth: Option<u32>,
}

#[tool_router]
impl Yomu {
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

        let http = Client::builder().build()?;

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
            auto_index_failures: Arc::new(AtomicU32::new(0)),
            tool_router: Self::tool_router(),
        })
    }

    #[tool(
        name = "explorer",
        description = "Semantic code search for frontend projects. Finds components, hooks, types, and patterns by meaning, not just text matching. Automatically builds the index on first use. Returns ranked results with full code chunks, file imports, and sibling definitions."
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

        let type_hints = query::extract_type_hints(&params.query);
        let hints_ref = if type_hints.is_empty() { None } else { Some(type_hints.as_slice()) };

        let stats = self.with_db(storage::get_stats).await?;
        let state = determine_index_state(&stats);
        let embed_error = self.ensure_indexed(embedder, state, hints_ref).await?;

        let results =
            query::search(Arc::clone(&self.conn), embedder, &params.query, limit, offset)
                .await
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        if results.is_empty() {
            let stats = self.with_db(storage::get_stats).await?;
            let msg = match embed_error {
                Some(ref err) => format!(
                    "{}\n\nNote: embedding failed: {err}",
                    format_no_results_message(&stats)
                ),
                None => format_no_results_message(&stats),
            };
            return Ok(CallToolResult::success(vec![Content::text(msg)]));
        }

        let (imports_map, siblings_map) = self.fetch_enrichment_context(&results).await?;
        let text = format_results_grouped(&results, &imports_map, &siblings_map);
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        name = "index",
        description = "Build or update the search index for the current project. Scans frontend files (TS, TSX, JS, JSX, CSS, HTML), splits them into semantic chunks using AST parsing, and embeds them incrementally. With force=true, does a full rebuild with re-embedding."
    )]
    async fn index(
        &self,
        Parameters(params): Parameters<IndexParams>,
    ) -> Result<CallToolResult, McpError> {
        let embedder = self.require_embedder()?;
        let force = params.force.unwrap_or(false);

        if force {
            let result = indexer::run_index(Arc::clone(&self.conn), &self.root, embedder, true)
                .await
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;
            self.auto_index_failures.store(0, Ordering::SeqCst);
            self.auto_indexed.store(false, Ordering::SeqCst);
            let text = format!(
                "Indexing complete: {} files processed, {} chunks created, {} files skipped (unchanged), {} files errored",
                result.files_processed, result.chunks_created, result.files_skipped, result.files_errored
            );
            return Ok(CallToolResult::success(vec![Content::text(text)]));
        }

        let chunk_result = indexer::run_chunk_only_index(Arc::clone(&self.conn), &self.root)
            .await
            .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        let embed_result =
            indexer::run_incremental_embed(Arc::clone(&self.conn), embedder, u32::MAX, None)
                .await
                .map_err(|e| McpError::internal_error(e.to_string(), None))?;

        self.auto_index_failures.store(0, Ordering::SeqCst);
        self.auto_indexed.store(false, Ordering::SeqCst);

        let text = format!(
            "Indexing complete: {} files chunked, {} chunks created, {} files skipped, {} chunks embedded, {} files embedded",
            chunk_result.files_processed,
            chunk_result.chunks_created,
            chunk_result.files_skipped,
            embed_result.chunks_embedded,
            embed_result.files_completed,
        );
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        name = "impact",
        description = "Analyze the impact of changes to a file or symbol. Shows which files depend on the target, both directly and transitively. Use the symbol parameter to filter to a specific export. Requires an existing index."
    )]
    async fn impact(
        &self,
        Parameters(params): Parameters<ImpactParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.target.is_empty() {
            return Err(McpError::invalid_params("target must not be empty", None));
        }

        let stats = self.with_db(storage::get_stats).await?;
        if stats.total_chunks == 0 {
            return Ok(CallToolResult::success(vec![Content::text(
                "Index is empty. Run the `index` tool first, or use `explorer` which auto-indexes on first use.",
            )]));
        }

        let (file_path, parsed_symbol) = parse_impact_target(&params.target);
        let symbol_filter = params.symbol.as_deref().or(parsed_symbol);
        let max_depth = params.depth.unwrap_or(3).min(10);
        let file_path_owned = file_path.to_string();

        let dependents = self
            .with_db(move |conn| {
                storage::get_transitive_dependents(conn, &file_path_owned, max_depth)
            })
            .await?;

        if dependents.is_empty() {
            return Ok(CallToolResult::success(vec![Content::text(format!(
                "No dependents found for `{}`.",
                params.target
            ))]));
        }

        let text = if let Some(symbol) = symbol_filter {
            let file_path_owned = file_path.to_string();
            let symbol_owned = symbol.to_string();
            let filtered = self
                .with_db(move |conn| {
                    storage::get_symbol_dependents(conn, &file_path_owned, &symbol_owned)
                })
                .await?;

            format_impact_results(&params.target, &filtered, &dependents)
        } else {
            format_impact_all(&params.target, &dependents)
        };

        Ok(CallToolResult::success(vec![Content::text(text)]))
    }

    #[tool(
        name = "status",
        description = "Show index statistics: number of indexed files, chunks, embedded chunks, references, and last update time."
    )]
    async fn status(&self) -> Result<CallToolResult, McpError> {
        let stats = self.with_db(storage::get_stats).await?;
        let ref_count = self.with_db(storage::get_reference_count).await?;

        let text = format!(
            "Index status:\n  Files: {}\n  Chunks: {}\n  Embedded: {}/{} ({}%)\n  References: {}\n  Last indexed: {}",
            stats.total_files,
            stats.total_chunks,
            stats.embedded_chunks,
            stats.total_chunks,
            stats.embed_percentage(),
            ref_count,
            stats.last_indexed_at.as_deref().unwrap_or("never")
        );
        Ok(CallToolResult::success(vec![Content::text(text)]))
    }
}

impl Yomu {
    /// Auto-index if needed and run incremental embedding.
    /// Returns embed error message if embedding failed but search can still proceed.
    async fn ensure_indexed(
        &self,
        embedder: &dyn Embed,
        state: IndexState,
        type_hints: Option<&[storage::ChunkType]>,
    ) -> Result<Option<String>, McpError> {
        let needs_embed = match state {
            IndexState::Empty => {
                if self.auto_index_failures.load(Ordering::SeqCst) < 3
                    && self
                        .auto_indexed
                        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                        .is_ok()
                {
                    tracing::info!("Index is empty, running chunk-only index");
                    if let Err(e) =
                        indexer::run_chunk_only_index(Arc::clone(&self.conn), &self.root).await
                    {
                        let failures = self.auto_index_failures.fetch_add(1, Ordering::SeqCst) + 1;
                        if failures >= 3 {
                            tracing::warn!(failures, "Auto-index failed {failures} times, disabling retries");
                        } else {
                            self.auto_indexed.store(false, Ordering::SeqCst);
                        }
                        return Err(McpError::internal_error(e.to_string(), None));
                    }
                    true
                } else {
                    false
                }
            }
            IndexState::ChunkedOnly | IndexState::PartiallyEmbedded => {
                // Re-chunk to pick up new/changed files before embedding
                if let Err(e) =
                    indexer::run_chunk_only_index(Arc::clone(&self.conn), &self.root).await
                {
                    tracing::warn!(error = %e, "Re-chunking failed, proceeding with existing chunks");
                }
                true
            }
            IndexState::FullyEmbedded => false,
        };

        if needs_embed
            && let Err(e) = indexer::run_incremental_embed(
                Arc::clone(&self.conn),
                embedder,
                max_chunks_per_budget(),
                type_hints,
            )
            .await
        {
            tracing::warn!(error = %e, "Incremental embed failed, searching existing embeddings");
            return Ok(Some(e.to_string()));
        }

        Ok(None)
    }

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

fn parse_impact_target(target: &str) -> (&str, Option<&str>) {
    if let Some(colon_pos) = target.rfind(':')
        && colon_pos > 0
        && colon_pos < target.len() - 1
    {
        let file = &target[..colon_pos];
        let symbol = &target[colon_pos + 1..];
        if !symbol.contains('/') && !symbol.contains('\\') {
            return (file, Some(symbol));
        }
    }
    (target, None)
}

fn format_dependents_by_depth(
    output: &mut String,
    dependents: &[storage::Dependent],
    heading_prefix: &str,
) {
    let mut current_depth = 0;
    for dep in dependents {
        if dep.depth != current_depth {
            current_depth = dep.depth;
            output.push_str(&format!("{heading_prefix} Depth {}\n", current_depth));
        }
        output.push_str(&format!("- {}\n", dep.file_path));
    }
}

fn format_impact_all(target: &str, dependents: &[storage::Dependent]) -> String {
    let mut output = format!("## Impact analysis: `{}`\n\n", target);
    format_dependents_by_depth(&mut output, dependents, "###");
    output.push_str(&format!(
        "\nTotal: {} dependent file(s)\n",
        dependents.len()
    ));
    output
}

fn format_impact_results(
    target: &str,
    symbol_refs: &[String],
    all_dependents: &[storage::Dependent],
) -> String {
    let mut output = format!("## Impact analysis: `{}`\n\n", target);

    if !symbol_refs.is_empty() {
        output.push_str("### Direct symbol references\n");
        for f in symbol_refs {
            output.push_str(&format!("- {}\n", f));
        }
    }

    output.push_str("\n### All transitive dependents\n");
    format_dependents_by_depth(&mut output, all_dependents, "####");
    output.push_str(&format!(
        "\nTotal: {} dependent file(s)\n",
        all_dependents.len()
    ));
    output
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
                .map(|(_, r)| r.score)
                .fold(f32::NEG_INFINITY, f32::max)
        };
        best(&b.1)
            .partial_cmp(&best(&a.1))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut output = String::new();
    for (file_path, chunks) in &sorted {
        format_file_group(&mut output, file_path, chunks, imports_map, siblings_map);
    }
    output
}

fn format_imports_line(imports_map: &std::collections::HashMap<String, String>, file_path: &str) -> Option<String> {
    let imports_text = imports_map.get(file_path)?;
    let items: Vec<&str> = imports_text.split('\n').filter(|s| !s.is_empty()).collect();
    if items.is_empty() { return None; }
    Some(format!("Imports: {}\n", items.join(", ")))
}

fn format_siblings_line(
    siblings_map: &std::collections::HashMap<String, Vec<storage::SiblingInfo>>,
    file_path: &str,
    result_ranges: &std::collections::HashSet<(u32, u32)>,
) -> Option<String> {
    let siblings = siblings_map.get(file_path)?;
    let filtered: Vec<String> = siblings
        .iter()
        .filter(|s| !result_ranges.contains(&(s.start_line, s.end_line)))
        .map(|s| {
            let name = s.name.as_deref().unwrap_or("(unnamed)");
            format!("{} [{}]", name, s.chunk_type.as_str())
        })
        .collect();
    if filtered.is_empty() { return None; }
    Some(format!("Siblings: {}\n", filtered.join(", ")))
}

fn format_file_group(
    output: &mut String,
    file_path: &str,
    chunks: &[(usize, &storage::SearchResult)],
    imports_map: &std::collections::HashMap<String, String>,
    siblings_map: &std::collections::HashMap<String, Vec<storage::SiblingInfo>>,
) {
    output.push_str(&format!("## {}\n", file_path));

    if let Some(line) = format_imports_line(imports_map, file_path) {
        output.push_str(&line);
    }

    let result_ranges: std::collections::HashSet<(u32, u32)> = chunks
        .iter()
        .map(|(_, r)| (r.chunk.start_line, r.chunk.end_line))
        .collect();
    if let Some(line) = format_siblings_line(siblings_map, file_path, &result_ranges) {
        output.push_str(&line);
    }

    output.push('\n');

    for (rank, result) in chunks {
        let chunk = &result.chunk;
        let name = chunk.name.as_deref().unwrap_or("(unnamed)");
        let score_label = format!("(similarity: {:.2})", result.score);
        output.push_str(&format!(
            "{}. {} [{}] — {}:{} {}\n",
            rank + 1,
            name,
            chunk.chunk_type.as_str(),
            chunk.start_line,
            chunk.end_line,
            score_label,
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
                 'explorer' auto-indexes on first use. Results include full code, imports, and sibling \
                 chunks. Use limit/offset for pagination. \
                 'impact' analyzes which files depend on a given file or symbol — use it to understand \
                 the blast radius before modifying code. Pass target=\"src/hooks/useAuth.ts\" for file-level \
                 analysis, or add symbol=\"useAuth\" to filter to a specific export. \
                 'index' rebuilds the search index (usually not needed — explorer auto-indexes on first use). \
                 Without force, it chunks all files then embeds incrementally. With force=true, it does a \
                 full rebuild. \
                 'status' shows index statistics."
                    .into(),
            ),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests;

