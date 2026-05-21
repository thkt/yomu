//! Brief expansion plan: TaskBrief -> forward closure -> chunks -> cap -> topo -> BriefOutput.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

use rusqlite::Connection;
use serde::Serialize;

use crate::injection_check::InjectionCheck;
use crate::storage::{
    Chunk, ChunkType, StorageError, get_chunks_for_files, get_edges_among_files, get_import_counts,
    get_transitive_dependencies,
};

#[derive(Debug, Clone, PartialEq)]
pub enum SeedKind {
    File,
    Symbol,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Seed {
    pub kind: SeedKind,
    pub value: String,
}

#[derive(Debug, Clone)]
pub struct TaskBrief {
    pub task: String,
    pub seeds: Vec<Seed>,
    pub depth: u32,
    pub max_chunks: u32,
    pub max_bytes: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkInclusionReason {
    Seed,
    Forward(u32),
    Sibling,
    ModDecl,
}

impl fmt::Display for ChunkInclusionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Seed => f.write_str("seed"),
            Self::Forward(n) => write!(f, "forward-{n}"),
            Self::Sibling => f.write_str("sibling"),
            Self::ModDecl => f.write_str("mod-decl"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BriefChunk {
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub chunk_type: ChunkType,
    pub content: String,
    pub included_reason: ChunkInclusionReason,
    pub source_kind: Option<String>,
    pub injection_flags: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
pub struct BriefOutput {
    pub chunks: Vec<BriefChunk>,
    pub degraded: bool,
    pub total_chunks: u32,
    pub total_bytes: u32,
}

fn collect_seed_paths(task: &TaskBrief) -> Vec<String> {
    task.seeds
        .iter()
        .filter(|s| s.kind == SeedKind::File)
        .map(|s| s.value.clone())
        .collect()
}

fn collect_closure(
    conn: &Connection,
    seeds: &[String],
    depth: u32,
) -> Result<HashMap<String, u32>, StorageError> {
    let mut depth_by_path: HashMap<String, u32> = HashMap::new();
    for seed in seeds {
        for dep in get_transitive_dependencies(conn, seed, depth)? {
            depth_by_path
                .entry(dep.file_path)
                .and_modify(|d| {
                    if dep.depth < *d {
                        *d = dep.depth;
                    }
                })
                .or_insert(dep.depth);
        }
    }
    Ok(depth_by_path)
}

fn determine_reason(depth: u32) -> ChunkInclusionReason {
    if depth == 0 {
        ChunkInclusionReason::Seed
    } else {
        ChunkInclusionReason::Forward(depth)
    }
}

fn build_brief_chunks(chunks: Vec<Chunk>, depth_by_path: &HashMap<String, u32>) -> Vec<BriefChunk> {
    chunks
        .into_iter()
        .map(|c| {
            let depth = depth_by_path.get(&c.file_path).copied().unwrap_or_else(|| {
                debug_assert!(
                    false,
                    "chunk file_path not in depth_by_path: {}",
                    c.file_path
                );
                0
            });
            BriefChunk {
                file_path: c.file_path,
                start_line: c.start_line,
                end_line: c.end_line,
                chunk_type: c.chunk_type,
                content: c.content,
                included_reason: determine_reason(depth),
                source_kind: c.source_kind,
                injection_flags: c.injection_flags,
            }
        })
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
fn under_cap(chunks: &[BriefChunk], max_chunks: u32, max_bytes: u32) -> bool {
    let count = chunks.len() as u32;
    let bytes: usize = chunks.iter().map(|c| c.content.len()).sum();
    let bytes_capped = u32::try_from(bytes).unwrap_or(u32::MAX);
    count <= max_chunks && bytes_capped <= max_bytes
}

fn deletion_priority(
    chunk: &BriefChunk,
    depth_by_path: &HashMap<String, u32>,
    incoming_counts: &HashMap<String, u32>,
) -> (u32, u32) {
    let depth = depth_by_path.get(&chunk.file_path).copied().unwrap_or(0);
    let incoming = incoming_counts.get(&chunk.file_path).copied().unwrap_or(0);
    (depth, incoming)
}

#[allow(clippy::cast_possible_truncation)]
fn select_drops(
    chunks: &[BriefChunk],
    depth_by_path: &HashMap<String, u32>,
    incoming_counts: &HashMap<String, u32>,
    max_chunks: u32,
    max_bytes: u32,
) -> HashSet<usize> {
    let mut order: Vec<(usize, u32, u32)> = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let (d, e) = deletion_priority(c, depth_by_path, incoming_counts);
            (i, d, e)
        })
        .collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));

    let mut drop_idx: HashSet<usize> = HashSet::new();
    let mut count = chunks.len() as u32;
    let mut bytes: usize = chunks.iter().map(|c| c.content.len()).sum();
    for (idx, _, _) in order {
        if count <= max_chunks && bytes as u32 <= max_bytes {
            break;
        }
        drop_idx.insert(idx);
        count -= 1;
        bytes -= chunks[idx].content.len();
    }
    drop_idx
}

/// Applies BR-001 cap deletion order: drop chunks first by
/// `(seed_distance DESC, incoming_edge_count ASC)`. Pure: takes pre-computed
/// `depth_by_path` and `incoming_counts` so callers can test deterministically
/// without DB access.
pub fn apply_cap(
    chunks: Vec<BriefChunk>,
    depth_by_path: &HashMap<String, u32>,
    incoming_counts: &HashMap<String, u32>,
    max_chunks: u32,
    max_bytes: u32,
) -> Vec<BriefChunk> {
    if under_cap(&chunks, max_chunks, max_bytes) {
        return chunks;
    }
    let drop_idx = select_drops(
        &chunks,
        depth_by_path,
        incoming_counts,
        max_chunks,
        max_bytes,
    );
    chunks
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop_idx.contains(i))
        .map(|(_, c)| c)
        .collect()
}

type DepGraph = (HashMap<String, u32>, HashMap<String, Vec<String>>);

fn build_dep_graph(chunks: &[BriefChunk], edges: &[(String, String)]) -> DepGraph {
    let in_scope: HashSet<&str> = chunks.iter().map(|c| c.file_path.as_str()).collect();
    let mut out_degree: HashMap<String, u32> =
        in_scope.iter().map(|p| ((*p).to_owned(), 0)).collect();
    let mut dependents: HashMap<String, Vec<String>> = HashMap::new();
    for (src, tgt) in edges {
        if src != tgt && in_scope.contains(src.as_str()) && in_scope.contains(tgt.as_str()) {
            *out_degree.entry(src.clone()).or_insert(0) += 1;
            dependents.entry(tgt.clone()).or_default().push(src.clone());
        }
    }
    (out_degree, dependents)
}

fn append_cycle_tail(
    positions: &mut HashMap<String, usize>,
    next_idx: &mut usize,
    all_files: &HashMap<String, u32>,
) {
    let mut leftover: Vec<&String> = all_files
        .keys()
        .filter(|p| !positions.contains_key(*p))
        .collect();
    leftover.sort();
    for p in leftover {
        positions.insert(p.clone(), *next_idx);
        *next_idx += 1;
    }
}

fn kahn_positions(graph: DepGraph) -> HashMap<String, usize> {
    let (mut out_degree, dependents) = graph;
    let mut available: BTreeSet<String> = out_degree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(p, _)| p.clone())
        .collect();
    let mut positions: HashMap<String, usize> = HashMap::new();
    let mut idx = 0;
    while let Some(p) = available.iter().next().cloned() {
        available.remove(&p);
        positions.insert(p.clone(), idx);
        idx += 1;
        if let Some(deps) = dependents.get(&p) {
            for dep in deps {
                if let Some(d) = out_degree.get_mut(dep) {
                    *d -= 1;
                    if *d == 0 {
                        available.insert(dep.clone());
                    }
                }
            }
        }
    }
    append_cycle_tail(&mut positions, &mut idx, &out_degree);
    positions
}

/// Applies BR-002 topological order: dependencies first (depended-upon files
/// come before their dependents), tie-breaking by file_path lex order.
/// Within the same file, chunks keep their `start_line` ordering. Cycles
/// degrade to lex order at the tail of the result.
pub fn topo_sort(chunks: Vec<BriefChunk>, edges: &[(String, String)]) -> Vec<BriefChunk> {
    let positions = kahn_positions(build_dep_graph(&chunks, edges));
    let mut sorted = chunks;
    sorted.sort_by_key(|c| {
        (
            positions.get(&c.file_path).copied().unwrap_or(usize::MAX),
            c.start_line,
        )
    });
    sorted
}

#[derive(Serialize)]
struct JsonOutput<'a> {
    degraded: bool,
    chunks: Vec<JsonChunk<'a>>,
    injection_check: InjectionCheck,
}

#[derive(Serialize)]
struct JsonChunk<'a> {
    file_path: &'a str,
    start_line: u32,
    end_line: u32,
    chunk_type: &'static str,
    content: &'a str,
    included_reason: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    source_kind: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    injection_flags: Option<Vec<&'a str>>,
}

/// Renders `BriefOutput` as compact JSON for FR-012 (jq-friendly,
/// no pretty-printing). See `JsonOutput` / `JsonChunk` for the shape.
pub fn render_json(output: &BriefOutput) -> String {
    let json = JsonOutput {
        degraded: output.degraded,
        chunks: output
            .chunks
            .iter()
            .map(|c| JsonChunk {
                file_path: &c.file_path,
                start_line: c.start_line,
                end_line: c.end_line,
                chunk_type: c.chunk_type.as_str(),
                content: &c.content,
                included_reason: c.included_reason.to_string(),
                source_kind: c.source_kind.as_deref(),
                injection_flags: c
                    .injection_flags
                    .as_ref()
                    .map(|v| v.iter().map(String::as_str).collect()),
            })
            .collect(),
        injection_check: InjectionCheck::Ran,
    };
    serde_json::to_string(&json).expect("BriefOutput JSON serialization is infallible")
}

const DEGRADED_NOTE: &str = "Note: degraded mode — FTS-only seed selection";

/// Plain CLI rendering (FR-011 + FR-014): each chunk becomes
/// `<file_path>:<start_line>-<end_line>\n<content>`, separated by `\n---\n`.
/// When `output.degraded` is true, prepends an advisory line so the caller
/// knows seed selection fell back to FTS-only. Empty + non-degraded renders
/// to an empty string.
pub fn render_plain(output: &BriefOutput) -> String {
    let body = output
        .chunks
        .iter()
        .map(|c| {
            format!(
                "{}:{}-{}\n{}",
                c.file_path, c.start_line, c.end_line, c.content
            )
        })
        .collect::<Vec<_>>()
        .join("\n---\n");
    match (output.degraded, body.is_empty()) {
        (true, true) => DEGRADED_NOTE.to_owned(),
        (true, false) => format!("{DEGRADED_NOTE}\n{body}"),
        (false, _) => body,
    }
}

#[allow(clippy::cast_possible_truncation)]
pub fn expand_plan(conn: &Connection, task: &TaskBrief) -> Result<BriefOutput, StorageError> {
    let seeds = collect_seed_paths(task);
    if seeds.is_empty() {
        tracing::warn!("expand_plan called with empty seeds; returning empty");
        return Ok(BriefOutput {
            chunks: Vec::new(),
            degraded: true,
            total_chunks: 0,
            total_bytes: 0,
        });
    }
    let depth_by_path = collect_closure(conn, &seeds, task.depth)?;

    let paths: Vec<&str> = depth_by_path.keys().map(String::as_str).collect();
    let chunks = get_chunks_for_files(conn, &paths)?;
    let brief_chunks = build_brief_chunks(chunks, &depth_by_path);

    // BR-001 priority data is only needed when chunks exceed the budget.
    let incoming_counts = if under_cap(&brief_chunks, task.max_chunks, task.max_bytes) {
        HashMap::new()
    } else {
        get_import_counts(conn, &paths)?
    };
    let capped = apply_cap(
        brief_chunks,
        &depth_by_path,
        &incoming_counts,
        task.max_chunks,
        task.max_bytes,
    );

    // Edges only matter for files that survived cap; querying the pre-cap
    // closure would fetch rows that topo_sort silently discards.
    let capped_paths: HashSet<&str> = capped.iter().map(|c| c.file_path.as_str()).collect();
    let capped_paths: Vec<&str> = capped_paths.into_iter().collect();
    let edges = get_edges_among_files(conn, &capped_paths)?;
    let ordered = topo_sort(capped, &edges);

    let total_chunks = ordered.len() as u32;
    let total_bytes: u32 = ordered.iter().map(|c| c.content.len() as u32).sum();

    Ok(BriefOutput {
        chunks: ordered,
        degraded: false,
        total_chunks,
        total_bytes,
    })
}

#[cfg(test)]
mod tests;
