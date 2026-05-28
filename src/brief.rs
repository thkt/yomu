//! Brief expansion plan: TaskBrief -> forward closure -> chunks -> cap -> topo -> BriefOutput.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

use rusqlite::Connection;
use serde::Serialize;

use crate::injection_check::InjectionCheck;
use crate::storage::{
    Chunk, ChunkType, SourceKind, StorageError, get_chunks_for_files, get_edges_among_files,
    get_import_counts, get_transitive_dependencies_multi, get_transitive_dependents,
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
    /// Keep test files (`source_kind = 'test'`) in the closure. Default false:
    /// brief is for working on the code under test, not the tests themselves.
    pub include_tests: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkInclusionReason {
    Seed,
    Forward(u32),
    Impact(u32),
    Sibling,
    ModDecl,
}

impl fmt::Display for ChunkInclusionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Seed => f.write_str("seed"),
            Self::Forward(n) => write!(f, "forward-{n}"),
            Self::Impact(n) => write!(f, "impact-{n}"),
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
    pub source_kind: Option<SourceKind>,
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

/// Bidirectional closure (FR-C): the seed's forward imports unioned with its
/// backward impact (callers). Returns the minimum distance per file and the set
/// of forward-reachable files so the caller can label backward-only files as
/// Impact. A file reachable both ways keeps its smaller distance and the forward
/// label. When a seed has no callers the backward query returns nothing, so the
/// result degrades to a forward-only closure.
fn collect_closure(
    conn: &Connection,
    seeds: &[String],
    depth: u32,
) -> Result<(HashMap<String, u32>, HashSet<String>), StorageError> {
    let seed_refs: Vec<&str> = seeds.iter().map(String::as_str).collect();
    let mut depth_by_path: HashMap<String, u32> = HashMap::new();
    let mut forward_paths: HashSet<String> = HashSet::new();

    // Forward: a single recursive query collapses per-seed distances via MIN.
    for dep in get_transitive_dependencies_multi(conn, &seed_refs, depth)? {
        forward_paths.insert(dep.file_path.clone());
        merge_min_depth(&mut depth_by_path, dep.file_path, dep.depth);
    }

    // Backward: `get_transitive_dependents` is single-target, so fan out over
    // seeds and merge. Seeds stay at depth 0 (forward injects them); callers
    // enter at depth >= 1.
    for seed in &seed_refs {
        for dep in get_transitive_dependents(conn, seed, depth)? {
            merge_min_depth(&mut depth_by_path, dep.file_path, dep.depth);
        }
    }

    Ok((depth_by_path, forward_paths))
}

fn merge_min_depth(depth_by_path: &mut HashMap<String, u32>, path: String, depth: u32) {
    depth_by_path
        .entry(path)
        .and_modify(|d| *d = (*d).min(depth))
        .or_insert(depth);
}

fn determine_reason(depth: u32, is_forward: bool) -> ChunkInclusionReason {
    if depth == 0 {
        ChunkInclusionReason::Seed
    } else if is_forward {
        ChunkInclusionReason::Forward(depth)
    } else {
        ChunkInclusionReason::Impact(depth)
    }
}

fn build_brief_chunks(
    chunks: Vec<Chunk>,
    depth_by_path: &HashMap<String, u32>,
    forward_paths: &HashSet<String>,
) -> Vec<BriefChunk> {
    chunks
        .into_iter()
        .map(|c| {
            let depth = depth_by_path.get(&c.file_path).copied().unwrap_or_else(|| {
                tracing::warn!(
                    file_path = %c.file_path,
                    "chunk file_path missing from depth_by_path; defaulting depth=0"
                );
                debug_assert!(
                    false,
                    "chunk file_path not in depth_by_path: {}",
                    c.file_path
                );
                0
            });
            let is_forward = forward_paths.contains(&c.file_path);
            BriefChunk {
                file_path: c.file_path,
                start_line: c.start_line,
                end_line: c.end_line,
                chunk_type: c.chunk_type,
                content: c.content,
                included_reason: determine_reason(depth, is_forward),
                source_kind: c.source_kind,
                injection_flags: c.injection_flags,
            }
        })
        .collect()
}

fn to_u32_saturating(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// Pure budget check on pre-computed totals. Callers compute `chunk_count`
/// and `total_bytes` once and pass them in, so the O(n) byte sum is not
/// repeated per call.
fn under_cap(chunk_count: usize, total_bytes: usize, max_chunks: u32, max_bytes: u32) -> bool {
    to_u32_saturating(chunk_count) <= max_chunks && to_u32_saturating(total_bytes) <= max_bytes
}

/// Selects which chunk indices to drop, breadth-first. Returns the drop set and
/// the surviving byte total. Groups chunks by file and hands each file one chunk
/// per round before any file gets a second, so the budget spreads across the
/// closure and maximizes file coverage (recall@N). Files are visited
/// closest-first (depth ASC) then by centrality (incoming DESC); within a file,
/// chunks keep source order.
///
/// Guarantee: under a chunk-count-bound budget, every closure file keeps at
/// least one chunk. Under a byte-bound budget this cannot always hold — a file
/// whose chunks all exceed the remaining byte budget is dropped, because the
/// hard `max_bytes` cap forbids including it.
fn select_drops(
    chunks: &[BriefChunk],
    depth_by_path: &HashMap<String, u32>,
    incoming_counts: &HashMap<String, u32>,
    max_chunks: u32,
    max_bytes: u32,
) -> (HashSet<usize>, usize) {
    let mut by_file: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, c) in chunks.iter().enumerate() {
        by_file.entry(c.file_path.as_str()).or_default().push(i);
    }
    for idxs in by_file.values_mut() {
        idxs.sort_by_key(|&i| chunks[i].start_line);
    }
    let mut files: Vec<&str> = by_file.keys().copied().collect();
    files.sort_by(|a, b| {
        let depth = |p: &str| depth_by_path.get(p).copied().unwrap_or(0);
        let incoming = |p: &str| incoming_counts.get(p).copied().unwrap_or(0);
        depth(a)
            .cmp(&depth(b))
            .then(incoming(b).cmp(&incoming(a)))
            .then(a.cmp(b))
    });

    let max_round = by_file.values().map(Vec::len).max().unwrap_or(0);
    let mut keep: HashSet<usize> = HashSet::new();
    let mut bytes: usize = 0;
    'rounds: for round in 0..max_round {
        for file in &files {
            let Some(&idx) = by_file[*file].get(round) else {
                continue;
            };
            if to_u32_saturating(keep.len() + 1) > max_chunks {
                break 'rounds;
            }
            let len = chunks[idx].content.len();
            if to_u32_saturating(bytes + len) > max_bytes {
                continue; // over the byte budget; skip (see the byte-bound caveat above)
            }
            keep.insert(idx);
            bytes += len;
        }
    }

    let drop_idx: HashSet<usize> = (0..chunks.len()).filter(|i| !keep.contains(i)).collect();
    (drop_idx, bytes)
}

/// Caps the chunk set to the budget via breadth-first retention (BR-001
/// revised, see [`select_drops`]). Pure: takes pre-computed `depth_by_path`,
/// `incoming_counts`, and `total_bytes` so callers can test deterministically
/// without DB access and the early under-budget check reuses the precomputed
/// byte sum instead of recomputing it. Returns the surviving chunks paired with
/// their total byte count.
pub fn apply_cap(
    chunks: Vec<BriefChunk>,
    depth_by_path: &HashMap<String, u32>,
    incoming_counts: &HashMap<String, u32>,
    max_chunks: u32,
    max_bytes: u32,
    total_bytes: usize,
) -> (Vec<BriefChunk>, usize) {
    if under_cap(chunks.len(), total_bytes, max_chunks, max_bytes) {
        return (chunks, total_bytes);
    }
    let (drop_idx, remaining_bytes) = select_drops(
        &chunks,
        depth_by_path,
        incoming_counts,
        max_chunks,
        max_bytes,
    );
    let kept = chunks
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop_idx.contains(i))
        .map(|(_, c)| c)
        .collect();
    (kept, remaining_bytes)
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
    source_kind: Option<&'static str>,
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
                source_kind: c.source_kind.map(SourceKind::as_str),
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
    let (depth_by_path, forward_paths) = collect_closure(conn, &seeds, task.depth)?;

    let paths: Vec<&str> = depth_by_path.keys().map(String::as_str).collect();
    let chunks = get_chunks_for_files(conn, &paths)?;
    let mut brief_chunks = build_brief_chunks(chunks, &depth_by_path, &forward_paths);
    // Drop test files from the closure unless the caller opted in. A test file
    // reached via a `mod tests;` edge (or a test seed) is noise for a task about
    // the code under test.
    if !task.include_tests {
        brief_chunks.retain(|c| c.source_kind != Some(SourceKind::Test));
    }
    // Single byte sum reused by the cap check and the final totals.
    let total_bytes: usize = brief_chunks.iter().map(|c| c.content.len()).sum();

    // BR-001 priority data is only needed when chunks exceed the budget.
    let incoming_counts = if under_cap(
        brief_chunks.len(),
        total_bytes,
        task.max_chunks,
        task.max_bytes,
    ) {
        HashMap::new()
    } else {
        get_import_counts(conn, &paths)?
    };
    let (capped, capped_bytes) = apply_cap(
        brief_chunks,
        &depth_by_path,
        &incoming_counts,
        task.max_chunks,
        task.max_bytes,
        total_bytes,
    );

    // Edges only matter for files that survived cap; querying the pre-cap
    // closure would fetch rows that topo_sort silently discards.
    let capped_paths: HashSet<&str> = capped.iter().map(|c| c.file_path.as_str()).collect();
    let capped_paths: Vec<&str> = capped_paths.into_iter().collect();
    let edges = get_edges_among_files(conn, &capped_paths)?;
    let ordered = topo_sort(capped, &edges);

    // topo_sort only reorders chunks, so the cap's surviving byte count holds.
    let total_chunks = to_u32_saturating(ordered.len());
    let total_bytes = to_u32_saturating(capped_bytes);

    Ok(BriefOutput {
        chunks: ordered,
        degraded: false,
        total_chunks,
        total_bytes,
    })
}

#[cfg(test)]
mod tests;
