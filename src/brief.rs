//! Brief expansion plan: TaskBrief -> forward closure -> chunks -> cap -> topo -> BriefOutput.

use std::collections::{HashMap, HashSet};
use std::fmt;

use rusqlite::Connection;
use serde::Serialize;

use crate::injection_check::InjectionCheck;
use crate::storage::{
    Chunk, ChunkType, SourceKind, StorageError, get_chunks_for_files, get_edges_among_files,
    get_import_counts, get_transitive_dependencies_multi, get_transitive_dependents,
};

mod cap;
mod topo;

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
    /// Distinct files reachable in the closure after the test filter and before
    /// the cap. Surfaced for recall measurement (FR-002): cap-fit's denominator
    /// is the must-include weight reachable here, isolating the cap's effect
    /// from the closure's coverage. Not serialized in the CLI output.
    pub reachable_files: Vec<String>,
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
            reachable_files: Vec::new(),
        });
    }
    let (depth_by_path, forward_paths) = collect_closure(conn, &seeds, task.depth)?;

    let paths: Vec<&str> = depth_by_path.keys().map(String::as_str).collect();
    let chunks = get_chunks_for_files(conn, &paths)?;
    let mut brief_chunks = build_brief_chunks(chunks, &depth_by_path, &forward_paths);
    // Drop test files reached transitively (e.g. via a `mod tests;` edge): they
    // are noise for a task about the code under test. An explicitly named seed
    // is the exception — the caller asked for that file, so keep it even when it
    // is a test. Dropping it would empty the closure and the CLI would then
    // misreport the cause as an FTS fallback (#236).
    if !task.include_tests {
        let seed_set: HashSet<&str> = seeds.iter().map(String::as_str).collect();
        brief_chunks.retain(|c| {
            c.source_kind != Some(SourceKind::Test) || seed_set.contains(c.file_path.as_str())
        });
    }

    // Distinct files reachable after the test filter and before the cap, in
    // chunk (source) order. Captured here so cap-fit (FR-002) can divide by the
    // must-include weight the closure actually reached, not what the cap kept.
    let reachable_files: Vec<String> = {
        let mut seen = HashSet::new();
        brief_chunks
            .iter()
            .filter(|c| seen.insert(c.file_path.as_str()))
            .map(|c| c.file_path.clone())
            .collect()
    };

    // Single byte sum reused by the cap check and the final totals.
    let total_bytes: usize = brief_chunks.iter().map(|c| c.content.len()).sum();

    // BR-001 priority data is only needed when chunks exceed the budget.
    let incoming_counts = if cap::under_cap(
        brief_chunks.len(),
        total_bytes,
        task.max_chunks,
        task.max_bytes,
    ) {
        HashMap::new()
    } else {
        get_import_counts(conn, &paths)?
    };
    let (capped, capped_bytes) = cap::apply_cap(
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
    let ordered = topo::topo_sort(capped, &edges);

    // topo_sort only reorders chunks, so the cap's surviving byte count holds.
    let total_chunks = to_u32_saturating(ordered.len());
    let total_bytes = to_u32_saturating(capped_bytes);

    Ok(BriefOutput {
        chunks: ordered,
        degraded: false,
        total_chunks,
        total_bytes,
        reachable_files,
    })
}

#[cfg(test)]
mod tests;
