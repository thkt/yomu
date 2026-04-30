//! Brief expansion plan: TaskBrief -> forward closure -> chunks -> BriefOutput.
//!
//! Phase 1c-3 scaffold. cap (BR-001) and topo sort (BR-002) are deferred to
//! follow-up commits in this phase.

use std::collections::{HashMap, HashSet};

use rusqlite::Connection;

use crate::storage::{
    Chunk, ChunkType, StorageError, get_chunks_for_files, get_import_counts,
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

#[derive(Debug, Clone)]
pub struct BriefChunk {
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub chunk_type: ChunkType,
    pub content: String,
    pub included_reason: ChunkInclusionReason,
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
            let depth = depth_by_path.get(&c.file_path).copied().unwrap_or(0);
            BriefChunk {
                file_path: c.file_path,
                start_line: c.start_line,
                end_line: c.end_line,
                chunk_type: c.chunk_type,
                content: c.content,
                included_reason: determine_reason(depth),
            }
        })
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
fn under_cap(chunks: &[BriefChunk], max_chunks: u32, max_bytes: u32) -> bool {
    let count = chunks.len() as u32;
    let bytes: usize = chunks.iter().map(|c| c.content.len()).sum();
    count <= max_chunks && bytes as u32 <= max_bytes
}

/// Applies BR-001 cap deletion order: drop chunks first by
/// `(seed_distance DESC, incoming_edge_count ASC)`. Stops once both
/// `max_chunks` and `max_bytes` constraints are satisfied. Pure: takes
/// pre-computed `depth_by_path` and `incoming_counts` so callers can test
/// deterministically without DB access.
#[allow(clippy::cast_possible_truncation)]
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
    let mut order: Vec<(usize, u32, u32)> = chunks
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let depth = depth_by_path.get(&c.file_path).copied().unwrap_or(0);
            let incoming = incoming_counts.get(&c.file_path).copied().unwrap_or(0);
            (i, depth, incoming)
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
    chunks
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop_idx.contains(i))
        .map(|(_, c)| c)
        .collect()
}

#[allow(clippy::cast_possible_truncation)]
pub fn expand_plan(conn: &Connection, task: &TaskBrief) -> Result<BriefOutput, StorageError> {
    let seeds = collect_seed_paths(task);
    let depth_by_path = collect_closure(conn, &seeds, task.depth)?;

    let paths: Vec<&str> = depth_by_path.keys().map(String::as_str).collect();
    let chunks = get_chunks_for_files(conn, &paths)?;
    let brief_chunks = build_brief_chunks(chunks, &depth_by_path);

    let incoming_counts = get_import_counts(conn, &paths)?;
    let capped = apply_cap(
        brief_chunks,
        &depth_by_path,
        &incoming_counts,
        task.max_chunks,
        task.max_bytes,
    );

    let total_chunks = capped.len() as u32;
    let total_bytes: u32 = capped.iter().map(|c| c.content.len() as u32).sum();

    Ok(BriefOutput {
        chunks: capped,
        degraded: false,
        total_chunks,
        total_bytes,
    })
}

#[cfg(test)]
mod tests;
