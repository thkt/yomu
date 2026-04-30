//! Brief expansion plan: TaskBrief -> forward closure -> chunks -> BriefOutput.
//!
//! Phase 1c-3 scaffold. cap (BR-001) and topo sort (BR-002) are deferred to
//! follow-up commits in this phase.

use std::collections::HashMap;

use rusqlite::Connection;

use crate::storage::{
    Chunk, ChunkType, StorageError, get_chunks_for_files, get_transitive_dependencies,
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
pub fn expand_plan(conn: &Connection, task: &TaskBrief) -> Result<BriefOutput, StorageError> {
    let seeds = collect_seed_paths(task);
    let depth_by_path = collect_closure(conn, &seeds, task.depth)?;

    let paths: Vec<&str> = depth_by_path.keys().map(String::as_str).collect();
    let chunks = get_chunks_for_files(conn, &paths)?;
    let brief_chunks = build_brief_chunks(chunks, &depth_by_path);

    let total_chunks = brief_chunks.len() as u32;
    let total_bytes: u32 = brief_chunks.iter().map(|c| c.content.len() as u32).sum();

    Ok(BriefOutput {
        chunks: brief_chunks,
        degraded: false,
        total_chunks,
        total_bytes,
    })
}

#[cfg(test)]
mod tests;
