//! Budget cap (BR-001 revised): breadth-first chunk retention under the
//! `max_chunks` / `max_bytes` budget. Pure functions over pre-computed totals.

use std::collections::{HashMap, HashSet};

use super::{BriefChunk, to_u32_saturating};

/// Pure budget check on pre-computed totals. Callers compute `chunk_count`
/// and `total_bytes` once and pass them in, so the O(n) byte sum is not
/// repeated per call.
pub(super) fn under_cap(
    chunk_count: usize,
    total_bytes: usize,
    max_chunks: u32,
    max_bytes: u32,
) -> bool {
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
pub(super) fn apply_cap(
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
