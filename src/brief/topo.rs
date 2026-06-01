//! Topological ordering of brief chunks (BR-002): dependencies first, ties by
//! file path lex order, cycles degraded to a lexicographic tail.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::BriefChunk;

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
pub(super) fn topo_sort(chunks: Vec<BriefChunk>, edges: &[(String, String)]) -> Vec<BriefChunk> {
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
