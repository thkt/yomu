//! Human-readable (Markdown) renderers for search and impact results.

use std::collections::{HashMap, HashSet};

use crate::storage;

// The items re-exported by `format.rs` are `pub(crate)`, not `pub(super)`:
// `render` is a grandchild of `tools`, and `format.rs` re-exports them up to
// the `tools` level via `pub(super) use`. `pub(super)` / `pub(in crate::tools)`
// here would scope them to `format` and fail that re-export with E0364.
// Effective reach stays `tools`-scoped because `mod render` is private.

#[derive(Debug)]
pub(crate) struct EnrichmentContext {
    pub imports: HashMap<String, String>,
    pub siblings: HashMap<String, Vec<storage::SiblingInfo>>,
}

fn format_dependents_by_depth(
    output: &mut String,
    dependents: &[storage::Dependent],
    heading_prefix: &str,
) {
    debug_assert!(
        dependents.windows(2).all(|w| w[0].depth <= w[1].depth),
        "dependents must be sorted by depth"
    );
    let mut current_depth = 0;
    for dep in dependents {
        if dep.depth != current_depth {
            current_depth = dep.depth;
            output.push_str(&format!("{heading_prefix} Depth {}\n", current_depth));
        }
        output.push_str(&format!("- {}\n", dep.file_path));
    }
}

fn format_semantic_section(output: &mut String, semantic_related: &[storage::SearchResult]) {
    output.push_str("\n### Semantic related (embedding search)\n");
    for r in semantic_related {
        output.push_str(&format!(
            "- {} (similarity: {:.2})\n",
            r.chunk.file_path, r.score
        ));
    }
}

pub(crate) fn format_impact_all(
    target: &str,
    dependents: &[storage::Dependent],
    semantic_related: &[storage::SearchResult],
) -> String {
    let mut output = format!("## Impact analysis: `{}`\n\n", target);

    if semantic_related.is_empty() {
        format_dependents_by_depth(&mut output, dependents, "###");
    } else {
        output.push_str("### Structural dependents (import graph)\n");
        format_dependents_by_depth(&mut output, dependents, "####");
        format_semantic_section(&mut output, semantic_related);
    }

    output.push_str(&format!(
        "\nTotal: {} dependent file(s)\n",
        dependents.len()
    ));
    output
}

pub(crate) fn format_impact_results(
    target: &str,
    symbol_refs: &[String],
    all_dependents: &[storage::Dependent],
    semantic_related: &[storage::SearchResult],
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

    if !semantic_related.is_empty() {
        format_semantic_section(&mut output, semantic_related);
    }

    output.push_str(&format!(
        "\nTotal: {} dependent file(s)\n",
        all_dependents.len()
    ));
    output
}

fn format_imports_line(imports: &HashMap<String, String>, file_path: &str) -> Option<String> {
    let imports_text = imports.get(file_path)?;
    let items: Vec<&str> = imports_text.split('\n').filter(|s| !s.is_empty()).collect();
    if items.is_empty() {
        return None;
    }
    Some(format!("Imports: {}\n", items.join(", ")))
}

fn format_siblings_line(
    siblings_map: &HashMap<String, Vec<storage::SiblingInfo>>,
    file_path: &str,
    result_ranges: &HashSet<(u32, u32)>,
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
    if filtered.is_empty() {
        return None;
    }
    Some(format!("Siblings: {}\n", filtered.join(", ")))
}

fn format_file_group(
    output: &mut String,
    file_path: &str,
    chunks: &[(usize, &storage::SearchResult)],
    ctx: &EnrichmentContext,
    parent_chunks: &HashMap<i64, storage::Chunk>,
    result_chunk_ids: &HashSet<i64>,
) {
    output.push_str(&format!("## {}\n", file_path));

    if let Some(line) = format_imports_line(&ctx.imports, file_path) {
        output.push_str(&line);
    }

    let result_ranges: HashSet<(u32, u32)> = chunks
        .iter()
        .map(|(_, r)| (r.chunk.start_line, r.chunk.end_line))
        .collect();
    if let Some(line) = format_siblings_line(&ctx.siblings, file_path, &result_ranges) {
        output.push_str(&line);
    }

    output.push('\n');

    for (rank, result) in chunks {
        let chunk = &result.chunk;
        let name = chunk.name.as_deref().unwrap_or("(unnamed)");
        output.push_str(&format!(
            "{}. {} [{}] — {}:{} (similarity: {:.2})\n",
            rank + 1,
            name,
            chunk.chunk_type.as_str(),
            chunk.start_line,
            chunk.end_line,
            result.score,
        ));
        output.push_str(&chunk.content);
        output.push_str("\n\n");

        if let Some(parent_id) = chunk.parent_chunk_id
            && !result_chunk_ids.contains(&parent_id)
            && let Some(parent) = parent_chunks.get(&parent_id)
        {
            let parent_name = parent.name.as_deref().unwrap_or("(unnamed)");
            output.push_str(&format!(
                "  Parent context: {} [{}] — {}:{}\n",
                parent_name,
                parent.chunk_type.as_str(),
                parent.start_line,
                parent.end_line,
            ));
            output.push_str(&parent.content);
            output.push_str("\n\n");
        }
    }
}

pub(crate) fn format_results_grouped(
    results: &[storage::SearchResult],
    ctx: &EnrichmentContext,
    parent_chunks: &HashMap<i64, storage::Chunk>,
) -> String {
    let mut groups: HashMap<&str, Vec<(usize, &storage::SearchResult)>> = HashMap::new();
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
        best(&b.1).total_cmp(&best(&a.1))
    });

    let mut output = String::new();
    let result_chunk_ids: HashSet<i64> = results.iter().filter_map(|r| r.chunk_id).collect();

    for (file_path, chunks) in &sorted {
        format_file_group(
            &mut output,
            file_path,
            chunks,
            ctx,
            parent_chunks,
            &result_chunk_ids,
        );
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{Chunk, ChunkType, Dependent, MatchSource, SearchResult};

    fn dependent(file_path: &str, depth: u32) -> Dependent {
        Dependent {
            file_path: file_path.to_owned(),
            depth,
        }
    }

    fn semantic_hit(file_path: &str, score: f32) -> SearchResult {
        SearchResult {
            chunk: Chunk {
                file_path: file_path.to_owned(),
                chunk_type: ChunkType::RustFn,
                name: None,
                content: String::new(),
                start_line: 1,
                end_line: 1,
                parent_chunk_id: None,
                source_kind: None,
                injection_flags: None,
            },
            chunk_id: None,
            distance: 0.1,
            match_source: MatchSource::Semantic,
            score,
        }
    }

    // T-706: format_impact_all_renders_semantic_section_when_related_present
    #[test]
    fn format_impact_all_renders_semantic_section_when_related_present() {
        let deps = vec![dependent("src/a.rs", 1)];
        let related = vec![semantic_hit("src/b.rs", 0.83)];
        let out = format_impact_all("foo", &deps, &related);
        assert!(
            out.contains("### Structural dependents (import graph)"),
            "expected structural heading, got: {out}"
        );
        assert!(
            out.contains("### Semantic related (embedding search)"),
            "expected semantic heading, got: {out}"
        );
        assert!(
            out.contains("src/b.rs (similarity: 0.83)"),
            "expected semantic hit line, got: {out}"
        );
    }

    // T-707: format_impact_results_renders_symbol_refs_and_semantic_section
    #[test]
    fn format_impact_results_renders_symbol_refs_and_semantic_section() {
        let deps = vec![dependent("src/a.rs", 1)];
        let related = vec![semantic_hit("src/b.rs", 0.5)];
        let out = format_impact_results("foo", &["bar".to_owned()], &deps, &related);
        assert!(
            out.contains("### Direct symbol references"),
            "expected symbol-refs heading, got: {out}"
        );
        assert!(
            out.contains("- bar"),
            "expected symbol ref line, got: {out}"
        );
        assert!(
            out.contains("### Semantic related (embedding search)"),
            "expected semantic heading, got: {out}"
        );
    }
}
