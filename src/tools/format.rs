use std::collections::{HashMap, HashSet};

use crate::storage;

pub(super) struct EnrichmentContext {
    pub imports: HashMap<String, String>,
    pub siblings: HashMap<String, Vec<storage::SiblingInfo>>,
}

pub(super) fn format_coverage(stats: &storage::IndexStatus) -> String {
    format!(
        "{}/{} chunks ({}%)",
        stats.embedded_chunks,
        stats.total_chunks,
        stats.embed_percentage()
    )
}

pub(super) fn format_no_results_message(stats: &storage::IndexStatus) -> String {
    format!(
        "No results found. Index coverage: {}. Repeat search to expand embedding coverage.",
        format_coverage(stats)
    )
}

pub(super) fn format_coverage_note(stats: &storage::IndexStatus) -> Option<String> {
    if stats.embedded_chunks < stats.total_chunks {
        Some(format!(
            "\n\nEmbedding coverage: {}. Use search to incrementally embed more.",
            format_coverage(stats)
        ))
    } else {
        None
    }
}

pub(super) fn format_dependents_by_depth(
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

pub(super) fn format_impact_all(target: &str, dependents: &[storage::Dependent]) -> String {
    let mut output = format!("## Impact analysis: `{}`\n\n", target);
    format_dependents_by_depth(&mut output, dependents, "###");
    output.push_str(&format!(
        "\nTotal: {} dependent file(s)\n",
        dependents.len()
    ));
    output
}

pub(super) fn format_impact_results(
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

pub(super) fn format_results_grouped(
    results: &[storage::SearchResult],
    ctx: &EnrichmentContext,
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
    for (file_path, chunks) in &sorted {
        format_file_group(&mut output, file_path, chunks, ctx);
    }
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
    }
}
