//! JSON envelope formatters for success-path routes.
//!
//! `degraded` and `notes` fields are always serialized (no `skip_serializing_if`)
//! so consumers can read `obj.degraded` without an existence guard
//! (OUTCOME.md Behavior #4, spec FR-001 / BR-002).

use std::collections::{HashMap, HashSet};

use serde::Serialize;

use crate::{indexer, storage};

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

fn coverage_if_partial(stats: &storage::IndexStatus) -> Option<String> {
    if stats.embedded_chunks < stats.embeddable_chunks {
        Some(format_coverage(stats))
    } else {
        None
    }
}

pub(super) fn format_coverage_note(stats: &storage::IndexStatus) -> Option<String> {
    coverage_if_partial(stats).map(|cov| {
        format!("\n\nEmbedding coverage: {cov}. Run `yomu index` again to finish embedding.")
    })
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

fn format_semantic_section(output: &mut String, semantic_related: &[storage::SearchResult]) {
    output.push_str("\n### Semantic related (embedding search)\n");
    for r in semantic_related {
        output.push_str(&format!(
            "- {} (similarity: {:.2})\n",
            r.chunk.file_path, r.score
        ));
    }
}

pub(super) fn format_impact_all(
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

pub(super) fn format_impact_results(
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

pub(super) fn format_results_grouped(
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

use crate::injection_check::InjectionCheck;

#[derive(Serialize)]
struct JsonChunk<'a> {
    file: &'a str,
    name: &'a str,
    r#type: &'a str,
    start_line: u32,
    end_line: u32,
    score: f32,
    content: &'a str,
    parent_chunk_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    source_kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    injection_flags: Option<Vec<&'a str>>,
}

#[derive(Serialize)]
struct JsonResponse<'a> {
    results: Vec<JsonChunk<'a>>,
    degraded: bool,
    notes: Vec<String>,
    injection_check: InjectionCheck,
}

pub(super) fn format_results_json(
    results: &[storage::SearchResult],
    degraded: bool,
    notes: Vec<String>,
) -> String {
    let items: Vec<JsonChunk> = results
        .iter()
        .map(|r| JsonChunk {
            file: &r.chunk.file_path,
            name: r.chunk.name.as_deref().unwrap_or(""),
            r#type: r.chunk.chunk_type.as_str(),
            start_line: r.chunk.start_line,
            end_line: r.chunk.end_line,
            score: r.score,
            content: &r.chunk.content,
            parent_chunk_id: r.chunk.parent_chunk_id,
            source_kind: r.chunk.source_kind.map(storage::SourceKind::as_str),
            injection_flags: r
                .chunk
                .injection_flags
                .as_ref()
                .map(|v| v.iter().map(String::as_str).collect()),
        })
        .collect();
    let response = JsonResponse {
        results: items,
        degraded,
        notes,
        injection_check: InjectionCheck::Ran,
    };
    serde_json::to_string(&response).unwrap_or_else(|e| {
        tracing::error!(error = %e, "JSON serialization failed");
        r#"{"results":[],"degraded":true,"notes":[],"injection_check":"ran"}"#.to_owned()
    })
}

#[derive(Serialize)]
struct JsonMutationResult {
    files_processed: u32,
    chunks_created: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    files_skipped: Option<u32>,
    files_errored: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    coverage: Option<String>,
    degraded: bool,
    notes: Vec<String>,
}

fn format_mutation_json(
    result: &indexer::IndexResult,
    stats: &storage::IndexStatus,
    include_skipped: bool,
    degraded: bool,
    notes: Vec<String>,
) -> String {
    let resp = JsonMutationResult {
        files_processed: result.files_processed,
        chunks_created: result.chunks_created,
        files_skipped: if include_skipped {
            Some(result.files_skipped)
        } else {
            None
        },
        files_errored: result.files_errored,
        coverage: coverage_if_partial(stats),
        degraded,
        notes,
    };
    serde_json::to_string(&resp).unwrap()
}

pub(super) fn format_index_json(
    result: &indexer::IndexResult,
    stats: &storage::IndexStatus,
    degraded: bool,
    notes: Vec<String>,
) -> String {
    format_mutation_json(result, stats, true, degraded, notes)
}

pub(super) fn format_rebuild_json(
    result: &indexer::IndexResult,
    stats: &storage::IndexStatus,
    degraded: bool,
    notes: Vec<String>,
) -> String {
    format_mutation_json(result, stats, false, degraded, notes)
}

#[derive(Serialize)]
struct JsonDryRunResult {
    files_to_process: u32,
    files_to_skip: u32,
    total_files: u32,
    files_errored: u32,
    orphans_to_remove: u32,
    degraded: bool,
    notes: Vec<String>,
}

pub(super) fn format_dry_run_json(
    result: &indexer::DryRunResult,
    degraded: bool,
    notes: Vec<String>,
) -> String {
    let resp = JsonDryRunResult {
        files_to_process: result.files_to_process,
        files_to_skip: result.files_to_skip,
        total_files: result.total_files,
        files_errored: result.files_errored,
        orphans_to_remove: result.orphans_to_remove,
        degraded,
        notes,
    };
    serde_json::to_string(&resp).unwrap()
}

#[derive(Serialize)]
struct JsonStatus {
    files: u32,
    chunks: u32,
    embedded_chunks: u32,
    embeddable_chunks: u32,
    embed_percentage: u32,
    references: u32,
    last_indexed: Option<String>,
    degraded: bool,
    notes: Vec<String>,
}

pub(super) fn format_status_json(
    stats: &storage::IndexStatus,
    ref_count: u32,
    degraded: bool,
    notes: Vec<String>,
) -> String {
    let resp = JsonStatus {
        files: stats.total_files,
        chunks: stats.total_chunks,
        embedded_chunks: stats.embedded_chunks,
        embeddable_chunks: stats.embeddable_chunks,
        embed_percentage: stats.embed_percentage(),
        references: ref_count,
        last_indexed: stats.last_indexed_at.clone(),
        degraded,
        notes,
    };
    serde_json::to_string(&resp).unwrap()
}

#[derive(Serialize)]
struct JsonReference<'a> {
    ref_kind: &'a str,
    via_symbol: Option<&'a str>,
}

#[derive(Serialize)]
struct JsonDependent<'a> {
    file_path: &'a str,
    depth: u32,
    /// Populated only for `depth == 1`. Transitive (depth >= 2) dependents are
    /// reached through intermediate files, so no direct edge data exists.
    references: Vec<JsonReference<'a>>,
}

#[derive(Serialize)]
struct JsonSemanticRelated<'a> {
    file_path: &'a str,
    score: f32,
}

#[derive(Serialize)]
struct JsonImpactResult<'a> {
    target: &'a str,
    in_index: bool,
    dependents: Vec<JsonDependent<'a>>,
    symbol_refs: &'a [String],
    #[serde(skip_serializing_if = "Vec::is_empty")]
    semantic_related: Vec<JsonSemanticRelated<'a>>,
    total: usize,
    degraded: bool,
    notes: Vec<String>,
}

// 8 args per Spec NFR-002: envelope shape stability over arg-count cap.
#[allow(clippy::too_many_arguments)]
pub(super) fn format_impact_json(
    target: &str,
    file_in_index: bool,
    dependents: &[storage::Dependent],
    direct_refs: &HashMap<String, Vec<storage::DirectReference>>,
    symbol_refs: &[String],
    semantic_related: &[storage::SearchResult],
    degraded: bool,
    notes: Vec<String>,
) -> String {
    let deps: Vec<JsonDependent> = dependents
        .iter()
        .map(|d| {
            let references = if d.depth == 1
                && let Some(refs) = direct_refs.get(&d.file_path)
            {
                refs.iter()
                    .map(|r| JsonReference {
                        ref_kind: r.ref_kind.as_str(),
                        via_symbol: r.via_symbol.as_deref(),
                    })
                    .collect()
            } else {
                Vec::new()
            };
            JsonDependent {
                file_path: &d.file_path,
                depth: d.depth,
                references,
            }
        })
        .collect();
    let sem: Vec<JsonSemanticRelated> = semantic_related
        .iter()
        .map(|r| JsonSemanticRelated {
            file_path: &r.chunk.file_path,
            score: r.score,
        })
        .collect();
    let resp = JsonImpactResult {
        target,
        in_index: file_in_index,
        dependents: deps,
        symbol_refs,
        semantic_related: sem,
        total: dependents.len(),
        degraded,
        notes,
    };
    serde_json::to_string(&resp).unwrap()
}

#[cfg(test)]
mod tests;
