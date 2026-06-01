//! JSON envelope formatters for success-path routes, plus shared
//! index-coverage helpers (used by both JSON and text output paths).
//!
//! `degraded` and `notes` fields are always serialized (no `skip_serializing_if`)
//! so consumers can read `obj.degraded` without an existence guard
//! (OUTCOME.md Behavior #4, spec FR-001 / BR-002).

use std::collections::HashMap;

use serde::Serialize;

use crate::injection_check::InjectionCheck;
use crate::{indexer, storage};

mod render;

pub(super) use render::{
    EnrichmentContext, format_impact_all, format_impact_results, format_results_grouped,
};

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
