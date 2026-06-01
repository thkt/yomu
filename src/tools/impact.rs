use std::collections::{HashMap, HashSet};

use crate::{query, storage};

use super::format::{format_impact_all, format_impact_json, format_impact_results};
use super::{
    InvalidInputKind, MAX_EMPTY_TARGET_CANDIDATES, MAX_IMPACT_DEPTH, Yomu, YomuError,
    never_degraded, parse_impact_target, validate_path,
};

const SEMANTIC_THRESHOLD: f32 = 0.7;

/// The data `Yomu::impact` gathers for one target before rendering: index
/// presence, transitive + symbol dependents, grouped direct references, and
/// semantically-related files. Bundling the DB results lets the fetch and
/// render seams pass one value instead of a long argument list.
#[derive(Debug)]
struct ImpactReport {
    file_in_index: bool,
    dependents: Vec<storage::Dependent>,
    symbol_refs: Vec<String>,
    direct_refs: HashMap<String, Vec<storage::DirectReference>>,
    semantic_related: Vec<storage::SearchResult>,
}

impl Yomu {
    /// Lexicographically-first `max` indexed file paths. Used to populate
    /// `EmptyTarget.candidates` when impact is invoked with an empty target
    /// (#197). Storage failures degrade to an empty vector so the primary
    /// `UsageError` code is preserved (FR-004 / BR-002). Ordering is
    /// alphabetical (not ranked) to keep results deterministic across runs.
    fn first_indexed_paths(&self, max: usize) -> Vec<String> {
        self.with_db(storage::get_all_file_paths)
            .map(|set| {
                let mut v: Vec<String> = set.into_iter().collect();
                v.sort();
                v.truncate(max);
                v
            })
            .unwrap_or_default()
    }

    pub fn impact(
        &self,
        target: &str,
        symbol: Option<&str>,
        depth: u32,
        json: bool,
        semantic: bool,
    ) -> Result<String, YomuError> {
        self.require_nonempty_target(target)?;

        let stats = self.with_db(storage::get_stats)?;
        if stats.total_chunks == 0 {
            return Err(YomuError::InvalidInput(InvalidInputKind::EmptyIndex));
        }

        let (file_path, parsed_symbol) = parse_impact_target(target);
        validate_path(file_path)?;
        let symbol_filter = symbol.or(parsed_symbol);
        let max_depth = depth.min(MAX_IMPACT_DEPTH);

        let report = self.gather_impact_report(file_path, symbol_filter, max_depth, semantic)?;
        Ok(render_impact_output(
            target,
            file_path,
            &report,
            symbol_filter,
            json,
        ))
    }

    /// Rejects an empty impact target, surfacing the first indexed paths as
    /// candidates so the caller can correct the argument (#197, FR-004 / BR-002).
    fn require_nonempty_target(&self, target: &str) -> Result<(), YomuError> {
        if target.is_empty() {
            let candidates = self.first_indexed_paths(MAX_EMPTY_TARGET_CANDIDATES);
            return Err(YomuError::InvalidInput(InvalidInputKind::EmptyTarget {
                candidates,
            }));
        }
        Ok(())
    }

    /// Gathers everything `impact` reports on for one target, isolating the DB
    /// access from the orchestration and rendering. The four graph queries run
    /// in one `with_db` transaction; the semantic pass is a separate query made
    /// only when requested.
    fn gather_impact_report(
        &self,
        file_path: &str,
        symbol_filter: Option<&str>,
        max_depth: u32,
        semantic: bool,
    ) -> Result<ImpactReport, YomuError> {
        let fp = file_path.to_owned();
        let sym_owned = symbol_filter.map(str::to_owned);
        let (file_in_index, dependents, symbol_refs, direct_refs) = self.with_db(move |conn| {
            let exists = storage::file_exists_in_index(conn, &fp)?;
            let dependents = storage::get_transitive_dependents(conn, &fp, max_depth)?;
            let refs = match &sym_owned {
                Some(sym) => storage::get_symbol_dependents(conn, &fp, sym)?,
                None => vec![],
            };
            let direct = storage::get_direct_references(conn, &fp)?;
            let mut grouped: HashMap<String, Vec<storage::DirectReference>> = HashMap::new();
            for r in direct {
                grouped.entry(r.source_file.clone()).or_default().push(r);
            }
            Ok((exists, dependents, refs, grouped))
        })?;

        let semantic_related = if semantic {
            self.semantic_search(file_path, symbol_filter)?
        } else {
            vec![]
        };

        Ok(ImpactReport {
            file_in_index,
            dependents,
            symbol_refs,
            direct_refs,
            semantic_related,
        })
    }

    fn semantic_search(
        &self,
        file_path: &str,
        symbol: Option<&str>,
    ) -> Result<Vec<storage::SearchResult>, YomuError> {
        let fp = file_path.to_owned();
        let sym = symbol.map(str::to_owned);
        let mut results = self.with_db(move |c| {
            let ids = storage::get_chunks_for_from_target(c, &fp, sym.as_deref())?;
            let bytes: Vec<Vec<u8>> = storage::get_sub_embeddings_for_chunks(c, &ids)?
                .into_iter()
                .map(|(_, b)| b)
                .collect();
            if bytes.is_empty() {
                return Ok(vec![]);
            }
            let source_ids: HashSet<i64> = ids.into_iter().collect();
            query::search_from_file(c, &bytes, &source_ids, None, 20, &[])
        })?;
        results.retain(|r| r.score >= SEMANTIC_THRESHOLD);
        let mut seen: HashSet<String> = HashSet::new();
        results.retain(|r| seen.insert(r.chunk.file_path.clone()));
        Ok(results)
    }
}

/// Renders a gathered [`ImpactReport`] to the user-facing string: JSON when
/// requested; an empty-result message (no dependents found, or the target not
/// in the index) when there is nothing to show; otherwise the symbol-scoped vs
/// whole-file view. Pure (depends only on its arguments), so it is
/// unit-testable without a DB.
fn render_impact_output(
    target: &str,
    file_path: &str,
    report: &ImpactReport,
    symbol_filter: Option<&str>,
    json: bool,
) -> String {
    if json {
        let (degraded, notes) = never_degraded();
        return format_impact_json(
            target,
            report.file_in_index,
            &report.dependents,
            &report.direct_refs,
            &report.symbol_refs,
            &report.semantic_related,
            degraded,
            notes,
        );
    }

    if report.dependents.is_empty() && report.semantic_related.is_empty() {
        return if report.file_in_index {
            format!("No dependents found for `{}`.", target)
        } else {
            format!(
                "`{}` not found in index. Run `yomu index` to update.",
                file_path
            )
        };
    }

    if symbol_filter.is_some() {
        format_impact_results(
            target,
            &report.symbol_refs,
            &report.dependents,
            &report.semantic_related,
        )
    } else {
        format_impact_all(target, &report.dependents, &report.semantic_related)
    }
}
