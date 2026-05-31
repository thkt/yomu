use std::collections::{HashMap, HashSet};

use crate::{query, storage};

use super::format::{format_impact_all, format_impact_json, format_impact_results};
use super::{
    InvalidInputKind, MAX_EMPTY_TARGET_CANDIDATES, MAX_IMPACT_DEPTH, Yomu, YomuError,
    never_degraded, parse_impact_target, validate_path,
};

const SEMANTIC_THRESHOLD: f32 = 0.7;

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
        if target.is_empty() {
            let candidates = self.first_indexed_paths(MAX_EMPTY_TARGET_CANDIDATES);
            return Err(YomuError::InvalidInput(InvalidInputKind::EmptyTarget {
                candidates,
            }));
        }

        let stats = self.with_db(storage::get_stats)?;
        if stats.total_chunks == 0 {
            return Err(YomuError::InvalidInput(InvalidInputKind::EmptyIndex));
        }

        let (file_path, parsed_symbol) = parse_impact_target(target);

        validate_path(file_path)?;

        let symbol_filter = symbol.or(parsed_symbol);
        let max_depth = depth.min(MAX_IMPACT_DEPTH);
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

        if json {
            let (degraded, notes) = never_degraded();
            return Ok(format_impact_json(
                target,
                file_in_index,
                &dependents,
                &direct_refs,
                &symbol_refs,
                &semantic_related,
                degraded,
                notes,
            ));
        }

        if dependents.is_empty() && semantic_related.is_empty() {
            return Ok(if file_in_index {
                format!("No dependents found for `{}`.", target)
            } else {
                format!(
                    "`{}` not found in index. Run `yomu index` to update.",
                    file_path
                )
            });
        }

        let text = if symbol_filter.is_some() {
            format_impact_results(target, &symbol_refs, &dependents, &semantic_related)
        } else {
            format_impact_all(target, &dependents, &semantic_related)
        };

        Ok(text)
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
