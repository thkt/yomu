//! Ground-truth (GT) corpus loading and the anti-tautology canary for `brief`
//! recall measurement.
//!
//! A GT entry pairs an investigation task with the file set a complete answer
//! must span (`must_include`, per-file weighted). The loader validates weights
//! (FR-016); the canary (FR-006 / BR-002) rejects entries whose every
//! must-include lies inside the seed's 1-hop forward closure, which would make
//! recall vacuously high.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;

use super::WeightedFile;

/// One ground-truth entry: an investigation task plus the file set a complete
/// answer must span. Deserialized from `tests/fixtures/brief/gt.yaml`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct GtEntry {
    /// Stable identifier, unique within the corpus (named in load errors and the canary).
    pub id: String,
    /// Target repository (`rurico` | `amici`).
    pub repo: String,
    /// Pinned submodule rev the seed and must-include paths resolve against.
    pub rev: String,
    /// Domain area the task exercises (e.g. `embedding`, `storage`).
    pub domain: String,
    /// One-sentence investigation or change task the brief must answer.
    pub task: String,
    /// Explicit seed paths (repo-relative) passed to `brief --seed-file`.
    pub seed: Vec<String>,
    /// Why these files form a complete answer: domain reasoning, not derived from
    /// brief output (AC-2 / AS-004).
    pub rationale: String,
    /// Files a complete answer must span, each with a positive importance weight.
    pub must_include: Vec<WeightedFile>,
}

/// A loaded ground-truth corpus.
#[derive(Debug, Clone, Deserialize)]
pub struct GtCorpus {
    pub entries: Vec<GtEntry>,
}

/// Errors returned by [`load_from_str`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GtLoadError {
    /// The yaml did not parse into a [`GtCorpus`].
    #[error("gt corpus parse error: {0}")]
    Parse(#[from] serde_yaml::Error),
    /// An entry's must-include set is empty (FR-016 / entity invariant).
    #[error("gt entry {id:?} has an empty must-include set")]
    EmptyMustInclude { id: String },
    /// A must-include file carries a zero (non-positive) weight (FR-016).
    #[error("gt entry {id:?} must-include {path:?} has zero weight; weight must be positive")]
    ZeroWeight { id: String, path: String },
}

/// Loads and validates a GT corpus from a yaml string.
///
/// # Errors
/// [`GtLoadError::Parse`] on malformed yaml, [`GtLoadError::EmptyMustInclude`] for an
/// entry with no must-include files, and [`GtLoadError::ZeroWeight`] when a must-include
/// weight is zero (FR-016: weights must be positive). Each non-parse error names the
/// offending entry id.
pub fn load_from_str(text: &str) -> Result<GtCorpus, GtLoadError> {
    let corpus: GtCorpus = serde_yaml::from_str(text)?;
    for entry in &corpus.entries {
        if entry.must_include.is_empty() {
            return Err(GtLoadError::EmptyMustInclude {
                id: entry.id.clone(),
            });
        }
        for file in &entry.must_include {
            if file.weight == 0 {
                return Err(GtLoadError::ZeroWeight {
                    id: entry.id.clone(),
                    path: file.path.clone(),
                });
            }
        }
    }
    Ok(corpus)
}

/// Returns the ids of entries that fail the anti-tautology canary (FR-006 / BR-002).
///
/// An entry fails when every must-include file is trivially reachable: it is either
/// the seed itself or lies in the seed's 1-hop forward closure. Such an entry needs
/// no bidirectional or multi-hop closure work, so `recall` would be vacuously high.
/// `forward_1hop` maps each entry id to the file set reachable by 1-hop forward edges
/// from its seed (the `forward_paths` element of `brief::collect_closure(seed, 1)`,
/// which includes the seeds at depth 0). The seed is also checked explicitly so the
/// canary stays strict even if a closure source omits seeds from `forward_1hop`. An
/// entry absent from the map fails the canary because its closure is unproven.
///
/// Pure set logic: the caller wires `collect_closure` (Gate1) or a synthetic set
/// (unit test).
pub fn canary_violations(
    entries: &[GtEntry],
    forward_1hop: &HashMap<String, HashSet<String>>,
) -> Vec<String> {
    entries
        .iter()
        .filter(|entry| match forward_1hop.get(&entry.id) {
            None => true,
            Some(reachable) => entry
                .must_include
                .iter()
                .all(|file| entry.seed.contains(&file.path) || reachable.contains(&file.path)),
        })
        .map(|entry| entry.id.clone())
        .collect()
}

/// The ground-truth corpus bundled with the crate (`tests/fixtures/brief/gt.yaml`).
const BUNDLED_GT_YAML: &str = include_str!("../../tests/fixtures/brief/gt.yaml");

/// Loads and validates the bundled ground-truth corpus.
///
/// Convenience over [`load_from_str`] for the two production call sites — the
/// Gate1 seeded-recall gate and the `yomu recall` CLI — that measure against the
/// same bundled corpus.
///
/// # Errors
/// Propagates [`load_from_str`]. The bundled corpus is also covered by a load
/// test, so an error here signals the fixture was edited into an invalid state.
pub fn load_bundled() -> Result<GtCorpus, GtLoadError> {
    load_from_str(BUNDLED_GT_YAML)
}

#[cfg(test)]
mod tests;
