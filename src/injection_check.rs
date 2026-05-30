//! ADR-0069 BR-303 sentinel: 3-variant closed set for response-level
//! matcher state. `Ran` is the only variant emitted in PR#3 onward success
//! paths; `Skipped` and `Unavailable` are reserved for matcher disable /
//! corpus init failure.
//!
//! Crate-root module so the 3 consumers (`tools::format`, `brief`, `verify`)
//! all reach the enum symmetrically (BR-406 single-source).

use serde::Serialize;

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum InjectionCheck {
    Ran,
    Skipped,
    Unavailable,
}

impl InjectionCheck {
    /// Closed-set guard for ADR-0069 BR-303: adding a variant without
    /// updating this match breaks compile. Mirrors `ChunkType::as_str` /
    /// `RefKind::as_str` idiom.
    #[allow(dead_code)]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ran => "ran",
            Self::Skipped => "skipped",
            Self::Unavailable => "unavailable",
        }
    }
}

#[cfg(test)]
mod tests;
