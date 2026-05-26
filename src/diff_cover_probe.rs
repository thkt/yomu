//! Throwaway probe for verifying the diff coverage gate. This module exists
//! only on the test/verify-diff-cover-fail branch and must be deleted with it.

/// Intentionally untested so diff-cover sees an uncovered added line and the
/// coverage gate fails. Not called from anywhere.
pub fn probe(n: usize) -> usize {
    n.saturating_add(1)
}
