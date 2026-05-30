use super::*;

// T-424a: injection_check_enum_lives_in_crate_root
// Spec FR-401 / BR-406: enum lives at src/injection_check.rs, not duplicated elsewhere.
#[test]
fn injection_check_enum_lives_in_crate_root() {
    // Compile-time presence + as_str exhaustiveness
    assert_eq!(InjectionCheck::Ran.as_str(), "ran");
    assert_eq!(InjectionCheck::Skipped.as_str(), "skipped");
    assert_eq!(InjectionCheck::Unavailable.as_str(), "unavailable");
}

// T-424b: format_rs_uses_crate_injection_check_path
// Spec FR-402a: src/tools/format.rs imports from crate::injection_check, not local enum.
#[test]
fn format_rs_uses_crate_injection_check_path() {
    const FORMAT_RS: &str = include_str!("../tools/format.rs");
    assert!(
        FORMAT_RS.contains("use crate::injection_check::InjectionCheck"),
        "src/tools/format.rs must import InjectionCheck from crate::injection_check (FR-402a)"
    );
    assert!(
        !FORMAT_RS.contains("pub(crate) enum InjectionCheck"),
        "src/tools/format.rs must NOT define InjectionCheck locally (FR-402a)"
    );
    assert!(
        !FORMAT_RS.contains("fn injection_check_str"),
        "src/tools/format.rs must NOT define injection_check_str (FR-402a, replaced by InjectionCheck::as_str)"
    );
}

// T-424c: brief_rs_uses_crate_injection_check_path
// Spec FR-402b: src/brief.rs imports from crate::injection_check, not tools::format.
#[test]
fn brief_rs_uses_crate_injection_check_path() {
    const BRIEF_RS: &str = include_str!("../brief.rs");
    assert!(
        BRIEF_RS.contains("use crate::injection_check::InjectionCheck"),
        "src/brief.rs must import InjectionCheck from crate::injection_check (FR-402b)"
    );
    assert!(
        !BRIEF_RS.contains("use crate::tools::format::InjectionCheck"),
        "src/brief.rs must NOT import InjectionCheck via tools::format (FR-402b)"
    );
}
