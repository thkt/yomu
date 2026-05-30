use super::*;
use crate::storage;

fn empty_response_with(injection_check: InjectionCheck) -> JsonResponse<'static> {
    JsonResponse {
        results: Vec::new(),
        degraded: false,
        notes: Vec::new(),
        injection_check,
    }
}

fn sample_chunk_with_flags<'a>(injection_flags: Option<Vec<&'a str>>) -> JsonChunk<'a> {
    JsonChunk {
        file: "src/foo.rs",
        name: "foo",
        r#type: "rust_fn",
        start_line: 1,
        end_line: 3,
        score: 0.5,
        content: "fn foo() {}",
        parent_chunk_id: None,
        source_kind: None,
        injection_flags,
    }
}

fn sample_chunk_with_source_kind(source_kind: Option<&'static str>) -> JsonChunk<'static> {
    JsonChunk {
        file: "src/foo.rs",
        name: "foo",
        r#type: "rust_fn",
        start_line: 1,
        end_line: 3,
        score: 0.5,
        content: "fn foo() {}",
        parent_chunk_id: None,
        source_kind,
        injection_flags: None,
    }
}

// T-309: json_chunk_skips_injection_flags_when_none
#[test]
fn json_chunk_skips_injection_flags_when_none() {
    let chunk = sample_chunk_with_flags(None);
    let serialized = serde_json::to_string(&chunk).unwrap();
    assert!(
        !serialized.contains("injection_flags"),
        "JsonChunk with injection_flags=None must omit the field via skip_serializing_if, got: {serialized}"
    );
}

// T-310: json_chunk_emits_empty_array_when_some_empty
#[test]
fn json_chunk_emits_empty_array_when_some_empty() {
    let chunk = sample_chunk_with_flags(Some(Vec::new()));
    let serialized = serde_json::to_string(&chunk).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    let array = parsed["injection_flags"]
        .as_array()
        .expect("injection_flags must be present as an array");
    assert!(
        array.is_empty(),
        "injection_flags=Some(vec![]) must serialize to empty array, got: {parsed}"
    );
}

// T-311: json_chunk_emits_flag_strings_when_some_non_empty
#[test]
fn json_chunk_emits_flag_strings_when_some_non_empty() {
    let chunk = sample_chunk_with_flags(Some(vec!["injection.instruction-override"]));
    let serialized = serde_json::to_string(&chunk).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    assert_eq!(
        parsed["injection_flags"][0], "injection.instruction-override",
        "first injection_flags entry must be the supplied marker, got: {parsed}"
    );
    assert_eq!(
        parsed["injection_flags"].as_array().unwrap().len(),
        1,
        "injection_flags must contain exactly the supplied entries, got: {parsed}"
    );
}

// T-312: json_response_emits_injection_check_ran_lowercase
#[test]
fn json_response_emits_injection_check_ran_lowercase() {
    let response = empty_response_with(InjectionCheck::Ran);
    let serialized = serde_json::to_string(&response).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    assert_eq!(
        parsed["injection_check"], "ran",
        "InjectionCheck::Ran must serialize to lowercase \"ran\", got: {parsed}"
    );
}

// T-313: format_results_json_emits_injection_check_and_per_chunk_flags
#[test]
fn format_results_json_emits_injection_check_and_per_chunk_flags() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/foo.rs".to_owned(),
            chunk_type: storage::ChunkType::RustFn,
            name: Some("foo".to_owned()),
            content: "fn foo() {}".to_owned(),
            start_line: 1,
            end_line: 3,
            parent_chunk_id: None,
            source_kind: Some(storage::SourceKind::Src),
            injection_flags: Some(vec!["x".to_owned()]),
        },
        chunk_id: Some(1),
        distance: 0.1,
        match_source: storage::MatchSource::Fts,
        score: 0.9,
    }];

    let json = format_results_json(&results, false, vec![]);
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(
        parsed["injection_check"], "ran",
        "FR-310a: format_results_json must emit top-level injection_check=\"ran\", got: {parsed}"
    );
    assert_eq!(
        parsed["results"][0]["injection_flags"][0], "x",
        "FR-310b: per-chunk injection_flags must be borrowed from SearchResult.chunk, got: {parsed}"
    );
}

// T-372: json_chunk_skips_source_kind_when_none
#[test]
fn json_chunk_skips_source_kind_when_none() {
    let chunk = sample_chunk_with_source_kind(None);
    let serialized = serde_json::to_string(&chunk).unwrap();
    assert!(
        !serialized.contains("source_kind"),
        "JsonChunk with source_kind=None must omit the field via skip_serializing_if, got: {serialized}"
    );
}

// T-373: format_results_json_emits_per_chunk_source_kind
#[test]
fn format_results_json_emits_per_chunk_source_kind() {
    let results = vec![storage::SearchResult {
        chunk: storage::Chunk {
            file_path: "src/foo.rs".to_owned(),
            chunk_type: storage::ChunkType::RustFn,
            name: Some("foo".to_owned()),
            content: "fn foo() {}".to_owned(),
            start_line: 1,
            end_line: 3,
            parent_chunk_id: None,
            source_kind: Some(storage::SourceKind::Src),
            injection_flags: None,
        },
        chunk_id: Some(1),
        distance: 0.1,
        match_source: storage::MatchSource::Fts,
        score: 0.9,
    }];

    let json = format_results_json(&results, false, vec![]);
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

    assert_eq!(
        parsed["results"][0]["source_kind"], "src",
        "FR-009a: per-chunk source_kind must be borrowed from SearchResult.chunk, got: {parsed}"
    );
}

// T-322: injection_check_as_str_is_exhaustive_over_all_variants
#[test]
fn injection_check_as_str_is_exhaustive_over_all_variants() {
    assert_eq!(InjectionCheck::Ran.as_str(), "ran");
    assert_eq!(InjectionCheck::Skipped.as_str(), "skipped");
    assert_eq!(InjectionCheck::Unavailable.as_str(), "unavailable");
}
