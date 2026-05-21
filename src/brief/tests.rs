use std::collections::HashMap;

use rusqlite::Connection;
use tempfile::{TempDir, tempdir};
use tracing_test::traced_test;

use super::*;
use crate::storage::{
    EMBEDDING_DIMS, NewChunk, RefKind, Reference, ce, insert_chunk, open_db,
    replace_file_references,
};

// T-574: expand_plan_empty_seeds_returns_empty_degraded [RC-014 #140]
#[traced_test]
#[test]
fn expand_plan_empty_seeds_returns_empty_degraded() {
    let (conn, _dir) = test_db();
    let task = task_with_seeds(vec![], 1);
    let output = expand_plan(&conn, &task).unwrap();
    assert!(output.chunks.is_empty(), "empty seeds yield empty chunks");
    assert!(
        output.degraded,
        "empty seeds must mark degraded as invariant violation"
    );
    assert_eq!(output.total_chunks, 0);
    assert_eq!(output.total_bytes, 0);
    assert!(
        logs_contain("expand_plan called with empty seeds"),
        "expected warn for empty seeds"
    );
}

fn test_db() -> (Connection, TempDir) {
    let dir = tempdir().unwrap();
    let conn = open_db(&dir.path().join("test.db")).unwrap();
    (conn, dir)
}

fn seed_file(value: &str) -> Seed {
    Seed {
        kind: SeedKind::File,
        value: value.to_owned(),
    }
}

fn task_with_seeds(seeds: Vec<Seed>, depth: u32) -> TaskBrief {
    TaskBrief {
        task: "find search".to_owned(),
        seeds,
        depth,
        max_chunks: 80,
        max_bytes: 80_000,
    }
}

fn insert_test_chunk(conn: &Connection, file_path: &str, name: &'static str, start: u32) {
    insert_chunk(
        conn,
        file_path,
        &NewChunk {
            chunk_type: &ChunkType::RustFn,
            name: Some(name),
            content: "fn body() {}",
            start_line: start,
            end_line: start + 2,
            parent_index: None,
            source_kind: None,
            injection_flags: None,
        },
        "h",
        &ce(vec![0.0_f32; EMBEDDING_DIMS]),
        None,
    )
    .unwrap();
}

fn make_brief_chunk(file_path: &str, content: &str) -> BriefChunk {
    BriefChunk {
        file_path: file_path.to_owned(),
        start_line: 1,
        end_line: 3,
        chunk_type: ChunkType::RustFn,
        content: content.to_owned(),
        included_reason: ChunkInclusionReason::Forward(1),
        source_kind: None,
        injection_flags: None,
    }
}

// T-601: apply_cap_drops_high_depth_low_incoming_first
#[test]
fn apply_cap_drops_high_depth_low_incoming_first() {
    let chunks = vec![
        make_brief_chunk("src/a.rs", "a"),
        make_brief_chunk("src/b.rs", "b"),
        make_brief_chunk("src/c.rs", "c"),
        make_brief_chunk("src/d.rs", "d"),
    ];
    let depth_by_path: HashMap<String, u32> = [
        ("src/a.rs".to_owned(), 0),
        ("src/b.rs".to_owned(), 1),
        ("src/c.rs".to_owned(), 1),
        ("src/d.rs".to_owned(), 2),
    ]
    .into_iter()
    .collect();
    let incoming_counts: HashMap<String, u32> = [
        ("src/a.rs".to_owned(), 10),
        ("src/b.rs".to_owned(), 5),
        ("src/c.rs".to_owned(), 2),
        ("src/d.rs".to_owned(), 1),
    ]
    .into_iter()
    .collect();

    let kept = apply_cap(chunks, &depth_by_path, &incoming_counts, 2, u32::MAX);

    assert_eq!(kept.len(), 2);
    let kept_paths: Vec<&str> = kept.iter().map(|c| c.file_path.as_str()).collect();
    assert!(
        kept_paths.contains(&"src/a.rs"),
        "seed (depth=0) must survive cap, got: {kept_paths:?}"
    );
    assert!(
        kept_paths.contains(&"src/b.rs"),
        "depth=1 incoming=5 must survive over depth=1 incoming=2, got: {kept_paths:?}"
    );
}

// T-605: render_json_emits_spec_shape
#[test]
fn render_json_emits_spec_shape() {
    let output = BriefOutput {
        chunks: vec![BriefChunk {
            file_path: "src/foo.rs".to_owned(),
            start_line: 10,
            end_line: 12,
            chunk_type: ChunkType::RustFn,
            content: "fn foo() {}".to_owned(),
            included_reason: ChunkInclusionReason::Forward(2),
            source_kind: None,
            injection_flags: None,
        }],
        degraded: true,
        total_chunks: 1,
        total_bytes: 11,
    };

    let rendered = render_json(&output);
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();

    assert_eq!(parsed["degraded"], true);
    assert_eq!(parsed["chunks"][0]["file_path"], "src/foo.rs");
    assert_eq!(parsed["chunks"][0]["start_line"], 10);
    assert_eq!(parsed["chunks"][0]["end_line"], 12);
    assert_eq!(parsed["chunks"][0]["chunk_type"], "rust_fn");
    assert_eq!(parsed["chunks"][0]["content"], "fn foo() {}");
    assert_eq!(parsed["chunks"][0]["included_reason"], "forward-2");
}

#[test]
fn render_json_includes_seed_and_modkinds() {
    let chunk = |reason: ChunkInclusionReason| BriefChunk {
        file_path: "x".to_owned(),
        start_line: 1,
        end_line: 1,
        chunk_type: ChunkType::Other,
        content: "".to_owned(),
        included_reason: reason,
        source_kind: None,
        injection_flags: None,
    };
    let output = BriefOutput {
        chunks: vec![
            chunk(ChunkInclusionReason::Seed),
            chunk(ChunkInclusionReason::Sibling),
            chunk(ChunkInclusionReason::ModDecl),
        ],
        degraded: false,
        total_chunks: 3,
        total_bytes: 0,
    };
    let parsed: serde_json::Value = serde_json::from_str(&render_json(&output)).unwrap();
    assert_eq!(parsed["chunks"][0]["included_reason"], "seed");
    assert_eq!(parsed["chunks"][1]["included_reason"], "sibling");
    assert_eq!(parsed["chunks"][2]["included_reason"], "mod-decl");
}

// T-604: render_plain_outputs_separator_and_header
#[test]
fn render_plain_outputs_separator_and_header() {
    let chunk = |path: &str, content: &str, start: u32, end: u32| BriefChunk {
        file_path: path.to_owned(),
        start_line: start,
        end_line: end,
        chunk_type: ChunkType::RustFn,
        content: content.to_owned(),
        included_reason: ChunkInclusionReason::Seed,
        source_kind: None,
        injection_flags: None,
    };
    let output = BriefOutput {
        chunks: vec![
            chunk("src/foo.rs", "fn foo() {}", 10, 12),
            chunk("src/bar.rs", "fn bar() {}", 20, 22),
        ],
        degraded: false,
        total_chunks: 2,
        total_bytes: 0,
    };

    let rendered = render_plain(&output);

    assert_eq!(
        rendered,
        "src/foo.rs:10-12\nfn foo() {}\n---\nsrc/bar.rs:20-22\nfn bar() {}"
    );
}

#[test]
fn render_plain_returns_empty_string_for_empty_output() {
    let output = BriefOutput {
        chunks: vec![],
        degraded: false,
        total_chunks: 0,
        total_bytes: 0,
    };
    assert_eq!(render_plain(&output), "");
}

// T-606: render_plain_prepends_degraded_note [Spec FR-014]
#[test]
fn render_plain_prepends_degraded_note() {
    let output = BriefOutput {
        chunks: vec![BriefChunk {
            file_path: "src/foo.rs".to_owned(),
            start_line: 1,
            end_line: 3,
            chunk_type: ChunkType::RustFn,
            content: "fn foo() {}".to_owned(),
            included_reason: ChunkInclusionReason::Seed,
            source_kind: None,
            injection_flags: None,
        }],
        degraded: true,
        total_chunks: 1,
        total_bytes: 11,
    };
    let rendered = render_plain(&output);
    assert!(
        rendered.starts_with("Note: degraded mode — FTS-only seed selection\n"),
        "expected degraded note prefix, got: {rendered}"
    );
    assert!(
        rendered.contains("src/foo.rs:1-3\nfn foo() {}"),
        "chunk body must follow the note, got: {rendered}"
    );
}

// T-607: render_plain_empty_degraded_returns_only_note
#[test]
fn render_plain_empty_degraded_returns_only_note() {
    let output = BriefOutput {
        chunks: vec![],
        degraded: true,
        total_chunks: 0,
        total_bytes: 0,
    };
    assert_eq!(
        render_plain(&output),
        "Note: degraded mode — FTS-only seed selection"
    );
}

// T-602: topo_sort_orders_dependencies_first
#[test]
fn topo_sort_orders_dependencies_first() {
    let chunks = vec![
        make_brief_chunk("src/a.rs", "a"),
        make_brief_chunk("src/b.rs", "b"),
        make_brief_chunk("src/c.rs", "c"),
    ];
    let edges = vec![
        ("src/a.rs".to_owned(), "src/b.rs".to_owned()),
        ("src/b.rs".to_owned(), "src/c.rs".to_owned()),
    ];

    let sorted = topo_sort(chunks, &edges);

    let order: Vec<&str> = sorted.iter().map(|c| c.file_path.as_str()).collect();
    assert_eq!(
        order,
        vec!["src/c.rs", "src/b.rs", "src/a.rs"],
        "BR-002: dependencies (depended-upon) come first"
    );
}

// T-600: expand_plan_returns_seed_and_forward_chunks
#[test]
fn expand_plan_returns_seed_and_forward_chunks() {
    let (conn, _dir) = test_db();
    insert_test_chunk(&conn, "src/a.rs", "a", 1);
    insert_test_chunk(&conn, "src/b.rs", "b", 1);
    replace_file_references(
        &conn,
        "src/a.rs",
        &[Reference {
            source_file: "src/a.rs".into(),
            target_file: "src/b.rs".into(),
            symbol_name: None,
            ref_kind: RefKind::Named,
        }],
    )
    .unwrap();

    let task = task_with_seeds(vec![seed_file("src/a.rs")], 1);
    let output = expand_plan(&conn, &task).unwrap();

    assert_eq!(output.chunks.len(), 2);
    assert!(!output.degraded);

    let by_path: HashMap<&str, &BriefChunk> = output
        .chunks
        .iter()
        .map(|c| (c.file_path.as_str(), c))
        .collect();
    assert_eq!(
        by_path["src/a.rs"].included_reason,
        ChunkInclusionReason::Seed
    );
    assert_eq!(
        by_path["src/b.rs"].included_reason,
        ChunkInclusionReason::Forward(1)
    );
}

// T-603: expand_plan_is_deterministic [Spec T-008]
#[test]
fn expand_plan_is_deterministic() {
    let (conn, _dir) = test_db();
    insert_test_chunk(&conn, "src/a.rs", "a", 1);
    insert_test_chunk(&conn, "src/b.rs", "b", 1);
    insert_test_chunk(&conn, "src/c.rs", "c", 1);
    let edge = |source: &str, target: &str| Reference {
        source_file: source.into(),
        target_file: target.into(),
        symbol_name: None,
        ref_kind: RefKind::Named,
    };
    replace_file_references(&conn, "src/a.rs", &[edge("src/a.rs", "src/b.rs")]).unwrap();
    replace_file_references(&conn, "src/b.rs", &[edge("src/b.rs", "src/c.rs")]).unwrap();

    let task = task_with_seeds(vec![seed_file("src/a.rs")], 2);
    let signature = |o: &BriefOutput| -> Vec<(String, u32, u32, String)> {
        o.chunks
            .iter()
            .map(|c| {
                (
                    c.file_path.clone(),
                    c.start_line,
                    c.end_line,
                    c.content.clone(),
                )
            })
            .collect()
    };

    let first = expand_plan(&conn, &task).unwrap();
    let second = expand_plan(&conn, &task).unwrap();

    assert_eq!(signature(&first), signature(&second));
    assert_eq!(first.total_chunks, second.total_chunks);
    assert_eq!(first.total_bytes, second.total_bytes);
}

// T-314: render_json_emits_per_chunk_injection_flags
#[test]
fn render_json_emits_per_chunk_injection_flags() {
    let output = BriefOutput {
        chunks: vec![BriefChunk {
            file_path: "src/foo.rs".to_owned(),
            start_line: 1,
            end_line: 3,
            chunk_type: ChunkType::RustFn,
            content: "fn foo() {}".to_owned(),
            included_reason: ChunkInclusionReason::Seed,
            source_kind: None,
            injection_flags: Some(vec!["y".to_owned()]),
        }],
        degraded: false,
        total_chunks: 1,
        total_bytes: 11,
    };

    let rendered = render_json(&output);
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();

    assert_eq!(
        parsed["chunks"][0]["injection_flags"][0], "y",
        "FR-311b/FR-313b: BriefChunk.injection_flags must propagate to JsonChunk, got: {parsed}"
    );
    assert_eq!(
        parsed["chunks"][0]["injection_flags"]
            .as_array()
            .unwrap()
            .len(),
        1,
        "per-chunk injection_flags must contain exactly the supplied entries, got: {parsed}"
    );
}

// T-390: render_json_emits_per_chunk_source_kind
#[test]
fn render_json_emits_per_chunk_source_kind() {
    let output = BriefOutput {
        chunks: vec![BriefChunk {
            file_path: "src/foo.rs".to_owned(),
            start_line: 1,
            end_line: 3,
            chunk_type: ChunkType::RustFn,
            content: "fn foo() {}".to_owned(),
            included_reason: ChunkInclusionReason::Seed,
            source_kind: Some("src".to_owned()),
            injection_flags: None,
        }],
        degraded: false,
        total_chunks: 1,
        total_bytes: 11,
    };

    let rendered = render_json(&output);
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();

    assert_eq!(
        parsed["chunks"][0]["source_kind"], "src",
        "FR-009a: BriefChunk.source_kind must propagate to JsonChunk, got: {parsed}"
    );
}

// T-395: render_json_skips_source_kind_when_none
#[test]
fn render_json_skips_source_kind_when_none() {
    let output = BriefOutput {
        chunks: vec![BriefChunk {
            file_path: "src/foo.rs".to_owned(),
            start_line: 1,
            end_line: 3,
            chunk_type: ChunkType::RustFn,
            content: "fn foo() {}".to_owned(),
            included_reason: ChunkInclusionReason::Seed,
            source_kind: None,
            injection_flags: None,
        }],
        degraded: false,
        total_chunks: 1,
        total_bytes: 11,
    };

    let rendered = render_json(&output);
    assert!(
        !rendered.contains("source_kind"),
        "JsonChunk with source_kind=None must omit the field via skip_serializing_if, got: {rendered}"
    );
}

// T-315: render_json_emits_injection_check_even_with_empty_chunks
#[test]
fn render_json_emits_injection_check_even_with_empty_chunks() {
    let output = BriefOutput {
        chunks: vec![],
        degraded: false,
        total_chunks: 0,
        total_bytes: 0,
    };

    let rendered = render_json(&output);
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();

    assert_eq!(
        parsed["injection_check"], "ran",
        "BR-302: injection_check must be present at top level even when chunks is empty (no skip_serializing_if), got: {parsed}"
    );
}
