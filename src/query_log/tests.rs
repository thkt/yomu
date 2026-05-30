use super::*;

fn sample_record() -> QueryLogRecord {
    QueryLogRecord {
        timestamp: "2026-05-19T03:33:27Z".to_owned(),
        yomu_version: "0.16.0".to_owned(),
        original_query: "auth handler".to_owned(),
        fts_results: vec![StageHit {
            chunk_id: 1,
            score: 0.5,
            source: "fts".to_owned(),
        }],
        vec_results: vec![],
        rrf_results: vec![],
        reranked_results: vec![],
        final_context_ids: vec![1, 2, 3],
        latency_ms: 123,
    }
}

// T-QL-001: serde round-trip preserves every field.
#[test]
fn record_round_trips_through_json() {
    let original = sample_record();
    let json = serde_json::to_string(&original).expect("serialize");
    let parsed: QueryLogRecord = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed, original);
}

// T-QL-002: multi-line query escapes to a single JSONL line.
#[test]
fn multi_line_query_serializes_to_one_line() {
    let mut record = sample_record();
    record.original_query = "line1\nline2".to_owned();
    let mut buf: Vec<u8> = Vec::new();
    QueryLogWriter::new(&mut buf)
        .write_record(&record)
        .expect("write");
    let s = String::from_utf8(buf).expect("utf8");
    assert!(s.ends_with('\n'), "trailing newline required");
    let body = &s[..s.len() - 1];
    assert!(
        !body.contains('\n'),
        "record body must occupy one line, got: {body}"
    );
    assert!(
        body.contains("line1\\nline2"),
        "newline must be JSON-escaped as \\\\n, got: {body}"
    );
}

// T-QL-003: unicode (Japanese + emoji) round-trips as UTF-8 bytes.
#[test]
fn unicode_query_serializes_as_utf8() {
    let mut record = sample_record();
    record.original_query = "認証 🔐".to_owned();
    let json = serde_json::to_string(&record).expect("serialize");
    assert!(
        json.contains("認証 🔐"),
        "expected literal UTF-8 bytes, got: {json}"
    );
    let parsed: QueryLogRecord = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(parsed.original_query, "認証 🔐");
}

// T-QL-004: append-only writer preserves pre-existing file contents.
#[test]
fn append_only_preserves_existing_lines() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("query_log.jsonl");
    fs::write(&path, "pre1\npre2\npre3\n").expect("seed");
    let mut writer = open_append_writer(&path).expect("open");
    writer.write_record(&sample_record()).expect("write");
    drop(writer);
    let content = fs::read_to_string(&path).expect("read");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 4, "expected 4 lines, got: {content}");
    assert_eq!(lines[0], "pre1");
    assert_eq!(lines[1], "pre2");
    assert_eq!(lines[2], "pre3");
}

// T-QL-007a: final_context_ids type and rank order on the yomu side.
#[test]
fn final_context_ids_is_vec_i64_in_rank_order() {
    let record = QueryLogRecord {
        final_context_ids: vec![10, 20, 30],
        ..sample_record()
    };
    assert_eq!(record.final_context_ids, vec![10_i64, 20, 30]);
}

// T-QL-007b: documented conversion produces Vec<Option<String>>.
#[test]
fn final_context_ids_extracts_to_option_string_vec() {
    let ids: Vec<i64> = vec![10, 20, 30];
    let extracted: Vec<Option<String>> = ids.iter().map(|id| Some(id.to_string())).collect();
    assert_eq!(
        extracted,
        vec![
            Some("10".to_owned()),
            Some("20".to_owned()),
            Some("30".to_owned())
        ]
    );
}

// T-QL-008: latency_ms serializes as a non-negative integer (amici
// QueryResult::latency_ms is u64). A weak `let _: u64 = ...` compile-time
// check would pass even if the field were later widened, so assert the
// wire representation instead.
#[test]
fn latency_ms_serializes_as_u64_integer() {
    let record = QueryLogRecord {
        latency_ms: 123,
        ..sample_record()
    };
    let value = serde_json::to_value(&record).expect("to_value");
    let latency = value
        .get("latency_ms")
        .and_then(serde_json::Value::as_u64)
        .expect("latency_ms must be a u64 JSON number");
    assert_eq!(latency, 123);
}

// T-QL-011: resolve_log_path falls back to $HOME/.local/share when XDG is unset.
#[test]
fn resolve_log_path_falls_back_to_home_local_share() {
    let resolved = resolve_log_path(None, "/tmp/test-home");
    assert_eq!(
        resolved,
        PathBuf::from("/tmp/test-home/.local/share/yomu/query_log.jsonl")
    );
}

// T-QL-011 supplement: XDG_DATA_HOME set is honoured.
#[test]
fn resolve_log_path_uses_xdg_when_set() {
    let resolved = resolve_log_path(Some("/var/xdg"), "/tmp/test-home");
    assert_eq!(resolved, PathBuf::from("/var/xdg/yomu/query_log.jsonl"));
}

// T-QL-011 supplement: empty XDG falls back per spec.
#[test]
fn resolve_log_path_empty_xdg_falls_back() {
    let resolved = resolve_log_path(Some(""), "/tmp/test-home");
    assert_eq!(
        resolved,
        PathBuf::from("/tmp/test-home/.local/share/yomu/query_log.jsonl")
    );
}

// T-QL-012: log emit median latency ≤ 10ms over 10 samples (in-memory writer).
// CI-skipped because wall-clock thresholds flake under load; run locally
// with `cargo test --lib log_emit_latency -- --ignored`.
#[ignore]
#[test]
fn log_emit_latency_median_under_10ms() {
    use std::time::Instant;
    let mut samples_us: Vec<u128> = Vec::with_capacity(10);
    for _ in 0..10 {
        let mut buf: Vec<u8> = Vec::with_capacity(4096);
        let start = Instant::now();
        QueryLogWriter::new(&mut buf)
            .write_record(&sample_record())
            .expect("write");
        samples_us.push(start.elapsed().as_micros());
    }
    samples_us.sort_unstable();
    let median = samples_us[samples_us.len() / 2];
    assert!(
        median <= 10_000,
        "median latency {median}us exceeded 10ms budget"
    );
}
