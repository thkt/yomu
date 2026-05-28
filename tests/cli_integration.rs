use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use tempfile::{TempDir, tempdir};

fn yomu_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_yomu"))
}

fn setup_brief_chain_project() -> TempDir {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("lib.rs"), "pub mod a;\npub mod b;\npub mod c;\n").unwrap();
    fs::write(
        src.join("a.rs"),
        "use crate::b;\npub fn run() { b::work(); }\n",
    )
    .unwrap();
    fs::write(
        src.join("b.rs"),
        "use crate::c;\npub fn work() { c::utility(); }\n",
    )
    .unwrap();
    fs::write(src.join("c.rs"), "pub fn utility() {}\n").unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"gt_chain\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .unwrap();
    let out = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    dir
}

// T-495: version_flag
#[test]
fn version_flag() {
    let output = yomu_cmd().arg("--version").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("yomu "),
        "expected 'yomu <version>', got: {stdout}"
    );
}

// T-496: help_flag
#[test]
fn help_flag() {
    let output = yomu_cmd().arg("--help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("search"),
        "help should mention search command"
    );
    assert!(
        stdout.contains("impact"),
        "help should mention impact command"
    );
}

// T-497: search_limit_out_of_range
#[test]
fn search_limit_out_of_range() {
    let output = yomu_cmd()
        .args(["search", "test", "--limit", "0"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(64),
        "clap parse failure must exit with sysexits USAGE (64)"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("0") || stderr.contains("invalid"),
        "should reject limit=0: {stderr}"
    );
}

// T-498: search_limit_too_large
#[test]
fn search_limit_too_large() {
    let output = yomu_cmd()
        .args(["search", "test", "--limit", "999"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(64),
        "clap parse failure must exit with sysexits USAGE (64)"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("999") || stderr.contains("invalid"),
        "should reject limit=999: {stderr}"
    );
}

// T-499: search_offset_too_large
#[test]
fn search_offset_too_large() {
    let output = yomu_cmd()
        .args(["search", "test", "--offset", "999"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(64),
        "clap parse failure must exit with sysexits USAGE (64)"
    );
}

// T-500: impact_depth_too_large
#[test]
fn impact_depth_too_large() {
    let output = yomu_cmd()
        .args(["impact", "src/foo.ts", "--depth", "99"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(64),
        "clap parse failure must exit with sysexits USAGE (64)"
    );
}

// T-501: search_default_limit_accepted
#[test]
fn search_default_limit_accepted() {
    // May fail for other reasons (e.g. no project root), but not argument validation
    let output = yomu_cmd().args(["search", "test"]).output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("invalid value"),
        "default limit should be accepted: {stderr}"
    );
}

// T-502: unknown_flag_fails
#[test]
fn unknown_flag_fails() {
    let output = yomu_cmd().arg("--nonexistent").output().unwrap();
    assert_eq!(
        output.status.code(),
        Some(64),
        "clap parse failure must exit with sysexits USAGE (64)"
    );
}

// T-503: no_subcommand_exits_usage
#[test]
fn no_subcommand_exits_usage() {
    let output = yomu_cmd().output().unwrap();
    assert_eq!(
        output.status.code(),
        Some(64),
        "missing subcommand must exit with sysexits USAGE (64)"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("subcommand"),
        "stderr should mention subcommand: {stderr}"
    );
}

// --- Success-path integration tests ---

fn setup_project() -> TempDir {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("Button.tsx"),
        "export function Button() { return <button>Click</button>; }\n",
    )
    .unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .unwrap();
    // pre-index so tests don't depend on auto-indexing
    let out = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "setup index failed");
    dir
}

// T-503: index_then_status_then_search
#[test]
fn index_then_status_then_search() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("Button.tsx"),
        "export function Button() { return <button>Click</button>; }\n",
    )
    .unwrap();
    // git init so yomu detects project root
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    // index
    let output = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("1 files chunked"),
        "expected 1 file: {stdout}"
    );

    // status
    let output = yomu_cmd()
        .arg("status")
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "status failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("Files: 1"), "expected Files: 1: {stdout}");
    assert!(stdout.contains("Chunks:"), "expected Chunks line: {stdout}");

    // search (read-only over the index built above)
    let output = yomu_cmd()
        .args(["search", "button"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Button"),
        "expected Button in results: {stdout}"
    );
}

// T-504: search_stdin_query
#[test]
fn search_stdin_query() {
    let dir = setup_project();
    let mut child = yomu_cmd()
        .args(["search", "-"])
        .current_dir(dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Write;
    child.stdin.take().unwrap().write_all(b"button").unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "stdin search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Button"),
        "expected Button in results: {stdout}"
    );
}

// T-505: search_format_json
#[test]
fn search_format_json() {
    let dir = setup_project();
    let output = yomu_cmd()
        .args(["search", "button", "--json"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "json search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    let results = parsed["results"]
        .as_array()
        .expect("should have results array");
    assert!(!results.is_empty(), "should have results: {stdout}");
    assert!(
        parsed.get("degraded").is_some(),
        "should have degraded field: {stdout}"
    );
    assert!(
        parsed["notes"].is_array(),
        "should have notes array: {stdout}"
    );
    let first = &results[0];
    for field in [
        "file",
        "name",
        "type",
        "start_line",
        "end_line",
        "score",
        "content",
    ] {
        assert!(
            first.get(field).is_some(),
            "result missing '{field}' field: {stdout}"
        );
    }
}

// T-506: search_json_degraded_note_when_model_unavailable
//
// The project is indexed (embeddings present via the stub), but the model is
// unavailable at search time (YOMU_TEST_EMBEDDER=unavailable, test-support
// seam). Read-only search must fall back to FTS, return results, and surface a
// degraded note pointing at `yomu model download`.
//
// `YOMU_TEST_EMBEDDER` is a test-support-only seam, so this test compiles only
// under `--features test-support` (the production binary ignores the env var).
#[cfg(feature = "test-support")]
#[test]
fn search_json_degraded_note_when_model_unavailable() {
    let dir = setup_project();

    let output = yomu_cmd()
        .args(["search", "button", "--json"])
        .current_dir(dir.path())
        .env("YOMU_TEST_EMBEDDER", "unavailable")
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "json search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    assert_eq!(
        parsed["degraded"],
        serde_json::Value::Bool(true),
        "model unavailable must mark the envelope degraded: {stdout}"
    );
    assert!(
        parsed["results"].as_array().is_some_and(|r| !r.is_empty()),
        "FTS fallback must still return results: {stdout}"
    );
    let notes = parsed["notes"].as_array().expect("should have notes array");
    assert!(
        notes.iter().any(|n| n
            .as_str()
            .is_some_and(|s| s.contains("yomu model download"))),
        "should include degraded note with download hint: {stdout}"
    );
}

// T-507: index_dry_run
#[test]
fn index_dry_run() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("App.tsx"), "export function App() {}").unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .unwrap();

    let output = yomu_cmd()
        .args(["index", "--dry-run"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("1 files to process"),
        "should show files to process: {stdout}"
    );
}

// T-508: rebuild_after_index
#[test]
fn rebuild_after_index() {
    let dir = setup_project();
    let output = yomu_cmd()
        .arg("rebuild")
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "rebuild failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("1 files chunked"),
        "rebuild should re-process files: {stdout}"
    );
}

// T-509: search_stdin_empty_query_fails
#[test]
fn search_stdin_empty_query_fails() {
    let dir = setup_project();
    let mut child = yomu_cmd()
        .args(["search", "-"])
        .current_dir(dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Close stdin immediately (empty input)
    drop(child.stdin.take());
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success(), "empty stdin should fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("empty"),
        "should mention empty query: {stderr}"
    );
}

// T-510: rebuild_dry_run_reports_all_files
#[test]
fn rebuild_dry_run_reports_all_files() {
    let dir = setup_project();
    let output = yomu_cmd()
        .args(["rebuild", "--dry-run"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "rebuild --dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("1 files to process"),
        "rebuild --dry-run should report files to process (force=true): {stdout}"
    );
}

// T-511: shorthand_query_runs_as_search
#[test]
fn shorthand_query_runs_as_search() {
    let dir = setup_project();
    let output = yomu_cmd()
        .arg("button")
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "shorthand search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Button"),
        "expected Button in results: {stdout}"
    );
}

// T-512: shorthand_explicit_search_still_works
#[test]
fn shorthand_explicit_search_still_works() {
    let dir = setup_project();
    let output = yomu_cmd()
        .args(["search", "button"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "explicit search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Button"),
        "expected Button in results: {stdout}"
    );
}

// T-513: shorthand_help_not_treated_as_search
#[test]
fn shorthand_help_not_treated_as_search() {
    let output = yomu_cmd().arg("help").output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("search"),
        "help should list search command: {stdout}"
    );
}

// T-514: shorthand_status_not_treated_as_search
#[test]
fn shorthand_status_not_treated_as_search() {
    let dir = setup_project();
    let output = yomu_cmd()
        .arg("status")
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "status failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("Files:"),
        "should show status output: {stdout}"
    );
}

// T-515: shorthand_typo_not_treated_as_search
#[test]
fn shorthand_typo_not_treated_as_search() {
    let output = yomu_cmd().arg("stauts").output().unwrap();
    assert!(
        !output.status.success(),
        "typo 'stauts' should fail, not become a search"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unrecognized subcommand"),
        "should show clap's unrecognized subcommand error: {stderr}"
    );
}

// T-516: shorthand_near_command_name_still_searches
#[test]
fn shorthand_near_command_name_still_searches() {
    // "state" is OSA distance 2 from "status" — should be treated as search, not typo
    let dir = setup_project();
    let output = yomu_cmd()
        .arg("state")
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "'state' should be a search query, not a typo: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Search may return 0 results for "state", but the output proves
    // it went through the search path (not clap error)
    assert!(
        stdout.contains("results") || stdout.contains("src/"),
        "should show search output (not clap error): {stdout}"
    );
}

// --- Global --json flag tests ---

// T-517: json_flag_with_status
#[test]
fn json_flag_with_status() {
    let dir = setup_project();
    let output = yomu_cmd()
        .args(["--json", "status"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "--json status failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    for field in ["files", "chunks", "embedded_chunks", "references"] {
        assert!(
            parsed.get(field).is_some(),
            "status missing '{field}' field: {stdout}"
        );
    }
}

// T-518: json_flag_with_index
#[test]
fn json_flag_with_index() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("App.tsx"),
        "export function App() { return <div/>; }\n",
    )
    .unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .unwrap();

    let output = yomu_cmd()
        .args(["--json", "index"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "--json index failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    assert_eq!(parsed["files_processed"], 1, "should process 1 file");
    assert!(
        parsed["chunks_created"].as_u64().unwrap() > 0,
        "should create chunks"
    );
}

// T-519: json_flag_with_rebuild
#[test]
fn json_flag_with_rebuild() {
    let dir = setup_project();
    let output = yomu_cmd()
        .args(["--json", "rebuild"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "--json rebuild failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    assert_eq!(parsed["files_processed"], 1);
    assert!(parsed["chunks_created"].as_u64().unwrap() > 0);
}

// T-520: json_flag_with_search_shorthand
#[test]
fn json_flag_with_search_shorthand() {
    let dir = setup_project();
    let output = yomu_cmd()
        .args(["--json", "button"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "--json shorthand failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    assert!(
        parsed["results"].is_array(),
        "shorthand --json should produce search JSON: {stdout}"
    );
}

// T-521: json_flag_after_subcommand
#[test]
fn json_flag_after_subcommand() {
    let dir = setup_project();
    let output = yomu_cmd()
        .args(["status", "--json"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "status --json failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    assert!(
        parsed.get("files").is_some(),
        "should produce JSON: {stdout}"
    );
}

// T-522: deprecated_format_json_still_works
#[test]
fn deprecated_format_json_still_works() {
    let dir = setup_project();
    let output = yomu_cmd()
        .args(["search", "button", "--format", "json"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "deprecated --format json failed: {stderr}"
    );
    let _: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    assert!(
        stderr.contains("--format is deprecated"),
        "should warn about deprecation: {stderr}"
    );
}

// T-523: json_flag_with_index_dry_run
#[test]
fn json_flag_with_index_dry_run() {
    let dir = setup_project();
    let output = yomu_cmd()
        .args(["--json", "index", "--dry-run"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "--json index --dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    for field in [
        "files_to_process",
        "files_to_skip",
        "total_files",
        "files_errored",
        "orphans_to_remove",
    ] {
        assert!(
            parsed.get(field).is_some(),
            "dry-run JSON missing '{field}': {stdout}"
        );
    }
}

// T-524: json_flag_with_rebuild_dry_run
#[test]
fn json_flag_with_rebuild_dry_run() {
    let dir = setup_project();
    let output = yomu_cmd()
        .args(["--json", "rebuild", "--dry-run"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "--json rebuild --dry-run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    assert!(
        parsed["total_files"].as_u64().unwrap() > 0,
        "should have files: {stdout}"
    );
}

// T-525: json_flag_with_impact
#[test]
fn json_flag_with_impact() {
    let dir = setup_project();
    // Add a file that imports Button so impact has dependents
    let src = dir.path().join("src");
    fs::write(
        src.join("App.tsx"),
        "import { Button } from './Button';\nexport function App() { return <Button/>; }\n",
    )
    .unwrap();
    // Re-index to pick up the new file + references
    let out = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "re-index failed");

    let output = yomu_cmd()
        .args(["--json", "impact", "src/Button.tsx"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "--json impact failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    assert_eq!(parsed["target"], "src/Button.tsx");
    assert!(
        parsed.get("in_index").is_some(),
        "should have in_index: {stdout}"
    );
    assert!(
        parsed["dependents"].is_array(),
        "should have dependents: {stdout}"
    );
    assert!(parsed.get("total").is_some(), "should have total: {stdout}");
}

// T-572: impact_json_includes_direct_references_with_kind_and_symbol
#[test]
fn impact_json_includes_direct_references_with_kind_and_symbol() {
    let dir = setup_project();
    let src = dir.path().join("src");
    fs::write(
        src.join("App.tsx"),
        "import { Button } from './Button';\nexport function App() { return <Button/>; }\n",
    )
    .unwrap();
    let out = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(out.status.success(), "re-index failed");

    let output = yomu_cmd()
        .args(["--json", "impact", "src/Button.tsx"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "impact failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    let app_dep = parsed["dependents"]
        .as_array()
        .unwrap_or_else(|| panic!("dependents should be array: {stdout}"))
        .iter()
        .find(|d| d["file_path"] == "src/App.tsx")
        .unwrap_or_else(|| panic!("App.tsx should be a dependent: {stdout}"));
    assert_eq!(app_dep["depth"], 1, "App.tsx is direct dependent: {stdout}");
    let refs = app_dep["references"]
        .as_array()
        .unwrap_or_else(|| panic!("should have references array: {stdout}"));
    assert!(
        refs.iter().any(|r| r["via_symbol"] == "Button"),
        "should include via_symbol=Button: {stdout}"
    );
    assert!(
        refs.iter().all(|r| r["ref_kind"].is_string()),
        "every reference should carry ref_kind: {stdout}"
    );
}

// T-550: probe_embed_flag_rejected
#[test]
fn probe_embed_flag_rejected() {
    let output = yomu_cmd()
        .args(["--probe-embed", "/nonexistent/model/dir"])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "--probe-embed should fail: {:?}",
        output.status
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unexpected argument") || stderr.contains("unrecognized"),
        "should show unrecognized flag error, got: {stderr}"
    );
}

// T-610: brief_integration_includes_forward_closure [Spec FR-015 minimal GT]
#[test]
fn brief_integration_includes_forward_closure() {
    let dir = setup_brief_chain_project();
    let output = yomu_cmd()
        .args([
            "brief",
            "find utility",
            "--seed-file",
            "src/a.rs",
            "--depth",
            "2",
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "brief failed: stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    for expected in ["src/a.rs", "src/b.rs", "src/c.rs"] {
        assert!(
            stdout.contains(expected),
            "forward closure recall: expected {expected} in output, got: {stdout}"
        );
    }
}

// T-611: brief_integration_rejects_empty_task [Spec FR-010b]
#[test]
fn brief_integration_rejects_empty_task() {
    let dir = setup_brief_chain_project();
    let output = yomu_cmd()
        .args(["brief", ""])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "empty task must fail, got success with: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(
        output.status.code(),
        Some(64),
        "empty task must exit 64 (sysexits USAGE), got: {:?} (stderr: {})",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
}

// T-706: brief_integration_rejects_seed_symbol [Issue #138 COV-008]
// FR-010b: --seed-symbol is not yet implemented. The CLI must exit 64
// (sysexits USAGE) and point the user at --seed-file.
#[test]
fn brief_integration_rejects_seed_symbol() {
    let dir = setup_brief_chain_project();
    let output = yomu_cmd()
        .args(["brief", "find work", "--seed-symbol", "work"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "--seed-symbol must fail, got success with: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert_eq!(
        output.status.code(),
        Some(64),
        "--seed-symbol must exit 64 (sysexits USAGE), got: {:?} (stderr: {})",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("seed-file"),
        "rejection must point the user at --seed-file, got stderr: {stderr}"
    );
}

// T-612: brief_integration_json_output_includes_chunks [Spec FR-012]
#[test]
fn brief_integration_json_output_includes_chunks() {
    let dir = setup_brief_chain_project();
    let output = yomu_cmd()
        .args(["--json", "brief", "anything", "--seed-file", "src/a.rs"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "brief --json failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("expected JSON output, parse error {e}: {stdout}"));
    assert!(parsed["chunks"].is_array(), "expected .chunks array");
    assert!(
        !parsed["chunks"].as_array().unwrap().is_empty(),
        "expected at least seed chunk, got: {stdout}"
    );
}

// Minimal TS project exercising a tsconfig `baseUrl` + `paths` alias: the seed
// imports `@/util/format`, which maps to `src/util/format.ts` only when the
// resolver factors `baseUrl` into the alias target.
fn setup_ts_alias_project() -> TempDir {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(src.join("components")).unwrap();
    fs::create_dir_all(src.join("util")).unwrap();
    fs::write(
        dir.path().join("tsconfig.json"),
        "{ \"compilerOptions\": { \"baseUrl\": \"src\", \"paths\": { \"@/*\": [\"*\"] } } }\n",
    )
    .unwrap();
    fs::write(
        src.join("components/widget.tsx"),
        "import { formatLabel } from \"@/util/format\";\n\
         export function Widget() { return formatLabel(\"x\"); }\n",
    )
    .unwrap();
    fs::write(
        src.join("util/format.ts"),
        "export function formatLabel(s: string): string { return s.toUpperCase(); }\n",
    )
    .unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .unwrap();
    let out = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    dir
}

/// Like `setup_ts_alias_project` but the seed reaches its target through a
/// `import type { ... }` specifier instead of a value import, exercising the
/// `RefKind::TypeOnly` edge under the same baseUrl path alias.
fn setup_ts_type_only_project() -> TempDir {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(src.join("components")).unwrap();
    fs::write(
        dir.path().join("tsconfig.json"),
        "{ \"compilerOptions\": { \"baseUrl\": \"src\", \"paths\": { \"@/*\": [\"*\"] } } }\n",
    )
    .unwrap();
    fs::write(
        src.join("components/widget.tsx"),
        "import type { Theme } from \"@/types\";\n\
         export function Widget(t: Theme) { return t.name; }\n",
    )
    .unwrap();
    fs::write(
        src.join("types.ts"),
        "export interface Theme { name: string; }\n",
    )
    .unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .unwrap();
    let out = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    dir
}

// T-613: brief_resolves_tsconfig_baseurl_path_alias [#127 Phase 2]
// The seed imports `@/util/format`; tsconfig maps `@/*` to `*` under
// `baseUrl: "src"`, so the forward closure must include `src/util/format.ts`.
// Before the baseUrl fix the alias target resolved to the repo root and the
// file was dropped from the closure.
#[test]
fn brief_resolves_tsconfig_baseurl_path_alias() {
    let dir = setup_ts_alias_project();
    let output = yomu_cmd()
        .args([
            "brief",
            "widget",
            "--seed-file",
            "src/components/widget.tsx",
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "brief failed: stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("src/util/format.ts"),
        "forward closure must resolve baseUrl path alias `@/util/format` → \
         src/util/format.ts, got: {stdout}"
    );
}

// T-614: brief_resolves_type_only_path_alias [#127 Phase 2]
// The seed reaches `src/types.ts` only through `import type { Theme } from
// "@/types"`. A type-only specifier must still emit a forward edge
// (`RefKind::TypeOnly`), so the same baseUrl path alias must land the target in
// the closure. Probes the last unverified TS gap from #127.
#[test]
fn brief_resolves_type_only_path_alias() {
    let dir = setup_ts_type_only_project();
    let output = yomu_cmd()
        .args(["brief", "theme", "--seed-file", "src/components/widget.tsx"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "brief failed: stdout={}, stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("src/types.ts"),
        "type-only import `@/types` must resolve into forward closure → \
         src/types.ts, got: {stdout}"
    );
}

// --- ADR-0066 Group 2 exit code + JSON error envelope ---

/// Returns the last line in stderr that begins with `{`, parsed as JSON.
/// Tolerates leading `tracing` warn lines emitted by amici::migration when
/// the shared `.yomu/index.db` is migrated by a concurrent subprocess.
fn parse_last_json_envelope(stderr: &str) -> serde_json::Value {
    let line = stderr
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .unwrap_or_else(|| panic!("no JSON envelope line in stderr: {stderr:?}"));
    serde_json::from_str(line.trim())
        .unwrap_or_else(|e| panic!("invalid JSON envelope: {e}: line={line:?}"))
}

/// True if stderr contains a line starting with `error:` (text-mode prefix),
/// tolerating leading `tracing` warn lines.
fn stderr_has_error_line(stderr: &str) -> bool {
    stderr.lines().any(|l| l.starts_with("error:"))
}

// T-EC201: NoQuery surfaces as sysexits USAGE (64) in text mode.
#[test]
fn search_no_query_exits_with_sysexits_usage_text_mode() {
    let output = yomu_cmd()
        .args(["search"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(
        output.status.code(),
        Some(64),
        "expected sysexits USAGE (64), got: {:?}",
        output.status.code()
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr_has_error_line(&stderr),
        "expected text-mode 'error:' line in stderr: {stderr}"
    );
}

// T-EC202: NoQuery with --json emits JSON envelope with USAGE_ERROR code.
#[test]
fn search_no_query_json_emits_usage_error_envelope() {
    let output = yomu_cmd()
        .args(["--json", "search"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(
        output.status.code(),
        Some(64),
        "expected exit code 64, got: {:?}",
        output.status.code()
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let parsed = parse_last_json_envelope(&stderr);
    assert_eq!(parsed["error"]["code"], "USAGE_ERROR", "stderr: {stderr}");
    assert!(
        parsed["error"]["message"].is_string(),
        "expected error.message string: {stderr}"
    );
}

// T-022: `yomu --json search` (no query, empty stdin) emits an envelope
// containing the kind-specific next_step plus retryable: false.
// Spec: Issue #192 Phase 2.2a — FR-002, FR-005.
// This is the integration counterpart to T-002 (next_step) and T-016
// (retryable=false for InvalidInput): the agent reading stderr must see
// both fields so it can branch without re-parsing the message.
#[test]
fn search_no_query_json_envelope_includes_next_step_and_retryable() {
    let output = yomu_cmd()
        .args(["--json", "search"])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(!output.status.success(), "expected non-zero exit");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let parsed = parse_last_json_envelope(&stderr);
    assert!(
        parsed["error"]["next_step"].is_string(),
        "expected error.next_step string per FR-002: {stderr}"
    );
    assert_eq!(
        parsed["error"]["retryable"], false,
        "expected error.retryable=false per FR-005/FR-006 for InvalidInput: {stderr}"
    );
}

// T-700: after_help_examples_present_for_all_commands [Issue #192 Phase 2.3]
#[test]
fn after_help_examples_present_for_all_commands() {
    for cmd in ["search", "index", "rebuild", "impact", "status", "brief"] {
        let output = yomu_cmd().args([cmd, "--help"]).output().unwrap();
        assert!(
            output.status.success(),
            "{cmd} --help failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        let (_, after_examples) = stdout
            .split_once("Examples:")
            .unwrap_or_else(|| panic!("{cmd} --help should contain 'Examples:', got: {stdout}"));
        let example_count = after_examples
            .lines()
            .filter(|l| l.trim_start().starts_with("yomu "))
            .count();
        assert!(
            example_count >= 1,
            "{cmd} --help should have at least 1 example line starting with 'yomu ', got: {stdout}"
        );
    }
}

// T-008: degraded_and_notes_present_in_all_five_json_routes [Issue #192 Phase 2.2b]
// FR-001: every JSON success envelope on index, rebuild, dry_run, status, and
// impact carries both `degraded` (boolean) and `notes` (array).
//
// Setup: indexed project with a single .tsx file plus an importer so impact has
// a target. `index` embeds via the test-support stub embedder, so every route
// produces its JSON envelope without a real model.
#[test]
fn degraded_and_notes_present_in_all_five_json_routes() {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(
        src.join("Button.tsx"),
        "export function Button() { return <button>Click</button>; }\n",
    )
    .unwrap();
    fs::write(
        src.join("App.tsx"),
        "import { Button } from './Button';\nexport function App() { return <Button/>; }\n",
    )
    .unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .unwrap();

    // Routes that produce JSON without external state. `index` / `rebuild`
    // embed via the test-support stub, so no real model or network is needed.
    let routes: &[(&str, &[&str])] = &[
        ("index --dry-run", &["--json", "index", "--dry-run"]),
        ("index", &["--json", "index"]),
        ("status", &["--json", "status"]),
        ("rebuild", &["--json", "rebuild"]),
        ("impact", &["--json", "impact", "src/Button.tsx"]),
    ];

    for (label, args) in routes {
        let output = yomu_cmd()
            .args(*args)
            .current_dir(dir.path())
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success(),
            "{label} should succeed, stderr: {stderr}, stdout: {stdout}"
        );
        let parsed: serde_json::Value = serde_json::from_str(&stdout)
            .unwrap_or_else(|e| panic!("{label} should emit JSON: {e}\n{stdout}"));
        assert!(
            parsed["degraded"].is_boolean(),
            "{label} must include `degraded` as boolean: {stdout}"
        );
        assert!(
            parsed["notes"].is_array(),
            "{label} must include `notes` as array: {stdout}"
        );
    }
}

// --- Issue #197: EmptyTarget candidates dynamic supply ---
//
// Spec: .claude/workspace/planning/2026-05-20-197-empty-target-candidates/spec.md

// T-004: impact_empty_target_json_envelope_includes_candidates_array
// FR-001, FR-005, NFR-003. Against a workspace with at least one indexed
// file, `yomu impact "" --json` must:
//   - exit non-zero with USAGE_ERROR (FR-001 routing via InvalidInput)
//   - emit `error.candidates` as a non-empty array of strings (FR-005 + NFR-003:
//     key present iff candidate list is non-empty)
#[test]
fn impact_empty_target_json_envelope_includes_candidates_array() {
    let dir = setup_project();
    let output = yomu_cmd()
        .args(["--json", "impact", ""])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "empty target must exit non-zero, stdout: {}, stderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    let parsed = parse_last_json_envelope(&stderr);
    assert_eq!(
        parsed["error"]["code"], "USAGE_ERROR",
        "FR-001: empty target routes through InvalidInput → USAGE_ERROR: {stderr}"
    );
    let candidates = parsed["error"]["candidates"]
        .as_array()
        .unwrap_or_else(|| panic!("FR-005: error.candidates must be an array: {stderr}"));
    assert!(
        !candidates.is_empty(),
        "FR-005 / NFR-003: candidates array must be non-empty when index has files: {stderr}"
    );
    for v in candidates {
        assert!(
            v.is_string(),
            "FR-005: every candidates entry must be a string, got: {v:?} in {stderr}"
        );
    }
}

fn setup_injection_e2e_project() -> TempDir {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src");
    let tests_dir = dir.path().join("tests");
    fs::create_dir_all(&src).unwrap();
    fs::create_dir_all(&tests_dir).unwrap();
    fs::write(
        src.join("lib.rs"),
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    )
    .unwrap();
    fs::write(
        tests_dir.join("foo_test.rs"),
        "#[test] fn it_runs() { assert_eq!(2 + 2, 4); }\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"injection_e2e\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .unwrap();
    let out = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    dir
}

/// Runs `yomu` with `args`, parses stdout as JSON, requires the named array
/// to be non-empty, and asserts that at least one entry satisfies `check`.
/// Returns the parsed JSON so callers can run additional top-level asserts.
fn run_json_array_check(
    dir: &TempDir,
    args: &[&str],
    array_key: &str,
    check: impl Fn(&serde_json::Value) -> bool,
    check_msg: &str,
) -> serde_json::Value {
    let output = yomu_cmd()
        .args(args)
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "yomu {args:?} failed: stdout={stdout}, stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("should parse as JSON: {e}\n{stdout}"));
    let entries = parsed[array_key]
        .as_array()
        .unwrap_or_else(|| panic!("{array_key} must be an array: {stdout}"));
    assert!(
        !entries.is_empty(),
        "{array_key} must be non-empty: {stdout}"
    );
    assert!(entries.iter().any(check), "{check_msg}: {stdout}");
    parsed
}

// T-212: index_populates_injection_flags_for_all_chunks
// Spec FR-214: After `yomu index`, every chunks row has non-NULL
// injection_flags (matcher走行 over every chunk).
#[test]
fn index_populates_injection_flags_for_all_chunks() {
    let dir = setup_injection_e2e_project();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let null_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE injection_flags IS NULL",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let total_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
        .unwrap();
    assert!(total_count > 0, "fixture must produce at least one chunk");
    assert_eq!(
        null_count, 0,
        "every chunk must have non-NULL injection_flags (got {null_count}/{total_count})"
    );
}

// T-213: index_source_kind_is_subset_of_src_and_test
// Spec FR-215: After `yomu index` on a repo with src + tests directories,
// DISTINCT source_kind is a subset of {"src", "test"} (walker excludes
// vendor via .gitignore).
#[test]
fn index_source_kind_is_subset_of_src_and_test() {
    let dir = setup_injection_e2e_project();
    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let mut stmt = conn
        .prepare("SELECT DISTINCT source_kind FROM chunks ORDER BY source_kind")
        .unwrap();
    let kinds: Vec<String> = stmt
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .filter_map(Result::ok)
        .collect();
    assert!(!kinds.is_empty(), "chunks must have source_kind values");
    for k in &kinds {
        assert!(
            k == "src" || k == "test",
            "source_kind must be 'src' or 'test' (walker excludes vendor), got: {k:?}"
        );
    }
    assert!(
        kinds.contains(&"src".to_owned()),
        "src fixture must produce src source_kind, got: {kinds:?}"
    );
}

// Helper for T-318: project with a parsable Rust file under src/ plus a
// parsable TypeScript file under vendor/. No .gitignore — vendor/ is reachable
// by the walker unless `--exclude-vendor` filters it out. NOT pre-indexed:
// each T-318 call site invokes `yomu index` (with or without the flag) itself.
fn setup_vendor_project_unindexed() -> TempDir {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src");
    let vendor = dir.path().join("vendor");
    fs::create_dir_all(&src).unwrap();
    fs::create_dir_all(&vendor).unwrap();
    fs::write(
        src.join("lib.rs"),
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    )
    .unwrap();
    fs::write(
        vendor.join("util.ts"),
        "export function util() { return 1; }\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"vendor_e2e\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .unwrap();
    dir
}

// T-318: index_exclude_vendor_drops_vendor_source_kind
// Spec FR-318 (also exercises FR-315a, FR-315b, FR-316a, FR-316b, FR-316c,
// FR-317a, FR-317b, FR-317c via the real CLI dispatch chain). Perspective:
// State + Hazard (silent-default-false avoidance for `--exclude-vendor`).
//
// Given a project with `vendor/util.ts` (no .gitignore excluding it), invoking
// `yomu index --exclude-vendor` SHALL leave 0 chunks with source_kind='vendor'
// in the DB. Regression: a parallel fresh project indexed WITHOUT the flag
// SHALL produce > 0 such chunks (so a flag that silently no-ops is caught).
#[test]
fn index_exclude_vendor_drops_vendor_source_kind() {
    // Arrange — Act (flag ON): fresh tempdir + `yomu index --exclude-vendor`
    let dir_excluded = setup_vendor_project_unindexed();
    let out = yomu_cmd()
        .args(["index", "--exclude-vendor"])
        .current_dir(dir_excluded.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "index --exclude-vendor failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Assert (flag ON): SQL count of source_kind='vendor' is 0
    let db_path = dir_excluded.path().join(".yomu").join("index.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let vendor_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE source_kind = 'vendor'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let total_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
        .unwrap();
    drop(conn);
    assert!(
        total_count > 0,
        "fixture must produce some chunks (src/lib.rs at minimum), got total={total_count}"
    );
    assert_eq!(
        vendor_count, 0,
        "FR-318: --exclude-vendor must drop vendor source_kind, got {vendor_count}/{total_count}"
    );

    // Arrange — Act (flag OFF, regression): separate fresh tempdir + `yomu index`
    let dir_baseline = setup_vendor_project_unindexed();
    let out = yomu_cmd()
        .arg("index")
        .current_dir(dir_baseline.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "baseline index failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Assert (flag OFF, regression): vendor count must be > 0 so the flag has
    // a non-vacuous effect (guards against the flag silently no-op'ing).
    let db_path = dir_baseline.path().join(".yomu").join("index.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let vendor_count_baseline: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE source_kind = 'vendor'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    drop(conn);
    assert!(
        vendor_count_baseline > 0,
        "regression: baseline `yomu index` (no flag) must produce vendor chunks for the \
         --exclude-vendor flag to be non-vacuous, got {vendor_count_baseline}"
    );
}

// Helper for T-209: a project whose test code lives in a Rust inline test
// module file `src/feature/tests.rs` (the `#[cfg(test)] mod tests;` split-out
// convention), alongside a plain `src/feature.rs`. NOT pre-indexed.
fn setup_inline_test_module_project() -> TempDir {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src");
    let feature = src.join("feature");
    fs::create_dir_all(&feature).unwrap();
    fs::write(src.join("lib.rs"), "pub mod feature;\n").unwrap();
    fs::write(
        src.join("feature.rs"),
        "pub fn sum(a: i32, b: i32) -> i32 { a + b }\n#[cfg(test)]\nmod tests;\n",
    )
    .unwrap();
    fs::write(
        feature.join("tests.rs"),
        "#[test]\nfn computes_sum() {\n    assert_eq!(super::sum(2, 3), 5);\n}\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"inline_test_e2e\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .unwrap();
    dir
}

// T-209: index_tags_rust_inline_test_module_as_test
// Spec FR-A1-1/FR-A1-4: after `yomu index`, chunks from a Rust inline test
// module file (`src/feature/tests.rs`) carry source_kind='test', while the
// sibling source file (`src/feature.rs`) stays 'src'. End-to-end guard that the
// classifier fix reaches the persisted column.
#[test]
fn index_tags_rust_inline_test_module_as_test() {
    let dir = setup_inline_test_module_project();
    let out = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let test_chunks: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE file_path LIKE '%feature/tests.rs' AND source_kind = 'test'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let mistagged: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE file_path LIKE '%feature/tests.rs' AND source_kind != 'test'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let parent_src: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE file_path LIKE '%/feature.rs' AND source_kind = 'src'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    drop(conn);

    assert!(
        test_chunks > 0,
        "inline test module src/feature/tests.rs must yield test chunks, got {test_chunks}"
    );
    assert_eq!(
        mistagged, 0,
        "no chunk from src/feature/tests.rs may carry a non-test source_kind, got {mistagged}"
    );
    assert!(
        parent_src > 0,
        "sibling source src/feature.rs must stay source_kind='src', got {parent_src}"
    );
}

// T-216: index_does_not_embed_test_chunks
// Spec FR-A2: after `yomu index`, test files are chunked/FTS-indexed but NOT
// embedded. No test chunk appears in embedded_chunk_ids, and the embedded count
// equals the embeddable (non-test, non-inner_fn) count so status is not degraded.
#[test]
fn index_does_not_embed_test_chunks() {
    let dir = setup_inline_test_module_project();
    let out = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let db_path = dir.path().join(".yomu").join("index.db");
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let embedded_test: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM embedded_chunk_ids e \
             JOIN chunks c ON e.chunk_id = c.id WHERE c.source_kind = 'test'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let embeddable_expected: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM chunks WHERE source_kind != 'test' AND chunk_type != 'inner_fn'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    let embedded_total: i64 = conn
        .query_row(
            "SELECT COUNT(DISTINCT chunk_id) FROM embedded_chunk_ids",
            [],
            |row| row.get(0),
        )
        .unwrap();
    drop(conn);

    assert!(
        embeddable_expected > 0,
        "fixture must have embeddable src chunks, got {embeddable_expected}"
    );
    assert_eq!(
        embedded_test, 0,
        "test chunks must not be embedded, got {embedded_test}"
    );
    assert_eq!(
        embedded_total, embeddable_expected,
        "embedded count must equal embeddable (non-test, non-inner_fn) count; \
         embedded={embedded_total} embeddable={embeddable_expected}"
    );
}

// Returns the distinct file_path values of a `yomu --json brief` invocation.
fn brief_files(dir: &Path, extra_args: &[&str]) -> Vec<String> {
    let mut args = vec![
        "--json",
        "brief",
        "feature sum helper",
        "--seed-file",
        "src/feature.rs",
    ];
    args.extend_from_slice(extra_args);
    let out = yomu_cmd().args(&args).current_dir(dir).output().unwrap();
    assert!(
        out.status.success(),
        "brief failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("brief stdout is not JSON: {e}; got: {stdout}"));
    let mut files: Vec<String> = parsed["chunks"]
        .as_array()
        .unwrap_or(&Vec::new())
        .iter()
        .filter_map(|c| c["file_path"].as_str().map(str::to_owned))
        .collect();
    files.sort();
    files.dedup();
    files
}

// T-217: brief_excludes_test_chunks_from_closure_by_default
// Spec FR-B: brief drops chunks whose source_kind is 'test' from the forward
// closure. Seeding on src/feature.rs (which declares `mod tests;`) must yield
// the src seed without its inline test module.
#[test]
fn brief_excludes_test_chunks_from_closure_by_default() {
    let dir = setup_inline_test_module_project();
    let idx = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        idx.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&idx.stderr)
    );

    let files = brief_files(dir.path(), &[]);
    assert!(
        files.iter().any(|f| f.ends_with("feature.rs")),
        "src seed src/feature.rs must be present, got: {files:?}"
    );
    assert!(
        !files.iter().any(|f| f.ends_with("feature/tests.rs")),
        "inline test module must be excluded by default, got: {files:?}"
    );
}

// T-218: brief_include_tests_flag_keeps_test_chunks
// Spec FR-B: `--include-tests` opts back into test code for test-fixing tasks.
#[test]
fn brief_include_tests_flag_keeps_test_chunks() {
    let dir = setup_inline_test_module_project();
    let idx = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        idx.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&idx.stderr)
    );

    let files = brief_files(dir.path(), &["--include-tests"]);
    assert!(
        files.iter().any(|f| f.ends_with("feature/tests.rs")),
        "--include-tests must keep the inline test module, got: {files:?}"
    );
}

// A test file reached by nothing (no `mod` declaration) and importing nothing
// resolvable: its bidirectional closure is the seed alone, all test-kind. This
// is the only shape that empties under the default test filter now that the
// closure is bidirectional — an inline `mod tests;` file always pulls its src
// parent in backward.
fn setup_orphan_test_seed_project() -> TempDir {
    let dir = tempdir().unwrap();
    let src = dir.path().join("src");
    fs::create_dir_all(src.join("widget")).unwrap();
    fs::write(src.join("lib.rs"), "pub fn noop() {}\n").unwrap();
    fs::write(
        src.join("widget/tests.rs"),
        "#[test]\nfn solo() {\n    assert!(true);\n}\n",
    )
    .unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        "[package]\nname = \"orphan_test_e2e\"\nversion = \"0.1.0\"\n",
    )
    .unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .unwrap();
    let out = yomu_cmd()
        .arg("index")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "index failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    dir
}

// T-616: brief_explicit_test_seed_keeps_seed_without_misattribution [#236]
// Seeding directly on a test file whose whole closure is test-kind, without
// --include-tests. The seed must survive (the caller named it), so the closure
// is non-empty and the CLI does not emit the degraded note — that note means
// "FTS-only seed selection" and would misreport the cause of an empty result.
#[test]
fn brief_explicit_test_seed_keeps_seed_without_misattribution() {
    let dir = setup_orphan_test_seed_project();
    let out = yomu_cmd()
        .args([
            "brief",
            "fix the failing solo assertion",
            "--seed-file",
            "src/widget/tests.rs",
        ])
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "brief failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("src/widget/tests.rs"),
        "explicit test seed must survive the default test filter, got: {stdout}"
    );
    assert!(
        !stdout.contains("FTS-only seed selection"),
        "a non-empty closure must not emit the FTS-fallback degraded note (#236), got: {stdout}"
    );
}

// T-319: search_json_emits_injection_check_and_per_chunk_flags
// Spec FR-319a (top-level `injection_check`) + FR-319b (per-chunk
// `injection_flags` when storage source is Some). Perspective: State +
// Equivalence (one representative result with populated injection_flags).
//
// Given a project indexed with PR#2 populate enabled (default), `yomu --json
// search "add"` SHALL emit a JSON object with `injection_check="ran"` at top
// level and at least one `results[]` entry carrying `injection_flags`.
#[test]
fn search_json_emits_injection_check_and_per_chunk_flags() {
    let dir = setup_injection_e2e_project();
    let parsed = run_json_array_check(
        &dir,
        &["--json", "search", "add"],
        "results",
        |r| r.get("injection_flags").is_some(),
        "FR-319b: at least one result chunk must include injection_flags field \
         (PR#2 populates every chunk)",
    );
    assert_eq!(
        parsed["injection_check"], "ran",
        "FR-319a: response must include top-level injection_check='ran': {parsed}"
    );
}

// T-320: brief_json_emits_injection_check_and_per_chunk_flags
// Spec FR-320a (top-level `injection_check`) + FR-320b (per-chunk
// `injection_flags` when storage source is Some). Perspective: State +
// Equivalence (seed-file chunk path is the obvious chunk source).
//
// Given a project indexed with PR#2 populate enabled, `yomu --json brief
// "task" --seed-file src/lib.rs` SHALL emit a JSON object with
// `injection_check="ran"` at top level and at least one `chunks[]` entry
// carrying `injection_flags`.
#[test]
fn brief_json_emits_injection_check_and_per_chunk_flags() {
    let dir = setup_injection_e2e_project();
    let parsed = run_json_array_check(
        &dir,
        &["--json", "brief", "task", "--seed-file", "src/lib.rs"],
        "chunks",
        |c| c.get("injection_flags").is_some(),
        "FR-320b: at least one brief chunk must include injection_flags field \
         (PR#2 populates every chunk)",
    );
    assert_eq!(
        parsed["injection_check"], "ran",
        "FR-320a: response must include top-level injection_check='ran': {parsed}"
    );
}

// T-576: search_json_emits_per_chunk_source_kind
// Spec FR-009a (per-chunk `source_kind` carried from chunks table to JSON
// envelope). Perspective: State + Equivalence (one representative result
// classified as "src" since the fixture is non-vendor / non-test).
//
// Given a project indexed with default config, `yomu --json search "add"`
// SHALL emit a JSON object whose `results[]` entries each carry
// `source_kind="src"` (the walker classifies fixture files under `src/`).
#[test]
fn search_json_emits_per_chunk_source_kind() {
    let dir = setup_injection_e2e_project();
    run_json_array_check(
        &dir,
        &["--json", "search", "add"],
        "results",
        |r| r["source_kind"].as_str() == Some("src"),
        "FR-009a: at least one result chunk must carry source_kind='src' \
         (default classification for non-vendor/non-test files)",
    );
}

// T-577: brief_json_emits_per_chunk_source_kind
// Spec FR-009a (per-chunk `source_kind` carried from chunks table to JSON
// envelope). Perspective: State + Equivalence (seed-file chunk path).
//
// Given a project indexed with default config, `yomu --json brief "task"
// --seed-file src/lib.rs` SHALL emit `chunks[]` entries each carrying
// `source_kind="src"` (the walker classifies the seed file under `src/`).
#[test]
fn brief_json_emits_per_chunk_source_kind() {
    let dir = setup_injection_e2e_project();
    run_json_array_check(
        &dir,
        &["--json", "brief", "task", "--seed-file", "src/lib.rs"],
        "chunks",
        |c| c["source_kind"].as_str() == Some("src"),
        "FR-009a: at least one brief chunk must carry source_kind='src'",
    );
}

// ── verify subcommand (T-402, T-403) ────────────────────────────────

// T-402: yomu --json verify emits structured precision/recall report
// Spec FR-406b / FR-407.
#[test]
fn yomu_json_verify_emits_precision_recall_schema() {
    let output = yomu_cmd()
        .arg("--json")
        .arg("verify")
        .output()
        .expect("yomu --json verify runs");
    assert!(
        output.status.success(),
        "yomu --json verify must exit 0, status={:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("verify JSON output parses");

    for field in [
        "precision",
        "recall",
        "tp",
        "fp",
        "fn_count",
        "details",
        "degraded",
        "injection_check",
    ] {
        assert!(
            parsed.get(field).is_some(),
            "FR-406b: JSON output must include top-level field `{field}`, got: {parsed}"
        );
    }
    assert_eq!(
        parsed["injection_check"], "ran",
        "FR-406b: injection_check must serialize to lowercase \"ran\", got: {parsed}"
    );
    // The bundled corpus must meet the gate at runtime as well as in the
    // dedicated verification test.
    let precision = parsed["precision"].as_f64().expect("precision is a number");
    let recall = parsed["recall"].as_f64().expect("recall is a number");
    assert!(
        precision >= 0.90,
        "BR-403 via CLI: precision must be >= 0.90, got {precision}: {parsed}"
    );
    assert!(
        recall >= 0.95,
        "BR-403 via CLI: recall must be >= 0.95, got {recall}: {parsed}"
    );
}

// T-403: yomu verify plain text emits the header line and a per-entry table
// Spec FR-406c / FR-407.
#[test]
fn yomu_verify_plain_text_emits_header_and_table() {
    let output = yomu_cmd().arg("verify").output().expect("yomu verify runs");
    assert!(
        output.status.success(),
        "yomu verify must exit 0, status={:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.starts_with("Verification: precision="),
        "FR-406c: plain output must start with `Verification: precision=`, got: {stdout}"
    );
    // Per-entry table header line is the second meaningful line.
    assert!(
        stdout.contains("\nid\tkind\tmatched\texpected\tactual\n"),
        "FR-406c: plain output must include the per-entry table header, got: {stdout}"
    );
}
