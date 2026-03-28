use std::process::Command;

fn yomu_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_yomu"))
}

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

#[test]
fn search_limit_out_of_range() {
    let output = yomu_cmd()
        .args(["search", "test", "--limit", "0"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("0") || stderr.contains("invalid"),
        "should reject limit=0: {stderr}"
    );
}

#[test]
fn search_limit_too_large() {
    let output = yomu_cmd()
        .args(["search", "test", "--limit", "999"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("999") || stderr.contains("invalid"),
        "should reject limit=999: {stderr}"
    );
}

#[test]
fn search_offset_too_large() {
    let output = yomu_cmd()
        .args(["search", "test", "--offset", "999"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn impact_depth_too_large() {
    let output = yomu_cmd()
        .args(["impact", "src/foo.ts", "--depth", "99"])
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn search_default_limit_accepted() {
    // Default limit=10 should be valid (command may fail for other reasons like no project root)
    let output = yomu_cmd().args(["search", "test"]).output().unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should NOT fail due to argument validation
    assert!(
        !stderr.contains("invalid value"),
        "default limit should be accepted: {stderr}"
    );
}

#[test]
fn unknown_subcommand_fails() {
    let output = yomu_cmd().arg("foobar").output().unwrap();
    assert!(!output.status.success());
}

// --- Success-path integration tests ---

fn setup_project() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(
        src.join("Button.tsx"),
        "export function Button() { return <button>Click</button>; }\n",
    )
    .unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
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

#[test]
fn index_then_status_then_search() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(
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

    // search (text-only fallback via probe — no YOMU_EMBED override)
    let output = yomu_cmd()
        .args(["search", "button"])
        .current_dir(dir.path())
        .env_remove("GEMINI_API_KEY")
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

#[test]
fn search_stdin_query() {
    let dir = setup_project();
    let mut child = yomu_cmd()
        .args(["search", "-"])
        .current_dir(dir.path())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
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

#[test]
fn search_format_json() {
    let dir = setup_project();
    let output = yomu_cmd()
        .args(["search", "button", "--format", "json"])
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

#[test]
fn index_dry_run() {
    let dir = tempfile::tempdir().unwrap();
    let src = dir.path().join("src");
    std::fs::create_dir_all(&src).unwrap();
    std::fs::write(src.join("App.tsx"), "export function App() {}").unwrap();
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
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

#[test]
fn search_stdin_empty_query_fails() {
    let dir = setup_project();
    let mut child = yomu_cmd()
        .args(["search", "-"])
        .current_dir(dir.path())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
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

#[test]
fn shorthand_typo_not_treated_as_search() {
    let output = yomu_cmd().arg("stauts").output().unwrap();
    assert!(
        !output.status.success(),
        "typo 'stauts' should fail, not become a search"
    );
}

#[test]
fn shorthand_near_command_name_still_searches() {
    // "state" is OSA distance 2 from "status" — should be treated as search, not typo
    let dir = setup_project();
    let output = yomu_cmd()
        .arg("state")
        .current_dir(dir.path())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "'state' should be a search query, not a typo: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn probe_embed_reaches_probe_path_without_subcommand() {
    // --probe-embed with an invalid dir should exit 10 (PROBE_EXIT_UNAVAILABLE),
    // NOT exit 2 ("requires a subcommand"). This proves the CLI routing fix works.
    let output = yomu_cmd()
        .args(["--probe-embed", "/nonexistent/model/dir"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(10),
        "should exit 10 (probe unavailable), not 2 (missing subcommand): status={:?} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}
