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

    // search (text-only, no GEMINI_API_KEY)
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
