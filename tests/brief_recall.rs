//! Gate1: seeded recall hard-block over the vendored ground-truth corpus
//! (#128 Phase 3, FR-007/008/009/010/017).
//!
//! For each pinned submodule the gate copies `src/` into a tempdir, indexes it
//! with the spawned `yomu` binary (built with the `test-support` stub embedder,
//! so no model is needed), runs seeded `brief` for every GT entry, and asserts
//! the mean seeded recall stays at or above the floor committed in
//! `fixtures/brief/baseline.json`. The seeded path is graph-only (import/impact
//! closure, no vector search), so the stub index suffices and the gate is
//! deterministic — empirically verified bit-identical across processes and
//! re-indexes (#128, NFR-001).
//!
//! Run with `--features test-support` so the spawned binary uses the stub
//! embedder. The gate carries no `#[ignore]` and runs in the default profile
//! (FR-008 / T-010).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use rusqlite::Connection;
use serde::Deserialize;
use tempfile::{TempDir, tempdir};

use yomu::brief::{ChunkInclusionReason, Seed, SeedKind, TaskBrief, expand_plan};
use yomu::recall::corpus::{GtEntry, canary_violations, load_bundled};
use yomu::recall::{gate_passes, measure};
use yomu::storage::open_db;

/// Production `brief` defaults (src/main.rs); recall is measured at these so the
/// floor reflects the values agents actually run with.
const DEPTH: u32 = 3;
const MAX_CHUNKS: u32 = 80;
const MAX_BYTES: u32 = 80_000;

/// The committed Gate1 baseline (`fixtures/brief/baseline.json`).
#[derive(Debug, Deserialize)]
struct Baseline {
    /// Minimum acceptable mean seeded recall (FR-010). The gate fails below it.
    seeded_floor: f64,
    /// Embedder identity behind the floor measurement (stub for the seeded gate).
    model_hash: String,
}

fn load_baseline() -> Baseline {
    let text = include_str!("fixtures/brief/baseline.json");
    serde_json::from_str(text).expect("baseline.json parses")
}

fn vendor_dir(repo: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("vendor")
        .join(repo)
}

/// Recursively copies a directory tree (regular files and subdirectories).
fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).expect("create dest dir");
    for entry in fs::read_dir(from).expect("read source dir") {
        let entry = entry.expect("dir entry");
        let kind = entry.file_type().expect("file type");
        let dst = to.join(entry.file_name());
        if kind.is_dir() {
            copy_tree(&entry.path(), &dst);
        } else if kind.is_file() {
            fs::copy(entry.path(), &dst).expect("copy file");
        }
    }
}

/// Copies `vendor/<repo>/src` into a fresh tempdir, indexes it with the spawned
/// stub-embedder binary, and opens the resulting db. GT seeds are crate-relative
/// (`src/...`), so only `src/` is copied; a minimal Cargo.toml gives the indexer
/// a project root (mirrors the cli_integration setup). The vendored checkout is
/// never written to (BR-005, hermetic).
fn index_repo_dir(repo: &str) -> TempDir {
    let dir = tempdir().expect("tempdir");
    copy_tree(&vendor_dir(repo).join("src"), &dir.path().join("src"));
    fs::write(
        dir.path().join("Cargo.toml"),
        format!("[package]\nname = \"{repo}_gt\"\nversion = \"0.0.0\"\n"),
    )
    .expect("write Cargo.toml");
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .expect("git init");
    let out = Command::new(env!("CARGO_BIN_EXE_yomu"))
        .arg("index")
        .current_dir(dir.path())
        .output()
        .expect("spawn yomu index");
    assert!(
        out.status.success(),
        "index {repo} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    dir
}

fn index_repo(repo: &str) -> (TempDir, Connection) {
    let dir = index_repo_dir(repo);
    let conn = open_db(&dir.path().join(".yomu/index.db")).expect("open indexed db");
    (dir, conn)
}

fn task_for(
    entry: &GtEntry,
    depth: u32,
    max_chunks: u32,
    max_bytes: u32,
    include_tests: bool,
) -> TaskBrief {
    TaskBrief {
        task: entry.task.clone(),
        seeds: entry
            .seed
            .iter()
            .map(|p| Seed {
                kind: SeedKind::File,
                value: p.clone(),
            })
            .collect(),
        depth,
        max_chunks,
        max_bytes,
        include_tests,
    }
}

/// Seeded recall for one entry at the production brief defaults.
fn entry_recall(conn: &Connection, entry: &GtEntry) -> f64 {
    let task = task_for(entry, DEPTH, MAX_CHUNKS, MAX_BYTES, false);
    let out = expand_plan(conn, &task).expect("expand_plan");
    let output: HashSet<String> = out.chunks.iter().map(|c| c.file_path.clone()).collect();
    let reachable: HashSet<String> = out.reachable_files.iter().cloned().collect();
    measure(&entry.must_include, &output, &reachable).recall
}

/// The seed's 1-hop forward closure: files reasoned Seed or Forward by a depth-1
/// brief with the cap maxed out, so a truncating cap cannot hide a forward
/// member (toothless canary). Mirrors `collect_closure(seed, 1).forward_paths`,
/// the BR-002 canary's reachability source.
fn forward_1hop(conn: &Connection, entry: &GtEntry) -> HashSet<String> {
    let task = task_for(entry, 1, u32::MAX, u32::MAX, true);
    let out = expand_plan(conn, &task).expect("expand_plan depth=1");
    out.chunks
        .iter()
        .filter(|c| {
            matches!(
                c.included_reason,
                ChunkInclusionReason::Seed | ChunkInclusionReason::Forward(_)
            )
        })
        .map(|c| c.file_path.clone())
        .collect()
}

/// Setup failure when a vendored submodule is missing or off its pinned rev.
#[derive(Debug)]
struct SetupError {
    repo: String,
    expected_rev: String,
    detail: String,
}

impl fmt::Display for SetupError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "submodule {} setup error (expected rev {}): {}",
            self.repo, self.expected_rev, self.detail
        )
    }
}

/// FR-017: fail when a submodule is absent or its HEAD differs from the pinned
/// rev. Pure over the observed HEAD so both paths are deterministically testable
/// (T-011).
fn check_rev(repo: &str, expected_rev: &str, actual_head: Option<&str>) -> Result<(), SetupError> {
    match actual_head {
        None => Err(SetupError {
            repo: repo.to_owned(),
            expected_rev: expected_rev.to_owned(),
            detail: "submodule absent or HEAD unreadable".to_owned(),
        }),
        Some(head) if head == expected_rev => Ok(()),
        Some(head) => Err(SetupError {
            repo: repo.to_owned(),
            expected_rev: expected_rev.to_owned(),
            detail: format!("HEAD is {head}"),
        }),
    }
}

/// Reads `vendor/<repo>` HEAD, or None when the submodule is absent/unreadable.
fn read_head(repo: &str) -> Option<String> {
    let out = Command::new("git")
        .args([
            "-C",
            vendor_dir(repo).to_str().expect("utf8 path"),
            "rev-parse",
            "HEAD",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

/// Groups corpus entries by repo (sorted), asserting one pinned rev per repo.
fn entries_by_repo(entries: Vec<GtEntry>) -> BTreeMap<String, (String, Vec<GtEntry>)> {
    let mut by_repo: BTreeMap<String, (String, Vec<GtEntry>)> = BTreeMap::new();
    for entry in entries {
        let slot = by_repo
            .entry(entry.repo.clone())
            .or_insert_with(|| (entry.rev.clone(), Vec::new()));
        assert_eq!(
            slot.0, entry.rev,
            "all GT entries for repo {} must share one pinned rev",
            entry.repo
        );
        slot.1.push(entry);
    }
    by_repo
}

// T-008 / T-006b / T-017 / NFR-001 + FR-009 wiring. The seeded recall over the
// vendored corpus must meet the committed floor, the corpus must pass the BR-002
// canary through the real index, recall must be deterministic, and baseline.json
// must record a floor consistent with this run. One test shares the index across
// these checks because indexing dominates the cost (NFR-002); nextest runs each
// test in its own process, so a shared fixture is not available.
#[test]
fn gate1_seeded_recall_meets_floor() {
    let baseline = load_baseline();
    let by_repo = entries_by_repo(load_bundled().expect("bundled corpus").entries);

    let mut recalls: Vec<f64> = Vec::new();
    let mut forward: HashMap<String, HashSet<String>> = HashMap::new();
    let mut all_entries: Vec<GtEntry> = Vec::new();

    for (repo, (expected_rev, entries)) in &by_repo {
        check_rev(repo, expected_rev, read_head(repo).as_deref()).unwrap_or_else(|e| panic!("{e}")); // FR-017
        let (_dir, conn) = index_repo(repo);
        for entry in entries {
            let recall = entry_recall(&conn, entry);
            // NFR-001: same input, same recall (re-measure on the same index).
            assert_eq!(
                recall,
                entry_recall(&conn, entry),
                "recall for {} must be deterministic",
                entry.id
            );
            assert!(
                (0.0..=1.0).contains(&recall),
                "recall for {} must be in [0,1], got {recall}",
                entry.id
            );
            eprintln!("RECALL {} = {recall:.4}", entry.id);
            recalls.push(recall);
            forward.insert(entry.id.clone(), forward_1hop(&conn, entry));
            all_entries.push(entry.clone());
        }
    }

    // T-006b: every GT entry has a must-include outside its 1-hop forward closure,
    // verified through the real index, so recall is not vacuously high.
    let violations = canary_violations(&all_entries, &forward);
    assert!(
        violations.is_empty(),
        "BR-002 canary flagged tautological entries: {violations:?}"
    );

    // T-008: mean seeded recall is computed over the full corpus.
    assert!(!recalls.is_empty(), "corpus produced no measurements");
    let mean = recalls.iter().sum::<f64>() / recalls.len() as f64;
    eprintln!(
        "MEAN_SEEDED_RECALL = {mean:.6} over {} entries (floor {:.6})",
        recalls.len(),
        baseline.seeded_floor
    );

    // T-017: the committed floor is a valid recall and not above this run's mean
    // (it was measured from a run like this one). model_hash is recorded.
    assert!(
        (0.0..=1.0).contains(&baseline.seeded_floor),
        "seeded_floor must be in [0,1], got {}",
        baseline.seeded_floor
    );
    assert!(
        !baseline.model_hash.is_empty(),
        "baseline must record the embedder identity"
    );

    // FR-009: the gate fails when mean seeded recall drops below the floor.
    assert!(
        gate_passes(mean, baseline.seeded_floor),
        "mean seeded recall {mean:.6} is below the committed floor {:.6}",
        baseline.seeded_floor
    );
}

// T-011: FR-017 setup check rejects a wrong rev and an absent submodule, and
// accepts a matching HEAD. Pure, no submodule required.
#[test]
fn check_rev_rejects_mismatch_and_absence() {
    let rurico_rev = "1d6650a86ffd5c377fd4b50e171edfc89e39c3f0";
    let err = check_rev(
        "rurico",
        rurico_rev,
        Some("deadbeefdeadbeefdeadbeefdeadbeefdeadbeef"),
    )
    .expect_err("a mismatched HEAD must fail");
    assert_eq!(err.repo, "rurico");
    assert_eq!(err.expected_rev, rurico_rev);
    let msg = err.to_string();
    assert!(
        msg.contains("rurico") && msg.contains(rurico_rev),
        "error message names repo and expected rev, got: {msg}"
    );

    assert!(
        check_rev("amici", "ae8c0682c4c5f663644f4cd9f9cbcf6b392904da", None).is_err(),
        "an absent submodule must fail"
    );
    assert!(
        check_rev("rurico", rurico_rev, Some(rurico_rev)).is_ok(),
        "a HEAD matching the pinned rev must pass"
    );
}
