//! Benchmark test: index → explorer flow against kagami.
//!
//! Requires: ~/GitHub/kagami to exist.
//! Usage: cargo test --release --test benchmark_kagami -- --nocapture

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn kagami_root() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME not set");
    PathBuf::from(home).join("GitHub/kagami")
}

fn skip_if_no_repo() -> bool {
    !kagami_root().join(".git").exists()
}

#[tokio::test]
async fn step1_index() {
    if skip_if_no_repo() {
        eprintln!("SKIP: ~/GitHub/kagami not found");
        return;
    }

    let root = kagami_root();
    // Clean DB + WAL/SHM for fresh start
    let yomu_dir = root.join(".yomu");
    for ext in ["index.db", "index.db-wal", "index.db-shm"] {
        let _ = std::fs::remove_file(yomu_dir.join(ext));
    }

    let db_path = yomu_dir.join("index.db");
    let conn = yomu::storage::open_db(&db_path).unwrap();
    let conn = Arc::new(parking_lot::Mutex::new(conn));

    let start = Instant::now();
    yomu::indexer::run_chunk_only_index(Arc::clone(&conn), &root)
        .await
        .unwrap();
    let elapsed = start.elapsed();

    let (stats, ref_count) = {
        let c = conn.lock();
        (
            yomu::storage::get_stats(&c).unwrap(),
            yomu::storage::get_reference_count(&c).unwrap(),
        )
    };

    eprintln!();
    eprintln!("=== index (chunk-only) ===");
    eprintln!("  Files:      {}", stats.total_files);
    eprintln!("  Chunks:     {}", stats.total_chunks);
    eprintln!("  Embedded:   {}/{}", stats.embedded_chunks, stats.total_chunks);
    eprintln!("  References: {}", ref_count);
    eprintln!("  Time:       {:.1}s", elapsed.as_secs_f64());
    eprintln!("  API calls:  0");
    eprintln!();
}

#[tokio::test]
async fn step2_explorer() {
    if skip_if_no_repo() {
        eprintln!("SKIP: ~/GitHub/kagami not found");
        return;
    }

    let root = kagami_root();
    let db_path = root.join(".yomu/index.db");
    if !db_path.exists() {
        eprintln!("SKIP: run step1_index first");
        return;
    }

    let http = reqwest::Client::new();
    let embedder = match yomu::indexer::embedder::Embedder::from_env(http) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("SKIP: GEMINI_API_KEY not set");
            return;
        }
    };

    let conn = yomu::storage::open_db(&db_path).unwrap();
    let conn = Arc::new(parking_lot::Mutex::new(conn));

    // Incremental embed (simulates what explorer does internally)
    let embed_start = Instant::now();
    let _ = yomu::indexer::run_incremental_embed(
        Arc::clone(&conn),
        &embedder,
        50,
        None,
    )
    .await;
    let embed_time = embed_start.elapsed();

    let stats = {
        let c = conn.lock();
        yomu::storage::get_stats(&c).unwrap()
    };

    eprintln!();
    eprintln!("=== explorer (incremental embed) ===");
    eprintln!("  Embedded:   {}/{}", stats.embedded_chunks, stats.total_chunks);
    eprintln!("  Coverage:   {}%", stats.embed_percentage());
    eprintln!("  Embed time: {:.1}s", embed_time.as_secs_f64());

    // Search queries
    let queries = [
        "data fetching with loading state",
        "chart visualization component",
        "number formatting helpers",
        "user registration page",
    ];

    for query in &queries {
        let start = Instant::now();
        let results = yomu::query::search(
            Arc::clone(&conn),
            &embedder,
            query,
            5,
            0,
        )
        .await;
        let search_time = start.elapsed();

        match results {
            Ok(results) => {
                eprintln!();
                eprintln!("  explorer(\"{query}\") — {:.0}ms", search_time.as_millis());
                for r in &results {
                    let name = r.chunk.name.as_deref().unwrap_or("(unnamed)");
                    let ctype = format!("{:?}", r.chunk.chunk_type).to_lowercase();
                    eprintln!(
                        "    → {} [{}] — {}:{} ({:.2})",
                        name, ctype, r.chunk.file_path, r.chunk.start_line, r.score
                    );
                }
                if results.is_empty() {
                    eprintln!("    (no results)");
                }
            }
            Err(e) => eprintln!("    ERROR: {e}"),
        }
    }
    eprintln!();
}
