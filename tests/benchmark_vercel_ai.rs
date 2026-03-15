//! Benchmark test: run chunk-only index + explorer against vercel/ai.
//!
//! Requires: /tmp/vercel-ai to exist (git clone of vercel/ai).
//! Skips gracefully when not present.
//!
//! Usage: cargo test --release --test benchmark_vercel_ai -- --nocapture

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

fn vercel_ai_root() -> PathBuf {
    PathBuf::from("/tmp/vercel-ai")
}

fn skip_if_no_repo() -> bool {
    !vercel_ai_root().join(".git").exists()
}

#[tokio::test]
async fn benchmark_chunk_only_index() {
    if skip_if_no_repo() {
        eprintln!("SKIP: /tmp/vercel-ai not found");
        return;
    }

    let root = vercel_ai_root();
    let db_dir = root.join(".yomu");
    let _ = std::fs::remove_file(db_dir.join("index.db"));

    let db_path = db_dir.join("index.db");
    let conn = yomu::storage::open_db(&db_path).unwrap();
    let conn = Arc::new(std::sync::Mutex::new(conn));

    let start = Instant::now();
    yomu::indexer::run_chunk_only_index(Arc::clone(&conn), &root)
        .await
        .unwrap();
    let chunk_time = start.elapsed();

    let stats = {
        let c = conn.lock().unwrap();
        yomu::storage::get_stats(&c).unwrap()
    };
    let ref_count = {
        let c = conn.lock().unwrap();
        yomu::storage::get_reference_count(&c).unwrap()
    };

    eprintln!();
    eprintln!("=== Chunk-only index benchmark ===");
    eprintln!("  Files:      {}", stats.total_files);
    eprintln!("  Chunks:     {}", stats.total_chunks);
    eprintln!(
        "  Embedded:   {}/{}",
        stats.embedded_chunks, stats.total_chunks
    );
    eprintln!("  References: {}", ref_count);
    eprintln!("  Time:       {:.1}s", chunk_time.as_secs_f64());
    eprintln!();
}

#[tokio::test]
async fn benchmark_incremental_embed() {
    if skip_if_no_repo() {
        eprintln!("SKIP: /tmp/vercel-ai not found");
        return;
    }

    let root = vercel_ai_root();
    let db_path = root.join(".yomu").join("index.db");
    if !db_path.exists() {
        eprintln!("SKIP: run benchmark_chunk_only_index first");
        return;
    }

    let conn = yomu::storage::open_db(&db_path).unwrap();
    let conn = Arc::new(std::sync::Mutex::new(conn));

    let http = reqwest::Client::new();
    let embedder = match yomu::indexer::embedder::Embedder::from_env(http) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("SKIP: GEMINI_API_KEY not set");
            return;
        }
    };

    let start = Instant::now();
    let result = yomu::indexer::run_incremental_embed(Arc::clone(&conn), &embedder, 50, None).await;
    let embed_time = start.elapsed();

    let stats = {
        let c = conn.lock().unwrap();
        yomu::storage::get_stats(&c).unwrap()
    };

    eprintln!();
    eprintln!("=== Incremental embed benchmark (budget=50) ===");
    eprintln!("  Result:     {:?}", result.map(|_| "ok"));
    eprintln!(
        "  Embedded:   {}/{}",
        stats.embedded_chunks, stats.total_chunks
    );
    eprintln!("  Coverage:   {}%", stats.embed_percentage());
    eprintln!("  Time:       {:.1}s", embed_time.as_secs_f64());
    eprintln!();
}

#[tokio::test]
async fn benchmark_explorer_query() {
    if skip_if_no_repo() {
        eprintln!("SKIP: /tmp/vercel-ai not found");
        return;
    }

    let root = vercel_ai_root();
    let db_path = root.join(".yomu").join("index.db");
    if !db_path.exists() {
        eprintln!("SKIP: run benchmark_chunk_only_index first");
        return;
    }

    let conn = match yomu::storage::open_db(&db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: DB busy or error: {e}");
            return;
        }
    };
    let stats = yomu::storage::get_stats(&conn).unwrap();
    if stats.embedded_chunks == 0 {
        eprintln!("SKIP: no embeddings, run benchmark_incremental_embed first");
        return;
    }
    drop(conn);

    let conn2 = match yomu::storage::open_db(&db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: DB busy or error: {e}");
            return;
        }
    };
    let conn2 = Arc::new(std::sync::Mutex::new(conn2));

    let http = reqwest::Client::new();
    let embedder = match yomu::indexer::embedder::Embedder::from_env(http) {
        Ok(e) => e,
        Err(_) => {
            eprintln!("SKIP: GEMINI_API_KEY not set");
            return;
        }
    };

    let queries = [
        "streaming chat hooks",
        "tool calling agent",
        "form validation logic",
    ];

    for query in &queries {
        let start = Instant::now();
        let results = yomu::query::search(Arc::clone(&conn2), &embedder, query, 5, 0).await;
        let search_time = start.elapsed();

        match results {
            Ok(outcome) => {
                eprintln!();
                eprintln!(
                    "=== explorer(\"{query}\") — {:.0}ms ===",
                    search_time.as_millis()
                );
                for r in &outcome.results {
                    let name = r.chunk.name.as_deref().unwrap_or("(unnamed)");
                    let ctype = format!("{:?}", r.chunk.chunk_type).to_lowercase();
                    eprintln!(
                        "  → {} [{}] — {}:{} ({:.2})",
                        name, ctype, r.chunk.file_path, r.chunk.start_line, r.score
                    );
                }
            }
            Err(e) => {
                eprintln!("  ERROR: {e}");
            }
        }
    }
    eprintln!();
}

#[tokio::test]
async fn benchmark_impact_query() {
    if skip_if_no_repo() {
        eprintln!("SKIP: /tmp/vercel-ai not found");
        return;
    }

    let root = vercel_ai_root();
    let db_path = root.join(".yomu").join("index.db");
    if !db_path.exists() {
        eprintln!("SKIP: run benchmark_chunk_only_index first");
        return;
    }

    let conn = match yomu::storage::open_db(&db_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: DB busy or error: {e}");
            return;
        }
    };
    let ref_count = match yomu::storage::get_reference_count(&conn) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: DB busy or error: {e}");
            return;
        }
    };
    if ref_count == 0 {
        eprintln!("SKIP: no references");
        return;
    }

    let targets = [
        ("packages/ai/src/ui/ui-messages.ts", Some("UIMessage"), 2u32),
        ("packages/react/src/use-chat.ts", None, 1u32),
    ];

    for (file, symbol, depth) in &targets {
        let label = match symbol {
            Some(s) => format!("{file}:{s}"),
            None => file.to_string(),
        };

        eprintln!();
        eprintln!("=== impact(\"{label}\", depth={depth}) ===");

        if let Some(sym) = symbol {
            let refs = yomu::storage::get_symbol_dependents(&conn, file, sym).unwrap();
            eprintln!("  Direct symbol references: {} files", refs.len());
            for r in refs.iter().take(5) {
                eprintln!("    - {r}");
            }
            if refs.len() > 5 {
                eprintln!("    ... {} more", refs.len() - 5);
            }
        }

        let dependents = yomu::storage::get_transitive_dependents(&conn, file, *depth).unwrap();
        // Group by depth
        let mut by_depth: std::collections::BTreeMap<u32, Vec<&str>> =
            std::collections::BTreeMap::new();
        for d in &dependents {
            by_depth.entry(d.depth).or_default().push(&d.file_path);
        }
        for (d, files) in &by_depth {
            eprintln!("  Depth {d} — {} files", files.len());
        }
        eprintln!("  Total: {} dependent file(s)", dependents.len());
    }
    eprintln!();
}
