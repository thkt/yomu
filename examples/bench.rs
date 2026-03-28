//! Benchmark pipeline for vercel/ai.
//!
//! Requires: /tmp/vercel-ai (git clone of vercel/ai).
//!
//! Usage:
//!   cargo run --release --example bench -- pipeline
//!   cargo run --release --example bench -- chunk
//!   cargo run --release --example bench -- embed
//!   cargo run --release --example bench -- explorer
//!   cargo run --release --example bench -- impact

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use clap::Parser;

const REPO_ROOT: &str = "/tmp/vercel-ai";

#[derive(Parser)]
#[command(name = "bench", about = "Benchmark pipeline for vercel/ai")]
struct Cli {
    #[command(subcommand)]
    stage: Stage,
}

#[derive(clap::Subcommand)]
enum Stage {
    /// All stages: chunk -> embed -> explorer -> impact
    Pipeline,
    /// Chunk-only index (fresh DB)
    Chunk,
    /// Incremental embedding
    Embed,
    /// Explorer search queries
    Explorer,
    /// Impact analysis
    Impact,
}

fn main() {
    let cli = Cli::parse();
    let root = PathBuf::from(REPO_ROOT);

    if !root.join(".git").exists() {
        eprintln!("ERROR: {REPO_ROOT} not found");
        std::process::exit(1);
    }

    match cli.stage {
        Stage::Pipeline => run_pipeline(&root),
        Stage::Chunk => {
            let conn = fresh_db(&root);
            run_chunk(&conn, &root);
        }
        Stage::Embed => {
            let conn = open_db(&root);
            require_chunks(&conn);
            let embedder = require_embedder();
            run_embed(&conn, &embedder);
        }
        Stage::Explorer => {
            let conn = open_db(&root);
            require_embeddings(&conn);
            let embedder = require_embedder();
            run_explorer(&conn, &embedder);
        }
        Stage::Impact => {
            let conn = open_db(&root);
            require_references(&conn);
            let c = conn.lock().unwrap();
            run_impact(&c);
        }
    }
}

fn run_pipeline(root: &Path) {
    let conn = fresh_db(root);
    run_chunk(&conn, root);

    if let Some(embedder) = load_embedder() {
        run_embed(&conn, &embedder);
        run_explorer(&conn, &embedder);
    } else {
        eprintln!("\n=== Embed/Explorer — SKIP (model unavailable) ===");
    }

    let c = conn.lock().unwrap();
    run_impact(&c);
}

fn db_path(root: &Path) -> PathBuf {
    root.join(".yomu").join("index.db")
}

fn fresh_db(root: &Path) -> Arc<Mutex<yomu::storage::Db>> {
    let path = db_path(root);
    let _ = std::fs::remove_file(&path);
    let conn = yomu::storage::open_db(&path).unwrap();
    Arc::new(Mutex::new(conn))
}

fn open_db(root: &Path) -> Arc<Mutex<yomu::storage::Db>> {
    let path = db_path(root);
    if !path.exists() {
        eprintln!("ERROR: no index at {path:?} — run `pipeline` or `chunk` first");
        std::process::exit(1);
    }
    let conn = yomu::storage::open_db(&path).unwrap();
    Arc::new(Mutex::new(conn))
}

fn load_embedder() -> Option<rurico::embed::Embedder> {
    let paths = rurico::embed::download_model().ok()?;
    rurico::embed::Embedder::new(&paths).ok()
}

fn require_chunks(conn: &Arc<Mutex<yomu::storage::Db>>) {
    let c = conn.lock().unwrap();
    let stats = yomu::storage::get_stats(&c).unwrap();
    if stats.total_chunks == 0 {
        eprintln!("ERROR: no chunks indexed; run 'chunk' or 'pipeline' first");
        std::process::exit(1);
    }
}

fn require_embeddings(conn: &Arc<Mutex<yomu::storage::Db>>) {
    let c = conn.lock().unwrap();
    let stats = yomu::storage::get_stats(&c).unwrap();
    if stats.embedded_chunks == 0 {
        eprintln!("ERROR: no embeddings; run 'embed' or 'pipeline' first");
        std::process::exit(1);
    }
}

fn require_references(conn: &Arc<Mutex<yomu::storage::Db>>) {
    let c = conn.lock().unwrap();
    let refs = yomu::storage::get_reference_count(&c).unwrap();
    if refs == 0 {
        eprintln!("ERROR: no references; run 'chunk' or 'pipeline' first");
        std::process::exit(1);
    }
}

fn require_embedder() -> rurico::embed::Embedder {
    load_embedder().unwrap_or_else(|| {
        eprintln!("ERROR: model unavailable");
        std::process::exit(1);
    })
}

fn run_chunk(conn: &Arc<Mutex<yomu::storage::Db>>, root: &Path) {
    let start = Instant::now();
    yomu::indexer::run_chunk_only_index(Arc::clone(conn), root).unwrap();
    let elapsed = start.elapsed();

    let c = conn.lock().unwrap();
    let stats = yomu::storage::get_stats(&c).unwrap();
    let refs = yomu::storage::get_reference_count(&c).unwrap();

    eprintln!("\n=== Chunk-only index ===");
    eprintln!("  Files:      {}", stats.total_files);
    eprintln!("  Chunks:     {}", stats.total_chunks);
    eprintln!(
        "  Embedded:   {}/{}",
        stats.embedded_chunks, stats.total_chunks
    );
    eprintln!("  References: {refs}");
    eprintln!("  Time:       {:.1}s", elapsed.as_secs_f64());
}

fn run_embed(conn: &Arc<Mutex<yomu::storage::Db>>, embedder: &rurico::embed::Embedder) {
    let start = Instant::now();
    let result = yomu::indexer::run_incremental_embed(Arc::clone(conn), embedder, 50, None);
    let elapsed = start.elapsed();

    let c = conn.lock().unwrap();
    let stats = yomu::storage::get_stats(&c).unwrap();

    eprintln!("\n=== Incremental embed (budget=50) ===");
    eprintln!("  Result:     {:?}", result.map(|_| "ok"));
    eprintln!(
        "  Embedded:   {}/{}",
        stats.embedded_chunks, stats.total_chunks
    );
    eprintln!("  Coverage:   {}%", stats.embed_percentage());
    eprintln!("  Time:       {:.1}s", elapsed.as_secs_f64());
}

fn run_explorer(conn: &Arc<Mutex<yomu::storage::Db>>, embedder: &rurico::embed::Embedder) {
    let queries = [
        "streaming chat hooks",
        "tool calling agent",
        "form validation logic",
    ];

    for query in &queries {
        let start = Instant::now();
        let results = yomu::query::search(Arc::clone(conn), embedder, query, 5, 0);
        let elapsed = start.elapsed();

        match results {
            Ok(outcome) => {
                eprintln!(
                    "\n=== explorer(\"{query}\") — {:.0}ms ===",
                    elapsed.as_millis()
                );
                for r in &outcome.results {
                    let name = r.chunk.name.as_deref().unwrap_or("(unnamed)");
                    let ctype = format!("{:?}", r.chunk.chunk_type).to_lowercase();
                    eprintln!(
                        "  → {name} [{ctype}] — {}:{} ({:.2})",
                        r.chunk.file_path, r.chunk.start_line, r.score
                    );
                }
            }
            Err(e) => eprintln!("  ERROR: {e}"),
        }
    }
}

fn run_impact(conn: &yomu::storage::Db) {
    let targets = [
        ("packages/ai/src/ui/ui-messages.ts", Some("UIMessage"), 2u32),
        ("packages/react/src/use-chat.ts", None, 1u32),
    ];

    for (file, symbol, depth) in &targets {
        let label = match symbol {
            Some(s) => format!("{file}:{s}"),
            None => file.to_string(),
        };
        eprintln!("\n=== impact(\"{label}\", depth={depth}) ===");

        if let Some(sym) = symbol {
            let sym_refs = yomu::storage::get_symbol_dependents(conn, file, sym).unwrap();
            eprintln!("  Direct symbol references: {} files", sym_refs.len());
            for r in sym_refs.iter().take(5) {
                eprintln!("    - {r}");
            }
            if sym_refs.len() > 5 {
                eprintln!("    ... {} more", sym_refs.len() - 5);
            }
        }

        let deps = yomu::storage::get_transitive_dependents(conn, file, *depth).unwrap();
        let mut by_depth: BTreeMap<u32, Vec<&str>> = BTreeMap::new();
        for d in &deps {
            by_depth.entry(d.depth).or_default().push(&d.file_path);
        }
        for (d, files) in &by_depth {
            eprintln!("  Depth {d} — {} files", files.len());
        }
        eprintln!("  Total: {} dependent(s)", deps.len());
    }
}
