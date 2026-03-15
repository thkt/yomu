use std::process::ExitCode;

use clap::{Parser, Subcommand};
use yomu::tools::{MAX_IMPACT_DEPTH, MAX_SEARCH_LIMIT, MAX_SEARCH_OFFSET, Yomu, YomuError};

#[derive(Parser)]
#[command(name = "yomu", version, about = "Frontend code search for AI agents")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Semantic code search. Finds components, hooks, types by meaning.
    Search {
        /// Natural language query
        query: String,
        /// Maximum results (default: 10)
        #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..=MAX_SEARCH_LIMIT as i64))]
        limit: u32,
        /// Skip N results (default: 0)
        #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=MAX_SEARCH_OFFSET as i64))]
        offset: u32,
    },
    /// Update chunk index incrementally. No API calls.
    Index,
    /// Rebuild chunk index from scratch. No API calls.
    Rebuild,
    /// Analyze impact of changes to a file or symbol.
    Impact {
        /// File path relative to project root (e.g. "src/hooks/useAuth.ts")
        target: String,
        /// Filter to specific symbol (e.g. "useAuth")
        #[arg(long)]
        symbol: Option<String>,
        /// Max traversal depth (default: 3)
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(0..=MAX_IMPACT_DEPTH as i64))]
        depth: u32,
    },
    /// Show index statistics.
    Status,
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("yomu=warn")),
        )
        .init();

    let cli = Cli::parse();

    let yomu = match Yomu::new() {
        Ok(y) => y,
        Err(e) => {
            eprintln!("error: {e}");
            return exit_code_for(&e);
        }
    };

    let result = match cli.command {
        Command::Search {
            query,
            limit,
            offset,
        } => yomu.search(&query, limit, offset).await,
        Command::Index => yomu.index().await,
        Command::Rebuild => yomu.rebuild().await,
        Command::Impact {
            target,
            symbol,
            depth,
        } => yomu.impact(&target, symbol.as_deref(), depth).await,
        Command::Status => yomu.status().await,
    };

    match result {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            exit_code_for(&e)
        }
    }
}

fn exit_code_for(e: &YomuError) -> ExitCode {
    match e {
        YomuError::InvalidInput(_) => ExitCode::from(2),
        YomuError::Network(_) => ExitCode::from(3),
        YomuError::Internal(_) => ExitCode::from(4),
        YomuError::Storage(_) | YomuError::Io(_) | YomuError::Index(_) | YomuError::Query(_) => {
            ExitCode::FAILURE
        }
    }
}
