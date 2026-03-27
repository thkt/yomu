use std::io::{IsTerminal, Read};
use std::process::ExitCode;

use clap::{CommandFactory, Parser, Subcommand};
use yomu::tools::{
    MAX_IMPACT_DEPTH, MAX_SEARCH_LIMIT, MAX_SEARCH_OFFSET, OutputFormat, Yomu, YomuError,
    probe_embedder,
};

#[derive(Parser)]
#[command(name = "yomu", version, about = "Frontend code search for AI agents")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Probe whether the embedding model can load safely (hidden, internal use)
    #[arg(long, hide = true)]
    probe_embed: Option<String>,
}

#[derive(Subcommand)]
enum Command {
    /// Semantic code search. Finds components, hooks, types by meaning.
    Search {
        /// Natural language query (reads from stdin if omitted or "-")
        query: Option<String>,
        /// Maximum results (default: 10)
        #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u32).range(1..=MAX_SEARCH_LIMIT as i64))]
        limit: u32,
        /// Skip N results (default: 0)
        #[arg(long, default_value_t = 0, value_parser = clap::value_parser!(u32).range(0..=MAX_SEARCH_OFFSET as i64))]
        offset: u32,
        /// Output format: text (default) or json
        #[arg(long, default_value = "text")]
        format: OutputFormat,
    },
    /// Update chunk index incrementally. No API calls.
    Index {
        /// Show what would be indexed without writing to the database
        #[arg(long)]
        dry_run: bool,
    },
    /// Rebuild chunk index from scratch. No API calls.
    Rebuild {
        /// Show what would be rebuilt without writing to the database
        #[arg(long)]
        dry_run: bool,
    },
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

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("yomu=warn")),
        )
        .init();

    let cli = Cli::parse();

    if let Some(model_dir) = cli.probe_embed {
        return probe_embedder(&model_dir);
    }

    let yomu = match Yomu::new() {
        Ok(y) => y,
        Err(e) => {
            eprintln!("error: {e}");
            return exit_code_for(&e);
        }
    };

    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            Cli::command()
                .error(
                    clap::error::ErrorKind::MissingSubcommand,
                    "requires a subcommand",
                )
                .exit();
        }
    };

    let result = match command {
        Command::Search {
            query,
            limit,
            offset,
            format,
        } => {
            let query = match resolve_query(query) {
                Ok(q) => q,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            yomu.search(&query, limit, offset, format)
        }
        Command::Index { dry_run } => {
            if dry_run {
                yomu.dry_run_index(false)
            } else {
                yomu.index()
            }
        }
        Command::Rebuild { dry_run } => {
            if dry_run {
                yomu.dry_run_index(true)
            } else {
                yomu.rebuild()
            }
        }
        Command::Impact {
            target,
            symbol,
            depth,
        } => yomu.impact(&target, symbol.as_deref(), depth),
        Command::Status => yomu.status(),
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

fn resolve_query(arg: Option<String>) -> Result<String, String> {
    let stdin = std::io::stdin();
    let is_terminal = stdin.is_terminal();
    resolve_query_with(arg, &mut stdin.lock(), is_terminal)
}

fn resolve_query_with(
    arg: Option<String>,
    stdin: &mut impl Read,
    stdin_is_terminal: bool,
) -> Result<String, String> {
    match arg {
        Some(q) if q != "-" => Ok(q),
        _ => {
            if stdin_is_terminal {
                return Err("query required: pass as argument or pipe via stdin".into());
            }
            let mut buf = String::new();
            stdin
                .read_to_string(&mut buf)
                .map_err(|e| format!("failed to read from stdin: {e}"))?;
            let trimmed = buf.trim().to_string();
            if trimmed.is_empty() {
                return Err("empty query from stdin".into());
            }
            Ok(trimmed)
        }
    }
}

fn exit_code_for(e: &YomuError) -> ExitCode {
    match e {
        YomuError::InvalidInput(_) => ExitCode::from(2),
        YomuError::Internal(_) => ExitCode::from(4),
        YomuError::Storage(_) | YomuError::Io(_) | YomuError::Index(_) | YomuError::Query(_) => {
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn resolve_query_with_direct_arg() {
        let mut stdin = Cursor::new(b"");
        let result = resolve_query_with(Some("auth hooks".into()), &mut stdin, true);
        assert_eq!(result.unwrap(), "auth hooks");
    }

    #[test]
    fn resolve_query_with_dash_reads_stdin() {
        let mut stdin = Cursor::new(b"piped query");
        let result = resolve_query_with(Some("-".into()), &mut stdin, false);
        assert_eq!(result.unwrap(), "piped query");
    }

    #[test]
    fn resolve_query_with_none_reads_stdin() {
        let mut stdin = Cursor::new(b"  streaming hooks  ");
        let result = resolve_query_with(None, &mut stdin, false);
        assert_eq!(result.unwrap(), "streaming hooks");
    }

    #[test]
    fn resolve_query_with_none_terminal_returns_error() {
        let mut stdin = Cursor::new(b"");
        let result = resolve_query_with(None, &mut stdin, true);
        assert!(result.unwrap_err().contains("query required"));
    }

    #[test]
    fn resolve_query_with_empty_stdin_returns_error() {
        let mut stdin = Cursor::new(b"   ");
        let result = resolve_query_with(None, &mut stdin, false);
        assert!(result.unwrap_err().contains("empty query"));
    }
}
