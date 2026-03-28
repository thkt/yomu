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

    let cli = parse_cli();

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

/// Parse CLI args, treating unknown first arguments as implicit `search` queries.
/// e.g. `yomu "auth hook"` becomes `yomu search "auth hook"`.
/// Near-matches of known subcommands (edit distance ≤ 2) are not rewritten,
/// so clap can show "did you mean?" suggestions for typos.
fn parse_cli() -> Cli {
    let args: Vec<String> = std::env::args().collect();
    let cmd = Cli::command();
    let known: Vec<&str> = cmd.get_subcommands().map(|s| s.get_name()).collect();
    if args.len() > 1
        && !args[1].starts_with('-')
        && args[1] != "help"
        && !known.contains(&args[1].as_str())
        && !is_near_subcommand(&args[1], &known)
    {
        let mut patched = vec![args[0].clone(), "search".to_string()];
        patched.extend_from_slice(&args[1..]);
        Cli::parse_from(patched)
    } else {
        Cli::parse()
    }
}

/// Returns true if `input` is within Damerau-Levenshtein distance 1 of any known subcommand.
/// Uses OSA (Optimal String Alignment) distance which counts adjacent transpositions as 1 edit.
fn is_near_subcommand(input: &str, known: &[&str]) -> bool {
    known.iter().any(|cmd| osa_distance(input, cmd) <= 1)
}

#[allow(clippy::needless_range_loop)]
fn osa_distance(a: &str, b: &str) -> usize {
    let a = a.as_bytes();
    let b = b.as_bytes();
    let (m, n) = (a.len(), b.len());
    if m.abs_diff(n) > 1 {
        return m.abs_diff(n);
    }
    let mut d = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m {
        d[i][0] = i;
    }
    for j in 0..=n {
        d[0][j] = j;
    }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            d[i][j] = (d[i - 1][j] + 1)
                .min(d[i][j - 1] + 1)
                .min(d[i - 1][j - 1] + cost);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                d[i][j] = d[i][j].min(d[i - 2][j - 2] + 1);
            }
        }
    }
    d[m][n]
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

    #[test]
    fn near_subcommand_catches_typos() {
        let known = ["search", "index", "rebuild", "impact", "status"];
        // Transpositions (OSA distance 1)
        assert!(is_near_subcommand("stauts", &known)); // status
        assert!(is_near_subcommand("serach", &known)); // search
        // Single deletion
        assert!(is_near_subcommand("indx", &known)); // index
    }

    #[test]
    fn near_subcommand_passes_valid_queries() {
        let known = ["search", "index", "rebuild", "impact", "status"];
        assert!(!is_near_subcommand("state", &known)); // OSA 2 from status
        assert!(!is_near_subcommand("auth", &known));
        assert!(!is_near_subcommand("button", &known));
        assert!(!is_near_subcommand("statusBar", &known));
        assert!(!is_near_subcommand("認証", &known));
    }
}
