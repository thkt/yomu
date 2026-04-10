mod shorthand;

use yomu::progress;

use std::io::{IsTerminal, Read};
use std::process::ExitCode;

use clap::{CommandFactory, Parser, Subcommand};
use yomu::tools::{MAX_IMPACT_DEPTH, MAX_SEARCH_LIMIT, MAX_SEARCH_OFFSET, Yomu, YomuError};

#[derive(Parser)]
#[command(name = "yomu", version, about = "Frontend code search for AI agents")]
struct Cli {
    /// Output as JSON
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
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
        /// Restrict search to files under this path prefix (repeatable)
        #[arg(long)]
        path: Vec<String>,
        /// Deprecated: use global --json instead
        #[arg(long, hide = true)]
        format: Option<String>,
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
    /// Manage the embedding model.
    #[command(
        subcommand_required = true,
        arg_required_else_help = true,
        after_help = "\
Examples:
  yomu model download
  YOMU_AUTO_DOWNLOAD_MODEL=1 yomu search \"auth hooks\""
    )]
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Download embedding model from Hugging Face Hub.
    Download,
}

fn main() -> ExitCode {
    rurico::embed::handle_probe_if_needed();

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("yomu=warn")),
        )
        .init();

    let cli = parse_cli_args(std::env::args_os()).unwrap_or_else(|e| e.exit());
    let json = cli.json;

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

    // model subcommands do not require a project root or DB
    if let Command::Model { command } = &command {
        let result = match command {
            ModelCommand::Download => run_model_download(json),
        };
        return match result {
            Ok(output) => {
                println!("{output}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("error: {e}");
                exit_code_for(&e)
            }
        };
    }

    let yomu = match Yomu::new() {
        Ok(y) => y,
        Err(e) => {
            eprintln!("error: {e}");
            return exit_code_for(&e);
        }
    };

    let result = match command {
        Command::Search {
            query,
            limit,
            offset,
            path,
            format,
        } => {
            if format.is_some() {
                eprintln!("warning: --format is deprecated, use --json instead");
            }
            let json = json || format.as_deref() == Some("json");
            let query = match resolve_query(query) {
                Ok(q) => q,
                Err(e) => {
                    eprintln!("error: {e}");
                    return ExitCode::from(2);
                }
            };
            yomu.search(&query, limit, offset, &path, json)
        }
        Command::Index { dry_run } => {
            if dry_run {
                yomu.dry_run_index(false, json)
            } else {
                yomu.index(json)
            }
        }
        Command::Rebuild { dry_run } => {
            if dry_run {
                yomu.dry_run_index(true, json)
            } else {
                yomu.rebuild(json)
            }
        }
        Command::Impact {
            target,
            symbol,
            depth,
        } => yomu.impact(&target, symbol.as_deref(), depth, json),
        Command::Status => yomu.status(json),
        Command::Model { .. } => unreachable!("handled before Yomu::new()"),
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

const KNOWN_SUBCOMMANDS: &[&str] = &["search", "index", "rebuild", "impact", "status", "model"];
const GLOBAL_FLAGS: &[&str] = &["--json"];

fn parse_cli_args<I, T>(args: I) -> Result<Cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let args: Vec<std::ffi::OsString> = args.into_iter().map(Into::into).collect();
    let expanded = shorthand::try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS);
    if let Some(expanded) = expanded
        && let Ok(cli) = Cli::try_parse_from(&expanded)
    {
        let display: Vec<_> = std::iter::once("yomu")
            .chain(expanded[1..].iter().filter_map(|a| a.to_str()))
            .collect();
        eprintln!("→ {}", display.join(" "));
        return Ok(cli);
    }
    Cli::try_parse_from(args)
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
            let trimmed = buf.trim();
            if trimmed.is_empty() {
                return Err("empty query from stdin".into());
            }
            Ok(trimmed.to_string())
        }
    }
}

fn run_model_download(json: bool) -> Result<String, YomuError> {
    use rurico::embed::{EmbedInitError, Embedder, ModelId, ProbeStatus};

    let spinner = progress::Spinner::new("Downloading model...");
    let paths = match rurico::embed::download_model(ModelId::default()) {
        Ok(p) => p,
        Err(e) => {
            spinner.cancel();
            tracing::error!(error = %e, "Model download failed");
            return Err(YomuError::Internal(format!(
                "Failed to download model: {e}"
            )));
        }
    };
    match Embedder::probe(&paths) {
        Ok(ProbeStatus::Available) => {}
        Ok(ProbeStatus::BackendUnavailable) => {
            spinner.cancel();
            return Err(YomuError::Internal(
                "Model downloaded but MLX backend is unavailable".to_string(),
            ));
        }
        Err(e) => {
            spinner.cancel();
            if matches!(e, EmbedInitError::ModelCorrupt { .. })
                && let Err(del_err) = paths.delete_files()
            {
                tracing::warn!(error = %del_err, "failed to delete corrupt model files");
            }
            return Err(YomuError::Internal(format!("Model probe failed: {e}")));
        }
    }
    match Embedder::new(&paths) {
        Ok(_) => {
            spinner.finish("Model ready");
            if json {
                Ok(serde_json::json!({"status": "ok"}).to_string())
            } else {
                Ok("Model downloaded and verified".to_string())
            }
        }
        Err(e) => {
            spinner.cancel();
            Err(YomuError::Internal(format!("Failed to verify model: {e}")))
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

    // T-030: explicit `yomu search "認証"` is not double-injected
    #[test]
    fn explicit_search_not_double_injected() {
        let cli = parse_cli_args(["yomu", "search", "認証"]).unwrap();
        match cli.command.unwrap() {
            Command::Search { query, .. } => assert_eq!(query.as_deref(), Some("認証")),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    // T-033: `yomu search` (missing arg) parses to stdin fallback path
    #[test]
    fn search_missing_arg_parses_for_stdin_fallback() {
        let cli = parse_cli_args(["yomu", "search"]).unwrap();
        match cli.command.unwrap() {
            Command::Search { query, .. } => assert_eq!(query, None),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    // T-049: parse_cli_args(["yomu", "query"]) → Command::Search (json=false) - regression
    #[test]
    fn shorthand_without_flags_has_json_false() {
        let cli = parse_cli_args(["yomu", "query"]).unwrap();
        assert!(!cli.json, "[T-049] json should default to false");
        match cli.command.unwrap() {
            Command::Search { query, .. } => assert_eq!(query.as_deref(), Some("query")),
            other => panic!("[T-049] expected Search, got {other:?}"),
        }
    }

    // T-076: --path parses into path vec
    #[test]
    fn search_path_filter_parses() {
        let cli = parse_cli_args(["yomu", "search", "query", "--path", "src/fetcher/"]).unwrap();
        match cli.command.unwrap() {
            Command::Search { path, .. } => {
                assert_eq!(path, vec!["src/fetcher/"]);
            }
            other => panic!("expected Search, got {other:?}"),
        }
    }

    // T-076: multiple --path values
    #[test]
    fn search_multiple_path_filters_parse() {
        let cli = parse_cli_args([
            "yomu",
            "search",
            "query",
            "--path",
            "src/fetcher/",
            "--path",
            "src/client/",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Command::Search { path, .. } => {
                assert_eq!(path, vec!["src/fetcher/", "src/client/"]);
            }
            other => panic!("expected Search, got {other:?}"),
        }
    }

    // T-076: --path absent → empty vec (full search)
    #[test]
    fn search_no_path_defaults_to_empty() {
        let cli = parse_cli_args(["yomu", "search", "query"]).unwrap();
        match cli.command.unwrap() {
            Command::Search { path, .. } => assert!(path.is_empty()),
            other => panic!("expected Search, got {other:?}"),
        }
    }

    // T-025: typo (OSA ≤ 1) → clap error, not shorthand expansion
    #[test]
    fn typo_subcommand_is_clap_error() {
        let result = parse_cli_args(["yomu", "serach"]);
        assert!(result.is_err(), "typo 'serach' should be clap error");
    }

    // TC-014: non-search subcommand names are not rewritten as search shorthand
    #[test]
    fn all_subcommands_not_shorthand() {
        for cmd in ["index", "rebuild", "impact", "status"] {
            let result = parse_cli_args(["yomu", cmd]);
            assert!(
                !matches!(
                    result.as_ref().map(|c| c.command.as_ref()),
                    Ok(Some(Command::Search { .. }))
                ),
                "[TC-014] subcommand '{cmd}' should not be rewritten as Search shorthand"
            );
        }
    }
}
