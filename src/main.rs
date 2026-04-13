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
        /// Search for code similar to the given file or symbol (e.g. "src/foo.rs" or "src/foo.rs:my_fn")
        #[arg(long)]
        from: Option<String>,
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
        /// Include semantically related files via embedding search (in addition to import graph)
        #[arg(long)]
        semantic: bool,
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

#[derive(Debug)]
enum QueryError {
    /// No query available (terminal, empty stdin) — expected with --from
    NoQuery(String),
    /// I/O failure reading stdin — must propagate
    Io(String),
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoQuery(msg) | Self::Io(msg) => f.write_str(msg),
        }
    }
}

fn main() -> ExitCode {
    rurico::model_probe::handle_probe_if_needed();

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
            from,
            format,
        } => {
            if format.is_some() {
                eprintln!("warning: --format is deprecated, use --json instead");
            }
            let json = json || format.as_deref() == Some("json");
            if from.is_some() {
                // Literal query: use as-is. "-" / None: try stdin (optional with --from).
                let is_literal = query.as_deref().is_some_and(|q| q != "-");
                let query = if is_literal {
                    query
                } else {
                    match resolve_query(query) {
                        Ok(q) => Some(q),
                        Err(QueryError::NoQuery(_)) => None,
                        Err(e @ QueryError::Io(_)) => {
                            eprintln!("error: {e}");
                            return ExitCode::FAILURE;
                        }
                    }
                };
                yomu.search(
                    query.as_deref(),
                    limit,
                    offset,
                    &path,
                    json,
                    from.as_deref(),
                )
            } else {
                let query = match resolve_query(query) {
                    Ok(q) => q,
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::from(2);
                    }
                };
                yomu.search(Some(&query), limit, offset, &path, json, None)
            }
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
            semantic,
        } => yomu.impact(&target, symbol.as_deref(), depth, json, semantic),
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

fn resolve_query(arg: Option<String>) -> Result<String, QueryError> {
    let stdin = std::io::stdin();
    let is_terminal = stdin.is_terminal();
    resolve_query_with(arg, &mut stdin.lock(), is_terminal)
}

fn resolve_query_with(
    arg: Option<String>,
    stdin: &mut impl Read,
    stdin_is_terminal: bool,
) -> Result<String, QueryError> {
    match arg {
        Some(q) if q != "-" => Ok(q),
        _ => {
            if stdin_is_terminal {
                return Err(QueryError::NoQuery(
                    "query required: pass as argument or pipe via stdin".into(),
                ));
            }
            let mut buf = String::new();
            stdin
                .read_to_string(&mut buf)
                .map_err(|e| QueryError::Io(format!("failed to read from stdin: {e}")))?;
            let trimmed = buf.trim();
            if trimmed.is_empty() {
                return Err(QueryError::NoQuery("empty query from stdin".into()));
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
    fn resolve_query_with_none_terminal_returns_no_query() {
        let mut stdin = Cursor::new(b"");
        let result = resolve_query_with(None, &mut stdin, true);
        let err = result.unwrap_err();
        assert!(matches!(err, QueryError::NoQuery(_)));
        assert!(err.to_string().contains("query required"));
    }

    #[test]
    fn resolve_query_with_empty_stdin_returns_no_query() {
        let mut stdin = Cursor::new(b"   ");
        let result = resolve_query_with(None, &mut stdin, false);
        let err = result.unwrap_err();
        assert!(matches!(err, QueryError::NoQuery(_)));
        assert!(err.to_string().contains("empty query"));
    }

    // RC-005: I/O errors must not be swallowed as NoQuery
    #[test]
    fn resolve_query_with_io_error_returns_io_variant() {
        struct FailingReader;
        impl std::io::Read for FailingReader {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "broken pipe",
                ))
            }
        }
        let result = resolve_query_with(None, &mut FailingReader, false);
        let err = result.unwrap_err();
        assert!(matches!(err, QueryError::Io(_)));
        assert!(err.to_string().contains("failed to read from stdin"));
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

    // T-015: --from without query parses OK
    #[test]
    fn from_flag_without_query_parses_ok() {
        let cli = parse_cli_args(["yomu", "search", "--from", "src/foo.rs"]).unwrap();
        match cli.command.unwrap() {
            Command::Search { query, from, .. } => {
                assert_eq!(query, None);
                assert_eq!(from.as_deref(), Some("src/foo.rs"));
            }
            other => panic!("expected Search, got {other:?}"),
        }
    }

    // T-078: --semantic flag on impact parses to semantic=true
    #[test]
    fn impact_semantic_flag_parses() {
        let cli = parse_cli_args(["yomu", "impact", "src/foo.rs", "--semantic"]).unwrap();
        match cli.command.unwrap() {
            Command::Impact {
                target, semantic, ..
            } => {
                assert_eq!(target, "src/foo.rs");
                assert!(semantic, "expected semantic=true");
            }
            other => panic!("expected Impact, got {other:?}"),
        }
    }

    // T-079: impact without --semantic defaults to semantic=false
    #[test]
    fn impact_no_semantic_flag_defaults_false() {
        let cli = parse_cli_args(["yomu", "impact", "src/foo.rs"]).unwrap();
        match cli.command.unwrap() {
            Command::Impact { semantic, .. } => {
                assert!(!semantic, "expected semantic=false by default");
            }
            other => panic!("expected Impact, got {other:?}"),
        }
    }

    // T-016: no --from, no query → from defaults to None (error comes from resolve_query)
    #[test]
    fn no_from_no_query_has_from_none() {
        let cli = parse_cli_args(["yomu", "search"]).unwrap();
        match cli.command.unwrap() {
            Command::Search { query, from, .. } => {
                assert_eq!(query, None);
                assert_eq!(from, None);
            }
            other => panic!("expected Search, got {other:?}"),
        }
    }
}
