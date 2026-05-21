use std::env;
use std::ffi::OsString;
use std::fmt;
use std::io::{self, IsTerminal, Read};
use std::iter;
use std::process::ExitCode;

use amici::cli::{deprecation_warn, exit_error, hint_arrow, try_expand_shorthand};
use amici::logging::init_subscriber;
use clap::error::ErrorKind;
use clap::{CommandFactory, Parser, Subcommand};
use rurico::handle_probe_if_needed;
use yomu::brief;
use yomu::error::{self, ErrorCode};
use yomu::io::write_output;
use yomu::tools::{
    InvalidInputKind, MAX_IMPACT_DEPTH, MAX_SEARCH_LIMIT, MAX_SEARCH_OFFSET, Yomu, YomuError,
    YomuOptions,
};

#[derive(Parser)]
#[command(name = "yomu", version, about = "Frontend code search for AI agents")]
struct Cli {
    /// Output as JSON
    #[arg(long, global = true)]
    json: bool,
    /// Append a JSONL record per search to the XDG-resolved query log (default off, see Issue #182)
    #[arg(long, global = true)]
    log_query: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Semantic code search. Finds components, hooks, types by meaning.
    #[command(after_help = "\
Examples:
  yomu search \"streaming chat hooks\"
  yomu search --from src/query.rs:rerank
  yomu search \"auth\" --path src/auth --limit 5
  yomu --json search \"useAuth\"")]
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
        /// Skip embedding lookups; use FTS5 only. Same effect as YOMU_EMBED=0.
        /// Conflicts with `--from` (similarity search requires stored embeddings).
        #[arg(long, conflicts_with = "from")]
        no_embed: bool,
    },
    /// Update chunk index incrementally. No API calls.
    #[command(after_help = "\
Examples:
  yomu index
  yomu index --dry-run")]
    Index {
        /// Show what would be indexed without writing to the database
        #[arg(long)]
        dry_run: bool,
        /// Skip files classified as vendor (e.g. node_modules, vendor/, dist/) at walker time. Default off.
        #[arg(long)]
        exclude_vendor: bool,
    },
    /// Rebuild chunk index from scratch. No API calls.
    #[command(after_help = "\
Examples:
  yomu rebuild
  yomu rebuild --dry-run")]
    Rebuild {
        /// Show what would be rebuilt without writing to the database
        #[arg(long)]
        dry_run: bool,
        /// Skip files classified as vendor (e.g. node_modules, vendor/, dist/) at walker time. Default off.
        #[arg(long)]
        exclude_vendor: bool,
    },
    /// Analyze impact of changes to a file or symbol.
    #[command(after_help = "\
Examples:
  yomu impact src/hooks/useAuth.ts
  yomu impact src/hooks/useAuth.ts --symbol useAuth --depth 2
  yomu impact src/hooks/useAuth.ts --semantic")]
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
    #[command(after_help = "\
Examples:
  yomu status
  yomu --json status")]
    Status,
    /// Embed pending chunks for semantic search.
    #[command(after_help = "\
Examples:
  yomu embed
  yomu --json embed")]
    Embed,
    /// Bundle forward-closure code for an agent (recall-complete brief).
    #[command(after_help = "\
Examples:
  yomu brief \"add OAuth login\" --seed-file src/auth.rs
  yomu brief \"fix rerank scoring\" --seed-symbol rerank --depth 2
  yomu --json brief \"refactor query layer\" --seed-file src/query.rs --max-chunks 40")]
    Brief {
        /// Free-form task description (must not be empty)
        task: String,
        /// Seed file path (repeatable)
        #[arg(long)]
        seed_file: Vec<String>,
        /// Seed symbol name (repeatable)
        #[arg(long)]
        seed_symbol: Vec<String>,
        /// Forward closure depth (1..=10)
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..=10))]
        depth: u32,
        /// Maximum chunks in output (1..=1000)
        #[arg(long, default_value_t = 80, value_parser = clap::value_parser!(u32).range(1..=1000))]
        max_chunks: u32,
        /// Maximum bytes in output (1000..=10000000)
        #[arg(long, default_value_t = 80_000, value_parser = clap::value_parser!(u32).range(1000..=10_000_000))]
        max_bytes: u32,
        /// Skip embedding lookups; use FTS5 only.
        #[arg(long)]
        no_embed: bool,
    },
    /// Manage the embedding model.
    #[command(
        subcommand_required = true,
        arg_required_else_help = true,
        after_help = "\
Examples:
  yomu model download"
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
enum NoQueryReason {
    /// stdin is a terminal and no query argument was provided.
    Terminal,
    /// stdin was piped but contained no query content.
    EmptyStdin,
}

#[derive(Debug)]
enum QueryError {
    /// No query available — expected with --from, an error otherwise.
    NoQuery(NoQueryReason),
    /// I/O failure reading stdin — must propagate.
    Io(String),
}

impl fmt::Display for QueryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoQuery(NoQueryReason::Terminal) => {
                f.write_str("query required: pass as argument or pipe via stdin")
            }
            Self::NoQuery(NoQueryReason::EmptyStdin) => f.write_str("empty query from stdin"),
            Self::Io(msg) => f.write_str(msg),
        }
    }
}

fn main() -> ExitCode {
    handle_probe_if_needed();

    init_subscriber("yomu=warn");

    let cli = match parse_cli_args(env::args_os()) {
        Ok(cli) => cli,
        Err(e) if is_clap_display_exit(&e) => e.exit(),
        Err(e) => return render_clap_error(&e),
    };
    let json = cli.json;

    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            let err = Cli::command().error(ErrorKind::MissingSubcommand, "requires a subcommand");
            return render_clap_error(&err);
        }
    };

    // model subcommands do not require a project root or DB
    if let Command::Model { command } = &command {
        let result = match command {
            ModelCommand::Download => Yomu::model_download(json),
        };
        return match result {
            Ok(output) => write_output(&output),
            Err(e) => emit_error(&e, json),
        };
    }

    let yomu_options = match &command {
        Command::Search { no_embed, .. } | Command::Brief { no_embed, .. } => YomuOptions {
            no_embed: *no_embed,
            log_query: cli.log_query,
        },
        _ => YomuOptions {
            log_query: cli.log_query,
            ..YomuOptions::default()
        },
    };

    let yomu = match Yomu::new(yomu_options) {
        Ok(y) => y,
        Err(e) => return emit_error(&e, json),
    };

    let result = match command {
        Command::Search {
            query,
            limit,
            offset,
            path,
            from,
            format,
            no_embed: _,
        } => {
            if format.is_some() {
                deprecation_warn("--format", "--json");
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
                            return emit_error_code(&e.to_string(), ErrorCode::IoError, json);
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
                    Err(QueryError::NoQuery(reason)) => {
                        let kind = match reason {
                            NoQueryReason::Terminal => InvalidInputKind::QueryOrFromRequired,
                            NoQueryReason::EmptyStdin => InvalidInputKind::EmptyQuery,
                        };
                        return emit_error(&YomuError::InvalidInput(kind), json);
                    }
                    Err(e @ QueryError::Io(_)) => {
                        return emit_error_code(&e.to_string(), ErrorCode::IoError, json);
                    }
                };
                yomu.search(Some(&query), limit, offset, &path, json, None)
            }
        }
        Command::Index {
            dry_run,
            exclude_vendor,
        } => {
            if dry_run {
                yomu.dry_run_index(false, json, exclude_vendor)
            } else {
                yomu.index(json, exclude_vendor)
            }
        }
        Command::Rebuild {
            dry_run,
            exclude_vendor,
        } => {
            if dry_run {
                yomu.dry_run_index(true, json, exclude_vendor)
            } else {
                yomu.rebuild(json, exclude_vendor)
            }
        }
        Command::Impact {
            target,
            symbol,
            depth,
            semantic,
        } => yomu.impact(&target, symbol.as_deref(), depth, json, semantic),
        Command::Status => yomu.status(json),
        Command::Embed => yomu.embed(json),
        Command::Brief {
            task,
            seed_file,
            seed_symbol,
            depth,
            max_chunks,
            max_bytes,
            no_embed: _,
        } => {
            let task_brief = brief::TaskBrief {
                task,
                seeds: build_seeds(seed_file, seed_symbol),
                depth,
                max_chunks,
                max_bytes,
            };
            yomu.brief(&task_brief, json)
        }
        Command::Model { .. } => unreachable!("handled before Yomu::new()"),
    };

    match result {
        Ok(output) => write_output(&output),
        Err(e) => emit_error(&e, json),
    }
}

fn emit_error(err: &YomuError, json: bool) -> ExitCode {
    let code = err.error_code();
    let message = err.to_string();
    if json {
        eprintln!(
            "{}",
            error::render_json_error_with(
                code,
                &message,
                err.next_step(),
                &err.candidates(),
                err.retryable(),
            )
        );
    } else {
        exit_error(&message);
    }
    ExitCode::from(code.exit_code())
}

fn emit_error_code(message: &str, code: ErrorCode, json: bool) -> ExitCode {
    if json {
        eprintln!("{}", error::render_json_error(code, message));
    } else {
        exit_error(message);
    }
    ExitCode::from(code.exit_code())
}

fn is_clap_display_exit(e: &clap::Error) -> bool {
    matches!(
        e.kind(),
        ErrorKind::DisplayHelp
            | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
            | ErrorKind::DisplayVersion
    )
}

fn render_clap_error(e: &clap::Error) -> ExitCode {
    let rendered = e.to_string();
    let message = rendered
        .strip_prefix("error: ")
        .unwrap_or(&rendered)
        .trim_end();
    emit_error_code(message, ErrorCode::UsageError, false)
}

const KNOWN_SUBCOMMANDS: &[&str] = &[
    "search", "index", "rebuild", "impact", "status", "embed", "brief", "model",
];

fn build_seeds(files: Vec<String>, symbols: Vec<String>) -> Vec<brief::Seed> {
    let mut seeds = Vec::with_capacity(files.len() + symbols.len());
    seeds.extend(files.into_iter().map(|value| brief::Seed {
        kind: brief::SeedKind::File,
        value,
    }));
    seeds.extend(symbols.into_iter().map(|value| brief::Seed {
        kind: brief::SeedKind::Symbol,
        value,
    }));
    seeds
}
const GLOBAL_FLAGS: &[&str] = &["--json"];

fn parse_cli_args<I, T>(args: I) -> Result<Cli, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    let args: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let expanded = try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS);
    if let Some(expanded) = expanded
        && let Ok(cli) = Cli::try_parse_from(&expanded)
    {
        let display: Vec<_> = iter::once("yomu")
            .chain(expanded[1..].iter().filter_map(|a| a.to_str()))
            .collect();
        hint_arrow(&display);
        return Ok(cli);
    }
    Cli::try_parse_from(args)
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
                return Err(QueryError::NoQuery(NoQueryReason::Terminal));
            }
            let mut buf = String::new();
            stdin
                .read_to_string(&mut buf)
                .map_err(|e| QueryError::Io(format!("failed to read from stdin: {e}")))?;
            let trimmed = buf.trim();
            if trimmed.is_empty() {
                return Err(QueryError::NoQuery(NoQueryReason::EmptyStdin));
            }
            Ok(trimmed.to_owned())
        }
    }
}

fn resolve_query(arg: Option<String>) -> Result<String, QueryError> {
    let stdin = io::stdin();
    let is_terminal = stdin.is_terminal();
    resolve_query_with(arg, &mut stdin.lock(), is_terminal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // T-305: resolve_query_with_direct_arg
    #[test]
    fn resolve_query_with_direct_arg() {
        let mut stdin = Cursor::new(b"");
        let result = resolve_query_with(Some("auth hooks".into()), &mut stdin, true);
        assert_eq!(result.unwrap(), "auth hooks");
    }

    // T-306: resolve_query_with_dash_reads_stdin
    #[test]
    fn resolve_query_with_dash_reads_stdin() {
        let mut stdin = Cursor::new(b"piped query");
        let result = resolve_query_with(Some("-".into()), &mut stdin, false);
        assert_eq!(result.unwrap(), "piped query");
    }

    // T-307: resolve_query_with_none_reads_stdin
    #[test]
    fn resolve_query_with_none_reads_stdin() {
        let mut stdin = Cursor::new(b"  streaming hooks  ");
        let result = resolve_query_with(None, &mut stdin, false);
        assert_eq!(result.unwrap(), "streaming hooks");
    }

    // T-308: resolve_query_with_none_terminal_returns_no_query
    #[test]
    fn resolve_query_with_none_terminal_returns_no_query() {
        let mut stdin = Cursor::new(b"");
        let result = resolve_query_with(None, &mut stdin, true);
        let err = result.unwrap_err();
        assert!(matches!(err, QueryError::NoQuery(_)));
        assert!(err.to_string().contains("query required"));
    }

    // T-309: resolve_query_with_empty_stdin_returns_no_query
    #[test]
    fn resolve_query_with_empty_stdin_returns_no_query() {
        let mut stdin = Cursor::new(b"   ");
        let result = resolve_query_with(None, &mut stdin, false);
        let err = result.unwrap_err();
        assert!(matches!(err, QueryError::NoQuery(_)));
        assert!(err.to_string().contains("empty query"));
    }

    // RC-005: I/O errors must not be swallowed as NoQuery
    // T-310: resolve_query_with_io_error_returns_io_variant
    #[test]
    fn resolve_query_with_io_error_returns_io_variant() {
        struct FailingReader;
        impl io::Read for FailingReader {
            fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::new(io::ErrorKind::BrokenPipe, "broken pipe"))
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

    // T-049: parse_cli_args(["yomu", "query"]) → Command::Search (json=false) - regression
    #[test]
    fn shorthand_without_flags_has_json_false() {
        let cli = parse_cli_args(["yomu", "query"]).unwrap();
        assert!(!cli.json, "json should default to false");
        match cli.command.unwrap() {
            Command::Search { query, .. } => assert_eq!(query.as_deref(), Some("query")),
            other => panic!("expected Search, got {other:?}"),
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

    // T-563: multiple --path values
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

    // T-564: --path absent → empty vec (full search)
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

    // T-014: non-search subcommand names are not rewritten as search shorthand
    #[test]
    fn all_subcommands_not_shorthand() {
        for cmd in ["index", "rebuild", "impact", "status"] {
            let result = parse_cli_args(["yomu", cmd]);
            assert!(
                !matches!(
                    result.as_ref().map(|c| c.command.as_ref()),
                    Ok(Some(Command::Search { .. }))
                ),
                "subcommand '{cmd}' should not be rewritten as Search shorthand"
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

    // T-565: brief_parses_with_required_task
    #[test]
    fn brief_parses_with_required_task() {
        let cli = parse_cli_args(["yomu", "brief", "implement search"]).unwrap();
        match cli.command.unwrap() {
            Command::Brief {
                task,
                seed_file,
                seed_symbol,
                depth,
                max_chunks,
                max_bytes,
                ..
            } => {
                assert_eq!(task, "implement search");
                assert!(seed_file.is_empty());
                assert!(seed_symbol.is_empty());
                assert_eq!(depth, 3);
                assert_eq!(max_chunks, 80);
                assert_eq!(max_bytes, 80_000);
            }
            other => panic!("expected Brief, got {other:?}"),
        }
    }

    // T-566: brief_rejects_depth_out_of_range [Spec FR-005b]
    #[test]
    fn brief_rejects_depth_out_of_range() {
        let result = parse_cli_args(["yomu", "brief", "task", "--depth", "11"]);
        assert!(result.is_err(), "depth=11 must fail (range 1..=10)");

        let result = parse_cli_args(["yomu", "brief", "task", "--depth", "0"]);
        assert!(result.is_err(), "depth=0 must fail (range 1..=10)");
    }

    // T-567: brief_rejects_max_chunks_or_bytes_out_of_range [Spec FR-009b]
    #[test]
    fn brief_rejects_max_chunks_or_bytes_out_of_range() {
        let result = parse_cli_args(["yomu", "brief", "task", "--max-chunks", "1001"]);
        assert!(
            result.is_err(),
            "max-chunks=1001 must fail (range 1..=1000)"
        );

        let result = parse_cli_args(["yomu", "brief", "task", "--max-bytes", "999"]);
        assert!(
            result.is_err(),
            "max-bytes=999 must fail (range 1000..=10000000)"
        );
    }

    // T-568: brief_accepts_multiple_seed_files [Spec FR-013]
    #[test]
    fn brief_accepts_multiple_seed_files() {
        let cli = parse_cli_args([
            "yomu",
            "brief",
            "task",
            "--seed-file",
            "src/a.rs",
            "--seed-file",
            "src/b.rs",
            "--seed-symbol",
            "Foo",
        ])
        .unwrap();
        match cli.command.unwrap() {
            Command::Brief {
                seed_file,
                seed_symbol,
                ..
            } => {
                assert_eq!(seed_file, vec!["src/a.rs", "src/b.rs"]);
                assert_eq!(seed_symbol, vec!["Foo"]);
            }
            other => panic!("expected Brief, got {other:?}"),
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
