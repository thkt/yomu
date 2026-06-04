mod cli;

use std::env;
use std::ffi::OsString;
use std::fmt;
use std::io::{self, IsTerminal, Read};
use std::process::ExitCode;

use amici::cli::{deprecation_warn, exit_error};
use amici::logging::init_subscriber;
use clap::CommandFactory;
use clap::error::ErrorKind;
use cli::{Command, ModelCommand, parse_cli_args};
use rurico::handle_probe_if_needed;
use yomu::brief;
use yomu::error::{self, ErrorCode};
use yomu::io::write_output;
use yomu::tools::{IndexRunOptions, InvalidInputKind, OutputFormat, Yomu, YomuError, YomuOptions};

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
    run(env::args_os())
}

fn run<I, T>(args: I) -> ExitCode
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    handle_probe_if_needed();
    init_subscriber("yomu=warn");

    let cli = match parse_cli_args(args) {
        Ok(cli) => cli,
        Err(e) if is_clap_display_exit(&e) => e.exit(),
        Err(e) => return render_clap_error(&e),
    };
    let Some(command) = cli.command else {
        let err = cli::Cli::command().error(ErrorKind::MissingSubcommand, "requires a subcommand");
        return render_clap_error(&err);
    };

    dispatch(command, cli.json, cli.log_query)
}

/// Routes a parsed subcommand to its handler. `model` needs no project root or
/// DB, so it runs before `Yomu::new`; every other command shares the DB-backed
/// `Yomu` and funnels its `Result` through [`finish`].
fn dispatch(command: Command, json: bool, log_query: bool) -> ExitCode {
    if let Command::Model { command } = &command {
        let result = match command {
            ModelCommand::Download => Yomu::model_download(json),
        };
        return finish(result, json);
    }

    let yomu = match Yomu::new(YomuOptions { log_query }) {
        Ok(y) => y,
        Err(e) => return emit_error(&e, json),
    };
    run_command(&yomu, command, json)
}

/// Subcommand fan-out: each arm is a thin wire to a `Yomu` method, except the
/// search query resolution ([`resolve_search_query`]) and the brief request
/// assembly. Kept as one match (over the 30-line guideline but well under
/// clippy's 100) so the dispatch table reads top-to-bottom.
fn run_command(yomu: &Yomu, command: Command, json: bool) -> ExitCode {
    match command {
        Command::Search {
            query,
            limit,
            offset,
            path,
            from,
            format,
        } => {
            if format.is_some() {
                deprecation_warn("--format", "--json");
            }
            let json = json || format.as_deref() == Some("json");
            let query = match resolve_search_query(query, from.is_some()) {
                Ok(q) => q,
                Err(e) => return render_query_error(&e, json),
            };
            finish(
                yomu.search(
                    query.as_deref(),
                    limit,
                    offset,
                    &path,
                    OutputFormat::from_json_flag(json),
                    from.as_deref(),
                ),
                json,
            )
        }
        Command::Index {
            dry_run,
            exclude_vendor,
        } => handle_index_run(yomu, dry_run, exclude_vendor, IndexMode::Fresh, json),
        Command::Rebuild {
            dry_run,
            exclude_vendor,
        } => handle_index_run(yomu, dry_run, exclude_vendor, IndexMode::Rebuild, json),
        Command::Impact {
            target,
            symbol,
            depth,
            semantic,
        } => finish(
            yomu.impact(
                &target,
                symbol.as_deref(),
                depth,
                OutputFormat::from_json_flag(json),
                semantic,
            ),
            json,
        ),
        Command::Status => finish(yomu.status(json), json),
        Command::Brief {
            task,
            seed_file,
            seed_symbol,
            depth,
            max_chunks,
            max_bytes,
            include_tests,
        } => {
            let task_brief = brief::TaskBrief {
                task,
                seeds: build_seeds(seed_file, seed_symbol),
                depth,
                max_chunks,
                max_bytes,
                include_tests,
            };
            finish(yomu.brief(&task_brief, json), json)
        }
        Command::Model { .. } => unreachable!("handled before Yomu::new()"),
    }
}

/// Single exit point for every command result, keeping each `run_command` arm a
/// one-liner: success writes the output, failure renders the typed error.
fn finish(result: Result<String, YomuError>, json: bool) -> ExitCode {
    match result {
        Ok(output) => write_output(&output),
        Err(e) => emit_error(&e, json),
    }
}

/// Whether an index command overwrites the existing index. `Rebuild` forces a
/// full rebuild; `Fresh` indexes without forcing.
#[derive(Debug, Clone, Copy)]
enum IndexMode {
    Fresh,
    Rebuild,
}

/// Runs an index or rebuild, collapsing the dry-run, fresh-index, and
/// force-rebuild variants. A dry run reports without writing; otherwise
/// `IndexMode::Rebuild` forces a rebuild over a plain index.
fn handle_index_run(
    yomu: &Yomu,
    dry_run: bool,
    exclude_vendor: bool,
    mode: IndexMode,
    json: bool,
) -> ExitCode {
    let force = matches!(mode, IndexMode::Rebuild);
    let opts = IndexRunOptions {
        force,
        exclude_vendor,
    };
    let result = if dry_run {
        yomu.dry_run_index(opts, json)
    } else if force {
        yomu.rebuild(opts, json)
    } else {
        yomu.index(opts, json)
    };
    finish(result, json)
}

/// Resolves the search query from the argument or stdin, reading the real
/// stdin. With `--from` a missing query is allowed (`Ok(None)`); without it a
/// missing query surfaces as a [`QueryError`] the caller renders via
/// [`render_query_error`].
fn resolve_search_query(
    query: Option<String>,
    has_from: bool,
) -> Result<Option<String>, QueryError> {
    let stdin = io::stdin();
    let is_terminal = stdin.is_terminal();
    resolve_search_query_with(query, has_from, &mut stdin.lock(), is_terminal)
}

/// Stdin-injectable core of [`resolve_search_query`] (the seam unit tests
/// drive). A literal `--from` query is returned as-is; otherwise the query is
/// read via [`resolve_query_with`], and `--from` turns a missing query into
/// `Ok(None)` rather than an error.
fn resolve_search_query_with(
    query: Option<String>,
    has_from: bool,
    stdin: &mut impl Read,
    stdin_is_terminal: bool,
) -> Result<Option<String>, QueryError> {
    // Literal query under --from: use as-is. "-" / None falls through to stdin.
    if has_from && query.as_deref().is_some_and(|q| q != "-") {
        return Ok(query);
    }
    match resolve_query_with(query, stdin, stdin_is_terminal) {
        Ok(q) => Ok(Some(q)),
        Err(QueryError::NoQuery(_)) if has_from => Ok(None),
        Err(e) => Err(e),
    }
}

/// Renders a search [`QueryError`] to the process exit code: a missing query is
/// a usage error (mapped to the matching [`InvalidInputKind`]); an I/O failure
/// keeps its message under [`ErrorCode::IoError`].
fn render_query_error(error: &QueryError, json: bool) -> ExitCode {
    match error {
        QueryError::NoQuery(reason) => {
            emit_error(&YomuError::InvalidInput(missing_query_kind(reason)), json)
        }
        QueryError::Io(msg) => emit_error_code(msg, ErrorCode::IoError, json),
    }
}

/// Maps a missing-query reason to its user-facing input error: no query at all
/// is a usage error; an empty pipe is a distinct empty-query error. Split out so
/// the mapping is unit-tested directly rather than only through the binary.
fn missing_query_kind(reason: &NoQueryReason) -> InvalidInputKind {
    match reason {
        NoQueryReason::Terminal => InvalidInputKind::QueryOrFromRequired,
        NoQueryReason::EmptyStdin => InvalidInputKind::EmptyQuery,
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

#[cfg(test)]
mod tests;
