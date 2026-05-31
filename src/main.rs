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
use yomu::tools::{IndexRunOptions, InvalidInputKind, Yomu, YomuError, YomuOptions};

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
    let json = cli.json;

    let command = match cli.command {
        Some(cmd) => cmd,
        None => {
            let err =
                cli::Cli::command().error(ErrorKind::MissingSubcommand, "requires a subcommand");
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

    let yomu_options = YomuOptions {
        log_query: cli.log_query,
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
            let opts = IndexRunOptions {
                force: false,
                exclude_vendor,
            };
            if dry_run {
                yomu.dry_run_index(opts, json)
            } else {
                yomu.index(opts, json)
            }
        }
        Command::Rebuild {
            dry_run,
            exclude_vendor,
        } => {
            let opts = IndexRunOptions {
                force: true,
                exclude_vendor,
            };
            if dry_run {
                yomu.dry_run_index(opts, json)
            } else {
                yomu.rebuild(opts, json)
            }
        }
        Command::Impact {
            target,
            symbol,
            depth,
            semantic,
        } => yomu.impact(&target, symbol.as_deref(), depth, json, semantic),
        Command::Status => yomu.status(json),
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
}
