use std::ffi::OsString;
use std::iter;

use amici::cli::{hint_arrow, try_expand_shorthand};
use clap::{Parser, Subcommand};
use yomu::tools::{MAX_IMPACT_DEPTH, MAX_SEARCH_LIMIT, MAX_SEARCH_OFFSET};

#[derive(Parser)]
#[command(name = "yomu", version, about = "Frontend code search for AI agents")]
pub(crate) struct Cli {
    /// Output as JSON
    #[arg(long, global = true)]
    pub(crate) json: bool,
    /// Append a JSONL record per search to the XDG-resolved query log (default off, see Issue #182)
    #[arg(long, global = true)]
    pub(crate) log_query: bool,
    #[command(subcommand)]
    pub(crate) command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Semantic code search. Finds components, hooks, types by meaning.
    #[command(after_help = "\
Examples:
  yomu search \"streaming chat hooks\"
  yomu search --from src/query.rs:rerank
  yomu search \"auth\" --path src/auth --limit 5
  yomu --json search \"useAuth\"

Search is read-only; build the index first with `yomu index`.")]
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
    /// Update the index incrementally (chunks + embeddings). No API calls.
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
    /// Rebuild the index from scratch (chunks + embeddings). No API calls.
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
        /// Include test files in the closure (default: test files are excluded)
        #[arg(long)]
        include_tests: bool,
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
pub(crate) enum ModelCommand {
    /// Download embedding model from Hugging Face Hub.
    Download,
}

const KNOWN_SUBCOMMANDS: &[&str] = &[
    "search", "index", "rebuild", "impact", "status", "brief", "model",
];
const GLOBAL_FLAGS: &[&str] = &["--json", "--log-query"];

pub(crate) fn parse_cli_args<I, T>(args: I) -> Result<Cli, clap::Error>
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

#[cfg(test)]
mod tests {
    use super::*;

    // T-030: explicit `yomu search "認証"` is not double-injected
    #[test]
    fn explicit_search_not_double_injected() {
        let cli = parse_cli_args(["yomu", "search", "認証"]).unwrap();
        assert!(
            matches!(cli.command.unwrap(), Command::Search { query, .. } if query.as_deref() == Some("認証")),
            "expected Search with query=認証",
        );
    }

    // T-049: parse_cli_args(["yomu", "query"]) → Command::Search (json=false) - regression
    #[test]
    fn shorthand_without_flags_has_json_false() {
        let cli = parse_cli_args(["yomu", "query"]).unwrap();
        assert!(!cli.json, "json should default to false");
        assert!(
            matches!(cli.command.unwrap(), Command::Search { query, .. } if query.as_deref() == Some("query")),
            "expected Search with query=query",
        );
    }

    // T-569: --log-query is a global flag and must be preserved during shorthand expansion.
    #[test]
    fn shorthand_with_log_query_flag_sets_log_query_true() {
        let cli = parse_cli_args(["yomu", "--log-query", "query"]).unwrap();
        assert!(cli.log_query, "log_query should be true");
        assert!(
            matches!(cli.command.unwrap(), Command::Search { query, .. } if query.as_deref() == Some("query")),
            "expected Search with query=query",
        );
    }

    // T-076: --path parses into path vec
    #[test]
    fn search_path_filter_parses() {
        let cli = parse_cli_args(["yomu", "search", "query", "--path", "src/fetcher/"]).unwrap();
        assert!(
            matches!(cli.command.unwrap(), Command::Search { path, .. } if path == ["src/fetcher/"]),
            "expected Search with path=[src/fetcher/]",
        );
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
        assert!(
            matches!(cli.command.unwrap(), Command::Search { path, .. } if path == ["src/fetcher/", "src/client/"]),
            "expected Search with path=[src/fetcher/, src/client/]",
        );
    }

    // T-564: --path absent → empty vec (full search)
    #[test]
    fn search_no_path_defaults_to_empty() {
        let cli = parse_cli_args(["yomu", "search", "query"]).unwrap();
        assert!(
            matches!(cli.command.unwrap(), Command::Search { path, .. } if path.is_empty()),
            "expected Search with empty path",
        );
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
        assert!(
            matches!(cli.command.unwrap(), Command::Search { query, from, .. } if query.is_none() && from.as_deref() == Some("src/foo.rs")),
            "expected Search with query=None, from=src/foo.rs",
        );
    }

    // T-078: --semantic flag on impact parses to semantic=true
    #[test]
    fn impact_semantic_flag_parses() {
        let cli = parse_cli_args(["yomu", "impact", "src/foo.rs", "--semantic"]).unwrap();
        assert!(
            matches!(cli.command.unwrap(), Command::Impact { target, semantic, .. } if target == "src/foo.rs" && semantic),
            "expected Impact with target=src/foo.rs, semantic=true",
        );
    }

    // T-079: impact without --semantic defaults to semantic=false
    #[test]
    fn impact_no_semantic_flag_defaults_false() {
        let cli = parse_cli_args(["yomu", "impact", "src/foo.rs"]).unwrap();
        assert!(
            matches!(cli.command.unwrap(), Command::Impact { semantic, .. } if !semantic),
            "expected Impact with semantic=false by default",
        );
    }

    // T-565: brief_parses_with_required_task
    #[test]
    fn brief_parses_with_required_task() {
        let cli = parse_cli_args(["yomu", "brief", "implement search"]).unwrap();
        assert!(
            matches!(
                cli.command.unwrap(),
                Command::Brief {
                    task,
                    seed_file,
                    seed_symbol,
                    depth,
                    max_chunks,
                    max_bytes,
                    ..
                } if task == "implement search"
                    && seed_file.is_empty()
                    && seed_symbol.is_empty()
                    && depth == 3
                    && max_chunks == 80
                    && max_bytes == 80_000
            ),
            "expected Brief with task=implement search and default depth/chunks/bytes",
        );
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
        assert!(
            matches!(
                cli.command.unwrap(),
                Command::Brief { seed_file, seed_symbol, .. }
                    if seed_file == ["src/a.rs", "src/b.rs"] && seed_symbol == ["Foo"]
            ),
            "expected Brief with seed_file=[src/a.rs, src/b.rs], seed_symbol=[Foo]",
        );
    }

    // T-016: no --from, no query → from defaults to None (error comes from resolve_query)
    #[test]
    fn no_from_no_query_has_from_none() {
        let cli = parse_cli_args(["yomu", "search"]).unwrap();
        assert!(
            matches!(cli.command.unwrap(), Command::Search { query, from, .. } if query.is_none() && from.is_none()),
            "expected Search with query=None, from=None",
        );
    }
}
