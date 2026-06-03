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
mod tests;
