//! Seed-less recall and weighted cap-fit measurement for `yomu brief`, run
//! against the bundled GT corpus with the real embedding model.
//!
//! A separate crate, not part of the `yomu` binary (ADR-0005). Measurement is a
//! maintainer diagnostic, not a product surface. Seed-less recall needs the real
//! model, which a production build links; yomu's integration tests build the
//! `test-support` stub embedder, where seed inference is meaningless, so this
//! cannot live in a test.
//!
//! Run: `cargo run -p recall-bench -- --repo rurico [--json]`.

use std::process::ExitCode;

use clap::Parser;
use rurico::handle_probe_if_needed;
use yomu::error::ErrorCode;
use yomu::io::write_output;
use yomu::tools::{Yomu, YomuOptions};

#[derive(Parser)]
#[command(
    about = "Measure GT-corpus recall: seed-less closure recall, or --compare for the seed-stage embedding-vs-no-embed arm comparison (#250)"
)]
struct Args {
    /// GT corpus repo to measure against the current index (e.g. rurico, amici).
    #[arg(long)]
    repo: String,
    /// Emit a JSON report instead of the plain-text report.
    #[arg(long)]
    json: bool,
    /// Run the seed-stage arm comparison (embedding vs no-embed) instead of
    /// seed-less closure recall (#250 Phase 1).
    #[arg(long)]
    compare: bool,
}

fn main() -> ExitCode {
    // Answer the embedder's mlx-forward probe before clap sees probe args, the
    // same ordering the yomu binary's main uses. Without this a recall-bench
    // re-exec'd as a probe never replies, so every embedding query degrades
    // (#250).
    handle_probe_if_needed();
    let args = Args::parse();

    let yomu = match Yomu::new(YomuOptions { log_query: false }) {
        Ok(yomu) => yomu,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(ErrorCode::IoError.exit_code());
        }
    };

    // Mirrors the recall dispatch removed from the yomu CLI (ADR-0005): emit the
    // possibly-degraded report and exit non-zero when degraded, so a missing
    // model is not a silent pass. `--compare` switches to the #250 Phase 1
    // seed-stage arm comparison; both share the degraded-exit contract.
    let outcome = if args.compare {
        yomu.recall_arms(&args.repo, args.json)
    } else {
        yomu.recall(&args.repo, args.json)
    };
    match outcome {
        Ok((text, degraded)) => {
            let code = write_output(&text);
            if degraded {
                ExitCode::from(ErrorCode::TempFailure.exit_code())
            } else {
                code
            }
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(ErrorCode::IoError.exit_code())
        }
    }
}
