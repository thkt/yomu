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
use yomu::error::ErrorCode;
use yomu::io::write_output;
use yomu::tools::{Yomu, YomuOptions};

#[derive(Parser)]
#[command(about = "Measure seed-less recall and weighted cap-fit against the bundled GT corpus")]
struct Args {
    /// GT corpus repo to measure against the current index (e.g. rurico, amici).
    #[arg(long)]
    repo: String,
    /// Emit a JSON report instead of the plain-text report.
    #[arg(long)]
    json: bool,
}

fn main() -> ExitCode {
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
    // model is not a silent pass.
    match yomu.recall(&args.repo, args.json) {
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
