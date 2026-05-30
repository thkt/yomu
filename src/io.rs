//! Stdout writer that converts `BrokenPipe` (EPIPE) to a clean exit.
//!
//! When yomu's stdout is piped to a consumer that closes early (`| head -0`,
//! `| true`), Rust's default panic-on-broken-pipe surfaces as ExitCode 101
//! and a noisy stderr message. AI agents reading the output then misclassify
//! this as a yomu failure even though the consumer chose to stop reading.
//! `write_output` drains `BrokenPipe` into `ExitCode::SUCCESS` while still
//! reporting other I/O failures with the `IO_ERR` sysexits code so genuine
//! disk / encoding faults remain observable.

use std::io::{self, Write};
use std::process::ExitCode;

use amici::cli::{exit_code::codes, exit_error};

/// Writes `output` plus a newline to stdout. See module docs for the
/// `BrokenPipe` / `IO_ERR` exit-code contract.
pub fn write_output(output: &str) -> ExitCode {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    write_output_to(&mut handle, output)
}

fn write_output_to<W: Write>(writer: &mut W, output: &str) -> ExitCode {
    match writeln!(writer, "{output}") {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(e) => {
            exit_error(&format!("stdout write failed: {e}"));
            ExitCode::from(codes::IO_ERR)
        }
    }
}

#[cfg(test)]
mod tests;
