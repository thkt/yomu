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

use amici::cli::exit_code::codes;

/// Writes `output` followed by a newline to stdout. `BrokenPipe` is treated as
/// successful completion (consumer chose to stop reading); other I/O errors
/// surface as `IO_ERR` exit code.
pub fn write_output(output: &str) -> ExitCode {
    let stdout = io::stdout();
    let mut handle = stdout.lock();
    write_output_to(&mut handle, output)
}

pub(crate) fn write_output_to<W: Write>(writer: &mut W, output: &str) -> ExitCode {
    match writeln!(writer, "{output}") {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("yomu: stdout write failed: {e}");
            ExitCode::from(codes::IO_ERR)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FailingWriter {
        kind: io::ErrorKind,
    }

    impl Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(self.kind, "induced"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    // T-IO-001: write_output_to writes the payload plus a trailing newline.
    #[test]
    fn write_output_to_appends_newline_and_succeeds() {
        let mut buf: Vec<u8> = Vec::new();
        let code = write_output_to(&mut buf, "hello");
        assert_eq!(code, ExitCode::SUCCESS);
        assert_eq!(buf, b"hello\n");
    }

    // T-IO-002: BrokenPipe collapses to SUCCESS (pipe consumer closed early).
    #[test]
    fn write_output_to_broken_pipe_returns_success() {
        let mut w = FailingWriter {
            kind: io::ErrorKind::BrokenPipe,
        };
        let code = write_output_to(&mut w, "ignored");
        assert_eq!(code, ExitCode::SUCCESS);
    }

    // T-IO-003: non-pipe I/O failures surface as IO_ERR (74).
    #[test]
    fn write_output_to_other_io_error_returns_io_err() {
        let mut w = FailingWriter {
            kind: io::ErrorKind::PermissionDenied,
        };
        let code = write_output_to(&mut w, "ignored");
        assert_eq!(code, ExitCode::from(codes::IO_ERR));
    }
}
