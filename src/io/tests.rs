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
