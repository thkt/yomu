use super::*;
use std::assert_matches;
use std::io::Cursor;

/// A reader whose every `read` fails, for exercising the stdin I/O error path.
struct FailingReader;
impl io::Read for FailingReader {
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::new(io::ErrorKind::BrokenPipe, "broken pipe"))
    }
}

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
    assert_matches!(err, QueryError::NoQuery(_));
    assert!(err.to_string().contains("query required"));
}

// T-309: resolve_query_with_empty_stdin_returns_no_query
#[test]
fn resolve_query_with_empty_stdin_returns_no_query() {
    let mut stdin = Cursor::new(b"   ");
    let result = resolve_query_with(None, &mut stdin, false);
    let err = result.unwrap_err();
    assert_matches!(err, QueryError::NoQuery(_));
    assert!(err.to_string().contains("empty query"));
}

// RC-005: I/O errors must not be swallowed as NoQuery
// T-310: resolve_query_with_io_error_returns_io_variant
#[test]
fn resolve_query_with_io_error_returns_io_variant() {
    let result = resolve_query_with(None, &mut FailingReader, false);
    let err = result.unwrap_err();
    assert_matches!(err, QueryError::Io(_));
    assert!(err.to_string().contains("failed to read from stdin"));
}

// T-712: resolve_search_query_with — a literal --from query is used as-is
#[test]
fn resolve_search_query_with_from_literal_used_as_is() {
    let mut stdin = Cursor::new(b"ignored");
    let result = resolve_search_query_with(Some("auth".into()), true, &mut stdin, false);
    assert_eq!(result.unwrap(), Some("auth".to_owned()));
}

// T-713: resolve_search_query_with — a missing --from query is optional (None)
#[test]
fn resolve_search_query_with_from_missing_returns_none() {
    let mut stdin = Cursor::new(b"");
    let result = resolve_search_query_with(None, true, &mut stdin, true);
    assert_eq!(result.unwrap(), None);
}

// T-714: resolve_search_query_with — "-" under --from reads stdin
#[test]
fn resolve_search_query_with_from_dash_reads_stdin() {
    let mut stdin = Cursor::new(b"piped");
    let result = resolve_search_query_with(Some("-".into()), true, &mut stdin, false);
    assert_eq!(result.unwrap(), Some("piped".to_owned()));
}

// T-715: resolve_search_query_with — a query without --from resolves to Some
#[test]
fn resolve_search_query_with_no_from_present_returns_some() {
    let mut stdin = Cursor::new(b"");
    let result = resolve_search_query_with(Some("hooks".into()), false, &mut stdin, true);
    assert_eq!(result.unwrap(), Some("hooks".to_owned()));
}

// T-716: resolve_search_query_with — a missing query without --from is an error
#[test]
fn resolve_search_query_with_no_from_missing_is_error() {
    let mut stdin = Cursor::new(b"");
    let result = resolve_search_query_with(None, false, &mut stdin, true);
    assert_matches!(
        result.unwrap_err(),
        QueryError::NoQuery(NoQueryReason::Terminal)
    );
}

// T-717: resolve_search_query_with — a stdin I/O failure propagates as Io
#[test]
fn resolve_search_query_with_io_error_propagates() {
    let result = resolve_search_query_with(None, false, &mut FailingReader, false);
    assert_matches!(result.unwrap_err(), QueryError::Io(_));
}

// T-718: resolve_search_query_with — no --from + empty piped stdin is EmptyStdin
#[test]
fn resolve_search_query_with_no_from_empty_stdin_is_empty_stdin() {
    let mut stdin = Cursor::new(b"   ");
    let result = resolve_search_query_with(None, false, &mut stdin, false);
    assert_matches!(
        result.unwrap_err(),
        QueryError::NoQuery(NoQueryReason::EmptyStdin)
    );
}

// T-719: missing_query_kind maps each NoQueryReason to its input error
#[test]
fn missing_query_kind_maps_each_reason() {
    assert_matches!(
        missing_query_kind(&NoQueryReason::Terminal),
        InvalidInputKind::QueryOrFromRequired
    );
    assert_matches!(
        missing_query_kind(&NoQueryReason::EmptyStdin),
        InvalidInputKind::EmptyQuery
    );
}
