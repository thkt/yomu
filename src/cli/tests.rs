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
