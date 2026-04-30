**English** | [日本語](README.ja.md)

# yomu

Frontend code search for AI agents. Find code by concept when you don't know the name.

## The problem

You need to find the chat streaming hook in [vercel/ai](https://github.com/vercel/ai) (3,535 files), but you don't know the function is called `useChat`.

Typical agent workflow:

```sh
glob "**/chat*"                                      → 12 files. None are it (it's called use-chat.ts).
grep "stream.*hook"                                  → 0 files.
grep "chat"                                          → too many results. Try reading a few...
read packages/react/src/use-chat.ts                  → Found it. What does it import?
read packages/ai/src/ui/process-ui-message-stream.ts → Now I have context.
```

3-5 tool calls, trial and error, noise in the context window.

With yomu:

```sh
yomu search "streaming chat hooks"

## packages/react/src/use-chat.ts
Imports: @ai-sdk/provider-utils, @ai-sdk/ui-utils
Siblings: UseChatOptions [type_def], UseChatHelpers [type_def]

1. useChat [hook] — 58:210 (similarity: 0.85)
export function useChat({ api, ...options }: UseChatOptions): UseChatHelpers {
  ...full function body...
}

## packages/rsc/src/streamable-value/use-streamable-value.ts
2. useStreamableValue [hook] — 12:45 (similarity: 0.72)
...

## examples/ai-e2e-next/.../chat-context.tsx
3. useSharedChatContext [hook] — 8:22 (similarity: 0.68)
...
```

1 call. The implementation is the first result — out of 130 files that contain "useChat", and 9,015 total chunks in the index.

Each result includes the full code body, file imports, and sibling definitions. No follow-up reads needed.

## Why not just grep?

[Claude Code's developers found](https://zenn.dev/acntechjp/articles/c1296f425baf03) that agentic search — letting the model use glob and grep iteratively — outperformed RAG for code navigation. They're right. When the agent can retry with different keywords, read directory structures, and refine its search, grep works remarkably well.

yomu doesn't compete with that workflow. It **reduces the iterations**:

| Approach              | Calls | Context window cost                       |
| --------------------- | ----- | ----------------------------------------- |
| grep/glob (iterative) | 3–5   | Each miss adds noise                      |
| yomu search           | 1     | Code + imports + siblings in one response |

The classic RAG problems — index sync lag, stale embeddings, cold starts — are addressed:

| Solution              | Details                                                                            |
| --------------------- | ---------------------------------------------------------------------------------- |
| No sync lag           | Every `search` checks index freshness and re-chunks automatically if files changed |
| No API key required   | Local embedding model, FTS5 full-text fallback when model unavailable              |
| Incremental embedding | 50 chunks per search call, most-imported files first. No upfront build             |

grep is the right tool when you know the name. yomu is for the moment before that — when you know the concept but not the identifier.

## When to use yomu (and when not to)

| yomu                                                | grep/glob                                            |
| --------------------------------------------------- | ---------------------------------------------------- |
| You don't know what the code is called              | You know the exact name (`grep "useAuth"` is faster) |
| grep returns too many results — you want the impl   | You need regex matching or exact string search       |
| You want code + imports + related types in one call | The codebase is small and familiar                   |

yomu doesn't replace grep. It covers the case grep can't: searching by concept.

## Setup

### Install

```sh
brew install thkt/tap/yomu
```

Or build from source (requires Rust 1.85+):

```sh
cargo build --release
```

### Configure

Semantic search uses a local embedding model ([Ruri v3](https://huggingface.co/cl-nagoya/ruri-v3-310m), ~1.2 GB). If the model is already cached locally, `search` uses it automatically — no API key required.

If the model is not installed or unavailable, `search` falls back to text-only mode (FTS5). All other commands (`index`, `rebuild`, `impact`, `status`) work without the model.

No manual indexing. `search` auto-indexes on first call.

#### Platform notes

| Platform              | Build command                                                   |
| --------------------- | --------------------------------------------------------------- |
| macOS (Apple Silicon) | `cargo build --release` (default: mlx backend)                  |
| Linux / x86           | `cargo build --release --no-default-features --features candle` |

## Commands

### Global flags

| Flag     | Description                   |
| -------- | ----------------------------- |
| `--json` | Output as JSON (all commands) |

`--json` can appear before or after the subcommand:

```sh
$ yomu --json status
{"files":42,"chunks":187,"embedded_chunks":187,"embeddable_chunks":187,"embed_percentage":100,"references":156,"last_indexed":"2025-03-29 01:23:45"}
```

### `yomu search [query]` — Search by concept

Returns ranked results with full context. Each result includes:

| Included       | Why                                              |
| -------------- | ------------------------------------------------ |
| Full code body | No follow-up `read` needed                       |
| File imports   | Dependency context without opening another file  |
| Sibling defs   | Other functions/types in the same file           |
| Chunk type     | component / hook / type_def / css_rule / rust_fn |

Options:

| Flag         | Default | Description                                                              |
| ------------ | ------- | ------------------------------------------------------------------------ |
| `--limit`    | 10      | Max results (max: 100)                                                   |
| `--offset`   | 0       | Pagination offset (max: 500)                                             |
| `--from`     | —       | Search for code similar to a file or symbol (`src/foo.rs` or `src/foo.rs:my_fn`). Query becomes optional |
| `--no-embed` | false   | Skip embedding lookups; use FTS5 only. Same effect as `YOMU_EMBED=0`     |

`--from` uses the stored embeddings of the target — no re-embedding needed:

```sh
yomu search --from src/query/mod.rs           # files similar to this file
yomu search --from src/query/mod.rs:rerank    # files similar to this function
yomu search --from src/query/mod.rs "filter"  # hybrid: similar to file + FTS on "filter"
```

### `yomu impact <target>` — Blast radius of a change

Shows which files depend on a target file or symbol.

```sh
$ yomu impact "packages/ai/src/ui/ui-messages.ts" --symbol UIMessage --depth 2

## Impact analysis: `packages/ai/src/ui/ui-messages.ts`

### Direct symbol references
- packages/ai/src/ui/process-ui-message-stream.ts
- packages/react/src/use-chat.ts
  ...34 files

### All transitive dependents
#### Depth 1
- packages/ai/src/ui/process-ui-message-stream.ts
  ...37 files
#### Depth 2
- packages/react/src/use-chat.ts
  ...18 files

Total: 55 dependent file(s)
```

Real output from vercel/ai. One call replaces manually tracing imports.

Options: `--symbol` (optional, filter to specific export), `--depth` (default: 3, max: 10)

`--json` returns a structured response. `dependents[].references` is populated for direct (depth=1) edges and lists every `(ref_kind, via_symbol)` pair from the source file. Transitive (depth>=2) dependents reach the target through intermediate files, so their `references` is `[]`:

```json
{
  "target": "src/storage.rs",
  "in_index": true,
  "dependents": [
    {
      "file_path": "src/indexer.rs",
      "depth": 1,
      "references": [
        {"ref_kind": "named", "via_symbol": "Db"},
        {"ref_kind": "named", "via_symbol": "open_db"}
      ]
    },
    {
      "file_path": "src/main.rs",
      "depth": 2,
      "references": []
    }
  ],
  "symbol_refs": [],
  "total": 2
}
```

`ref_kind` is one of `named` / `default` / `namespace` / `type_only` / `side_effect`. `via_symbol` is `null` for namespace/side-effect imports where no individual symbol is named at the import site.

### `yomu index` / `yomu rebuild` / `yomu status`

| Command   | Details                                                                                                |
| --------- | ------------------------------------------------------------------------------------------------------ |
| `index`   | Update the chunk index. No API calls, ~2.5s on 3,535 files. Usually not needed — `search` auto-indexes |
| `rebuild` | Full re-parse from scratch                                                                             |
| `status`  | Files, chunks, embedding coverage, references                                                          |

## How it works

```text
Source files → tree-sitter AST → Semantic chunks → Local embeddings (Ruri v3) → Hybrid search
```

| Stage     | Details                                                                                                                                                                                                                                  |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Indexing  | tree-sitter splits code at function/component/type boundaries. Each chunk is one searchable unit. The import graph is built in the same pass. On vercel/ai: 3,535 files → 9,015 chunks + 5,026 import references in 2.5s, zero API calls |
| Embedding | Chunks are embedded incrementally via a local model (Ruri v3, 310M params). 50 chunks per `search` call, prioritized by import count — the most-used code gets searchable first. No upfront build required                               |
| Search    | Three-tier hybrid: vector similarity → name/path matching → FTS5 full-text. Reranked with IDF-weighted keyword scoring. Frequently-imported files rank higher, test files are pushed down                                                |

## Supported file types

| Type             | Parser      | Chunk types                                            |
| ---------------- | ----------- | ------------------------------------------------------ |
| TypeScript / TSX | tree-sitter | component, hook, type_def, test_case, other            |
| JavaScript / JSX | tree-sitter | component, hook, type_def, test_case, other            |
| Rust             | tree-sitter | rust_fn, rust_struct, rust_enum, rust_trait, rust_impl |
| CSS              | tree-sitter | css_rule (selectors, @media, @keyframes)               |
| HTML             | tree-sitter | html_element                                           |

Other files fall back to character-based chunking with overlap.

## Limitations

| Limitation                | Details                                                                                     |
| ------------------------- | ------------------------------------------------------------------------------------------- |
| Model not auto-downloaded | The ~1.2 GB embedding model must be pre-cached; `search` does not download it automatically |
| SCSS/Sass not supported   | Only plain CSS                                                                              |
| Cold start                | First `search` call takes a few seconds for chunking + initial embedding                    |
| Large files skipped       | Files over 1 MB are excluded from indexing                                                  |
| Embedding opt-out         | Pass `--no-embed` to `yomu search`, or set `YOMU_EMBED=0`; `search` falls back to text-only mode |

## Development

### Setup

Run once after cloning:

```sh
git config --local core.hooksPath .githooks
```

This installs a pre-commit hook that runs `cargo fmt --check` and `cargo clippy --all-targets --all-features -- -D warnings` before each commit. Violations abort the commit. To skip for one commit: `git commit --no-verify`.

### Common commands

```sh
cargo test                                                # all tests
cargo clippy --all-targets --all-features -- -D warnings  # lint (matches CI)
cargo fmt -- --check                                      # format check
```

## Architecture

```text
src/
├── main.rs              CLI entry point (clap)
├── lib.rs               Crate root, public API
├── config.rs            Runtime configuration
├── modernbert.rs        ModernBERT model (mlx backend)
├── tools/               Application facade — orchestrates indexer, query, storage per command
├── indexer/
│   ├── mod.rs           Orchestration: incremental index, embed budget
│   ├── chunker/         tree-sitter AST → semantic chunks
│   ├── embedder.rs      Local embedding (ModernBERT/Ruri v3 via mlx-rs or candle)
│   └── walker.rs        File discovery, .gitignore filtering
├── resolver.rs          Import path resolution (tsconfig aliases, index.ts probing)
├── query/               Hybrid search + IDF reranking
└── storage/
    ├── mod.rs           Schema, CRUD, types
    ├── search.rs        Vector similarity, name matching, FTS5
    ├── embed.rs         Embedding storage, vec_chunks
    └── graph.rs         Import graph traversal, dependents, siblings
```

Single binary, zero runtime dependencies. SQLite and sqlite-vec are statically linked.

## License

MIT
