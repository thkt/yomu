# yomu

Search frontend code by what it does, not what it's named. An MCP server for AI agents.

## The problem

Searching [vercel/ai](https://github.com/vercel/ai) (1,824 files) for the chat streaming hook:

```
Grep "useChat"         → 130 files — 125 are tests, examples, codemods. 5 are source.
Grep "stream"          → 882 files
Grep "chat hook"       → 0 files

Which of those 130 files is the actual implementation?
```

With yomu:

```
explorer("streaming chat hooks")

  → useChat [hook]           — packages/react/src/use-chat.ts:58  (0.72)
  → UseChatOptions [type_def] — packages/react/src/use-chat.ts:42  (0.68)
  → UseChatHelpers [type_def] — packages/react/src/use-chat.ts:12  (0.68)
  + full code, imports, siblings for each result
```

One call. Source files at the top, 125 test/example files ranked below.

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

You need a [Gemini API key](https://aistudio.google.com/apikey) (free tier works).

Add to your MCP client config (e.g. Claude Code `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "yomu": {
      "command": "yomu",
      "env": {
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

yomu auto-detects the project root from `.yomu/` or `.git/` upward from the working directory. Index is stored at `{project_root}/.yomu/index.db`. No manual indexing step — `explorer` auto-indexes on first call.

## How it works

```
Source files → tree-sitter AST → Semantic chunks → Gemini embeddings → Hybrid search
```

1. **AST-aware chunking** — Code is split at function, component, type, and CSS rule boundaries. A React component stays as one searchable unit, not split across arbitrary lines.
2. **Semantic embedding** — Each chunk is embedded via Gemini (`gemini-embedding-001`, 768 dims, free tier). Code that does similar things gets similar vectors, regardless of naming.
3. **Hybrid search + rerank** — Vector similarity first, then name-based fallback. Results are reranked: query type hints boost matching chunks, frequently-imported files rank higher, test/example files are pushed down.

### Incremental indexing

The index grows through usage. No full upfront build required.

```
First explorer call on vercel/ai (1,824 files):

  1. Chunk-only index    1,824 files → 6,889 chunks    1.7s   zero API calls
  2. Incremental embed   Top 5 most-imported files      ~3s    43 chunks embedded
  3. Search              Results from embedded chunks    instant

Each subsequent explorer call:
  → Embeds 50 more chunks, prioritized by import count
  → Search coverage expands automatically
```

Embedding priority is based on import count, so the most-used API surface gets searchable first. Queries like "auth hooks" further prioritize files containing matching chunk types.

## Tools

### `explorer`

Semantic code search. The primary tool. Each result includes:

- **Full code chunk** — The complete function/component, not a single line
- **Imports** — File dependencies, visible without opening the file
- **Siblings** — Other functions and types in the same file
- **chunk_type** — component / hook / type_def / css_rule

`grep "useChat"` returns 130 file paths with single-line matches. `explorer("streaming chat hooks")` returns the implementation with full code and context.

### `impact`

Import graph analysis. Shows which files depend on a target file or symbol.

Real output from vercel/ai — querying the `UIMessage` type at depth 2:

```
> impact({ "target": "packages/ai/src/ui/ui-messages.ts:UIMessage", "depth": 2 })

### Direct symbol references (34 files)
- packages/ai/src/ui/chat.ts
- packages/ai/src/ui/chat-transport.ts
- packages/ai/src/ui/process-ui-message-stream.ts
- packages/ai/src/agent/create-agent-ui-stream.ts
  ... 30 more

### All transitive dependents
#### Depth 1 — 37 files
#### Depth 2 — 18 files

Total: 55 dependent file(s)
```

The full transitive blast radius of a change, computed from the import graph built during indexing. Useful for scoping refactoring or estimating PR impact.

### `index`

Builds or updates the full index. Use when you want 100% search coverage upfront rather than incremental.

```json
{ "force": false }
```

### `status`

Index statistics: files, chunks, embedding coverage percentage, references, last update time.

## Supported file types

| Type             | Parser      | Chunk types                                 |
| ---------------- | ----------- | ------------------------------------------- |
| TypeScript / TSX | tree-sitter | component, hook, type_def, test_case, other |
| JavaScript / JSX | tree-sitter | component, hook, type_def, test_case, other |
| CSS              | tree-sitter | css_rule (selectors, @media, @keyframes)    |
| HTML             | tree-sitter | html_element                                |

Other files fall back to character-based chunking with overlap.

## When to use yomu

**yomu is for when:**

- You don't know what things are named in this codebase
- You want to find code by concept ("error boundary", "auth flow", "data fetching layer")
- You need to discover related components/hooks/types across files
- The codebase is large enough that grep returns too many results

**Use grep/glob instead when:**

- You know the exact string, class name, or function name
- You need regex pattern matching
- The codebase is small and well-known

## Limitations

- **Gemini free-tier rate limits** — 100 requests/min, 1,500/day. Incremental indexing works within these limits, but full `index` on a large project (1,000+ files) may hit the daily cap.
- **SCSS/Sass not supported** — Only plain CSS via tree-sitter.
- **Embedding quality depends on Gemini** — Highly project-specific abbreviations or domain jargon may not match well against natural language queries.
- **Cold start on first search** — The initial `explorer` call takes a few seconds for chunking + first batch embedding. Subsequent calls are faster.
- **Large files skipped** — Files over 1 MB are excluded from indexing. This covers virtually all hand-written frontend source; generated bundles or large SVG-in-TSX may be skipped.

## Architecture

```
src/
├── main.rs          MCP server entry point (stdio transport)
├── lib.rs           Library root
├── config.rs        Project root detection
├── tools/mod.rs     MCP tool handlers (explorer, index, impact, status)
├── indexer/
│   ├── mod.rs       Indexing orchestration + incremental embed
│   ├── walker.rs    File discovery (respects .gitignore patterns)
│   ├── chunker/     tree-sitter AST chunking + import parsing
│   └── embedder.rs  Gemini embedding API client
├── resolver.rs      Import path resolution (tsconfig aliases, index.ts probing)
├── query.rs         Hybrid search (vector + name fallback + rerank)
└── storage/mod.rs   SQLite + sqlite-vec storage layer
```

Single binary, zero runtime dependencies. SQLite and sqlite-vec are statically linked.

## License

Apache-2.0
