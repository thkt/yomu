# yomu

[日本語](README.ja.md)

Frontend code search for AI agents. Find code by concept when you don't know the name.

## The problem

An AI agent needs to find the chat streaming hook in [vercel/ai](https://github.com/vercel/ai) (3,535 files). It doesn't know the function is called `useChat`.

**Typical agent workflow:**

```
glob "**/chat*"       → 12 files. None are it (it's called use-chat.ts).
grep "stream.*hook"   → 0 files.
grep "chat"           → too many results. Try reading a few...
read packages/react/src/use-chat.ts  → Found it. What does it import?
read packages/ai/src/ui/process-ui-message-stream.ts → Now I have context.
```

3-5 tool calls, trial and error, noise in the context window.

**With yomu:**

```
explorer("streaming chat hooks")

  #1  useChat [hook] — packages/react/src/use-chat.ts:58
      Full function body, imports, sibling type definitions included.

  #2  useStreamableValue [hook] — packages/rsc/src/streamable-value/...
  #3  useSharedChatContext [hook] — examples/ai-e2e-next/.../chat-context.tsx
```

1 call. The implementation is the first result — out of 130 files that contain "useChat", and 9,015 total chunks in the index.

Each result includes the full code body, file imports, and sibling definitions. No follow-up reads needed.

## When to use yomu (and when not to)

**Use yomu when:**

- You don't know what the code is called — "error boundary logic", "data fetching layer"
- grep returns too many results and you need the implementation, not the 125 test files
- You want code + imports + related types in one response

**Use grep/glob when:**

- You know the exact name — `grep "useAuth"` is faster and more precise
- You need regex matching or exact string search
- The codebase is small and familiar

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

You need a [Gemini API key](https://aistudio.google.com/apikey) (free tier works).

Add to your MCP client config (e.g. Claude Code `~/.claude/.mcp.json`):

```json
{
  "mcpServers": {
    "yomu": {
      "command": "yomu",
      "env": {
        "GEMINI_API_KEY": "${GEMINI_API_KEY}"
      }
    }
  }
}
```

No manual indexing. `explorer` auto-indexes on first call.

## Tools

### `explorer` — Search by concept

Returns ranked results with full context. Each result includes:

| Included       | Why                                             |
| -------------- | ----------------------------------------------- |
| Full code body | No follow-up `read` needed                      |
| File imports   | Dependency context without opening another file |
| Sibling defs   | Other functions/types in the same file          |
| Chunk type     | component / hook / type_def / css_rule          |

### `impact` — Blast radius of a change

Shows which files depend on a target file or symbol.

```
> impact("packages/ai/src/ui/ui-messages.ts:UIMessage", depth=2)

  Direct symbol references: 34 files
  Depth 1: 37 files
  Depth 2: 18 files
  Total: 55 dependent files
```

Real output from vercel/ai. One call replaces manually tracing imports.

### `index` / `rebuild` / `status`

- `index` — Update the chunk index. No API calls, ~2.5s on 3,535 files. Usually not needed — `explorer` auto-indexes.
- `rebuild` — Full re-parse from scratch.
- `status` — Files, chunks, embedding coverage, references.

## How it works

```
Source files → tree-sitter AST → Semantic chunks → Gemini embeddings → Hybrid search
```

**Indexing** — tree-sitter splits code at function/component/type boundaries. Each chunk is one searchable unit. The import graph is built in the same pass. On vercel/ai: 3,535 files → 9,015 chunks + 5,026 import references in 2.5s, zero API calls.

**Embedding** — Chunks are embedded incrementally via Gemini (free tier). 50 chunks per `explorer` call, prioritized by import count — the most-used code gets searchable first. No upfront build required.

**Search** — Three-tier hybrid: vector similarity → name/path matching → FTS5 full-text. Results are reranked with IDF-weighted keyword scoring. Frequently-imported files rank higher, test files are pushed down.

## Supported file types

| Type             | Parser      | Chunk types                                 |
| ---------------- | ----------- | ------------------------------------------- |
| TypeScript / TSX | tree-sitter | component, hook, type_def, test_case, other |
| JavaScript / JSX | tree-sitter | component, hook, type_def, test_case, other |
| CSS              | tree-sitter | css_rule (selectors, @media, @keyframes)    |
| HTML             | tree-sitter | html_element                                |

Other files fall back to character-based chunking with overlap.

## Limitations

- **Gemini free-tier rate limits** — 100 RPM, 1,500 requests/day. Heavy usage across projects can exhaust the daily quota (resets midnight Pacific Time).
- **SCSS/Sass not supported** — Only plain CSS.
- **Cold start** — First `explorer` call takes a few seconds for chunking + initial embedding.
- **Large files skipped** — Files over 1 MB are excluded from indexing.

## Architecture

```
src/
├── main.rs              MCP server (stdio transport)
├── lib.rs               Crate root, public API
├── config.rs            Runtime configuration
├── tools/               MCP tool handlers (explorer, index, rebuild, impact, status)
├── indexer/
│   ├── mod.rs           Orchestration: incremental index, embed budget
│   ├── chunker/         tree-sitter AST → semantic chunks
│   ├── embedder.rs      Gemini embedding client (batchEmbedContents)
│   └── walker.rs        File discovery, .gitignore filtering
├── resolver.rs          Import path resolution (tsconfig aliases, index.ts probing)
├── query.rs             Hybrid search + IDF reranking
└── storage/
    ├── mod.rs           Schema, CRUD, types
    ├── search.rs        Vector similarity, name matching, FTS5
    └── graph.rs         Import graph traversal, dependents, siblings
```

Single binary, zero runtime dependencies. SQLite and sqlite-vec are statically linked.

## License

Apache-2.0
