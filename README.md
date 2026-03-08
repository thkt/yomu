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

- **No sync lag** — Every `search` checks index freshness and re-chunks automatically if files changed
- **No embedding required** — FTS5 full-text fallback works without any API key
- **Incremental embedding** — 50 chunks per search call, most-imported files first. No upfront build

grep is the right tool when you know the name. yomu is for the moment before that — when you know the concept but not the identifier.

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

For semantic search, set a [Gemini API key](https://aistudio.google.com/apikey) (free tier works):

```sh
export GEMINI_API_KEY="your-key-here"
```

Without an API key, `search` falls back to text-only mode (FTS5). All other commands (`index`, `rebuild`, `impact`, `status`) work without a key.

No manual indexing. `search` auto-indexes on first call.

## Commands

### `yomu search <query>` — Search by concept

Returns ranked results with full context. Each result includes:

| Included       | Why                                             |
| -------------- | ----------------------------------------------- |
| Full code body | No follow-up `read` needed                      |
| File imports   | Dependency context without opening another file |
| Sibling defs   | Other functions/types in the same file          |
| Chunk type     | component / hook / type_def / css_rule          |

Options: `--limit` (default: 10, max: 100), `--offset` (default: 0, max: 500)

### `yomu impact <target>` — Blast radius of a change

Shows which files depend on a target file or symbol.

```
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

Options: `--symbol` (filter to specific export), `--depth` (default: 3, max: 10)

### `yomu index` / `yomu rebuild` / `yomu status`

- `index` — Update the chunk index. No API calls, ~2.5s on 3,535 files. Usually not needed — `search` auto-indexes.
- `rebuild` — Full re-parse from scratch.
- `status` — Files, chunks, embedding coverage, references.

## How it works

```
Source files → tree-sitter AST → Semantic chunks → Gemini embeddings → Hybrid search
```

**Indexing** — tree-sitter splits code at function/component/type boundaries. Each chunk is one searchable unit. The import graph is built in the same pass. On vercel/ai: 3,535 files → 9,015 chunks + 5,026 import references in 2.5s, zero API calls.

**Embedding** — Chunks are embedded incrementally via Gemini (free tier). 50 chunks per `search` call, prioritized by import count — the most-used code gets searchable first. No upfront build required.

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
- **Cold start** — First `search` call takes a few seconds for chunking + initial embedding.
- **Large files skipped** — Files over 1 MB are excluded from indexing.

## Architecture

```
src/
├── main.rs              CLI entry point (clap)
├── lib.rs               Crate root, public API
├── config.rs            Runtime configuration
├── tools/               Application facade — orchestrates indexer, query, storage per command
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

MIT
