# yomu

Frontend code search for AI agents. One call instead of six.

## What it does

An AI agent asked to find "the chat streaming hook" in [vercel/ai](https://github.com/vercel/ai) (3,535 files):

**Without yomu** — 6+ tool calls, each consuming context window:

```
grep "useChat"    → 130 files. Which one is the implementation?
grep "stream"     → 885 files. Worse.
grep "chat hook"  → 0 files. Name unknown.
glob "**/chat*"   → Files found, contents unknown.
read use-chat.ts  → Got the code. What does it import?
read 2 more files → Now I have context.
```

**With yomu** — 1 tool call, everything included:

```
explorer("streaming chat hooks")

  useChat [hook] — packages/react/src/use-chat.ts:58
  │ code:     Full function body (not a single-line match)
  │ imports:  import { processUIMessageStream } from '@ai-sdk/ui-utils'
  │ siblings: UseChatOptions [type_def], UseChatHelpers [type_def]
  │
  useStreamableValue [hook] — packages/rsc/src/streamable-value/...
  │ ...
```

No name guessing. No follow-up reads. The agent gets code, imports, and related definitions in one response.

## Why this matters for agents

AI agents pay for every tool call in two ways: **latency** and **context window consumption**. Each `grep` → `read` → `read` chain adds thousands of tokens of intermediate results to the conversation. Most of those tokens are noise — the 129 test files that matched `useChat`, the file listing that didn't show contents.

yomu returns only what the agent needs: the relevant code with its dependencies. This means:

- **Fewer tool calls** — 1 instead of 5-7 for concept-based search
- **Less context waste** — No grep noise, no exploratory reads
- **No name required** — Search by what code does, not what it's called

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

### `explorer` — Find code by concept

The primary tool. Returns ranked results with full context:

| Included       | Why                                             |
| -------------- | ----------------------------------------------- |
| Full code body | No follow-up `read` needed                      |
| File imports   | Dependency context without opening another file |
| Sibling defs   | Other functions/types in the same file          |
| Chunk type     | component / hook / type_def / css_rule          |

### `impact` — Blast radius of a change

Shows which files depend on a target file or symbol. Real output from vercel/ai:

```
> impact("packages/ai/src/ui/ui-messages.ts:UIMessage", depth=2)

  Direct symbol references: 34 files
  Depth 1: 37 files
  Depth 2: 18 files
  Total: 55 dependent files
```

One call replaces manually tracing imports across dozens of files.

### `index` / `rebuild` / `status`

- `index` — Update the chunk index incrementally. No API calls, ~2.5s on 3,535 files. Usually not needed since `explorer` auto-indexes.
- `rebuild` — Full re-parse from scratch.
- `status` — Files, chunks, embedding coverage, references.

## How it works

```
Source files → tree-sitter AST → Semantic chunks → Gemini embeddings → Hybrid search
```

**Indexing** — tree-sitter splits code at function/component/type boundaries. Each chunk is one searchable unit. Import graph is built in the same pass. On vercel/ai: 3,535 files → 9,015 chunks + 5,026 import references in 2.5s, zero API calls.

**Embedding** — Chunks are embedded incrementally via Gemini (free tier). 50 chunks per `explorer` call, prioritized by import count — the most-used code gets searchable first.

**Search** — Three-tier hybrid: vector similarity → name/path matching → FTS5 full-text. Results are reranked with IDF-weighted keyword scoring. Frequently-imported files rank higher, test/example files are pushed down.

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
- **Cold start** — First `explorer` call takes a few seconds for chunking + initial embedding. Subsequent calls are faster.
- **Large files skipped** — Files over 1 MB are excluded from indexing.

## Architecture

```
src/
├── main.rs            MCP server entry point (stdio transport)
├── tools/             MCP tool handlers (explorer, index, rebuild, impact, status)
├── indexer/           Chunking, file walking, Gemini embedding client
├── resolver.rs        Import path resolution (tsconfig aliases, index.ts probing)
├── query.rs           Hybrid search + IDF reranking
└── storage/           SQLite + sqlite-vec (search, import graph, CRUD)
```

Single binary, zero runtime dependencies. SQLite and sqlite-vec are statically linked.

## License

Apache-2.0
