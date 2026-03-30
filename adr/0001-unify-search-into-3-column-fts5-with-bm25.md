# Unify search into 3-column FTS5 with bm25

- Status: accepted
- Deciders: thkt
- Date: 2026-03-30
- Confidence: medium — DA challenge passed with revisions; bm25 weight tuning needs production validation
- Refs: #37, #36 (closed)

## Context

`search_by_name` uses `LIKE '%keyword%'` on `chunks.name` and `chunks.file_path`, resulting in O(N) full table scans on every search request. `search_by_content` already uses FTS5 MATCH at O(log N). The two search paths run sequentially with exclude_ids coordination, and reranking uses separate `MatchSource::NameMatch` / `ContentMatch` with manually tuned base score constants.

## Decision Drivers

- Every search request executes the O(N) LIKE scan before FTS, becoming a bottleneck as chunk count grows
- Two search functions + 6 scoring constants + exclude_ids coordination add maintenance surface
- FTS5 `bm25()` provides principled cross-column ranking that replaces manual score tuning

## Considered Options

### A: Add name to FTS, keep file_path as LIKE

Add name column to FTS table. Use `{name}: keyword` column filter for name search. Keep file_path as LIKE.

- Good: minimal change, reranking logic preserved
- Bad: file_path remains O(N) LIKE
- Bad: two FTS queries per search (name + content separately)

### B: Unified 3-column FTS (name, content, file_path)

Replace both `search_by_name` and `search_by_content` with a single FTS query across 3 columns. Use `bm25()` column weights for ranking.

- Good: eliminates all LIKE scans
- Good: single FTS query replaces two search functions
- Good: net code reduction (2 functions, 6 constants, escape_like removed)
- Bad: loses arbitrary substring matching (e.g., `etch` no longer matches `fetch`)
- Bad: bm25 weights need tuning via testing
- Bad: broader test modifications due to MatchSource unification

## Decision

Option B. Merge `search_by_name` and `search_by_content` into `search_by_fts` with `fts5(name, content, file_path)` and `bm25()` ranking.

Key design choices:

- **name pre-processing**: `split_identifier(name).join(" ")` at INSERT time to handle camelCase/PascalCase (Unicode61 treats `useAuth` as one token; splitting produces `use Auth` which matches `auth`)
- **file_path**: stored as-is; Unicode61 tokenizes on `/`, `.`, `-` naturally
- **FTS query building**: reuse `prepare_match_query` (from rurico) for prefix expansion via `fts_chunks_vocab`
- **MatchSource**: `Semantic | Fts` (collapse NameMatch + ContentMatch)
- **FTS base score**: `1.0 / (1.0 + bm25.abs())`; type_bonus / import_bonus / test_penalty preserved as additive modifiers
- **keyword_hit_ratio**: retained for Semantic results only
- **Migration v6->v7**: DROP + recreate `fts_chunks` and `fts_chunks_vocab`, re-populate with split names

### Positive Consequences

- All search paths are O(log N) via FTS5
- Net code reduction: `search_by_name`, `search_by_content`, `escape_like`, 6 scoring constants removed
- `bm25()` provides corpus-aware ranking without manual constant tuning
- Single FTS query eliminates exclude_ids coordination between search steps

### Negative Consequences

- Arbitrary substring match lost: token-level matching only (accepted as spec change)
- bm25 score normalization may need revision if ranking degrades at extreme corpus sizes
- Existing tests for `search_by_name` and `search_by_content` require rewrite

## Architecture

```
Before:
  semantic -> search_by_name (LIKE O(N)) -> search_by_content (FTS) -> rerank

After:
  semantic -> search_by_fts (FTS bm25) -> rerank
```

## Quality Attributes

| Attribute       | Priority | Approach                                     |
| --------------- | -------- | -------------------------------------------- |
| Performance     | high     | O(N) LIKE eliminated, single FTS query       |
| Maintainability | high     | 2 functions collapsed, 6 constants removed   |
| Search quality  | medium   | bm25 column weights; needs tuning validation |

## Trade-offs

- Token-level matching precision in exchange for O(log N) performance on all search paths
- Manual scoring control in exchange for bm25's corpus-aware ranking

## Reassessment Triggers

- bm25 ranking quality degrades noticeably at >50K chunks
- Users report missing results that LIKE substring match would have found
- Need for CJK search (may require trigram tokenizer, revisit #36)
