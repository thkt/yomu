use std::collections::{HashMap, HashSet};

use amici::storage::filter::{
    append_exclude_ids, append_in_filter, append_include_ids, append_like_prefix_filter,
};
use bytemuck::cast_slice;
use rurico::storage::{fts_quote, prepare_match_query};
use rusqlite::{Connection, Error as RusqliteError, Row, params, params_from_iter, types::ToSql};

use super::{
    Chunk, ChunkType, MatchSource, SearchResult, StorageError, anon_placeholders,
    fts_normalization, in_placeholders,
};

const VEC_MAXSIM_OVERSAMPLE: u32 = 10;

/// Hard cap on candidates passed to a single `WHERE id IN (?, ...)` query.
/// SQLite's default `SQLITE_MAX_VARIABLE_NUMBER` is 32766; we leave headroom
/// for filter params.
const MAX_VEC_CANDIDATES: usize = 30_000;

/// Hard cap on keywords for the bulk UNION ALL doc-frequency query. Above
/// this we skip the bulk path entirely so we do not build SQL that SQLite
/// will refuse to compile (`SQLITE_MAX_COMPOUND_SELECT` default is 500).
const MAX_BULK_DOC_FREQ_KEYWORDS: usize = 500;

/// KNN-only query over `vec_chunks`. Returns `(chunk_id, distance)` pairs
/// ordered by ascending distance. Shared by [`vec_search`] (single embedding)
/// and [`vec_search_multi`] (multiple embeddings sharing one metadata fetch).
fn knn_only(conn: &Connection, embedding: &[f32], k: u32) -> Result<Vec<(i64, f32)>, StorageError> {
    let query_bytes = cast_slice::<f32, u8>(embedding);
    let mut stmt = conn.prepare_cached(
        "SELECT chunk_id, distance FROM vec_chunks \
         WHERE embedding MATCH ?1 AND k = ?2 \
         ORDER BY distance",
    )?;
    let rows = stmt.query_map(params![query_bytes, k], |row| {
        Ok((row.get(0)?, row.get(1)?))
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

/// Assemble [`SearchResult`]s from a `chunk_id → min distance` map and a
/// filtered metadata map. Entries missing from `meta` (filtered out) are
/// dropped. Result is sorted by ascending distance and truncated to `limit`.
fn build_semantic_results(
    best: HashMap<i64, f32>,
    meta: &HashMap<i64, Chunk>,
    limit: u32,
) -> Vec<SearchResult> {
    let mut results: Vec<SearchResult> = best
        .into_iter()
        .filter_map(|(chunk_id, distance)| {
            meta.get(&chunk_id).map(|chunk| SearchResult {
                chunk: chunk.clone(),
                chunk_id: Some(chunk_id),
                distance,
                match_source: MatchSource::Semantic,
                score: 1.0 / (1.0 + distance),
            })
        })
        .collect();
    results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    results.truncate(limit as usize);
    results
}

fn chunk_from_row(row: &Row<'_>, offset: usize) -> rusqlite::Result<Chunk> {
    Ok(Chunk {
        file_path: row.get(offset)?,
        chunk_type: ChunkType::from_db(row.get::<_, String>(offset + 1)?.as_ref()),
        name: row.get(offset + 2)?,
        content: row.get(offset + 3)?,
        start_line: row.get(offset + 4)?,
        end_line: row.get(offset + 5)?,
        parent_chunk_id: row.get(offset + 6)?,
    })
}

/// Appends a chunk-type `IN (...)` filter, treating both `None` and
/// `Some(&[])` as "no filter". Contrast with [`append_in_filter`], which
/// renders `Some(&[])` as `AND 1 = 0`.
///
/// Precondition: `sql` ends inside an open WHERE clause — callers anchor with
/// `MATCH ?` (FTS path) or a trailing `IN (...)` (vec path) so the leading
/// ` AND ` from amici helpers attaches cleanly.
fn append_chunk_type_filter(
    sql: &mut String,
    params: &mut Vec<Box<dyn ToSql>>,
    column: &'static str,
    types: Option<&[ChunkType]>,
) {
    let Some(types) = types.filter(|t| !t.is_empty()) else {
        return;
    };
    let strings: Vec<String> = types.iter().map(|c| c.as_str().to_owned()).collect();
    append_in_filter(sql, params, column, Some(&strings));
}

pub fn vec_search(
    conn: &Connection,
    query_embedding: &[f32],
    limit: u32,
    type_filter: Option<&[ChunkType]>,
    path_filter: &[String],
) -> Result<Vec<SearchResult>, StorageError> {
    let k = limit.saturating_mul(VEC_MAXSIM_OVERSAMPLE);
    let knn_rows = knn_only(conn, query_embedding, k)?;

    if knn_rows.is_empty() {
        return Ok(Vec::new());
    }

    // MaxSim: keep the sub-embedding with the smallest distance per chunk_id.
    let mut best: HashMap<i64, f32> = HashMap::new();
    for (chunk_id, distance) in &knn_rows {
        best.entry(*chunk_id)
            .and_modify(|d| {
                if *distance < *d {
                    *d = *distance;
                }
            })
            .or_insert(*distance);
    }

    let chunk_ids: Vec<i64> = best.keys().copied().collect();
    let meta = get_chunks_by_ids(conn, &chunk_ids, type_filter, path_filter)?;
    Ok(build_semantic_results(best, &meta, limit))
}

#[allow(clippy::cast_possible_truncation)]
pub fn search_by_fts(
    conn: &Connection,
    keywords: &[&str],
    type_filter: Option<&[ChunkType]>,
    exclude_ids: &HashSet<i64>,
    include_ids: Option<&HashSet<i64>>,
    limit: u32,
    path_filter: &[String],
) -> Result<Vec<SearchResult>, StorageError> {
    if keywords.is_empty() {
        return Ok(Vec::new());
    }

    let normalization = fts_normalization();
    let mut failed: Vec<(&str, String)> = Vec::new();
    let parts: Vec<String> = keywords
        .iter()
        .filter_map(
            |k| match prepare_match_query(conn, k, "fts_chunks_vocab", &normalization) {
                Ok(m) if !m.as_str().is_empty() => Some(m.into_string()),
                Err(e) if !k.trim().is_empty() => {
                    failed.push((*k, e.to_string()));
                    Some(fts_quote(k))
                }
                _ => None,
            },
        )
        .collect();
    if !failed.is_empty() {
        let failed_keywords: Vec<&str> = failed.iter().map(|(k, _)| *k).collect();
        tracing::warn!(
            keywords = ?failed_keywords,
            count = failed.len(),
            first_error = %failed[0].1,
            "prepare_match_query failed, falling back to fts_quote"
        );
    }
    if parts.is_empty() {
        return Ok(Vec::new());
    }
    let fts_query = parts.join(" AND ");

    const BM25_EXPR: &str = "bm25(fts_chunks, 5.0, 1.0, 3.0)";

    let mut sql = format!(
        "SELECT c.id, c.file_path, c.chunk_type, c.name, c.content,
                c.start_line, c.end_line, c.parent_chunk_id,
                {BM25_EXPR}
         FROM fts_chunks f
         INNER JOIN chunks c ON c.id = f.rowid
         WHERE fts_chunks MATCH ?1",
    );

    let mut params: Vec<Box<dyn ToSql>> = Vec::new();
    params.push(Box::new(fts_query));

    append_chunk_type_filter(&mut sql, &mut params, "c.chunk_type", type_filter);
    append_exclude_ids(&mut sql, &mut params, "c.id", exclude_ids);
    append_include_ids(&mut sql, &mut params, "c.id", include_ids);
    append_like_prefix_filter(&mut sql, &mut params, "c.file_path", path_filter);
    sql.push_str(&format!(" ORDER BY {BM25_EXPR} LIMIT ?"));
    params.push(Box::new(limit));

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(params.iter()), |row| {
        let bm25_score: f64 = row.get(8)?;
        let abs = (bm25_score as f32).abs();
        let base_score = abs / (1.0 + abs);
        Ok(SearchResult {
            chunk: chunk_from_row(row, 1)?,
            chunk_id: Some(row.get(0)?),
            distance: f32::INFINITY,
            match_source: MatchSource::Fts,
            score: base_score,
        })
    })?;

    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_chunk_by_id(conn: &Connection, chunk_id: i64) -> Result<Option<Chunk>, StorageError> {
    let mut stmt = conn.prepare_cached(
        "SELECT file_path, chunk_type, name, content, start_line, end_line, parent_chunk_id
         FROM chunks WHERE id = ?1",
    )?;
    match stmt.query_row([chunk_id], |row| chunk_from_row(row, 0)) {
        Ok(chunk) => Ok(Some(chunk)),
        Err(RusqliteError::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Bulk-fetch chunk metadata by ids with optional type/path filters applied
/// at the SQL layer. Shared by [`vec_search`], [`vec_search_multi`], and
/// callers that need a plain id→chunk lookup.
pub fn get_chunks_by_ids(
    conn: &Connection,
    ids: &[i64],
    type_filter: Option<&[ChunkType]>,
    path_filter: &[String],
) -> Result<HashMap<i64, Chunk>, StorageError> {
    if ids.is_empty() {
        return Ok(HashMap::new());
    }
    let mut sql = format!(
        "SELECT id, file_path, chunk_type, name, content, start_line, end_line, parent_chunk_id \
         FROM chunks WHERE id IN ({})",
        anon_placeholders(ids.len())
    );
    let mut params: Vec<Box<dyn ToSql>> = ids
        .iter()
        .map(|id| Box::new(*id) as Box<dyn ToSql>)
        .collect();
    append_like_prefix_filter(&mut sql, &mut params, "file_path", path_filter);
    append_chunk_type_filter(&mut sql, &mut params, "chunk_type", type_filter);

    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(params.iter()), |row| {
        Ok((row.get::<_, i64>(0)?, chunk_from_row(row, 1)?))
    })?;
    rows.collect::<Result<HashMap<_, _>, _>>()
        .map_err(Into::into)
}

/// Bulk-fetch every Chunk (with body content) belonging to the given files.
/// Output is sorted by `(file_path, start_line)` so brief expansion can apply
/// topological ordering on top without an extra sort pass.
pub fn get_chunks_for_files(conn: &Connection, paths: &[&str]) -> Result<Vec<Chunk>, StorageError> {
    if paths.is_empty() {
        return Ok(Vec::new());
    }
    let placeholders = in_placeholders(paths.len());
    let sql = format!(
        "SELECT file_path, chunk_type, name, content, start_line, end_line, parent_chunk_id
         FROM chunks WHERE file_path IN ({placeholders})
         ORDER BY file_path, start_line"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_from_iter(paths.iter()), |row| chunk_from_row(row, 0))?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_keyword_doc_frequencies(
    conn: &Connection,
    keywords: &[&str],
    total_chunks: u32,
) -> Result<Vec<u32>, StorageError> {
    if keywords.is_empty() {
        return Ok(Vec::new());
    }

    // Above SQLite's SQLITE_MAX_COMPOUND_SELECT (default 500) the UNION ALL
    // refuses to compile. Drop to per-keyword immediately rather than build
    // a SQL string we know will fail.
    if keywords.len() > MAX_BULK_DOC_FREQ_KEYWORDS {
        return doc_frequencies_per_keyword(conn, keywords, total_chunks);
    }

    // SQLite preserves SELECT order across UNION ALL in practice, but we
    // attach an explicit `idx` + ORDER BY to defend against re-ordering.
    let terms: Vec<String> = keywords.iter().map(|k| fts_quote(k)).collect();
    let mut sql = String::with_capacity(terms.len() * 80);
    for i in 0..terms.len() {
        if i > 0 {
            sql.push_str(" UNION ALL ");
        }
        sql.push_str(&format!(
            "SELECT {i} AS idx, COUNT(*) AS df FROM fts_chunks WHERE fts_chunks MATCH ?"
        ));
    }
    sql.push_str(" ORDER BY idx");

    let bulk = conn.prepare(&sql).and_then(|mut stmt| {
        stmt.query_map(params_from_iter(terms.iter()), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, u32>(1)?))
        })?
        .collect::<Result<Vec<(i64, u32)>, _>>()
    });

    match bulk {
        Ok(rows) if rows.len() == keywords.len() => {
            Ok(rows.into_iter().map(|(_, df)| df).collect())
        }
        Ok(rows) => {
            tracing::warn!(
                expected = keywords.len(),
                got = rows.len(),
                "FTS5 bulk doc-freq query returned unexpected row count, falling back to per-keyword"
            );
            doc_frequencies_per_keyword(conn, keywords, total_chunks)
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                count = keywords.len(),
                "FTS5 bulk doc-freq query failed, falling back to per-keyword"
            );
            doc_frequencies_per_keyword(conn, keywords, total_chunks)
        }
    }
}

/// Per-keyword fallback used when the bulk UNION ALL query fails. Substitutes
/// `total_chunks` for individual keyword failures so IDF stays neutral.
fn doc_frequencies_per_keyword(
    conn: &Connection,
    keywords: &[&str],
    total_chunks: u32,
) -> Result<Vec<u32>, StorageError> {
    let mut dfs = Vec::with_capacity(keywords.len());
    let mut failed: Vec<(&str, String)> = Vec::new();
    for kw in keywords {
        let fts_term = fts_quote(kw);
        let count: u32 = match conn.query_row(
            "SELECT COUNT(*) FROM fts_chunks WHERE fts_chunks MATCH ?1",
            [&fts_term],
            |row| row.get(0),
        ) {
            Ok(c) => c,
            Err(e) => {
                failed.push((kw, e.to_string()));
                total_chunks
            }
        };
        dfs.push(count);
    }
    if !failed.is_empty() {
        let failed_keywords: Vec<&str> = failed.iter().map(|(k, _)| *k).collect();
        tracing::warn!(
            keywords = ?failed_keywords,
            count = failed.len(),
            first_error = %failed[0].1,
            "FTS5 doc freq query failed, treating as neutral"
        );
    }
    Ok(dfs)
}

pub fn vec_search_multi(
    conn: &Connection,
    query_embeddings: &[&[f32]],
    limit: u32,
    type_filter: Option<&[ChunkType]>,
    path_filter: &[String],
) -> Result<Vec<SearchResult>, StorageError> {
    if query_embeddings.is_empty() {
        return Ok(Vec::new());
    }
    let k = limit.saturating_mul(VEC_MAXSIM_OVERSAMPLE);

    // KNN per embedding, MaxSim merge across all embeddings before the shared
    // metadata fetch.
    let mut best: HashMap<i64, f32> = HashMap::new();
    for emb in query_embeddings {
        for (chunk_id, distance) in knn_only(conn, emb, k)? {
            best.entry(chunk_id)
                .and_modify(|d| {
                    if distance < *d {
                        *d = distance;
                    }
                })
                .or_insert(distance);
        }
    }
    if best.is_empty() {
        return Ok(Vec::new());
    }

    // Cap candidates by min distance so the IN-clause stays under SQLite's
    // SQLITE_MAX_VARIABLE_NUMBER (32766). Without this, large-corpus searches
    // through `search_from_file` (up to 20 sub-embeddings × fetch_limit × 10
    // oversample) can overflow the prepared-statement parameter limit.
    if best.len() > MAX_VEC_CANDIDATES {
        let mut pairs: Vec<(i64, f32)> = best.into_iter().collect();
        pairs.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        pairs.truncate(MAX_VEC_CANDIDATES);
        best = pairs.into_iter().collect();
    }

    let chunk_ids: Vec<i64> = best.keys().copied().collect();
    let meta = get_chunks_by_ids(conn, &chunk_ids, type_filter, path_filter)?;
    Ok(build_semantic_results(best, &meta, limit))
}

pub fn get_chunks_for_from_target(
    conn: &Connection,
    file_path: &str,
    symbol: Option<&str>,
) -> Result<Vec<i64>, StorageError> {
    let mut sql = String::from(
        "SELECT id FROM chunks \
         WHERE file_path = ?1 AND chunk_type != 'inner_fn'",
    );
    if symbol.is_some() {
        sql.push_str(" AND name = ?2");
    }

    let mut stmt = conn.prepare(&sql)?;
    let ids = if let Some(sym) = symbol {
        stmt.query_map(params![file_path, sym], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?
    } else {
        stmt.query_map([file_path], |row| row.get::<_, i64>(0))?
            .collect::<Result<Vec<_>, _>>()?
    };
    Ok(ids)
}

pub fn get_sub_embeddings_for_chunks(
    conn: &Connection,
    chunk_ids: &[i64],
) -> Result<Vec<(i64, Vec<u8>)>, StorageError> {
    if chunk_ids.is_empty() {
        return Ok(Vec::new());
    }

    // Step 1: Get (chunk_id, vec_rowid) from embedded_chunk_ids, ordered by sub_idx.
    let placeholders = anon_placeholders(chunk_ids.len());
    let sql = format!(
        "SELECT chunk_id, vec_rowid FROM embedded_chunk_ids \
         WHERE chunk_id IN ({placeholders}) ORDER BY chunk_id, sub_idx"
    );
    let mut stmt = conn.prepare(&sql)?;
    let mappings: Vec<(i64, i64)> = stmt
        .query_map(params_from_iter(chunk_ids.iter()), |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?
        .collect::<Result<Vec<_>, _>>()?;

    if mappings.is_empty() {
        return Ok(Vec::new());
    }

    // Step 2: Batch-fetch embedding bytes (vec0 supports WHERE rowid IN (...)).
    let emb_placeholders = anon_placeholders(mappings.len());
    let emb_sql =
        format!("SELECT rowid, embedding FROM vec_chunks WHERE rowid IN ({emb_placeholders})");
    let mut by_rowid: HashMap<i64, Vec<u8>> = conn
        .prepare(&emb_sql)?
        .query_map(
            params_from_iter(mappings.iter().map(|(_, vid)| vid)),
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?
        .collect::<Result<_, _>>()?;

    let results: Vec<(i64, Vec<u8>)> = mappings
        .iter()
        .filter_map(|(chunk_id, vec_rowid)| by_rowid.remove(vec_rowid).map(|b| (*chunk_id, b)))
        .collect();
    Ok(results)
}
