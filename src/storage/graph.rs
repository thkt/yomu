use std::collections::HashMap;

use rusqlite::Connection;

#[cfg(test)]
use super::Reference;
use super::{ChunkType, RefKind, StorageError, as_sql_params, in_placeholders};

#[derive(Debug, Clone, PartialEq)]
pub struct Dependent {
    pub file_path: String,
    pub depth: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dependency {
    pub file_path: String,
    pub depth: u32,
}

/// One reference edge from `source_file` to a target file, retaining the
/// `ref_kind` (named/default/...) and the symbol that triggered it.
///
/// `via_symbol` is `None` for namespace/side-effect imports where no
/// individual symbol is named at the import site.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectReference {
    pub source_file: String,
    pub ref_kind: RefKind,
    pub via_symbol: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SiblingInfo {
    pub name: Option<String>,
    pub chunk_type: ChunkType,
    pub start_line: u32,
    pub end_line: u32,
}

const SQL_FILES_BY_IMPORT_COUNT: &str = "SELECT c.file_path
         FROM chunks c
         LEFT JOIN vec_chunks v ON c.id = v.chunk_id
         WHERE v.chunk_id IS NULL
         GROUP BY c.file_path
         ORDER BY (
             SELECT COUNT(*) FROM file_references r
             WHERE r.target_file = c.file_path
         ) + (
             SELECT CASE WHEN EXISTS(
                 SELECT 1 FROM chunks c2
                 WHERE c2.file_path = c.file_path
                 AND c2.chunk_type IN ('hook', 'component')
             ) THEN 3 ELSE 0 END
         ) DESC, c.file_path ASC";

pub fn get_transitive_dependents(
    conn: &Connection,
    target_file: &str,
    max_depth: u32,
) -> Result<Vec<Dependent>, StorageError> {
    let max_depth = max_depth.min(10);
    let mut stmt = conn.prepare_cached(
        "WITH RECURSIVE deps(file_path, depth, visited) AS (
            SELECT DISTINCT source_file, 1,
                   ',' || ?1 || ',' || source_file || ','
            FROM file_references
            WHERE target_file = ?1
          UNION
            SELECT r.source_file, d.depth + 1,
                   d.visited || r.source_file || ','
            FROM file_references r
            INNER JOIN deps d ON r.target_file = d.file_path
            WHERE d.depth < ?2
              AND INSTR(d.visited, ',' || r.source_file || ',') = 0
        )
        SELECT file_path, MIN(depth) as depth
        FROM deps GROUP BY file_path ORDER BY depth, file_path",
    )?;
    let rows = stmt.query_map(rusqlite::params![target_file, max_depth], |row| {
        Ok(Dependent {
            file_path: row.get(0)?,
            depth: row.get(1)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

/// Forward closure: files that `seed` transitively depends on, ordered by
/// distance ascending. `seed` itself is included at depth 0 so brief output
/// can group seed chunks alongside their forward dependencies.
pub fn get_transitive_dependencies(
    conn: &Connection,
    seed: &str,
    max_depth: u32,
) -> Result<Vec<Dependency>, StorageError> {
    let max_depth = max_depth.min(10);
    let mut stmt = conn.prepare_cached(
        "WITH RECURSIVE deps(file_path, depth, visited) AS (
            SELECT ?1, 0, ',' || ?1 || ','
          UNION
            SELECT r.target_file, d.depth + 1,
                   d.visited || r.target_file || ','
            FROM file_references r
            INNER JOIN deps d ON r.source_file = d.file_path
            WHERE d.depth < ?2
              AND INSTR(d.visited, ',' || r.target_file || ',') = 0
        )
        SELECT file_path, MIN(depth) as depth
        FROM deps GROUP BY file_path ORDER BY depth, file_path",
    )?;
    let rows = stmt.query_map(rusqlite::params![seed, max_depth], |row| {
        Ok(Dependency {
            file_path: row.get(0)?,
            depth: row.get(1)?,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

/// Returns every direct (depth=1) reference edge that points at `target_file`.
///
/// Each row is one `(source_file, ref_kind, via_symbol)` triple. Rows are
/// distinct on that triple — duplicate parses of the same import collapse to
/// a single row, but a file that imports the target twice with different
/// symbols or kinds still produces multiple rows.
pub fn get_direct_references(
    conn: &Connection,
    target_file: &str,
) -> Result<Vec<DirectReference>, StorageError> {
    let mut stmt = conn.prepare_cached(
        "SELECT DISTINCT source_file, ref_kind, symbol_name
         FROM file_references
         WHERE target_file = ?1
         ORDER BY source_file, ref_kind, symbol_name",
    )?;
    let rows = stmt.query_map([target_file], |row| {
        let ref_kind = RefKind::from_db(row.get::<_, String>(1)?.as_ref());
        let symbol: Option<String> = row.get(2)?;
        let via_symbol = match ref_kind {
            RefKind::Namespace | RefKind::SideEffect => None,
            _ => symbol,
        };
        Ok(DirectReference {
            source_file: row.get(0)?,
            ref_kind,
            via_symbol,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_symbol_dependents(
    conn: &Connection,
    target_file: &str,
    symbol_name: &str,
) -> Result<Vec<String>, StorageError> {
    let mut stmt = conn.prepare_cached(
        "SELECT DISTINCT source_file FROM file_references
         WHERE target_file = ?1 AND symbol_name = ?2",
    )?;
    let rows = stmt.query_map(rusqlite::params![target_file, symbol_name], |row| {
        row.get::<_, String>(0)
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_reference_count(conn: &Connection) -> Result<u32, StorageError> {
    let count: u32 =
        conn.query_row("SELECT COUNT(*) FROM file_references", [], |row| row.get(0))?;
    Ok(count)
}

/// Returns un-embedded file paths ordered by import count (most-imported first).
pub fn get_files_by_import_count(conn: &Connection) -> Result<Vec<String>, StorageError> {
    let mut stmt = conn.prepare_cached(SQL_FILES_BY_IMPORT_COUNT)?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_import_counts(
    conn: &Connection,
    file_paths: &[&str],
) -> Result<HashMap<String, u32>, StorageError> {
    if file_paths.is_empty() {
        return Ok(HashMap::new());
    }
    let sql = format!(
        "SELECT fp.path, COALESCE(cnt.c, 0)
         FROM (SELECT value AS path FROM ({})) fp
         LEFT JOIN (
             SELECT target_file, COUNT(DISTINCT source_file) AS c
             FROM file_references
             GROUP BY target_file
         ) cnt ON cnt.target_file = fp.path",
        file_paths
            .iter()
            .enumerate()
            .map(|(i, _)| format!("SELECT ?{} AS value", i + 1))
            .collect::<Vec<_>>()
            .join(" UNION ALL ")
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(as_sql_params(file_paths).as_slice(), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, u32>(1)?))
    })?;
    rows.collect::<Result<HashMap<_, _>, _>>()
        .map_err(Into::into)
}

pub fn get_file_mtimes(
    conn: &Connection,
    file_paths: &[&str],
) -> Result<HashMap<String, i64>, StorageError> {
    if file_paths.is_empty() {
        return Ok(HashMap::new());
    }
    let placeholders = in_placeholders(file_paths.len());
    let sql = format!(
        "SELECT file_path, mtime_epoch FROM file_context WHERE file_path IN ({placeholders}) AND mtime_epoch IS NOT NULL"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(as_sql_params(file_paths).as_slice(), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;
    rows.collect::<Result<HashMap<_, _>, _>>()
        .map_err(Into::into)
}

pub fn get_file_contexts(
    conn: &Connection,
    file_paths: &[&str],
) -> Result<HashMap<String, String>, StorageError> {
    if file_paths.is_empty() {
        return Ok(HashMap::new());
    }
    let placeholders = in_placeholders(file_paths.len());
    let sql = format!(
        "SELECT file_path, imports_text FROM file_context WHERE file_path IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(as_sql_params(file_paths).as_slice(), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    rows.collect::<Result<HashMap<_, _>, _>>()
        .map_err(Into::into)
}

pub fn get_file_siblings(
    conn: &Connection,
    file_paths: &[&str],
) -> Result<HashMap<String, Vec<SiblingInfo>>, StorageError> {
    if file_paths.is_empty() {
        return Ok(HashMap::new());
    }
    let placeholders = in_placeholders(file_paths.len());
    let sql = format!(
        "SELECT file_path, name, chunk_type, start_line, end_line \
         FROM chunks WHERE file_path IN ({placeholders}) \
         ORDER BY file_path, start_line"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(as_sql_params(file_paths).as_slice(), |row| {
        Ok((
            row.get::<_, String>(0)?,
            SiblingInfo {
                name: row.get(1)?,
                chunk_type: ChunkType::from_db(row.get::<_, String>(2)?.as_ref()),
                start_line: row.get(3)?,
                end_line: row.get(4)?,
            },
        ))
    })?;
    let mut map: HashMap<String, Vec<SiblingInfo>> = HashMap::new();
    for row in rows {
        let (path, info) = row?;
        map.entry(path).or_default().push(info);
    }
    Ok(map)
}

#[cfg(test)]
pub fn replace_file_references(
    conn: &Connection,
    source_file: &str,
    refs: &[Reference],
) -> Result<(), StorageError> {
    let tx = conn.unchecked_transaction()?;
    tx.execute(
        "DELETE FROM file_references WHERE source_file = ?1",
        [source_file],
    )?;
    for r in refs {
        tx.execute(
            "INSERT INTO file_references (source_file, target_file, symbol_name, ref_kind)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![
                r.source_file,
                r.target_file,
                r.symbol_name,
                r.ref_kind.as_str()
            ],
        )?;
    }
    tx.commit()?;
    Ok(())
}

#[cfg(test)]
pub fn get_dependents(
    conn: &Connection,
    target_file: &str,
) -> Result<Vec<Dependent>, StorageError> {
    let mut stmt = conn.prepare_cached(
        "SELECT DISTINCT source_file FROM file_references WHERE target_file = ?1",
    )?;
    let rows = stmt.query_map([target_file], |row| {
        Ok(Dependent {
            file_path: row.get(0)?,
            depth: 1,
        })
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}
