use rusqlite::Connection;

use super::{sql_placeholders, ChunkType, StorageError};
#[cfg(test)]
use super::Reference;

#[derive(Debug, Clone, PartialEq)]
pub struct Dependent {
    pub file_path: String,
    pub depth: u32,
}

#[derive(Debug, Clone)]
pub struct SiblingInfo {
    pub name: Option<String>,
    pub chunk_type: ChunkType,
    pub start_line: u32,
    pub end_line: u32,
}

const HOOK_COMPONENT_PRIORITY_BOOST: u32 = 3;

pub fn get_transitive_dependents(
    conn: &Connection,
    target_file: &str,
    max_depth: u32,
) -> Result<Vec<Dependent>, StorageError> {
    let max_depth = max_depth.min(10);
    let mut stmt = conn.prepare(
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

pub fn get_symbol_dependents(
    conn: &Connection,
    target_file: &str,
    symbol_name: &str,
) -> Result<Vec<String>, StorageError> {
    let mut stmt = conn.prepare(
        "SELECT DISTINCT source_file FROM file_references
         WHERE target_file = ?1 AND symbol_name = ?2",
    )?;
    let rows = stmt.query_map(rusqlite::params![target_file, symbol_name], |row| {
        row.get::<_, String>(0)
    })?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_reference_count(conn: &Connection) -> Result<u32, StorageError> {
    let count: u32 = conn.query_row(
        "SELECT COUNT(*) FROM file_references",
        [],
        |row| row.get(0),
    )?;
    Ok(count)
}

/// Returns un-embedded file paths ordered by import count (most-imported first).
pub fn get_files_by_import_count(
    conn: &Connection,
) -> Result<Vec<String>, StorageError> {
    let sql = format!(
        "SELECT c.file_path
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
             ) THEN {HOOK_COMPONENT_PRIORITY_BOOST} ELSE 0 END
         ) DESC, c.file_path ASC"
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
}

pub fn get_import_counts(
    conn: &Connection,
    file_paths: &[&str],
) -> Result<std::collections::HashMap<String, u32>, StorageError> {
    if file_paths.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let sql = format!(
        "SELECT fp.path, COALESCE(cnt.c, 0)
         FROM (SELECT value AS path FROM ({})) fp
         LEFT JOIN (
             SELECT target_file, COUNT(DISTINCT source_file) AS c
             FROM file_references
             GROUP BY target_file
         ) cnt ON cnt.target_file = fp.path",
        file_paths.iter().enumerate()
            .map(|(i, _)| format!("SELECT ?{} AS value", i + 1))
            .collect::<Vec<_>>()
            .join(" UNION ALL ")
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> = file_paths
        .iter()
        .map(|s| s as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, u32>(1)?))
    })?;
    rows.collect::<Result<std::collections::HashMap<_, _>, _>>()
        .map_err(Into::into)
}

pub fn get_file_contexts(
    conn: &Connection,
    file_paths: &[&str],
) -> Result<std::collections::HashMap<String, String>, StorageError> {
    if file_paths.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let placeholders = sql_placeholders(file_paths.len());
    let sql = format!(
        "SELECT file_path, imports_text FROM file_context WHERE file_path IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> = file_paths
        .iter()
        .map(|s| s as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = stmt.query_map(params.as_slice(), |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    rows.collect::<Result<std::collections::HashMap<_, _>, _>>()
        .map_err(Into::into)
}

pub fn get_file_siblings(
    conn: &Connection,
    file_paths: &[&str],
) -> Result<std::collections::HashMap<String, Vec<SiblingInfo>>, StorageError> {
    if file_paths.is_empty() {
        return Ok(std::collections::HashMap::new());
    }
    let placeholders = sql_placeholders(file_paths.len());
    let sql = format!(
        "SELECT file_path, name, chunk_type, start_line, end_line \
         FROM chunks WHERE file_path IN ({placeholders}) \
         ORDER BY file_path, start_line"
    );
    let mut stmt = conn.prepare(&sql)?;
    let params: Vec<&dyn rusqlite::types::ToSql> = file_paths
        .iter()
        .map(|s| s as &dyn rusqlite::types::ToSql)
        .collect();
    let rows = stmt.query_map(params.as_slice(), |row| {
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
    let mut map: std::collections::HashMap<String, Vec<SiblingInfo>> =
        std::collections::HashMap::new();
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
            rusqlite::params![r.source_file, r.target_file, r.symbol_name, r.ref_kind.as_str()],
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
    let mut stmt = conn.prepare(
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
