//! Verify vec_chunks data integrity in an actual yomu DB.
//!
//! Usage: cargo test --test verify_kai_db -- --nocapture
//!
//! Requires: ~/GitHub/kai/main/.yomu/index.db to exist
//! (run yomu explorer on kai/main first)

use std::path::PathBuf;

fn kai_db_path() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME not set");
    PathBuf::from(home)
        .join("GitHub/kai/main/.yomu/index.db")
}

fn skip_if_no_db() -> bool {
    !kai_db_path().exists()
}

#[test]
fn vec_chunks_row_count_matches_stats() {
    if skip_if_no_db() {
        eprintln!("SKIP: kai/main DB not found");
        return;
    }

    let conn = yomu::storage::open_db(&kai_db_path()).unwrap();
    let stats = yomu::storage::get_stats(&conn).unwrap();

    // vec_chunks count should match embedded_chunks in stats
    let vec_count: u32 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |row| row.get(0))
        .unwrap();

    println!("stats.embedded_chunks = {}", stats.embedded_chunks);
    println!("vec_chunks COUNT(*)   = {}", vec_count);
    assert_eq!(
        stats.embedded_chunks, vec_count,
        "embedded_chunks stat must match vec_chunks row count"
    );
    assert!(vec_count > 0, "vec_chunks should not be empty");
}

#[test]
fn vec_chunks_ids_reference_valid_chunks() {
    if skip_if_no_db() {
        eprintln!("SKIP: kai/main DB not found");
        return;
    }

    let conn = yomu::storage::open_db(&kai_db_path()).unwrap();

    // Every chunk_id in vec_chunks must exist in chunks table
    let orphan_count: u32 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks v
             WHERE v.chunk_id NOT IN (SELECT id FROM chunks)",
            [],
            |row| row.get(0),
        )
        .unwrap();

    println!("orphan vec_chunks (no matching chunk): {}", orphan_count);
    assert_eq!(orphan_count, 0, "all vec_chunks must reference valid chunks");
}

#[test]
fn vec_chunks_embeddings_have_correct_dimensions() {
    if skip_if_no_db() {
        eprintln!("SKIP: kai/main DB not found");
        return;
    }

    let conn = yomu::storage::open_db(&kai_db_path()).unwrap();

    // Each embedding should be 768 * 4 bytes = 3072 bytes (f32 little-endian)
    let expected_bytes = yomu::storage::EMBEDDING_DIMS as usize * 4;

    let mut stmt = conn
        .prepare("SELECT chunk_id, LENGTH(embedding) FROM vec_chunks")
        .unwrap();

    let rows: Vec<(i64, usize)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    println!("vec_chunks rows: {}", rows.len());
    println!("expected embedding size: {} bytes (768 * f32)", expected_bytes);

    let mut bad_dims = Vec::new();
    for (chunk_id, byte_len) in &rows {
        if *byte_len != expected_bytes {
            bad_dims.push((*chunk_id, *byte_len));
        }
    }

    if !bad_dims.is_empty() {
        for (id, len) in &bad_dims {
            eprintln!(
                "BAD: chunk_id={} has {} bytes ({} dims, expected 768)",
                id,
                len,
                len / 4
            );
        }
    }
    assert!(
        bad_dims.is_empty(),
        "{} embeddings have wrong dimensions",
        bad_dims.len()
    );
}

#[test]
fn vec_chunks_embeddings_are_not_zero_vectors() {
    if skip_if_no_db() {
        eprintln!("SKIP: kai/main DB not found");
        return;
    }

    let conn = yomu::storage::open_db(&kai_db_path()).unwrap();

    let mut stmt = conn
        .prepare("SELECT chunk_id, embedding FROM vec_chunks")
        .unwrap();

    let mut zero_count = 0u32;
    let mut total = 0u32;
    let mut sample_norms: Vec<(i64, f64)> = Vec::new();

    let rows = stmt
        .query_map([], |row| {
            let chunk_id: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((chunk_id, blob))
        })
        .unwrap();

    for row in rows {
        let (chunk_id, blob) = row.unwrap();
        total += 1;

        // Reinterpret bytes as f32 slice
        let floats: &[f32] = bytemuck::cast_slice(&blob);

        // Check if all zeros
        let norm_sq: f64 = floats.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let norm = norm_sq.sqrt();

        if norm < 1e-10 {
            zero_count += 1;
        }

        // Collect first 5 norms as samples
        if sample_norms.len() < 5 {
            sample_norms.push((chunk_id, norm));
        }
    }

    println!("total vec_chunks: {}", total);
    println!("zero vectors: {}", zero_count);
    for (id, norm) in &sample_norms {
        println!("  chunk_id={}: L2 norm = {:.6}", id, norm);
    }

    assert_eq!(zero_count, 0, "no embedding should be a zero vector");
}

#[test]
fn vec_chunks_embedded_files_match_most_imported_order() {
    if skip_if_no_db() {
        eprintln!("SKIP: kai/main DB not found");
        return;
    }

    let conn = yomu::storage::open_db(&kai_db_path()).unwrap();

    // Get files that have embeddings (via JOIN with chunks)
    let mut stmt = conn
        .prepare(
            "SELECT DISTINCT c.file_path
             FROM vec_chunks v
             INNER JOIN chunks c ON c.id = v.chunk_id",
        )
        .unwrap();

    let embedded_files: Vec<String> = stmt
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    println!("embedded files: {:?}", embedded_files);

    // Get most-imported files (should be superset of embedded files)
    let most_imported = yomu::storage::get_files_by_import_count(&conn).unwrap();
    println!(
        "most-imported order (top 5): {:?}",
        &most_imported.iter().take(5).collect::<Vec<_>>()
    );

    // All embedded files should be in the most-imported list
    // (they may not be at the very top since get_files_by_import_count
    //  only returns UN-embedded files, but the original ordering should match)
    assert!(
        !embedded_files.is_empty(),
        "should have at least one embedded file"
    );

    // Verify embedded file import counts are >= all un-embedded file import counts
    // i.e., the most important files were prioritized
    let mut embedded_min_imports = u32::MAX;
    for f in &embedded_files {
        let count: u32 = conn
            .query_row(
                "SELECT COUNT(*) FROM file_references WHERE target_file = ?1",
                [f.as_str()],
                |row| row.get(0),
            )
            .unwrap();
        println!("  embedded: {} (import count: {})", f, count);
        if count < embedded_min_imports {
            embedded_min_imports = count;
        }
    }

    // Check that un-embedded files with MORE imports than our minimum don't exist
    let higher_unembedded: u32 = conn
        .query_row(
            "SELECT COUNT(DISTINCT c.file_path) FROM chunks c
             WHERE c.file_path NOT IN (
                 SELECT DISTINCT c2.file_path FROM vec_chunks v
                 INNER JOIN chunks c2 ON c2.id = v.chunk_id
             )
             AND (SELECT COUNT(*) FROM file_references r
                  WHERE r.target_file = c.file_path) > ?1",
            [embedded_min_imports],
            |row| row.get(0),
        )
        .unwrap();

    println!(
        "un-embedded files with import count > {}: {}",
        embedded_min_imports, higher_unembedded
    );
    assert_eq!(
        higher_unembedded, 0,
        "no un-embedded file should have more imports than the least-imported embedded file"
    );
}

#[test]
fn vector_search_returns_results_from_embedded_chunks() {
    if skip_if_no_db() {
        eprintln!("SKIP: kai/main DB not found");
        return;
    }

    let conn = yomu::storage::open_db(&kai_db_path()).unwrap();

    // Get an actual embedding to use as a query vector
    let query_vec: Vec<u8> = conn
        .query_row("SELECT embedding FROM vec_chunks LIMIT 1", [], |row| {
            row.get(0)
        })
        .unwrap();

    let query_floats: &[f32] = bytemuck::cast_slice(&query_vec);
    assert_eq!(
        query_floats.len(),
        yomu::storage::EMBEDDING_DIMS as usize,
        "query vector should have 768 dims"
    );

    // Search using the embedding itself — should find at least itself
    let results = yomu::storage::search_similar(&conn, query_floats, 5, 0).unwrap();

    println!("search results: {}", results.len());
    for r in &results {
        let sim = 1.0 / (1.0 + r.distance);
        println!(
            "  {} [{}] {}:{} distance={:.4} sim={:.4}",
            r.chunk.name.as_deref().unwrap_or("(unnamed)"),
            r.chunk.chunk_type.as_str(),
            r.chunk.start_line,
            r.chunk.end_line,
            r.distance,
            sim,
        );
    }

    assert!(!results.is_empty(), "search should return results");
    assert!(
        results[0].distance < 0.01,
        "searching with own embedding should return near-zero distance, got {}",
        results[0].distance
    );
}
