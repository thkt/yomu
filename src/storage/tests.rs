use super::*;

fn test_db() -> (Connection, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");
    let conn = open_db(&db_path).unwrap();
    (conn, dir)
}

#[test]
fn init_db_creates_tables() {
    let (conn, _dir) = test_db();

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn insert_and_read_chunk() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.1_f32; 768];

    let id = insert_chunk(
        &conn,
        "src/Button.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("Button"),
            content: "function Button() { return <div/>; }",
            start_line: 1,
            end_line: 3,
        },
        "abc123",
        &embedding,
    )
    .unwrap();

    assert!(id > 0);

    let chunk: (String, String, String) = conn
        .query_row(
            "SELECT file_path, chunk_type, name FROM chunks WHERE id = ?1",
            [id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();

    assert_eq!(chunk.0, "src/Button.tsx");
    assert_eq!(chunk.1, "component");
    assert_eq!(chunk.2, "Button");
}

#[test]
fn should_reindex_returns_false_for_same_hash() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; 768];

    insert_chunk(
        &conn,
        "src/App.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("App"),
            content: "function App() {}",
            start_line: 1,
            end_line: 1,
        },
        "hash_abc",
        &embedding,
    )
    .unwrap();

    assert!(!should_reindex(&conn, "src/App.tsx", "hash_abc").unwrap());
}

#[test]
fn should_reindex_returns_true_for_different_hash() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; 768];

    insert_chunk(
        &conn,
        "src/App.tsx",
        &NewChunk {
            chunk_type: &ChunkType::Component,
            name: Some("App"),
            content: "function App() {}",
            start_line: 1,
            end_line: 1,
        },
        "hash_abc",
        &embedding,
    )
    .unwrap();

    assert!(should_reindex(&conn, "src/App.tsx", "hash_xyz").unwrap());
}

#[test]
fn should_reindex_returns_true_for_new_file() {
    let (conn, _dir) = test_db();
    assert!(should_reindex(&conn, "src/New.tsx", "any_hash").unwrap());
}

#[test]
fn get_stats_returns_counts() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; 768];

    insert_chunk(
        &conn, "src/A.tsx",
        &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code", start_line: 1, end_line: 5 },
        "h1", &embedding,
    ).unwrap();
    insert_chunk(
        &conn, "src/A.tsx",
        &NewChunk { chunk_type: &ChunkType::Hook, name: Some("useA"), content: "code", start_line: 6, end_line: 10 },
        "h1", &embedding,
    ).unwrap();
    insert_chunk(
        &conn, "src/B.tsx",
        &NewChunk { chunk_type: &ChunkType::Component, name: Some("B"), content: "code", start_line: 1, end_line: 3 },
        "h2", &embedding,
    ).unwrap();

    let stats = get_stats(&conn).unwrap();
    assert_eq!(stats.total_chunks, 3);
    assert_eq!(stats.total_files, 2);
}

#[test]
fn get_all_file_paths_returns_distinct_paths() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; 768];

    insert_chunk(&conn, "src/A.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code", start_line: 1, end_line: 3 }, "h1", &embedding).unwrap();
    insert_chunk(&conn, "src/A.tsx", &NewChunk { chunk_type: &ChunkType::Hook, name: Some("useA"), content: "code", start_line: 4, end_line: 6 }, "h1", &embedding).unwrap();
    insert_chunk(&conn, "src/B.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("B"), content: "code", start_line: 1, end_line: 3 }, "h2", &embedding).unwrap();

    let paths = get_all_file_paths(&conn).unwrap();
    assert_eq!(paths.len(), 2);
    assert!(paths.contains("src/A.tsx"));
    assert!(paths.contains("src/B.tsx"));
}

#[test]
fn delete_file_chunks_removes_all_chunks_for_file() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; 768];

    insert_chunk(&conn, "src/A.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code", start_line: 1, end_line: 3 }, "h1", &embedding).unwrap();
    insert_chunk(&conn, "src/B.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("B"), content: "code", start_line: 1, end_line: 3 }, "h2", &embedding).unwrap();

    delete_file_chunks(&conn, "src/A.tsx").unwrap();

    let stats = get_stats(&conn).unwrap();
    assert_eq!(stats.total_files, 1);
    assert_eq!(stats.total_chunks, 1);

    let paths = get_all_file_paths(&conn).unwrap();
    assert!(!paths.contains("src/A.tsx"));
    assert!(paths.contains("src/B.tsx"));
}

#[test]
fn replace_file_chunks_replaces_existing() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; 768];

    insert_chunk(
        &conn, "src/A.tsx",
        &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "old code", start_line: 1, end_line: 5 },
        "h1", &embedding,
    ).unwrap();

    let new_chunks = vec![
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useA"), content: "new code", start_line: 1, end_line: 3 },
        NewChunk { chunk_type: &ChunkType::Component, name: Some("B"), content: "more code", start_line: 4, end_line: 8 },
    ];
    let embeddings = vec![embedding.clone(), embedding.clone()];

    replace_file_chunks(&conn, "src/A.tsx", &new_chunks, &embeddings, "h2", "import { x } from 'y'", &[]).unwrap();

    let stats = get_stats(&conn).unwrap();
    assert_eq!(stats.total_chunks, 2);
    assert_eq!(stats.total_files, 1);
    assert!(stats.last_indexed_at.is_some());
}

#[test]
fn search_similar_returns_ordered_results() {
    let (conn, _dir) = test_db();
    let mut emb_a = vec![0.0_f32; 768];
    emb_a[0] = 1.0;
    let mut emb_b = vec![0.0_f32; 768];
    emb_b[1] = 1.0;

    insert_chunk(&conn, "src/A.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code a", start_line: 1, end_line: 3 }, "h1", &emb_a).unwrap();
    insert_chunk(&conn, "src/B.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("B"), content: "code b", start_line: 1, end_line: 3 }, "h2", &emb_b).unwrap();

    let results = search_similar(&conn, &emb_a, 10, 0).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].chunk.name.as_deref(), Some("A"));
    assert!(results[0].distance <= results[1].distance);
}

#[test]
fn chunk_type_roundtrip() {
    let variants = [
        ChunkType::Component,
        ChunkType::Hook,
        ChunkType::TypeDef,
        ChunkType::CssRule,
        ChunkType::HtmlElement,
        ChunkType::TestCase,
        ChunkType::Other,
    ];
    for variant in &variants {
        let s = variant.as_str();
        let restored = ChunkType::from_db(s);
        assert_eq!(&restored, variant, "roundtrip failed for {s}");
    }
}

#[test]
fn chunk_type_from_db_unknown_defaults_to_other() {
    assert_eq!(ChunkType::from_db("nonexistent"), ChunkType::Other);
}

#[test]
fn delete_file_chunks_also_removes_file_context() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; 768];
    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code", start_line: 1, end_line: 5 },
    ];
    let embeddings = vec![embedding];
    replace_file_chunks(&conn, "src/A.tsx", &chunks, &embeddings, "h1", "import React from 'react'", &[]).unwrap();

    let contexts = get_file_contexts(&conn, &["src/A.tsx"]).unwrap();
    assert_eq!(contexts.len(), 1, "pre-condition: file_context should exist");

    delete_file_chunks(&conn, "src/A.tsx").unwrap();

    let contexts = get_file_contexts(&conn, &["src/A.tsx"]).unwrap();
    assert!(contexts.is_empty(), "file_context should be removed after delete: {contexts:?}");
}

#[test]
fn replace_file_chunks_stores_file_context() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; 768];
    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Component, name: Some("App"), content: "code", start_line: 1, end_line: 5 },
    ];
    let embeddings = vec![embedding];
    replace_file_chunks(&conn, "src/App.tsx", &chunks, &embeddings, "h1", "import { useState } from 'react'", &[]).unwrap();

    let contexts = get_file_contexts(&conn, &["src/App.tsx"]).unwrap();
    assert_eq!(contexts.len(), 1);
    assert_eq!(contexts["src/App.tsx"], "import { useState } from 'react'");
}

#[test]
fn get_file_contexts_returns_empty_for_missing_files() {
    let (conn, _dir) = test_db();
    let contexts = get_file_contexts(&conn, &["src/Missing.tsx"]).unwrap();
    assert!(contexts.is_empty());
}

#[test]
fn get_file_contexts_returns_empty_for_empty_input() {
    let (conn, _dir) = test_db();
    let contexts = get_file_contexts(&conn, &[]).unwrap();
    assert!(contexts.is_empty());
}

#[test]
fn get_file_siblings_returns_all_chunks_for_file() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; 768];
    insert_chunk(&conn, "src/A.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("App"), content: "code", start_line: 1, end_line: 5 }, "h1", &embedding).unwrap();
    insert_chunk(&conn, "src/A.tsx", &NewChunk { chunk_type: &ChunkType::Hook, name: Some("useAuth"), content: "code", start_line: 6, end_line: 10 }, "h1", &embedding).unwrap();
    insert_chunk(&conn, "src/B.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("Button"), content: "code", start_line: 1, end_line: 3 }, "h2", &embedding).unwrap();

    let siblings = get_file_siblings(&conn, &["src/A.tsx"]).unwrap();
    assert_eq!(siblings.len(), 1);
    let a_siblings = &siblings["src/A.tsx"];
    assert_eq!(a_siblings.len(), 2);
    assert_eq!(a_siblings[0].name.as_deref(), Some("App"));
    assert_eq!(a_siblings[1].name.as_deref(), Some("useAuth"));
}

#[test]
fn get_file_siblings_returns_empty_for_empty_input() {
    let (conn, _dir) = test_db();
    let siblings = get_file_siblings(&conn, &[]).unwrap();
    assert!(siblings.is_empty());
}

// ── T-017: replace_file_references stores references ─────────────

#[test]
fn replace_file_references_stores_refs() {
    let (conn, _dir) = test_db();
    let refs = vec![
        Reference { source_file: "src/A.tsx".into(), target_file: "src/B.tsx".into(), symbol_name: Some("Button".into()), ref_kind: RefKind::Named },
        Reference { source_file: "src/A.tsx".into(), target_file: "src/C.tsx".into(), symbol_name: None, ref_kind: RefKind::Namespace },
        Reference { source_file: "src/A.tsx".into(), target_file: "src/D.tsx".into(), symbol_name: Some("useAuth".into()), ref_kind: RefKind::Named },
    ];
    replace_file_references(&conn, "src/A.tsx", &refs).unwrap();

    let count = get_reference_count(&conn).unwrap();
    assert_eq!(count, 3);
}

// ── T-018: re-index replaces old references ──────────────────────

#[test]
fn replace_file_references_replaces_existing() {
    let (conn, _dir) = test_db();
    let old_refs = vec![
        Reference { source_file: "src/A.tsx".into(), target_file: "src/B.tsx".into(), symbol_name: Some("B".into()), ref_kind: RefKind::Named },
        Reference { source_file: "src/A.tsx".into(), target_file: "src/C.tsx".into(), symbol_name: None, ref_kind: RefKind::Default },
        Reference { source_file: "src/A.tsx".into(), target_file: "src/D.tsx".into(), symbol_name: Some("D".into()), ref_kind: RefKind::Named },
    ];
    replace_file_references(&conn, "src/A.tsx", &old_refs).unwrap();

    let new_refs = vec![
        Reference { source_file: "src/A.tsx".into(), target_file: "src/E.tsx".into(), symbol_name: Some("E".into()), ref_kind: RefKind::Named },
        Reference { source_file: "src/A.tsx".into(), target_file: "src/F.tsx".into(), symbol_name: None, ref_kind: RefKind::TypeOnly },
    ];
    replace_file_references(&conn, "src/A.tsx", &new_refs).unwrap();

    let count = get_reference_count(&conn).unwrap();
    assert_eq!(count, 2);

    let dependents = get_dependents(&conn, "src/E.tsx").unwrap();
    assert_eq!(dependents.len(), 1);
    assert_eq!(dependents[0].file_path, "src/A.tsx");
}

// ── T-019: delete file also deletes references ───────────────────

#[test]
fn delete_file_chunks_also_removes_references() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; 768];
    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code", start_line: 1, end_line: 5 },
    ];
    let embeddings = vec![embedding];
    replace_file_chunks(&conn, "src/A.tsx", &chunks, &embeddings, "h1", "", &[]).unwrap();

    let refs = vec![
        Reference { source_file: "src/A.tsx".into(), target_file: "src/B.tsx".into(), symbol_name: Some("B".into()), ref_kind: RefKind::Named },
    ];
    replace_file_references(&conn, "src/A.tsx", &refs).unwrap();

    delete_file_chunks(&conn, "src/A.tsx").unwrap();

    let count = get_reference_count(&conn).unwrap();
    assert_eq!(count, 0);
}

// ── T-020: transitive dependents A→B→C ───────────────────────────

#[test]
fn get_transitive_dependents_chain() {
    let (conn, _dir) = test_db();
    // A imports B, B imports C → C's dependents = [B(1), A(2)]
    replace_file_references(&conn, "src/A.tsx", &[
        Reference { source_file: "src/A.tsx".into(), target_file: "src/B.tsx".into(), symbol_name: None, ref_kind: RefKind::Named },
    ]).unwrap();
    replace_file_references(&conn, "src/B.tsx", &[
        Reference { source_file: "src/B.tsx".into(), target_file: "src/C.tsx".into(), symbol_name: None, ref_kind: RefKind::Named },
    ]).unwrap();

    let deps = get_transitive_dependents(&conn, "src/C.tsx", 3).unwrap();
    assert_eq!(deps.len(), 2);
    assert_eq!(deps[0], Dependent { file_path: "src/B.tsx".into(), depth: 1 });
    assert_eq!(deps[1], Dependent { file_path: "src/A.tsx".into(), depth: 2 });
}

// ── T-021: circular dependents A→B→A ─────────────────────────────

#[test]
fn get_transitive_dependents_circular() {
    let (conn, _dir) = test_db();
    // A imports B, B imports A → circular
    replace_file_references(&conn, "src/A.tsx", &[
        Reference { source_file: "src/A.tsx".into(), target_file: "src/B.tsx".into(), symbol_name: None, ref_kind: RefKind::Named },
    ]).unwrap();
    replace_file_references(&conn, "src/B.tsx", &[
        Reference { source_file: "src/B.tsx".into(), target_file: "src/A.tsx".into(), symbol_name: None, ref_kind: RefKind::Named },
    ]).unwrap();

    let deps = get_transitive_dependents(&conn, "src/A.tsx", 5).unwrap();
    // Should return B at depth 1, but NOT loop infinitely
    assert_eq!(deps.len(), 1);
    assert_eq!(deps[0], Dependent { file_path: "src/B.tsx".into(), depth: 1 });
}

// ══════════════════════════════════════════════════════════════════
// Incremental Indexing – Phase 1 Storage (T-001 〜 T-010)
// ══════════════════════════════════════════════════════════════════

/// Helper: query chunk IDs for a file (ordered by id ASC).
fn get_chunk_ids(conn: &Connection, file_path: &str) -> Vec<i64> {
    let mut stmt = conn
        .prepare("SELECT id FROM chunks WHERE file_path = ?1 ORDER BY id")
        .unwrap();
    stmt.query_map([file_path], |row| row.get::<_, i64>(0))
        .unwrap()
        .map(|r| r.unwrap())
        .collect()
}

/// Helper: count rows in vec_chunks.
fn vec_chunks_count(conn: &Connection) -> u32 {
    conn.query_row("SELECT COUNT(*) FROM vec_chunks", [], |row| row.get(0))
        .unwrap()
}

// ── T-001: replace_file_chunks_only → chunks あり, vec_chunks なし ──

#[test]
fn replace_file_chunks_only_inserts_chunks_without_embeddings() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Component, name: Some("Button"), content: "function Button() {}", start_line: 1, end_line: 3 },
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useClick"), content: "function useClick() {}", start_line: 5, end_line: 8 },
    ];
    let refs: Vec<Reference> = vec![];

    replace_file_chunks_only(&conn, "src/Button.tsx", &chunks, "hash_a", "import React from 'react'", &refs).unwrap();

    // chunks テーブルに 2 行
    let chunk_count: u32 = conn
        .query_row("SELECT COUNT(*) FROM chunks WHERE file_path = 'src/Button.tsx'", [], |row| row.get(0))
        .unwrap();
    assert_eq!(chunk_count, 2);

    // vec_chunks テーブルに 0 行
    assert_eq!(vec_chunks_count(&conn), 0);

    // file_context も保存されている
    let contexts = get_file_contexts(&conn, &["src/Button.tsx"]).unwrap();
    assert_eq!(contexts["src/Button.tsx"], "import React from 'react'");

    // index_meta の last_indexed_at が更新されている
    let last: String = conn
        .query_row("SELECT value FROM index_meta WHERE key = 'last_indexed_at'", [], |row| row.get(0))
        .unwrap();
    assert!(!last.is_empty());
}

// ── T-002: replace_file_chunks_only with existing embed → 古い vec_chunks も削除 ──

#[test]
fn replace_file_chunks_only_deletes_old_embeddings() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS as usize];

    // 既存の embed 済みチャンクを挿入
    insert_chunk(
        &conn, "src/App.tsx",
        &NewChunk { chunk_type: &ChunkType::Component, name: Some("App"), content: "old code", start_line: 1, end_line: 5 },
        "hash_old", &embedding,
    ).unwrap();
    assert_eq!(vec_chunks_count(&conn), 1);

    // replace_file_chunks_only で同じファイルを上書き
    let new_chunks = vec![
        NewChunk { chunk_type: &ChunkType::Component, name: Some("AppV2"), content: "new code", start_line: 1, end_line: 3 },
    ];
    replace_file_chunks_only(&conn, "src/App.tsx", &new_chunks, "hash_new", "", &[]).unwrap();

    // 古い vec_chunks は削除されている
    assert_eq!(vec_chunks_count(&conn), 0);

    // 新しい chunks は存在する
    let chunk_count: u32 = conn
        .query_row("SELECT COUNT(*) FROM chunks WHERE file_path = 'src/App.tsx'", [], |row| row.get(0))
        .unwrap();
    assert_eq!(chunk_count, 1);

    // 新しいチャンクの name は "AppV2"
    let name: String = conn
        .query_row("SELECT name FROM chunks WHERE file_path = 'src/App.tsx'", [], |row| row.get(0))
        .unwrap();
    assert_eq!(name, "AppV2");
}

// ── T-003: add_embeddings → vec_chunks に行追加、dims 一致 ──

#[test]
fn add_embeddings_inserts_into_vec_chunks() {
    let (conn, _dir) = test_db();

    // chunk-only 状態を作成
    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Component, name: Some("Card"), content: "function Card() {}", start_line: 1, end_line: 3 },
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useCard"), content: "function useCard() {}", start_line: 5, end_line: 8 },
    ];
    replace_file_chunks_only(&conn, "src/Card.tsx", &chunks, "hash_c", "", &[]).unwrap();
    assert_eq!(vec_chunks_count(&conn), 0);

    // chunk_id を取得して embedding を追加
    let ids = get_chunk_ids(&conn, "src/Card.tsx");
    assert_eq!(ids.len(), 2);

    let mut emb1 = vec![0.0_f32; EMBEDDING_DIMS as usize];
    emb1[0] = 1.0;
    let mut emb2 = vec![0.0_f32; EMBEDDING_DIMS as usize];
    emb2[1] = 1.0;

    let embeddings = vec![(ids[0], emb1), (ids[1], emb2)];
    let inserted = add_embeddings(&conn, &embeddings).unwrap();

    assert_eq!(inserted, 2);
    assert_eq!(vec_chunks_count(&conn), 2);
}

// ── T-004: add_embeddings with already embedded chunk_id → スキップ ──

#[test]
fn add_embeddings_skips_already_embedded() {
    let (conn, _dir) = test_db();

    // chunk-only → add_embeddings で 1 件 embed
    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Component, name: Some("Modal"), content: "function Modal() {}", start_line: 1, end_line: 3 },
    ];
    replace_file_chunks_only(&conn, "src/Modal.tsx", &chunks, "hash_m", "", &[]).unwrap();

    let ids = get_chunk_ids(&conn, "src/Modal.tsx");
    let emb = vec![0.5_f32; EMBEDDING_DIMS as usize];
    let embeddings = vec![(ids[0], emb.clone())];

    let inserted = add_embeddings(&conn, &embeddings).unwrap();
    assert_eq!(inserted, 1);

    // 同じ chunk_id で再度 add_embeddings → INSERT OR IGNORE でスキップ
    let embeddings_dup = vec![(ids[0], emb)];
    let inserted_dup = add_embeddings(&conn, &embeddings_dup).unwrap();
    assert_eq!(inserted_dup, 0);

    // vec_chunks は 1 行のまま
    assert_eq!(vec_chunks_count(&conn), 1);
}

// ── T-005: get_unembedded_file_paths with mixed state → 未 embed のみ ──

#[test]
fn get_unembedded_file_paths_returns_only_unembedded() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS as usize];

    // ファイル A: embed 済み（insert_chunk 経由）
    insert_chunk(
        &conn, "src/A.tsx",
        &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code", start_line: 1, end_line: 3 },
        "h1", &embedding,
    ).unwrap();

    // ファイル B: chunk-only（embed なし）
    let chunks_b = vec![
        NewChunk { chunk_type: &ChunkType::Component, name: Some("B"), content: "code b", start_line: 1, end_line: 3 },
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useB"), content: "code b2", start_line: 5, end_line: 8 },
    ];
    replace_file_chunks_only(&conn, "src/B.tsx", &chunks_b, "h2", "", &[]).unwrap();

    // ファイル C: chunk-only（embed なし）
    let chunks_c = vec![
        NewChunk { chunk_type: &ChunkType::Component, name: Some("C"), content: "code c", start_line: 1, end_line: 3 },
    ];
    replace_file_chunks_only(&conn, "src/C.tsx", &chunks_c, "h3", "", &[]).unwrap();

    let unembedded = get_unembedded_file_paths(&conn).unwrap();

    // A は embed 済みなので含まれない
    let paths: Vec<&str> = unembedded.iter().map(|(p, _)| p.as_str()).collect();
    assert!(!paths.contains(&"src/A.tsx"));
    assert!(paths.contains(&"src/B.tsx"));
    assert!(paths.contains(&"src/C.tsx"));

    // B のチャンク数は 2
    let b_count = unembedded.iter().find(|(p, _)| p == "src/B.tsx").unwrap().1;
    assert_eq!(b_count, 2);

    // C のチャンク数は 1
    let c_count = unembedded.iter().find(|(p, _)| p == "src/C.tsx").unwrap().1;
    assert_eq!(c_count, 1);
}

// ── T-006: needs_embedding for chunk-only file → true ──

#[test]
fn needs_embedding_returns_true_for_chunk_only_file() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Component, name: Some("Nav"), content: "function Nav() {}", start_line: 1, end_line: 3 },
    ];
    replace_file_chunks_only(&conn, "src/Nav.tsx", &chunks, "hash_nav", "", &[]).unwrap();

    // hash は同一だが embed がないので true
    assert!(needs_embedding(&conn, "src/Nav.tsx", "hash_nav").unwrap());
}

// ── T-007: needs_embedding for fully embedded file → false ──

#[test]
fn needs_embedding_returns_false_for_fully_embedded() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS as usize];

    insert_chunk(
        &conn, "src/Footer.tsx",
        &NewChunk { chunk_type: &ChunkType::Component, name: Some("Footer"), content: "function Footer() {}", start_line: 1, end_line: 3 },
        "hash_footer", &embedding,
    ).unwrap();

    // hash 同一 + 全チャンク embed 済み → false
    assert!(!needs_embedding(&conn, "src/Footer.tsx", "hash_footer").unwrap());
}

// ── T-008: needs_embedding with hash change → true ──

#[test]
fn needs_embedding_returns_true_when_hash_changed() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS as usize];

    insert_chunk(
        &conn, "src/Header.tsx",
        &NewChunk { chunk_type: &ChunkType::Component, name: Some("Header"), content: "function Header() {}", start_line: 1, end_line: 3 },
        "hash_v1", &embedding,
    ).unwrap();

    // hash が変わったので true（再チャンク + 再 embed 必要）
    assert!(needs_embedding(&conn, "src/Header.tsx", "hash_v2").unwrap());
}

// ── T-009: get_stats with mixed state → embedded_chunks < total_chunks ──

#[test]
fn get_stats_reports_embedded_vs_total_chunks() {
    let (conn, _dir) = test_db();
    let embedding = vec![0.0_f32; EMBEDDING_DIMS as usize];

    // ファイル A: embed 済み (2 chunks)
    insert_chunk(
        &conn, "src/A.tsx",
        &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code", start_line: 1, end_line: 3 },
        "h1", &embedding,
    ).unwrap();
    insert_chunk(
        &conn, "src/A.tsx",
        &NewChunk { chunk_type: &ChunkType::Hook, name: Some("useA"), content: "code", start_line: 5, end_line: 8 },
        "h1", &embedding,
    ).unwrap();

    // ファイル B: chunk-only (1 chunk, embed なし)
    let chunks_b = vec![
        NewChunk { chunk_type: &ChunkType::Component, name: Some("B"), content: "code b", start_line: 1, end_line: 3 },
    ];
    replace_file_chunks_only(&conn, "src/B.tsx", &chunks_b, "h2", "", &[]).unwrap();

    let stats = get_stats(&conn).unwrap();
    assert_eq!(stats.total_chunks, 3);
    assert_eq!(stats.total_files, 2);
    assert_eq!(stats.embedded_chunks, 2); // FR-005: NEW field
    assert!(stats.embedded_chunks < stats.total_chunks);
}

// ── T-010: get_files_by_import_count → 被参照数降順 ──

#[test]
fn get_files_by_import_count_returns_most_imported_first() {
    let (conn, _dir) = test_db();

    // 3 ファイルを chunk-only で登録
    for (path, hash) in [("src/utils.tsx", "h1"), ("src/Button.tsx", "h2"), ("src/App.tsx", "h3")] {
        let chunks = vec![
            NewChunk { chunk_type: &ChunkType::Component, name: Some("X"), content: "code", start_line: 1, end_line: 3 },
        ];
        replace_file_chunks_only(&conn, path, &chunks, hash, "", &[]).unwrap();
    }

    // file_references: utils は 3 回参照、Button は 1 回参照、App は 0 回参照
    replace_file_references(&conn, "src/App.tsx", &[
        Reference { source_file: "src/App.tsx".into(), target_file: "src/utils.tsx".into(), symbol_name: Some("format".into()), ref_kind: RefKind::Named },
        Reference { source_file: "src/App.tsx".into(), target_file: "src/Button.tsx".into(), symbol_name: Some("Button".into()), ref_kind: RefKind::Named },
    ]).unwrap();
    replace_file_references(&conn, "src/Button.tsx", &[
        Reference { source_file: "src/Button.tsx".into(), target_file: "src/utils.tsx".into(), symbol_name: Some("cn".into()), ref_kind: RefKind::Named },
    ]).unwrap();
    // 別ファイルからも utils を参照（3 回目）
    replace_file_references(&conn, "src/Other.tsx", &[
        Reference { source_file: "src/Other.tsx".into(), target_file: "src/utils.tsx".into(), symbol_name: None, ref_kind: RefKind::Namespace },
    ]).unwrap();

    let ordered = get_files_by_import_count(&conn).unwrap();

    // utils (3 refs) > Button (1 ref) > App (0 refs)
    assert_eq!(ordered.len(), 3);
    assert_eq!(ordered[0], "src/utils.tsx");
    assert_eq!(ordered[1], "src/Button.tsx");
    assert_eq!(ordered[2], "src/App.tsx");
}

// ══════════════════════════════════════════════════════════════════
// Search Quality – Phase 1 Storage (T-001 〜 T-003, T-006)
// ══════════════════════════════════════════════════════════════════

// ── T-001: search_by_name matches chunk name with LIKE ──

#[test]
fn search_by_name_matches_keyword() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useAuth"), content: "function useAuth() {}", start_line: 1, end_line: 3 },
        NewChunk { chunk_type: &ChunkType::Component, name: Some("LoginForm"), content: "function LoginForm() {}", start_line: 5, end_line: 10 },
    ];
    replace_file_chunks_only(&conn, "src/hooks.tsx", &chunks, "h1", "", &[]).unwrap();

    let results = search_by_name(&conn, &["auth"], None, &HashSet::new(), 10).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.name.as_deref(), Some("useAuth"));
    assert_eq!(results[0].match_source, MatchSource::NameMatch);
    assert_eq!(results[0].distance, f32::INFINITY);
}

// ── T-002: search_by_name with type_filter → only matching type ──

#[test]
fn search_by_name_filters_by_type() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useBtn"), content: "function useBtn() {}", start_line: 1, end_line: 3 },
        NewChunk { chunk_type: &ChunkType::Component, name: Some("BtnGroup"), content: "function BtnGroup() {}", start_line: 5, end_line: 10 },
    ];
    replace_file_chunks_only(&conn, "src/btn.tsx", &chunks, "h1", "", &[]).unwrap();

    let results = search_by_name(&conn, &["btn"], Some(&[ChunkType::Hook]), &HashSet::new(), 10).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.name.as_deref(), Some("useBtn"));
    assert_eq!(results[0].chunk.chunk_type, ChunkType::Hook);
}

// ── T-003: search_by_name with exclude_ids → excluded chunk absent ──

#[test]
fn search_by_name_excludes_ids() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useAuth"), content: "function useAuth() {}", start_line: 1, end_line: 3 },
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useAuthProvider"), content: "function useAuthProvider() {}", start_line: 5, end_line: 10 },
    ];
    replace_file_chunks_only(&conn, "src/auth.tsx", &chunks, "h1", "", &[]).unwrap();

    let ids = get_chunk_ids(&conn, "src/auth.tsx");
    let mut exclude = HashSet::new();
    exclude.insert(ids[0]);

    let results = search_by_name(&conn, &["auth"], None, &exclude, 10).unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].chunk.name.as_deref(), Some("useAuthProvider"));
}

// ── T-006: get_files_by_import_count with hook/component boost ──

#[test]
fn get_files_by_import_count_boosts_hook_component_files() {
    let (conn, _dir) = test_db();

    // Hook file: 0 import refs
    let hook_chunks = vec![
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useAuth"), content: "function useAuth() {}", start_line: 1, end_line: 3 },
    ];
    replace_file_chunks_only(&conn, "src/useAuth.tsx", &hook_chunks, "h1", "", &[]).unwrap();

    // TypeDef file: 0 import refs
    let typedef_chunks = vec![
        NewChunk { chunk_type: &ChunkType::TypeDef, name: Some("AuthConfig"), content: "interface AuthConfig {}", start_line: 1, end_line: 3 },
    ];
    replace_file_chunks_only(&conn, "src/types.tsx", &typedef_chunks, "h2", "", &[]).unwrap();

    // Component file: 0 import refs
    let comp_chunks = vec![
        NewChunk { chunk_type: &ChunkType::Component, name: Some("LoginButton"), content: "function LoginButton() {}", start_line: 1, end_line: 3 },
    ];
    replace_file_chunks_only(&conn, "src/LoginButton.tsx", &comp_chunks, "h3", "", &[]).unwrap();

    let ordered = get_files_by_import_count(&conn).unwrap();

    // hook/component files (+3 boost) should come before type_def (no boost)
    assert_eq!(ordered.len(), 3);
    // First two should be hook or component files (both have +3 boost)
    let boosted: Vec<&str> = ordered.iter().take(2).map(|s| s.as_str()).collect();
    assert!(boosted.contains(&"src/useAuth.tsx"), "hook file should be boosted: {ordered:?}");
    assert!(boosted.contains(&"src/LoginButton.tsx"), "component file should be boosted: {ordered:?}");
    // Last should be typedef
    assert_eq!(ordered[2], "src/types.tsx", "typedef file should be last: {ordered:?}");
}

// ── search_similar sets MatchSource::Semantic ──

#[test]
fn search_similar_sets_semantic_match_source() {
    let (conn, _dir) = test_db();
    let mut emb = vec![0.0_f32; 768];
    emb[0] = 1.0;

    insert_chunk(&conn, "src/A.tsx", &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code", start_line: 1, end_line: 3 }, "h1", &emb).unwrap();

    let results = search_similar(&conn, &emb, 10, 0).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].match_source, MatchSource::Semantic);
}

// ── search_by_name returns empty for empty keywords ──

#[test]
fn search_by_name_empty_keywords_returns_empty() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useAuth"), content: "code", start_line: 1, end_line: 3 },
    ];
    replace_file_chunks_only(&conn, "src/auth.tsx", &chunks, "h1", "", &[]).unwrap();

    let results = search_by_name(&conn, &[], None, &HashSet::new(), 10).unwrap();
    assert!(results.is_empty());
}

// ══════════════════════════════════════════════════════════════════
// Rerank – Phase 1 Storage (T-001, T-002, T-006)
// ══════════════════════════════════════════════════════════════════

// ── T-001 (rerank spec): search_similar sets initial score ──

#[test]
fn search_similar_sets_initial_score() {
    let (conn, _dir) = test_db();
    let mut emb = vec![0.0_f32; EMBEDDING_DIMS as usize];
    emb[0] = 1.0;

    insert_chunk(
        &conn, "src/A.tsx",
        &NewChunk { chunk_type: &ChunkType::Component, name: Some("A"), content: "code", start_line: 1, end_line: 3 },
        "h1", &emb,
    ).unwrap();

    let results = search_similar(&conn, &emb, 10, 0).unwrap();
    assert_eq!(results.len(), 1);

    // score should be initialized to 1.0 / (1.0 + distance)
    let expected_score = 1.0 / (1.0 + results[0].distance);
    assert!((results[0].score - expected_score).abs() < 1e-6,
        "expected score {expected_score}, got {}", results[0].score);
}

// ── T-002 (rerank spec): search_by_name sets score 0.5 ──

#[test]
fn search_by_name_sets_base_score() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useAuth"), content: "function useAuth() {}", start_line: 1, end_line: 3 },
    ];
    replace_file_chunks_only(&conn, "src/auth.tsx", &chunks, "h1", "", &[]).unwrap();

    let results = search_by_name(&conn, &["auth"], None, &HashSet::new(), 10).unwrap();
    assert_eq!(results.len(), 1);
    assert!((results[0].score - 0.5).abs() < 1e-6,
        "NameMatch score should be 0.5, got {}", results[0].score);
}

// ── T-006 (rerank spec): get_import_counts batch SQL ──

#[test]
fn get_import_counts_returns_correct_counts() {
    let (conn, _dir) = test_db();

    // Set up files with references
    for (path, hash) in [("src/utils.tsx", "h1"), ("src/Button.tsx", "h2"), ("src/App.tsx", "h3")] {
        let chunks = vec![
            NewChunk { chunk_type: &ChunkType::Component, name: Some("X"), content: "code", start_line: 1, end_line: 3 },
        ];
        replace_file_chunks_only(&conn, path, &chunks, hash, "", &[]).unwrap();
    }

    // utils is imported 3 times, Button 1 time, App 0 times
    replace_file_references(&conn, "src/App.tsx", &[
        Reference { source_file: "src/App.tsx".into(), target_file: "src/utils.tsx".into(), symbol_name: None, ref_kind: RefKind::Named },
        Reference { source_file: "src/App.tsx".into(), target_file: "src/Button.tsx".into(), symbol_name: None, ref_kind: RefKind::Named },
    ]).unwrap();
    replace_file_references(&conn, "src/Button.tsx", &[
        Reference { source_file: "src/Button.tsx".into(), target_file: "src/utils.tsx".into(), symbol_name: None, ref_kind: RefKind::Named },
    ]).unwrap();
    replace_file_references(&conn, "src/Other.tsx", &[
        Reference { source_file: "src/Other.tsx".into(), target_file: "src/utils.tsx".into(), symbol_name: None, ref_kind: RefKind::Named },
    ]).unwrap();

    let counts = get_import_counts(&conn, &["src/utils.tsx", "src/Button.tsx", "src/App.tsx"]).unwrap();

    assert_eq!(counts.len(), 3);
    assert_eq!(counts["src/utils.tsx"], 3);
    assert_eq!(counts["src/Button.tsx"], 1);
    assert_eq!(counts["src/App.tsx"], 0);
}

#[test]
fn get_import_counts_returns_empty_for_empty_input() {
    let (conn, _dir) = test_db();
    let counts = get_import_counts(&conn, &[]).unwrap();
    assert!(counts.is_empty());
}

// ── search_by_name respects limit ──

#[test]
fn search_by_name_respects_limit() {
    let (conn, _dir) = test_db();

    let chunks = vec![
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useAuthA"), content: "code", start_line: 1, end_line: 3 },
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useAuthB"), content: "code", start_line: 5, end_line: 8 },
        NewChunk { chunk_type: &ChunkType::Hook, name: Some("useAuthC"), content: "code", start_line: 10, end_line: 13 },
    ];
    replace_file_chunks_only(&conn, "src/auth.tsx", &chunks, "h1", "", &[]).unwrap();

    let results = search_by_name(&conn, &["auth"], None, &HashSet::new(), 2).unwrap();
    assert_eq!(results.len(), 2);
}
