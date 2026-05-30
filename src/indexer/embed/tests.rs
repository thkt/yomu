use super::*;

// T-360: validate_rejects_empty_chunks
#[test]
fn validate_rejects_empty_chunks() {
    let embs = vec![ChunkedEmbedding::new(vec![])];
    assert!(validate_chunked_embeddings(embs).is_err());
}

// T-361: validate_passes_non_empty_chunks
#[test]
fn validate_passes_non_empty_chunks() {
    let embs = vec![
        ChunkedEmbedding::new(vec![vec![1.0_f32; 3]]),
        ChunkedEmbedding::new(vec![vec![2.0_f32; 3], vec![3.0_f32; 3]]),
    ];
    assert!(validate_chunked_embeddings(embs).is_ok());
}
