use super::*;

// T-361: validate_passes_non_empty_chunks
#[test]
fn validate_passes_non_empty_chunks() {
    let embs = vec![
        ChunkedEmbedding::try_new(vec![vec![1.0_f32; 3]]).unwrap(),
        ChunkedEmbedding::try_new(vec![vec![2.0_f32; 3], vec![3.0_f32; 3]]).unwrap(),
    ];
    assert!(validate_chunked_embeddings(embs).is_ok());
}
