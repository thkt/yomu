use std::path::Path;

use super::*;

#[test]
fn mean_pooling_excludes_masked_tokens() {
    #[rustfmt::skip]
    let data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0,       // token 0, mask=1
        5.0, 6.0, 7.0, 8.0,       // token 1, mask=1
        100.0, 100.0, 100.0, 100.0, // token 2, mask=0 (excluded)
    ];
    let mask = vec![1u32, 1, 0];
    let result = mean_pooling(&data, 3, 4, &mask);
    assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn mean_pooling_all_masked() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mask = vec![0u32, 0];
    let result = mean_pooling(&data, 2, 2, &mask);
    assert_eq!(result, vec![0.0, 0.0]);
}

#[test]
fn l2_normalize_produces_unit_norm() {
    let mut v = vec![3.0f32, 4.0];
    l2_normalize(&mut v);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6, "norm should be 1.0, got {norm}");
    assert!((v[0] - 0.6).abs() < 1e-6);
    assert!((v[1] - 0.8).abs() < 1e-6);
}

#[test]
fn l2_normalize_zero_vector() {
    let mut v = vec![0.0f32, 0.0];
    l2_normalize(&mut v);
    assert_eq!(v, vec![0.0, 0.0]);
}

#[test]
fn mean_pooling_single_token() {
    let data = vec![1.0f32, 2.0];
    let mask = vec![1u32];
    let result = mean_pooling(&data, 1, 2, &mask);
    assert_eq!(result, vec![1.0, 2.0]);
}

#[test]
fn postprocess_embedding_zero_seq_len() {
    let result = postprocess_embedding(&[], 0, &[]);
    let err = result.unwrap_err();
    assert!(
        matches!(err, EmbedError::DimensionMismatch { expected: _, actual: 0 }),
        "expected DimensionMismatch with actual=0, got: {err}"
    );
}

#[test]
fn postprocess_embedding_wrong_dims() {
    // 3 floats / 1 token = hidden_size 3, but EMBEDDING_DIMS is 768
    let data = vec![1.0f32, 2.0, 3.0];
    let mask = vec![1u32];
    let result = postprocess_embedding(&data, 1, &mask);
    let err = result.unwrap_err();
    assert!(
        matches!(err, EmbedError::DimensionMismatch { expected: 768, actual: 3 }),
        "expected DimensionMismatch{{768, 3}}, got: {err}"
    );
}

#[test]
fn validate_partial_download_reports_missing_file() {
    let dir = tempfile::tempdir().unwrap();
    // Only model.safetensors exists; config.json and tokenizer.json are missing
    std::fs::write(dir.path().join("model.safetensors"), b"fake").unwrap();
    let paths = ModelPaths::from_dir(dir.path());
    let err = paths.validate().unwrap_err();
    match &err {
        EmbedError::ModelNotFound { path } => {
            assert!(
                path.ends_with("config.json"),
                "should report config.json as missing, got: {path:?}"
            );
        }
        other => panic!("expected ModelNotFound, got: {other}"),
    }
}

#[test]
fn embedder_new_model_not_found() {
    let paths = ModelPaths::from_dir(Path::new("/nonexistent/path"));
    let err = Embedder::new(&paths).unwrap_err();
    assert!(
        matches!(err, EmbedError::ModelNotFound { .. }),
        "expected ModelNotFound, got: {err}"
    );
    assert!(
        err.to_string().contains("yomu index"),
        "error message should mention `yomu index`: {err}"
    );
}

#[tokio::test]
async fn mock_embedder_query_returns_correct_dims() {
    let embedder = MockEmbedder;
    let result = embedder.embed_query("test").await.unwrap();
    assert_eq!(result.len(), EMBEDDING_DIMS as usize);
    assert_eq!(result[0], 1.0);
}

#[tokio::test]
async fn mock_embedder_documents_returns_distinct_vectors() {
    let embedder = MockEmbedder;
    let texts = vec!["a".to_string(), "b".to_string()];
    let result = embedder.embed_documents(&texts).await.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0][0], 1.0);
    assert_eq!(result[1][1], 1.0);
}

#[tokio::test]
async fn failing_embedder_all_fail() {
    let embedder = FailingEmbedder::all_fail("test failure");
    assert!(embedder.embed_query("test").await.is_err());
    let texts = vec!["a".to_string()];
    assert!(embedder.embed_documents(&texts).await.is_err());
}

#[tokio::test]
async fn failing_embedder_query_only() {
    let embedder = FailingEmbedder::query_only("test failure");
    assert!(embedder.embed_query("test").await.is_err());
    let texts = vec!["a".to_string()];
    assert!(embedder.embed_documents(&texts).await.is_ok());
}

#[tokio::test]
#[ignore] // requires model download
async fn embed_query_returns_768_dims() {
    let paths = download_model().expect("download model");
    let embedder = Embedder::new(&paths).expect("load model");
    let embedding = embedder.embed_query("authentication logic").await.unwrap();
    assert_eq!(
        embedding.len(),
        768,
        "expected 768 dims, got {}",
        embedding.len()
    );
}

#[tokio::test]
#[ignore] // requires model download
async fn embed_documents_batch() {
    let paths = download_model().expect("download model");
    let embedder = Embedder::new(&paths).expect("load model");
    let texts = vec![
        "function useAuth() { return user; }".to_string(),
        "function Button() { return <div/>; }".to_string(),
    ];
    let embeddings = embedder.embed_documents(&texts).await.unwrap();
    assert_eq!(embeddings.len(), 2);
    assert_eq!(embeddings[0].len(), 768);
}
