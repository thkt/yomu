#[cfg(all(feature = "candle", feature = "mlx"))]
compile_error!("features `candle` and `mlx` are mutually exclusive — enable only one");

#[cfg(not(any(feature = "candle", feature = "mlx")))]
compile_error!("enable either `mlx` or `candle` feature");

#[cfg(feature = "candle")]
mod candle;
#[cfg(feature = "mlx")]
mod mlx;

#[cfg(any(test, feature = "test-support"))]
mod test_support;

#[cfg(test)]
mod tests;

#[cfg(feature = "candle")]
use self::candle::EmbedderInner;
#[cfg(feature = "mlx")]
use self::mlx::EmbedderInner;

#[cfg(any(test, feature = "test-support"))]
pub(crate) use test_support::{
    AlternatingEmbedder, FailingEmbedder, MismatchEmbedder, MockEmbedder,
};

use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Mutex;

pub use crate::storage::EMBEDDING_DIMS;

pub(crate) const QUERY_PREFIX: &str = "検索クエリ: ";
pub(crate) const DOCUMENT_PREFIX: &str = "検索文書: ";
const MODEL_REPO: &str = "cl-nagoya/ruri-v3-310m";
const MODEL_REVISION: &str = "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3";

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("embedding model not available")]
    ModelNotAvailable,
    #[error("model not found at {path}. Run `yomu index` to download.")]
    ModelNotFound { path: PathBuf },
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("inference error: {0}")]
    Inference(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
}

pub(crate) type EmbedFuture<'a, T> =
    Pin<Box<dyn std::future::Future<Output = Result<T, EmbedError>> + Send + 'a>>;

/// Code embedding provider. Returns [`EMBEDDING_DIMS`]-dimensional f32 vectors.
///
/// Object-safe: methods return boxed futures so `dyn Embed` can be used.
///
/// # Contract
/// Implementations MUST return vectors of exactly [`EMBEDDING_DIMS`] elements.
pub trait Embed: Send + Sync {
    fn embed_query<'a>(&'a self, text: &'a str) -> EmbedFuture<'a, Vec<f32>>;
    fn embed_documents<'a>(&'a self, texts: &'a [String]) -> EmbedFuture<'a, Vec<Vec<f32>>>;
}

#[derive(Debug, Clone)]
pub struct ModelPaths {
    pub model: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
}

impl ModelPaths {
    #[cfg(test)]
    pub(crate) fn from_dir(dir: &std::path::Path) -> Self {
        Self {
            model: dir.join("model.safetensors"),
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), EmbedError> {
        for path in [&self.model, &self.config, &self.tokenizer] {
            if !path.exists() {
                return Err(EmbedError::ModelNotFound { path: path.clone() });
            }
        }
        Ok(())
    }
}

/// Download model files from Hugging Face Hub (cached after first download).
pub fn download_model() -> Result<ModelPaths, EmbedError> {
    let api = hf_hub::api::sync::Api::new()
        .map_err(|e| EmbedError::Inference(format!("HF Hub init failed: {e}")))?;
    let repo = api.repo(hf_hub::Repo::with_revision(
        MODEL_REPO.to_string(),
        hf_hub::RepoType::Model,
        MODEL_REVISION.to_string(),
    ));

    let model = repo
        .get("model.safetensors")
        .map_err(|e| EmbedError::Inference(format!("model download failed: {e}")))?;
    let config = repo
        .get("config.json")
        .map_err(|e| EmbedError::Inference(format!("config download failed: {e}")))?;
    let tokenizer = repo
        .get("tokenizer.json")
        .map_err(|e| EmbedError::Inference(format!("tokenizer download failed: {e}")))?;

    Ok(ModelPaths {
        model,
        config,
        tokenizer,
    })
}

pub(crate) fn mean_pooling(
    data: &[f32],
    seq_len: usize,
    hidden_size: usize,
    attention_mask: &[u32],
) -> Vec<f32> {
    let mut result = vec![0.0f32; hidden_size];
    let mut mask_sum = 0.0f32;

    for (t, &m) in attention_mask.iter().enumerate().take(seq_len) {
        if m > 0 {
            let mf = m as f32;
            let offset = t * hidden_size;
            for d in 0..hidden_size {
                result[d] += data[offset + d] * mf;
            }
            mask_sum += mf;
        }
    }

    if mask_sum > 0.0 {
        for v in &mut result {
            *v /= mask_sum;
        }
    }

    result
}

pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub(crate) fn postprocess_embedding(
    flat: &[f32],
    seq_len: usize,
    attention_mask: &[u32],
) -> Result<Vec<f32>, EmbedError> {
    if seq_len == 0 {
        return Err(EmbedError::DimensionMismatch {
            expected: EMBEDDING_DIMS as usize,
            actual: 0,
        });
    }
    let hidden_size = flat.len() / seq_len;
    if hidden_size != EMBEDDING_DIMS as usize {
        return Err(EmbedError::DimensionMismatch {
            expected: EMBEDDING_DIMS as usize,
            actual: hidden_size,
        });
    }
    let mut pooled = mean_pooling(flat, seq_len, hidden_size, attention_mask);
    l2_normalize(&mut pooled);
    Ok(pooled)
}

pub(crate) fn read_config<T: serde::de::DeserializeOwned>(
    path: &std::path::Path,
) -> Result<T, EmbedError> {
    let text =
        std::fs::read_to_string(path).map_err(|e| EmbedError::Inference(e.to_string()))?;
    serde_json::from_str(&text)
        .map_err(|e| EmbedError::Inference(format!("config.json parse error: {e}")))
}

pub(crate) fn load_tokenizer(
    path: &std::path::Path,
) -> Result<tokenizers::Tokenizer, EmbedError> {
    tokenizers::Tokenizer::from_file(path).map_err(|e| EmbedError::Tokenizer(e.to_string()))
}

pub(crate) struct TokenizedInput {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub seq_len: usize,
}

pub(crate) fn tokenize_with_prefix(
    tokenizer: &tokenizers::Tokenizer,
    text: &str,
    prefix: &str,
) -> Result<TokenizedInput, EmbedError> {
    let prefixed = format!("{prefix}{text}");
    let encoding = tokenizer
        .encode(prefixed, true)
        .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;
    let input_ids = encoding.get_ids().to_vec();
    let attention_mask = encoding.get_attention_mask().to_vec();
    let seq_len = input_ids.len();
    Ok(TokenizedInput {
        input_ids,
        attention_mask,
        seq_len,
    })
}

pub struct Embedder {
    inner: Mutex<EmbedderInner>,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

impl Embedder {
    pub fn new(paths: &ModelPaths) -> Result<Self, EmbedError> {
        let inner = EmbedderInner::new(paths)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }
}

impl Embed for Embedder {
    fn embed_query<'a>(&'a self, text: &'a str) -> EmbedFuture<'a, Vec<f32>> {
        // Compute synchronously, wrap result in a ready future.
        // MutexGuard is dropped before the future is returned.
        let result = {
            let mut inner = self.inner.lock().unwrap();
            inner.embed_with_prefix(text, QUERY_PREFIX)
        };
        Box::pin(async move { result })
    }

    fn embed_documents<'a>(&'a self, texts: &'a [String]) -> EmbedFuture<'a, Vec<Vec<f32>>> {
        let result = {
            let mut inner = self.inner.lock().unwrap();
            inner.embed_batch(texts)
        };
        Box::pin(async move { result })
    }
}
