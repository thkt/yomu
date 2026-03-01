use std::pin::Pin;
use std::time::Duration;

use reqwest::Client;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";
const EMBED_PATH: &str = "/models/gemini-embedding-001:embedContent";
const BATCH_PATH: &str = "/models/gemini-embedding-001:batchEmbedContents";

pub use crate::storage::EMBEDDING_DIMS;
const MAX_RETRIES: u32 = 5;
const BATCH_SIZE: usize = 100;

fn is_retryable(status: u16) -> bool {
    matches!(status, 500 | 503)
}

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey")]
    ApiKeyNotSet,
    #[error("Gemini API error ({status}): {message}")]
    Api { status: u16, message: String },
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("Unexpected response format")]
    BadResponse,
}

type EmbedFuture<'a, T> = Pin<Box<dyn std::future::Future<Output = Result<T, EmbedError>> + Send + 'a>>;

/// Text embedding provider. Returns [`EMBEDDING_DIMS`]-dimensional f32 vectors.
///
/// Object-safe: methods return boxed futures so `dyn Embed` can be used.
///
/// # Contract
/// Implementations MUST return vectors of exactly [`EMBEDDING_DIMS`] elements.
/// Returning vectors of other lengths may cause storage layer errors or corrupted search results.
pub trait Embed: Send + Sync {
    /// Embed a single query string (uses RETRIEVAL_QUERY task type).
    fn embed_query<'a>(&'a self, text: &'a str) -> EmbedFuture<'a, Vec<f32>>;
    /// Embed multiple documents in batch (uses RETRIEVAL_DOCUMENT task type).
    fn embed_documents<'a>(&'a self, texts: &'a [String]) -> EmbedFuture<'a, Vec<Vec<f32>>>;
}

#[derive(Clone)]
pub struct Embedder {
    http: Client,
    api_key: String,
    embed_url: String,
    batch_url: String,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder")
            .field("api_key", &"[REDACTED]")
            .finish()
    }
}

impl Embedder {
    pub fn from_env(http: Client) -> Result<Self, EmbedError> {
        Self::from_env_with(http, |k| std::env::var(k).ok())
    }

    pub fn from_env_with(
        http: Client,
        get_var: impl Fn(&str) -> Option<String>,
    ) -> Result<Self, EmbedError> {
        let api_key = get_var("GEMINI_API_KEY").ok_or(EmbedError::ApiKeyNotSet)?;
        if api_key.is_empty() {
            return Err(EmbedError::ApiKeyNotSet);
        }
        let embed_url = format!("{DEFAULT_BASE_URL}{EMBED_PATH}");
        let batch_url = format!("{DEFAULT_BASE_URL}{BATCH_PATH}");
        Ok(Self { http, api_key, embed_url, batch_url })
    }

    #[cfg(test)]
    pub(crate) fn with_base_url(http: Client, api_key: String, base_url: String) -> Self {
        let embed_url = format!("{base_url}{EMBED_PATH}");
        let batch_url = format!("{base_url}{BATCH_PATH}");
        Self { http, api_key, embed_url, batch_url }
    }

    async fn post_with_retry(
        &self,
        url: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, EmbedError> {
        const PER_REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
        const TOTAL_RETRY_TIMEOUT: Duration = Duration::from_secs(180);

        let result = tokio::time::timeout(TOTAL_RETRY_TIMEOUT, async {
            let api_key_header = {
                let mut v = reqwest::header::HeaderValue::from_str(&self.api_key)
                    .map_err(|_| EmbedError::ApiKeyNotSet)?;
                v.set_sensitive(true);
                v
            };
            let mut last_status = 0;
            for attempt in 0..=MAX_RETRIES {
                if attempt > 0 {
                    let base = 500 * 2u64.pow(attempt - 1);
                    let jitter = (std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .subsec_nanos() as u64)
                        % (base / 2 + 1);
                    let delay = Duration::from_millis(base + jitter);
                    tracing::warn!(
                        attempt,
                        max_retries = MAX_RETRIES,
                        status = last_status,
                        url,
                        "Gemini API transient error, retrying in {delay:?}"
                    );
                    tokio::time::sleep(delay).await;
                }

                let resp = self
                    .http
                    .post(url)
                    .header("x-goog-api-key", api_key_header.clone())
                    .json(body)
                    .timeout(PER_REQUEST_TIMEOUT)
                    .send()
                    .await?;

                if resp.status().is_success() {
                    return resp.json().await.map_err(EmbedError::Network);
                }

                let status = resp.status().as_u16();
                last_status = status;
                if is_retryable(status) {
                    let resp_body = resp.text().await.unwrap_or_default();
                    let safe_body = resp_body.replace(&self.api_key, "[REDACTED]");
                    let truncated = truncate_str(&safe_body, 500);
                    tracing::warn!(status, body = truncated, "Gemini API error response");
                } else {
                    tracing::warn!(status, "Gemini API error response");
                }
                if !is_retryable(status) || attempt == MAX_RETRIES {
                    return Err(EmbedError::Api {
                        status,
                        message: sanitize_api_error(status),
                    });
                }
            }
            unreachable!()
        })
        .await;

        match result {
            Ok(inner) => inner,
            Err(_) => Err(EmbedError::Api {
                status: 0,
                message: "Gemini API did not respond within retry window (180s). Check network connectivity.".into(),
            }),
        }
    }
}

impl Embed for Embedder {
    fn embed_query<'a>(&'a self, text: &'a str) -> EmbedFuture<'a, Vec<f32>> {
        Box::pin(async move {
            let body = serde_json::json!({
                "model": "models/gemini-embedding-001",
                "content": { "parts": [{ "text": text }] },
                "taskType": "RETRIEVAL_QUERY",
                "outputDimensionality": EMBEDDING_DIMS
            });

            let json = self.post_with_retry(&self.embed_url, &body).await?;
            parse_single_embedding(&json)
        })
    }

    fn embed_documents<'a>(&'a self, texts: &'a [String]) -> EmbedFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            if texts.is_empty() {
                return Ok(Vec::new());
            }

            let mut all_embeddings = Vec::with_capacity(texts.len());

            for batch in texts.chunks(BATCH_SIZE) {
                let requests: Vec<serde_json::Value> = batch
                    .iter()
                    .map(|text| {
                        serde_json::json!({
                            "model": "models/gemini-embedding-001",
                            "content": { "parts": [{ "text": text }] },
                            "taskType": "RETRIEVAL_DOCUMENT",
                            "outputDimensionality": EMBEDDING_DIMS
                        })
                    })
                    .collect();

                let body = serde_json::json!({ "requests": requests });
                let json = self.post_with_retry(&self.batch_url, &body).await?;
                let embeddings = parse_batch_embeddings(&json)?;
                all_embeddings.extend(embeddings);
            }

            Ok(all_embeddings)
        })
    }
}

fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn sanitize_api_error(status: u16) -> String {
    match status {
        400 => "Bad request to Gemini API".to_string(),
        401 | 403 => "Authentication failed — check GEMINI_API_KEY".to_string(),
        404 => "Gemini API endpoint not found".to_string(),
        429 => "Gemini API rate limit exceeded".to_string(),
        _ if status >= 500 => format!("Gemini API server error ({status})"),
        _ => format!("Gemini API error ({status})"),
    }
}

fn parse_values(arr: &[serde_json::Value]) -> Result<Vec<f32>, EmbedError> {
    arr.iter()
        .map(|v| {
            v.as_f64()
                .map(|f| f as f32)
                .filter(|f| f.is_finite())
                .ok_or(EmbedError::BadResponse)
        })
        .collect()
}

fn parse_single_embedding(json: &serde_json::Value) -> Result<Vec<f32>, EmbedError> {
    let arr = json
        .get("embedding")
        .and_then(|e| e.get("values"))
        .and_then(|v| v.as_array())
        .ok_or(EmbedError::BadResponse)?;
    let values = parse_values(arr)?;
    if values.len() != EMBEDDING_DIMS as usize {
        tracing::warn!(expected = EMBEDDING_DIMS, actual = values.len(), "unexpected embedding dimension");
        return Err(EmbedError::BadResponse);
    }
    Ok(values)
}

fn parse_batch_embeddings(json: &serde_json::Value) -> Result<Vec<Vec<f32>>, EmbedError> {
    let items = json
        .get("embeddings")
        .and_then(|e| e.as_array())
        .ok_or(EmbedError::BadResponse)?;

    items
        .iter()
        .map(|item| {
            let vals = item
                .get("values")
                .and_then(|v| v.as_array())
                .ok_or(EmbedError::BadResponse)?;
            let values = parse_values(vals)?;
            if values.len() != EMBEDDING_DIMS as usize {
                tracing::warn!(expected = EMBEDDING_DIMS, actual = values.len(), "unexpected embedding dimension");
                return Err(EmbedError::BadResponse);
            }
            Ok(values)
        })
        .collect()
}

/// Mock embedder returning deterministic vectors for offline tests.
///
/// - `embed_query` returns a unit vector (v[0] = 1.0) for meaningful distance tests.
/// - `embed_documents` returns index-based unit vectors so different documents
///   produce distinct embeddings, enabling search ordering verification.
#[cfg(any(test, feature = "test-support"))]
pub(crate) struct MockEmbedder;

#[cfg(any(test, feature = "test-support"))]
impl Embed for MockEmbedder {
    fn embed_query<'a>(&'a self, _text: &'a str) -> EmbedFuture<'a, Vec<f32>> {
        Box::pin(async {
            let mut v = vec![0.0_f32; EMBEDDING_DIMS as usize];
            v[0] = 1.0;
            Ok(v)
        })
    }

    fn embed_documents<'a>(&'a self, texts: &'a [String]) -> EmbedFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let mut v = vec![0.0_f32; EMBEDDING_DIMS as usize];
                    v[i % EMBEDDING_DIMS as usize] = 1.0;
                    v
                })
                .collect())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::{Mock, MockServer, ResponseTemplate};
    use wiremock::matchers::method;

    #[test]
    fn from_env_returns_error_without_api_key() {
        let result = Embedder::from_env_with(Client::new(), |_| None);
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("GEMINI_API_KEY not set"),
            "got: {err}"
        );
    }

    #[test]
    fn from_env_returns_error_for_empty_api_key() {
        let result = Embedder::from_env_with(Client::new(), |_| Some(String::new()));
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("GEMINI_API_KEY not set"),
            "got: {err}"
        );
    }

    #[test]
    fn from_env_succeeds_with_api_key() {
        let result = Embedder::from_env_with(Client::new(), |_| Some("test-key".to_string()));
        assert!(result.is_ok());
    }

    #[test]
    fn parse_single_embedding_ok() {
        let mut values = vec![0.0_f64; EMBEDDING_DIMS as usize];
        values[0] = 0.1;
        values[1] = 0.2;
        let json = serde_json::json!({
            "embedding": { "values": values }
        });
        let result = parse_single_embedding(&json).unwrap();
        assert_eq!(result.len(), EMBEDDING_DIMS as usize);
        assert!((result[0] - 0.1).abs() < 0.001);
    }

    #[test]
    fn parse_single_embedding_rejects_wrong_dims() {
        let json = serde_json::json!({
            "embedding": { "values": [0.1, 0.2, 0.3] }
        });
        assert!(parse_single_embedding(&json).is_err());
    }

    #[test]
    fn parse_batch_embeddings_ok() {
        let values = vec![0.0_f64; EMBEDDING_DIMS as usize];
        let json = serde_json::json!({
            "embeddings": [
                { "values": values.clone() },
                { "values": values }
            ]
        });
        let result = parse_batch_embeddings(&json).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), EMBEDDING_DIMS as usize);
    }

    #[test]
    fn parse_batch_embeddings_rejects_wrong_dims() {
        let json = serde_json::json!({
            "embeddings": [
                { "values": [0.1, 0.2] }
            ]
        });
        assert!(parse_batch_embeddings(&json).is_err());
    }

    #[test]
    fn parse_single_embedding_rejects_non_numeric() {
        let json = serde_json::json!({
            "embedding": {
                "values": [0.1, "bad", 0.3]
            }
        });
        assert!(parse_single_embedding(&json).is_err());
    }

    #[test]
    fn parse_batch_embeddings_rejects_malformed_entry() {
        let json = serde_json::json!({
            "embeddings": [
                { "values": [0.1, 0.2] },
                { "broken": true }
            ]
        });
        assert!(parse_batch_embeddings(&json).is_err());
    }

    #[test]
    fn parse_batch_embeddings_rejects_non_numeric_value() {
        let json = serde_json::json!({
            "embeddings": [
                { "values": [0.1, null] }
            ]
        });
        assert!(parse_batch_embeddings(&json).is_err());
    }

    #[test]
    fn retryable_status_codes() {
        assert!(!is_retryable(429));
        assert!(is_retryable(500));
        assert!(is_retryable(503));
        assert!(!is_retryable(400));
        assert!(!is_retryable(401));
        assert!(!is_retryable(404));
    }

    fn mock_embedding_response() -> serde_json::Value {
        let values = vec![0.0_f64; EMBEDDING_DIMS as usize];
        serde_json::json!({ "embedding": { "values": values } })
    }

    #[tokio::test]
    async fn post_with_retry_succeeds_on_first_try() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(mock_embedding_response()))
            .expect(1)
            .mount(&server)
            .await;

        let embedder = Embedder::with_base_url(Client::new(), "test-key".into(), server.uri());
        let result = embedder.embed_query("test").await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), EMBEDDING_DIMS as usize);
    }

    #[tokio::test]
    async fn post_with_retry_does_not_retry_on_429() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .expect(1)
            .mount(&server)
            .await;

        let embedder = Embedder::with_base_url(Client::new(), "test-key".into(), server.uri());
        let result = embedder.embed_query("test").await;
        assert!(result.is_err(), "429 should fail immediately without retry");
        match result.unwrap_err() {
            EmbedError::Api { status, .. } => assert_eq!(status, 429),
            other => panic!("expected Api error with 429, got: {other}"),
        }
    }

    #[tokio::test]
    async fn post_with_retry_aborts_on_non_retryable_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .expect(1)
            .mount(&server)
            .await;

        let embedder = Embedder::with_base_url(Client::new(), "test-key".into(), server.uri());
        let result = embedder.embed_query("test").await;
        assert!(result.is_err());
        match result.unwrap_err() {
            EmbedError::Api { status, .. } => assert_eq!(status, 400),
            other => panic!("expected Api error, got: {other}"),
        }
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY"]
    async fn embed_query_returns_768_dims() {
        let embedder = Embedder::from_env(Client::new()).unwrap();
        let embedding = embedder.embed_query("authentication logic").await.unwrap();
        assert_eq!(embedding.len(), 768, "expected 768 dims, got {}", embedding.len());
    }

    #[tokio::test]
    #[ignore = "requires GEMINI_API_KEY"]
    async fn embed_documents_batch() {
        let embedder = Embedder::from_env(Client::new()).unwrap();
        let texts = vec![
            "function useAuth() { return user; }".to_string(),
            "function Button() { return <div/>; }".to_string(),
        ];
        let embeddings = embedder.embed_documents(&texts).await.unwrap();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 768);
    }
}
