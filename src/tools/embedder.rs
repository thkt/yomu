use std::sync::Arc;

use rurico::embed::{ChunkedEmbedding, Embed, EmbedError};

// Real-model loading is compiled only into the production binary. Under
// `--features test-support` the embedder is a deterministic stub (see the
// `cfg(feature = "test-support")` variant of `try_load_embedder`), so these
// model-cache imports would be unused and trip `-D warnings`.
#[cfg(not(feature = "test-support"))]
use amici::model::embedder::try_load_embedder_with;
#[cfg(not(feature = "test-support"))]
use rurico::embed::{ModelId, cached_artifacts};

#[cfg(feature = "test-support")]
use rurico::embed::MockEmbedder;
#[cfg(feature = "test-support")]
use std::env;

pub(super) use amici::model::embedder::{DegradedReason, degraded_reason_user_note};

use super::Yomu;

#[cfg(test)]
use std::cell::RefCell;
#[cfg(test)]
use std::path::PathBuf;
#[cfg(test)]
use std::sync::{Mutex, OnceLock};

pub(super) fn record_embedder_warning(reason: DegradedReason, detail: &str) {
    tracing::warn!(reason = ?reason, detail, "Embedder unavailable, using text search only");
    #[cfg(test)]
    RECORDED_WARNINGS.with(|w| w.borrow_mut().push((reason, detail.to_owned())));
}

#[cfg(test)]
thread_local! {
    pub(super) static RECORDED_WARNINGS: RefCell<Vec<(DegradedReason, String)>> =
        const { RefCell::new(Vec::new()) };
}

#[cfg(test)]
pub(super) fn get_recorded_warnings() -> Vec<(DegradedReason, String)> {
    RECORDED_WARNINGS.with(|w| w.borrow().clone())
}

struct NoOpEmbedder;

impl Embed for NoOpEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Err(EmbedError::Inference("embedder not available".into()))
    }
    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        Err(EmbedError::Inference("embedder not available".into()))
    }
    fn embed_text(&self, _text: &str, _prefix: &str) -> Result<Vec<f32>, EmbedError> {
        Err(EmbedError::Inference("embedder not available".into()))
    }
}

#[cfg(not(feature = "test-support"))]
fn try_load_embedder() -> Result<Arc<dyn Embed>, DegradedReason> {
    let result = try_load_embedder_with(
        || cached_artifacts(ModelId::default()),
        |e| tracing::warn!(error = %e, "failed to delete corrupt model files"),
        |e| tracing::warn!(error = %e, "embedder probe failed"),
    );
    if let Err(reason) = result.as_ref() {
        let detail = match reason {
            DegradedReason::NotInstalled => "model not installed",
            DegradedReason::BackendUnavailable => "MLX backend unavailable",
            DegradedReason::ProbeFailed => "probe failed",
            DegradedReason::Disabled => unreachable!("yomu no longer disables embedding"),
        };
        record_embedder_warning(*reason, detail);
    } else {
        tracing::info!("Embedding model loaded successfully");
    }
    result
}

/// Test-support build: return a deterministic in-memory embedder so `yomu
/// index` populates embeddings without a real model (CI runners have none).
/// `YOMU_TEST_EMBEDDER=unavailable` instead simulates a missing model so the
/// degraded FTS fallback can be exercised at the process boundary. Compiled
/// only under `--features test-support`; the production binary loads the real
/// model via the `cfg(not(...))` variant above and never reads this env var.
#[cfg(feature = "test-support")]
fn try_load_embedder() -> Result<Arc<dyn Embed>, DegradedReason> {
    if env::var("YOMU_TEST_EMBEDDER").as_deref() == Ok("unavailable") {
        let reason = DegradedReason::NotInstalled;
        record_embedder_warning(reason, "test-support: simulated model absence");
        return Err(reason);
    }
    Ok(Arc::new(MockEmbedder::default()))
}

impl Yomu {
    pub(super) fn try_embedder(&self) -> Result<&dyn Embed, DegradedReason> {
        self.embedder
            .get_or_init(try_load_embedder)
            .as_deref()
            .map_err(|r| *r)
    }

    pub(super) fn try_embedder_arc(&self) -> Result<Arc<dyn Embed>, DegradedReason> {
        self.embedder
            .get_or_init(try_load_embedder)
            .as_ref()
            .map(Arc::clone)
            .map_err(|r| *r)
    }

    pub(super) fn get_embedder(&self) -> &dyn Embed {
        static NOOP: NoOpEmbedder = NoOpEmbedder;
        self.try_embedder().unwrap_or(&NOOP)
    }

    pub(super) fn degraded_reason(&self) -> Option<&DegradedReason> {
        self.embedder.get().and_then(|r| r.as_ref().err())
    }
}

#[cfg(test)]
use crate::storage::Db;

#[cfg(test)]
impl Yomu {
    pub(super) fn for_test_raw(
        conn: Db,
        root: PathBuf,
        state: Result<Arc<dyn Embed>, DegradedReason>,
    ) -> Self {
        let embedder_lock = OnceLock::new();
        let _ = embedder_lock.set(state);
        Self {
            conn: Arc::new(Mutex::new(conn)),
            embedder: embedder_lock,
            root,
            rerank_enabled: false,
            reranker: OnceLock::new(),
            log_query: false,
        }
    }
}
