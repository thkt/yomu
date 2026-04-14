use std::sync::Arc;

use rurico::embed::{ChunkedEmbedding, Embed, EmbedError};

pub(super) use amici::model::embedder::DegradedReason;

use super::Yomu;

pub(super) const DEFAULT_EMBED_BUDGET: u32 = 50;

pub(super) fn degraded_reason_user_note(reason: DegradedReason) -> Option<&'static str> {
    match reason {
        DegradedReason::Disabled => None,
        DegradedReason::NotInstalled => {
            Some("embedding model not installed; results from text search only")
        }
        DegradedReason::BackendUnavailable | DegradedReason::ProbeFailed => {
            Some("embedding model unavailable; results from text search only")
        }
    }
}

pub(super) fn record_embedder_warning(reason: DegradedReason, detail: &str) {
    tracing::warn!(reason = ?reason, detail, "Embedder unavailable, using text search only");
    #[cfg(test)]
    RECORDED_WARNINGS.with(|w| w.borrow_mut().push((reason, detail.to_string())));
}

#[cfg(test)]
thread_local! {
    pub(super) static RECORDED_WARNINGS: std::cell::RefCell<Vec<(DegradedReason, String)>> =
        const { std::cell::RefCell::new(Vec::new()) };
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

pub(super) fn parse_embed_budget() -> u32 {
    parse_budget_value(std::env::var("YOMU_EMBED_BUDGET").ok().as_deref())
}

pub(super) fn parse_budget_value(value: Option<&str>) -> u32 {
    match value {
        Some(v) => match v.parse::<u32>() {
            Ok(n) if (super::MIN_EMBED_BUDGET..=super::MAX_EMBED_BUDGET).contains(&n) => n,
            Ok(n) => {
                tracing::warn!(
                    value = n,
                    "YOMU_EMBED_BUDGET out of range ({}..={}), using default",
                    super::MIN_EMBED_BUDGET,
                    super::MAX_EMBED_BUDGET
                );
                DEFAULT_EMBED_BUDGET
            }
            Err(_) => {
                tracing::warn!(value = %v, "Invalid YOMU_EMBED_BUDGET, using default");
                DEFAULT_EMBED_BUDGET
            }
        },
        None => DEFAULT_EMBED_BUDGET,
    }
}

fn try_load_embedder(disabled: bool) -> Result<Arc<dyn Embed>, DegradedReason> {
    if disabled {
        tracing::info!("Embedding disabled via YOMU_EMBED=0");
        return Err(DegradedReason::Disabled);
    }
    let result = amici::model::embedder::try_load_embedder_with(
        || rurico::embed::cached_artifacts(rurico::embed::ModelId::default()),
        |e| tracing::warn!(error = %e, "failed to delete corrupt model files"),
    );
    if let Err(reason) = result.as_ref() {
        let detail = match reason {
            DegradedReason::NotInstalled => "model not installed",
            DegradedReason::BackendUnavailable => "MLX backend unavailable",
            DegradedReason::ProbeFailed => "probe failed",
            DegradedReason::Disabled => unreachable!("disabled handled above"),
        };
        record_embedder_warning(*reason, detail);
    } else {
        tracing::info!("Embedding model loaded successfully");
    }
    result
}

impl Yomu {
    pub(super) fn try_embedder(&self) -> Result<&dyn Embed, DegradedReason> {
        let disabled = self.embed_disabled;
        self.embedder
            .get_or_init(|| try_load_embedder(disabled))
            .as_deref()
            .map_err(|r| *r)
    }

    pub(super) fn get_embedder(&self) -> &dyn Embed {
        static NOOP: NoOpEmbedder = NoOpEmbedder;
        self.try_embedder().unwrap_or(&NOOP)
    }

    pub(super) fn degraded_reason(&self) -> Option<&DegradedReason> {
        self.embedder.get().and_then(|r| r.as_ref().err())
    }

    pub(super) fn embedding_available(&self) -> bool {
        self.embedder.get().is_some_and(|r| r.is_ok())
    }
}

#[cfg(test)]
impl Yomu {
    pub(super) fn for_test_raw(
        conn: crate::storage::Db,
        root: std::path::PathBuf,
        state: Result<Arc<dyn Embed>, DegradedReason>,
    ) -> Self {
        let embedder_lock = std::sync::OnceLock::new();
        let _ = embedder_lock.set(state);
        Self {
            conn: Arc::new(std::sync::Mutex::new(conn)),
            embedder: embedder_lock,
            root,
            embed_budget: DEFAULT_EMBED_BUDGET,
            embed_disabled: false,
            rerank_enabled: false,
            reranker: std::sync::OnceLock::new(),
        }
    }

    /// Exercises the real `try_load_embedder` path with `embed_disabled=true`.
    pub(super) fn for_test_embed_disabled(
        conn: crate::storage::Db,
        root: std::path::PathBuf,
    ) -> Self {
        Self {
            conn: Arc::new(std::sync::Mutex::new(conn)),
            embedder: std::sync::OnceLock::new(),
            root,
            embed_budget: DEFAULT_EMBED_BUDGET,
            embed_disabled: true,
            rerank_enabled: false,
            reranker: std::sync::OnceLock::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_returns_disabled_reason() {
        let result = try_load_embedder(true);
        assert!(matches!(result, Err(DegradedReason::Disabled)));
    }
}
