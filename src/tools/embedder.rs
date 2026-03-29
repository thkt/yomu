use std::sync::Arc;

use rurico::embed::{Embed, EmbedError, Embedder};

use super::Yomu;

pub(super) const DEFAULT_EMBED_BUDGET: u32 = 50;
const MIN_EMBED_BUDGET: u32 = 1;
const MAX_EMBED_BUDGET: u32 = 500;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DegradedReason {
    Disabled,
    NotInstalled,
    BackendUnavailable,
    ProbeFailed,
}

impl DegradedReason {
    pub(super) fn user_note(&self) -> Option<&'static str> {
        match self {
            Self::Disabled => None,
            Self::NotInstalled => {
                Some("embedding model not installed; results from text search only")
            }
            Self::BackendUnavailable | Self::ProbeFailed => {
                Some("embedding model unavailable; results from text search only")
            }
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
    fn embed_document(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Err(EmbedError::Inference("embedder not available".into()))
    }
}

pub(super) fn parse_embed_budget() -> u32 {
    parse_budget_value(std::env::var("YOMU_EMBED_BUDGET").ok().as_deref())
}

pub(super) fn parse_budget_value(value: Option<&str>) -> u32 {
    match value {
        Some(v) => match v.parse::<u32>() {
            Ok(n) if (MIN_EMBED_BUDGET..=MAX_EMBED_BUDGET).contains(&n) => n,
            Ok(n) => {
                tracing::warn!(
                    value = n,
                    "YOMU_EMBED_BUDGET out of range ({MIN_EMBED_BUDGET}..={MAX_EMBED_BUDGET}), using default"
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
    use rurico::embed::{ProbeStatus, model_paths_if_cached};

    fn probe_failed(e: &dyn std::fmt::Display) -> DegradedReason {
        record_embedder_warning(DegradedReason::ProbeFailed, &e.to_string());
        DegradedReason::ProbeFailed
    }

    if disabled {
        tracing::info!("Embedding disabled via YOMU_EMBED=0");
        return Err(DegradedReason::Disabled);
    }
    let paths = match model_paths_if_cached() {
        Ok(Some(p)) => p,
        Ok(None) => return Err(DegradedReason::NotInstalled),
        Err(e) => return Err(probe_failed(&e)),
    };
    match Embedder::probe(&paths) {
        Ok(ProbeStatus::Available) => {}
        Ok(ProbeStatus::BackendUnavailable) => {
            record_embedder_warning(
                DegradedReason::BackendUnavailable,
                "MLX backend unavailable",
            );
            return Err(DegradedReason::BackendUnavailable);
        }
        Err(e) => return Err(probe_failed(&e)),
    }
    let embedder = Embedder::new(&paths).map_err(|e| probe_failed(&e))?;
    tracing::info!("Embedding model loaded successfully");
    Ok(Arc::new(embedder) as Arc<dyn Embed>)
}

impl Yomu {
    pub(super) fn get_embedder(&self) -> &dyn Embed {
        static NOOP: NoOpEmbedder = NoOpEmbedder;
        let disabled = self.embed_disabled;
        self.embedder
            .get_or_init(|| try_load_embedder(disabled))
            .as_deref()
            .ok()
            .unwrap_or(&NOOP)
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
        }
    }

    /// Create a Yomu with `embed_disabled` and an empty OnceLock so that
    /// `get_embedder()` exercises the real `try_load_embedder` path.
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
        }
    }
}
