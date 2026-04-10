use std::sync::Arc;

use rurico::embed::{ChunkedEmbedding, Embed, EmbedError, Embedder};

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
    DownloadFailed,
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
            Self::DownloadFailed => {
                Some("embedding model download failed; results from text search only")
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

pub(super) fn parse_auto_download(value: Option<&str>) -> bool {
    match value {
        Some("1" | "true" | "yes" | "on") => true,
        Some("0" | "false" | "no" | "off") | None => false,
        Some(v) => {
            tracing::warn!(
                value = %v,
                "YOMU_AUTO_DOWNLOAD_MODEL only accepts 1/true/yes/on, ignoring"
            );
            false
        }
    }
}

fn try_load_embedder(disabled: bool) -> Result<Arc<dyn Embed>, DegradedReason> {
    if disabled {
        tracing::info!("Embedding disabled via YOMU_EMBED=0");
        return Err(DegradedReason::Disabled);
    }
    let auto_download =
        parse_auto_download(std::env::var("YOMU_AUTO_DOWNLOAD_MODEL").ok().as_deref());
    try_load_embedder_with(
        auto_download,
        || rurico::embed::cached_artifacts(rurico::embed::ModelId::default()),
        || {
            eprintln!("yomu: auto-downloading embedding model...");
            let result = rurico::embed::download_model(rurico::embed::ModelId::default());
            if result.is_ok() {
                eprintln!("yomu: model ready");
            }
            result
        },
    )
}

fn try_load_embedder_with<CE: std::fmt::Display, DE: std::fmt::Display>(
    auto_download: bool,
    cache_check: impl FnOnce() -> Result<Option<rurico::embed::Artifacts>, CE>,
    download_fn: impl FnOnce() -> Result<rurico::embed::Artifacts, DE>,
) -> Result<Arc<dyn Embed>, DegradedReason> {
    use rurico::embed::ProbeStatus;

    fn probe_failed(e: &dyn std::fmt::Display) -> DegradedReason {
        record_embedder_warning(DegradedReason::ProbeFailed, &e.to_string());
        DegradedReason::ProbeFailed
    }

    let paths = match cache_check() {
        Ok(Some(p)) => p,
        Ok(None) if auto_download => match download_fn() {
            Ok(p) => p,
            Err(e) => {
                record_embedder_warning(DegradedReason::DownloadFailed, &e.to_string());
                return Err(DegradedReason::DownloadFailed);
            }
        },
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

#[cfg(test)]
mod tests {
    use super::*;

    // disabled=true → DegradedReason::Disabled
    #[test]
    fn disabled_returns_disabled_reason() {
        let result = try_load_embedder(true);
        assert!(matches!(result, Err(DegradedReason::Disabled)));
    }

    // cache returns None, no auto_download → DegradedReason::NotInstalled
    #[test]
    fn absent_returns_not_installed() {
        let result = try_load_embedder_with(
            false,
            || Ok::<_, &str>(None),
            || -> Result<rurico::embed::Artifacts, &str> {
                unreachable!("download_fn must not be called when auto_download=false")
            },
        );
        assert!(matches!(result, Err(DegradedReason::NotInstalled)));
    }

    // cache_check fails → DegradedReason::ProbeFailed
    #[test]
    fn cache_error_returns_probe_failed() {
        let result = try_load_embedder_with(
            false,
            || Err::<Option<rurico::embed::Artifacts>, _>("cache broken"),
            || -> Result<rurico::embed::Artifacts, &str> {
                unreachable!("download_fn must not be called when cache_check fails")
            },
        );
        assert!(matches!(result, Err(DegradedReason::ProbeFailed)));
    }

    // auto_download=true, model absent, download fails → DegradedReason::DownloadFailed
    #[test]
    fn auto_download_failure_returns_download_failed() {
        let result = try_load_embedder_with(
            true,
            || Ok::<_, &str>(None),
            || Err::<rurico::embed::Artifacts, _>("download failed"),
        );
        assert!(matches!(result, Err(DegradedReason::DownloadFailed)));
    }

    #[test]
    fn parse_auto_download_truthy_values() {
        assert!(parse_auto_download(Some("1")));
        assert!(parse_auto_download(Some("true")));
        assert!(parse_auto_download(Some("yes")));
        assert!(parse_auto_download(Some("on")));
    }

    #[test]
    fn parse_auto_download_falsy_values() {
        assert!(!parse_auto_download(Some("0")));
        assert!(!parse_auto_download(Some("false")));
        assert!(!parse_auto_download(Some("no")));
        assert!(!parse_auto_download(Some("off")));
        assert!(!parse_auto_download(None));
    }

    #[test]
    fn parse_auto_download_unrecognized_returns_false() {
        assert!(!parse_auto_download(Some("TRUE")));
        assert!(!parse_auto_download(Some("2")));
        assert!(!parse_auto_download(Some("enabled")));
    }
}
