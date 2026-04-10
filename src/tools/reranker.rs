use rurico::reranker::{Rerank, Reranker};

/// Model load state shared across yomu and sae (amici extraction target).
///
/// Identical definition exists in sae — intentional DRY violation
/// until amici crate extraction.
pub(crate) enum ModelLoad<T> {
    Ready(T),
    Absent,
    Failed(String),
}

impl<T> std::fmt::Debug for ModelLoad<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ready(_) => write!(f, "ModelLoad::Ready(..)"),
            Self::Absent => write!(f, "ModelLoad::Absent"),
            Self::Failed(msg) => write!(f, "ModelLoad::Failed({msg:?})"),
        }
    }
}

pub(crate) fn try_load_reranker() -> ModelLoad<Box<dyn Rerank>> {
    try_load_reranker_with(|| {
        rurico::reranker::cached_artifacts(rurico::reranker::RerankerModelId::default())
    })
}

fn try_load_reranker_with<E: std::fmt::Display>(
    cache_check: impl FnOnce() -> Result<Option<rurico::reranker::Artifacts>, E>,
) -> ModelLoad<Box<dyn Rerank>> {
    use rurico::reranker::ProbeStatus;

    let artifacts = match cache_check() {
        Ok(Some(a)) => a,
        Ok(None) => {
            tracing::debug!("reranker model not cached");
            return ModelLoad::Absent;
        }
        Err(e) => {
            tracing::debug!(error = %e, "reranker model cache check failed");
            return ModelLoad::Failed(e.to_string());
        }
    };
    match Reranker::probe(&artifacts) {
        Ok(ProbeStatus::Available) => {}
        Ok(ProbeStatus::BackendUnavailable) => {
            tracing::debug!("MLX backend unavailable for reranker");
            return ModelLoad::Failed("MLX backend is unavailable".to_string());
        }
        Err(e) => {
            tracing::debug!(error = %e, "reranker model probe failed");
            return ModelLoad::Failed(e.to_string());
        }
    }
    match Reranker::new(&artifacts) {
        Ok(r) => {
            tracing::debug!("reranker model loaded");
            ModelLoad::Ready(Box::new(r))
        }
        Err(e) => {
            tracing::debug!(error = %e, "reranker model load failed");
            ModelLoad::Failed(e.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-004: cached_artifacts が Ok(None) を返す → ModelLoad::Absent
    #[test]
    fn t004_cache_absent_returns_model_load_absent() {
        let result = try_load_reranker_with(|| Ok::<_, &str>(None));
        assert!(matches!(result, ModelLoad::Absent));
    }

    // T-005: cached_artifacts がエラーを返す → ModelLoad::Failed(msg)
    #[test]
    fn t005_cache_error_returns_model_load_failed() {
        let result = try_load_reranker_with(|| {
            Err::<Option<rurico::reranker::Artifacts>, _>("cache broken")
        });
        match result {
            ModelLoad::Failed(msg) => assert!(msg.contains("cache broken")),
            other => panic!("expected ModelLoad::Failed, got {other:?}"),
        }
    }
}
