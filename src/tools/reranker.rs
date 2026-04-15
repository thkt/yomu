use rurico::reranker::{Rerank, RerankerModelId, cached_artifacts};

use amici::model::reranker::try_load_reranker_with;
use amici::model::{DegradedReason, ModelLoad};

use super::Yomu;

pub(crate) fn try_load_reranker() -> ModelLoad<Box<dyn Rerank>> {
    match try_load_reranker_with(
        || cached_artifacts(RerankerModelId::default()),
        |e| tracing::warn!(error = %e, "failed to delete corrupt reranker model files"),
        |e| tracing::warn!(error = %e, "reranker failed to load"),
    ) {
        Ok(r) => ModelLoad::Ready(r),
        Err(DegradedReason::NotInstalled) => ModelLoad::Absent,
        Err(reason) => ModelLoad::Failed(reason.to_string()),
    }
}

impl Yomu {
    pub(super) fn get_reranker(&self) -> Option<&dyn Rerank> {
        if !self.rerank_enabled {
            return None;
        }
        self.reranker
            .get_or_init(try_load_reranker)
            .as_ref()
            .map(|b| b.as_ref() as &dyn Rerank)
    }

    pub(super) fn reranker_note(&self) -> Option<String> {
        if !self.rerank_enabled {
            return None;
        }
        match self.reranker.get() {
            Some(ModelLoad::Absent) => Some(
                "reranking requested (YOMU_RERANK=1) but model not cached; install with `sae download reranker`".into(),
            ),
            Some(ModelLoad::Failed(msg)) => Some(format!("reranker failed to load: {msg}")),
            _ => None,
        }
    }
}
