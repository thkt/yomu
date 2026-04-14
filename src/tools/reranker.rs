use rurico::reranker::Rerank;

use amici::model::ModelLoad;
use amici::model::reranker::try_load_reranker_with;

use super::Yomu;

pub(crate) fn try_load_reranker() -> ModelLoad<Box<dyn Rerank>> {
    try_load_reranker_with(|| {
        rurico::reranker::cached_artifacts(rurico::reranker::RerankerModelId::default())
    })
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
