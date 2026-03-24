use std::sync::atomic::{AtomicU32, Ordering};

use super::{EMBEDDING_DIMS, Embed, EmbedError, EmbedFuture};

/// Mock embedder returning deterministic vectors for offline tests.
pub(crate) struct MockEmbedder;

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

/// Embedder that always fails. For tests.
pub(crate) struct FailingEmbedder {
    message: &'static str,
    docs_fail: bool,
}

impl FailingEmbedder {
    pub fn all_fail(message: &'static str) -> Self {
        Self {
            message,
            docs_fail: true,
        }
    }

    pub fn query_only(message: &'static str) -> Self {
        Self {
            message,
            docs_fail: false,
        }
    }
}

impl Embed for FailingEmbedder {
    fn embed_query<'a>(&'a self, _text: &'a str) -> EmbedFuture<'a, Vec<f32>> {
        let message = self.message;
        Box::pin(async move { Err(EmbedError::Inference(message.into())) })
    }

    fn embed_documents<'a>(&'a self, _texts: &'a [String]) -> EmbedFuture<'a, Vec<Vec<f32>>> {
        if self.docs_fail {
            let message = self.message;
            Box::pin(async move { Err(EmbedError::Inference(message.into())) })
        } else {
            Box::pin(async { Ok(vec![]) })
        }
    }
}

/// Embedder that always returns exactly 1 vector regardless of input count.
pub(crate) struct MismatchEmbedder;

impl Embed for MismatchEmbedder {
    fn embed_query<'a>(&'a self, _text: &'a str) -> EmbedFuture<'a, Vec<f32>> {
        Box::pin(async {
            let mut v = vec![0.0_f32; EMBEDDING_DIMS as usize];
            v[0] = 1.0;
            Ok(v)
        })
    }

    fn embed_documents<'a>(&'a self, _texts: &'a [String]) -> EmbedFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async {
            let mut v = vec![0.0_f32; EMBEDDING_DIMS as usize];
            v[0] = 1.0;
            Ok(vec![v])
        })
    }
}

/// Embedder that alternates: fail on even calls (0, 2, 4…), succeed on odd (1, 3, 5…).
pub(crate) struct AlternatingEmbedder {
    call_count: AtomicU32,
}

impl AlternatingEmbedder {
    pub fn new() -> Self {
        Self {
            call_count: AtomicU32::new(0),
        }
    }
}

impl Embed for AlternatingEmbedder {
    fn embed_query<'a>(&'a self, _text: &'a str) -> EmbedFuture<'a, Vec<f32>> {
        Box::pin(async {
            let mut v = vec![0.0_f32; EMBEDDING_DIMS as usize];
            v[0] = 1.0;
            Ok(v)
        })
    }

    fn embed_documents<'a>(&'a self, texts: &'a [String]) -> EmbedFuture<'a, Vec<Vec<f32>>> {
        let n = self.call_count.fetch_add(1, Ordering::SeqCst);
        if n % 2 == 0 {
            Box::pin(async { Err(EmbedError::Inference("alternating failure".into())) })
        } else {
            Box::pin(async move {
                Ok(texts
                    .iter()
                    .map(|_| {
                        let mut v = vec![0.0_f32; EMBEDDING_DIMS as usize];
                        v[0] = 1.0;
                        v
                    })
                    .collect())
            })
        }
    }
}
