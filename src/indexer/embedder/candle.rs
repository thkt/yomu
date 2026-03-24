use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert;

use super::{
    DOCUMENT_PREFIX, EmbedError, ModelPaths, load_tokenizer, postprocess_embedding, read_config,
    tokenize_with_prefix,
};

pub(super) struct EmbedderInner {
    model: modernbert::ModernBert,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

impl EmbedderInner {
    pub(super) fn new(paths: &ModelPaths) -> Result<Self, EmbedError> {
        paths.validate()?;
        let device = Device::Cpu;

        let config: modernbert::Config = read_config(&paths.config)?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&paths.model),
                DType::F32,
                &device,
            )
            .map_err(|e| EmbedError::Inference(e.to_string()))?
        };

        // ruri-v3-310m safetensors keys have "model." prefix (e.g. "model.embeddings.tok_embeddings.weight")
        // but candle ModernBert expects keys without it (e.g. "embeddings.tok_embeddings.weight").
        // rename_f maps requested name → safetensors lookup key, so prepend "model.".
        let vb = vb.rename_f(|name| format!("model.{name}"));

        let model = modernbert::ModernBert::load(vb, &config)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        let tokenizer = load_tokenizer(&paths.tokenizer)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    pub(super) fn embed_with_prefix(
        &mut self,
        text: &str,
        prefix: &str,
    ) -> Result<Vec<f32>, EmbedError> {
        let tok = tokenize_with_prefix(&self.tokenizer, text, prefix)?;

        let input_ids_tensor = Tensor::new(tok.input_ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| EmbedError::Inference(e.to_string()))?;
        let attention_mask_tensor = Tensor::new(tok.attention_mask.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        let output = self
            .model
            .forward(&input_ids_tensor, &attention_mask_tensor)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        let flat: Vec<f32> = output
            .squeeze(0)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        postprocess_embedding(&flat, tok.seq_len, &tok.attention_mask)
    }

    // candle backend: sequential inference per document (no batched forward pass)
    pub(super) fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts
            .iter()
            .map(|t| self.embed_with_prefix(t, DOCUMENT_PREFIX))
            .collect()
    }
}
