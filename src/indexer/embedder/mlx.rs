use super::{
    DOCUMENT_PREFIX, EmbedError, ModelPaths, load_tokenizer, postprocess_embedding, read_config,
    tokenize_with_prefix,
};

pub(super) struct EmbedderInner {
    model: crate::modernbert::ModernBert,
    tokenizer: tokenizers::Tokenizer,
}

impl EmbedderInner {
    pub(super) fn new(paths: &ModelPaths) -> Result<Self, EmbedError> {
        paths.validate()?;

        let config: crate::modernbert::Config = read_config(&paths.config)?;

        let model = crate::modernbert::ModernBert::load(&paths.model, &config)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        let tokenizer = load_tokenizer(&paths.tokenizer)?;

        Ok(Self { model, tokenizer })
    }

    pub(super) fn embed_with_prefix(
        &mut self,
        text: &str,
        prefix: &str,
    ) -> Result<Vec<f32>, EmbedError> {
        let tok = tokenize_with_prefix(&self.tokenizer, text, prefix)?;

        let output = self
            .model
            .forward(&tok.input_ids, &tok.attention_mask, 1, tok.seq_len as i32)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        output
            .eval()
            .map_err(|e| EmbedError::Inference(e.to_string()))?;
        let flat: &[f32] = output.as_slice();

        postprocess_embedding(flat, tok.seq_len, &tok.attention_mask)
    }

    pub(super) fn embed_batch(&mut self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{DOCUMENT_PREFIX}{t}"))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(prefixed, true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let batch_size = encodings.len();
        let max_seq_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap();

        let mut input_ids = vec![0u32; batch_size * max_seq_len];
        let mut attention_mask = vec![0u32; batch_size * max_seq_len];
        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let offset = i * max_seq_len;
            input_ids[offset..offset + ids.len()].copy_from_slice(ids);
            attention_mask[offset..offset + mask.len()].copy_from_slice(mask);
        }

        let output = self
            .model
            .forward(
                &input_ids,
                &attention_mask,
                batch_size as i32,
                max_seq_len as i32,
            )
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        output
            .eval()
            .map_err(|e| EmbedError::Inference(e.to_string()))?;
        let flat: &[f32] = output.as_slice();

        let hidden_size = flat.len() / (batch_size * max_seq_len);
        let stride = max_seq_len * hidden_size;
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let seq_data = &flat[i * stride..(i + 1) * stride];
            let mask_slice = &attention_mask[i * max_seq_len..(i + 1) * max_seq_len];
            results.push(postprocess_embedding(seq_data, max_seq_len, mask_slice)?);
        }

        Ok(results)
    }
}
