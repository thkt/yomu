#[cfg(feature = "mlx")]
use std::path::Path;

#[cfg(feature = "mlx")]
use mlx_rs::{
    Array,
    builder::Builder,
    error::Exception,
    fast::scaled_dot_product_attention,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt},
    nn,
    ops::indexing::IndexOp,
};

use super::Config;

#[cfg(feature = "mlx")]
#[derive(Debug, Clone, ModuleParameters)]
#[allow(non_snake_case)]
struct Attention {
    #[param]
    Wqkv: nn::Linear,
    #[param]
    Wo: nn::Linear,
    #[param]
    rope: nn::Rope,
    num_heads: i32,
    head_dim: i32,
    scale: f32,
    uses_local_attention: bool,
}

#[cfg(feature = "mlx")]
impl Attention {
    fn new(
        config: &Config,
        rope_theta: f32,
        uses_local_attention: bool,
    ) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        let head_dim = h / config.num_attention_heads as i32;

        #[allow(non_snake_case)]
        let Wqkv = nn::LinearBuilder::new(h, h * 3).bias(false).build()?;
        #[allow(non_snake_case)]
        let Wo = nn::LinearBuilder::new(h, h).bias(false).build()?;
        let rope = nn::RopeBuilder::new(head_dim)
            .traditional(false)
            .base(rope_theta)
            .build()?;

        Ok(Self {
            Wqkv,
            Wo,
            rope,
            num_heads: config.num_attention_heads as i32,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            uses_local_attention,
        })
    }

    fn forward(
        &mut self,
        xs: &Array,
        global_mask: &Array,
        local_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let shape = xs.shape();
        let (b, seq_len) = (shape[0], shape[1]);

        // Packed QKV: [B, L, 3*H] → [B, L, 3, n_heads, head_dim]
        let qkv = self.Wqkv.forward(xs)?;
        let qkv = qkv.reshape(&[b, seq_len, 3, self.num_heads, self.head_dim])?;
        // → [3, B, n_heads, L, head_dim]
        let qkv = qkv.transpose_axes(&[2, 0, 3, 1, 4])?;

        let q = qkv.index(0);
        let k = qkv.index(1);
        let v = qkv.index(2);

        // RoPE on Q and K
        let rope_input_q = nn::RopeInputBuilder::new(&q).build()?;
        let rope_input_k = nn::RopeInputBuilder::new(&k).build()?;
        let q = self.rope.forward(rope_input_q)?;
        let k = self.rope.forward(rope_input_k)?;

        // Combine global + local mask for local attention layers
        let mask = if self.uses_local_attention {
            if let Some(local) = local_mask {
                global_mask.add(local)?
            } else {
                global_mask.clone()
            }
        } else {
            global_mask.clone()
        };

        // Scaled dot-product attention
        let output = scaled_dot_product_attention(q, &k, &v, self.scale, &mask)?;

        // [B, n_heads, L, head_dim] → [B, L, H]
        let output = output.transpose_axes(&[0, 2, 1, 3])?;
        let output = output.reshape(&[b, seq_len, -1])?;
        self.Wo.forward(&output)
    }
}

#[cfg(feature = "mlx")]
#[derive(Debug, Clone, ModuleParameters)]
#[allow(non_snake_case)]
struct Mlp {
    #[param]
    Wi: nn::Linear,
    #[param]
    Wo: nn::Linear,
}

#[cfg(feature = "mlx")]
impl Mlp {
    fn new(config: &Config) -> Result<Self, Exception> {
        let h = config.hidden_size as i32;
        let inter = config.intermediate_size as i32;
        #[allow(non_snake_case)]
        let Wi = nn::LinearBuilder::new(h, inter * 2).bias(false).build()?;
        #[allow(non_snake_case)]
        let Wo = nn::LinearBuilder::new(inter, h).bias(false).build()?;
        Ok(Self { Wi, Wo })
    }

    fn forward(&mut self, xs: &Array) -> Result<Array, Exception> {
        let xs = self.Wi.forward(xs)?;
        // GeGLU: split in half, gelu(first) * second
        let chunks = xs.split(2, Some(-1))?;
        let gate = nn::gelu(&chunks[0])?;
        let xs = gate.multiply(&chunks[1])?;
        self.Wo.forward(&xs)
    }
}

#[cfg(feature = "mlx")]
#[derive(Debug, Clone, ModuleParameters)]
struct TransformerLayer {
    #[param]
    attn_norm: nn::LayerNorm,
    #[param]
    attn: Attention,
    #[param]
    mlp_norm: nn::LayerNorm,
    #[param]
    mlp: Mlp,
    uses_attn_norm: bool,
}

#[cfg(feature = "mlx")]
impl TransformerLayer {
    fn new(config: &Config, layer_id: usize) -> Result<Self, Exception> {
        let uses_local = !layer_id.is_multiple_of(config.global_attn_every_n_layers);
        let rope_theta = if uses_local {
            config.local_rope_theta as f32
        } else {
            config.global_rope_theta as f32
        };
        let h = config.hidden_size as i32;
        let eps = config.layer_norm_eps as f32;

        let attn_norm = nn::LayerNormBuilder::new(h).eps(eps).build()?;
        let attn = Attention::new(config, rope_theta, uses_local)?;
        let mlp_norm = nn::LayerNormBuilder::new(h).eps(eps).build()?;
        let mlp = Mlp::new(config)?;

        Ok(Self {
            attn_norm,
            attn,
            mlp_norm,
            mlp,
            uses_attn_norm: layer_id != 0,
        })
    }

    fn forward(
        &mut self,
        xs: &Array,
        global_mask: &Array,
        local_mask: Option<&Array>,
    ) -> Result<Array, Exception> {
        let residual = xs.clone();

        let normed = if self.uses_attn_norm {
            self.attn_norm.forward(xs)?
        } else {
            xs.clone()
        };

        let attn_out = self.attn.forward(&normed, global_mask, local_mask)?;
        let xs = attn_out.add(&residual)?;

        let mlp_out = self.mlp.forward(&self.mlp_norm.forward(&xs)?)?;
        xs.add(&mlp_out)
    }
}

/// Embeddings sub-module: matches safetensors key prefix "embeddings."
#[cfg(feature = "mlx")]
#[derive(Debug, Clone, ModuleParameters)]
struct Embeddings {
    #[param]
    tok_embeddings: nn::Embedding,
    #[param]
    norm: nn::LayerNorm,
}

/// ModernBERT backbone (mlx-rs). Ported from cl-nagoya/ruri-v3-310m.
#[cfg(feature = "mlx")]
#[derive(Debug, Clone, ModuleParameters)]
pub(crate) struct ModernBert {
    #[param]
    embeddings: Embeddings,
    #[param]
    layers: Vec<TransformerLayer>,
    #[param]
    final_norm: nn::LayerNorm,
    local_attention_half: usize,
    /// Cached local attention mask keyed by seq_len.
    local_mask_cache: Option<(i32, Array)>,
}

#[cfg(feature = "mlx")]
impl ModernBert {
    /// Create a new model with randomly initialized weights.
    pub fn new(config: &Config) -> Result<Self, Exception> {
        config
            .validate()
            .map_err(|e| Exception::custom(format!("invalid config: {e}")))?;
        let h = config.hidden_size as i32;
        let eps = config.layer_norm_eps as f32;

        let tok_embeddings = nn::Embedding::new(config.vocab_size as i32, h)?;
        let emb_norm = nn::LayerNormBuilder::new(h).eps(eps).build()?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| TransformerLayer::new(config, i))
            .collect::<Result<Vec<_>, _>>()?;

        let final_norm = nn::LayerNormBuilder::new(h).eps(eps).build()?;

        Ok(Self {
            embeddings: Embeddings {
                tok_embeddings,
                norm: emb_norm,
            },
            layers,
            final_norm,
            local_attention_half: config.local_attention / 2,
            local_mask_cache: None,
        })
    }

    /// Load model weights from a SafeTensors file.
    pub fn load(path: impl AsRef<Path>, config: &Config) -> Result<Self, Exception> {
        let mut model = Self::new(config)?;
        model
            .load_safetensors(path)
            .map_err(|e| Exception::custom(format!("SafeTensors load error: {e}")))?;
        Ok(model)
    }

    /// Forward pass: input_ids + attention_mask → hidden states [batch, seq_len, hidden_size].
    pub fn forward(
        &mut self,
        input_ids: &[u32],
        attention_mask: &[u32],
        batch_size: i32,
        seq_len: i32,
    ) -> Result<Array, Exception> {
        debug_assert_eq!(
            input_ids.len(),
            (batch_size * seq_len) as usize,
            "input_ids length must equal batch_size * seq_len"
        );
        debug_assert_eq!(
            attention_mask.len(),
            (batch_size * seq_len) as usize,
            "attention_mask length must equal batch_size * seq_len"
        );

        let ids = Array::from_slice(input_ids, &[batch_size, seq_len]);
        let mask = Array::from_slice(attention_mask, &[batch_size, seq_len]);

        // Token embeddings + norm
        let mut xs = self.embeddings.tok_embeddings.forward(&ids)?;
        xs = self.embeddings.norm.forward(&xs)?;

        // Attention masks
        let global_mask = prepare_4d_attention_mask(&mask, batch_size, seq_len)?;
        let local_mask = match &self.local_mask_cache {
            Some((cached_len, cached)) if *cached_len == seq_len => cached.clone(),
            _ => {
                let m = get_local_attention_mask(seq_len, self.local_attention_half as i32)?;
                self.local_mask_cache = Some((seq_len, m.clone()));
                m
            }
        };

        // Transformer layers
        for layer in &mut self.layers {
            xs = layer.forward(&xs, &global_mask, Some(&local_mask))?;
        }

        self.final_norm.forward(&xs)
    }
}

/// [1, seq_len] mask → [1, 1, seq_len, seq_len] 4D attention mask.
#[cfg(feature = "mlx")]
fn prepare_4d_attention_mask(
    mask: &Array,
    batch_size: i32,
    seq_len: i32,
) -> Result<Array, Exception> {
    let expanded = mask
        .reshape(&[batch_size, 1, 1, seq_len])?
        .as_dtype(mlx_rs::Dtype::Float32)?;
    // Invert: 1→0, 0→-inf
    // f32::MIN (-3.4e38), not NEG_INFINITY: 0.0 * -inf = NaN (IEEE 754)
    let ones = Array::from_f32(1.0);
    let neg_inf = Array::from_f32(f32::MIN);
    let inverted = ones.subtract(&expanded)?;
    inverted.multiply(neg_inf)
}

/// Sliding window mask: [seq_len, seq_len].
#[cfg(feature = "mlx")]
fn get_local_attention_mask(seq_len: i32, half_window: i32) -> Result<Array, Exception> {
    let n = seq_len as usize;
    let hw = half_window as usize;
    let mut data = vec![f32::NEG_INFINITY; n * n];
    for i in 0..n {
        let lo = i.saturating_sub(hw);
        let hi = (i + hw + 1).min(n);
        for j in lo..hi {
            data[i * n + j] = 0.0;
        }
    }
    Ok(Array::from_slice(&data, &[seq_len, seq_len]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modernbert::config::tests::test_config;

    #[cfg(feature = "mlx")]
    mod mlx_tests {
        use serial_test::serial;

        use super::*;

        #[test]
        #[serial]
        fn forward_produces_correct_shape() {
            let config = test_config();
            let mut model = ModernBert::new(&config).expect("create model");

            let input_ids: Vec<u32> = vec![1, 2, 3, 4, 5];
            let mask: Vec<u32> = vec![1, 1, 1, 1, 1];

            let output = model.forward(&input_ids, &mask, 1, 5).expect("forward");
            assert_eq!(output.shape(), &[1, 5, 768]);
        }

        #[test]
        #[serial]
        fn forward_different_seq_lengths() {
            let config = test_config();
            let mut model = ModernBert::new(&config).expect("create model");

            let output = model
                .forward(&[1, 2, 3], &[1, 1, 1], 1, 3)
                .expect("forward");
            assert_eq!(output.shape(), &[1, 3, 768]);
        }

        #[test]
        #[serial]
        fn forward_with_padding_mask() {
            let config = test_config();
            let mut model = ModernBert::new(&config).expect("create model");

            let input_ids: Vec<u32> = vec![1, 2, 3, 0, 0];
            let mask: Vec<u32> = vec![1, 1, 1, 0, 0];

            let output = model.forward(&input_ids, &mask, 1, 5).expect("forward");
            assert_eq!(output.shape(), &[1, 5, 768]);
        }

        #[test]
        #[serial]
        fn global_mask_values() {
            let mask = Array::from_slice(&[1u32, 1, 0], &[1, 3]);
            let result = prepare_4d_attention_mask(&mask, 1, 3).expect("mask");
            result.eval().unwrap();

            assert_eq!(result.shape(), &[1, 1, 1, 3]);
            let data: &[f32] = result.as_slice();
            assert_eq!(data[0], 0.0, "unmasked should be 0.0");
            assert_eq!(data[1], 0.0, "unmasked should be 0.0");
            assert!(
                data[2] < -1e30 && data[2].is_finite(),
                "masked should be large negative finite, got {}",
                data[2]
            );
        }

        #[test]
        #[serial]
        fn global_mask_all_ones() {
            let mask = Array::from_slice(&[1u32, 1, 1, 1], &[1, 4]);
            let result = prepare_4d_attention_mask(&mask, 1, 4).expect("mask");
            result.eval().unwrap();

            let data: &[f32] = result.as_slice();
            for (i, &v) in data.iter().enumerate() {
                assert_eq!(v, 0.0, "all-ones mask should produce 0.0 at index {i}");
            }
        }

        #[test]
        #[serial]
        fn local_mask_window() {
            let result = get_local_attention_mask(5, 1).expect("local mask");
            result.eval().unwrap();

            assert_eq!(result.shape(), &[5, 5]);
            let data: &[f32] = result.as_slice();

            for i in 0..5usize {
                for j in 0..5usize {
                    let val = data[i * 5 + j];
                    let dist = (i as isize - j as isize).unsigned_abs();
                    if dist <= 1 {
                        assert_eq!(val, 0.0, "({i},{j}) within window should be 0.0");
                    } else {
                        assert!(
                            val.is_infinite() && val.is_sign_negative(),
                            "({i},{j}) outside window should be -inf, got {val}"
                        );
                    }
                }
            }
        }

        #[test]
        #[serial]
        fn local_mask_zero_window() {
            let result = get_local_attention_mask(3, 0).expect("local mask");
            result.eval().unwrap();

            let data: &[f32] = result.as_slice();
            for i in 0..3usize {
                for j in 0..3usize {
                    let val = data[i * 3 + j];
                    if i == j {
                        assert_eq!(val, 0.0, "diagonal ({i},{j}) should be 0.0");
                    } else {
                        assert!(val.is_infinite() && val.is_sign_negative());
                    }
                }
            }
        }

        #[test]
        #[serial]
        fn load_nonexistent_path_errors() {
            let config = test_config();
            let result = ModernBert::load("/nonexistent/model.safetensors", &config);
            assert!(result.is_err());
        }
    }
}
