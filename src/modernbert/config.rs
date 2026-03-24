use serde::Deserialize;

/// ModernBERT config (config.json).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    #[allow(dead_code)]
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    #[allow(dead_code)]
    pub pad_token_id: u32,
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    pub local_attention: usize,
    pub local_rope_theta: f64,
}

impl Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.num_attention_heads == 0 {
            return Err("num_attention_heads must be > 0".into());
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            ));
        }
        if self.global_attn_every_n_layers == 0 {
            return Err("global_attn_every_n_layers must be > 0".into());
        }
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".into());
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn test_config() -> Config {
        Config {
            vocab_size: 1000,
            hidden_size: 768,
            num_hidden_layers: 2,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,
            global_attn_every_n_layers: 3,
            global_rope_theta: 160000.0,
            local_attention: 128,
            local_rope_theta: 10000.0,
        }
    }

    #[test]
    fn config_validate_valid() {
        assert!(test_config().validate().is_ok());
    }

    #[test]
    fn config_validate_zero_attention_heads() {
        let mut c = test_config();
        c.num_attention_heads = 0;
        assert!(c.validate().unwrap_err().contains("num_attention_heads"));
    }

    #[test]
    fn config_validate_indivisible_hidden_size() {
        let mut c = test_config();
        c.hidden_size = 100;
        c.num_attention_heads = 3;
        assert!(c.validate().unwrap_err().contains("divisible"));
    }

    #[test]
    fn config_validate_zero_global_attn() {
        let mut c = test_config();
        c.global_attn_every_n_layers = 0;
        assert!(
            c.validate()
                .unwrap_err()
                .contains("global_attn_every_n_layers")
        );
    }

    #[test]
    fn config_validate_zero_vocab_size() {
        let mut c = test_config();
        c.vocab_size = 0;
        assert!(c.validate().unwrap_err().contains("vocab_size"));
    }

    #[test]
    fn config_deserialize_from_json() {
        let json = r#"{
            "vocab_size": 50368,
            "hidden_size": 768,
            "num_hidden_layers": 22,
            "num_attention_heads": 12,
            "intermediate_size": 1152,
            "max_position_embeddings": 8192,
            "layer_norm_eps": 1e-5,
            "pad_token_id": 50283,
            "global_attn_every_n_layers": 3,
            "global_rope_theta": 160000.0,
            "local_attention": 128,
            "local_rope_theta": 10000.0
        }"#;
        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.vocab_size, 50368);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 22);
        assert_eq!(config.num_attention_heads, 12);
        assert_eq!(config.global_attn_every_n_layers, 3);
    }

    #[test]
    fn config_deserialize_rejects_missing_field() {
        let json = r#"{ "vocab_size": 1000 }"#;
        let result: Result<Config, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }
}
