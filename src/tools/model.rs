use amici::model::download_and_verify_model;

use super::{Yomu, YomuError};

impl Yomu {
    pub fn model_download(json: bool) -> Result<String, YomuError> {
        download_and_verify_model().map_err(|e| YomuError::Internal(e.to_string()))?;
        if json {
            Ok(serde_json::json!({"status": "ok"}).to_string())
        } else {
            Ok("Model downloaded and verified".to_owned())
        }
    }
}
