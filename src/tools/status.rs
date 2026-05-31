use crate::storage;

use super::format::{format_coverage, format_status_json};
use super::{Yomu, YomuError, never_degraded};

impl Yomu {
    pub fn status(&self, json: bool) -> Result<String, YomuError> {
        let (stats, ref_count) = self.with_db(|conn| {
            let stats = storage::get_stats(conn)?;
            let ref_count = storage::get_reference_count(conn)?;
            Ok((stats, ref_count))
        })?;

        if json {
            let (degraded, notes) = never_degraded();
            return Ok(format_status_json(&stats, ref_count, degraded, notes));
        }

        Ok(format!(
            "Index status:\n  Files: {}\n  Chunks: {}\n  Embedded: {}\n  References: {}\n  Last indexed: {}",
            stats.total_files,
            stats.total_chunks,
            format_coverage(&stats),
            ref_count,
            stats.last_indexed_at.as_deref().unwrap_or("never")
        ))
    }
}
