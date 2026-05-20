#[cfg(test)]
use std::fs;
use std::io;
#[cfg(test)]
use std::path::Path;

use regex::Regex;
use serde::Deserialize;

#[derive(Debug, thiserror::Error)]
pub enum CorpusError {
    #[error("IO error reading corpus: {0}")]
    Io(#[from] io::Error),
    #[error("YAML parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("Invalid regex in entry {id}: {source}")]
    InvalidRegex {
        id: String,
        #[source]
        source: regex::Error,
    },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PatternType {
    Literal,
    Regex,
}

// `severity`, `category`, `description`, `source` are deserialized for schema
// documentation but not read at runtime; per FR-206 / reviewer-spec they MAY
// remain.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct CorpusEntry {
    pub id: String,
    pub pattern_type: PatternType,
    pub pattern: String,
    pub severity: String,
    pub category: String,
    pub expected_flags: Vec<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CorpusFile {
    entries: Vec<CorpusEntry>,
}

#[derive(Debug)]
enum CompiledPattern {
    Literal(String),
    Regex(Regex),
}

#[derive(Debug)]
struct CompiledEntry {
    pattern: CompiledPattern,
    expected_flags: Vec<String>,
}

#[derive(Debug)]
pub struct Corpus {
    entries: Vec<CompiledEntry>,
}

impl Corpus {
    #[cfg(test)]
    pub fn load_from_yaml(path: &Path) -> Result<Self, CorpusError> {
        let text = fs::read_to_string(path)?;
        Self::load_from_str(&text)
    }

    pub fn load_from_str(text: &str) -> Result<Self, CorpusError> {
        let raw: CorpusFile = serde_yaml::from_str(text)?;
        let entries = raw
            .entries
            .into_iter()
            .map(compile_entry)
            .collect::<Result<Vec<_>, CorpusError>>()?;
        Ok(Self { entries })
    }

    pub fn check_chunk(&self, content: &str) -> Vec<String> {
        let mut flags = Vec::new();
        for entry in &self.entries {
            let matched = match &entry.pattern {
                CompiledPattern::Literal(s) => content.contains(s.as_str()),
                CompiledPattern::Regex(re) => re.is_match(content),
            };
            if matched {
                flags.extend(entry.expected_flags.iter().cloned());
            }
        }
        flags
    }
}

fn compile_entry(entry: CorpusEntry) -> Result<CompiledEntry, CorpusError> {
    let CorpusEntry {
        id,
        pattern_type,
        pattern,
        expected_flags,
        ..
    } = entry;
    let compiled = match pattern_type {
        PatternType::Literal => CompiledPattern::Literal(pattern),
        PatternType::Regex => {
            let re = Regex::new(&pattern).map_err(|source| CorpusError::InvalidRegex {
                id: id.clone(),
                source,
            })?;
            CompiledPattern::Regex(re)
        }
    };
    Ok(CompiledEntry {
        pattern: compiled,
        expected_flags,
    })
}

#[cfg(test)]
mod tests;
