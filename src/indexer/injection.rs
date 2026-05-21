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
    /// FR-403e / FR-412: consumed by verify pipeline, not the production
    /// matcher. `Option<String>` keeps PR#1 fixtures (corpus.minimal.yaml /
    /// corpus.empty.yaml / corpus.multi-match.yaml / corpus.invalid-regex.yaml)
    /// backward-compatible. The non-empty invariant is enforced by `verify.rs`
    /// when deriving `PositiveCase`, not by `Corpus::load_from_str`.
    #[serde(default)]
    pub test_content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct CorpusFile {
    pub(crate) entries: Vec<CorpusEntry>,
}

/// Negative-corpus fixture (`tests/fixtures/injection/corpus.negative.yaml`).
/// Consumed by `verify.rs` for precision measurement. Each entry pairs with a
/// positive entry via `corresponds_to:` and SHALL produce zero matcher flags.
#[derive(Debug, Deserialize)]
pub struct NegativeEntry {
    pub id: String,
    pub corresponds_to: String,
    pub content: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct NegativeFile {
    pub entries: Vec<NegativeEntry>,
}

impl NegativeFile {
    pub fn load_from_str(text: &str) -> Result<Self, CorpusError> {
        serde_yaml::from_str(text).map_err(CorpusError::from)
    }
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
        let (corpus, _) = Self::load_with_entries(text)?;
        Ok(corpus)
    }

    /// Parses `text` once and returns both the compiled `Corpus` and the raw
    /// `Vec<CorpusEntry>`. Callers that need the entry-level data (e.g.
    /// `verify::positives_from_entries` for `test_content`) avoid a second
    /// `serde_yaml::from_str` pass.
    pub fn load_with_entries(text: &str) -> Result<(Self, Vec<CorpusEntry>), CorpusError> {
        let raw: CorpusFile = serde_yaml::from_str(text)?;
        let entries = raw
            .entries
            .iter()
            .map(compile_entry)
            .collect::<Result<Vec<_>, CorpusError>>()?;
        Ok((Self { entries }, raw.entries))
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

fn compile_entry(entry: &CorpusEntry) -> Result<CompiledEntry, CorpusError> {
    let compiled = match entry.pattern_type {
        PatternType::Literal => CompiledPattern::Literal(entry.pattern.clone()),
        PatternType::Regex => {
            let re = Regex::new(&entry.pattern).map_err(|source| CorpusError::InvalidRegex {
                id: entry.id.clone(),
                source,
            })?;
            CompiledPattern::Regex(re)
        }
    };
    Ok(CompiledEntry {
        pattern: compiled,
        expected_flags: entry.expected_flags.clone(),
    })
}

#[cfg(test)]
mod tests;
