use std::io;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq)]
pub enum ChunkType {
    Component,
    Hook,
    TypeDef,
    CssRule,
    HtmlElement,
    TestCase,
    RustFn,
    RustStruct,
    RustEnum,
    RustTrait,
    RustImpl,
    MdSection,
    Other,
    InnerFn,
}

impl ChunkType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Component => "component",
            Self::Hook => "hook",
            Self::TypeDef => "type_def",
            Self::CssRule => "css_rule",
            Self::HtmlElement => "html_element",
            Self::TestCase => "test_case",
            Self::RustFn => "rust_fn",
            Self::RustStruct => "rust_struct",
            Self::RustEnum => "rust_enum",
            Self::RustTrait => "rust_trait",
            Self::RustImpl => "rust_impl",
            Self::MdSection => "md_section",
            Self::Other => "other",
            Self::InnerFn => "inner_fn",
        }
    }

    pub fn from_db(s: &str) -> Self {
        match s {
            "component" => Self::Component,
            "hook" => Self::Hook,
            "type_def" => Self::TypeDef,
            "css_rule" => Self::CssRule,
            "html_element" => Self::HtmlElement,
            "test_case" => Self::TestCase,
            "rust_fn" => Self::RustFn,
            "rust_struct" => Self::RustStruct,
            "rust_enum" => Self::RustEnum,
            "rust_trait" => Self::RustTrait,
            "rust_impl" => Self::RustImpl,
            "md_section" => Self::MdSection,
            "other" => Self::Other,
            "inner_fn" => Self::InnerFn,
            other => {
                tracing::warn!(
                    chunk_type = other,
                    "Unknown chunk_type in DB, defaulting to Other"
                );
                Self::Other
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub file_path: String,
    pub chunk_type: ChunkType,
    pub name: Option<String>,
    pub content: String,
    pub start_line: u32,
    pub end_line: u32,
    pub parent_chunk_id: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IndexStatus {
    pub total_files: u32,
    pub total_chunks: u32,
    pub embeddable_chunks: u32,
    pub embedded_chunks: u32,
    pub last_indexed_at: Option<String>,
}

impl IndexStatus {
    pub fn embed_coverage(&self) -> f32 {
        if self.embeddable_chunks > 0 {
            self.embedded_chunks as f32 / self.embeddable_chunks as f32
        } else {
            0.0
        }
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn embed_percentage(&self) -> u32 {
        (self.embed_coverage() as f64 * 100.0) as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatchSource {
    Semantic,
    Fts,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: Chunk,
    pub chunk_id: Option<i64>,
    pub distance: f32,
    pub match_source: MatchSource,
    /// Reranked score (higher = better).
    pub score: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Database error: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    #[error("chunks/embeddings length mismatch: {chunks} chunks vs {embeddings} embeddings")]
    LengthMismatch { chunks: usize, embeddings: usize },
    #[error("embedding dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("schema mismatch: missing columns {missing:?} in table '{table}' in {} — delete this file and re-run to recreate the index", path.display())]
    SchemaMismatch {
        table: &'static str,
        missing: Vec<String>,
        path: PathBuf,
    },
}

pub struct NewChunk<'a> {
    pub chunk_type: &'a ChunkType,
    pub name: Option<&'a str>,
    pub content: &'a str,
    pub start_line: u32,
    pub end_line: u32,
    pub parent_index: Option<usize>,
}

pub struct FileData<'a> {
    pub file_path: &'a str,
    pub chunks: &'a [NewChunk<'a>],
    pub file_hash: &'a str,
    pub imports_text: &'a str,
    pub refs: &'a [Reference],
    pub mtime_epoch: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefKind {
    Named,
    Default,
    Namespace,
    TypeOnly,
    SideEffect,
    ModDecl,
}

impl RefKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Named => "named",
            Self::Default => "default",
            Self::Namespace => "namespace",
            Self::TypeOnly => "type_only",
            Self::SideEffect => "side_effect",
            Self::ModDecl => "mod_decl",
        }
    }

    pub fn from_db(s: &str) -> Self {
        match s {
            "named" => Self::Named,
            "default" => Self::Default,
            "namespace" => Self::Namespace,
            "type_only" => Self::TypeOnly,
            "side_effect" => Self::SideEffect,
            "mod_decl" => Self::ModDecl,
            other => {
                tracing::warn!(
                    ref_kind = other,
                    "Unknown ref_kind in DB, defaulting to Named"
                );
                Self::Named
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reference {
    pub source_file: String,
    pub target_file: String,
    pub symbol_name: Option<String>,
    pub ref_kind: RefKind,
}
