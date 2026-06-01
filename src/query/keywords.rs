use std::collections::HashSet;

use crate::storage::ChunkType;
use crate::text::split_identifier;

const STOP_WORDS: &[&str] = &["the", "a", "an", "in", "for", "of", "with", "and", "or"];

// Words ending in -ing that are not gerunds (stripping -ing produces a non-word).
const ING_DENY: &[&str] = &[
    "string",
    "bring",
    "thing",
    "nothing",
    "something",
    "everything",
    "ring",
    "king",
    "spring",
    "swing",
    "sing",
    "sting",
    "wing",
];

// Words ending in -s that are not plurals (stripping -s produces a non-word).
const S_DENY: &[&str] = &[
    "class", "this", "alias", "canvas", "focus", "status", "bus", "process", "address", "access",
    "express", "progress",
];

fn stem_keyword(kw: &str) -> Option<&str> {
    let stem = if kw.len() > 5 && kw.ends_with("ing") && !ING_DENY.contains(&kw) {
        Some(&kw[..kw.len() - 3])
    } else if kw.len() > 3 && kw.ends_with('s') && !kw.ends_with("ss") && !S_DENY.contains(&kw) {
        Some(&kw[..kw.len() - 1])
    } else {
        None
    };
    stem.filter(|s| s.len() >= 2)
}

pub fn extract_keywords(query: &str) -> Vec<String> {
    let mut base: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for token in query.split_whitespace() {
        let lower = token.to_lowercase();
        if lower.chars().count() >= 2
            && !STOP_WORDS.contains(&lower.as_str())
            && seen.insert(lower.clone())
        {
            base.push(lower);
        }
        for part in split_identifier(token) {
            let part_lower = part.to_lowercase();
            if part_lower.chars().count() >= 2
                && !STOP_WORDS.contains(&part_lower.as_str())
                && seen.insert(part_lower.clone())
            {
                base.push(part_lower);
            }
        }
    }

    let stems: Vec<String> = base
        .iter()
        .filter_map(|kw| stem_keyword(kw))
        .filter(|s| !seen.contains(*s))
        .map(str::to_owned)
        .collect();
    let mut all = base;
    for s in stems {
        seen.insert(s.clone());
        all.push(s);
    }
    all
}

pub fn extract_type_hints(query: &str) -> Vec<ChunkType> {
    let mut hints = Vec::new();
    for token in query.split_whitespace() {
        let token = token.to_lowercase();
        let hint = match token.as_str() {
            "hook" | "hooks" => Some(ChunkType::Hook),
            "component" | "components" => Some(ChunkType::Component),
            "type" | "types" | "interface" => Some(ChunkType::TypeDef),
            "css" | "style" | "styles" => Some(ChunkType::CssRule),
            "test" | "tests" | "spec" => Some(ChunkType::TestCase),
            _ => None,
        };
        if let Some(h) = hint
            && !hints.contains(&h)
        {
            hints.push(h);
        }
    }
    hints
}
