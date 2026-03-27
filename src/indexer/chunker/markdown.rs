use crate::storage::ChunkType;

use super::{RawChunk, chunk_fallback};

pub(super) fn chunk_markdown(source: &str) -> Vec<RawChunk> {
    let lines: Vec<&str> = source.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let headings = find_md_headings(&lines);
    if headings.is_empty() {
        return chunk_fallback(source);
    }

    let mut chunks = Vec::new();

    if headings[0].0 > 0 {
        let content: String = lines[..headings[0].0].join("\n");
        if !content.trim().is_empty() {
            chunks.push(RawChunk {
                chunk_type: ChunkType::Other,
                parent_index: None,
                name: None,
                content,
                start_line: 1,
                end_line: headings[0].0 as u32,
                ast_start_line: 1,
            });
        }
    }

    for (idx, (line_idx, title)) in headings.iter().enumerate() {
        let end_idx = if idx + 1 < headings.len() {
            headings[idx + 1].0
        } else {
            lines.len()
        };
        let content: String = lines[*line_idx..end_idx].join("\n");
        if !content.trim().is_empty() {
            chunks.push(RawChunk {
                chunk_type: ChunkType::MdSection,
                parent_index: None,
                name: if title.is_empty() {
                    None
                } else {
                    Some(title.clone())
                },
                content,
                start_line: (*line_idx + 1) as u32,
                end_line: end_idx as u32,
                ast_start_line: (*line_idx + 1) as u32,
            });
        }
    }

    if chunks.is_empty() {
        return chunk_fallback(source);
    }

    chunks
}

fn find_md_headings(lines: &[&str]) -> Vec<(usize, String)> {
    let mut headings = Vec::new();
    let mut in_fence = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if let Some(title) = parse_md_heading(line) {
            headings.push((i, title));
        }
    }

    if in_fence {
        tracing::warn!("Unclosed code fence detected, headings after fence may be suppressed");
    }

    headings
}

fn parse_md_heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    if line.len() - trimmed.len() > 3 {
        return None;
    }
    let level = trimmed.bytes().take_while(|&b| b == b'#').count();
    if level == 0 || level > 6 {
        return None;
    }
    let rest = &trimmed[level..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim().trim_end_matches('#').trim();
    Some(title.to_string())
}
