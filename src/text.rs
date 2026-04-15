use std::mem;

pub fn split_identifier(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = s.chars().collect();

    for i in 0..chars.len() {
        let c = chars[i];
        if c == '-' || c == '_' {
            if !current.is_empty() {
                parts.push(mem::take(&mut current));
            }
        } else if c.is_uppercase() {
            let prev_lower = i > 0 && chars[i - 1].is_lowercase();
            let prev_upper = i > 0 && chars[i - 1].is_uppercase();
            let next_lower = i + 1 < chars.len() && chars[i + 1].is_lowercase();
            if (prev_lower || (prev_upper && next_lower)) && !current.is_empty() {
                parts.push(mem::take(&mut current));
            }
            current.push(c);
        } else {
            current.push(c);
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-347: camel_case
    #[test]
    fn camel_case() {
        assert_eq!(split_identifier("useChat"), vec!["use", "Chat"]);
        assert_eq!(split_identifier("DataTable"), vec!["Data", "Table"]);
    }

    // T-348: acronym
    #[test]
    fn acronym() {
        assert_eq!(split_identifier("HTMLElement"), vec!["HTML", "Element"]);
        assert_eq!(
            split_identifier("getXMLParser"),
            vec!["get", "XML", "Parser"]
        );
        assert_eq!(split_identifier("UIMessage"), vec!["UI", "Message"]);
    }

    // T-349: kebab_and_snake
    #[test]
    fn kebab_and_snake() {
        assert_eq!(split_identifier("data-table"), vec!["data", "table"]);
        assert_eq!(split_identifier("get_value"), vec!["get", "value"]);
        assert_eq!(split_identifier("use-chat"), vec!["use", "chat"]);
    }

    // T-350: single_word
    #[test]
    fn single_word() {
        assert_eq!(split_identifier("stream"), vec!["stream"]);
        assert_eq!(split_identifier("chat"), vec!["chat"]);
    }

    // T-351: empty_string
    #[test]
    fn empty_string() {
        let result: Vec<String> = split_identifier("");
        assert!(result.is_empty());
    }

    // T-352: delimiter_only
    #[test]
    fn delimiter_only() {
        assert!(split_identifier("_").is_empty());
        assert!(split_identifier("---").is_empty());
        assert!(split_identifier("__").is_empty());
    }

    // T-353: single_char
    #[test]
    fn single_char() {
        assert_eq!(split_identifier("a"), vec!["a"]);
        assert_eq!(split_identifier("A"), vec!["A"]);
    }

    // T-354: trailing_leading_delimiters
    #[test]
    fn trailing_leading_delimiters() {
        assert_eq!(split_identifier("_foo_"), vec!["foo"]);
        assert_eq!(split_identifier("-bar-"), vec!["bar"]);
    }

    // T-355: all_uppercase
    #[test]
    fn all_uppercase() {
        assert_eq!(split_identifier("HTTP"), vec!["HTTP"]);
        assert_eq!(split_identifier("URL"), vec!["URL"]);
    }

    // T-356: mixed_delimiters
    #[test]
    fn mixed_delimiters() {
        assert_eq!(split_identifier("foo-bar_baz"), vec!["foo", "bar", "baz"]);
    }
}
