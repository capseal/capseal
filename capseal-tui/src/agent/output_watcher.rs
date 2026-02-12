/// Watches PTY output for patterns indicating the agent is waiting for user input
/// or requesting approval. Writes events to .capseal/pty_events.jsonl.
use std::path::PathBuf;
use std::time::Instant;

const BUFFER_CAP: usize = 4096;
const IDLE_THRESHOLD_SECS: f64 = 3.0;

const QUESTION_PREFIXES: &[&str] = &[
    "did you",
    "would you",
    "should i",
    "do you want",
    "is there",
    "can i",
    "shall i",
    "are you sure",
    "do you",
    "would it be",
];

const APPROVAL_PATTERNS: &[&str] = &[
    "accept edits on",
    "do you want to proceed",
    "allow this action",
    "apply these changes",
    "write these files",
    "do you want to make this edit",
    "do you want to make these edits",
    "esc to cancel",
];

pub struct AgentOutputWatcher {
    buffer: Vec<u8>,
    last_output: Instant,
    pty_events_path: PathBuf,
    fired_for_current_idle: bool,
    /// Cooldown: don't fire another approval event within 10 seconds
    last_approval_emit: Option<Instant>,
}

impl AgentOutputWatcher {
    pub fn new(capseal_dir: &std::path::Path) -> Self {
        Self {
            buffer: Vec::with_capacity(BUFFER_CAP),
            last_output: Instant::now(),
            pty_events_path: capseal_dir.join("pty_events.jsonl"),
            fired_for_current_idle: false,
            last_approval_emit: None,
        }
    }

    /// Feed bytes from PTY output. Call this whenever the terminal produces output.
    pub fn feed(&mut self, data: &[u8]) {
        self.last_output = Instant::now();
        self.fired_for_current_idle = false;

        self.buffer.extend_from_slice(data);
        if self.buffer.len() > BUFFER_CAP {
            let drain_to = self.buffer.len() - BUFFER_CAP;
            self.buffer.drain(0..drain_to);
        }

        // Immediate approval pattern detection with 10s cooldown dedup
        let cooldown_ok = match self.last_approval_emit {
            Some(t) => t.elapsed().as_secs_f64() >= 10.0,
            None => true,
        };

        if cooldown_ok {
            let text = String::from_utf8_lossy(&self.buffer).to_string();
            if let Some(matched) = self.looks_like_approval_request(&text) {
                self.last_approval_emit = Some(Instant::now());
                self.fired_for_current_idle = true;
                self.emit_approval_event(&text, &matched);
            }
        }
    }

    /// Check if the agent appears to be waiting. Call this on each tick.
    /// Returns true if an event was emitted.
    pub fn check_idle(&mut self) -> bool {
        if self.buffer.is_empty() {
            return false;
        }

        let elapsed = self.last_output.elapsed().as_secs_f64();

        // General question detection (idle-gated)
        if !self.fired_for_current_idle && elapsed >= IDLE_THRESHOLD_SECS {
            let text = String::from_utf8_lossy(&self.buffer).to_string();
            if self.looks_like_question(&text) {
                self.fired_for_current_idle = true;
                self.emit_event(&text);
                return true;
            }
        }

        false
    }

    fn looks_like_approval_request(&self, text: &str) -> Option<String> {
        let lower = text.to_lowercase();
        for line in lower.lines().rev().take(10) {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            for pattern in APPROVAL_PATTERNS {
                if trimmed.contains(pattern) {
                    return Some(pattern.to_string());
                }
            }
            // Claude Code diff summary followed by ❯❯ (e.g. "42 files +108 -7")
            if (trimmed.contains("files") && (trimmed.contains('+') || trimmed.contains('-')))
                && text.contains('\u{276f}')
            {
                return Some("accept edits on".to_string());
            }
        }
        None
    }

    fn looks_like_question(&self, text: &str) -> bool {
        for line in text.lines().rev().take(5) {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.ends_with('?') {
                return true;
            }

            let lower = trimmed.to_lowercase();
            for prefix in QUESTION_PREFIXES {
                if lower.starts_with(prefix) {
                    return true;
                }
            }

            if trimmed.contains('\u{276f}') || trimmed.contains("\u{276f}") {
                return true;
            }

            if trimmed.ends_with("[Y/n]")
                || trimmed.ends_with("[y/N]")
                || trimmed.ends_with("(yes/no)")
                || trimmed.ends_with("(y/n)")
            {
                return true;
            }

            break;
        }

        false
    }

    fn safe_truncate(text: &str, max_chars: usize) -> String {
        let char_count = text.chars().count();
        if char_count <= max_chars {
            text.to_string()
        } else {
            text.chars().skip(char_count - max_chars).collect()
        }
    }

    fn clean_for_json(text: &str) -> String {
        let prompt_text = Self::safe_truncate(text, 200);
        prompt_text
            .chars()
            .map(|c| if c.is_control() && c != '\n' { ' ' } else { c })
            .collect::<String>()
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    }

    fn write_event(&self, event_json: &str) {
        if let Ok(()) =
            std::fs::create_dir_all(self.pty_events_path.parent().unwrap_or(&PathBuf::from(".")))
        {
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.pty_events_path)
            {
                use std::io::Write;
                let _ = writeln!(f, "{}", event_json);
            }
        }
    }

    fn emit_event(&self, text: &str) {
        let clean = Self::clean_for_json(text);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let event = format!(
            r#"{{"type":"agent_waiting","timestamp":{},"summary":"Agent waiting for input","data":{{"prompt_text":"{}","source":"pty_detect"}}}}"#,
            ts, clean,
        );
        self.write_event(&event);
    }

    fn emit_approval_event(&self, text: &str, matched_pattern: &str) {
        let clean = Self::clean_for_json(text);
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let event = format!(
            r#"{{"type":"agent_approval_request","timestamp":{},"summary":"Agent requesting approval","data":{{"prompt_text":"{}","matched_pattern":"{}","source":"pty_detect"}}}}"#,
            ts, clean, matched_pattern,
        );
        self.write_event(&event);
    }
}
