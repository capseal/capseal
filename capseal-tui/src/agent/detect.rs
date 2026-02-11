/// Detects when the user types an agent command in the PTY input stream.
pub struct AgentDetector {
    buffer: Vec<u8>,
    max_len: usize,
}

const AGENTS: &[(&str, &str)] = &[
    ("claude", "claude_code"),
    ("codex", "codex"),
    ("gemini", "gemini"),
    ("cursor", "cursor"),
    ("aider", "aider"),
];

impl AgentDetector {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(256),
            max_len: 256,
        }
    }

    /// Feed a byte from user input. Returns the agent name if Enter was pressed
    /// and the buffer matches an agent command.
    pub fn feed(&mut self, byte: u8) -> Option<&'static str> {
        if byte == b'\r' || byte == b'\n' {
            // User pressed Enter — check buffer
            let result = self.check_buffer();
            self.buffer.clear();
            return result;
        }

        if byte == 0x7f || byte == 0x08 {
            // Backspace
            self.buffer.pop();
            return None;
        }

        if byte == 0x03 {
            // Ctrl+C — clear
            self.buffer.clear();
            return None;
        }

        self.buffer.push(byte);
        if self.buffer.len() > self.max_len {
            self.buffer.drain(0..self.buffer.len() - self.max_len);
        }

        None
    }

    fn check_buffer(&self) -> Option<&'static str> {
        let input = String::from_utf8_lossy(&self.buffer);
        let trimmed = input.trim();

        for (prefix, agent_name) in AGENTS {
            if trimmed == *prefix || trimmed.starts_with(&format!("{} ", prefix)) {
                return Some(agent_name);
            }
        }
        None
    }
}
