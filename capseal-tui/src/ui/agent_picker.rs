use crossterm::event::{KeyCode, KeyEvent};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph, Widget};

pub struct AgentOption {
    pub label: &'static str,
    pub command: &'static str,
    pub detected: bool,
}

pub struct AgentPicker {
    pub selected: usize,
    pub agents: Vec<AgentOption>,
}

pub enum AgentPickerResult {
    Continue,
    Selected(String),
    Cancelled,
}

impl AgentPicker {
    /// Create a new agent picker.
    /// `default_agent` is checked first (e.g. "claude", "codex", "aider"),
    /// then falls back to `provider` (e.g. "anthropic", "openai").
    pub fn new(default_agent: &str, provider: &str) -> Self {
        let agents = vec![
            AgentOption {
                label: "Claude Code",
                command: "claude",
                detected: which_exists("claude"),
            },
            AgentOption {
                label: "Codex (OpenAI)",
                command: "codex",
                detected: which_exists("codex"),
            },
            AgentOption {
                label: "Gemini CLI",
                command: "gemini",
                detected: which_exists("gemini"),
            },
            AgentOption {
                label: "Aider",
                command: "aider",
                detected: which_exists("aider"),
            },
            AgentOption {
                label: "None (just a shell)",
                command: "",
                detected: true,
            },
        ];

        // Pre-select: check default_agent first, then provider
        let default_idx = if !default_agent.is_empty() {
            agents
                .iter()
                .position(|a| a.command == default_agent)
                .unwrap_or(0)
        } else if provider.contains("openai") {
            1
        } else if provider.contains("gemini") || provider.contains("google") {
            2
        } else {
            0
        };

        Self {
            selected: default_idx,
            agents,
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> AgentPickerResult {
        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                if self.selected > 0 {
                    self.selected -= 1;
                }
                AgentPickerResult::Continue
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if self.selected < self.agents.len() - 1 {
                    self.selected += 1;
                }
                AgentPickerResult::Continue
            }
            KeyCode::Enter => {
                let agent = &self.agents[self.selected];
                if !agent.detected && !agent.command.is_empty() {
                    // Can't select a missing agent
                    AgentPickerResult::Continue
                } else {
                    AgentPickerResult::Selected(agent.command.to_string())
                }
            }
            KeyCode::Esc => AgentPickerResult::Cancelled,
            _ => AgentPickerResult::Continue,
        }
    }
}

impl Widget for &AgentPicker {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let width = 36u16.min(area.width.saturating_sub(4));
        let height = (self.agents.len() as u16) + 6;
        let x = area.x + (area.width.saturating_sub(width)) / 2;
        let y = area.y + (area.height.saturating_sub(height)) / 2;
        let overlay = Rect::new(x, y, width, height);

        Clear.render(overlay, buf);

        let block = Block::default()
            .title(" Select Agent ")
            .title_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(overlay);
        block.render(overlay, buf);

        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::raw(""));

        for (i, agent) in self.agents.iter().enumerate() {
            let is_selected = i == self.selected;
            let prefix = if is_selected { " \u{25b8} " } else { "   " };

            let style = if !agent.detected && !agent.command.is_empty() {
                Style::default().fg(Color::DarkGray)
            } else if is_selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };

            let suffix = if !agent.detected && !agent.command.is_empty() {
                " (not found)"
            } else {
                ""
            };

            lines.push(Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(agent.label, style),
                Span::styled(suffix, Style::default().fg(Color::DarkGray)),
            ]));
        }

        lines.push(Line::raw(""));
        lines.push(Line::styled(
            " \u{2191}\u{2193} select  enter launch  esc cancel",
            Style::default().fg(Color::DarkGray),
        ));

        Paragraph::new(lines).render(inner, buf);
    }
}

/// Check if a binary exists on PATH
fn which_exists(cmd: &str) -> bool {
    std::env::var_os("PATH")
        .map(|paths| {
            std::env::split_paths(&paths).any(|dir| {
                let full = dir.join(cmd);
                full.is_file() || full.with_extension("exe").is_file()
            })
        })
        .unwrap_or(false)
}
