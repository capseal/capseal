use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Widget};

pub struct ControlPanel<'a> {
    pub workspace_name: &'a str,
    pub model_loaded: bool,
    pub session_count: usize,
    pub focused: bool,
    pub pty_active: bool,
    pub action_count: u32,
    pub denied_count: u32,
    pub chain_intact: bool,
    pub tick: u64,

    // Training state
    pub training_in_progress: bool,
    pub training_round: u32,
    pub training_total_rounds: u32,
}

impl<'a> Widget for ControlPanel<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let border_color = if self.focused {
            Color::Cyan
        } else {
            Color::DarkGray
        };

        let block = Block::default()
            .title(" CAPSEAL ")
            .title_style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(border_color));

        let inner = block.inner(area);
        block.render(area, buf);

        let cyan = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let dim = Style::default().fg(Color::DarkGray);
        let white = Style::default().fg(Color::White);
        let rule_char = "\u{2500}"; // ─

        let mut lines: Vec<Line> = Vec::new();
        let panel_w = inner.width as usize;
        let rule = rule_char.repeat(panel_w.saturating_sub(2));

        // Workspace name (truncated)
        let ws_display = truncate(self.workspace_name, panel_w.saturating_sub(2));
        lines.push(Line::from(vec![
            Span::styled(" ", dim),
            Span::styled(ws_display, white.add_modifier(Modifier::BOLD)),
        ]));

        // Model status + sessions
        let (dot, dot_color, model_text) = if self.model_loaded {
            ("\u{25cf}", Color::Green, "trained")
        } else {
            ("\u{25cf}", Color::Red, "no model")
        };
        let session_text = if self.session_count > 0 {
            format!("{} sess", self.session_count)
        } else {
            "no sessions".to_string()
        };
        lines.push(Line::from(vec![
            Span::styled(format!(" {} ", dot), Style::default().fg(dot_color)),
            Span::styled(model_text, Style::default().fg(dot_color)),
            Span::styled(format!(" \u{00b7} {}", session_text), dim),
        ]));

        // Training spinner
        if self.training_in_progress {
            let spinner = ['\u{25d0}', '\u{25d3}', '\u{25d1}', '\u{25d2}'];
            let ch = spinner[(self.tick / 4) as usize % spinner.len()];
            lines.push(Line::styled(
                format!(
                    " {} Training {}/{}",
                    ch, self.training_round, self.training_total_rounds
                ),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
        }

        lines.push(Line::raw(""));

        // Session stats (only when PTY active)
        if self.pty_active {
            lines.push(Line::styled(" Session", cyan));
            lines.push(Line::from(vec![
                Span::styled(" Acts: ", dim),
                Span::styled(format!("{}", self.action_count), white),
            ]));
            lines.push(Line::from(vec![
                Span::styled(" Deny: ", dim),
                Span::styled(
                    format!("{}", self.denied_count),
                    if self.denied_count > 0 {
                        Style::default().fg(Color::Red)
                    } else {
                        white
                    },
                ),
            ]));

            let chain_color = if self.chain_intact {
                Color::Green
            } else {
                Color::Red
            };
            lines.push(Line::from(vec![
                Span::styled(" Chain: ", dim),
                Span::styled(
                    if self.chain_intact {
                        "\u{25cf} ok"
                    } else {
                        "\u{25cf} BROKEN"
                    },
                    Style::default().fg(chain_color),
                ),
            ]));

            lines.push(Line::raw(""));
        }

        // Separator
        lines.push(Line::styled(format!(" {}", rule), dim));

        // Single-letter shortcuts
        let shortcuts: &[(&str, &str)] = &[
            ("s", "Scan & fix"),
            ("v", "Sessions"),
            ("r", "Risk report"),
            ("c", "Configure"),
            ("q", "Quit"),
        ];

        for (key, label) in shortcuts {
            lines.push(Line::from(vec![
                Span::styled(
                    format!(" {} ", key),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(*label, dim),
            ]));
        }

        Paragraph::new(lines).render(inner, buf);
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len > 3 {
        format!("{}\u{2026}", &s[..max_len.saturating_sub(1)])
    } else {
        s[..max_len].to_string()
    }
}
