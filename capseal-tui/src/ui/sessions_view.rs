use crate::capseal::sessions::SessionSummary;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Widget};

pub struct SessionsView<'a> {
    pub sessions: &'a [SessionSummary],
    pub selected: usize,
}

impl<'a> Widget for SessionsView<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Sessions ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        block.render(area, buf);

        if self.sessions.is_empty() {
            Paragraph::new("  No sessions found\n\n  Run an agent with CapSeal to create sessions.\n\n  Press Esc to go back")
                .style(Style::default().fg(Color::DarkGray))
                .render(inner, buf);
            return;
        }

        let mut lines = Vec::new();

        // Header
        lines.push(Line::styled(
            format!(
                " {:<20} {:<12} {:>6} {:>8} {:<16}",
                "Date", "Agent", "Acts", "Proof", "Type"
            ),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        lines.push(Line::styled(
            " ".to_string() + &"-".repeat(inner.width.saturating_sub(2) as usize),
            Style::default().fg(Color::DarkGray),
        ));

        // Sessions
        for (i, session) in self.sessions.iter().enumerate() {
            let style = if i == self.selected {
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };

            let prefix = if i == self.selected { ">" } else { " " };
            let proof_icon = if session.proof_verified {
                Span::styled(" \u{2713}", Style::default().fg(Color::Green))
            } else {
                Span::styled(" \u{2717}", Style::default().fg(Color::Red))
            };

            lines.push(Line::from(vec![
                Span::styled(
                    format!(
                        "{}{:<20} {:<12} {:>6} ",
                        prefix,
                        truncate(&session.date, 20),
                        truncate(&session.agent, 12),
                        session.action_count,
                    ),
                    style,
                ),
                proof_icon,
                Span::styled(
                    format!("       {:<16}", truncate(&session.proof_type, 16)),
                    style,
                ),
            ]));
        }

        lines.push(Line::raw(""));
        lines.push(Line::styled(
            " Esc Back  |  Up/Down Navigate",
            Style::default().fg(Color::DarkGray),
        ));

        Paragraph::new(lines).render(inner, buf);
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
