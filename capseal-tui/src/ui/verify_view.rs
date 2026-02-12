use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Widget};

#[derive(Debug, Clone, Default)]
pub struct VerifyInfo {
    pub cap_path: String,
    pub status: String,
    pub chain_hash: String,
    pub count_label: String,
    pub count_value: String,
    pub error: Option<String>,
}

pub struct VerifyView<'a> {
    pub info: &'a VerifyInfo,
}

impl<'a> Widget for VerifyView<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Verify Receipt ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        block.render(area, buf);

        let (status_text, status_style) = match self.info.status.as_str() {
            "VERIFIED" => (
                "VERIFIED",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            "REJECTED" => (
                "REJECTED",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            _ => (
                if self.info.status.is_empty() {
                    "UNKNOWN"
                } else {
                    &self.info.status
                },
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        };

        let mut lines: Vec<Line> = vec![
            Line::raw(""),
            Line::from(vec![
                Span::styled("  Latest .cap: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    truncate_str(&self.info.cap_path, inner.width.saturating_sub(15) as usize),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled("  Status:      ", Style::default().fg(Color::DarkGray)),
                Span::styled(status_text, status_style),
            ]),
        ];

        if !self.info.chain_hash.is_empty() {
            lines.push(Line::from(vec![
                Span::styled("  Chain hash:  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    truncate_str(&self.info.chain_hash, 40),
                    Style::default().fg(Color::White),
                ),
            ]));
        }

        if !self.info.count_label.is_empty() {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("  {:<12}", format!("{}:", self.info.count_label)),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(&self.info.count_value, Style::default().fg(Color::White)),
            ]));
        }

        if let Some(err) = &self.info.error {
            lines.push(Line::raw(""));
            lines.push(Line::styled("  Error:", Style::default().fg(Color::Red)));
            lines.push(Line::styled(
                format!(
                    "  {}",
                    truncate_str(err, inner.width.saturating_sub(4) as usize)
                ),
                Style::default().fg(Color::Red),
            ));
        }

        lines.push(Line::raw(""));
        lines.push(Line::styled(
            "  F5 Refresh  |  Esc Close",
            Style::default().fg(Color::DarkGray),
        ));

        Paragraph::new(lines).render(inner, buf);
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_len {
        s.to_string()
    } else if max_len > 1 {
        let truncated: String = s.chars().take(max_len - 1).collect();
        format!("{}…", truncated)
    } else {
        s.chars().take(max_len).collect()
    }
}
