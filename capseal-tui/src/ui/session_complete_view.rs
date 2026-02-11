use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph, Widget};

pub struct SessionCompleteInfo {
    pub action_count: u32,
    pub denied_count: u32,
    pub duration_secs: u64,
    pub cap_file: Option<String>,
    pub chain_intact: bool,
}

pub struct SessionCompleteView<'a> {
    pub info: &'a SessionCompleteInfo,
}

impl<'a> SessionCompleteView<'a> {
    /// Compute the centered overlay rect within the given area.
    pub fn overlay_rect(area: Rect) -> Rect {
        let width = 60u16.min(area.width.saturating_sub(4));
        let height = 14u16.min(area.height.saturating_sub(4));
        Rect::new(
            area.x + (area.width.saturating_sub(width)) / 2,
            area.y + (area.height.saturating_sub(height)) / 2,
            width,
            height,
        )
    }
}

impl<'a> Widget for SessionCompleteView<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        // Clear area behind overlay
        Clear.render(area, buf);

        let block = Block::default()
            .title(" Session Complete ")
            .title_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        block.render(area, buf);

        let label = Style::default().fg(Color::DarkGray);
        let value = Style::default().fg(Color::White);

        // Duration
        let duration = if self.info.duration_secs >= 3600 {
            format!(
                "{}h {}m",
                self.info.duration_secs / 3600,
                (self.info.duration_secs % 3600) / 60
            )
        } else if self.info.duration_secs >= 60 {
            format!(
                "{}m {}s",
                self.info.duration_secs / 60,
                self.info.duration_secs % 60
            )
        } else {
            format!("{}s", self.info.duration_secs)
        };

        let approved = self.info.action_count.saturating_sub(self.info.denied_count);

        // Chain status
        let (chain_dot, chain_color, chain_text) = if self.info.chain_intact {
            ("\u{25cf}", Color::Green, "intact")
        } else {
            ("\u{25cf}", Color::Red, "BROKEN")
        };

        let receipt_display = self
            .info
            .cap_file
            .as_deref()
            .unwrap_or("(no receipt)");

        let lines = vec![
            Line::raw(""),
            Line::from(vec![
                Span::styled("  Duration:    ", label),
                Span::styled(duration, value),
            ]),
            Line::from(vec![
                Span::styled("  Actions:     ", label),
                Span::styled(format!("{}", self.info.action_count), value),
            ]),
            Line::from(vec![
                Span::styled("  Approved:    ", label),
                Span::styled(format!("{}", approved), value),
            ]),
            Line::from(vec![
                Span::styled("  Denied:      ", label),
                Span::styled(format!("{}", self.info.denied_count), value),
            ]),
            Line::from(vec![
                Span::styled("  Receipt:     ", label),
                Span::styled(receipt_display, value),
            ]),
            Line::from(vec![
                Span::styled("  Chain:       ", label),
                Span::styled(
                    format!("{} {}", chain_dot, chain_text),
                    Style::default().fg(chain_color),
                ),
            ]),
            Line::raw(""),
            Line::raw(""),
            Line::styled(
                "  [Enter] Back   [v] Verify   [q] Quit",
                Style::default().fg(Color::DarkGray),
            ),
        ];

        Paragraph::new(lines).render(inner, buf);
    }
}
