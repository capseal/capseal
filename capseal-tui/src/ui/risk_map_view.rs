use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Widget};

pub struct RiskMapView<'a> {
    pub risk_scores: &'a [(String, f64)],
    pub model_loaded: bool,
    pub episode_count: u32,
    pub profile_count: u32,
}

impl<'a> Widget for RiskMapView<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Risk Map ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        block.render(area, buf);

        if !self.model_loaded {
            Paragraph::new("  No risk model loaded\n\n  Run `capseal learn .` to train the model.\n\n  Press Esc to go back")
                .style(Style::default().fg(Color::DarkGray))
                .render(inner, buf);
            return;
        }

        let mut lines = Vec::new();

        lines.push(Line::styled(
            format!(
                "  Risk Model: {} episodes, {} active profiles",
                self.episode_count, self.profile_count
            ),
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
        lines.push(Line::raw(""));

        // Legend
        lines.push(Line::from(vec![
            Span::styled("  Low ", Style::default().fg(Color::Green)),
            Span::raw("< 0.3  "),
            Span::styled("Medium ", Style::default().fg(Color::Yellow)),
            Span::raw("< 0.6  "),
            Span::styled("High ", Style::default().fg(Color::Red)),
            Span::raw(">= 0.6"),
        ]));
        lines.push(Line::raw(""));

        // Header
        lines.push(Line::styled(
            format!("  {:<20} {:>8} {}", "Profile", "p_fail", "Risk"),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));
        lines.push(Line::styled(
            "  ".to_string() + &"-".repeat(50),
            Style::default().fg(Color::DarkGray),
        ));

        // Risk scores
        for (name, p_fail) in self.risk_scores {
            let color = risk_color(*p_fail);
            let bar_width = 20;
            let filled = (*p_fail * bar_width as f64).round() as usize;
            let empty = bar_width - filled.min(bar_width);
            let bar = format!(
                "{}{}",
                "\u{2588}".repeat(filled),  // █
                "\u{2591}".repeat(empty)      // ░
            );

            lines.push(Line::from(vec![
                Span::raw(format!("  {:<20} ", truncate(name, 20))),
                Span::styled(format!("{:>5.2}  ", p_fail), Style::default().fg(color)),
                Span::styled(bar, Style::default().fg(color)),
            ]));
        }

        if self.risk_scores.is_empty() {
            lines.push(Line::styled(
                "  No active profiles (need more training data)",
                Style::default().fg(Color::DarkGray),
            ));
        }

        lines.push(Line::raw(""));
        lines.push(Line::styled(
            "  Press Esc to close",
            Style::default().fg(Color::DarkGray),
        ));

        Paragraph::new(lines).render(inner, buf);
    }
}

fn risk_color(p_fail: f64) -> Color {
    if p_fail < 0.3 {
        Color::Green
    } else if p_fail < 0.6 {
        Color::Yellow
    } else {
        Color::Red
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}
