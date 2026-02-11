use crate::capseal::ActionEntry;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Widget, Wrap};

#[allow(dead_code)]
pub struct ActionChainView<'a> {
    pub chain: &'a [ActionEntry],
    pub scroll_offset: usize,
}

impl<'a> Widget for ActionChainView<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Action Chain ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray));
        let inner = block.inner(area);
        block.render(area, buf);

        // Debug: log panel dimensions
        if let Ok(mut log) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/tmp/chain-debug.log")
        {
            use std::io::Write;
            let _ = writeln!(
                log,
                "ActionChain panel: {}x{}, entries: {}",
                inner.width, inner.height, self.chain.len()
            );
        }

        if inner.width < 10 || inner.height < 2 {
            return;
        }

        let gates: Vec<&ActionEntry> = self
            .chain
            .iter()
            .filter(|e| e.action_type == "gate")
            .collect();

        if gates.is_empty() {
            let msg = "Waiting for actions\u{2026}";
            let x = inner.x + (inner.width.saturating_sub(msg.len() as u16)) / 2;
            let y = inner.y + inner.height / 2;
            buf.set_string(x, y, msg, Style::default().fg(Color::DarkGray));
            return;
        }

        // Build single-line horizontal chain
        let mut spans: Vec<Span> = Vec::new();
        let total = gates.len();
        let hashed = gates.iter().filter(|e| e.receipt_hash.is_some()).count();

        for (i, entry) in gates.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(
                    " \u{2192} ",
                    Style::default().fg(Color::DarkGray),
                ));
            }

            let (dec_icon, dec_color) = match entry.decision.as_str() {
                "approve" | "approved" | "pass" => ("\u{2713}", Color::Green),
                "deny" | "denied" | "skip" => ("\u{2717}", Color::Red),
                "flag" | "flagged" | "human_review" => ("\u{26a0}", Color::Yellow),
                "pending" => ("\u{25cf}", Color::Cyan),
                _ => ("\u{25cf}", Color::White),
            };

            // Basename only
            let basename = entry
                .target
                .rsplit('/')
                .next()
                .unwrap_or(&entry.target);

            let p_str = match entry.p_fail {
                Some(p) => format!("{:.2}", p),
                None => "-".to_string(),
            };

            let hash_str = match &entry.receipt_hash {
                Some(h) if h.chars().count() >= 4 => format!("#{}", h.chars().take(4).collect::<String>()),
                Some(h) => format!("#{}", h),
                None => "#----".to_string(),
            };

            let text = format!(
                "[{}:{} {} {} {}]",
                i + 1,
                basename,
                dec_icon,
                p_str,
                hash_str,
            );

            spans.push(Span::styled(text, Style::default().fg(dec_color)));
        }

        let chain_line = Line::from(spans);

        // Chain status line
        let (status_color, status_text) = if hashed == total && total > 0 {
            (
                Color::Green,
                format!("Chain: \u{25cf} intact ({}/{})", total, total),
            )
        } else if total > 0 {
            (
                Color::Yellow,
                format!("Chain: {}/{} recorded", hashed, total),
            )
        } else {
            (Color::DarkGray, "Chain: no actions".to_string())
        };

        let status_line = Line::from(Span::styled(
            status_text,
            Style::default().fg(status_color),
        ));

        let paragraph = Paragraph::new(vec![chain_line, status_line])
            .wrap(Wrap { trim: false });

        paragraph.render(inner, buf);
    }
}
