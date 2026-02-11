use crate::capseal::CapSealState;
use crate::ui::action_chain_view::ActionChainView;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Widget};

pub struct SessionMonitor<'a> {
    pub state: &'a CapSealState,
    pub scroll_offset: usize,
}

impl<'a> Widget for SessionMonitor<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Session Monitor ")
            .title_style(Style::default().fg(Color::White))
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner = block.inner(area);
        block.render(area, buf);

        if inner.height < 6 {
            return;
        }

        // Split: Latest Gate 40% | Files+Stats 30% | Event Log 30%
        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(40),
                Constraint::Percentage(30),
                Constraint::Percentage(30),
            ])
            .split(inner);

        render_latest_gate(self.state, sections[0], buf);
        render_middle(self.state, self.scroll_offset, sections[1], buf);
        render_event_log(self.state, self.scroll_offset, sections[2], buf);
    }
}

// ── Section 1: Latest Gate ──────────────────────────────────────────────

fn render_latest_gate(state: &CapSealState, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
        .title(" Latest Gate ")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    block.render(area, buf);

    if inner.height == 0 || inner.width < 10 {
        return;
    }

    let latest_gate = state
        .action_chain
        .iter()
        .rev()
        .find(|e| e.action_type == "gate");

    match latest_gate {
        None => {
            let lines = vec![
                Line::raw(""),
                Line::styled(
                    "  Waiting for agent actions\u{2026}",
                    Style::default().fg(Color::DarkGray),
                ),
                Line::raw(""),
                Line::styled(
                    "  Gate decisions will appear here",
                    Style::default().fg(Color::DarkGray),
                ),
                Line::styled(
                    "  as the agent edits files.",
                    Style::default().fg(Color::DarkGray),
                ),
            ];
            Paragraph::new(lines).render(inner, buf);
        }
        Some(gate) => {
            let (icon, color, label) = decision_style(&gate.decision);

            let mut lines = vec![Line::from(vec![
                Span::styled(
                    format!("  {} {}  ", icon, label),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    truncate_str(&gate.target, 30),
                    Style::default().fg(Color::White),
                ),
                Span::styled(
                    format!("  p_fail: {:.2}", gate.p_fail.unwrap_or(0.0)),
                    Style::default().fg(Color::DarkGray),
                ),
            ])];

            // Show diff if available
            if let Some(ref diff) = gate.diff {
                lines.push(Line::raw(""));
                let max_diff_lines = (inner.height as usize).saturating_sub(4).min(6);
                for diff_line in diff.lines().take(max_diff_lines) {
                    let style = if diff_line.starts_with('+') {
                        Style::default().fg(Color::Green)
                    } else if diff_line.starts_with('-') {
                        Style::default().fg(Color::Red)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    };
                    lines.push(Line::styled(
                        format!("  {}", truncate_str(diff_line, inner.width as usize - 4)),
                        style,
                    ));
                }
            }

            // Show risk factors if available
            if let Some(ref factors) = gate.risk_factors {
                if !factors.is_empty() {
                    lines.push(Line::raw(""));
                    lines.push(Line::styled(
                        format!("  {}", truncate_str(factors, inner.width as usize - 4)),
                        Style::default().fg(Color::DarkGray),
                    ));
                }
            }

            Paragraph::new(lines).render(inner, buf);
        }
    }
}

// ── Section 2: Files Touched + Live Stats ───────────────────────────────

fn render_middle(state: &CapSealState, scroll_offset: usize, area: Rect, buf: &mut Buffer) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(area);

    ActionChainView {
        chain: &state.action_chain,
        scroll_offset,
    }
    .render(cols[0], buf);
    render_live_stats(state, cols[1], buf);
}

fn render_live_stats(state: &CapSealState, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
        .title(" Live Stats ")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    block.render(area, buf);

    if inner.height == 0 {
        return;
    }

    let total = state.action_count;
    let denied = state.denied_count;
    let approved = total.saturating_sub(denied);

    // Trust score
    let trust = if total == 0 {
        100.0
    } else {
        (approved as f64 / total as f64) * 100.0
    };

    let trust_filled = (trust / 10.0).round() as usize;
    let trust_filled = trust_filled.min(10);
    let trust_bar = format!(
        "{}{}",
        "\u{2588}".repeat(trust_filled),
        "\u{2591}".repeat(10 - trust_filled)
    );

    let trust_color = if trust >= 80.0 {
        Color::Green
    } else if trust >= 50.0 {
        Color::Yellow
    } else {
        Color::Red
    };

    let chain_color = if state.chain_intact {
        Color::Green
    } else {
        Color::Red
    };

    // Session duration
    let secs = state.session_duration_secs();
    let duration = format!("{}m {:02}s", secs / 60, secs % 60);

    // Operator status
    let (op_icon, op_color, op_text) = if state.operator_online {
        (
            "\u{25cf}",
            Color::Green,
            format!("online ({} ch)", state.operator_channels),
        )
    } else {
        ("\u{25cb}", Color::DarkGray, "offline".to_string())
    };

    let lines = vec![
        Line::raw(format!(" Duration:  {}", duration)),
        Line::raw(format!(
            " Actions:   {} ({}\u{2713} {}\u{2717})",
            total, approved, denied
        )),
        Line::from(vec![
            Span::raw(format!(" Trust:     {:.0}% ", trust)),
            Span::styled(trust_bar, Style::default().fg(trust_color)),
        ]),
        Line::from(vec![
            Span::raw(" Chain:     "),
            Span::styled(
                format!(
                    "\u{25cf} {} ({}/{})",
                    if state.chain_intact { "intact" } else { "BROKEN" },
                    total,
                    total,
                ),
                Style::default().fg(chain_color),
            ),
        ]),
        Line::from(vec![
            Span::raw(" Operator:  "),
            Span::styled(
                format!("{} {}", op_icon, op_text),
                Style::default().fg(op_color),
            ),
        ]),
    ];

    Paragraph::new(lines).render(inner, buf);
}

// ── Section 3: Event Log ────────────────────────────────────────────────

fn render_event_log(state: &CapSealState, scroll_offset: usize, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
        .title(" Event Log ")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    block.render(area, buf);

    if inner.height == 0 {
        return;
    }

    if state.action_chain.is_empty() {
        Paragraph::new(Line::styled(
            " No events yet",
            Style::default().fg(Color::DarkGray),
        ))
        .render(inner, buf);
        return;
    }

    let mut lines: Vec<Line> = Vec::new();
    let target_max = (inner.width as usize).saturating_sub(32).max(4);

    for event in &state.action_chain {
        let (icon, color) = match event.decision.as_str() {
            "approve" | "approved" | "pass" => ("\u{2713}", Color::Green),
            d if d.starts_with("recorded") => ("\u{2713}", Color::Green),
            "deny" | "denied" | "skip" => ("\u{2717}", Color::Red),
            "flag" | "flagged" | "human_review" => ("\u{25b2}", Color::Yellow),
            "waiting" => ("\u{25cf}", Color::Cyan),
            "pending" => ("\u{25cf}", Color::Magenta),
            _ => ("\u{25cf}", Color::White),
        };

        let time: String = event.timestamp.chars().take(5).collect();

        let detail = if let Some(p) = event.p_fail {
            format!("p={:.2}  {}", p, event.decision)
        } else if let Some(ref hash) = event.receipt_hash {
            let h: String = hash.chars().take(8).collect();
            format!("sha:{}\u{2026}", h)
        } else {
            event.decision.clone()
        };

        lines.push(Line::from(vec![
            Span::styled(format!(" {}  ", time), Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{} ", icon), Style::default().fg(color)),
            Span::raw(format!("{:<6} ", event.action_type)),
            Span::styled(
                format!("{:<w$}", truncate_str(&event.target, target_max), w = target_max),
                Style::default().fg(Color::White),
            ),
            Span::styled(format!(" {}", detail), Style::default().fg(Color::DarkGray)),
        ]));
    }

    // Scrolling: auto-scroll to newest unless manually scrolled
    let visible_lines = inner.height as usize;
    let total_lines = lines.len();

    let start = if scroll_offset > 0 && total_lines > visible_lines {
        let max_offset = total_lines.saturating_sub(visible_lines);
        let clamped = scroll_offset.min(max_offset);
        max_offset.saturating_sub(clamped)
    } else {
        total_lines.saturating_sub(visible_lines)
    };

    let end = (start + visible_lines).min(total_lines);
    let visible = &lines[start..end];
    Paragraph::new(visible.to_vec()).render(inner, buf);
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn decision_style(decision: &str) -> (&str, Color, &str) {
    match decision {
        "approve" | "approved" | "pass" => ("\u{2713}", Color::Green, "APPROVED"),
        "deny" | "denied" => ("\u{2717}", Color::Red, "DENIED"),
        "skip" => ("\u{2717}", Color::Red, "SKIPPED"),
        "flag" | "flagged" | "human_review" => ("\u{25b2}", Color::Yellow, "FLAGGED"),
        "waiting" => ("\u{25cf}", Color::Cyan, "WAITING"),
        "pending" => ("\u{25cf}", Color::Magenta, "PENDING"),
        _ => ("\u{25cf}", Color::White, "UNKNOWN"),
    }
}

fn truncate_str(s: &str, max_len: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_len {
        s.to_string()
    } else if max_len > 1 {
        let truncated: String = s.chars().take(max_len - 1).collect();
        format!("{}\u{2026}", truncated)
    } else {
        s.chars().take(max_len).collect()
    }
}
