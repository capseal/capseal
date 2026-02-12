use crate::capseal::CapSealState;
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Widget};
use std::collections::HashMap;

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
            let (icon, color, label) = plain_decision_style(&gate.decision);
            let risk_label = gate.label.as_deref().unwrap_or("unclassified");
            let p = gate.p_fail.unwrap_or(0.0).clamp(0.0, 1.0);
            let chance = format!("{:.0}%", p * 100.0);
            let chance_color = if p < 0.3 {
                Color::Green
            } else if p < 0.6 {
                Color::Yellow
            } else {
                Color::Red
            };

            let mut lines = vec![Line::from(vec![
                Span::styled(
                    format!("  {} {}  ", icon, label),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    truncate_str(&gate.target, 24),
                    Style::default().fg(Color::White),
                ),
                Span::styled(
                    format!("  {}", truncate_str(risk_label, 26)),
                    Style::default().fg(Color::Cyan),
                ),
                Span::styled(
                    format!("  chance this breaks: {}", chance),
                    Style::default().fg(chance_color),
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

fn render_middle(state: &CapSealState, _scroll_offset: usize, area: Rect, buf: &mut Buffer) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
        .split(area);

    render_files_touched(state, cols[0], buf);
    render_live_stats(state, cols[1], buf);
}

#[derive(Clone)]
struct FileRiskEntry {
    file: String,
    p_fail: Option<f64>,
    label: Option<String>,
}

fn render_files_touched(state: &CapSealState, area: Rect, buf: &mut Buffer) {
    let block = Block::default()
        .title(" Files Touched ")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray));
    let inner = block.inner(area);
    block.render(area, buf);

    if inner.height == 0 || inner.width < 12 {
        return;
    }

    // Deduplicate by file and keep the latest gate decision for each file.
    let mut latest_by_file: HashMap<String, FileRiskEntry> = HashMap::new();
    for event in &state.action_chain {
        if event.action_type != "gate" {
            continue;
        }
        if event.target.trim().is_empty() {
            continue;
        }
        latest_by_file.insert(
            event.target.clone(),
            FileRiskEntry {
                file: event.target.clone(),
                p_fail: event.p_fail,
                label: event.label.clone(),
            },
        );
    }

    if latest_by_file.is_empty() {
        Paragraph::new(vec![
            Line::raw(""),
            Line::styled(
                "  No gated file actions yet",
                Style::default().fg(Color::DarkGray),
            ),
        ])
        .render(inner, buf);
        return;
    }

    let mut entries: Vec<FileRiskEntry> = latest_by_file.into_values().collect();
    entries.sort_by(|a, b| {
        let ap = a.p_fail.unwrap_or(-1.0);
        let bp = b.p_fail.unwrap_or(-1.0);
        bp.partial_cmp(&ap).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::styled(
        format!("  {} unique files", entries.len()),
        Style::default().fg(Color::DarkGray),
    ));

    let max_rows = inner.height.saturating_sub(1) as usize;
    let file_w = (inner.width as usize).saturating_sub(18).max(8);

    for entry in entries.into_iter().take(max_rows.saturating_sub(1)) {
        let (bar, bar_color, p_text, warn, warn_color) = match entry.p_fail {
            Some(p_raw) => {
                let p = p_raw.clamp(0.0, 1.0);
                let filled = ((p * 10.0).round() as usize).min(10);
                let bar = format!(
                    "{}{}",
                    "\u{2588}".repeat(filled),
                    "\u{2591}".repeat(10 - filled)
                );
                let color = if p < 0.3 {
                    Color::Green
                } else if p < 0.6 {
                    Color::Yellow
                } else {
                    Color::Red
                };
                let warn = if p >= 0.7 { "\u{26a0}" } else { " " };
                let warn_color = if p >= 0.7 {
                    Color::Red
                } else {
                    Color::DarkGray
                };
                (bar, color, format!("{:.0}%", p * 100.0), warn, warn_color)
            }
            None => (
                "\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}\u{2591}"
                    .to_string(),
                Color::DarkGray,
                "--".to_string(),
                " ",
                Color::DarkGray,
            ),
        };

        lines.push(Line::from(vec![
            Span::styled(format!("{} ", warn), Style::default().fg(warn_color)),
            Span::styled(
                format!("{:<w$}", truncate_str(&entry.file, file_w), w = file_w),
                Style::default().fg(Color::White),
            ),
            Span::raw(" "),
            Span::styled(bar, Style::default().fg(bar_color)),
            Span::styled(format!(" {}", p_text), Style::default().fg(Color::DarkGray)),
        ]));
        if let Some(label) = entry.label {
            if !label.is_empty() {
                lines.push(Line::styled(
                    format!("    {}", truncate_str(&label, inner.width as usize - 6)),
                    Style::default().fg(Color::DarkGray),
                ));
            }
        }
    }

    Paragraph::new(lines).render(inner, buf);
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

    let attempted = state.gates_attempted;
    let executed = state.actions_recorded;
    let denied = state.gates_denied;

    let chain_color = if !state.chain_verified {
        Color::Yellow
    } else if state.chain_intact {
        Color::Green
    } else {
        Color::Red
    };

    // Session duration
    let secs = state.session_duration_secs();
    let duration = format!("{}m {:02}s", secs / 60, secs % 60);

    // Operator status
    let channel_names = if state.operator_channel_types.is_empty() {
        format!("{} ch", state.operator_channels)
    } else {
        state.operator_channel_types.join("+")
    };
    let (op_icon, op_color, op_text) = if state.operator_online {
        (
            "\u{25cf}",
            Color::Green,
            if state.operator_voice_connected {
                format!("online ({}, voice)", channel_names)
            } else {
                format!("online ({})", channel_names)
            },
        )
    } else {
        ("\u{25cb}", Color::DarkGray, "offline".to_string())
    };

    let lines = vec![
        Line::raw(format!(" Duration:  {}", duration)),
        Line::raw(format!(" Try/Exec:  {} / {}", attempted, executed)),
        Line::raw(format!(" Denied:    {}", denied)),
        Line::from(vec![
            Span::raw(" Chain:     "),
            Span::styled(
                if !state.chain_verified {
                    "\u{25cf} unverified".to_string()
                } else if state.chain_intact {
                    "\u{25cf} intact \u{2713}".to_string()
                } else {
                    "\u{25cf} broken \u{2717}".to_string()
                },
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
        let time: String = event.timestamp.chars().take(5).collect();
        match event.action_type.as_str() {
            "gate" => {
                let (icon, color, plain) = plain_decision_style(&event.decision);
                let p = event.p_fail.unwrap_or(0.0).clamp(0.0, 1.0);
                let chance = format!("{:>3.0}%", p * 100.0);
                let mut note = String::new();
                if let Some(reason) = &event.risk_factors {
                    if reason.to_lowercase().contains("override") {
                        note = "  operator override".to_string();
                    }
                }
                lines.push(Line::from(vec![
                    Span::styled(format!(" {}  ", time), Style::default().fg(Color::DarkGray)),
                    Span::styled(format!("{} ", icon), Style::default().fg(color)),
                    Span::styled(
                        format!("{:<22}", truncate_str(plain, 22)),
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        format!(
                            "{:<w$}",
                            truncate_str(&event.target, target_max),
                            w = target_max
                        ),
                        Style::default().fg(Color::White),
                    ),
                    Span::styled(format!("  {}", chance), Style::default().fg(color)),
                    Span::styled(note, Style::default().fg(Color::Magenta)),
                ]));
            }
            "record" => {
                let h = event
                    .receipt_hash
                    .as_ref()
                    .map(|s| format!(" sha:{}…", s.chars().take(8).collect::<String>()))
                    .unwrap_or_default();
                lines.push(Line::from(vec![
                    Span::styled(format!(" {}  ", time), Style::default().fg(Color::DarkGray)),
                    Span::styled("\u{00b7} ", Style::default().fg(Color::DarkGray)),
                    Span::styled("Action recorded      ", Style::default().fg(Color::Gray)),
                    Span::styled(
                        format!(
                            "{:<w$}",
                            truncate_str(&event.target, target_max),
                            w = target_max
                        ),
                        Style::default().fg(Color::Gray),
                    ),
                    Span::styled(h, Style::default().fg(Color::DarkGray)),
                ]));
            }
            "session_start" => {
                lines.push(Line::from(vec![
                    Span::styled(format!(" {}  ", time), Style::default().fg(Color::DarkGray)),
                    Span::styled("\u{25a0} ", Style::default().fg(Color::Blue)),
                    Span::styled("Session started", Style::default().fg(Color::Blue)),
                ]));
            }
            "seal" => {
                let hash = event
                    .receipt_hash
                    .as_ref()
                    .map(|h| format!(" receipt {}…", h.chars().take(8).collect::<String>()))
                    .unwrap_or_default();
                let color = if event.decision == "seal_broken" {
                    Color::Red
                } else {
                    Color::Green
                };
                lines.push(Line::from(vec![
                    Span::styled(format!(" {}  ", time), Style::default().fg(Color::DarkGray)),
                    Span::styled("\u{25a0} ", Style::default().fg(color)),
                    Span::styled(
                        "Session sealed",
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(hash, Style::default().fg(Color::DarkGray)),
                ]));
            }
            "chain_break" => {
                lines.push(Line::from(vec![
                    Span::styled(format!(" {}  ", time), Style::default().fg(Color::DarkGray)),
                    Span::styled("\u{2717} ", Style::default().fg(Color::Red)),
                    Span::styled(
                        "Chain broken",
                        Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
                    ),
                ]));
            }
            _ => {
                let (icon, color) = match event.decision.as_str() {
                    "waiting" => ("\u{25cf}", Color::Cyan),
                    "pending" => ("\u{25cf}", Color::Magenta),
                    _ => ("\u{25cf}", Color::White),
                };
                lines.push(Line::from(vec![
                    Span::styled(format!(" {}  ", time), Style::default().fg(Color::DarkGray)),
                    Span::styled(format!("{} ", icon), Style::default().fg(color)),
                    Span::raw(format!("{:<6} ", event.action_type)),
                    Span::styled(
                        format!(
                            "{:<w$}",
                            truncate_str(&event.target, target_max),
                            w = target_max
                        ),
                        Style::default().fg(Color::White),
                    ),
                    Span::styled(
                        format!(" {}", event.decision),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            }
        }
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

fn plain_decision_style(decision: &str) -> (&str, Color, &str) {
    match decision {
        "approve" | "approved" | "pass" => ("\u{2713}", Color::Green, "Safe to proceed"),
        "deny" | "denied" | "skip" => ("\u{2717}", Color::Red, "Blocked for safety"),
        "flag" | "flagged" | "human_review" => ("\u{25b2}", Color::Yellow, "Proceed with caution"),
        "waiting" => ("\u{25cf}", Color::Cyan, "WAITING"),
        "pending" => ("\u{25cf}", Color::Magenta, "PENDING"),
        _ => ("\u{25cf}", Color::White, "Unknown decision"),
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
