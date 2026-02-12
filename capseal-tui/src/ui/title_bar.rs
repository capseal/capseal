use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::Widget;

pub struct TitleBar<'a> {
    pub workspace_name: &'a str,
    pub process_name: Option<&'a str>,
    pub model_trained: bool,
    pub session_count: usize,
    pub session_active: bool,
    pub operator_online: bool,
    pub operator_channel_types: &'a [String],
    pub operator_voice_connected: bool,
    pub voice_active: bool,
    pub operator_workspace_name: Option<&'a str>,
    pub operator_workspace_mismatch: bool,
    pub operator_last_alert_age: Option<&'a str>,
    pub pending_intervention: Option<(&'a str, &'a str)>,
}

impl<'a> Widget for TitleBar<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let brand_style = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let bg_style = Style::default().bg(Color::DarkGray).fg(Color::White);
        // Fill background
        for x in area.x..area.x + area.width {
            buf.cell_mut((x, area.y))
                .map(|c| c.set_style(bg_style).set_symbol(" "));
        }

        // Left side: CAPSEAL · workspace [· process]
        let mut left = format!(" CAPSEAL \u{00b7} {}", self.workspace_name);
        if let Some(proc) = self.process_name {
            left.push_str(&format!(" \u{00b7} {}", proc));
        }

        // Render left side
        for (i, ch) in left.chars().enumerate() {
            let x = area.x + i as u16;
            if x >= area.x + area.width {
                break;
            }
            // "CAPSEAL" portion (first 8 chars including space) gets brand style
            let style = if i < 8 { brand_style } else { bg_style };
            buf.cell_mut((x, area.y))
                .map(|c| c.set_symbol(&ch.to_string()).set_style(style));
        }

        // Right side: ● status · N sessions
        let (dot, dot_color) = if self.model_trained {
            ("\u{25cf}", Color::Green) // ●
        } else {
            ("\u{25cf}", Color::Red)
        };
        let model_text = if self.model_trained {
            "trained"
        } else {
            "no model"
        };

        let mut right_parts = format!(
            "{} {} \u{00b7} {} sessions ",
            dot, model_text, self.session_count
        );
        if self.session_active {
            right_parts = format!(
                "{} {} \u{00b7} {} sessions \u{00b7} active ",
                dot, model_text, self.session_count
            );
        }

        let mut op_status = if self.operator_online {
            let channels = if self.operator_channel_types.is_empty() {
                "channels".to_string()
            } else {
                self.operator_channel_types
                    .iter()
                    .map(|s| s.to_lowercase())
                    .collect::<Vec<String>>()
                    .join(", ")
            };
            if self.operator_voice_connected {
                format!("operator: online ({channels}, voice)")
            } else {
                format!("operator: online ({channels})")
            }
        } else {
            "operator: offline".to_string()
        };
        let voice_text = if self.voice_active {
            "voice: \u{1f50a} ON"
        } else {
            "voice: \u{1f507} OFF"
        };
        op_status.push_str(&format!(" \u{00b7} {}", voice_text));

        if let Some(age) = self.operator_last_alert_age {
            op_status.push_str(&format!(" \u{00b7} last alert {age}"));
        }
        if let Some((action, source)) = self.pending_intervention {
            op_status.push_str(&format!(
                " \u{00b7} \u{26a0} pending {} ({})",
                action, source
            ));
        }
        if self.operator_workspace_mismatch {
            if let Some(ws) = self.operator_workspace_name {
                op_status.push_str(&format!(" \u{00b7} \u{26a0} op ws: {}", ws));
            } else {
                op_status.push_str(" \u{00b7} \u{26a0} op ws mismatch");
            }
        }

        right_parts.push_str(&format!("\u{00b7} {} ", op_status));

        // Render right side
        let max_right = area.width.saturating_sub(10) as usize;
        if right_parts.chars().count() > max_right && max_right > 4 {
            let trimmed: String = right_parts
                .chars()
                .rev()
                .take(max_right - 1)
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            right_parts = format!("\u{2026}{}", trimmed);
        }
        let right_len = right_parts.chars().count() as u16;
        if right_len < area.width {
            let start_x = area.x + area.width - right_len;
            for (i, ch) in right_parts.chars().enumerate() {
                let rx = start_x + i as u16;
                if rx >= area.x + area.width {
                    break;
                }
                // Color the dot character
                let style = if i == 0 {
                    Style::default().bg(Color::DarkGray).fg(dot_color)
                } else {
                    bg_style
                };
                buf.cell_mut((rx, area.y))
                    .map(|c| c.set_symbol(&ch.to_string()).set_style(style));
            }
        }
    }
}
