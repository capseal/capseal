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

        let mut right_parts = format!("{} {} \u{00b7} {} sessions ", dot, model_text, self.session_count);
        if self.session_active {
            right_parts = format!("{} {} \u{00b7} {} sessions \u{00b7} active ", dot, model_text, self.session_count);
        }

        // Render right side
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
