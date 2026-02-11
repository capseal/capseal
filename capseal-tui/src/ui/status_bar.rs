use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Style};
use ratatui::widgets::Widget;

pub struct StatusBar<'a> {
    pub hints: &'a [(&'a str, &'a str)], // (key, description)
}

impl<'a> Widget for StatusBar<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let bg_style = Style::default().bg(Color::DarkGray).fg(Color::White);
        let key_style = Style::default().bg(Color::DarkGray).fg(Color::Cyan);

        // Fill background
        for x in area.x..area.x + area.width {
            buf.cell_mut((x, area.y))
                .map(|c| c.set_style(bg_style).set_symbol(" "));
        }

        let mut x = area.x + 1;
        for (i, (key, desc)) in self.hints.iter().enumerate() {
            if i > 0 {
                // Separator
                if x + 3 < area.x + area.width {
                    buf.cell_mut((x, area.y))
                        .map(|c| c.set_symbol(" ").set_style(bg_style));
                    buf.cell_mut((x + 1, area.y))
                        .map(|c| c.set_symbol("|").set_style(bg_style));
                    buf.cell_mut((x + 2, area.y))
                        .map(|c| c.set_symbol(" ").set_style(bg_style));
                    x += 3;
                }
            }

            // Key
            for ch in key.chars() {
                if x >= area.x + area.width {
                    return;
                }
                buf.cell_mut((x, area.y))
                    .map(|c| c.set_symbol(&ch.to_string()).set_style(key_style));
                x += 1;
            }

            // Space
            if x < area.x + area.width {
                buf.cell_mut((x, area.y))
                    .map(|c| c.set_symbol(" ").set_style(bg_style));
                x += 1;
            }

            // Description
            for ch in desc.chars() {
                if x >= area.x + area.width {
                    return;
                }
                buf.cell_mut((x, area.y))
                    .map(|c| c.set_symbol(&ch.to_string()).set_style(bg_style));
                x += 1;
            }
        }
    }
}
