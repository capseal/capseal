use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::Widget;

pub struct TerminalWidget<'a> {
    pub screen: &'a vt100::Screen,
}

impl<'a> Widget for TerminalWidget<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let (screen_rows, screen_cols) = self.screen.size();

        for row in 0..area.height.min(screen_rows) {
            for col in 0..area.width.min(screen_cols) {
                let cell = match self.screen.cell(row, col) {
                    Some(c) => c,
                    None => continue,
                };

                let contents = cell.contents();

                let fg = vt_color_to_ratatui(cell.fgcolor());
                let bg = vt_color_to_ratatui_bg(cell.bgcolor());

                let mut style = Style::default().fg(fg).bg(bg);
                if cell.bold() {
                    style = style.add_modifier(Modifier::BOLD);
                }
                if cell.italic() {
                    style = style.add_modifier(Modifier::ITALIC);
                }
                if cell.underline() {
                    style = style.add_modifier(Modifier::UNDERLINED);
                }
                if cell.inverse() {
                    // Swap fg and bg
                    let swapped = Style::default().fg(bg).bg(fg);
                    style = swapped;
                    if cell.bold() {
                        style = style.add_modifier(Modifier::BOLD);
                    }
                }

                let buf_x = area.x + col;
                let buf_y = area.y + row;

                if let Some(buf_cell) = buf.cell_mut((buf_x, buf_y)) {
                    if contents.is_empty() {
                        buf_cell.set_symbol(" ");
                    } else {
                        buf_cell.set_symbol(&contents);
                    }
                    buf_cell.set_style(style);
                }
            }
        }

        // Render cursor
        let (cursor_row, cursor_col) = self.screen.cursor_position();
        if cursor_row < area.height && cursor_col < area.width {
            if let Some(cell) = buf.cell_mut((area.x + cursor_col, area.y + cursor_row)) {
                cell.set_style(
                    cell.style()
                        .add_modifier(Modifier::REVERSED),
                );
            }
        }
    }
}

fn vt_color_to_ratatui(color: vt100::Color) -> Color {
    match color {
        vt100::Color::Default => Color::Reset,
        vt100::Color::Idx(i) => Color::Indexed(i),
        vt100::Color::Rgb(r, g, b) => Color::Rgb(r, g, b),
    }
}

/// For background colors, map Default to Black instead of Reset
fn vt_color_to_ratatui_bg(color: vt100::Color) -> Color {
    match color {
        vt100::Color::Default => Color::Black,
        vt100::Color::Idx(i) => Color::Indexed(i),
        vt100::Color::Rgb(r, g, b) => Color::Rgb(r, g, b),
    }
}
