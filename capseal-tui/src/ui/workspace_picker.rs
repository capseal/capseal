use crate::workspace::PickerState;
use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

pub struct WorkspacePicker<'a> {
    pub state: &'a PickerState,
}

impl<'a> Widget for WorkspacePicker<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let cyan = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let dim = Style::default().fg(Color::DarkGray);
        let white = Style::default().fg(Color::White);
        let green = Style::default().fg(Color::Green);
        let selected_style = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);

        let mut lines: Vec<Line> = Vec::new();

        // Header
        lines.push(Line::raw(""));
        lines.push(Line::styled("  CAPSEAL", cyan));
        lines.push(Line::raw(""));
        lines.push(Line::styled("  Select a project workspace", white));
        lines.push(Line::raw(""));

        // Recent projects section
        if !self.state.recent.is_empty() {
            lines.push(Line::styled("  Recent projects:", dim));
            lines.push(Line::raw(""));

            for (i, project) in self.state.recent.iter().enumerate() {
                let is_sel = self.state.selected == i;
                let prefix = if is_sel { "  \u{25b8} " } else { "    " };
                let style = if is_sel { selected_style } else { white };

                let session_text = if project.session_count == 1 {
                    "1 session".to_string()
                } else {
                    format!("{} sessions", project.session_count)
                };

                lines.push(Line::from(vec![
                    Span::styled(format!("{}{:<40}", prefix, project.display_path), style),
                    Span::styled(
                        format!(" \u{25cf} {}", session_text),
                        if is_sel { green } else { dim },
                    ),
                ]));
            }

            lines.push(Line::raw(""));
            let rule = "\u{2500}".repeat(56);
            lines.push(Line::styled(format!("  {}", rule), dim));
            lines.push(Line::raw(""));
        }

        // Git repositories section
        if !self.state.git_repos.is_empty() {
            lines.push(Line::styled("  Git repositories:", dim));
            lines.push(Line::raw(""));

            let recent_len = self.state.recent.len();
            let scroll = self.state.scroll_offset;

            // Calculate how many git repo lines fit
            let used_lines = lines.len() + 3; // +3 for footer (blank + hint + blank)
            let available = (area.height as usize).saturating_sub(used_lines);
            let visible_end = (scroll + available).min(self.state.git_repos.len());

            if scroll > 0 {
                lines.push(Line::styled(
                    format!("    \u{2191} {} more above", scroll),
                    dim,
                ));
            }

            for git_idx in scroll..visible_end {
                let project = &self.state.git_repos[git_idx];
                let global_idx = recent_len + git_idx;
                let is_sel = self.state.selected == global_idx;
                let prefix = if is_sel { "  \u{25b8} " } else { "    " };
                let style = if is_sel {
                    selected_style
                } else {
                    Style::default().fg(Color::Cyan)
                };

                let mut spans = vec![Span::styled(
                    format!("{}{}", prefix, project.display_path),
                    style,
                )];

                // Show session count if it has .capseal
                if project.session_count > 0 {
                    let session_text = if project.session_count == 1 {
                        "1 session".to_string()
                    } else {
                        format!("{} sessions", project.session_count)
                    };
                    spans.push(Span::styled(format!("  \u{25cf} {}", session_text), green));
                }

                lines.push(Line::from(spans));
            }

            if visible_end < self.state.git_repos.len() {
                lines.push(Line::styled(
                    format!(
                        "    \u{2193} {} more below",
                        self.state.git_repos.len() - visible_end
                    ),
                    dim,
                ));
            }
        }

        if self.state.recent.is_empty() && self.state.git_repos.is_empty() {
            lines.push(Line::styled(
                "  No projects found. Use / to type a path.",
                dim,
            ));
        }

        lines.push(Line::raw(""));

        // Footer
        if self.state.path_input_mode {
            lines.push(Line::from(vec![
                Span::styled("  Path: ", cyan),
                Span::styled(&self.state.path_input, white),
                Span::styled("\u{2588}", Style::default().fg(Color::Cyan)),
            ]));
        } else {
            lines.push(Line::styled(
                "  \u{2191}\u{2193} navigate   enter select   / type a path   q quit",
                dim,
            ));
        }

        // Render top-aligned, horizontally centered
        let content_width: u16 = 72;
        let x_offset = area.width.saturating_sub(content_width) / 2;
        let render_area = Rect::new(
            area.x + x_offset,
            area.y,
            content_width.min(area.width),
            area.height,
        );

        Paragraph::new(lines)
            .alignment(Alignment::Left)
            .render(render_area, buf);
    }
}
