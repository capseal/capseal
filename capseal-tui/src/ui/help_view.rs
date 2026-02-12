use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::Line;
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Widget};

pub struct HelpView;

impl Widget for HelpView {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let block = Block::default()
            .title(" Help ")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::Cyan));

        let inner = block.inner(area);
        block.render(area, buf);

        let header_style = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let key_style = Style::default().fg(Color::Yellow);
        let desc_style = Style::default().fg(Color::White);

        let lines = vec![
            Line::raw(""),
            Line::styled("  Keybindings", header_style),
            Line::raw(""),
            Line::raw(""),
            Line::styled("  Global", header_style),
            Line::raw(""),
            help_line("  Ctrl+Q", "Quit application", key_style, desc_style),
            help_line("  F1", "Toggle this help", key_style, desc_style),
            help_line("  F4", "Toggle voice operator", key_style, desc_style),
            help_line("  F5", "Verify latest receipt", key_style, desc_style),
            help_line(
                "  Ctrl+H",
                "Toggle focus (panel/terminal)",
                key_style,
                desc_style,
            ),
            Line::raw(""),
            Line::styled("  Hub (no session)", header_style),
            Line::raw(""),
            help_line("  Up/Down, j/k", "Navigate menu", key_style, desc_style),
            help_line("  Enter", "Select item", key_style, desc_style),
            help_line("  s", "Quick: autopilot scan", key_style, desc_style),
            help_line("  v", "Quick: view sessions", key_style, desc_style),
            help_line("  c", "Quick: configure", key_style, desc_style),
            help_line("  q, Esc", "Quit", key_style, desc_style),
            Line::raw(""),
            Line::styled("  Session (3-zone layout)", header_style),
            Line::raw(""),
            help_line(
                "  Ctrl+H",
                "Switch focus panel/terminal",
                key_style,
                desc_style,
            ),
            help_line("  (panel) s", "Scan & fix", key_style, desc_style),
            help_line("  (panel) v", "Sessions overlay", key_style, desc_style),
            help_line(
                "  (panel) r",
                "Risk profiles overlay",
                key_style,
                desc_style,
            ),
            help_line("  (panel) c", "Configure", key_style, desc_style),
            help_line("  (panel) q", "Quit", key_style, desc_style),
            help_line(
                "  (terminal)",
                "All keys forwarded to shell",
                key_style,
                desc_style,
            ),
            help_line("  F2", "Sessions overlay", key_style, desc_style),
            help_line("  F3", "Risk profiles overlay", key_style, desc_style),
            help_line("  F4", "Toggle voice operator", key_style, desc_style),
            help_line("  F5", "Verify overlay", key_style, desc_style),
            help_line("  F10", "Quit", key_style, desc_style),
            Line::raw(""),
            Line::styled("  Mouse", header_style),
            Line::raw(""),
            help_line(
                "  Click panel",
                "Focus control panel",
                key_style,
                desc_style,
            ),
            help_line("  Click terminal", "Focus terminal", key_style, desc_style),
            help_line(
                "  Scroll chain",
                "Scroll action history",
                key_style,
                desc_style,
            ),
            Line::raw(""),
            Line::styled("  Agent Picker", header_style),
            Line::raw(""),
            help_line("  Up/Down", "Select agent", key_style, desc_style),
            help_line("  Enter", "Launch selected agent", key_style, desc_style),
            help_line("  Esc", "Cancel", key_style, desc_style),
            Line::raw(""),
            Line::styled("  Press Esc to close", Style::default().fg(Color::DarkGray)),
        ];

        Paragraph::new(lines).render(inner, buf);
    }
}

fn help_line<'a>(key: &'a str, desc: &'a str, key_style: Style, desc_style: Style) -> Line<'a> {
    Line::from(vec![
        ratatui::text::Span::styled(format!("{:<24}", key), key_style),
        ratatui::text::Span::styled(desc, desc_style),
    ])
}
