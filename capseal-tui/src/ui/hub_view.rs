use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};

// Block-character CAPSEAL banner (matches Python CLI's init_tui.py banner)
const BANNER: [&str; 6] = [
    " \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557} \u{2588}\u{2588}\u{2557}",
    "\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{255d}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{255d}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2550}\u{2550}\u{255d}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}",
    "\u{2588}\u{2588}\u{2551}     \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2554}\u{255d}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}  \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2551}",
    "\u{2588}\u{2588}\u{2551}     \u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2550}\u{255d} \u{255a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{255d}  \u{2588}\u{2588}\u{2554}\u{2550}\u{2550}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2551}",
    "\u{255a}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}  \u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2551}     \u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}\u{2588}\u{2588}\u{2551}  \u{2588}\u{2588}\u{2551}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2588}\u{2557}",
    " \u{255a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255d}\u{255a}\u{2550}\u{255d}  \u{255a}\u{2550}\u{255d}\u{255a}\u{2550}\u{255d}     \u{255a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255d}\u{255a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255d}\u{255a}\u{2550}\u{255d}  \u{255a}\u{2550}\u{255d}\u{255a}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{2550}\u{255d}",
];

/// Hub display mode derived from workspace state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HubMode {
    /// No .capseal/ directory — guide user to initialize
    Uninitialized,
    /// Workspace exists but no trained model
    Untrained,
    /// Fully ready — clean menu with "Start a coding session" first
    Ready,
}

struct MenuItem {
    label: &'static str,
    description: &'static str,
    value: &'static str,
    enabled: bool,
}

pub struct HubView<'a> {
    pub workspace_name: &'a str,
    pub provider: &'a str,
    pub agent_name: &'a str,
    pub model_name: &'a str,
    pub initialized: bool,
    pub model_loaded: bool,
    pub episode_count: u32,
    pub profile_count: u32,
    pub session_count: usize,
    pub selected: usize,
}

impl<'a> HubView<'a> {
    pub fn mode(&self) -> HubMode {
        if !self.initialized {
            HubMode::Uninitialized
        } else if !self.model_loaded {
            HubMode::Untrained
        } else {
            HubMode::Ready
        }
    }

    fn menu_items(&self) -> Vec<MenuItem> {
        match self.mode() {
            HubMode::Uninitialized => vec![
                MenuItem {
                    label: "Initialize workspace",
                    description: "Set up .capseal/ with provider and scan config",
                    value: "configure",
                    enabled: true,
                },
                MenuItem {
                    label: "Train risk model",
                    description: "Learn from git history to predict failures",
                    value: "train",
                    enabled: false,
                },
                MenuItem {
                    label: "Start a coding session",
                    description: "Launch agent with CapSeal protection",
                    value: "session",
                    enabled: false,
                },
            ],
            HubMode::Untrained => vec![
                MenuItem {
                    label: "Train risk model",
                    description: "Learn from git history to predict failures",
                    value: "train",
                    enabled: true,
                },
                MenuItem {
                    label: "Start a coding session",
                    description: "Launch agent with CapSeal protection",
                    value: "session",
                    enabled: true,
                },
                MenuItem {
                    label: "Scan & fix this codebase",
                    description: "Run autopilot scan and fix pipeline",
                    value: "autopilot",
                    enabled: true,
                },
                MenuItem {
                    label: "Configure",
                    description: "Providers, agents, profiles",
                    value: "configure",
                    enabled: true,
                },
                MenuItem {
                    label: "Exit",
                    description: "Quit CapSeal",
                    value: "exit",
                    enabled: true,
                },
            ],
            HubMode::Ready => vec![
                MenuItem {
                    label: "Start a coding session",
                    description: "Launch agent with CapSeal protection",
                    value: "session",
                    enabled: true,
                },
                MenuItem {
                    label: "Scan & fix this codebase",
                    description: "Run autopilot scan and fix pipeline",
                    value: "autopilot",
                    enabled: true,
                },
                MenuItem {
                    label: "View past sessions",
                    description: "Browse receipts & verification",
                    value: "sessions",
                    enabled: true,
                },
                MenuItem {
                    label: "Run a command",
                    description: "Open CapSeal shell",
                    value: "shell",
                    enabled: true,
                },
                MenuItem {
                    label: "Risk report",
                    description: "View learned risk model",
                    value: "report",
                    enabled: true,
                },
                MenuItem {
                    label: "Configure",
                    description: "Providers, agents, profiles",
                    value: "configure",
                    enabled: true,
                },
                MenuItem {
                    label: "Exit",
                    description: "Quit CapSeal",
                    value: "exit",
                    enabled: true,
                },
            ],
        }
    }

    pub fn menu_item_count(&self) -> usize {
        self.menu_items().len()
    }

    pub fn action_for_index(&self, idx: usize) -> &'static str {
        let items = self.menu_items();
        items.get(idx).map(|i| i.value).unwrap_or("exit")
    }

    pub fn is_enabled(&self, idx: usize) -> bool {
        let items = self.menu_items();
        items.get(idx).map(|i| i.enabled).unwrap_or(false)
    }
}

impl<'a> Widget for HubView<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let cyan = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let dim = Style::default().fg(Color::DarkGray);
        let white = Style::default().fg(Color::White);
        let green = Style::default().fg(Color::Green);
        let rule_char = "\u{2500}"; // ─

        let mut lines: Vec<Line> = Vec::new();
        let mode = self.mode();

        // Banner
        lines.push(Line::raw(""));
        for banner_line in &BANNER {
            lines.push(Line::styled(*banner_line, cyan));
        }
        lines.push(Line::raw(""));

        // Status card
        let rule = rule_char.repeat(48);
        lines.push(Line::styled(format!("    {}", rule), dim));

        // Workspace
        lines.push(Line::from(vec![
            Span::styled("    Workspace   ", dim),
            Span::styled(self.workspace_name, white),
        ]));

        match mode {
            HubMode::Uninitialized => {
                lines.push(Line::from(vec![
                    Span::styled("    Status      ", dim),
                    Span::styled("Not initialized", Style::default().fg(Color::Yellow)),
                ]));
            }
            _ => {
                // Provider
                if !self.provider.is_empty() {
                    let provider_display = if !self.model_name.is_empty() {
                        format!("{} ({})", self.provider, self.model_name)
                    } else {
                        self.provider.to_string()
                    };
                    lines.push(Line::from(vec![
                        Span::styled("    Provider    ", dim),
                        Span::styled(provider_display, white),
                    ]));
                }

                // Agent
                let agent_display = if self.agent_name.trim().is_empty() {
                    "not configured"
                } else {
                    self.agent_name
                };
                lines.push(Line::from(vec![
                    Span::styled("    Agent       ", dim),
                    Span::styled(agent_display, white),
                ]));

                // Model status
                let (dot, dot_color, model_text) = if self.model_loaded {
                    (
                        "\u{25cf}", // ●
                        Color::Green,
                        format!(
                            "trained  ({} profiles, {} episodes)",
                            self.profile_count, self.episode_count
                        ),
                    )
                } else {
                    ("\u{25cf}", Color::Red, "not trained yet".to_string())
                };
                lines.push(Line::from(vec![
                    Span::styled("    Model       ", dim),
                    Span::styled(format!("{} ", dot), Style::default().fg(dot_color)),
                    Span::styled(model_text, white),
                ]));

                // Sessions
                let session_text = if self.session_count > 0 {
                    format!("{} past", self.session_count)
                } else {
                    "none yet".to_string()
                };
                lines.push(Line::from(vec![
                    Span::styled("    Sessions    ", dim),
                    Span::styled(session_text, white),
                ]));
            }
        }

        lines.push(Line::styled(format!("    {}", rule), dim));
        lines.push(Line::raw(""));

        // Guided flow header (Modes A & B)
        if mode != HubMode::Ready {
            lines.push(Line::styled("    Get Started", cyan));
            lines.push(Line::raw(""));

            let done_marker = |done: bool| -> Span {
                if done {
                    Span::styled("\u{25cf} ", green) // ●
                } else {
                    Span::styled("\u{25cb} ", dim) // ○
                }
            };

            // Step 1: Initialize
            let step1_done = self.initialized;
            let step1_style = if step1_done {
                dim
            } else {
                white
            };
            let step1_suffix = if step1_done {
                Span::styled("  done", green)
            } else if mode == HubMode::Uninitialized {
                Span::styled("  \u{2190} press Enter", Style::default().fg(Color::Yellow))
            } else {
                Span::raw("")
            };
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled("Step 1  ", dim),
                done_marker(step1_done),
                Span::styled("Initialize workspace", step1_style),
                step1_suffix,
            ]));

            // Step 2: Train
            let step2_active = self.initialized && !self.model_loaded;
            let step2_style = if step2_active { white } else { dim };
            let step2_suffix = if step2_active {
                Span::styled("  \u{2190} press Enter", Style::default().fg(Color::Yellow))
            } else {
                Span::raw("")
            };
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled("Step 2  ", dim),
                done_marker(self.model_loaded),
                Span::styled("Train risk model", step2_style),
                step2_suffix,
            ]));

            // Step 3: Code
            let step3_style = dim;
            lines.push(Line::from(vec![
                Span::raw("    "),
                Span::styled("Step 3  ", dim),
                Span::styled("\u{25cb} ", dim),
                Span::styled("Start a coding session", step3_style),
            ]));

            lines.push(Line::raw(""));
        }

        // Menu items
        let items = self.menu_items();
        if mode == HubMode::Ready {
            // Full menu for ready mode
            for (i, item) in items.iter().enumerate() {
                let is_selected = i == self.selected;
                let style = if is_selected {
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(Color::White)
                };
                let prefix = if is_selected {
                    "    \u{25b8} " // ▸
                } else {
                    "      "
                };
                lines.push(Line::styled(format!("{}{}", prefix, item.label), style));
                if is_selected {
                    lines.push(Line::styled(
                        format!("        {}", item.description),
                        dim,
                    ));
                }
            }
        } else if mode == HubMode::Untrained {
            // Quick-action hints for untrained mode (below the guided steps)
            lines.push(Line::from(vec![
                Span::styled("    Quick actions:  ", dim),
                Span::styled("s", Style::default().fg(Color::Yellow)),
                Span::styled(" scan  ", dim),
                Span::styled("v", Style::default().fg(Color::Yellow)),
                Span::styled(" sessions  ", dim),
                Span::styled("c", Style::default().fg(Color::Yellow)),
                Span::styled(" configure  ", dim),
                Span::styled("q", Style::default().fg(Color::Yellow)),
                Span::styled(" quit", dim),
            ]));
        }

        lines.push(Line::raw(""));
        let hint_text = match mode {
            HubMode::Uninitialized => "    enter initialize   q quit",
            HubMode::Untrained => "    enter train   q quit",
            HubMode::Ready => "    \u{2191}\u{2193} navigate   enter select   q quit",
        };
        lines.push(Line::styled(hint_text, dim));

        // Center the content
        let content_height = lines.len() as u16;
        let content_width: u16 = 64;
        let y_offset = area.height.saturating_sub(content_height) / 2;
        let x_offset = area.width.saturating_sub(content_width) / 2;
        let centered = Rect::new(
            area.x + x_offset,
            area.y + y_offset,
            content_width.min(area.width),
            content_height.min(area.height),
        );

        Paragraph::new(lines)
            .alignment(Alignment::Left)
            .render(centered, buf);
    }
}
