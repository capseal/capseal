use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use ratatui::widgets::Widget;
use std::io::{Read, Stdout};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::agent::detect::AgentDetector;
use crate::agent::inject;
use crate::agent::output_watcher::AgentOutputWatcher;
use crate::capseal::watcher::EventWatcher;
use crate::capseal::CapSealState;
use crate::config::CapSealConfig;
use crate::terminal::pty::PtyHandle;
use crate::terminal::renderer::TerminalWidget;
use crate::ui::session_monitor::SessionMonitor;
use crate::ui::agent_picker::{AgentPicker, AgentPickerResult};
use crate::ui::control_panel::ControlPanel;
use crate::ui::help_view::HelpView;
use crate::ui::hub_view::HubView;
use crate::ui::layout::{HubLayout, SessionLayout};
use crate::ui::workspace_picker::WorkspacePicker;
use crate::ui::risk_map_view::RiskMapView;
use crate::ui::session_complete_view::{SessionCompleteInfo, SessionCompleteView};
use crate::ui::sessions_view::SessionsView;
use crate::ui::status_bar::StatusBar;
use crate::ui::title_bar::TitleBar;
use crate::workspace::PickerState;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AppMode {
    WorkspacePicker,
    Hub,
    Sessions,
    RiskMap,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Focus {
    ControlPanel,
    Terminal,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Action {
    Continue,
    Quit,
}

pub struct App {
    pub config: CapSealConfig,
    pub mode: AppMode,
    pub hub_selected: usize,
    pub start_in_shell: bool,
    pub focus: Focus,

    // PTY state
    pty: Option<PtyHandle>,
    vt_parser: Option<vt100::Parser>,
    pty_rx: Option<mpsc::Receiver<Vec<u8>>>,
    pty_alive: bool,
    last_pty_cols: u16,
    last_pty_rows: u16,

    // CapSeal state + watcher
    capseal_state: CapSealState,
    watcher: EventWatcher,

    // Agent detection
    agent_detector: AgentDetector,
    output_watcher: AgentOutputWatcher,

    // Animation / display state
    tick_count: u64,
    shell_name: String,

    // Overlay state
    show_help: bool,
    show_session_complete: bool,
    session_complete_info: Option<SessionCompleteInfo>,
    session_start_time: Option<Instant>,
    session_selected: usize,

    // Agent picker
    show_agent_picker: bool,
    agent_picker: Option<AgentPicker>,

    // Pending PTY write (for agent command after spawn)
    pending_pty_write: Option<String>,
    pending_write_delay: Option<u32>,

    // Action chain scrolling
    chain_scroll_offset: usize,

    // Layout rects for mouse hit testing
    last_terminal_rect: Option<ratatui::layout::Rect>,
    last_chain_rect: Option<ratatui::layout::Rect>,
    last_control_rect: Option<ratatui::layout::Rect>,

    // Workspace picker
    picker: Option<PickerState>,
    last_height: u16,
    last_width: u16,
}

impl App {
    pub fn new(config: CapSealConfig, start_in_shell: bool) -> Self {
        let capseal_dir = config.capseal_dir();
        eprintln!("[TUI] capseal_dir = {:?}", capseal_dir);
        let capseal_state = CapSealState::new(&config.workspace);
        let watcher = EventWatcher::new(&capseal_dir);

        // Only skip the workspace picker when the directory has .git
        let has_git = config.workspace.join(".git").exists();
        let (initial_mode, picker) = if has_git {
            (AppMode::Hub, None)
        } else {
            (AppMode::WorkspacePicker, Some(PickerState::new()))
        };

        Self {
            config,
            mode: initial_mode,
            hub_selected: 0,
            start_in_shell,
            focus: Focus::ControlPanel,
            pty: None,
            vt_parser: None,
            pty_rx: None,
            pty_alive: false,
            last_pty_cols: 0,
            last_pty_rows: 0,
            capseal_state,
            watcher,
            agent_detector: AgentDetector::new(),
            output_watcher: AgentOutputWatcher::new(&capseal_dir),
            tick_count: 0,
            shell_name: {
                let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string());
                std::path::Path::new(&shell)
                    .file_name()
                    .map(|n| n.to_string_lossy().to_string())
                    .unwrap_or_else(|| "shell".to_string())
            },
            show_help: false,
            show_session_complete: false,
            session_complete_info: None,
            session_start_time: None,
            session_selected: 0,
            show_agent_picker: false,
            agent_picker: None,
            pending_pty_write: None,
            pending_write_delay: None,
            chain_scroll_offset: 0,
            last_terminal_rect: None,
            last_chain_rect: None,
            last_control_rect: None,
            picker,
            last_height: 40,
            last_width: 120,
        }
    }

    fn pty_active(&self) -> bool {
        self.pty.is_some() && self.pty_alive
    }

    pub fn run(&mut self, terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
        // Initial data load
        self.watcher.initial_load(&mut self.capseal_state);

        // If --shell flag, skip hub and go straight to terminal (only if valid project)
        if self.start_in_shell && self.mode != AppMode::WorkspacePicker {
            let area = terminal.get_frame().area();
            let layout = SessionLayout::compute(area);
            let inner_w = layout.terminal.width.saturating_sub(2);
            let inner_h = layout.terminal.height.saturating_sub(2);
            self.spawn_terminal(inner_w, inner_h)?;
        }

        let tick_rate = Duration::from_millis(16);
        let mut last_tick = Instant::now();

        loop {
            terminal.draw(|frame| self.draw(frame))?;

            // Read PTY output
            self.drain_pty_output();

            // Poll capseal file watcher
            self.watcher.poll(&mut self.capseal_state);

            // Inject bytes from operator (pty_input.txt → PTY stdin)
            if let Some(data) = self.capseal_state.pending_pty_injection.take() {
                if let Some(pty) = &mut self.pty {
                    let _ = pty.write(&data);
                }
            }

            // Check if agent is waiting for input
            self.output_watcher.check_idle();

            // Drain pending PTY writes (delayed agent command)
            if let Some(ref mut delay) = self.pending_write_delay {
                if *delay == 0 {
                    if let Some(cmd) = self.pending_pty_write.take() {
                        if let Some(pty) = &mut self.pty {
                            let _ = pty.write(cmd.as_bytes());
                        }
                    }
                    self.pending_write_delay = None;
                } else {
                    *delay -= 1;
                }
            }

            // Check if PTY child exited
            if self.pty.is_some() && !self.pty_alive {
                // Build session complete info before cleanup
                let duration_secs = self
                    .session_start_time
                    .map(|t| t.elapsed().as_secs())
                    .unwrap_or(0);

                let has_actions = self.capseal_state.action_count > 0;

                if has_actions {
                    // Find latest .cap file from runs dir
                    let cap_file = self
                        .capseal_state
                        .sessions
                        .first() // sessions sorted newest-first
                        .map(|s| {
                            self.config
                                .capseal_dir()
                                .join("runs")
                                .join(format!("{}.cap", s.name))
                                .display()
                                .to_string()
                        })
                        .filter(|p| std::path::Path::new(p).exists());

                    self.session_complete_info = Some(SessionCompleteInfo {
                        action_count: self.capseal_state.action_count,
                        denied_count: self.capseal_state.denied_count,
                        duration_secs,
                        cap_file,
                        chain_intact: self.capseal_state.chain_intact,
                    });
                    self.show_session_complete = true;
                } else {
                    self.cleanup_terminal();
                    self.focus = Focus::ControlPanel;
                    self.session_start_time = None;
                }
            }

            let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                match event::read()? {
                    Event::Key(key) => {
                        if self.handle_key(key)? == Action::Quit {
                            break;
                        }
                    }
                    Event::Mouse(mouse) => {
                        self.handle_mouse(mouse)?;
                    }
                    Event::Resize(_w, _h) => {}
                    _ => {}
                }
            }

            if last_tick.elapsed() >= tick_rate {
                last_tick = Instant::now();
                self.tick_count = self.tick_count.wrapping_add(1);
            }
        }

        Ok(())
    }

    fn spawn_terminal(&mut self, cols: u16, rows: u16) -> Result<()> {
        // Clear stale events from previous session
        self.capseal_state.action_chain.clear();
        self.capseal_state.action_count = 0;
        self.capseal_state.denied_count = 0;
        self.capseal_state.session_active = true;
        self.capseal_state.session_start = Some(std::time::Instant::now());
        self.watcher.reset_events_position();

        // Auto-launch operator daemon if not already running
        if !self.capseal_state.operator_online {
            let op_config = self.config.workspace.join(".capseal").join("operator.json");
            if op_config.exists() {
                let ws = self.config.workspace.display().to_string();
                if let Ok(_child) = std::process::Command::new("capseal")
                    .args(["operator", &ws, "--bg"])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()
                {
                    eprintln!("[tui] Operator daemon auto-launched");
                }
            }
        }

        let mut pty = PtyHandle::spawn(&self.config.workspace, cols, rows)?;
        let reader = pty.take_reader();

        let (tx, rx) = mpsc::channel::<Vec<u8>>();
        std::thread::spawn(move || {
            pty_reader_thread(reader, tx);
        });

        self.vt_parser = Some(vt100::Parser::new(rows, cols, 10000));
        self.pty = Some(pty);
        self.pty_rx = Some(rx);
        self.pty_alive = true;
        self.last_pty_cols = cols;
        self.last_pty_rows = rows;
        self.session_start_time = Some(Instant::now());
        self.chain_scroll_offset = 0;
        self.focus = Focus::Terminal;
        Ok(())
    }

    fn drain_pty_output(&mut self) {
        if let Some(rx) = &self.pty_rx {
            while let Ok(data) = rx.try_recv() {
                if data.is_empty() {
                    self.pty_alive = false;
                    return;
                }
                if let Some(parser) = &mut self.vt_parser {
                    parser.process(&data);
                }
                // Feed output to agent waiting detector
                self.output_watcher.feed(&data);
            }
        }
        if let Some(pty) = &mut self.pty {
            if !pty.is_alive() {
                self.pty_alive = false;
            }
        }
    }

    fn cleanup_terminal(&mut self) {
        self.pty = None;
        self.vt_parser = None;
        self.pty_rx = None;
        self.pty_alive = false;
        self.pending_pty_write = None;
        self.pending_write_delay = None;
    }

    fn draw(&mut self, frame: &mut ratatui::Frame) {
        let area = frame.area();
        self.last_height = area.height;
        self.last_width = area.width;

        // Workspace picker: full-screen, no panels
        if self.mode == AppMode::WorkspacePicker {
            if let Some(ref picker) = self.picker {
                frame.render_widget(
                    WorkspacePicker { state: picker },
                    area,
                );
            }
            return;
        }

        let ws_name = self
            .config
            .workspace
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| self.config.workspace.display().to_string());

        let process_name = if self.pty_active() {
            Some(self.shell_name.as_str())
        } else {
            None
        };

        let pty_visible = self.pty_active() || self.show_session_complete;

        if pty_visible {
            // ── 3-zone session layout ──
            let layout = SessionLayout::compute(area);

            // Store rects for mouse hit testing
            self.last_control_rect = Some(layout.control_panel);
            self.last_chain_rect = Some(layout.action_chain);
            self.last_terminal_rect = Some(layout.terminal);

            // Title bar
            frame.render_widget(
                TitleBar {
                    workspace_name: &ws_name,
                    process_name,
                    model_trained: self.capseal_state.model_loaded,
                    session_count: self.capseal_state.sessions.len(),
                    session_active: self.capseal_state.session_active,
                },
                layout.title_bar,
            );

            // Left: compact control panel
            frame.render_widget(
                ControlPanel {
                    workspace_name: &ws_name,
                    model_loaded: self.capseal_state.model_loaded,
                    session_count: self.capseal_state.sessions.len(),
                    focused: self.focus == Focus::ControlPanel,
                    pty_active: self.pty_active(),
                    action_count: self.capseal_state.action_count,
                    denied_count: self.capseal_state.denied_count,
                    chain_intact: self.capseal_state.chain_intact,
                    tick: self.tick_count,
                    training_in_progress: self.capseal_state.training_in_progress,
                    training_round: self.capseal_state.training_round,
                    training_total_rounds: self.capseal_state.training_total_rounds,
                },
                layout.control_panel,
            );

            // Top-right: session monitor
            frame.render_widget(
                SessionMonitor {
                    state: &self.capseal_state,
                    scroll_offset: self.chain_scroll_offset,
                },
                layout.action_chain,
            );

            // Bottom-right: terminal with border
            if layout.terminal.width > 0 && layout.terminal.height > 0 {
                let term_border_color = if self.focus == Focus::Terminal {
                    ratatui::style::Color::Cyan
                } else {
                    ratatui::style::Color::DarkGray
                };

                let term_block = ratatui::widgets::Block::default()
                    .title(" Terminal ")
                    .title_style(ratatui::style::Style::default().fg(ratatui::style::Color::White))
                    .borders(ratatui::widgets::Borders::ALL)
                    .border_type(ratatui::widgets::BorderType::Rounded)
                    .border_style(ratatui::style::Style::default().fg(term_border_color));

                let term_inner = term_block.inner(layout.terminal);
                term_block.render(layout.terminal, frame.buffer_mut());

                // Fill terminal area with black background
                for y in term_inner.y..term_inner.y + term_inner.height {
                    for x in term_inner.x..term_inner.x + term_inner.width {
                        if let Some(cell) = frame.buffer_mut().cell_mut((x, y)) {
                            cell.set_style(ratatui::style::Style::default().bg(ratatui::style::Color::Black));
                            cell.set_symbol(" ");
                        }
                    }
                }

                // Resize PTY if terminal inner area changed
                let new_cols = term_inner.width;
                let new_rows = term_inner.height;
                if (new_cols != self.last_pty_cols || new_rows != self.last_pty_rows)
                    && new_cols > 0
                    && new_rows > 0
                {
                    if let Some(pty) = &self.pty {
                        let _ = pty.resize(new_cols, new_rows);
                    }
                    if let Some(parser) = &mut self.vt_parser {
                        parser.set_size(new_rows, new_cols);
                    }
                    self.last_pty_cols = new_cols;
                    self.last_pty_rows = new_rows;
                }

                if let Some(parser) = &self.vt_parser {
                    frame.render_widget(
                        TerminalWidget {
                            screen: parser.screen(),
                        },
                        term_inner,
                    );
                }
            }

            // Status bar
            let hints = self.status_hints();
            frame.render_widget(StatusBar { hints: &hints }, layout.status_bar);
        } else {
            // ── Full-screen hub view (no PTY) ──
            let layout = HubLayout::compute(area);

            // Title bar
            frame.render_widget(
                TitleBar {
                    workspace_name: &ws_name,
                    process_name: None,
                    model_trained: self.capseal_state.model_loaded,
                    session_count: self.capseal_state.sessions.len(),
                    session_active: self.capseal_state.session_active,
                },
                layout.title_bar,
            );

            // Hub view
            let hub = HubView {
                workspace_name: &ws_name,
                provider: &self.config.provider,
                model_name: &self.config.model,
                initialized: self.config.initialized,
                model_loaded: self.capseal_state.model_loaded,
                episode_count: self.capseal_state.episode_count,
                profile_count: self.capseal_state.profile_count,
                session_count: self.capseal_state.sessions.len(),
                selected: self.hub_selected,
            };
            frame.render_widget(hub, layout.hub_area);

            // Status bar
            let hints = self.status_hints();
            frame.render_widget(StatusBar { hints: &hints }, layout.status_bar);
        }

        // ── Overlays (rendered on top of everything) ──

        // Sessions overlay
        if self.mode == AppMode::Sessions {
            let overlay_w = 72.min(area.width.saturating_sub(4));
            let overlay_h = 28.min(area.height.saturating_sub(4));
            let overlay_area = ratatui::layout::Rect::new(
                area.x + (area.width.saturating_sub(overlay_w)) / 2,
                area.y + (area.height.saturating_sub(overlay_h)) / 2,
                overlay_w,
                overlay_h,
            );
            ratatui::widgets::Clear.render(overlay_area, frame.buffer_mut());
            frame.render_widget(
                SessionsView {
                    sessions: &self.capseal_state.sessions,
                    selected: self.session_selected,
                },
                overlay_area,
            );
        }

        // Risk Map overlay
        if self.mode == AppMode::RiskMap {
            let overlay_w = 72.min(area.width.saturating_sub(4));
            let overlay_h = 28.min(area.height.saturating_sub(4));
            let overlay_area = ratatui::layout::Rect::new(
                area.x + (area.width.saturating_sub(overlay_w)) / 2,
                area.y + (area.height.saturating_sub(overlay_h)) / 2,
                overlay_w,
                overlay_h,
            );
            ratatui::widgets::Clear.render(overlay_area, frame.buffer_mut());
            frame.render_widget(
                RiskMapView {
                    risk_scores: &self.capseal_state.risk_scores,
                    model_loaded: self.capseal_state.model_loaded,
                    episode_count: self.capseal_state.episode_count,
                    profile_count: self.capseal_state.profile_count,
                },
                overlay_area,
            );
        }

        // Session Complete overlay
        if self.show_session_complete {
            if let Some(ref info) = self.session_complete_info {
                let overlay_area = SessionCompleteView::overlay_rect(area);
                ratatui::widgets::Clear.render(overlay_area, frame.buffer_mut());
                frame.render_widget(SessionCompleteView { info }, overlay_area);
            }
        }

        // Agent Picker overlay
        if self.show_agent_picker {
            if let Some(ref picker) = self.agent_picker {
                frame.render_widget(picker, area);
            }
        }

        // Help overlay (always on top)
        if self.show_help {
            let help_width = 50.min(area.width.saturating_sub(4));
            let help_height = 30.min(area.height.saturating_sub(4));
            let help_area = ratatui::layout::Rect::new(
                area.x + (area.width.saturating_sub(help_width)) / 2,
                area.y + (area.height.saturating_sub(help_height)) / 2,
                help_width,
                help_height,
            );
            ratatui::widgets::Clear.render(help_area, frame.buffer_mut());
            frame.render_widget(HelpView, help_area);
        }
    }

    fn status_hints(&self) -> Vec<(&str, &str)> {
        if self.pty_active() {
            match self.focus {
                Focus::Terminal => vec![
                    ("Ctrl+H", "Panel"),
                    ("F1", "Help"),
                    ("F2", "Sessions"),
                    ("F3", "Risk Map"),
                    ("F10", "Quit"),
                ],
                Focus::ControlPanel => vec![
                    ("Ctrl+H", "Terminal"),
                    ("s/v/r/c", "Actions"),
                    ("F1", "Help"),
                    ("F10", "Quit"),
                ],
            }
        } else {
            vec![
                ("\u{2191}\u{2193}", "Navigate"),
                ("Enter", "Select"),
                ("q", "Quit"),
            ]
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> Result<Action> {
        // Ctrl+Q always quits
        if key.code == KeyCode::Char('q') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Ok(Action::Quit);
        }

        // F1 toggles help in any mode
        if key.code == KeyCode::F(1) {
            self.show_help = !self.show_help;
            return Ok(Action::Continue);
        }

        // If help is showing, Esc closes it
        if self.show_help {
            if key.code == KeyCode::Esc {
                self.show_help = false;
            }
            return Ok(Action::Continue);
        }

        // Agent picker overlay
        if self.show_agent_picker {
            if let Some(ref mut picker) = self.agent_picker {
                match picker.handle_key(key) {
                    AgentPickerResult::Selected(command) => {
                        self.show_agent_picker = false;
                        self.agent_picker = None;
                        self.launch_with_agent(command)?;
                    }
                    AgentPickerResult::Cancelled => {
                        self.show_agent_picker = false;
                        self.agent_picker = None;
                    }
                    AgentPickerResult::Continue => {}
                }
            }
            return Ok(Action::Continue);
        }

        // Session Complete overlay — dismiss then cleanup
        if self.show_session_complete {
            match key.code {
                KeyCode::Enter | KeyCode::Esc => {
                    self.show_session_complete = false;
                    self.session_complete_info = None;
                    self.cleanup_terminal();
                    self.focus = Focus::ControlPanel;
                    self.session_start_time = None;
                }
                KeyCode::Char('v') => {
                    let cap_file = self
                        .session_complete_info
                        .as_ref()
                        .and_then(|i| i.cap_file.clone());
                    self.show_session_complete = false;
                    self.session_complete_info = None;
                    self.cleanup_terminal();
                    self.focus = Focus::ControlPanel;
                    self.session_start_time = None;
                    if let Some(path) = cap_file {
                        let cmd = format!("capseal verify {}", path);
                        let (cols, rows) = self.terminal_spawn_size();
                        self.spawn_terminal_with_cmd(cols, rows, &cmd)?;
                    }
                }
                KeyCode::Char('q') => {
                    return Ok(Action::Quit);
                }
                _ => {}
            }
            return Ok(Action::Continue);
        }

        // Overlay modes: Sessions/RiskMap
        match self.mode {
            AppMode::Sessions => return self.handle_sessions_key(key),
            AppMode::RiskMap => {
                if key.code == KeyCode::Esc {
                    self.mode = AppMode::Hub;
                    return Ok(Action::Continue);
                } else if key.code == KeyCode::Char('q') && !self.pty_active() {
                    return Ok(Action::Quit);
                } else if key.code == KeyCode::Char('q') {
                    self.mode = AppMode::Hub;
                    return Ok(Action::Continue);
                }
                return Ok(Action::Continue);
            }
            _ => {}
        }

        // Workspace picker
        if self.mode == AppMode::WorkspacePicker {
            return self.handle_picker_key(key, self.last_height);
        }

        // Ctrl+H toggles focus when PTY is active
        if self.pty_active() {
            if key.code == KeyCode::Char('h') && key.modifiers.contains(KeyModifiers::CONTROL) {
                self.focus = match self.focus {
                    Focus::ControlPanel => Focus::Terminal,
                    Focus::Terminal => Focus::ControlPanel,
                };
                return Ok(Action::Continue);
            }
        }

        // Hub mode — dispatch based on PTY state and focus
        if !self.pty_active() {
            return self.handle_hub_key(key);
        }

        // PTY active — dispatch based on focus
        match self.focus {
            Focus::ControlPanel => self.handle_control_panel_key(key),
            Focus::Terminal => self.handle_terminal_key(key),
        }
    }

    fn handle_sessions_key(&mut self, key: KeyEvent) -> Result<Action> {
        match key.code {
            KeyCode::Esc => {
                self.mode = AppMode::Hub;
                Ok(Action::Continue)
            }
            KeyCode::Char('q') => {
                if self.pty_active() {
                    self.mode = AppMode::Hub;
                    Ok(Action::Continue)
                } else {
                    Ok(Action::Quit)
                }
            }
            KeyCode::Up | KeyCode::Char('k') => {
                if self.session_selected > 0 {
                    self.session_selected -= 1;
                }
                Ok(Action::Continue)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                let max = self.capseal_state.sessions.len().saturating_sub(1);
                if self.session_selected < max {
                    self.session_selected += 1;
                }
                Ok(Action::Continue)
            }
            _ => Ok(Action::Continue),
        }
    }

    fn hub_view(&self) -> HubView<'_> {
        let ws_name = self.config.workspace
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| self.config.workspace.display().to_string());
        HubView {
            workspace_name: Box::leak(ws_name.into_boxed_str()),
            provider: Box::leak(self.config.provider.clone().into_boxed_str()),
            model_name: Box::leak(self.config.model.clone().into_boxed_str()),
            initialized: self.capseal_state.initialized,
            model_loaded: self.capseal_state.model_loaded,
            episode_count: self.capseal_state.episode_count,
            profile_count: self.capseal_state.profile_count,
            session_count: self.capseal_state.sessions.len(),
            selected: self.hub_selected,
        }
    }

    fn hub_menu_item_count(&self) -> usize {
        self.hub_view().menu_item_count()
    }

    fn hub_action_for_index(&self, idx: usize) -> &'static str {
        self.hub_view().action_for_index(idx)
    }

    fn hub_is_enabled(&self, idx: usize) -> bool {
        self.hub_view().is_enabled(idx)
    }

    fn spawn_terminal_with_cmd(&mut self, cols: u16, rows: u16, cmd: &str) -> Result<()> {
        self.spawn_terminal(cols, rows)?;
        if let Some(pty) = &mut self.pty {
            let _ = pty.write(cmd.as_bytes());
            let _ = pty.write(b"\n");
        }
        Ok(())
    }

    /// Launch PTY and optionally queue an agent command to be written after a short delay
    fn launch_with_agent(&mut self, agent_command: String) -> Result<()> {
        let (cols, rows) = self.terminal_spawn_size();
        self.spawn_terminal(cols, rows)?;

        if !agent_command.is_empty() {
            self.pending_pty_write = Some(format!("{}\n", agent_command));
            self.pending_write_delay = Some(2); // wait 2 ticks (~32ms)
        }

        Ok(())
    }

    fn handle_picker_key(&mut self, key: KeyEvent, viewport_h: u16) -> Result<Action> {
        let picker = match self.picker.as_mut() {
            Some(p) => p,
            None => return Ok(Action::Continue),
        };

        // Path input mode
        if picker.path_input_mode {
            match key.code {
                KeyCode::Esc => {
                    picker.path_input_mode = false;
                    picker.path_input.clear();
                }
                KeyCode::Enter => {
                    let path = picker.path_input.clone();
                    picker.path_input_mode = false;
                    picker.path_input.clear();
                    let expanded = if path.starts_with('~') {
                        if let Some(home) = dirs::home_dir() {
                            home.join(path[1..].trim_start_matches('/'))
                        } else {
                            PathBuf::from(&path)
                        }
                    } else {
                        PathBuf::from(&path)
                    };
                    if expanded.is_dir() {
                        return self.switch_to_workspace(expanded);
                    }
                }
                KeyCode::Backspace => {
                    picker.path_input.pop();
                }
                KeyCode::Char(c) => {
                    picker.path_input.push(c);
                }
                _ => {}
            }
            return Ok(Action::Continue);
        }

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => Ok(Action::Quit),
            KeyCode::Up | KeyCode::Char('k') => {
                picker.move_up();
                picker.adjust_scroll(viewport_h);
                Ok(Action::Continue)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                picker.move_down();
                picker.adjust_scroll(viewport_h);
                Ok(Action::Continue)
            }
            KeyCode::Enter => {
                if let Some(path) = picker.selected_path() {
                    let path = path.to_path_buf();
                    return self.switch_to_workspace(path);
                }
                Ok(Action::Continue)
            }
            KeyCode::Char('/') => {
                picker.path_input_mode = true;
                picker.path_input.clear();
                Ok(Action::Continue)
            }
            _ => Ok(Action::Continue),
        }
    }

    fn switch_to_workspace(&mut self, workspace: PathBuf) -> Result<Action> {
        if let Ok(new_config) = crate::config::CapSealConfig::load(&workspace) {
            self.config = new_config;
            self.capseal_state = crate::capseal::CapSealState::new(&workspace);
            let capseal_dir = self.config.capseal_dir();
            self.watcher = crate::capseal::watcher::EventWatcher::new(&capseal_dir);
            self.watcher.initial_load(&mut self.capseal_state);
            self.mode = AppMode::Hub;
            self.hub_selected = 0;
            self.picker = None;
        }
        Ok(Action::Continue)
    }

    fn handle_hub_key(&mut self, key: KeyEvent) -> Result<Action> {
        use crate::ui::hub_view::HubMode;

        let mode = self.hub_view().mode();
        let max_idx = self.hub_menu_item_count().saturating_sub(1);

        // Quick-action hotkeys (work in Untrained and Ready modes)
        if mode != HubMode::Uninitialized {
            match key.code {
                KeyCode::Char('s') => {
                    let (cols, rows) = self.terminal_spawn_size();
                    self.spawn_terminal_with_cmd(cols, rows, "capseal autopilot .")?;
                    return Ok(Action::Continue);
                }
                KeyCode::Char('v') => {
                    self.mode = AppMode::Sessions;
                    return Ok(Action::Continue);
                }
                KeyCode::Char('c') => {
                    let (cols, rows) = self.terminal_spawn_size();
                    self.spawn_terminal_with_cmd(cols, rows, "capseal init")?;
                    return Ok(Action::Continue);
                }
                _ => {}
            }
        }

        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => Ok(Action::Quit),
            KeyCode::Up | KeyCode::Char('k') => {
                if mode == HubMode::Ready && self.hub_selected > 0 {
                    self.hub_selected -= 1;
                }
                Ok(Action::Continue)
            }
            KeyCode::Down | KeyCode::Char('j') => {
                if mode == HubMode::Ready && self.hub_selected < max_idx {
                    self.hub_selected += 1;
                }
                Ok(Action::Continue)
            }
            KeyCode::Enter => {
                match mode {
                    HubMode::Uninitialized => {
                        let (cols, rows) = self.terminal_spawn_size();
                        self.spawn_terminal_with_cmd(cols, rows, "capseal init")?;
                    }
                    HubMode::Untrained => {
                        let action = self.hub_action_for_index(self.hub_selected);
                        if action == "exit" {
                            return Ok(Action::Quit);
                        }
                        if !self.hub_is_enabled(self.hub_selected) {
                            let (cols, rows) = self.terminal_spawn_size();
                            self.spawn_terminal_with_cmd(cols, rows, "capseal learn . --from-git")?;
                        } else {
                            self.dispatch_hub_action(action)?;
                        }
                    }
                    HubMode::Ready => {
                        let action = self.hub_action_for_index(self.hub_selected);
                        if action == "exit" {
                            return Ok(Action::Quit);
                        }
                        self.dispatch_hub_action(action)?;
                    }
                }
                Ok(Action::Continue)
            }
            _ => Ok(Action::Continue),
        }
    }

    /// Compute terminal inner dimensions for spawning (accounts for 3-zone layout + border)
    fn terminal_spawn_size(&self) -> (u16, u16) {
        let area = ratatui::layout::Rect::new(0, 0, self.last_width, self.last_height);
        let layout = SessionLayout::compute(area);
        let cols = layout.terminal.width.saturating_sub(2).max(40);
        let rows = layout.terminal.height.saturating_sub(2).max(10);
        (cols, rows)
    }

    fn dispatch_hub_action(&mut self, action: &str) -> Result<()> {
        match action {
            "session" => {
                // Show agent picker overlay instead of spawning immediately
                self.agent_picker = Some(AgentPicker::new(&self.config.default_agent, &self.config.provider));
                self.show_agent_picker = true;
            }
            "autopilot" => {
                let (cols, rows) = self.terminal_spawn_size();
                self.spawn_terminal_with_cmd(cols, rows, "capseal autopilot .")?;
            }
            "train" => {
                let (cols, rows) = self.terminal_spawn_size();
                self.spawn_terminal_with_cmd(cols, rows, "capseal learn . --from-git")?;
            }
            "sessions" => {
                self.mode = AppMode::Sessions;
            }
            "report" => {
                self.mode = AppMode::RiskMap;
            }
            "configure" => {
                let (cols, rows) = self.terminal_spawn_size();
                self.spawn_terminal_with_cmd(cols, rows, "capseal init")?;
            }
            "exit" => {
                // Caller should handle quit
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_control_panel_key(&mut self, key: KeyEvent) -> Result<Action> {
        match key.code {
            // Single-letter shortcuts
            KeyCode::Char('s') => {
                // Scan & fix — write command to existing PTY
                if let Some(pty) = &mut self.pty {
                    let _ = pty.write(b"capseal autopilot .\n");
                }
                self.focus = Focus::Terminal;
                Ok(Action::Continue)
            }
            KeyCode::Char('v') => {
                self.mode = AppMode::Sessions;
                Ok(Action::Continue)
            }
            KeyCode::Char('r') => {
                self.mode = AppMode::RiskMap;
                Ok(Action::Continue)
            }
            KeyCode::Char('c') => {
                // Configure — write command to existing PTY
                if let Some(pty) = &mut self.pty {
                    let _ = pty.write(b"capseal init\n");
                }
                self.focus = Focus::Terminal;
                Ok(Action::Continue)
            }
            KeyCode::Char('q') => Ok(Action::Quit),
            KeyCode::F(2) => {
                self.mode = AppMode::Sessions;
                Ok(Action::Continue)
            }
            KeyCode::F(3) => {
                self.mode = AppMode::RiskMap;
                Ok(Action::Continue)
            }
            KeyCode::F(10) => Ok(Action::Quit),
            _ => Ok(Action::Continue),
        }
    }

    fn handle_terminal_key(&mut self, key: KeyEvent) -> Result<Action> {
        match key.code {
            KeyCode::F(2) => {
                self.mode = AppMode::Sessions;
                Ok(Action::Continue)
            }
            KeyCode::F(3) => {
                self.mode = AppMode::RiskMap;
                Ok(Action::Continue)
            }
            KeyCode::F(10) => Ok(Action::Quit),
            _ => {
                let bytes = key_to_bytes(key);
                if !bytes.is_empty() {
                    // Feed bytes to agent detector
                    for &b in &bytes {
                        if let Some(_agent_name) = self.agent_detector.feed(b) {
                            let _ = inject::inject_mcp(&self.config.workspace);
                        }
                    }

                    // Forward to PTY
                    if let Some(pty) = &mut self.pty {
                        let _ = pty.write(&bytes);
                    }
                }
                Ok(Action::Continue)
            }
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) -> Result<Action> {
        match mouse.kind {
            MouseEventKind::Down(MouseButton::Left) => {
                let col = mouse.column;
                let row = mouse.row;

                // Click on terminal area → focus terminal
                if let Some(rect) = self.last_terminal_rect {
                    if col >= rect.x && col < rect.x + rect.width
                        && row >= rect.y && row < rect.y + rect.height
                    {
                        self.focus = Focus::Terminal;
                        return Ok(Action::Continue);
                    }
                }

                // Click on control panel → focus control panel
                if let Some(rect) = self.last_control_rect {
                    if col >= rect.x && col < rect.x + rect.width
                        && row >= rect.y && row < rect.y + rect.height
                    {
                        self.focus = Focus::ControlPanel;
                        return Ok(Action::Continue);
                    }
                }

                // Click on action chain → no focus change, just acknowledge
                Ok(Action::Continue)
            }
            MouseEventKind::ScrollUp => {
                // Scroll in action chain → scroll history up (older)
                if let Some(rect) = self.last_chain_rect {
                    if mouse.column >= rect.x && mouse.column < rect.x + rect.width
                        && mouse.row >= rect.y && mouse.row < rect.y + rect.height
                    {
                        self.chain_scroll_offset = self.chain_scroll_offset.saturating_add(3);
                        return Ok(Action::Continue);
                    }
                }

                // Scroll in terminal → forward to PTY (check rect)
                if self.pty_active() {
                    if let Some(rect) = self.last_terminal_rect {
                        if mouse.column >= rect.x && mouse.column < rect.x + rect.width
                            && mouse.row >= rect.y && mouse.row < rect.y + rect.height
                        {
                            if let Some(pty) = &mut self.pty {
                                let _ = pty.write(b"\x1b[A\x1b[A\x1b[A");
                            }
                            return Ok(Action::Continue);
                        }
                    }
                }
                Ok(Action::Continue)
            }
            MouseEventKind::ScrollDown => {
                // Scroll in session monitor → scroll event log up (older)
                if let Some(rect) = self.last_chain_rect {
                    if mouse.column >= rect.x && mouse.column < rect.x + rect.width
                        && mouse.row >= rect.y && mouse.row < rect.y + rect.height
                    {
                        self.chain_scroll_offset = self.chain_scroll_offset.saturating_sub(3);
                        return Ok(Action::Continue);
                    }
                }

                // Scroll in terminal → forward to PTY (check rect)
                if self.pty_active() {
                    if let Some(rect) = self.last_terminal_rect {
                        if mouse.column >= rect.x && mouse.column < rect.x + rect.width
                            && mouse.row >= rect.y && mouse.row < rect.y + rect.height
                        {
                            if let Some(pty) = &mut self.pty {
                                let _ = pty.write(b"\x1b[B\x1b[B\x1b[B");
                            }
                            return Ok(Action::Continue);
                        }
                    }
                }
                Ok(Action::Continue)
            }
            _ => Ok(Action::Continue),
        }
    }
}

fn pty_reader_thread(mut reader: Box<dyn Read + Send>, tx: mpsc::Sender<Vec<u8>>) {
    let mut buf = [0u8; 4096];
    loop {
        match reader.read(&mut buf) {
            Ok(0) => {
                let _ = tx.send(Vec::new());
                break;
            }
            Ok(n) => {
                if tx.send(buf[..n].to_vec()).is_err() {
                    break;
                }
            }
            Err(_) => {
                let _ = tx.send(Vec::new());
                break;
            }
        }
    }
}

fn key_to_bytes(key: KeyEvent) -> Vec<u8> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);

    match key.code {
        KeyCode::Char(c) => {
            if ctrl {
                let code = (c.to_ascii_lowercase() as u8).wrapping_sub(b'a').wrapping_add(1);
                if code <= 26 {
                    return vec![code];
                }
            }
            if alt {
                let mut bytes = vec![0x1b];
                let mut char_buf = [0u8; 4];
                bytes.extend_from_slice(c.encode_utf8(&mut char_buf).as_bytes());
                return bytes;
            }
            let mut char_buf = [0u8; 4];
            let s = c.encode_utf8(&mut char_buf);
            s.as_bytes().to_vec()
        }
        KeyCode::Enter => vec![b'\r'],
        KeyCode::Backspace => vec![0x7f],
        KeyCode::Tab => vec![b'\t'],
        KeyCode::BackTab => b"\x1b[Z".to_vec(),
        KeyCode::Esc => vec![0x1b],
        KeyCode::Up => b"\x1b[A".to_vec(),
        KeyCode::Down => b"\x1b[B".to_vec(),
        KeyCode::Right => b"\x1b[C".to_vec(),
        KeyCode::Left => b"\x1b[D".to_vec(),
        KeyCode::Home => b"\x1b[H".to_vec(),
        KeyCode::End => b"\x1b[F".to_vec(),
        KeyCode::PageUp => b"\x1b[5~".to_vec(),
        KeyCode::PageDown => b"\x1b[6~".to_vec(),
        KeyCode::Insert => b"\x1b[2~".to_vec(),
        KeyCode::Delete => b"\x1b[3~".to_vec(),
        KeyCode::F(n) => match n {
            1 => b"\x1bOP".to_vec(),
            2 => b"\x1bOQ".to_vec(),
            3 => b"\x1bOR".to_vec(),
            4 => b"\x1bOS".to_vec(),
            5 => b"\x1b[15~".to_vec(),
            6 => b"\x1b[17~".to_vec(),
            7 => b"\x1b[18~".to_vec(),
            8 => b"\x1b[19~".to_vec(),
            9 => b"\x1b[20~".to_vec(),
            10 => b"\x1b[21~".to_vec(),
            11 => b"\x1b[23~".to_vec(),
            12 => b"\x1b[24~".to_vec(),
            _ => vec![],
        },
        _ => vec![],
    }
}
