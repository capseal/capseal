use ratatui::layout::{Constraint, Direction, Layout, Rect};

/// Layout for active session: 3-zone split
/// Left 25%: compact control panel
/// Top-right 60%: session monitor
/// Bottom-right 40%: terminal
pub struct SessionLayout {
    pub title_bar: Rect,
    pub control_panel: Rect,
    pub action_chain: Rect,
    pub terminal: Rect,
    pub status_bar: Rect,
}

impl SessionLayout {
    pub fn compute(area: Rect) -> Self {
        // Step 1: Vertical — title(1) | main | status(1)
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(10),
                Constraint::Length(1),
            ])
            .split(area);

        let title_bar = outer[0];
        let main_area = outer[1];
        let status_bar = outer[2];

        // Step 2: Horizontal — left 25% | right 75%
        let horizontal = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(25), Constraint::Percentage(75)])
            .split(main_area);

        let control_panel = horizontal[0];
        let right_area = horizontal[1];

        // Step 3: Right side vertical — session monitor 60% | terminal 40%
        let right_split = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(right_area);

        let action_chain = right_split[0];
        let terminal = right_split[1];

        Self {
            title_bar,
            control_panel,
            action_chain,
            terminal,
            status_bar,
        }
    }
}

/// Layout for hub view (no active session): full-screen
pub struct HubLayout {
    pub title_bar: Rect,
    pub hub_area: Rect,
    pub status_bar: Rect,
}

impl HubLayout {
    pub fn compute(area: Rect) -> Self {
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(10),
                Constraint::Length(1),
            ])
            .split(area);

        Self {
            title_bar: outer[0],
            hub_area: outer[1],
            status_bar: outer[2],
        }
    }
}
