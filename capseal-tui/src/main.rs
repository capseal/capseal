mod agent;
mod app;
mod capseal;
mod config;
mod terminal;
mod ui;
mod workspace;

use anyhow::Result;
use clap::Parser;
use crossterm::event::{DisableMouseCapture, EnableMouseCapture};
use crossterm::terminal::{EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use std::io;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "capseal-tui",
    about = "CapSeal - AI agent trust layer",
    version
)]
struct Cli {
    /// Workspace directory
    #[arg(short, long, default_value = ".")]
    workspace: String,

    /// Skip hub menu, launch shell directly
    #[arg(long)]
    shell: bool,

    /// Launch a specific agent immediately
    #[arg(long)]
    agent: Option<String>,

    /// Theme name
    #[arg(long, default_value = "default")]
    theme: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let workspace = PathBuf::from(&cli.workspace)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(&cli.workspace));

    let config = config::CapSealConfig::load(&workspace)?;

    // Initialize terminal
    crossterm::terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    // Run app
    let mut app = app::App::new(config, cli.shell);
    let result = app.run(&mut terminal);

    // Restore terminal
    crossterm::terminal::disable_raw_mode()?;
    crossterm::execute!(terminal.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;

    result
}
