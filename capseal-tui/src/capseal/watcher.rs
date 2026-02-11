use crate::capseal::models::RiskModel;
use crate::capseal::sessions;
use crate::capseal::events;
use crate::capseal::CapSealState;
use notify::{Event as NotifyEvent, RecursiveMode, Watcher};
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Instant, SystemTime};

pub struct EventWatcher {
    tx: mpsc::Sender<NotifyEvent>,
    rx: mpsc::Receiver<NotifyEvent>,
    _watcher: Option<notify::RecommendedWatcher>,
    capseal_dir: PathBuf,
    events_path: PathBuf,
    pty_events_path: PathBuf,
    posteriors_path: PathBuf,
    runs_dir: PathBuf,
    operator_status_path: PathBuf,
    pty_input_path: PathBuf,
    last_events_pos: u64,
    last_pty_events_pos: u64,

    // Mtime-based fallback polling
    last_poll: Instant,
    last_events_mtime: Option<SystemTime>,
    last_events_size: u64,
    last_model_mtime: Option<SystemTime>,
    last_sessions_mtime: Option<SystemTime>,
}

impl EventWatcher {
    pub fn new(capseal_dir: &std::path::Path) -> Self {
        let (tx, rx) = mpsc::channel();

        let watcher = if capseal_dir.exists() {
            Self::create_watcher(&tx, capseal_dir)
        } else {
            None
        };

        let dir = capseal_dir.to_path_buf();
        Self {
            tx,
            rx,
            _watcher: watcher,
            events_path: dir.join("events.jsonl"),
            pty_events_path: dir.join("pty_events.jsonl"),
            posteriors_path: dir.join("models").join("beta_posteriors.npz"),
            runs_dir: dir.join("runs"),
            operator_status_path: dir.join("operator_status.json"),
            pty_input_path: dir.join("pty_input.txt"),
            capseal_dir: dir,
            last_events_pos: 0,
            last_pty_events_pos: 0,
            last_poll: Instant::now(),
            last_events_mtime: None,
            last_events_size: 0,
            last_model_mtime: None,
            last_sessions_mtime: None,
        }
    }

    fn create_watcher(
        tx: &mpsc::Sender<NotifyEvent>,
        capseal_dir: &std::path::Path,
    ) -> Option<notify::RecommendedWatcher> {
        let tx = tx.clone();
        let mut w = notify::recommended_watcher(move |res: notify::Result<NotifyEvent>| {
            if let Ok(event) = res {
                let _ = tx.send(event);
            }
        })
        .ok()?;

        let _ = w.watch(capseal_dir, RecursiveMode::Recursive);
        Some(w)
    }

    /// Non-blocking poll for file system changes.
    /// Call this in the main event loop.
    pub fn poll(&mut self, state: &mut CapSealState) {
        // Lazy init: if .capseal/ appeared after startup, start watching
        if self._watcher.is_none() && self.capseal_dir.exists() {
            self._watcher = Self::create_watcher(&self.tx, &self.capseal_dir);
            if self._watcher.is_some() {
                state.initialized = true;
                // Do a full initial load since we just discovered the directory
                self.initial_load(state);
                return;
            }
        }

        let mut events_changed = false;
        let mut model_changed = false;
        let mut sessions_changed = false;
        let mut pty_input_changed = false;

        // Drain all pending file system events from notify
        while let Ok(event) = self.rx.try_recv() {
            for path in &event.paths {
                let filename = path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .unwrap_or("");

                match filename {
                    "events.jsonl" | "pty_events.jsonl" => events_changed = true,
                    "pty_input.txt" => pty_input_changed = true,
                    "beta_posteriors.npz" => model_changed = true,
                    "config.json" => {
                        state.initialized = self.capseal_dir.join("config.json").exists();
                    }
                    f if f.ends_with(".cap") => sessions_changed = true,
                    "actions.jsonl" | "agent_capsule.json" | "run_metadata.json" => {
                        sessions_changed = true;
                    }
                    _ => {}
                }
            }
        }

        // Mtime-based fallback: check every 250ms in case notify missed events
        let now = Instant::now();
        if now.duration_since(self.last_poll) > std::time::Duration::from_millis(250) {
            self.last_poll = now;

            // Check events.jsonl mtime and size
            if let Ok(meta) = std::fs::metadata(&self.events_path) {
                let size = meta.len();
                let mtime = meta.modified().ok();
                if size != self.last_events_size || mtime != self.last_events_mtime {
                    self.last_events_size = size;
                    self.last_events_mtime = mtime;
                    events_changed = true;
                }
            }

            // Check beta_posteriors.npz mtime
            if let Ok(meta) = std::fs::metadata(&self.posteriors_path) {
                if let Ok(mtime) = meta.modified() {
                    if self.last_model_mtime != Some(mtime) {
                        self.last_model_mtime = Some(mtime);
                        model_changed = true;
                    }
                }
            }

            // Check runs/ dir mtime (any new file triggers session reload)
            if let Ok(meta) = std::fs::metadata(&self.runs_dir) {
                if let Ok(mtime) = meta.modified() {
                    if self.last_sessions_mtime != Some(mtime) {
                        self.last_sessions_mtime = Some(mtime);
                        sessions_changed = true;
                    }
                }
            }

            // Check operator_status.json
            if let Ok(contents) = std::fs::read_to_string(&self.operator_status_path) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&contents) {
                    state.operator_online = val
                        .get("online")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    state.operator_channels = val
                        .get("channels")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                }
            } else {
                state.operator_online = false;
                state.operator_channels = 0;
            }
        }

        // Check for pty_input.txt (operator PTY injection)
        // Always check — it's a one-shot file deleted after reading
        if pty_input_changed || self.pty_input_path.exists() {
            if let Ok(contents) = std::fs::read(&self.pty_input_path) {
                if !contents.is_empty() {
                    state.pending_pty_injection = Some(contents);
                    let _ = std::fs::remove_file(&self.pty_input_path);
                }
            }
        }

        // Process changes
        if events_changed {
            self.last_events_pos =
                events::read_new_events(&self.events_path, self.last_events_pos, state);
            self.last_pty_events_pos =
                events::read_new_events(&self.pty_events_path, self.last_pty_events_pos, state);
        }

        if model_changed {
            self.reload_model(state);
        }

        if sessions_changed {
            self.reload_sessions(state);
        }
    }

    /// Initial load of model and sessions
    pub fn initial_load(&mut self, state: &mut CapSealState) {
        self.reload_model(state);
        self.reload_sessions(state);

        // Read any existing events
        self.last_events_pos = events::read_new_events(&self.events_path, 0, state);

        // Snapshot current mtimes/sizes so we don't re-trigger on the first poll
        if let Ok(meta) = std::fs::metadata(&self.events_path) {
            self.last_events_mtime = meta.modified().ok();
            self.last_events_size = meta.len();
        }
        if let Ok(meta) = std::fs::metadata(&self.posteriors_path) {
            self.last_model_mtime = meta.modified().ok();
        }
        if let Ok(meta) = std::fs::metadata(&self.runs_dir) {
            self.last_sessions_mtime = meta.modified().ok();
        }
    }

    /// Reset events position to current EOF — only new events will be shown
    pub fn reset_events_position(&mut self) {
        if let Ok(meta) = std::fs::metadata(&self.events_path) {
            self.last_events_pos = meta.len();
            self.last_events_size = meta.len();
            self.last_events_mtime = meta.modified().ok();
        }
    }

    fn reload_model(&self, state: &mut CapSealState) {
        if self.posteriors_path.exists() {
            match RiskModel::load(&self.posteriors_path) {
                Ok(model) => {
                    state.episode_count = model.episode_count();
                    state.profile_count = model.profile_count();
                    state.risk_scores = model
                        .active_profiles(1.0)
                        .into_iter()
                        .map(|(idx, p)| (format!("profile_{}", idx), p))
                        .take(10) // Show top 10
                        .collect();
                    state.model_loaded = true;

                    // Get file modification time
                    if let Ok(meta) = std::fs::metadata(&self.posteriors_path) {
                        if let Ok(modified) = meta.modified() {
                            let elapsed = modified.elapsed().unwrap_or_default();
                            state.model_updated = Some(format_elapsed(elapsed));
                        }
                    }
                }
                Err(e) => {
                    eprintln!("[capseal] Failed to load risk model: {}", e);
                    state.model_loaded = false;
                }
            }
        }
    }

    fn reload_sessions(&self, state: &mut CapSealState) {
        state.sessions = sessions::list_sessions(&self.runs_dir);
    }
}

fn format_elapsed(elapsed: std::time::Duration) -> String {
    let secs = elapsed.as_secs();
    if secs < 60 {
        format!("{}s ago", secs)
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
}
