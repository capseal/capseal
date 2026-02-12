pub mod events;
pub mod models;
pub mod sessions;
pub mod watcher;

use serde::Deserialize;

/// Aggregated CapSeal state updated by file watcher
pub struct CapSealState {
    pub initialized: bool,

    // Risk model
    pub model_loaded: bool,
    pub episode_count: u32,
    pub profile_count: u32,
    pub model_updated: Option<String>,
    pub risk_scores: Vec<(String, f64)>,

    // Training state
    pub training_in_progress: bool,
    pub training_round: u32,
    pub training_total_rounds: u32,
    pub training_profiles: Vec<(String, f64)>,

    // Current session
    pub session_active: bool,
    pub gates_attempted: u32,
    pub gates_approved: u32,
    pub gates_denied: u32,
    pub actions_recorded: u32,
    pub action_count: u32,
    pub denied_count: u32,
    pub chain_verified: bool,
    pub chain_intact: bool,
    pub action_chain: Vec<ActionEntry>,
    pub session_start: Option<std::time::Instant>,

    // PTY injection from operator (pty_input.txt)
    pub pending_pty_injection: Option<Vec<u8>>,

    // Operator status
    pub operator_online: bool,
    pub operator_channels: u32,
    pub operator_channel_types: Vec<String>,
    pub operator_events_processed: u64,
    pub operator_voice_connected: bool,
    pub operator_last_alert_ts: Option<f64>,
    pub operator_workspace: Option<String>,
    pub voice_active: bool,

    // Intervention visibility
    pub pending_intervention: Option<PendingIntervention>,

    // History
    pub sessions: Vec<sessions::SessionSummary>,
}

impl CapSealState {
    pub fn new(workspace: &std::path::Path) -> Self {
        let initialized = workspace.join(".capseal").exists();
        Self {
            initialized,
            model_loaded: false,
            episode_count: 0,
            profile_count: 0,
            model_updated: None,
            risk_scores: Vec::new(),
            training_in_progress: false,
            training_round: 0,
            training_total_rounds: 0,
            training_profiles: Vec::new(),
            session_active: false,
            gates_attempted: 0,
            gates_approved: 0,
            gates_denied: 0,
            actions_recorded: 0,
            action_count: 0,
            denied_count: 0,
            chain_verified: false,
            chain_intact: true,
            action_chain: Vec::new(),
            session_start: None,
            pending_pty_injection: None,
            operator_online: false,
            operator_channels: 0,
            operator_channel_types: Vec::new(),
            operator_events_processed: 0,
            operator_voice_connected: false,
            operator_last_alert_ts: None,
            operator_workspace: None,
            voice_active: false,
            pending_intervention: None,
            sessions: Vec::new(),
        }
    }

    pub fn session_duration_secs(&self) -> u64 {
        self.session_start
            .map(|s| s.elapsed().as_secs())
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct PendingIntervention {
    pub action: String,
    pub source: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CapSealEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    pub timestamp: f64,
    pub summary: String,
    #[serde(default)]
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ActionEntry {
    pub action_type: String,
    pub target: String,
    pub decision: String,
    pub p_fail: Option<f64>,
    pub label: Option<String>,
    pub observations: Option<u32>,
    pub receipt_hash: Option<String>,
    pub timestamp: String,
    pub diff: Option<String>,
    pub risk_factors: Option<String>,
}
