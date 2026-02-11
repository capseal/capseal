use anyhow::Result;
use serde::Deserialize;
use std::path::{Path, PathBuf};

// Gate thresholds — deserialized from config.json, used by Python MCP server.
// Kept here so the full config.json schema deserializes without errors.
#[derive(Debug, Clone, Deserialize, Default)]
#[allow(dead_code)]
pub struct GateConfig {
    #[serde(default = "default_threshold")]
    pub threshold: f64,
    #[serde(default = "default_uncertainty")]
    pub uncertainty_threshold: f64,
}

fn default_threshold() -> f64 {
    0.6
}
fn default_uncertainty() -> f64 {
    0.15
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct CapSealConfig {
    #[serde(default)]
    #[allow(dead_code)]
    pub version: String,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub default_command: String,
    #[serde(default)]
    pub default_agent: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub scan_profile: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub test_cmd: String,
    #[serde(default)]
    #[allow(dead_code)]
    pub gate: GateConfig,

    /// Resolved workspace path (not from JSON)
    #[serde(skip)]
    pub workspace: PathBuf,
    /// Whether .capseal/ directory exists
    #[serde(skip)]
    pub initialized: bool,
}

impl CapSealConfig {
    pub fn load(workspace: &Path) -> Result<Self> {
        let capseal_dir = workspace.join(".capseal");
        let config_path = capseal_dir.join("config.json");

        let mut config = if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            serde_json::from_str::<CapSealConfig>(&content).unwrap_or_default()
        } else {
            CapSealConfig::default()
        };

        config.workspace = workspace.to_path_buf();
        config.initialized = capseal_dir.exists();

        Ok(config)
    }

    pub fn capseal_dir(&self) -> PathBuf {
        self.workspace.join(".capseal")
    }
}
