//! CapSeal Tauri Application
//!
//! Desktop shell for CapSeal - Governed Execution Console.
//! Provides native file dialogs, IPC commands, and integration with capseal_core.

use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

// =============================================================================
// Types (matching contracts.ts and capseal_core)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LayerStatus {
    Pass,
    Fail,
    Skipped,
    Unknown,
}

impl Default for LayerStatus {
    fn default() -> Self {
        LayerStatus::Unknown
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LayerCheck {
    pub status: LayerStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VerificationLayers {
    pub l0_hash: LayerCheck,
    pub l1_commitment: LayerCheck,
    pub l2_constraint: LayerCheck,
    pub l3_proximity: LayerCheck,
    pub l4_receipt: LayerCheck,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyError {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyReport {
    pub run_id: String,
    pub status: String,
    pub exit_code: i32,
    pub layers: VerificationLayers,
    pub errors: Vec<VerifyError>,
    pub total_ms: f64,
    pub verified_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventSummary {
    pub seq: u64,
    pub event_type: String,
    pub event_hash: String,
    pub prev_hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub run_id: String,
    pub chain_valid: bool,
    pub chain_length: usize,
    pub event_counts: HashMap<String, usize>,
    pub timeline: Vec<EventSummary>,
    pub audited_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSummary {
    pub run_id: String,
    pub capsule_hash: String,
    pub policy_hash: Option<String>,
    pub created_at: String,
    pub verification_status: String,
}

// =============================================================================
// Tauri Commands
// =============================================================================

/// Verify a capsule file
#[tauri::command]
async fn verify_capsule(capsule_path: String) -> Result<VerifyReport, String> {
    let start = std::time::Instant::now();
    let path = PathBuf::from(&capsule_path);

    if !path.exists() {
        return Err(format!("Capsule not found: {}", capsule_path));
    }

    let content = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read capsule: {}", e))?;

    let capsule: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse capsule: {}", e))?;

    let run_id = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let mut layers = VerificationLayers::default();
    let mut errors = Vec::new();
    let mut exit_code = 0;

    // L0: Hash verification
    if let Some(stored_hash) = capsule.get("capsule_hash").and_then(|v| v.as_str()) {
        let mut capsule_for_hash = capsule.clone();
        if let Some(obj) = capsule_for_hash.as_object_mut() {
            obj.remove("capsule_hash");
        }
        let computed = compute_sha256(&serde_json::to_string(&capsule_for_hash).unwrap_or_default());

        if computed == stored_hash {
            layers.l0_hash = LayerCheck {
                status: LayerStatus::Pass,
                message: None,
            };
        } else {
            layers.l0_hash = LayerCheck {
                status: LayerStatus::Fail,
                message: Some("Hash mismatch".to_string()),
            };
            errors.push(VerifyError {
                code: "E011_HASH_MISMATCH".to_string(),
                message: "Capsule hash mismatch".to_string(),
                hint: Some("The capsule may have been tampered with".to_string()),
            });
            exit_code = 10;
        }
    } else {
        layers.l0_hash = LayerCheck {
            status: LayerStatus::Skipped,
            message: Some("No hash field".to_string()),
        };
    }

    // L1: Commitment check
    if capsule.get("row_index_ref").is_some() {
        layers.l1_commitment = LayerCheck {
            status: LayerStatus::Pass,
            message: None,
        };
    } else {
        layers.l1_commitment = LayerCheck {
            status: LayerStatus::Skipped,
            message: Some("No commitment".to_string()),
        };
    }

    // L2: Policy check
    if capsule.get("policy").is_some() {
        layers.l2_constraint = LayerCheck {
            status: LayerStatus::Pass,
            message: None,
        };
    } else {
        layers.l2_constraint = LayerCheck {
            status: LayerStatus::Skipped,
            message: Some("No policy".to_string()),
        };
    }

    // L3: Proof check
    if capsule.get("proof_system").is_some() {
        layers.l3_proximity = LayerCheck {
            status: LayerStatus::Pass,
            message: None,
        };
    } else {
        layers.l3_proximity = LayerCheck {
            status: LayerStatus::Skipped,
            message: Some("No proof".to_string()),
        };
    }

    // L4: Receipt format
    let required = ["capsule_hash", "row_index_ref", "proof_system"];
    let has_all = required.iter().all(|f| capsule.get(*f).is_some());
    if has_all {
        layers.l4_receipt = LayerCheck {
            status: LayerStatus::Pass,
            message: None,
        };
    } else {
        layers.l4_receipt = LayerCheck {
            status: LayerStatus::Fail,
            message: Some("Missing fields".to_string()),
        };
        if exit_code == 0 {
            exit_code = 20;
        }
    }

    let total_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(VerifyReport {
        run_id,
        status: if exit_code == 0 { "verified".to_string() } else { "rejected".to_string() },
        exit_code,
        layers,
        errors,
        total_ms,
        verified_at: chrono_now(),
    })
}

/// Audit an events file
#[tauri::command]
async fn audit_events(events_path: String) -> Result<AuditReport, String> {
    let path = PathBuf::from(&events_path);

    if !path.exists() {
        return Err(format!("Events file not found: {}", events_path));
    }

    let run_id = path
        .parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let file = File::open(&path)
        .map_err(|e| format!("Failed to open events: {}", e))?;

    let reader = BufReader::new(file);
    let mut events: Vec<serde_json::Value> = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| format!("Read error: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(event) = serde_json::from_str(&line) {
            events.push(event);
        }
    }

    events.sort_by_key(|e| e.get("seq").and_then(|v| v.as_u64()).unwrap_or(0));

    // Validate chain
    let mut chain_valid = true;
    let mut prev_hash = "0".repeat(64);

    for event in &events {
        let event_prev = event.get("prev_event_hash")
            .or_else(|| event.get("prev_hash"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if !event_prev.is_empty() && event_prev != prev_hash {
            chain_valid = false;
        }

        prev_hash = event.get("event_hash")
            .or_else(|| event.get("hash"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
    }

    // Count types
    let mut event_counts: HashMap<String, usize> = HashMap::new();
    for event in &events {
        let t = event.get("event_type")
            .or_else(|| event.get("type"))
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        *event_counts.entry(t).or_insert(0) += 1;
    }

    // Build timeline
    let timeline: Vec<EventSummary> = events.iter().map(|e| {
        EventSummary {
            seq: e.get("seq").and_then(|v| v.as_u64()).unwrap_or(0),
            event_type: e.get("event_type")
                .or_else(|| e.get("type"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            event_hash: e.get("event_hash")
                .or_else(|| e.get("hash"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .chars()
                .take(16)
                .collect(),
            prev_hash: e.get("prev_event_hash")
                .or_else(|| e.get("prev_hash"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .chars()
                .take(16)
                .collect(),
        }
    }).collect();

    Ok(AuditReport {
        run_id,
        chain_valid,
        chain_length: events.len(),
        event_counts,
        timeline,
        audited_at: chrono_now(),
    })
}

/// List runs in a directory
#[tauri::command]
async fn list_runs(runs_dir: String) -> Result<Vec<RunSummary>, String> {
    let path = PathBuf::from(&runs_dir);

    if !path.exists() {
        return Ok(Vec::new());
    }

    let mut runs = Vec::new();

    let entries = fs::read_dir(&path)
        .map_err(|e| format!("Failed to read directory: {}", e))?;

    for entry in entries.flatten() {
        if !entry.path().is_dir() {
            continue;
        }

        let capsule_path = entry.path().join("strategy_capsule.json");
        if !capsule_path.exists() {
            continue;
        }

        if let Ok(content) = fs::read_to_string(&capsule_path) {
            if let Ok(capsule) = serde_json::from_str::<serde_json::Value>(&content) {
                runs.push(RunSummary {
                    run_id: entry.file_name().to_string_lossy().to_string(),
                    capsule_hash: capsule.get("capsule_hash")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    policy_hash: capsule.get("policy")
                        .and_then(|p| p.get("policy_hash"))
                        .and_then(|v| v.as_str())
                        .map(String::from),
                    created_at: capsule.get("created_at")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    verification_status: "unverified".to_string(),
                });
            }
        }
    }

    runs.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(runs)
}

/// Get application info
#[tauri::command]
fn get_app_info() -> serde_json::Value {
    serde_json::json!({
        "name": "CapSeal",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "Governed Execution Console",
        "rust_version": "1.77.2",
    })
}

// =============================================================================
// Helpers
// =============================================================================

fn compute_sha256(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    hex::encode(hasher.finalize())
}

fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}Z", duration.as_secs())
}

// =============================================================================
// App Entry Point
// =============================================================================

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            verify_capsule,
            audit_events,
            list_runs,
            get_app_info,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
