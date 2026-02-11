use std::io::Read;
use std::path::Path;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct SessionSummary {
    pub name: String,
    pub date: String,
    pub agent: String,
    pub action_count: u32,
    pub proof_verified: bool,
    pub proof_type: String,
}

/// List sessions from .capseal/runs/ directory.
/// Sessions can be directories (run dirs) or .cap files.
pub fn list_sessions(runs_dir: &Path) -> Vec<SessionSummary> {
    if !runs_dir.exists() {
        return Vec::new();
    }

    let mut sessions_by_name: HashMap<String, SessionSummary> = HashMap::new();

    if let Ok(entries) = std::fs::read_dir(runs_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let entry_name = entry.file_name().to_string_lossy().to_string();

            if entry_name == "latest" || entry_name == "latest.cap" {
                continue;
            }

            if path.is_dir() {
                // Run directory — try to read manifest or metadata
                if let Some(summary) = read_run_dir(&path) {
                    sessions_by_name.entry(summary.name.clone()).or_insert(summary);
                }
            } else if path.extension().map(|e| e == "cap").unwrap_or(false) {
                // .cap file — try to read manifest from tar.gz
                if let Some(summary) = read_cap_file(&path) {
                    if let Some(existing) = sessions_by_name.get_mut(&summary.name) {
                        if existing.agent == "unknown" && summary.agent != "unknown" {
                            existing.agent = summary.agent.clone();
                        }
                        if existing.action_count == 0 && summary.action_count > 0 {
                            existing.action_count = summary.action_count;
                        }
                        if !existing.proof_verified && summary.proof_verified {
                            existing.proof_verified = true;
                        }
                        if existing.proof_type == "none" && summary.proof_type != "none" {
                            existing.proof_type = summary.proof_type.clone();
                        }
                    } else {
                        sessions_by_name.insert(summary.name.clone(), summary);
                    }
                }
            }
        }
    }

    let mut sessions: Vec<SessionSummary> = sessions_by_name.into_values().collect();

    // Sort by name (which is typically a timestamp)
    sessions.sort_by(|a, b| b.name.cmp(&a.name));
    sessions
}

fn read_run_dir(dir: &Path) -> Option<SessionSummary> {
    let name = dir.file_name()?.to_string_lossy().to_string();

    // Try to read run_metadata.json
    let meta_path = dir.join("run_metadata.json");
    let capsule_path = dir.join("agent_capsule.json");
    let actions_path = dir.join("actions.jsonl");

    let mut agent = String::from("unknown");
    let mut action_count: u32 = 0;
    let mut proof_verified = false;
    let mut proof_type = String::from("none");

    // Count actions
    if actions_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&actions_path) {
            action_count = content.lines().filter(|l| !l.trim().is_empty()).count() as u32;
        }
    }

    // Read metadata
    if meta_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&meta_path) {
            if let Ok(meta) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(a) = meta.get("agent").and_then(|v| v.as_str()) {
                    agent = a.to_string();
                }
            }
        }
    }

    // Read capsule for proof status
    if capsule_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&capsule_path) {
            if let Ok(capsule) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(v) = capsule.get("verification") {
                    proof_verified = v
                        .get("constraints_valid")
                        .or_else(|| v.get("constraints_satisfied"))
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false);
                    if let Some(pt) = v.get("proof_type").and_then(|v| v.as_str()) {
                        proof_type = pt.to_string();
                    }
                }
            }
        }
    }

    // Extract date from name (format: YYYYMMDDTHHMMSS-type)
    let date = if name.len() >= 15 && name.contains('T') {
        let parts: Vec<&str> = name.splitn(2, '-').collect();
        parts.first().unwrap_or(&name.as_str()).to_string()
    } else {
        name.clone()
    };

    Some(SessionSummary {
        name,
        date,
        agent,
        action_count,
        proof_verified,
        proof_type,
    })
}

fn read_cap_file(path: &Path) -> Option<SessionSummary> {
    let name = path.file_stem()?.to_string_lossy().to_string();

    // .cap files are tar.gz — try to extract manifest.json
    let file = std::fs::File::open(path).ok()?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(gz);

    let mut agent = String::from("unknown");
    let mut action_count: u32 = 0;
    let mut proof_verified = false;
    let mut proof_type = String::from("none");

    for entry in archive.entries().ok()? {
        if let Ok(mut entry) = entry {
            let entry_path = entry.path().ok()?.to_path_buf();
            let file_name = entry_path.file_name()?.to_string_lossy().to_string();

            if file_name == "manifest.json" {
                let mut content = String::new();
                if entry.read_to_string(&mut content).is_ok() {
                    if let Ok(manifest) = serde_json::from_str::<serde_json::Value>(&content) {
                        if let Some(a) = manifest.get("agent")
                            .or_else(|| manifest.get("extras").and_then(|e| e.get("agent")))
                            .and_then(|v| v.as_str())
                        {
                            agent = a.to_string();
                        }
                        if let Some(ac) = manifest
                            .get("actions_count")
                            .or_else(|| manifest.get("extras").and_then(|e| e.get("actions_count")))
                            .and_then(|v| v.as_u64())
                        {
                            action_count = ac as u32;
                        }
                        if let Some(pv) = manifest
                            .get("proof_verified")
                            .or_else(|| manifest.get("extras").and_then(|e| e.get("proof_verified")))
                            .and_then(|v| v.as_bool())
                        {
                            proof_verified = pv;
                        }
                        if let Some(pt) = manifest
                            .get("proof_type")
                            .or_else(|| manifest.get("extras").and_then(|e| e.get("proof_type")))
                            .and_then(|v| v.as_str())
                        {
                            proof_type = pt.to_string();
                        }
                    }
                }
                break; // Got what we need
            }
        }
    }

    let date = if name.len() >= 15 && name.contains('T') {
        let parts: Vec<&str> = name.splitn(2, '-').collect();
        parts.first().unwrap_or(&name.as_str()).to_string()
    } else {
        name.clone()
    };

    Some(SessionSummary {
        name,
        date,
        agent,
        action_count,
        proof_verified,
        proof_type,
    })
}
