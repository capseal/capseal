use crate::capseal::{ActionEntry, CapSealEvent, CapSealState};
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

fn safe_truncate(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        text.to_string()
    } else {
        text.chars().skip(char_count - max_chars).collect()
    }
}

/// Reads new events from events.jsonl starting at the given file position.
/// Returns the new file position after reading.
pub fn read_new_events(events_path: &Path, last_pos: u64, state: &mut CapSealState) -> u64 {
    if !events_path.exists() {
        return 0;
    }

    let file = match std::fs::File::open(events_path) {
        Ok(f) => f,
        Err(_) => return last_pos,
    };

    let mut reader = BufReader::new(file);
    if reader.seek(SeekFrom::Start(last_pos)).is_err() {
        return last_pos;
    }

    let mut line = String::new();
    let mut pos = last_pos;

    while reader.read_line(&mut line).unwrap_or(0) > 0 {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            if let Ok(event) = serde_json::from_str::<CapSealEvent>(trimmed) {
                process_event(&event, state);
            }
        }
        pos = reader.stream_position().unwrap_or(pos);
        line.clear();
    }

    pos
}

fn process_event(event: &CapSealEvent, state: &mut CapSealState) {
    let ts = format_timestamp(event.timestamp);

    match event.event_type.as_str() {
        "session_start" => {
            // Start a new on-screen session timeline.
            state.session_active = true;
            state.session_start = Some(std::time::Instant::now());
            state.chain_verified = false;
            state.chain_intact = true;
            state.gates_attempted = 0;
            state.gates_approved = 0;
            state.gates_denied = 0;
            state.actions_recorded = 0;
            state.action_count = 0;
            state.denied_count = 0;
            state.action_chain.clear();

            let mut target = "Session started".to_string();
            if let Some(ref data) = event.data {
                if let Some(agent) = data.get("agent_name").and_then(|v| v.as_str()) {
                    target = format!("Session started ({agent})");
                } else if let Some(session_id) = data.get("session_id").and_then(|v| v.as_str()) {
                    target = format!("Session started ({session_id})");
                }
            }

            state.action_chain.push(ActionEntry {
                action_type: "session_start".to_string(),
                target,
                decision: "started".to_string(),
                p_fail: None,
                label: None,
                observations: None,
                receipt_hash: None,
                timestamp: ts,
                diff: None,
                risk_factors: None,
            });
        }
        "gate" => {
            // Parse summary as fallback: "approve: Fix SQL injection (p_fail=0.12)"
            let (mut decision, rest) = event
                .summary
                .split_once(": ")
                .map(|(d, r)| (d.to_string(), r))
                .unwrap_or_else(|| ("unknown".to_string(), event.summary.as_str()));

            let mut p_fail = rest.rfind("(p_fail=").and_then(|idx| {
                let start = idx + 8;
                let end = rest[start..].find(')')?;
                rest[start..start + end].parse::<f64>().ok()
            });

            let observations = rest.rfind("n=").and_then(|idx| {
                let start = idx + 2;
                let num_str: String = rest[start..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                num_str.parse::<u32>().ok()
            });

            let mut target = rest.split(" (p_fail=").next().unwrap_or("?").to_string();

            // Overlay structured data if available (richer than summary parsing)
            let mut diff = None;
            let mut risk_factors = None;
            let mut label = None;

            if let Some(ref data) = event.data {
                if let Some(d) = data.get("decision").and_then(|v| v.as_str()) {
                    decision = d.to_string();
                }
                if let Some(p) = data.get("p_fail").and_then(|v| v.as_f64()) {
                    p_fail = Some(p);
                }
                if let Some(files) = data.get("files").and_then(|v| v.as_array()) {
                    if let Some(first) = files.first().and_then(|f| f.as_str()) {
                        target = first.to_string();
                    }
                }
                diff = data
                    .get("diff")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                risk_factors = data
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                label = data
                    .get("label")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
            }

            let decision_norm = decision.to_lowercase();
            state.gates_attempted += 1;
            if matches!(decision_norm.as_str(), "skip" | "deny" | "denied") {
                state.gates_denied += 1;
            } else if matches!(
                decision_norm.as_str(),
                "approve" | "approved" | "pass" | "flag" | "flagged" | "human_review"
            ) {
                state.gates_approved += 1;
            }
            state.denied_count = state.gates_denied;

            state.action_chain.push(ActionEntry {
                action_type: "gate".to_string(),
                target,
                decision,
                p_fail,
                label,
                observations,
                receipt_hash: None,
                timestamp: ts,
                diff,
                risk_factors,
            });
            state.session_active = true;
        }
        "record" => {
            state.actions_recorded += 1;
            state.action_count = state.actions_recorded;

            // Try to extract receipt hash from summary
            let mut receipt_hash = event.summary.find("sha:").map(|idx| {
                let start = idx + 4;
                event.summary[start..].chars().take(16).collect::<String>()
            });

            let mut target = event.summary.clone();
            let mut decision = "recorded".to_string();
            let mut p_fail = None;
            let mut label = None;

            // Prefer structured data if available
            if let Some(ref data) = event.data {
                if let Some(rh) = data.get("receipt_hash").and_then(|v| v.as_str()) {
                    receipt_hash = Some(rh.chars().take(16).collect());
                }
                if let Some(files) = data.get("files").and_then(|v| v.as_array()) {
                    if let Some(first) = files.first().and_then(|f| f.as_str()) {
                        target = first.to_string();
                    }
                }
                if let Some(desc) = data.get("description").and_then(|v| v.as_str()) {
                    if !desc.trim().is_empty() {
                        target = desc.to_string();
                    }
                }
                if let Some(ok) = data.get("success").and_then(|v| v.as_bool()) {
                    decision = if ok { "recorded" } else { "record_failed" }.to_string();
                }
                p_fail = data.get("gate_p_fail").and_then(|v| v.as_f64());
                label = data
                    .get("gate_label")
                    .or_else(|| data.get("risk_label"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
            }

            let record_receipt_hash = receipt_hash.clone();

            // Merge receipt_hash into the most recent gate entry that lacks one
            // This pairs gate decisions with their corresponding record hashes
            if let Some(rh) = receipt_hash {
                // Find the index first for logging
                let merge_idx = state
                    .action_chain
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, e)| e.action_type == "gate" && e.receipt_hash.is_none())
                    .map(|(i, _)| i);

                if let Some(idx) = merge_idx {
                    if let Ok(mut log) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open("/tmp/chain-debug.log")
                    {
                        use std::io::Write;
                        let _ = writeln!(log, "Merged hash {} into gate entry {}", rh, idx);
                    }
                    state.action_chain[idx].receipt_hash = Some(rh);
                }
            }

            state.action_chain.push(ActionEntry {
                action_type: "record".to_string(),
                target,
                decision,
                p_fail,
                label,
                observations: None,
                receipt_hash: record_receipt_hash,
                timestamp: ts,
                diff: None,
                risk_factors: None,
            });
            state.session_active = true;
        }
        "seal" | "session_seal" => {
            state.session_active = false;
            let mut chain_intact = true;
            let mut total_actions: Option<u32> = None;
            let mut target = "Session sealed".to_string();
            let mut receipt_hash: Option<String> = None;

            if let Some(ref data) = event.data {
                if let Some(ci) = data.get("chain_intact").and_then(|v| v.as_bool()) {
                    chain_intact = ci;
                }
                total_actions = data
                    .get("total_actions")
                    .and_then(|v| v.as_u64())
                    .and_then(|v| u32::try_from(v).ok());
                if let Some(name) = data.get("session_name").and_then(|v| v.as_str()) {
                    target = format!("Session sealed ({name})");
                } else if let Some(cap_file) = data.get("cap_file").and_then(|v| v.as_str()) {
                    target = format!("Session sealed ({cap_file})");
                }
                receipt_hash = data
                    .get("receipt_hash")
                    .and_then(|v| v.as_str())
                    .map(|s| s.chars().take(16).collect());
            }

            state.chain_verified = true;
            state.chain_intact = chain_intact;
            state.action_chain.push(ActionEntry {
                action_type: "seal".to_string(),
                target,
                decision: if chain_intact {
                    "sealed".to_string()
                } else {
                    "seal_broken".to_string()
                },
                p_fail: None,
                label: None,
                observations: total_actions,
                receipt_hash,
                timestamp: ts,
                diff: None,
                risk_factors: total_actions.map(|n| format!("actions: {n}")),
            });
        }
        "chain_break" => {
            state.chain_verified = true;
            state.chain_intact = false;
            state.action_chain.push(ActionEntry {
                action_type: "chain_break".to_string(),
                target: "Chain integrity break detected".to_string(),
                decision: "broken".to_string(),
                p_fail: None,
                label: None,
                observations: None,
                receipt_hash: None,
                timestamp: ts,
                diff: None,
                risk_factors: event.data.as_ref().and_then(|d| {
                    d.get("reason")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                }),
            });
        }
        "train_start" => {
            state.training_in_progress = true;
            state.training_round = 0;
            // Parse total from summary like "round 0/5"
            if let Some(total) = parse_round_total(&event.summary) {
                state.training_total_rounds = total;
            }
        }
        "train_progress" => {
            state.training_in_progress = true;
            // Parse round from summary like "round 3/5 simple+style=0.86 large=0.43"
            if let Some((round, total)) = parse_round_progress(&event.summary) {
                state.training_round = round;
                state.training_total_rounds = total;
            }
            state.training_profiles = parse_training_profiles(&event.summary);
        }
        "train_complete" => {
            state.training_in_progress = false;
            state.training_round = 0;
            state.training_total_rounds = 0;
            state.training_profiles.clear();
        }
        "agent_approval_request" => {
            let mut prompt_text = event.summary.clone();
            if let Some(ref data) = event.data {
                if let Some(pt) = data.get("prompt_text").and_then(|v| v.as_str()) {
                    prompt_text = pt.to_string();
                }
            }
            if prompt_text.chars().count() > 150 {
                prompt_text = format!("{}\u{2026}", safe_truncate(&prompt_text, 150));
            }
            state.action_chain.push(ActionEntry {
                action_type: "approval".to_string(),
                target: prompt_text,
                decision: "pending".to_string(),
                p_fail: None,
                label: None,
                observations: None,
                receipt_hash: None,
                timestamp: ts,
                diff: None,
                risk_factors: None,
            });
        }
        "agent_waiting" => {
            let mut prompt_text = event.summary.clone();
            if let Some(ref data) = event.data {
                if let Some(pt) = data.get("prompt_text").and_then(|v| v.as_str()) {
                    prompt_text = pt.to_string();
                }
            }
            // Truncate for display
            if prompt_text.chars().count() > 150 {
                prompt_text = format!("{}\u{2026}", safe_truncate(&prompt_text, 150));
            }
            state.action_chain.push(ActionEntry {
                action_type: "waiting".to_string(),
                target: prompt_text,
                decision: "waiting".to_string(),
                p_fail: None,
                label: None,
                observations: None,
                receipt_hash: None,
                timestamp: ts,
                diff: None,
                risk_factors: None,
            });
        }
        _ => {}
    }
}

/// Parse "round N/M" -> Some(M) for total only
fn parse_round_total(summary: &str) -> Option<u32> {
    parse_round_progress(summary).map(|(_, total)| total)
}

/// Parse "round N/M" -> Some((N, M))
fn parse_round_progress(summary: &str) -> Option<(u32, u32)> {
    // Look for pattern "round N/M" or "N/M"
    let s = summary.to_lowercase();
    let slash_part = if let Some(idx) = s.find("round ") {
        &summary[idx + 6..]
    } else {
        summary
    };
    let slash_part = slash_part.trim();
    let (left, right) = slash_part.split_once('/')?;
    let round = left.trim().parse::<u32>().ok()?;
    let total = right
        .trim()
        .split_whitespace()
        .next()?
        .parse::<u32>()
        .ok()?;
    Some((round, total))
}

/// Parse per-profile data from "round 3/5 simple+style=0.86 large=0.43"
fn parse_training_profiles(summary: &str) -> Vec<(String, f64)> {
    let mut profiles = Vec::new();
    for token in summary.split_whitespace() {
        if let Some((name, val)) = token.split_once('=') {
            if let Ok(p_fail) = val.parse::<f64>() {
                profiles.push((name.to_string(), p_fail));
            }
        }
    }
    profiles
}

/// Convert unix timestamp (f64) to HH:MM:SS string
fn format_timestamp(ts: f64) -> String {
    let secs = ts as u64;
    let h = (secs / 3600) % 24;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", h, m, s)
}

#[cfg(test)]
mod tests {
    use super::read_new_events;
    use crate::capseal::CapSealState;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_events_path(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("capseal-events-{name}-{nanos}"));
        fs::create_dir_all(&dir).expect("mkdir");
        dir.join("events.jsonl")
    }

    #[test]
    fn reads_full_session_flow_into_action_chain() {
        let events_path = temp_events_path("full-flow");
        let lines = [
            r#"{"type":"session_start","timestamp":1730000000.0,"summary":"Session started: test","data":{"agent_name":"codex"}}"#,
            r#"{"type":"gate","timestamp":1730000001.0,"summary":"flagged: demo (p_fail=0.50)","data":{"decision":"flagged","p_fail":0.5,"files":["src/e2e_demo.py"],"label":"simple + single-file","reason":"baseline prior"}}"#,
            r#"{"type":"record","timestamp":1730000002.0,"summary":"edit_file: e2e sha:99aadbb1","data":{"action_type":"edit_file","files":["src/e2e_demo.py"],"receipt_hash":"99aadbb12222"}}"#,
            r#"{"type":"seal","timestamp":1730000003.0,"summary":"Sealed 1 actions","data":{"total_actions":1,"chain_intact":true,"receipt_hash":"99aadbb12222"}}"#,
        ];
        fs::write(&events_path, format!("{}\n", lines.join("\n"))).expect("write events");

        let mut state = CapSealState::new(std::path::Path::new("."));
        let pos = read_new_events(&events_path, 0, &mut state);
        assert!(pos > 0);

        let kinds: Vec<&str> = state
            .action_chain
            .iter()
            .map(|e| e.action_type.as_str())
            .collect();
        assert_eq!(kinds, vec!["session_start", "gate", "record", "seal"]);
        assert_eq!(state.gates_attempted, 1);
        assert_eq!(state.actions_recorded, 1);
        assert_eq!(state.gates_denied, 0);
        assert!(state.chain_verified);
        assert!(state.chain_intact);
    }

    #[test]
    fn chain_is_unverified_until_seal_arrives() {
        let events_path = temp_events_path("chain-state");
        let mut state = CapSealState::new(std::path::Path::new("."));

        let first_chunk = [
            r#"{"type":"session_start","timestamp":1730001000.0,"summary":"Session started: test","data":{"agent_name":"codex"}}"#,
            r#"{"type":"gate","timestamp":1730001001.0,"summary":"denied: demo (p_fail=0.91)","data":{"decision":"denied","p_fail":0.91,"files":["src/risky.py"],"reason":"Operator override: denied"}}"#,
        ];
        fs::write(&events_path, format!("{}\n", first_chunk.join("\n")))
            .expect("write first chunk");
        let pos = read_new_events(&events_path, 0, &mut state);
        assert!(pos > 0);
        assert!(!state.chain_verified);
        assert!(state.chain_intact);
        assert_eq!(state.gates_denied, 1);

        let second_chunk = r#"{"type":"session_seal","timestamp":1730001002.0,"summary":"session sealed","data":{"total_actions":0,"chain_intact":false}}"#;
        let mut existing = fs::read_to_string(&events_path).expect("read existing");
        existing.push_str(second_chunk);
        existing.push('\n');
        fs::write(&events_path, existing).expect("append seal");
        let _ = read_new_events(&events_path, pos, &mut state);
        assert!(state.chain_verified);
        assert!(!state.chain_intact);
        assert_eq!(
            state.action_chain.last().map(|e| e.action_type.as_str()),
            Some("seal")
        );
    }
}
