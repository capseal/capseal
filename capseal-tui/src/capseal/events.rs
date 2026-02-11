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
                state.latest_event = Some(event);
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
        "gate" => {
            // Parse summary as fallback: "approve: Fix SQL injection (p_fail=0.12)"
            let (mut decision, rest) = event
                .summary
                .split_once(": ")
                .map(|(d, r)| (d.to_string(), r))
                .unwrap_or_else(|| ("unknown".to_string(), event.summary.as_str()));

            let mut p_fail = rest
                .rfind("(p_fail=")
                .and_then(|idx| {
                    let start = idx + 8;
                    let end = rest[start..].find(')')?;
                    rest[start..start + end].parse::<f64>().ok()
                });

            let observations = rest
                .rfind("n=")
                .and_then(|idx| {
                    let start = idx + 2;
                    let num_str: String = rest[start..].chars().take_while(|c| c.is_ascii_digit()).collect();
                    num_str.parse::<u32>().ok()
                });

            let mut target = rest
                .split(" (p_fail=")
                .next()
                .unwrap_or("?")
                .to_string();

            // Overlay structured data if available (richer than summary parsing)
            let mut diff = None;
            let mut risk_factors = None;

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
                diff = data.get("diff").and_then(|v| v.as_str()).map(|s| s.to_string());
                risk_factors = data.get("reason").and_then(|v| v.as_str()).map(|s| s.to_string());
            }

            if decision == "skip" || decision == "deny" {
                state.denied_count += 1;
            }

            state.action_chain.push(ActionEntry {
                action_type: "gate".to_string(),
                target,
                decision,
                p_fail,
                observations,
                receipt_hash: None,
                timestamp: ts,
                diff,
                risk_factors,
            });
            state.session_active = true;
        }
        "record" => {
            state.action_count += 1;

            // Try to extract receipt hash from summary
            let mut receipt_hash = event
                .summary
                .find("sha:")
                .map(|idx| {
                    let start = idx + 4;
                    event.summary[start..].chars().take(16).collect::<String>()
                });

            // Prefer structured data receipt_hash if available
            if let Some(ref data) = event.data {
                if let Some(rh) = data.get("receipt_hash").and_then(|v| v.as_str()) {
                    receipt_hash = Some(rh.chars().take(16).collect());
                }
            }

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

            state.session_active = true;
        }
        "seal" => {
            state.session_active = false;
            state.chain_intact = true;
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
    let total = right.trim().split_whitespace().next()?.parse::<u32>().ok()?;
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
