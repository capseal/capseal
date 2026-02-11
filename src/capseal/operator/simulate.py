#!/usr/bin/env python3
"""
CapSeal Operator — Event Simulator

Writes realistic events to events.jsonl so you can test
the operator daemon without running a live agent session.

Usage:
    python simulate.py --workspace /path/to/project
    python simulate.py --workspace . --scenario full_session
    python simulate.py --workspace . --scenario chain_break
    python simulate.py --workspace . --event denial  # Single event
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone
from argparse import ArgumentParser


def ts():
    return datetime.now(timezone.utc).isoformat()


def write_event(events_path: Path, event: dict):
    """Append an event to events.jsonl."""
    with open(events_path, "a") as f:
        f.write(json.dumps(event) + "\n")
    etype = event.get("type", "?")
    data = event.get("data", {})
    decision = data.get("decision", "")
    files = data.get("files", [])
    print(f"  → {etype} {decision} {files}")


def scenario_full_session(events_path: Path, delay: float = 2.0):
    """Simulate a complete agent session with mixed events."""
    print("🎬 Scenario: Full session with denials and approvals\n")

    # Session start
    write_event(events_path, {
        "type": "session_start",
        "ts": ts(),
        "data": {
            "session_id": "sim-001",
            "agent": "claude-code",
            "workspace": str(events_path.parent.parent),
            "model": "claude-sonnet-4-20250514",
        }
    })
    time.sleep(delay)

    # Routine approval — low risk file
    write_event(events_path, {
        "type": "gate",
        "ts": ts(),
        "data": {
            "action_type": "edit",
            "files": ["src/utils.py"],
            "description": "Add helper function for string formatting",
            "decision": "approved",
            "p_fail": 0.12,
            "observations": ["Low complexity change", "Test file exists"],
            "diff": "+def format_name(first, last):\n+    return f'{first} {last}'",
            "risk_factors": "Low risk: utility function, has tests",
        }
    })
    time.sleep(delay)

    # Another routine approval
    write_event(events_path, {
        "type": "gate",
        "ts": ts(),
        "data": {
            "action_type": "edit",
            "files": ["tests/test_utils.py"],
            "description": "Add tests for format_name",
            "decision": "approved",
            "p_fail": 0.05,
            "observations": ["Test file only"],
            "diff": "+def test_format_name():\n+    assert format_name('John', 'Doe') == 'John Doe'",
            "risk_factors": "Minimal risk: test file only",
        }
    })
    time.sleep(delay)

    # Risky approval — database file
    write_event(events_path, {
        "type": "gate",
        "ts": ts(),
        "data": {
            "action_type": "edit",
            "files": ["src/db/models.py"],
            "description": "Add email_verified column to users table",
            "decision": "approved",
            "p_fail": 0.52,
            "observations": ["Schema change", "Migration needed", "No destructive ops"],
            "diff": "+    email_verified = Column(Boolean, default=False)\n+    verified_at = Column(DateTime, nullable=True)",
            "risk_factors": "Moderate: schema change, additive only, no data loss",
        }
    })
    time.sleep(delay)

    # DENIAL — auth file, high risk
    write_event(events_path, {
        "type": "gate",
        "ts": ts(),
        "data": {
            "action_type": "edit",
            "files": ["src/auth/token.py"],
            "description": "Remove token expiry check for development convenience",
            "decision": "denied",
            "p_fail": 0.86,
            "observations": ["Security-critical file", "Removes validation", "High historical failure rate"],
            "diff": "-    if token.expires_at < datetime.now():\n-        raise TokenExpiredError()\n+    # TODO: re-enable expiry check",
            "risk_factors": "Critical: removes security validation from auth module",
        }
    })
    time.sleep(delay)

    # Second denial on same file — streak
    write_event(events_path, {
        "type": "gate",
        "ts": ts(),
        "data": {
            "action_type": "edit",
            "files": ["src/auth/token.py"],
            "description": "Modify token validation to skip signature check",
            "decision": "denied",
            "p_fail": 0.91,
            "observations": ["Security-critical file", "Removes crypto validation", "Consecutive denial"],
            "diff": "-    if not verify_signature(token, public_key):\n-        raise InvalidSignatureError()\n+    pass  # skip for now",
            "risk_factors": "Critical: disables cryptographic verification in auth",
        }
    })
    time.sleep(delay)

    # Clean approval after denials
    write_event(events_path, {
        "type": "gate",
        "ts": ts(),
        "data": {
            "action_type": "edit",
            "files": ["src/auth/token.py"],
            "description": "Add input validation for token claims",
            "decision": "approved",
            "p_fail": 0.28,
            "observations": ["Additive change", "Adds validation", "Security improvement"],
            "diff": "+def validate_claims(token):\n+    if not token.get('sub'):\n+        raise ValueError('Missing subject claim')",
            "risk_factors": "Low risk: adds validation, security improvement",
        }
    })
    time.sleep(delay)

    # Session seal
    write_event(events_path, {
        "type": "session_seal",
        "ts": ts(),
        "data": {
            "session_id": "sim-001",
            "total_actions": 6,
            "approved": 4,
            "denied": 2,
            "chain_intact": True,
            "receipt_hash": "a1b2c3d4e5f6...",
        }
    })

    print("\n✅ Full session scenario complete (6 actions, 2 denied)")


def scenario_chain_break(events_path: Path, delay: float = 2.0):
    """Simulate a chain break — the most critical alert."""
    print("🎬 Scenario: Chain break detection\n")

    write_event(events_path, {
        "type": "session_start",
        "ts": ts(),
        "data": {
            "session_id": "sim-002",
            "agent": "claude-code",
            "workspace": str(events_path.parent.parent),
            "model": "claude-sonnet-4-20250514",
        }
    })
    time.sleep(delay)

    # A few normal actions
    for i, f in enumerate(["README.md", "src/main.py", "src/config.py"]):
        write_event(events_path, {
            "type": "gate",
            "ts": ts(),
            "data": {
                "action_type": "edit",
                "files": [f],
                "description": f"Update {f}",
                "decision": "approved",
                "p_fail": 0.1 + (i * 0.05),
            }
        })
        time.sleep(delay * 0.5)

    # CHAIN BREAK
    write_event(events_path, {
        "type": "chain_break",
        "ts": ts(),
        "data": {
            "action_number": 4,
            "file": "src/config.py",
            "expected_hash": "abc123...",
            "actual_hash": "def456...",
            "detail": "Receipt hash mismatch: .cap file was modified between writes",
        }
    })

    print("\n🚨 Chain break scenario complete")


def scenario_sensitive_files(events_path: Path, delay: float = 2.0):
    """Agent touches a bunch of sensitive files."""
    print("🎬 Scenario: Sensitive file access\n")

    write_event(events_path, {
        "type": "session_start",
        "ts": ts(),
        "data": {"session_id": "sim-003", "agent": "claude-code"}
    })
    time.sleep(delay)

    sensitive = [
        (".env", "denied", 0.95, "Adding API keys to .env"),
        ("src/auth/middleware.py", "approved", 0.55, "Modifying auth middleware"),
        ("migrations/20240201_add_admin.sql", "denied", 0.78, "Adding admin user migration"),
        ("deploy/k8s/secrets.yaml", "denied", 0.92, "Modifying Kubernetes secrets"),
    ]

    for filename, decision, p_fail, desc in sensitive:
        write_event(events_path, {
            "type": "gate",
            "ts": ts(),
            "data": {
                "action_type": "edit",
                "files": [filename],
                "description": desc,
                "decision": decision,
                "p_fail": p_fail,
                "risk_factors": f"Sensitive file pattern detected: {filename}",
            }
        })
        time.sleep(delay)

    print("\n✅ Sensitive files scenario complete")


def single_event(events_path: Path, event_type: str):
    """Write a single event for quick testing."""
    events = {
        "approval": {
            "type": "gate",
            "ts": ts(),
            "data": {
                "action_type": "edit",
                "files": ["src/app.py"],
                "decision": "approved",
                "p_fail": 0.15,
                "description": "Add logging to main endpoint",
            }
        },
        "denial": {
            "type": "gate",
            "ts": ts(),
            "data": {
                "action_type": "edit",
                "files": ["src/auth/token.py"],
                "decision": "denied",
                "p_fail": 0.88,
                "description": "Remove token validation",
                "diff": "-    validate_token(token)\n+    pass",
                "risk_factors": "Removes security validation from auth module",
            }
        },
        "chain_break": {
            "type": "chain_break",
            "ts": ts(),
            "data": {
                "action_number": 7,
                "file": "src/config.py",
            }
        },
        "session_start": {
            "type": "session_start",
            "ts": ts(),
            "data": {
                "session_id": f"sim-{int(time.time())}",
                "agent": "claude-code",
                "workspace": str(events_path.parent.parent),
            }
        },
        "session_seal": {
            "type": "session_seal",
            "ts": ts(),
            "data": {
                "session_id": "sim-001",
                "total_actions": 8,
                "approved": 6,
                "denied": 2,
                "chain_intact": True,
            }
        },
    }

    if event_type not in events:
        print(f"Unknown event type: {event_type}")
        print(f"Available: {', '.join(events.keys())}")
        return

    write_event(events_path, events[event_type])
    print(f"\n✅ Single {event_type} event written")


def main():
    parser = ArgumentParser(description="CapSeal Event Simulator")
    parser.add_argument("--workspace", "-w", type=Path, default=Path("."))
    parser.add_argument("--scenario", "-s", type=str, default="full_session",
                        choices=["full_session", "chain_break", "sensitive_files"])
    parser.add_argument("--event", "-e", type=str, default=None,
                        help="Single event: approval, denial, chain_break, session_start, session_seal")
    parser.add_argument("--delay", "-d", type=float, default=2.0,
                        help="Delay between events in seconds")
    parser.add_argument("--clear", action="store_true",
                        help="Clear events.jsonl before starting")
    args = parser.parse_args()

    capseal_dir = args.workspace / ".capseal"
    capseal_dir.mkdir(parents=True, exist_ok=True)
    events_path = capseal_dir / "events.jsonl"

    if args.clear and events_path.exists():
        events_path.unlink()
        print("🗑️  Cleared events.jsonl\n")

    print(f"📁 Writing to: {events_path}\n")

    if args.event:
        single_event(events_path, args.event)
    else:
        scenarios = {
            "full_session": scenario_full_session,
            "chain_break": scenario_chain_break,
            "sensitive_files": scenario_sensitive_files,
        }
        scenarios[args.scenario](events_path, args.delay)


if __name__ == "__main__":
    main()
