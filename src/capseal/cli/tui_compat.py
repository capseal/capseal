"""Utilities for Python CLI ↔ Rust TUI integration.

When the Python CLI runs inside the Rust TUI's embedded terminal,
CAPSEAL_TUI=1 is set in the environment. Commands can use this to:
  - Skip heavy Rich output that doesn't render well in nested terminals
  - Emit events to events.jsonl for the TUI sidebar to pick up
"""
from __future__ import annotations

import os


def is_inside_tui() -> bool:
    """Return True when running inside the CapSeal Rust TUI."""
    return os.environ.get("CAPSEAL_TUI") == "1"


def suppress_cpr() -> None:
    """Suppress prompt_toolkit CPR warnings when inside TUI."""
    if is_inside_tui():
        os.environ.setdefault("PROMPT_TOOLKIT_NO_CPR", "1")


def emit_tui_event(event_type: str, summary: str) -> None:
    """Write a JSON event to events.jsonl for TUI sidebar consumption.

    Only emits when running inside the TUI (CAPSEAL_TUI=1).
    The Rust TUI's file watcher picks these up and updates the sidebar.
    """
    if not is_inside_tui():
        return

    import json
    import time

    workspace = os.environ.get("CAPSEAL_WORKSPACE", ".")
    events_path = os.path.join(workspace, ".capseal", "events.jsonl")
    try:
        with open(events_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "type": event_type,
                        "timestamp": time.time(),
                        "summary": summary,
                    }
                )
                + "\n"
            )
    except OSError:
        pass
