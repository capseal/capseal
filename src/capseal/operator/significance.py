"""
CapSeal Operator — Significance Filter

Scores each event 0.0-1.0 to determine notification priority.

Score ranges:
  0.0 - 0.3  Silent (routine, no notification)
  0.3 - 0.5  Log only (shown in TUI console)
  0.5 - 0.7  Text notification
  0.7 - 0.9  Voice note via messaging
  0.9 - 1.0  Live voice alert (critical)
"""

import re
from typing import Optional

# Files that are inherently sensitive
SENSITIVE_PATTERNS = [
    r"auth", r"login", r"password", r"secret", r"credential",
    r"token", r"\.env", r"migration", r"admin", r"payment",
    r"billing", r"stripe", r"key", r"cert", r"private",
    r"security", r"permission", r"role", r"session",
    r"deploy", r"docker", r"k8s", r"terraform", r"ci",
    r"Makefile", r"\.github", r"Cargo\.toml", r"pyproject",
]
SENSITIVE_RE = re.compile("|".join(SENSITIVE_PATTERNS), re.IGNORECASE)


class SignificanceFilter:
    def __init__(self, config: dict):
        self.config = config
        self.notify_threshold = config.get("notify_threshold", 0.5)

    def score(self, event: dict, context) -> float:
        """
        Score an event's significance from 0.0 to 1.0.
        
        Considers: event type, decision, p_fail, file sensitivity,
        consecutive denial streaks, anomalies, session context.
        """
        etype = event.get("type", "")
        data = event.get("data", {})

        # --- Chain break is always critical ---
        if etype == "chain_break":
            return 1.0

        # --- Session lifecycle events ---
        if etype == "session_start":
            return 0.55  # Always notify session start

        if etype == "session_seal":
            # More interesting if there were denials
            if context.denials > 0:
                return 0.7
            return 0.6

        # --- Gate events (the meat) ---
        if etype == "gate":
            return self._score_gate(data, context)

        # --- Agent approval request (blocked, needs response) ---
        if etype == "agent_approval_request":
            return 0.75  # Voice note tier — agent is blocked

        # --- Agent waiting for input ---
        if etype == "agent_waiting":
            return 0.6  # Text notification tier

        # --- Budget/cost events ---
        if etype == "budget_alert":
            return 0.7

        # --- Unknown event types ---
        return 0.2

    def _score_gate(self, data: dict, context) -> float:
        """Score a gate decision event."""
        decision = data.get("decision", "")
        p_fail = data.get("p_fail")
        files = data.get("files", [])
        action_type = data.get("action_type", "")

        score = 0.0

        # --- Decision type ---
        if decision == "denied":
            score = 0.8  # Denials are always important
        elif decision == "approved":
            score = 0.15  # Routine approvals are quiet

        # --- p_fail modifiers ---
        if p_fail is not None:
            if p_fail >= 0.8:
                score = max(score, 0.85)
            elif p_fail >= 0.7:
                score = max(score, 0.7)
            elif p_fail >= 0.5:
                score = max(score, 0.4)
            elif p_fail >= 0.3:
                score = max(score, 0.25)
            # Approved but risky is noteworthy
            if decision == "approved" and p_fail >= 0.5:
                score = max(score, 0.55)

        # --- File sensitivity ---
        sensitive = any(SENSITIVE_RE.search(f) for f in files)
        if sensitive:
            score = max(score, 0.6)
            if decision == "denied":
                score = max(score, 0.85)

        # --- Consecutive denial streak ---
        for f in files:
            streak = context.consecutive_denials.get(f, 0)
            if streak >= 3:
                score = max(score, 0.95)  # Three strikes, escalate hard
            elif streak >= 2:
                score = max(score, 0.9)

        # --- Anomaly: unusual number of files touched ---
        total_unique = len(context.files_touched)
        if context.total_actions > 5 and total_unique > context.total_actions * 0.8:
            # Agent is scattershot — touching lots of different files
            score = max(score, 0.6)

        # --- Action type modifiers ---
        if action_type in ("delete", "remove", "drop"):
            score = max(score, 0.7)

        return min(score, 1.0)

    def should_notify(self, score: float) -> bool:
        return score >= self.notify_threshold

    def tier(self, score: float) -> str:
        """Return the notification tier for a score."""
        if score >= 0.9:
            return "critical"
        elif score >= 0.7:
            return "voice_note"
        elif score >= 0.5:
            return "text"
        elif score >= 0.3:
            return "log"
        return "silent"
