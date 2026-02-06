"""Unified hash-chained event logging for CapSeal.

All auditable events (run lifecycle, oracle calls, tool invocations) are logged
to a hash-chained append-only log. This provides:

1. **Tamper evidence**: Any modification breaks the hash chain
2. **Auditability**: Complete history of all actions
3. **Reproducibility**: Events can be replayed/verified

Event Types:
- run_started, run_completed, run_failed
- trace_simulated, spec_locked, statement_locked
- row_root_finalized, proof_artifact, capsule_sealed
- oracle_call, context_pack_created, budget_checkpoint
- tool_invocation (MCP)
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Event type constants
EVENT_RUN_STARTED = "run_started"
EVENT_RUN_COMPLETED = "run_completed"
EVENT_RUN_FAILED = "run_failed"
EVENT_TRACE_SIMULATED = "trace_simulated"
EVENT_SPEC_LOCKED = "spec_locked"
EVENT_STATEMENT_LOCKED = "statement_locked"
EVENT_ROW_ROOT_FINALIZED = "row_root_finalized"
EVENT_PROOF_ARTIFACT = "proof_artifact"
EVENT_CAPSULE_SEALED = "capsule_sealed"

# Oracle/governance events
EVENT_ORACLE_CALL = "oracle_call"
EVENT_CONTEXT_PACK_CREATED = "context_pack_created"
EVENT_BUDGET_CHECKPOINT = "budget_checkpoint"
EVENT_BUDGET_EXCEEDED = "budget_exceeded"

# Tool events
EVENT_TOOL_INVOCATION = "tool_invocation"

HASH_PREFIX = b"CAPSEAL_EVENT_V1::"
GENESIS_HASH = "0" * 64


@dataclass
class Event:
    """A single auditable event in the hash chain."""

    seq: int
    event_type: str
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    prev_hash: str = GENESIS_HASH
    event_hash: str = ""

    def compute_hash(self) -> str:
        """Compute hash of this event (excluding event_hash field)."""
        payload = {
            "seq": self.seq,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "prev_hash": self.prev_hash,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(HASH_PREFIX + canonical.encode()).hexdigest()

    def finalize(self) -> "Event":
        """Set event_hash and return self."""
        self.event_hash = self.compute_hash()
        return self

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "type": self.event_type,
            "event_type": self.event_type,  # Alias for compatibility
            "data": self.data,
            "timestamp": self.timestamp,
            "prev_event_hash": self.prev_hash,
            "prev_hash": self.prev_hash,  # Alias
            "event_hash": self.event_hash,
            "hash": self.event_hash,  # Alias
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(
            seq=d.get("seq", 0),
            event_type=d.get("type") or d.get("event_type", "unknown"),
            data=d.get("data", {}),
            timestamp=d.get("timestamp", 0),
            prev_hash=d.get("prev_event_hash") or d.get("prev_hash", GENESIS_HASH),
            event_hash=d.get("event_hash") or d.get("hash", ""),
        )


class EventLog:
    """Hash-chained append-only event log.

    Usage:
        log = EventLog(Path("out/my_run"))
        log.emit(EVENT_RUN_STARTED, {"run_id": "my_run"})
        log.emit(EVENT_ORACLE_CALL, {"oracle_id": "claude", "tokens": 1000})
        log.emit(EVENT_RUN_COMPLETED, {"status": "ok"})

        # Verify chain integrity
        assert log.verify_chain()
    """

    def __init__(self, output_dir: Path, filename: str = "events.jsonl"):
        self.output_dir = Path(output_dir)
        self.log_path = self.output_dir / filename
        self.events: list[Event] = []
        self._seq = 0
        self._last_hash = GENESIS_HASH

        # Load existing events if any
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing events from file."""
        if not self.log_path.exists():
            return

        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = Event.from_dict(json.loads(line))
                    self.events.append(event)
                    self._seq = max(self._seq, event.seq)
                    self._last_hash = event.event_hash
                except json.JSONDecodeError:
                    pass

    def emit(self, event_type: str, data: Optional[dict] = None) -> Event:
        """Emit a new event to the log."""
        self._seq += 1
        event = Event(
            seq=self._seq,
            event_type=event_type,
            data=data or {},
            prev_hash=self._last_hash,
        ).finalize()

        self.events.append(event)
        self._last_hash = event.event_hash

        # Append to file
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(event.to_json() + "\n")

        return event

    def emit_oracle_call(
        self,
        oracle_id: str,
        call_id: str,
        context_root: str,
        tokens_in: int,
        tokens_out: int,
        model: str = "",
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        success: bool = True,
        error: Optional[str] = None,
    ) -> Event:
        """Emit an oracle call event with full tracking data."""
        return self.emit(EVENT_ORACLE_CALL, {
            "oracle_id": oracle_id,
            "call_id": call_id,
            "context_root": context_root,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "tokens_total": tokens_in + tokens_out,
            "model": model,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
            "success": success,
            "error": error,
        })

    def emit_context_pack(
        self,
        pack_id: str,
        context_root: str,
        chunk_count: int,
        total_bytes: int,
        estimated_tokens: int,
        selection_algorithm: str = "manual",
    ) -> Event:
        """Emit a context pack creation event."""
        return self.emit(EVENT_CONTEXT_PACK_CREATED, {
            "pack_id": pack_id,
            "context_root": context_root,
            "chunk_count": chunk_count,
            "total_bytes": total_bytes,
            "estimated_tokens": estimated_tokens,
            "selection_algorithm": selection_algorithm,
        })

    def emit_budget_checkpoint(
        self,
        run_id: str,
        spent_tokens: int,
        spent_calls: int,
        spent_usd: float,
        remaining_tokens: int,
        remaining_calls: int,
        remaining_usd: float,
    ) -> Event:
        """Emit a budget checkpoint event."""
        return self.emit(EVENT_BUDGET_CHECKPOINT, {
            "run_id": run_id,
            "spent_tokens": spent_tokens,
            "spent_calls": spent_calls,
            "spent_usd": spent_usd,
            "remaining_tokens": remaining_tokens,
            "remaining_calls": remaining_calls,
            "remaining_usd": remaining_usd,
        })

    def verify_chain(self) -> bool:
        """Verify the hash chain integrity."""
        if not self.events:
            return True

        prev_hash = GENESIS_HASH
        for event in self.events:
            # Check prev_hash links correctly
            if event.prev_hash != prev_hash:
                return False

            # Check event_hash is correct
            expected_hash = event.compute_hash()
            if event.event_hash != expected_hash:
                return False

            prev_hash = event.event_hash

        return True

    @property
    def head_hash(self) -> str:
        return self._last_hash

    @property
    def chain_length(self) -> int:
        return len(self.events)

    def get_events_by_type(self, event_type: str) -> list[Event]:
        return [e for e in self.events if e.event_type == event_type]

    def get_oracle_calls(self) -> list[Event]:
        return self.get_events_by_type(EVENT_ORACLE_CALL)

    def get_total_token_spend(self) -> dict:
        """Get aggregate token spend from all oracle calls."""
        calls = self.get_oracle_calls()
        return {
            "tokens_in": sum(e.data.get("tokens_in", 0) for e in calls),
            "tokens_out": sum(e.data.get("tokens_out", 0) for e in calls),
            "tokens_total": sum(e.data.get("tokens_total", 0) for e in calls),
            "cost_usd": sum(e.data.get("cost_usd", 0) for e in calls),
            "call_count": len(calls),
        }

    def to_summary(self) -> dict:
        """Generate audit summary."""
        event_counts: dict[str, int] = {}
        for event in self.events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1

        return {
            "chain_valid": self.verify_chain(),
            "chain_length": self.chain_length,
            "genesis_hash": GENESIS_HASH,
            "head_hash": self.head_hash,
            "event_counts": event_counts,
            "oracle_spend": self.get_total_token_spend(),
            "first_event": self.events[0].to_dict() if self.events else None,
            "last_event": self.events[-1].to_dict() if self.events else None,
        }


# Global convenience functions for simple usage
_default_log: Optional[EventLog] = None


def init_event_log(output_dir: Path) -> EventLog:
    """Initialize the default event log."""
    global _default_log
    _default_log = EventLog(output_dir)
    return _default_log


def get_event_log() -> Optional[EventLog]:
    """Get the default event log (if initialized)."""
    return _default_log


def emit(event_type: str, data: Optional[dict] = None) -> Optional[Event]:
    """Emit to the default event log."""
    if _default_log:
        return _default_log.emit(event_type, data)
    return None
