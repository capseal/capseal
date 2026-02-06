"""Event logging utilities for CapsuleBench runs."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional


class EventLogger:
    """Writes canonical JSON events to a log file and exposes progress callbacks."""

    def __init__(
        self,
        path: Path,
        run_id: str,
        *,
        trace_id: str | None = None,
        default_source: str = "runner",
    ) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        self.run_id = run_id
        self.trace_id = trace_id
        self._seq = 0
        self._last_hash = "0" * 64
        self._default_source = default_source

    def emit(
        self,
        event_type: str,
        data: Optional[dict] = None,
        *,
        source: str | None = None,
        trace_id: str | None = None,
    ) -> str:
        if self._fh is None:
            raise RuntimeError("event logger is closed")
        base_event = {
            "schema": "bef_capsule_stream_v1",
            "v": 1,
            "run_id": self.run_id,
            "seq": self._next_seq(),
            "ts_ms": int(time.time() * 1000),
            "source": source or self._default_source,
            "type": event_type,
            "data": data or {},
        }
        effective_trace_id = trace_id or self.trace_id
        if effective_trace_id:
            base_event["trace_id"] = effective_trace_id
        serialized = json.dumps(base_event, sort_keys=True, separators=(",", ":")).encode("utf-8")
        prev_hash = self._last_hash
        event_hash = hashlib.sha256(bytes.fromhex(prev_hash) + serialized).hexdigest()
        payload = dict(base_event)
        payload["prev_event_hash"] = prev_hash
        payload["event_hash"] = event_hash
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self._fh.write(line + os.linesep)
        self._fh.flush()
        self._last_hash = event_hash
        return event_hash

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def path_str(self) -> str:
        return str(self.path)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class ProgressSink:
    """Bridges TraceAdapter progress callbacks into the EventLogger."""

    def __init__(self, logger: EventLogger) -> None:
        self.logger = logger

    def callback(self, payload: dict) -> None:
        event_type = payload.get("type", "progress")
        data = payload.get("data", {})
        self.logger.emit(event_type, data)
