"""Minimal event storage backend for CapsuleBench relay."""
from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class EventStore:
    """Persists events per run_id to disk and keeps in-memory history."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._buffers: Dict[str, List[str]] = defaultdict(list)

    def append(self, run_id: str, event_line: str) -> None:
        with self._lock:
            self._buffers[run_id].append(event_line)
            path = self.root / f"{run_id}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(event_line + "\n")

    def history(self, run_id: str, last_seq: int | None = None) -> List[str]:
        with self._lock:
            events = list(self._buffers.get(run_id, []))
        if last_seq is None:
            return events
        filtered = []
        for line in events:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(payload.get("seq", 0)) > last_seq:
                filtered.append(line)
        return filtered

    def load_existing(self) -> None:
        for path in self.root.glob("*.jsonl"):
            run_id = path.stem
            with path.open("r", encoding="utf-8") as fh:
                lines = [line.rstrip("\n") for line in fh]
            self._buffers[run_id] = lines

    def run_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._buffers.keys())
