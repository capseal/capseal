"""Postgres-backed event store compatible with EventStore interface."""
from __future__ import annotations

import json
import threading
import time
from typing import Dict, List, Optional

import psycopg
from psycopg.rows import dict_row


class PostgresEventStore:
    """Persist events in Postgres with the same interface as EventStore."""

    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("PostgresEventStore requires a DSN")
        self.dsn = dsn
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self):
        return psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)

    def _ensure_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS runs (
          run_id      TEXT PRIMARY KEY,
          created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE TABLE IF NOT EXISTS events (
          run_id   TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
          seq      BIGINT NOT NULL,
          ts       TIMESTAMPTZ NOT NULL DEFAULT now(),
          payload  JSONB NOT NULL,
          raw_line TEXT NOT NULL,
          PRIMARY KEY (run_id, seq)
        );

        CREATE INDEX IF NOT EXISTS events_run_id_seq_idx ON events(run_id, seq);
        """
        with self._connect() as conn:
            conn.execute(ddl)

    def append(self, run_id: str, event_line: str) -> None:
        try:
            payload = json.loads(event_line)
        except json.JSONDecodeError:
            return

        seq = payload.get("seq")
        if seq is None:
            seq = int(time.time() * 1_000_000)
            payload["seq"] = seq
        try:
            seq_i = int(seq)
        except (TypeError, ValueError):
            seq_i = int(time.time() * 1_000_000)
            payload["seq"] = seq_i

        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (run_id,),
            )
            conn.execute(
                """
                INSERT INTO events(run_id, seq, payload, raw_line)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id, seq)
                DO UPDATE SET payload = EXCLUDED.payload, raw_line = EXCLUDED.raw_line
                """,
                (run_id, seq_i, json.dumps(payload), event_line),
            )

    def history(self, run_id: str, last_seq: Optional[int] = None) -> List[str]:
        if last_seq is None:
            query = "SELECT raw_line FROM events WHERE run_id=%s ORDER BY seq"
            params = (run_id,)
        else:
            query = "SELECT raw_line FROM events WHERE run_id=%s AND seq>%s ORDER BY seq"
            params = (run_id, int(last_seq))

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [row["raw_line"] for row in rows]

    def load_existing(self) -> None:
        # Interface compatibility; nothing to do for Postgres.
        return

    def run_ids(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT run_id FROM runs ORDER BY created_at DESC"
            ).fetchall()
        return [row["run_id"] for row in rows]
