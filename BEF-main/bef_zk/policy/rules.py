"""Runtime policy enforcement utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import json


class PolicyError(RuntimeError):
    """Raised when a policy rule is violated."""


@dataclass
class PolicyConfig:
    raw: Dict[str, Any]

    @property
    def rules(self) -> Dict[str, Any]:
        return self.raw.get("rules", {})


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PolicyError(f"policy file is not valid JSON: {path}") from exc


def load_policy_config(path: Path | None) -> PolicyConfig | None:
    """Load a policy JSON file if provided."""
    if path is None:
        return None
    data = _load_json(path)
    # backwards compatibility with legacy schema (tracks[])
    if "rules" not in data and data.get("tracks"):
        track = data["tracks"][0]
        data = {
            "policy_id": data.get("policy_id") or track.get("track_id"),
            "policy_version": data.get("policy_version", "unspecified"),
            "rules": track.get("rules", {}),
        }
    return PolicyConfig(raw=data)


def enforce_dataset_rules(config: PolicyConfig | None, *, dataset_id: str, files: Iterable[Dict[str, Any]]) -> None:
    if not config:
        return
    dataset_rules = config.rules.get("dataset_integrity")
    if not dataset_rules:
        return
    allowlist = dataset_rules.get("dataset_allowlist")
    if allowlist and dataset_id not in allowlist:
        raise PolicyError(f"dataset '{dataset_id}' not permitted by policy")
    files_list = list(files)
    total_files = len(files_list)
    total_bytes = 0
    for entry in files_list:
        try:
            size = int(entry.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        total_bytes += max(0, size)
        if dataset_rules.get("hash_algorithm") and dataset_rules["hash_algorithm"].lower() != "sha256":
            raise PolicyError("only sha256 dataset commitments are supported by policy")
    max_files = dataset_rules.get("max_total_files")
    if max_files is not None and total_files > int(max_files):
        raise PolicyError(
            f"dataset '{dataset_id}' has {total_files} files exceeds policy limit {max_files}"
        )
    max_bytes = dataset_rules.get("max_total_bytes")
    if max_bytes is not None and total_bytes > int(max_bytes):
        raise PolicyError(
            f"dataset '{dataset_id}' has {total_bytes} bytes exceeds policy limit {max_bytes}"
        )


def enforce_access_rules(
    config: PolicyConfig | None,
    *,
    access_entries: Iterable[Dict[str, Any]],
) -> None:
    if not config:
        return
    rules = config.rules.get("access_control")
    if not rules:
        return
    entries = list(access_entries)
    if rules.get("require_access_log") and not entries:
        raise PolicyError("policy requires access log but no entries were recorded")
    if not entries:
        return
    total_reads = len(entries)
    total_bytes = 0
    for entry in entries:
        try:
            size = int(((entry.get("extra") or {}).get("size")) or 0)
        except (TypeError, ValueError):
            size = 0
        total_bytes += max(0, size)
        if rules.get("forbid_out_of_manifest_reads"):
            if entry.get("dataset_id") is None or entry.get("chunk_id") is None:
                raise PolicyError("access entry missing dataset/chunk reference")
    max_files = rules.get("max_files_read_total")
    if max_files is not None and total_reads > int(max_files):
        raise PolicyError(
            f"policy limits reads to {max_files} entries but {total_reads} recorded"
        )
    max_bytes = rules.get("max_bytes_read_total")
    if max_bytes is not None and total_bytes > int(max_bytes):
        raise PolicyError(
            f"policy limits read volume to {max_bytes} bytes but {total_bytes} recorded"
        )


def enforce_pii_guardrail(config: PolicyConfig | None, *, telemetry: Dict[str, Any]) -> None:
    if not config:
        return
    guard = config.rules.get("pii_guardrail")
    if not guard or not guard.get("enabled"):
        return
    counters = (telemetry.get("counters") or {}) if telemetry else {}
    total_hits = int(counters.get("pii_hits") or 0)
    max_total = int(guard.get("thresholds", {}).get("max_pii_hits_total", 0))
    if total_hits > max_total:
        raise PolicyError(
            f"policy forbids PII but {total_hits} hits recorded (limit {max_total})"
        )
    per_row_hits = int(counters.get("pii_hits_per_row_max") or 0)
    max_per_row = int(guard.get("thresholds", {}).get("max_pii_hits_per_row", 0))
    if per_row_hits > max_per_row:
        raise PolicyError(
            f"policy forbids PII per row but max {per_row_hits} hits recorded"
        )


def enforce_execution_limits(config: PolicyConfig | None, *, counters: Dict[str, Any]) -> None:
    """Enforce global execution budget limits such as rows/time/bytes."""

    if not config:
        return
    limits = config.rules.get("execution_limits")
    if not limits:
        return
    counters = counters or {}

    def _coerce(value: Any) -> int | float:
        if value is None:
            return 0
        if isinstance(value, (int, float)):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0

    rows_total = _coerce(counters.get("rows_total"))
    max_rows = limits.get("max_rows_total")
    if max_rows is not None and rows_total > float(max_rows):
        raise PolicyError(
            f"policy limits rows_total to {max_rows} but {rows_total} recorded"
        )

    bytes_total = _coerce(counters.get("bytes_read_total"))
    max_bytes = limits.get("max_bytes_read_total")
    if max_bytes is not None and bytes_total > float(max_bytes):
        raise PolicyError(
            f"policy limits bytes_read_total to {max_bytes} but {bytes_total} recorded"
        )

    wall_time = _coerce(counters.get("wall_time_sec"))
    max_wall = limits.get("max_wall_time_sec")
    if max_wall is not None and wall_time > float(max_wall):
        raise PolicyError(
            f"policy limits wall_time_sec to {max_wall} but {wall_time} recorded"
        )

    access_events = _coerce(counters.get("access_events"))
    max_access_events = limits.get("max_access_events")
    if max_access_events is not None and access_events > float(max_access_events):
        raise PolicyError(
            f"policy limits access_events to {max_access_events} but {access_events} recorded"
        )
