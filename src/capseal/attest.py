"""Typed attestation helpers for files traced under project_trace_v1."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from canonical.project_trace import sha256_bytes

from capseal.review_agent import load_trace_index

try:  # Optional numpy for tensor stats
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None  # type: ignore


SEVERITY_ORDER = {"info": 1, "warning": 2, "error": 3}


def _severity_meets(severity: str, threshold: str) -> bool:
    return SEVERITY_ORDER.get(severity, 0) >= SEVERITY_ORDER.get(threshold, 0)


def _match_globs(patterns: Iterable[str], candidates: Iterable[str], root: Path) -> list[str]:
    if not patterns:
        return []
    matched = set()
    candidate_set = set(candidates)
    for pattern in patterns:
        for match in root.glob(pattern):
            rel = match.relative_to(root).as_posix()
            if rel in candidate_set:
                matched.add(rel)
    return sorted(matched)


def _fingerprint(*parts: str) -> str:
    return sha256_bytes("|".join(parts).encode())


def _sanitize_filename(rel_path: str) -> str:
    return rel_path.replace("/", "__")


@dataclass
class AttestationViolation:
    severity: str
    message: str
    file_path: str
    file_hash: str
    tracer: str

    def fingerprint(self) -> str:
        return _fingerprint(self.file_path, self.file_hash, self.tracer, self.message)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "severity": self.severity,
            "message": self.message,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "tracer": self.tracer,
            "fingerprint": self.fingerprint(),
        }
        return data


@dataclass
class ColumnStats:
    name: str
    value_kind: str = "unknown"
    null_count: int = 0
    count: int = 0
    min_value: float | None = None
    max_value: float | None = None
    sum_value: float = 0.0
    sum_sq: float = 0.0
    min_length: int | None = None
    max_length: int | None = None
    unique_strings: list[str] = field(default_factory=list)

    def add(self, value: Any) -> None:
        if value is None:
            self.null_count += 1
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if math.isnan(value) or math.isinf(value):
                self.null_count += 1
                return
            if self.value_kind == "string":
                # Already string typed – treat as null to avoid mixing types
                self.null_count += 1
                return
            if self.value_kind in {"unknown", "int"} and isinstance(value, float) and not value.is_integer():
                self.value_kind = "float"
            elif self.value_kind == "unknown":
                self.value_kind = "int" if isinstance(value, int) else "float"
            elif self.value_kind == "int" and not isinstance(value, int):
                self.value_kind = "float"
            numeric = float(value)
            self.count += 1
            self.sum_value += numeric
            self.sum_sq += numeric * numeric
            if self.min_value is None or numeric < self.min_value:
                self.min_value = numeric
            if self.max_value is None or numeric > self.max_value:
                self.max_value = numeric
            return

        # Treat the rest as strings
        text = str(value)
        if self.value_kind in {"unknown", "int", "float"}:
            self.value_kind = "string"
        self.count += 1
        length = len(text)
        if self.min_length is None or length < self.min_length:
            self.min_length = length
        if self.max_length is None or length > self.max_length:
            self.max_length = length
        if len(self.unique_strings) < 5 and text not in self.unique_strings:
            self.unique_strings.append(text)

    def add_null(self) -> None:
        self.null_count += 1

    def mean(self) -> float | None:
        if self.value_kind not in {"int", "float"} or self.count == 0:
            return None
        return self.sum_value / self.count

    def stddev(self) -> float | None:
        if self.value_kind not in {"int", "float"} or self.count == 0:
            return None
        mean = self.mean()
        if mean is None:
            return None
        variance = max(self.sum_sq / self.count - mean * mean, 0.0)
        return math.sqrt(variance)

    def to_dict(self, row_count: int) -> dict[str, Any]:
        data: dict[str, Any] = {
            "type": self.value_kind if self.value_kind != "unknown" else "string",
            "values": self.count,
            "nulls": self.null_count,
            "null_fraction": (self.null_count / row_count) if row_count else 0.0,
        }
        if self.value_kind in {"int", "float"}:
            data.update({
                "min": self.min_value,
                "max": self.max_value,
                "mean": self.mean(),
                "stddev": self.stddev(),
            })
        else:
            if self.min_length is not None:
                data["min_length"] = self.min_length
            if self.max_length is not None:
                data["max_length"] = self.max_length
            if self.unique_strings:
                data["sample_values"] = self.unique_strings
        return data


class TabularTracer:
    name = "tabular_v1"
    SUPPORTED_SUFFIXES = {".csv", ".tsv", ".jsonl", ".ndjson"}

    def process(self, file_path: Path, rel_path: str, config: dict[str, Any]) -> tuple[dict, list[AttestationViolation]]:
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported tabular format for {rel_path}: {file_path.suffix}")

        expected_columns = set(config.get("columns", {}).keys())
        stats: dict[str, ColumnStats] = {col: ColumnStats(col) for col in expected_columns}
        rows = 0
        sample_rows: list[str] = []

        def _record_row(row_map: dict[str, Any]) -> None:
            nonlocal rows
            rows += 1
            present = set(row_map.keys())
            for column in present:
                stats.setdefault(column, ColumnStats(column)).add(_normalize_scalar(row_map[column]))
            for column in stats:
                if column not in present:
                    stats[column].add_null()
            if len(sample_rows) < 32:
                sample_rows.append(json.dumps(row_map, sort_keys=True))

        if suffix in {".csv", ".tsv"}:
            import csv

            delimiter = "\t" if suffix == ".tsv" else ","
            with open(file_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                headers = reader.fieldnames or []
                for header in headers:
                    stats.setdefault(header, ColumnStats(header))
                for row in reader:
                    _record_row(row)
        else:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise ValueError(f"JSONL row is not an object: {rel_path}")
                    _record_row(obj)

        stats_dict = {
            column: stat.to_dict(rows)
            for column, stat in sorted(stats.items())
        }
        sample_hash = sha256_bytes("".join(sample_rows).encode()) if sample_rows else ""
        table_stats = {
            "row_count": rows,
            "columns": stats_dict,
            "sample_hash": sample_hash,
        }

        violations = self._check_violations(stats, rows, rel_path, config)
        return table_stats, violations

    def _check_violations(
        self,
        column_stats: dict[str, ColumnStats],
        row_count: int,
        rel_path: str,
        config: dict[str, Any],
    ) -> list[AttestationViolation]:
        violations: list[AttestationViolation] = []
        file_hash = ""
        max_missing_global = config.get("max_missing_frac")
        column_cfg = config.get("columns", {})

        for column_name, stat in column_stats.items():
            stats_dict = stat.to_dict(row_count)
            missing_frac = stats_dict["null_fraction"]
            if max_missing_global is not None and missing_frac > max_missing_global:
                violations.append(AttestationViolation(
                    severity="warning",
                    message=f"Column {column_name} missing fraction {missing_frac:.3f} exceeds {max_missing_global}",
                    file_path=rel_path,
                    file_hash=file_hash,
                    tracer=self.name,
                ))

        for column_name, checks in column_cfg.items():
            stat = column_stats.get(column_name)
            if stat is None or stat.count == 0:
                violations.append(AttestationViolation(
                    severity="error",
                    message=f"Column {column_name} missing from table",
                    file_path=rel_path,
                    file_hash=file_hash,
                    tracer=self.name,
                ))
                continue
            stats_dict = stat.to_dict(row_count)
            min_allowed = checks.get("min")
            max_allowed = checks.get("max")
            if min_allowed is not None and stats_dict.get("min") is not None and stats_dict["min"] < min_allowed:
                violations.append(AttestationViolation(
                    severity="error",
                    message=f"Column {column_name} min {stats_dict['min']} < {min_allowed}",
                    file_path=rel_path,
                    file_hash=file_hash,
                    tracer=self.name,
                ))
            if max_allowed is not None and stats_dict.get("max") is not None and stats_dict["max"] > max_allowed:
                violations.append(AttestationViolation(
                    severity="error",
                    message=f"Column {column_name} max {stats_dict['max']} > {max_allowed}",
                    file_path=rel_path,
                    file_hash=file_hash,
                    tracer=self.name,
                ))
            column_missing = checks.get("max_missing_frac")
            if column_missing is not None and stats_dict["null_fraction"] > column_missing:
                violations.append(AttestationViolation(
                    severity="warning",
                    message=(
                        f"Column {column_name} missing fraction {stats_dict['null_fraction']:.3f} exceeds {column_missing}"
                    ),
                    file_path=rel_path,
                    file_hash=file_hash,
                    tracer=self.name,
                ))

        for violation in violations:
            if not violation.file_hash:
                violation.file_hash = ""
        return violations


class TensorTracer:
    name = "tensor_v1"

    def process(self, file_path: Path, rel_path: str, config: dict[str, Any]) -> tuple[dict, list[AttestationViolation]]:
        if np is None:  # pragma: no cover - requires numpy
            raise RuntimeError("numpy is required for tensor attestation (install numpy)")
        if file_path.suffix.lower() != ".npy":
            raise ValueError(f"Only .npy tensors are supported ({rel_path})")

        array = np.load(file_path, allow_pickle=False)
        is_numeric = np.issubdtype(array.dtype, np.number)
        stats = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "size": int(array.size),
            "min": float(np.min(array)) if array.size and is_numeric else None,
            "max": float(np.max(array)) if array.size and is_numeric else None,
            "mean": float(np.mean(array)) if array.size and is_numeric else None,
            "stddev": float(np.std(array)) if array.size and is_numeric else None,
            "nan_count": int(np.isnan(array).sum()) if is_numeric else 0,
            "inf_count": int(np.isinf(array).sum()) if is_numeric else 0,
        }

        violations: list[AttestationViolation] = []
        file_hash = ""
        max_nan = config.get("max_nan")
        max_inf = config.get("max_inf")
        min_allowed = config.get("min")
        max_allowed = config.get("max")
        if max_nan is not None and stats["nan_count"] > max_nan:
            violations.append(AttestationViolation(
                severity="error",
                message=f"Tensor has {stats['nan_count']} NaNs (limit {max_nan})",
                file_path=rel_path,
                file_hash=file_hash,
                tracer=self.name,
            ))
        if max_inf is not None and stats["inf_count"] > max_inf:
            violations.append(AttestationViolation(
                severity="error",
                message=f"Tensor has {stats['inf_count']} inf values (limit {max_inf})",
                file_path=rel_path,
                file_hash=file_hash,
                tracer=self.name,
            ))
        if min_allowed is not None and stats["min"] is not None and stats["min"] < min_allowed:
            violations.append(AttestationViolation(
                severity="warning",
                message=f"Tensor min {stats['min']} < {min_allowed}",
                file_path=rel_path,
                file_hash=file_hash,
                tracer=self.name,
            ))
        if max_allowed is not None and stats["max"] is not None and stats["max"] > max_allowed:
            violations.append(AttestationViolation(
                severity="warning",
                message=f"Tensor max {stats['max']} > {max_allowed}",
                file_path=rel_path,
                file_hash=file_hash,
                tracer=self.name,
            ))

        for violation in violations:
            violation.file_hash = ""
        return stats, violations


TRACERS = {
    "tabular_v1": TabularTracer(),
    "tensor_v1": TensorTracer(),
}


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"na", "nan", "null", "none"}:
            return None
        try:
            return int(stripped)
        except ValueError:
            try:
                return float(stripped)
            except ValueError:
                return stripped
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    return str(value)


def load_profile(profile_path: Path, profile_id: str) -> dict[str, Any]:
    data = json.loads(profile_path.read_text())
    profiles = data.get("profiles", {})
    if profile_id not in profiles:
        raise KeyError(f"Profile {profile_id} not found in {profile_path}")
    return profiles[profile_id]


def run_attestation(
    run_path: Path,
    project_dir: Path,
    profile_id: str,
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    trace_index = load_trace_index(run_path)
    commitments_path = run_path / "commitments.json"
    trace_root = ""
    if commitments_path.exists():
        trace_root = json.loads(commitments_path.read_text()).get("head_T", "")
    attest_dir = run_path / "attestations" / profile_id
    attest_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema": "attestation_summary_v1",
        "profile": profile_id,
        "trace_root": trace_root,
        "results": [],
        "violations": [],
    }
    stats_counts: dict[str, dict[str, int]] = {}

    for tracer_name, tracer in TRACERS.items():
        tracer_cfg = profile.get(tracer_name)
        if not tracer_cfg:
            continue
        include_patterns = tracer_cfg.get("include", [])
        matched_paths = _match_globs(include_patterns, trace_index.files_by_path.keys(), project_dir)
        tracer_dir = attest_dir / tracer_name
        tracer_dir.mkdir(parents=True, exist_ok=True)
        counts = stats_counts.setdefault(tracer_name, {"files": 0, "violations": 0})
        for rel_path in matched_paths:
            file_meta = trace_index.files_by_path.get(rel_path)
            if not file_meta:
                continue
            file_abs = project_dir / rel_path
            if not file_abs.exists():
                raise FileNotFoundError(f"File {rel_path} missing under {project_dir}")
            stats, violations = tracer.process(file_abs, rel_path, tracer_cfg)
            for violation in violations:
                violation.file_hash = file_meta.file_hash if file_meta else ""
            attestation = {
                "schema": f"attestation_{tracer_name}",
                "trace_root": summary["trace_root"],
                "profile": profile_id,
                "file_path": rel_path,
                "file_hash": file_meta.file_hash if file_meta else "",
                "stats": stats,
                "violations": [v.to_dict() for v in violations],
            }
            file_name = _sanitize_filename(rel_path) + ".json"
            att_path = tracer_dir / file_name
            with open(att_path, "w") as f:
                json.dump(attestation, f, indent=2, sort_keys=True)
            counts["files"] += 1
            counts["violations"] += len(violations)
            summary["results"].append({
                "file_path": rel_path,
                "file_hash": file_meta.file_hash if file_meta else "",
                "tracer": tracer_name,
                "attestation_path": str(att_path.relative_to(run_path)),
                "num_violations": len(violations),
            })
            for violation in violations:
                entry = violation.to_dict()
                summary["violations"].append(entry)

    summary_path = attest_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    return summary, stats_counts


def load_summary(run_path: Path, profile_id: str) -> dict[str, Any]:
    summary_path = run_path / "attestations" / profile_id / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Attestation summary not found: {summary_path}")
    return json.loads(summary_path.read_text())


def diff_summaries(base: dict[str, Any], head: dict[str, Any]) -> dict[str, Any]:
    base_map = {v["fingerprint"]: v for v in base.get("violations", [])}
    head_map = {v["fingerprint"]: v for v in head.get("violations", [])}
    new = [head_map[k] for k in head_map.keys() - base_map.keys()]
    resolved = [base_map[k] for k in base_map.keys() - head_map.keys()]
    unchanged = [head_map[k] for k in head_map.keys() & base_map.keys()]
    return {
        "new": new,
        "resolved": resolved,
        "unchanged": unchanged,
    }
