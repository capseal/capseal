"""Lightweight receipt generation for agent bench rounds.

This module provides cryptographic binding for bench evaluation rounds
without pulling in the full capsule verification stack. Receipts enable:

1. Verifiable artifact chains across rounds
2. Reproducibility audits (trace_spec_hash pins configuration)
3. Tamper detection (statement_hash commits to all artifacts)

Receipt Chain Structure:
    round_receipt.json → statement_hash (pins artifacts + config)
    run_receipt.json → chain_hash (commits to all round statements)
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# Known artifacts to hash in each round directory
ROUND_ARTIFACTS = [
    "beta_posteriors.npz",
    "enn.npz",
    "enn.pt",
    "fusion.npz",
    "plan_out.json",
    "active_sampling_plan.json",
    "agent_results.csv",
    "metrics.json",
    "tallies.csv",
    "round_pre.json",
    "round_post.json",
]


def hash_file(path: Path) -> str:
    """Compute SHA256 hex digest of file contents.

    Args:
        path: Path to file.

    Returns:
        SHA256 hex string (64 chars).

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    """Compute SHA256 hex digest of bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_str(data: str) -> str:
    """Compute SHA256 hex digest of UTF-8 string."""
    return hash_bytes(data.encode("utf-8"))


def canonical_json(obj: Any) -> str:
    """Produce canonical JSON encoding (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _to_json_serializable(value: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def compute_trace_spec_hash(round_config: dict) -> str:
    """Compute hash of round configuration (pins reproducibility).

    The trace_spec_hash commits to all parameters that affect round execution:
    - grid_version: Which parameter grid was used
    - targets_per_round: K targets selected
    - episodes_per_budget_unit: Episodes per target
    - seed: RNG seed for this round
    - use_synthetic: Whether synthetic mode was used
    - acquisition_weights: (optional) w1, w2, tau, sigma

    Args:
        round_config: Dict with configuration parameters.

    Returns:
        SHA256 hex string.
    """
    # Extract and normalize known fields, converting numpy types
    grid_version = round_config.get("grid_version", "unknown")
    grid_version = _to_json_serializable(grid_version)
    if isinstance(grid_version, list):
        grid_version = str(grid_version)

    normalized = {
        "grid_version": grid_version,
        "targets_per_round": int(round_config.get("targets_per_round", 64)),
        "episodes_per_budget_unit": int(round_config.get("episodes_per_budget_unit", 1)),
        "seed": _to_json_serializable(round_config.get("seed")),
        "use_synthetic": bool(round_config.get("use_synthetic", False)),
    }

    # Include acquisition weights if present
    if "acquisition_weights" in round_config:
        aw = round_config["acquisition_weights"]
        normalized["acquisition_weights"] = {
            k: float(_to_json_serializable(v)) for k, v in aw.items()
        }
    elif any(k in round_config for k in ("tau", "sigma", "w1", "w2")):
        normalized["acquisition_weights"] = {
            "tau": float(round_config.get("tau", 0.2)),
            "sigma": float(round_config.get("sigma", 0.05)),
            "w1": float(round_config.get("w1", 1.0)),
            "w2": float(round_config.get("w2", 0.5)),
        }

    return hash_str(canonical_json(normalized))


def compute_statement_hash(
    trace_spec_hash: str,
    artifact_hashes: Dict[str, str],
) -> str:
    """Compute hash that commits to configuration AND artifacts.

    The statement_hash is the cryptographic commitment for the round.
    It binds the trace_spec (configuration) to all output artifacts.

    Args:
        trace_spec_hash: Hash of round configuration.
        artifact_hashes: Dict mapping artifact names to their hashes.

    Returns:
        SHA256 hex string.
    """
    # Sort artifacts for deterministic ordering
    sorted_artifacts = sorted(artifact_hashes.items())

    # Concatenate: trace_spec_hash || artifact1_name:hash || artifact2_name:hash || ...
    parts = [trace_spec_hash]
    for name, h in sorted_artifacts:
        parts.append(f"{name}:{h}")

    combined = "|".join(parts)
    return hash_str(combined)


def build_round_receipt(
    round_dir: Path,
    round_config: dict,
) -> dict:
    """Build a complete receipt for a single round.

    Scans the round directory for known artifacts, hashes each one,
    computes trace_spec_hash and statement_hash, extracts metrics snapshot.

    Args:
        round_dir: Path to round directory (e.g., run_dir/rounds/R0001_...).
        round_config: Configuration dict for this round.

    Returns:
        Receipt dict with:
        - round_id: Round identifier
        - trace_spec_hash: Hash of configuration
        - statement_hash: Hash of config + artifacts
        - artifact_hashes: Dict of artifact name → hash
        - metrics_snapshot: {tube_var_sum, tube_coverage, status}
        - created_at: ISO timestamp
    """
    round_dir = Path(round_dir)
    round_id = round_dir.name

    # Hash all present artifacts (round-local only)
    # Note: We don't hash parent-level artifacts (beta_posteriors.npz, enn.npz, etc.)
    # because they get updated each round. Each round's metrics.json captures
    # the state at that point in time.
    artifact_hashes = {}
    for artifact_name in ROUND_ARTIFACTS:
        artifact_path = round_dir / artifact_name
        if artifact_path.exists():
            artifact_hashes[artifact_name] = hash_file(artifact_path)

    # Compute hashes
    trace_spec_hash = compute_trace_spec_hash(round_config)
    statement_hash = compute_statement_hash(trace_spec_hash, artifact_hashes)

    # Extract metrics snapshot
    metrics_snapshot = {}
    metrics_path = round_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        tube = metrics.get("tube", {})
        metrics_snapshot = {
            "tube_var_sum": tube.get("tube_var_sum"),
            "tube_coverage": tube.get("tube_coverage"),
            "status": metrics.get("status"),
        }

    return {
        "schema": "round_receipt_v1",
        "round_id": round_id,
        "trace_spec_hash": trace_spec_hash,
        "statement_hash": statement_hash,
        "artifact_hashes": artifact_hashes,
        "metrics_snapshot": metrics_snapshot,
        "round_config": round_config,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def build_run_receipt(
    run_dir: Path,
    round_receipts: List[dict],
) -> dict:
    """Build a receipt for the entire run (chains all rounds).

    The chain_hash commits to the sequence of all round statement_hashes,
    providing a single hash that verifies the entire evaluation run.

    Args:
        run_dir: Path to run directory.
        round_receipts: List of round receipts (in order).

    Returns:
        Run receipt dict with:
        - run_id: Run identifier (from run_metadata.json or directory name)
        - rounds: List of {round_id, statement_hash}
        - chain_hash: SHA256 of concatenated statement_hashes
        - final_metrics: Metrics from last round
        - created_at: ISO timestamp
    """
    run_dir = Path(run_dir)

    # Extract run_id from metadata or directory name
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        run_id = metadata.get("run_uuid", run_dir.name)
    else:
        run_id = run_dir.name

    # Build chain hash from statement hashes
    statement_hashes = [r["statement_hash"] for r in round_receipts]
    chain_hash = hash_str("||".join(statement_hashes))

    # Extract round summaries
    rounds_summary = [
        {"round_id": r["round_id"], "statement_hash": r["statement_hash"]}
        for r in round_receipts
    ]

    # Final metrics from last round
    final_metrics = {}
    if round_receipts:
        final_metrics = round_receipts[-1].get("metrics_snapshot", {})

    return {
        "schema": "run_receipt_v1",
        "run_id": run_id,
        "rounds": rounds_summary,
        "chain_hash": chain_hash,
        "total_rounds": len(round_receipts),
        "final_metrics": final_metrics,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def verify_round_receipt(round_dir: Path) -> dict:
    """Verify a round receipt by recomputing all hashes.

    Args:
        round_dir: Path to round directory containing round_receipt.json.

    Returns:
        Verification result:
        - verified: bool (True if all hashes match)
        - mismatches: List of mismatch descriptions
        - artifacts_checked: Number of artifacts verified
    """
    round_dir = Path(round_dir)
    receipt_path = round_dir / "round_receipt.json"

    if not receipt_path.exists():
        return {
            "verified": False,
            "mismatches": ["round_receipt.json not found"],
            "artifacts_checked": 0,
        }

    with open(receipt_path) as f:
        receipt = json.load(f)

    mismatches = []
    artifacts_checked = 0

    # Recompute trace_spec_hash
    round_config = receipt.get("round_config", {})
    computed_trace_spec = compute_trace_spec_hash(round_config)
    if computed_trace_spec != receipt.get("trace_spec_hash"):
        mismatches.append(
            f"trace_spec_hash mismatch: stored={receipt.get('trace_spec_hash')[:16]}..., "
            f"computed={computed_trace_spec[:16]}..."
        )

    # Recompute artifact hashes (round-local only)
    stored_hashes = receipt.get("artifact_hashes", {})
    computed_hashes = {}

    for artifact_name, stored_hash in stored_hashes.items():
        # Skip parent-level artifacts (they change across rounds)
        if artifact_name.startswith("../"):
            continue

        artifact_path = round_dir / artifact_name

        if artifact_path.exists():
            computed_hash = hash_file(artifact_path)
            computed_hashes[artifact_name] = computed_hash
            artifacts_checked += 1

            if computed_hash != stored_hash:
                mismatches.append(
                    f"artifact {artifact_name}: stored={stored_hash[:16]}..., "
                    f"computed={computed_hash[:16]}..."
                )
        else:
            mismatches.append(f"artifact {artifact_name}: file not found")

    # Recompute statement_hash
    computed_statement = compute_statement_hash(computed_trace_spec, computed_hashes)
    if computed_statement != receipt.get("statement_hash"):
        mismatches.append(
            f"statement_hash mismatch: stored={receipt.get('statement_hash')[:16]}..., "
            f"computed={computed_statement[:16]}..."
        )

    return {
        "verified": len(mismatches) == 0,
        "mismatches": mismatches,
        "artifacts_checked": artifacts_checked,
        "round_id": receipt.get("round_id"),
    }


def verify_run_receipt(run_dir: Path) -> dict:
    """Verify a run receipt by checking the chain hash.

    Args:
        run_dir: Path to run directory containing run_receipt.json.

    Returns:
        Verification result:
        - verified: bool (True if chain hash matches)
        - mismatches: List of mismatch descriptions
        - rounds_verified: Number of rounds checked
    """
    run_dir = Path(run_dir)
    receipt_path = run_dir / "run_receipt.json"

    if not receipt_path.exists():
        return {
            "verified": False,
            "mismatches": ["run_receipt.json not found"],
            "rounds_verified": 0,
        }

    with open(receipt_path) as f:
        run_receipt = json.load(f)

    mismatches = []
    rounds_verified = 0

    # Load all round receipts
    rounds_dir = run_dir / "rounds"
    round_receipts = []

    for round_info in run_receipt.get("rounds", []):
        round_id = round_info["round_id"]
        round_dir = rounds_dir / round_id
        receipt_path = round_dir / "round_receipt.json"

        if not receipt_path.exists():
            mismatches.append(f"round {round_id}: round_receipt.json not found")
            continue

        with open(receipt_path) as f:
            round_receipt = json.load(f)

        # Check statement_hash matches
        if round_receipt.get("statement_hash") != round_info["statement_hash"]:
            mismatches.append(
                f"round {round_id}: statement_hash mismatch in run_receipt vs round_receipt"
            )

        round_receipts.append(round_receipt)
        rounds_verified += 1

    # Recompute chain hash
    if round_receipts:
        statement_hashes = [r["statement_hash"] for r in round_receipts]
        computed_chain = hash_str("||".join(statement_hashes))

        if computed_chain != run_receipt.get("chain_hash"):
            mismatches.append(
                f"chain_hash mismatch: stored={run_receipt.get('chain_hash')[:16]}..., "
                f"computed={computed_chain[:16]}..."
            )

    return {
        "verified": len(mismatches) == 0,
        "mismatches": mismatches,
        "rounds_verified": rounds_verified,
        "run_id": run_receipt.get("run_id"),
    }


def collect_round_dirs(run_dir: Path) -> List[Path]:
    """Collect all round directories in order.

    Args:
        run_dir: Path to run directory.

    Returns:
        Sorted list of round directory paths.
    """
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.exists():
        return []

    round_dirs = [d for d in rounds_dir.iterdir() if d.is_dir()]
    # Sort by round number (R0001, R0002, etc.)
    return sorted(round_dirs, key=lambda d: d.name)
