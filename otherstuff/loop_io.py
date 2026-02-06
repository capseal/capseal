#!/usr/bin/env python3
"""I/O utilities for the adaptive sampling loop.

Handles artifact directory structure, copying, and report generation.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shutil
import sys
import uuid
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np

from loop_metrics import RoundMetrics, compute_all_fingerprints


# ---------------------------------------------------------------------------
# Run directory initialization
# ---------------------------------------------------------------------------


def init_run_dir(
    run_dir: Path,
    base_seed: int,
    seed_mode: str,
    grid_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Initialize a run directory with required artifacts and metadata.
    
    Creates:
      - rounds/ subdirectory
      - beta_posteriors.npz with Beta(1,1) priors if missing
      - run_metadata.json with run_uuid and config
    
    Args:
        run_dir: The run directory to initialize.
        base_seed: Base seed for reproducibility.
        seed_mode: Seed mode ('fixed', 'increment', 'random').
        grid_path: Optional path to grid.npz for determining n_points.
    
    Returns:
        Dict with run_uuid and initialization status.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create rounds directory
    rounds_dir = run_dir / "rounds"
    rounds_dir.mkdir(exist_ok=True)
    
    # Load or create run metadata
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        run_uuid = metadata.get("run_uuid")
        print(f"[Init] Resuming run: {run_uuid}")
    else:
        run_uuid = str(uuid.uuid4())[:8]
        metadata = {
            "run_uuid": run_uuid,
            "base_seed": base_seed,
            "seed_mode": seed_mode,
            "created_at": datetime.now().isoformat(),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[Init] New run created: {run_uuid}")
    
    # Initialize beta_posteriors.npz if missing
    beta_path = run_dir / "beta_posteriors.npz"
    if not beta_path.exists():
        # Determine grid size from grid.npz or use default
        n_points = 4000  # Default
        
        if grid_path and grid_path.exists():
            grid_data = np.load(grid_path)
            if "x" in grid_data:
                n_points = len(grid_data["x"])
            elif "points" in grid_data:
                n_points = len(grid_data["points"])
        elif (run_dir / "grid.npz").exists():
            grid_data = np.load(run_dir / "grid.npz")
            if "x" in grid_data:
                n_points = len(grid_data["x"])
            elif "points" in grid_data:
                n_points = len(grid_data["points"])
        
        # Initialize with Beta(1,1) uniform priors (no evidence yet)
        alpha = np.ones(n_points, dtype=np.float64)
        beta = np.ones(n_points, dtype=np.float64)
        np.savez(beta_path, alpha=alpha, beta=beta)
        print(f"[Init] Created beta_posteriors.npz with {n_points} points (Beta(1,1) priors)")
    else:
        print(f"[Init] Using existing beta_posteriors.npz")
    
    return {
        "run_uuid": run_uuid,
        "metadata_path": metadata_path,
        "beta_initialized": not beta_path.exists(),
    }


def get_next_round_num(run_dir: Path) -> int:
    """Get the next round number based on existing complete rounds.
    
    Scans the rounds directory and returns max(complete_round_nums) + 1.
    A round is complete if it has a metrics.json file.
    
    Returns 1 if no complete rounds exist.
    """
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.exists():
        return 1
    
    max_complete = 0
    for d in rounds_dir.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith("R"):
            continue
        
        # Check if round is complete
        if not (d / "metrics.json").exists():
            continue
        
        # Parse round number
        try:
            parts = d.name.split("_", 1)
            round_num = int(parts[0][1:])  # Remove 'R' prefix
            max_complete = max(max_complete, round_num)
        except (ValueError, IndexError):
            continue
    
    return max_complete + 1


def get_run_uuid(run_dir: Path) -> Optional[str]:
    """Get the run_uuid from run_metadata.json, if it exists."""
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            with open(metadata_path) as f:
                return json.load(f).get("run_uuid")
        except (json.JSONDecodeError, IOError):
            pass
    return None


# ---------------------------------------------------------------------------
# Round directory management
# ---------------------------------------------------------------------------


def get_round_dir(base_dir: Path, round_id: str) -> Path:
    """Get the directory for a specific round."""
    return base_dir / "rounds" / round_id


def create_round_dir(base_dir: Path, round_id: str) -> Path:
    """Create a round directory and return its path."""
    round_dir = get_round_dir(base_dir, round_id)
    round_dir.mkdir(parents=True, exist_ok=True)
    return round_dir


def copy_artifacts_to_round(
    run_dir: Path,
    round_dir: Path,
    artifacts: List[str],
) -> Dict[str, bool]:
    """Copy artifacts from run_dir to round_dir."""
    results = {}
    for artifact in artifacts:
        src = run_dir / artifact
        dst = round_dir / artifact
        if src.exists():
            shutil.copy2(src, dst)
            results[artifact] = True
        else:
            results[artifact] = False
    return results


def copy_plan_as_input(run_dir: Path, round_dir: Path) -> bool:
    """Copy the current sampling plan as plan_in.json."""
    src = run_dir / "active_sampling_plan.json"
    dst = round_dir / "plan_in.json"
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False


def copy_plan_as_output(run_dir: Path, round_dir: Path) -> bool:
    """Copy the new sampling plan as plan_out.json."""
    src = run_dir / "active_sampling_plan.json"
    dst = round_dir / "plan_out.json"
    if src.exists():
        shutil.copy2(src, dst)
        return True
    return False


def copy_tallies(src_path: Path, round_dir: Path) -> bool:
    """Copy tallies CSV to round directory."""
    dst = round_dir / "tallies.csv"
    if src_path.exists():
        shutil.copy2(src_path, dst)
        return True
    return False


# ---------------------------------------------------------------------------
# JSON serialization (deterministic)
# ---------------------------------------------------------------------------


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # Round to 8 decimal places for reproducibility
            return round(float(obj), 8)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_json(data: Dict[str, Any], path: Path, atomic: bool = True) -> None:
    """Save JSON with deterministic formatting and atomic writes.
    
    For atomic=True (default), uses the safe pattern:
      1. Write to temp file with unique name
      2. Flush and fsync to ensure data is on disk
      3. Atomic rename via os.replace()
      4. Fsync directory to persist rename (crash-safe)
    
    This ensures that `path` either contains the old data or the new data,
    never a partial/corrupted file, even on crash or power loss.
    
    Args:
        data: Data to save.
        path: Target path.
        atomic: If True, use atomic write pattern (default True).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Encode once to avoid partial writes
    encoded = json.dumps(data, cls=NumpyEncoder, indent=2, sort_keys=True)
    
    if atomic:
        # Use unique temp file name to avoid collisions
        tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
        
        with open(tmp_path, "w") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())  # Ensure data is on disk before rename
        
        # Atomic rename (POSIX guarantees this is atomic on same filesystem)
        os.replace(tmp_path, path)
        
        # Fsync directory to persist the rename across power loss
        try:
            dir_fd = os.open(path.parent, os.O_DIRECTORY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            pass  # O_DIRECTORY not available on all platforms
    else:
        with open(path, "w") as f:
            f.write(encoded)


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def _fsync_dir(dir_path: Path) -> None:
    """Fsync a directory to persist renames across power loss."""
    try:
        dir_fd = os.open(dir_path, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except (OSError, AttributeError):
        pass  # O_DIRECTORY not available on all platforms


def save_npz_atomic(path: Path, **arrays) -> None:
    """Atomic write for NPZ files: temp -> fsync -> replace -> fsync dir.
    
    CRITICAL: temp file MUST end in .npz - np.savez appends .npz if missing.
    
    Args:
        path: Target path (must end in .npz).
        **arrays: Arrays to save in the NPZ file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # CRITICAL: temp file MUST end in .npz - np.savez appends .npz if missing
    tmp = path.with_name(f".{path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}.npz")
    
    np.savez(tmp, **arrays)
    # Reopen to fsync (np.savez closes the file)
    with open(tmp, 'r+b') as f:
        f.flush()
        os.fsync(f.fileno())
    
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def save_csv_atomic(path: Path, rows: List[Dict], columns: List[str]) -> None:
    """Atomic write for CSV files.
    
    Args:
        path: Target path (should end in .csv).
        rows: List of dicts, each representing a row.
        columns: Column names (determines order and header).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Temp file ends in .csv for consistency
    tmp = path.with_name(f".{path.stem}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}.csv")
    
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    
    os.replace(tmp, path)
    _fsync_dir(path.parent)


# ---------------------------------------------------------------------------
# Metrics file management
# ---------------------------------------------------------------------------


def save_metrics(metrics: RoundMetrics, round_dir: Path) -> Path:
    """Save metrics.json to round directory."""
    path = round_dir / "metrics.json"
    save_json(metrics.to_dict(), path)
    return path


def save_fingerprints(round_dir: Path) -> Path:
    """Compute and save fingerprints.json for all artifacts."""
    fingerprints = compute_all_fingerprints(round_dir)
    path = round_dir / "fingerprints.json"
    save_json(fingerprints, path)
    return path


# ---------------------------------------------------------------------------
# Summary CSV management
# ---------------------------------------------------------------------------


def append_to_summary_csv(
    base_dir: Path,
    metrics: RoundMetrics,
    run_uuid: Optional[str] = None,
) -> Path:
    """Append a row to summary.csv.
    
    Args:
        base_dir: Run directory containing summary.csv.
        metrics: RoundMetrics to append.
        run_uuid: Optional run UUID for series identification.
    
    Returns:
        Path to summary.csv.
        
    Note:
        Checks for duplicate round_ids and refuses to append if a row
        with the same round_id already exists (to prevent resume bugs).
    """
    csv_path = base_dir / "summary.csv"
    row = metrics.summary_row()
    
    # Add run_uuid if provided
    if run_uuid:
        row["run_uuid"] = run_uuid

    # Define column order (current schema)
    columns = [
        "run_uuid", "round_id", "seed", "tube_coverage", "tube_var", "tube_var_delta",
        "tube_var_delta_prev", "tube_var_delta_baseline",
        "targets_selected", "total_budget", "beta_std_mean_tube",
        "epistemic_mean_tube", "aleatoric_mean_tube", "enn_changed",
        "fusion_changed", "status",
    ]
    
    # Ensure run_uuid column has a value even if not provided
    if "run_uuid" not in row or not row["run_uuid"]:
        row["run_uuid"] = ""

    # Check if file exists and if header matches current schema
    write_header = not csv_path.exists()
    existing_round_ids = set()
    
    if csv_path.exists():
        # Check if header matches - if not, backup and start fresh
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
                if existing_header != columns:
                    # Header mismatch - backup old file
                    backup_path = csv_path.with_suffix(".csv.bak")
                    shutil.copy2(csv_path, backup_path)
                    print(f"[WARNING] CSV schema changed. Old file backed up to {backup_path}")
                    write_header = True
                    # Truncate and rewrite
                    csv_path.unlink()
                else:
                    # Collect existing round_ids to check for duplicates
                    for csv_row in reader:
                        if len(csv_row) > 1:  # round_id is at index 1
                            existing_round_ids.add(csv_row[1])
            except StopIteration:
                # Empty file
                write_header = True

    # Check for duplicate round_id - this is a bug, not a recoverable condition
    if row["round_id"] in existing_round_ids:
        raise ValueError(
            f"BUG: Attempted to append duplicate round_id '{row['round_id']}' to summary.csv. "
            f"This indicates a bug in round numbering logic - round_id should never be minted twice."
        )

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return csv_path


def load_summary_csv(base_dir: Path) -> List[Dict[str, Any]]:
    """Load summary.csv as list of dicts."""
    csv_path = base_dir / "summary.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def format_target(target: Dict[str, Any], idx: int) -> str:
    """Format a single target for the report."""
    xy = target.get("xy", [0, 0])
    return (
        f"  {idx+1}. idx={target.get('grid_idx', '?'):5d}  "
        f"xy=({xy[0]:+.3f}, {xy[1]:+.3f})  "
        f"score={target.get('score', 0):.3f}  "
        f"budget={target.get('budget', 0)}"
    )


def generate_report_md(
    metrics: RoundMetrics,
    seed_mode: str,
    commands_run: List[str],
) -> str:
    """Generate a short (~20 line) markdown report."""
    lines = []

    # Header
    lines.append(f"# Round {metrics.round_id}")
    lines.append(f"Timestamp: {metrics.timestamp}")
    lines.append(f"Seed: {metrics.seed} (mode: {seed_mode})")
    lines.append("")

    # Commands
    lines.append("## Commands")
    for cmd in commands_run[:3]:  # Limit to 3
        lines.append(f"- `{cmd}`")
    lines.append("")

    # Status
    status_emoji = {"IMPROVED": "+", "REGRESSED": "!", "NO_CHANGE": "=", "FIRST_ROUND": "*"}
    lines.append(f"## Status: [{status_emoji.get(metrics.status, '?')}] {metrics.status}")
    lines.append("")

    # Scoreboard
    lines.append("## Scoreboard")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| tube_coverage | {metrics.tube.tube_coverage:.3f} |")
    lines.append(f"| tube_var | {metrics.tube.tube_var_sum:.6f} |")
    delta_str = f"{metrics.tube.tube_var_delta:+.6f}" if metrics.tube.tube_var_delta is not None else "N/A (first round)"
    lines.append(f"| tube_var_delta | {delta_str} |")
    lines.append(f"| targets_selected | {metrics.plan.targets_selected} |")
    lines.append(f"| total_budget | {metrics.plan.total_budget} |")
    lines.append("")

    # Changes
    lines.append("## Changes")
    enn_mark = "CHANGED" if metrics.model_change.enn_changed else "same"
    fusion_mark = "CHANGED" if metrics.model_change.fusion_changed else "same"
    lines.append(f"- enn.npz: [{enn_mark}]")
    lines.append(f"- fusion.npz: [{fusion_mark}]")
    lines.append("")

    # Top targets
    if metrics.plan.top_targets:
        lines.append("## Top Targets")
        for i, t in enumerate(metrics.plan.top_targets[:3]):
            lines.append(format_target(t, i))
    lines.append("")

    # Warnings
    warnings = []
    if metrics.fusion.status == "SUSPECT":
        warnings.append("FUSION: ENN changed but fusion byte-identical - check fusion!")
    if metrics.status == "REGRESSED":
        warnings.append("REGRESSED: tube_var increased this round")
    if not metrics.model_change.enn_changed and metrics.counts.sampled_points_new > 0:
        warnings.append("ENN: New data added but ENN unchanged - check training!")

    if warnings:
        lines.append("## Warnings")
        for w in warnings:
            lines.append(f"- {w}")

    return "\n".join(lines)


def save_report_md(
    metrics: RoundMetrics,
    round_dir: Path,
    seed_mode: str,
    commands_run: List[str],
) -> Path:
    """Generate and save report.md."""
    report = generate_report_md(metrics, seed_mode, commands_run)
    path = round_dir / "report.md"
    path.write_text(report)
    return path


# ---------------------------------------------------------------------------
# Stdout capture
# ---------------------------------------------------------------------------


class TeeWriter:
    """Write to both a file and the original stream."""

    def __init__(self, original: TextIO, capture: io.StringIO):
        self.original = original
        self.capture = capture

    def write(self, text: str) -> int:
        self.original.write(text)
        return self.capture.write(text)

    def flush(self) -> None:
        self.original.flush()
        self.capture.flush()


class CaptureOutput:
    """Context manager to capture stdout/stderr to a file while still printing."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.capture = io.StringIO()
        self._stdout_tee: Optional[TeeWriter] = None
        self._stderr_tee: Optional[TeeWriter] = None
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

    def __enter__(self) -> "CaptureOutput":
        self._stdout_tee = TeeWriter(self._orig_stdout, self.capture)
        self._stderr_tee = TeeWriter(self._orig_stderr, self.capture)
        sys.stdout = self._stdout_tee  # type: ignore
        sys.stderr = self._stderr_tee  # type: ignore
        return self

    def __exit__(self, *args: Any) -> None:
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        # Write captured output to file
        self.log_path.write_text(self.capture.getvalue())


# ---------------------------------------------------------------------------
# Console output formatting
# ---------------------------------------------------------------------------


def print_round_summary(metrics: RoundMetrics, verbose: bool = False) -> None:
    """Print a concise round summary to console."""
    status_symbol = {"IMPROVED": "+", "REGRESSED": "!", "NO_CHANGE": "=", "FIRST_ROUND": "*"}

    print()
    print("=" * 60)
    print(f" ROUND {metrics.round_id} SUMMARY ".center(60, "="))
    print("=" * 60)
    print(f"  seed:           {metrics.seed}")
    print(f"  status:         [{status_symbol.get(metrics.status, '?')}] {metrics.status}")
    print(f"  tube_coverage:  {metrics.tube.tube_coverage:.3f}")
    print(f"  tube_var:       {metrics.tube.tube_var_sum:.6f}")
    delta_display = f"{metrics.tube.tube_var_delta:+.6f}" if metrics.tube.tube_var_delta is not None else "N/A"
    print(f"  tube_var_delta: {delta_display}")
    print(f"  enn_changed:    {metrics.model_change.enn_changed}")
    print(f"  fusion_changed: {metrics.model_change.fusion_changed}")
    print("-" * 60)
    print(f"  Top 3 targets:")
    for i, t in enumerate(metrics.plan.top_targets[:3]):
        xy = t.get("xy", [0, 0])
        print(f"    {i+1}. idx={t.get('grid_idx', '?'):5d} "
              f"xy=({xy[0]:+.3f},{xy[1]:+.3f}) "
              f"score={t.get('score', 0):.3f} "
              f"budget={t.get('budget', 0)}")
    print("=" * 60)
    print()

    # Warnings
    if metrics.fusion.status == "SUSPECT":
        print("WARNING: Fusion appears unchanged despite ENN change!")
    if metrics.status == "REGRESSED":
        print("WARNING: tube_var increased this round!")


# ---------------------------------------------------------------------------
# Previous round loading
# ---------------------------------------------------------------------------


def get_previous_round_info(
    base_dir: Path, 
    exclude_round_id: Optional[str] = None,
    current_round_num: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Load info from the most recent COMPLETE round (has metrics.json).

    Args:
        base_dir: The run directory containing the 'rounds' subdirectory.
        exclude_round_id: Optional round ID to exclude (e.g., the current round being created).
        current_round_num: If provided, only consider rounds with round_num < current_round_num.
            This ensures we don't pick up rounds from previous loop series when starting fresh.

    Returns:
        Dict with round_dir, round_id, tube_var, sampled, coverage, or None if no valid previous round.

    Note: This function iterates through rounds in descending order by timestamp and returns 
    the first one that has a valid metrics.json. For round 1 of a new series, it will return
    None (no previous round) rather than picking up the last round from a different series.
    """
    rounds_dir = base_dir / "rounds"
    if not rounds_dir.exists():
        return None

    # Parse round directories to extract round number and timestamp
    def parse_round_dir(d: Path) -> Optional[Tuple[int, str, Path]]:
        """Parse round dir name like R0001_20260131_092722 -> (1, '20260131_092722', path)"""
        name = d.name
        if not name.startswith("R"):
            return None
        try:
            parts = name.split("_", 1)
            round_num = int(parts[0][1:])  # Remove 'R' prefix
            timestamp = parts[1] if len(parts) > 1 else ""
            return (round_num, timestamp, d)
        except (ValueError, IndexError):
            return None
    
    parsed_rounds = []
    for d in rounds_dir.iterdir():
        if d.is_dir():
            parsed = parse_round_dir(d)
            if parsed:
                parsed_rounds.append(parsed)
    
    # Sort by timestamp descending (newest first), then by round_num descending
    # This ensures we get the most recent round of each number
    parsed_rounds.sort(key=lambda x: (x[1], x[0]), reverse=True)
    
    if not parsed_rounds:
        return None

    # Find the first valid round that satisfies constraints
    for round_num, timestamp, round_dir in parsed_rounds:
        # Skip the excluded round if specified
        if exclude_round_id and round_dir.name == exclude_round_id:
            continue
        
        # If current_round_num specified, only consider rounds with smaller round_num
        # This prevents cross-series contamination when starting a new loop
        if current_round_num is not None and round_num >= current_round_num:
            continue

        metrics_path = round_dir / "metrics.json"
        if not metrics_path.exists():
            # Skip this round - it's incomplete (no metrics.json yet)
            continue

        try:
            metrics = load_json(metrics_path)
        except (json.JSONDecodeError, IOError):
            # Skip corrupted metrics files
            continue

        tube_data = metrics.get("tube", {})
        counts_data = metrics.get("counts", {})

        return {
            "round_dir": round_dir,
            "round_id": round_dir.name,
            "round_num": round_num,
            "tube_var": tube_data.get("tube_var_sum", 0),
            "sampled": counts_data.get("sampled_points_total", 0),
            "coverage": tube_data.get("tube_coverage", 0),
            "baseline_tube_var": tube_data.get("tube_var_baseline", None),
        }

    # No valid previous round found
    return None
