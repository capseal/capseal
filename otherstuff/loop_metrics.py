#!/usr/bin/env python3
"""Metrics computation for the adaptive sampling loop.

This module provides stable, comparable metrics across rounds.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Fingerprint computation
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> Optional[str]:
    """Compute SHA256 hash of a file."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_npz(path: Path) -> Optional[str]:
    """Compute fingerprint of npz arrays (deterministic)."""
    if not path.exists():
        return None
    try:
        data = np.load(path)
        keys = sorted(data.files)
        if not keys:
            return "empty"
        h = hashlib.sha256()
        for k in keys:
            arr = data[k]
            h.update(k.encode())
            h.update(arr.tobytes())
        return h.hexdigest()[:32]
    except Exception as e:
        return f"error:{e}"


def fingerprint_json(path: Path) -> Optional[str]:
    """Compute fingerprint of JSON file."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        # Serialize deterministically
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode()).hexdigest()[:32]
    except Exception as e:
        return f"error:{e}"


def compute_all_fingerprints(round_dir: Path) -> Dict[str, Optional[str]]:
    """Compute fingerprints for all artifacts in a round directory."""
    artifacts = {
        "plan_in.json": fingerprint_json,
        "tallies.csv": sha256_file,
        "beta_posteriors.npz": fingerprint_npz,
        "enn.pt": sha256_file,
        "enn.npz": fingerprint_npz,
        "fusion.npz": fingerprint_npz,
        "plan_out.json": fingerprint_json,
    }
    return {name: func(round_dir / name) for name, func in artifacts.items()}


# ---------------------------------------------------------------------------
# Array comparison
# ---------------------------------------------------------------------------


def array_diff_stats(
    arr1: np.ndarray, arr2: np.ndarray
) -> Dict[str, float]:
    """Compute L2 norm and max abs diff between two arrays."""
    diff = arr1.flatten() - arr2.flatten()
    return {
        "l2_norm": float(np.linalg.norm(diff)),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
    }


def compare_npz_arrays(
    path1: Path, path2: Path, array_name: str
) -> Optional[Dict[str, float]]:
    """Compare a specific array between two npz files."""
    if not path1.exists() or not path2.exists():
        return None
    try:
        data1 = np.load(path1)
        data2 = np.load(path2)
        if array_name not in data1.files or array_name not in data2.files:
            return None
        return array_diff_stats(data1[array_name], data2[array_name])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Uncertainty statistics
# ---------------------------------------------------------------------------


@dataclass
class ArrayStats:
    """Statistics for an array over a masked region."""
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0

    @classmethod
    def from_array(cls, arr: np.ndarray, mask: Optional[np.ndarray] = None) -> "ArrayStats":
        if mask is not None:
            arr = arr[mask]
        if len(arr) == 0:
            return cls()
        return cls(
            min=float(np.min(arr)),
            max=float(np.max(arr)),
            mean=float(np.mean(arr)),
            median=float(np.median(arr)),
            std=float(np.std(arr)),
        )


@dataclass
class TubeMetrics:
    """Metrics for the transition tube region.

    Metric definitions:
    - tube_var_sum: Sum of Beta variance over all points in the tube [0.4, 0.6].
      Formula: sum(alpha * beta / ((alpha + beta)^2 * (alpha + beta + 1)))
    - tube_var_mean: tube_var_sum / tube_points_total
    - tube_var_delta: prev_tube_var_sum - current_tube_var_sum (positive = improved)
      None if no previous round exists (first round).
    """
    tube_points_total: int = 0
    tube_sampled: int = 0
    tube_coverage: float = 0.0
    tube_var_sum: float = 0.0
    tube_var_mean: float = 0.0
    tube_var_delta: Optional[float] = None  # DEPRECATED: use delta_prev
    
    tube_var_delta_prev: Optional[float] = None # Improvement vs previous round
    tube_var_delta_baseline: Optional[float] = None # Improvement vs baseline
    tube_var_baseline: Optional[float] = None # The baseline value used

    beta_std: ArrayStats = field(default_factory=ArrayStats)
    epistemic: ArrayStats = field(default_factory=ArrayStats)
    aleatoric: ArrayStats = field(default_factory=ArrayStats)


def compute_tube_metrics(
    run_dir: Path,
    tube_low: float = 0.4,
    tube_high: float = 0.6,
    prev_tube_var: Optional[float] = None,
    baseline_tube_var: Optional[float] = None,
) -> TubeMetrics:
    """Compute comprehensive tube metrics."""
    beta_path = run_dir / "beta_posteriors.npz"
    enn_path = run_dir / "enn.npz"

    metrics = TubeMetrics()

    if not beta_path.exists() or not enn_path.exists():
        return metrics

    beta = np.load(beta_path)
    enn = np.load(enn_path)

    alpha, beta_arr = beta["alpha"], beta["beta"]
    q_enn = enn["q_enn"]

    # Beta variance
    beta_var = alpha * beta_arr / ((alpha + beta_arr) ** 2 * (alpha + beta_arr + 1.0))
    beta_std = np.sqrt(beta_var)

    # Tube mask
    in_tube = (q_enn >= tube_low) & (q_enn <= tube_high)
    sampled = (alpha + beta_arr) > 2.0

    metrics.tube_points_total = int(np.sum(in_tube))
    metrics.tube_sampled = int(np.sum(in_tube & sampled))
    metrics.tube_coverage = (
        metrics.tube_sampled / max(metrics.tube_points_total, 1)
    )

    metrics.tube_var_sum = float(np.sum(beta_var[in_tube]))
    metrics.tube_var_mean = metrics.tube_var_sum / max(metrics.tube_points_total, 1)

    # Compute delta vs previous round (if exists)
    if prev_tube_var is not None:
        metrics.tube_var_delta = prev_tube_var - metrics.tube_var_sum
        metrics.tube_var_delta_prev = prev_tube_var - metrics.tube_var_sum
    # Note: if prev_tube_var is None, delta fields stay None (not 0.0)
    
    # Compute baseline and delta vs baseline
    # SPEC: FIRST_ROUND has delta_prev=None AND delta_base=None
    # We store baseline_tube_var_sum separately for later rounds to use,
    # but FIRST_ROUND itself reports no delta.
    if baseline_tube_var is not None:
        # Use provided baseline (not first round)
        metrics.tube_var_baseline = baseline_tube_var
        metrics.tube_var_delta_baseline = baseline_tube_var - metrics.tube_var_sum
    else:
        # No baseline provided - this IS the baseline (first round of series)
        # Store baseline value for later rounds, but delta_baseline = None
        metrics.tube_var_baseline = metrics.tube_var_sum
        metrics.tube_var_delta_baseline = None  # NOT 0.0 - first round has no delta

    # Uncertainty stats over tube
    metrics.beta_std = ArrayStats.from_array(beta_std, in_tube)

    # Epistemic (MC dropout) if available
    if "var_epistemic" in enn.files:
        epi_std = np.sqrt(enn["var_epistemic"])
        metrics.epistemic = ArrayStats.from_array(epi_std, in_tube)

    # Aleatoric if available
    if "var_aleatoric" in enn.files:
        ale_std = np.sqrt(enn["var_aleatoric"])
        metrics.aleatoric = ArrayStats.from_array(ale_std, in_tube)

    return metrics


# ---------------------------------------------------------------------------
# Plan summary
# ---------------------------------------------------------------------------


@dataclass
class PlanSummary:
    """Summary of a sampling plan."""
    targets_selected: int = 0
    total_budget: int = 0
    budget_min: int = 0
    budget_max: int = 0
    budget_mean: float = 0.0

    score_min: float = 0.0
    score_max: float = 0.0
    score_mean: float = 0.0
    score_median: float = 0.0

    top_targets: List[Dict[str, Any]] = field(default_factory=list)


def compute_plan_summary(plan: Dict, k_top: int = 10) -> PlanSummary:
    """Compute summary statistics for a sampling plan."""
    summary = PlanSummary()

    indices = plan.get("selected_indices", [])
    points = plan.get("points", [])
    scores = plan.get("scores", [])
    budgets = plan.get("budget", [])
    components = plan.get("components", {})

    if not indices:
        return summary

    summary.targets_selected = len(indices)
    summary.total_budget = int(np.sum(budgets)) if budgets else 0

    if budgets:
        budgets_arr = np.array(budgets)
        summary.budget_min = int(np.min(budgets_arr))
        summary.budget_max = int(np.max(budgets_arr))
        summary.budget_mean = float(np.mean(budgets_arr))

    if scores:
        scores_arr = np.array(scores)
        summary.score_min = float(np.min(scores_arr))
        summary.score_max = float(np.max(scores_arr))
        summary.score_mean = float(np.mean(scores_arr))
        summary.score_median = float(np.median(scores_arr))

    # Top K targets
    if scores:
        sorted_idx = np.argsort(scores)[::-1][:k_top]
        for i in sorted_idx:
            target = {
                "grid_idx": indices[i] if i < len(indices) else -1,
                "xy": points[i] if i < len(points) else [0, 0],
                "score": scores[i] if i < len(scores) else 0.0,
                "budget": budgets[i] if i < len(budgets) else 0,
            }
            # Add component breakdown
            for comp_name in ["uncertainty", "tube", "epistemic", "disagreement", "fusion_delta"]:
                if comp_name in components and i < len(components[comp_name]):
                    target[comp_name] = components[comp_name][i]
            summary.top_targets.append(target)

    return summary


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


@dataclass
class Counts:
    """Point counts for a round."""
    sampled_points_total: int = 0
    sampled_points_new: int = 0
    total_candidates: int = 0
    tube_points_total: int = 0


def compute_counts(
    run_dir: Path,
    prev_sampled: int = 0,
    tube_low: float = 0.4,
    tube_high: float = 0.6,
) -> Counts:
    """Compute point counts."""
    counts = Counts()

    beta_path = run_dir / "beta_posteriors.npz"
    enn_path = run_dir / "enn.npz"
    grid_path = run_dir / "grid.npz"

    if grid_path.exists():
        grid = np.load(grid_path)
        counts.total_candidates = len(grid["x"])

    if beta_path.exists():
        beta = np.load(beta_path)
        alpha, beta_arr = beta["alpha"], beta["beta"]
        sampled = (alpha + beta_arr) > 2.0
        counts.sampled_points_total = int(np.sum(sampled))
        counts.sampled_points_new = counts.sampled_points_total - prev_sampled

    if enn_path.exists():
        enn = np.load(enn_path)
        q_enn = enn["q_enn"]
        in_tube = (q_enn >= tube_low) & (q_enn <= tube_high)
        counts.tube_points_total = int(np.sum(in_tube))

    return counts


# ---------------------------------------------------------------------------
# Model change detection
# ---------------------------------------------------------------------------


@dataclass
class ModelChange:
    """Track changes between rounds."""
    enn_changed: bool = False
    fusion_changed: bool = False

    enn_q_diff: Optional[Dict[str, float]] = None
    fusion_q_diff: Optional[Dict[str, float]] = None

    fingerprint_changes: Dict[str, Tuple[Optional[str], Optional[str]]] = field(
        default_factory=dict
    )


def compute_model_change(
    prev_round_dir: Optional[Path],
    curr_round_dir: Path,
) -> ModelChange:
    """Detect changes between rounds."""
    change = ModelChange()

    if prev_round_dir is None:
        # First round - everything is "new"
        change.enn_changed = True
        change.fusion_changed = True
        return change

    # Compare fingerprints
    prev_fp = compute_all_fingerprints(prev_round_dir)
    curr_fp = compute_all_fingerprints(curr_round_dir)

    for name in set(prev_fp.keys()) | set(curr_fp.keys()):
        if prev_fp.get(name) != curr_fp.get(name):
            change.fingerprint_changes[name] = (prev_fp.get(name), curr_fp.get(name))

    change.enn_changed = "enn.npz" in change.fingerprint_changes
    change.fusion_changed = "fusion.npz" in change.fingerprint_changes

    # Compute array diffs for q values
    change.enn_q_diff = compare_npz_arrays(
        prev_round_dir / "enn.npz",
        curr_round_dir / "enn.npz",
        "q_enn",
    )
    change.fusion_q_diff = compare_npz_arrays(
        prev_round_dir / "fusion.npz",
        curr_round_dir / "fusion.npz",
        "q_fused",
    )

    return change


# ---------------------------------------------------------------------------
# Fusion consistency
# ---------------------------------------------------------------------------


@dataclass
class FusionConsistency:
    """Check fusion vs ENN consistency."""
    max_diff: float = 0.0
    mean_diff: float = 0.0
    status: str = "OK"  # OK, NOOP_OK, SUSPECT


def check_fusion_consistency(
    run_dir: Path,
    tolerance: float = 0.01,
) -> FusionConsistency:
    """Check if fusion is meaningfully different from ENN."""
    result = FusionConsistency()

    enn_path = run_dir / "enn.npz"
    fusion_path = run_dir / "fusion.npz"

    if not enn_path.exists() or not fusion_path.exists():
        result.status = "MISSING"
        return result

    try:
        enn = np.load(enn_path)
        fusion = np.load(fusion_path)

        q_enn = enn["q_enn"]
        q_fused = fusion["q_fused"]

        diff = np.abs(q_fused - q_enn)
        result.max_diff = float(np.max(diff))
        result.mean_diff = float(np.mean(diff))

        if result.max_diff < tolerance and result.mean_diff < tolerance:
            result.status = "NOOP_OK"
        else:
            result.status = "OK"

    except Exception as e:
        result.status = f"ERROR:{e}"

    return result


# ---------------------------------------------------------------------------
# Progress status
# ---------------------------------------------------------------------------


def determine_status(
    tube_var_delta: Optional[float],
    tube_coverage: float,
    prev_coverage: float,
    eps_var: float = 0.001,
    eps_var_up: float = 0.01,
) -> str:
    """Determine round status: IMPROVED, REGRESSED, NO_CHANGE, or FIRST_ROUND.

    Status is determined primarily by tube_var_delta (the variance reduction metric):
    - FIRST_ROUND: No previous round to compare (delta is None)
    - IMPROVED: tube_var decreased significantly (delta > eps_var)
    - REGRESSED: tube_var increased significantly (delta < -eps_var_up)
    - NO_CHANGE: delta is within noise threshold

    Args:
        tube_var_delta: prev_tube_var_sum - current_tube_var_sum (positive = improved).
            None if no previous round (first round).
        tube_coverage: Current tube coverage ratio.
        prev_coverage: Previous tube coverage ratio.
        eps_var: Minimum delta to count as improved.
        eps_var_up: Minimum negative delta to count as regressed.

    Returns:
        Status string.
    """
    # First round has no delta to compare
    if tube_var_delta is None:
        return "FIRST_ROUND"

    # Status is determined by variance delta sign
    # Note: coverage_improved no longer overrides REGRESSED status
    # (to ensure status accurately reflects delta sign per Task C)
    if tube_var_delta >= eps_var:
        return "IMPROVED"
    elif tube_var_delta <= -eps_var_up:
        return "REGRESSED"
    else:
        return "NO_CHANGE"


# ---------------------------------------------------------------------------
# Full round metrics
# ---------------------------------------------------------------------------


@dataclass
class RoundMetrics:
    """Complete metrics for a single round."""
    round_id: str = ""
    timestamp: str = ""
    seed: int = 0

    counts: Counts = field(default_factory=Counts)
    tube: TubeMetrics = field(default_factory=TubeMetrics)
    plan: PlanSummary = field(default_factory=PlanSummary)
    model_change: ModelChange = field(default_factory=ModelChange)
    fusion: FusionConsistency = field(default_factory=FusionConsistency)

    status: str = "NO_CHANGE"
    fingerprints: Dict[str, Optional[str]] = field(default_factory=dict)
    
    # Audit fields
    posteriors_sha256: Optional[str] = None
    train_dataset_sha256: Optional[str] = None
    train_dataset_points: int = 0
    train_dataset_total_trials: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "round_id": self.round_id,
            "timestamp": self.timestamp,
            "seed": self.seed,
            "counts": asdict(self.counts),
            "tube": {
                **{k: v for k, v in asdict(self.tube).items()
                   if not isinstance(v, dict)},
                "beta_std": asdict(self.tube.beta_std),
                "epistemic": asdict(self.tube.epistemic),
                "aleatoric": asdict(self.tube.aleatoric),
            },
            "plan": {
                **{k: v for k, v in asdict(self.plan).items() if k != "top_targets"},
                "top_targets": self.plan.top_targets,
            },
            "model_change": {
                "enn_changed": self.model_change.enn_changed,
                "fusion_changed": self.model_change.fusion_changed,
                "enn_q_diff": self.model_change.enn_q_diff,
                "fusion_q_diff": self.model_change.fusion_q_diff,
                "fingerprint_changes": {
                    k: list(v) for k, v in self.model_change.fingerprint_changes.items()
                },
            },
            "fusion": asdict(self.fusion),
            "status": self.status,
            "fingerprints": self.fingerprints,
            "audit": {
                "posteriors_sha256": self.posteriors_sha256,
                "train_dataset_sha256": self.train_dataset_sha256,
                "train_dataset_points": self.train_dataset_points,
                "train_dataset_total_trials": self.train_dataset_total_trials,
            }
        }

    def summary_row(self) -> Dict[str, Any]:
        """Return a flat dict suitable for summary.csv.

        Column definitions:
        - tube_var: tube_var_sum (sum of Beta variance in tube)
        - tube_var_delta: prev_tube_var_sum - current_tube_var_sum (positive = improved)
          Empty string if first round (no previous).
        """
        # Handle None delta (first round) - write empty string for CSV
        delta_prev = self.tube.tube_var_delta_prev if self.tube.tube_var_delta_prev is not None else ""
        delta_base = self.tube.tube_var_delta_baseline if self.tube.tube_var_delta_baseline is not None else ""

        return {
            "round_id": self.round_id,
            "seed": self.seed,
            "tube_coverage": self.tube.tube_coverage,
            "tube_var": self.tube.tube_var_sum,
            "tube_var_delta": delta_prev, # Legacy column name for compatibility
            "tube_var_delta_prev": delta_prev,
            "tube_var_delta_baseline": delta_base,
            "targets_selected": self.plan.targets_selected,
            "total_budget": self.plan.total_budget,
            "beta_std_mean_tube": self.tube.beta_std.mean,
            "epistemic_mean_tube": self.tube.epistemic.mean,
            "aleatoric_mean_tube": self.tube.aleatoric.mean,
            "enn_changed": self.model_change.enn_changed,
            "fusion_changed": self.model_change.fusion_changed,
            "status": self.status,
        }


def compute_round_metrics(
    round_dir: Path,
    round_id: str,
    seed: int,
    prev_round_dir: Optional[Path] = None,
    prev_tube_var: Optional[float] = None,
    baseline_tube_var: Optional[float] = None,
    prev_sampled: int = 0,
    prev_coverage: float = 0.0,
) -> RoundMetrics:
    """Compute all metrics for a round."""
    metrics = RoundMetrics(
        round_id=round_id,
        timestamp=datetime.now().isoformat(),
        seed=seed,
    )

    # Counts
    metrics.counts = compute_counts(round_dir, prev_sampled=prev_sampled)

    # Tube metrics
    metrics.tube = compute_tube_metrics(
        round_dir, 
        prev_tube_var=prev_tube_var,
        baseline_tube_var=baseline_tube_var,
    )

    # Plan summary
    plan_path = round_dir / "plan_out.json"
    if plan_path.exists():
        with open(plan_path) as f:
            plan = json.load(f)
        metrics.plan = compute_plan_summary(plan)

    # Model change
    metrics.model_change = compute_model_change(prev_round_dir, round_dir)

    # Fusion consistency
    metrics.fusion = check_fusion_consistency(round_dir)

    # Status
    metrics.status = determine_status(
        metrics.tube.tube_var_delta,
        metrics.tube.tube_coverage,
        prev_coverage,
    )

    # Fingerprints
    metrics.fingerprints = compute_all_fingerprints(round_dir)
    
    # Audit fields
    import pandas as pd
    metrics.posteriors_sha256 = fingerprint_npz(round_dir / "beta_posteriors.npz")
    
    train_csv = round_dir / "training_data_merged.csv"
    if train_csv.exists():
        metrics.train_dataset_sha256 = sha256_file(train_csv)
        try:
            df = pd.read_csv(train_csv)
            metrics.train_dataset_points = len(df)
            if "weight" in df.columns:
                metrics.train_dataset_total_trials = float(df["weight"].sum())
        except Exception:
            pass

    return metrics
