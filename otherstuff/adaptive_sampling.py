#!/usr/bin/env python3
"""Adaptive sampling loop for FusionAlpha + ENN + BICEP.

This script scores states, selects high-value sampling targets, and tracks
Beta posteriors for Bernoulli committor hits. It does NOT launch BICEP jobs;
instead it outputs a plan enumerating where to spend simulation budget.

Usage:
    python adaptive_sampling.py suggest --run-dir artifacts/latest_bicep
    python adaptive_sampling.py update --run-dir artifacts/latest_bicep --results new_hits.csv

`new_hits.csv` must contain columns: index, successes, trials
(indices refer to row offsets in grid.npz / enn.npz / fusion.npz arrays).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Beta posterior bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class BetaPosterior:
    alpha: np.ndarray
    beta: np.ndarray

    @classmethod
    def initialize(cls, n: int, prior_alpha: float = 1.0, prior_beta: float = 1.0) -> "BetaPosterior":
        alpha = np.full(n, prior_alpha, dtype=np.float64)
        beta = np.full(n, prior_beta, dtype=np.float64)
        return cls(alpha=alpha, beta=beta)

    @classmethod
    def load(cls, path: Path) -> "BetaPosterior":
        data = np.load(path)
        return cls(alpha=data["alpha"], beta=data["beta"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, alpha=self.alpha, beta=self.beta)

    def mean(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> np.ndarray:
        num = self.alpha * self.beta
        denom = (self.alpha + self.beta) ** 2 * (self.alpha + self.beta + 1.0)
        return num / np.maximum(denom, 1e-24)

    def update_batch(self, indices: np.ndarray, successes: np.ndarray, trials: np.ndarray) -> None:
        successes = np.asarray(successes, dtype=np.float64)
        trials = np.asarray(trials, dtype=np.float64)
        if np.any(trials < successes):
            raise ValueError("Trials must be >= successes for all updates")
        self.alpha[indices] += successes
        self.beta[indices] += (trials - successes)


# ---------------------------------------------------------------------------
# Acquisition computation
# ---------------------------------------------------------------------------


def _percentile_scale(values: np.ndarray, p: float = 90.0, eps: float = 1e-4) -> float:
    scale = float(np.percentile(np.abs(values), p))
    return max(scale, eps)


def _normalize_metric(values: np.ndarray, max_ratio: float = 10.0) -> np.ndarray:
    """Normalize by 90th percentile, then cap outliers at max_ratio times the scale."""
    scale = _percentile_scale(values)
    normalized = values / scale
    # Cap outliers to prevent blow-up from extreme values
    return np.clip(normalized, 0.0, max_ratio)


def estimate_gradient_norm(points: np.ndarray, values: np.ndarray, neighbors: int = 12) -> np.ndarray:
    n, dim = points.shape
    if n == 0:
        return np.zeros(0)
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    np.fill_diagonal(dist2, np.inf)
    k = min(neighbors, max(n - 1, 1))
    idx = np.argpartition(dist2, k, axis=1)[:, :k]
    grads = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nbr_idx = idx[i]
        nbr_points = points[nbr_idx] - points[i]
        nbr_values = values[nbr_idx] - values[i]
        if nbr_points.size == 0:
            grads[i] = 0.0
            continue
        # Weighted least squares for gradient
        w = 1.0 / (np.linalg.norm(nbr_points, axis=1) + 1e-9)
        A = nbr_points * w[:, None]
        b = nbr_values * w
        try:
            grad_vec, *_ = np.linalg.lstsq(A, b, rcond=None)
            grads[i] = np.linalg.norm(grad_vec)
        except np.linalg.LinAlgError:
            grads[i] = 0.0
    return grads


@dataclass
class AcquisitionComponents:
    uncertainty: np.ndarray  # Beta variance drop (how much we'd learn from more samples)
    tube: np.ndarray  # Gradient magnitude * tube proximity
    disagreement: np.ndarray  # ENN vs Beta posterior disagreement
    fusion_delta: np.ndarray  # Fusion correction magnitude
    epistemic: np.ndarray  # MC dropout epistemic uncertainty (model doesn't know)

    def normalized(self) -> "AcquisitionComponents":
        return AcquisitionComponents(
            uncertainty=_normalize_metric(self.uncertainty),
            tube=_normalize_metric(self.tube),
            disagreement=_normalize_metric(self.disagreement),
            fusion_delta=_normalize_metric(self.fusion_delta),
            epistemic=_normalize_metric(self.epistemic),
        )

    def subset(self, mask: np.ndarray) -> "AcquisitionComponents":
        return AcquisitionComponents(
            uncertainty=self.uncertainty[mask],
            tube=self.tube[mask],
            disagreement=self.disagreement[mask],
            fusion_delta=self.fusion_delta[mask],
            epistemic=self.epistemic[mask],
        )


def compute_acquisition_components(
    points: np.ndarray,
    beta_post: BetaPosterior,
    q_prior: np.ndarray,
    q_fused: np.ndarray,
    var_epistemic: np.ndarray | None = None,
    eps: float = 1e-6,
    sigma_q: float = 0.08,
    grad_neighbors: int = 12,
    preview_batch: int = 32,
    beta_std_floor: float = 0.03,  # ~30 samples worth of certainty
) -> AcquisitionComponents:
    beta_mean = beta_post.mean()
    beta_std = np.sqrt(beta_post.variance())
    grad_mag = estimate_gradient_norm(points, q_prior, neighbors=grad_neighbors)
    tube_gate = np.exp(-((q_prior - 0.5) ** 2) / (2 * sigma_q * sigma_q))
    tube_metric = grad_mag * tube_gate
    # Clamp denominator to prevent explosion when beta_std is tiny
    beta_std_clamped = np.maximum(beta_std, beta_std_floor)
    disagreement = np.abs(q_prior - beta_mean) / beta_std_clamped
    fusion_delta = np.abs(q_fused - q_prior)
    variance_drop = beta_variance_pref_drop(beta_post, preview_batch)

    # Epistemic uncertainty from MC dropout (if available)
    # This signals "model hasn't seen this region" vs "region is inherently noisy"
    if var_epistemic is None:
        # Fallback: use zeros (no epistemic signal)
        epistemic = np.zeros_like(q_prior)
    else:
        # Weight epistemic by tube proximity - we care about epistemic uncertainty
        # mainly in the transition region
        epistemic = np.sqrt(var_epistemic) * tube_gate

    return AcquisitionComponents(
        uncertainty=variance_drop,
        tube=tube_metric,
        disagreement=disagreement,
        fusion_delta=fusion_delta,
        epistemic=epistemic,
    )


def combine_acquisition(components: AcquisitionComponents, score_cap: float = 50.0) -> np.ndarray:
    """Combine acquisition components with weights, capping final score for safety.

    Weight breakdown:
        uncertainty (0.30): Expected Beta variance reduction from more samples
        tube (0.25): Gradient magnitude near q=0.5 (transition region importance)
        epistemic (0.25): MC dropout variance (model doesn't know this region)
        disagreement (0.15): ENN vs Beta posterior disagreement
        fusion_delta (0.05): Fusion correction magnitude
    """
    norm = components.normalized()

    # Weights: uncertainty + epistemic together = 0.55 (knowing vs not knowing)
    #          tube = 0.25 (where it matters)
    #          disagreement + fusion = 0.20 (conflict signals)
    w_u, w_t, w_e, w_d, w_f = 0.30, 0.25, 0.25, 0.15, 0.05

    raw_score = (
        w_u * norm.uncertainty +
        w_t * norm.tube +
        w_e * norm.epistemic +
        w_d * norm.disagreement +
        w_f * norm.fusion_delta
    )
    # Final safety cap - scores above this indicate something pathological
    return np.clip(raw_score, 0.0, score_cap)


def beta_variance_pref_drop(beta_post: BetaPosterior, batch: int) -> np.ndarray:
    """Expected variance reduction after `batch` more samples for each Beta posterior."""
    a = beta_post.alpha
    b = beta_post.beta
    current = a * b / ((a + b) ** 2 * (a + b + 1.0))
    a_b = a + b
    mean_success = batch * (a / a_b)
    var_success = batch * a * b * (a_b + batch) / (a_b**2 * (a_b + 1.0))
    second_moment = var_success + mean_success**2
    numerator = a * b + a * batch - a * mean_success + b * mean_success + batch * mean_success - second_moment
    denom = (a_b + batch) ** 2 * (a_b + batch + 1.0)
    expected_future = numerator / np.maximum(denom, 1e-24)
    drop = current - expected_future
    return np.maximum(drop, 0.0)


# ---------------------------------------------------------------------------
# Selection + scheduling
# ---------------------------------------------------------------------------


def _bin_indices(values: np.ndarray, n_bins: int) -> List[np.ndarray]:
    bins = [list() for _ in range(n_bins)]
    clipped = np.clip(values, 0.0, 0.999999)
    ids = np.floor(clipped * n_bins).astype(int)
    for idx, b in enumerate(ids):
        bins[b].append(idx)
    return [np.array(bin_list, dtype=np.int32) for bin_list in bins]


def _passes_distance(point: np.ndarray, selected_points: List[np.ndarray], min_distance: float) -> bool:
    if not selected_points:
        return True
    for other in selected_points:
        if np.linalg.norm(point - other) < min_distance:
            return False
    return True


def diversify_selection(
    points: np.ndarray,
    scores: np.ndarray,
    q_prior: np.ndarray,
    num_select: int,
    min_distance: float,
    n_bins: int = 10,
    score_percentile_threshold: float = 50.0,
    min_score_margin: float = 0.1,
) -> List[int]:
    """Select diverse high-value points, filtering out low-score filler.

    Args:
        score_percentile_threshold: Only consider points above this percentile of scores.
        min_score_margin: Only consider points with score > median + margin.
    """
    # Filter out low-value points BEFORE selection
    finite_mask = np.isfinite(scores)
    if not np.any(finite_mask):
        return []

    finite_scores = scores[finite_mask]
    percentile_cutoff = np.percentile(finite_scores, score_percentile_threshold)
    median_score = np.median(finite_scores)
    min_score = max(percentile_cutoff, median_score + min_score_margin)

    # Points that pass quality threshold
    quality_mask = (scores >= min_score) & finite_mask

    # If too few quality points, relax to percentile only
    if np.sum(quality_mask) < num_select // 2:
        quality_mask = (scores >= percentile_cutoff) & finite_mask

    bins = _bin_indices(q_prior, n_bins)
    sorted_bins: List[List[int]] = []
    for bin_indices in bins:
        if bin_indices.size == 0:
            sorted_bins.append([])
            continue
        # Only include points that pass quality threshold
        quality_in_bin = bin_indices[quality_mask[bin_indices]]
        if quality_in_bin.size == 0:
            sorted_bins.append([])
            continue
        order = quality_in_bin[np.argsort(scores[quality_in_bin])[::-1]]
        sorted_bins.append(order.tolist())

    selected: List[int] = []
    selected_points: List[np.ndarray] = []
    exhausted = False
    while len(selected) < num_select and not exhausted:
        exhausted = True
        for bin_order in sorted_bins:
            while bin_order:
                candidate = bin_order.pop(0)
                if not np.isfinite(scores[candidate]):
                    continue
                if _passes_distance(points[candidate], selected_points, min_distance):
                    selected.append(candidate)
                    selected_points.append(points[candidate])
                    exhausted = False
                    break
            if len(selected) >= num_select:
                break
    return selected


def allocate_trajectory_budget(
    scores: np.ndarray,
    min_traj: int = 32,
    max_traj: int = 256,
    base: int = 32,
) -> np.ndarray:
    if scores.size == 0:
        return np.zeros(0, dtype=np.int32)
    norm = (scores - scores.min()) / max(scores.max() - scores.min(), 1e-9)
    budget = base + norm * (max_traj - min_traj)
    budget = np.clip(budget, min_traj, max_traj)
    return budget.astype(np.int32)


# ---------------------------------------------------------------------------
# Adaptive sampler orchestration
# ---------------------------------------------------------------------------


class AdaptiveSampler:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.grid = np.load(run_dir / "grid.npz")
        self.points = np.stack([self.grid["x"], self.grid["y"]], axis=1)
        self.beta_state_path = run_dir / "beta_posteriors.npz"
        self.beta_post = self._load_beta()

    def _load_beta(self) -> BetaPosterior:
        if self.beta_state_path.exists():
            beta = BetaPosterior.load(self.beta_state_path)
        else:
            beta = BetaPosterior.initialize(len(self.points))
            beta.save(self.beta_state_path)
        return beta

    def load_priors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Load ENN predictions, fused field, and epistemic variance.

        Returns:
            q_enn: ENN committor prediction
            q_fused: Fusion-corrected committor (or ENN if fusion not run)
            var_epistemic: MC dropout epistemic variance (or None if not available)
        """
        enn = np.load(self.run_dir / "enn.npz")
        fusion_path = self.run_dir / "fusion.npz"
        if fusion_path.exists():
            fusion = np.load(fusion_path)
            q_fused = fusion["q_fused"]
        else:
            q_fused = enn["q_enn"]

        # Load epistemic variance if available (from MC dropout)
        var_epistemic = None
        if "var_epistemic" in enn.files:
            var_epistemic = enn["var_epistemic"]

        return enn["q_enn"], q_fused, var_epistemic

    def compute_plan(
        self,
        num_select: int = 64,
        min_distance: float = 0.08,
        n_bins: int = 10,
        ci_half_width: float = 0.1,
        ci_z: float = 1.96,
        preview_batch: int = 32,
        score_percentile_threshold: float = 50.0,
        min_score_margin: float = 0.1,
    ) -> Dict:
        q_prior, q_fused, var_epistemic = self.load_priors()

        # Report epistemic uncertainty status
        if var_epistemic is not None:
            epi_mean = float(np.mean(var_epistemic))
            epi_max = float(np.max(var_epistemic))
            print(f"[AdaptiveSampling] Epistemic uncertainty available: mean={epi_mean:.6f}, max={epi_max:.6f}")
        else:
            print(f"[AdaptiveSampling] No epistemic uncertainty available (var_epistemic not in enn.npz)")

        components = compute_acquisition_components(
            self.points,
            self.beta_post,
            q_prior,
            q_fused,
            var_epistemic=var_epistemic,
            preview_batch=preview_batch,
        )
        beta_mean = self.beta_post.mean()
        beta_std = np.sqrt(self.beta_post.variance())
        ci_low = np.clip(beta_mean - ci_z * beta_std, 0.0, 1.0)
        ci_high = np.clip(beta_mean + ci_z * beta_std, 0.0, 1.0)
        lo_thresh = 0.5 - ci_half_width
        hi_thresh = 0.5 + ci_half_width
        active_mask = (ci_low <= hi_thresh) & (ci_high >= lo_thresh)
        skipped_confident = np.where(~active_mask)[0].tolist()
        if not np.any(active_mask):
            active_mask = np.ones_like(beta_mean, dtype=bool)
        active_components = components.subset(active_mask)
        scores_active = combine_acquisition(active_components)
        scores = np.full_like(beta_mean, -np.inf)
        scores[active_mask] = scores_active

        # Compute score distribution stats for quality reporting
        finite_scores = scores[np.isfinite(scores)]
        score_stats = {}
        if len(finite_scores) > 0:
            score_stats = {
                "min": float(np.min(finite_scores)),
                "max": float(np.max(finite_scores)),
                "mean": float(np.mean(finite_scores)),
                "median": float(np.median(finite_scores)),
                "p25": float(np.percentile(finite_scores, 25)),
                "p75": float(np.percentile(finite_scores, 75)),
                "p90": float(np.percentile(finite_scores, 90)),
                "threshold_used": float(max(
                    np.percentile(finite_scores, score_percentile_threshold),
                    np.median(finite_scores) + min_score_margin
                )),
            }

        bbox = np.max(self.points, axis=0) - np.min(self.points, axis=0)
        spatial_scale = float(np.linalg.norm(bbox))
        min_dist = min_distance * spatial_scale
        selected_idx = diversify_selection(
            self.points, scores, q_prior, num_select, min_dist, n_bins,
            score_percentile_threshold=score_percentile_threshold,
            min_score_margin=min_score_margin,
        )

        # Handle case where fewer points selected than requested
        if len(selected_idx) < num_select:
            print(f"[AdaptiveSampling] Note: Only {len(selected_idx)} points passed quality threshold "
                  f"(requested {num_select}). This is good - not wasting budget on low-value targets.")

        sel_scores = scores[selected_idx] if selected_idx else np.array([])
        budgets = allocate_trajectory_budget(sel_scores)
        plan = {
            "selected_indices": selected_idx,
            "points": self.points[selected_idx].tolist() if selected_idx else [],
            "scores": sel_scores.tolist() if len(sel_scores) > 0 else [],
            "budget": budgets.tolist() if len(budgets) > 0 else [],
            "components": {
                "uncertainty": components.uncertainty[selected_idx].tolist() if selected_idx else [],
                "tube": components.tube[selected_idx].tolist() if selected_idx else [],
                "epistemic": components.epistemic[selected_idx].tolist() if selected_idx else [],
                "disagreement": components.disagreement[selected_idx].tolist() if selected_idx else [],
                "fusion_delta": components.fusion_delta[selected_idx].tolist() if selected_idx else [],
            },
            "metadata": {
                "ci_half_width": ci_half_width,
                "ci_z": ci_z,
                "preview_batch": preview_batch,
                "skipped_confident": skipped_confident,
                "active_candidate_count": int(np.sum(active_mask)),
                "total_candidates": int(len(beta_mean)),
                "score_stats": score_stats,
                "quality_threshold": {
                    "percentile": score_percentile_threshold,
                    "min_margin": min_score_margin,
                },
            },
        }
        return plan

    def save_plan(self, plan: Dict, path: Path, round_id: str = None, seed: int = None) -> None:
        """Save plan to JSON with optional provenance metadata.
        
        Args:
            plan: The sampling plan dict
            path: Output path
            round_id: Optional round ID for provenance tracking
            seed: Optional seed for provenance tracking
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add provenance fields to metadata if provided
        if "metadata" not in plan:
            plan["metadata"] = {}
        if round_id is not None:
            plan["metadata"]["round_id"] = round_id
        if seed is not None:
            plan["metadata"]["seed"] = seed
            
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(plan, fh, indent=2)

    def update_from_csv(self, csv_path: Path) -> None:
        data = np.genfromtxt(csv_path, delimiter=",", names=True)
        if data.size == 0:
            raise ValueError("CSV appears empty")
        if data.ndim == 0:
            data = np.array([data], dtype=data.dtype)
        for field in ("index", "successes", "trials"):
            if field not in data.dtype.names:
                raise ValueError(f"CSV missing column '{field}'")
        indices = data["index"].astype(np.int64)
        successes = data["successes"].astype(np.float64)
        trials = data["trials"].astype(np.float64)
        self.beta_post.update_batch(indices, successes, trials)
        self.beta_post.save(self.beta_state_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_suggest(args: argparse.Namespace) -> None:
    sampler = AdaptiveSampler(Path(args.run_dir))
    plan = sampler.compute_plan(num_select=args.num_select, min_distance=args.min_distance, n_bins=args.n_bins)
    out_path = Path(args.run_dir) / args.output
    sampler.save_plan(plan, out_path)
    print(f"[AdaptiveSampling] Wrote plan with {len(plan['selected_indices'])} points to {out_path}")


def cmd_update(args: argparse.Namespace) -> None:
    sampler = AdaptiveSampler(Path(args.run_dir))
    sampler.update_from_csv(Path(args.results))
    print(f"[AdaptiveSampling] Updated Beta posteriors using {args.results}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adaptive sampling planner for FusionAlpha")
    sub = parser.add_subparsers(dest="mode", required=True)

    suggest = sub.add_parser("suggest", help="Generate a sampling plan")
    suggest.add_argument("--run-dir", required=True)
    suggest.add_argument("--num-select", type=int, default=64)
    suggest.add_argument("--min-distance", type=float, default=0.08, help="Fraction of diagonal for diversity")
    suggest.add_argument("--n-bins", type=int, default=10, help="Bins across committor values")
    suggest.add_argument("--output", type=str, default="active_sampling_plan.json")
    suggest.set_defaults(func=cmd_suggest)

    update = sub.add_parser("update", help="Update Beta posteriors from CSV results")
    update.add_argument("--run-dir", required=True)
    update.add_argument("--results", required=True)
    update.set_defaults(func=cmd_update)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
