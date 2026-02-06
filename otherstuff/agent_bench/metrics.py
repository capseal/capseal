#!/usr/bin/env python3
"""Agent-mode tube metrics for AgentEvalBench.

Tube Definition for Agent Mode:
- Threshold: tau = 0.2
- Tube (mean-based): { i : mu[i] <= tau } (points believed safe)
- Future v2: Conservative tube using credible sets

Posterior moments:
- mu[i] = alpha[i] / (alpha[i] + beta[i])
- var[i] = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# Default threshold for "safe" classification
DEFAULT_TAU = 0.2


def compute_agent_tube_metrics(
    alpha: np.ndarray,
    beta: np.ndarray,
    round_id: str,
    round_num: int,
    prev_tube_var: Optional[float] = None,
    baseline_tube_var: Optional[float] = None,
    selected: Optional[np.ndarray] = None,
    episodes_per_target: int = 0,
    tau: float = DEFAULT_TAU,
) -> Dict[str, Any]:
    """Compute tube metrics for agent evaluation mode.
    
    Args:
        alpha: Failure counts + 1 (prior) array.
        beta: Success counts + 1 (prior) array.
        round_id: Round identifier string.
        round_num: Round number (1-indexed).
        prev_tube_var: Previous round's tube_var_sum (for delta).
        baseline_tube_var: Baseline tube_var_sum (for delta_baseline).
        selected: Selected grid indices for this round.
        episodes_per_target: Episodes per target in this round.
        tau: Threshold for tube classification.
        
    Returns:
        Metrics dict compatible with BICEP loop reporting.
    """
    # Cast to float for computation
    alpha_f = alpha.astype(np.float64)
    beta_f = beta.astype(np.float64)
    total = alpha_f + beta_f
    
    # Posterior mean (mu = alpha / (alpha + beta))
    mu = alpha_f / total
    
    # Posterior variance (Beta distribution variance)
    var = (alpha_f * beta_f) / (total ** 2 * (total + 1))
    
    # Tube definition: points with mu <= tau (believed safe)
    tube_mask = mu <= tau
    tube_indices = np.where(tube_mask)[0]
    n_tube = len(tube_indices)
    n_total = len(alpha)
    
    # Tube coverage
    tube_coverage = n_tube / n_total if n_total > 0 else 0.0
    
    # Tube variance sum (epistemic uncertainty in tube)
    tube_var_sum = float(np.sum(var[tube_mask]))
    
    # Epistemic statistics (variance in tube)
    if n_tube > 0:
        tube_var_array = var[tube_mask]
        epistemic = {
            "mean": float(np.mean(tube_var_array)),
            "max": float(np.max(tube_var_array)),
            "median": float(np.median(tube_var_array)),
            "sum": tube_var_sum,
        }
    else:
        epistemic = {"mean": 0.0, "max": 0.0, "median": 0.0, "sum": 0.0}
    
    # Aleatoric is 0 for v1 (no separate aleatoric component)
    aleatoric = {"mean": 0.0, "max": 0.0, "median": 0.0}
    
    # Compute deltas
    tube_var_delta_prev: Optional[float] = None
    tube_var_delta_baseline: Optional[float] = None
    tube_var_baseline: Optional[float] = None
    
    if prev_tube_var is not None:
        # Delta = prev - current (positive means improvement)
        tube_var_delta_prev = prev_tube_var - tube_var_sum
    
    if baseline_tube_var is not None:
        tube_var_baseline = baseline_tube_var
        tube_var_delta_baseline = baseline_tube_var - tube_var_sum
    elif round_num == 1:
        # First round: baseline is current
        tube_var_baseline = tube_var_sum
        tube_var_delta_baseline = None  # N/A for first round
    
    # Determine status
    status = determine_status(tube_var_delta_prev, round_num)
    
    # Count sampled points
    sampled_new = len(selected) if selected is not None else 0
    total_trajectories = sampled_new * episodes_per_target
    
    # Build metrics dict
    metrics = {
        "round_id": round_id,
        "round_num": round_num,
        "status": status,
        "tube": {
            "tube_var_sum": tube_var_sum,
            "tube_coverage": tube_coverage,
            "tube_points_total": n_tube,
            "tube_definition": "mean",  # Document which definition used
            "tau": tau,
            "tube_var_delta_prev": tube_var_delta_prev,
            "tube_var_delta_baseline": tube_var_delta_baseline,
            "tube_var_baseline": tube_var_baseline,
            "epistemic": epistemic,
            "aleatoric": aleatoric,
        },
        "counts": {
            "sampled_points_new": sampled_new,
            "sampled_points_total": int(np.sum(total - 2)),  # Points with any evidence
            "trajectories_new": total_trajectories,
            "trajectories_total": int(np.sum(total - 2)),  # Total evidence count
        },
        "plan": {
            "targets_selected": sampled_new,
            "total_budget": total_trajectories,
        },
    }
    
    return metrics


def determine_status(
    tube_var_delta: Optional[float],
    round_num: int,
) -> str:
    """Determine round status based on delta.
    
    Status values:
    - FIRST_ROUND: First round, no comparison possible
    - IMPROVED: tube_var decreased (delta > 0)
    - REGRESSED: tube_var increased (delta < 0)
    - NO_CHANGE: tube_var unchanged (delta == 0)
    
    Args:
        tube_var_delta: Change from previous round (prev - current).
        round_num: Round number (1-indexed).
        
    Returns:
        Status string.
    """
    if tube_var_delta is None or round_num == 1:
        return "FIRST_ROUND"
    
    if tube_var_delta > 1e-9:
        return "IMPROVED"
    elif tube_var_delta < -1e-9:
        return "REGRESSED"
    else:
        return "NO_CHANGE"


def compute_recall_at_tau(
    alpha: np.ndarray,
    beta: np.ndarray,
    true_p_fail: np.ndarray,
    tau: float = DEFAULT_TAU,
) -> Dict[str, float]:
    """Compute recall of unsafe points at threshold tau.
    
    This is a research metric for evaluating active learning strategies.
    
    Args:
        alpha: Failure counts + 1.
        beta: Success counts + 1.
        true_p_fail: Ground truth failure probabilities.
        tau: Threshold for unsafe classification.
        
    Returns:
        Dict with recall, precision, and f1 metrics.
    """
    # Estimated unsafe: mu > tau
    alpha_f = alpha.astype(np.float64)
    beta_f = beta.astype(np.float64)
    mu = alpha_f / (alpha_f + beta_f)
    
    estimated_unsafe = mu > tau
    true_unsafe = true_p_fail > tau
    
    # True positives, false positives, false negatives
    tp = np.sum(estimated_unsafe & true_unsafe)
    fp = np.sum(estimated_unsafe & ~true_unsafe)
    fn = np.sum(~estimated_unsafe & true_unsafe)
    
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # F1 = 2 * precision * recall / (precision + recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "true_unsafe_count": int(np.sum(true_unsafe)),
        "estimated_unsafe_count": int(np.sum(estimated_unsafe)),
    }
