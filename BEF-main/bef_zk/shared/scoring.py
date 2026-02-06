"""Acquisition scoring and tube metrics for committor gate.

Copied from otherstuff/agent_bench/runner.py and metrics.py to avoid
cross-dependency between shell and bench packages.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# Default threshold for "safe" classification
DEFAULT_TAU = 0.2


def compute_acquisition_score(
    alpha: np.ndarray,
    beta: np.ndarray,
    tau: float = 0.2,
    sigma: float = 0.05,
    w1: float = 1.0,
    w2: float = 0.5,
) -> np.ndarray:
    """Compute acquisition scores for all grid points.

    The acquisition function combines:
    - Epistemic uncertainty (posterior variance)
    - Boundary bonus (proximity to decision threshold tau)

    Args:
        alpha: Failure counts + 1 (prior) array.
        beta: Success counts + 1 (prior) array.
        tau: Decision threshold (default 0.2 = 20% failure rate).
        sigma: Width of boundary bonus kernel.
        w1: Weight for variance term.
        w2: Weight for boundary bonus term.

    Returns:
        Acquisition scores array (higher = more valuable to sample).
    """
    alpha_f = alpha.astype(np.float64)
    beta_f = beta.astype(np.float64)
    total = alpha_f + beta_f
    mu = alpha_f / total
    var = (alpha_f * beta_f) / (total ** 2 * (total + 1))
    boundary_bonus = np.exp(-np.abs(mu - tau) / sigma)
    return var * w1 + boundary_bonus * w2


def select_targets(scores: np.ndarray, K: int) -> np.ndarray:
    """Select top-K targets with deterministic tie-breaking.

    Args:
        scores: Acquisition scores array.
        K: Number of targets to select.

    Returns:
        Array of selected indices (sorted by score descending).
    """
    n = len(scores)
    order = np.lexsort((np.arange(n), -scores))
    return order[:min(K, n)]


def compute_tube_metrics(
    alpha: np.ndarray,
    beta: np.ndarray,
    tau: float = DEFAULT_TAU,
) -> Dict[str, Any]:
    """Compute tube metrics from beta posteriors.

    Tube Definition:
    - Points with mu <= tau are considered "in the tube" (believed safe)
    - tube_var_sum is the total epistemic uncertainty in the tube
    - tube_coverage is the fraction of points in the tube

    Args:
        alpha: Failure counts + 1 (prior) array.
        beta: Success counts + 1 (prior) array.
        tau: Threshold for tube classification.

    Returns:
        Dict with tube_var_sum, tube_coverage, tube_points_total, epistemic stats.
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

    return {
        "tube_var_sum": tube_var_sum,
        "tube_coverage": tube_coverage,
        "tube_points_total": n_tube,
        "tube_definition": "mean",
        "tau": tau,
        "epistemic": epistemic,
    }


def lookup_posterior_at_idx(
    alpha: np.ndarray,
    beta: np.ndarray,
    idx: int,
) -> Dict[str, float]:
    """Look up posterior statistics for a single grid index.

    Args:
        alpha: Failure counts + 1 array.
        beta: Success counts + 1 array.
        idx: Grid index to look up.

    Returns:
        Dict with q (mean p_fail), uncertainty (posterior std), alpha, beta.
    """
    if idx < 0 or idx >= len(alpha):
        return {
            "q": 0.5,  # Uninformative prior
            "uncertainty": 0.5,
            "alpha": 1,
            "beta": 1,
            "valid": False,
        }

    a = float(alpha[idx])
    b = float(beta[idx])
    total = a + b

    # Mean of Beta(alpha, beta) is alpha / (alpha + beta)
    # This is the estimated p_fail
    q = a / total

    # Standard deviation of Beta distribution
    var = (a * b) / (total ** 2 * (total + 1))
    uncertainty = float(np.sqrt(var))

    return {
        "q": float(q),
        "uncertainty": uncertainty,
        "alpha": int(alpha[idx]),
        "beta": int(beta[idx]),
        "valid": True,
    }
