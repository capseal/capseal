from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from fusion_utils import build_knn_graph, conjugate_gradient, estimate_diag_inverse

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "artifacts" / "latest_bicep"
EPS = 1e-12


def negative_log_likelihood(mean: np.ndarray, var: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sum(((truth - mean) ** 2) / (var + EPS) + np.log(var + EPS)))


def solve_for_mu(
    mu: float,
    neighbors,
    weights,
    degrees,
    confidence: np.ndarray,
    q_prior: np.ndarray,
    x0: np.ndarray,
    diag_samples: int = 8,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    b_vec = mu * confidence * q_prior
    fused, iters, residual = conjugate_gradient(
        neighbors,
        weights,
        degrees,
        mu,
        confidence,
        b_vec,
        x0,
        tol=1e-6,
        max_iter=4000,
    )
    diag = estimate_diag_inverse(
        neighbors,
        weights,
        degrees,
        mu,
        confidence,
        samples=diag_samples,
        tol=2e-4,
        max_iter=2000,
    )
    return fused, diag, iters, residual


def main() -> None:
    grid = np.load(RUN_DIR / "grid.npz")
    bicep = np.load(RUN_DIR / "bicep.npz")
    enn = np.load(RUN_DIR / "enn.npz")

    x = grid["x"]
    y = grid["y"]
    truth = bicep["q_bicep"]
    var_bicep = bicep["var_bicep"]
    q_prior = enn["q_enn"]

    n = len(x)
    rng = np.random.default_rng(2025)
    holdout_ratio = 0.2
    holdout_size = max(1, int(holdout_ratio * n))
    holdout_idx = rng.choice(n, size=holdout_size, replace=False)
    mask = np.ones(n, dtype=bool)
    mask[holdout_idx] = False

    points = np.stack([x, y], axis=1)
    neighbors, weights, degrees, length_scale = build_knn_graph(points, k=15)

    base_conf = 1.0 / (var_bicep + 1e-9)
    base_conf = np.clip(base_conf, 1.0, 1e6)

    mu_candidates: List[float] = [0.05, 0.1, 0.2, 0.4]
    summaries = []

    best_mu = mu_candidates[0]
    best_nll = float("inf")

    for mu in mu_candidates:
        confidence = base_conf * mask.astype(np.float64)
        fused, diag, iters, residual = solve_for_mu(
            mu,
            neighbors,
            weights,
            degrees,
            confidence,
            q_prior,
            q_prior.copy(),
            diag_samples=8,
        )
        nll = negative_log_likelihood(fused[~mask], diag[~mask], truth[~mask])
        summaries.append((mu, nll, iters, residual))
        print(
            f"mu={mu:.3f} holdout NLL={nll:.3f} iterations={iters} residual={residual:.2e}"
        )
        if nll < best_nll:
            best_nll = nll
            best_mu = mu

    print(f"Best mu={best_mu:.3f} with holdout NLL={best_nll:.3f}")

    confidence_full = base_conf
    fused, diag, iters, residual = solve_for_mu(
        best_mu,
        neighbors,
        weights,
        degrees,
        confidence_full,
        q_prior,
        q_prior.copy(),
        diag_samples=16,
    )

    summary_arr = np.array(
        summaries,
        dtype=[("mu", "f8"), ("nll", "f8"), ("iters", "f8"), ("residual", "f8")],
    )

    np.savez(
        RUN_DIR / "fusion.npz",
        q_fused=fused,
        var_fused=diag,
        mu=best_mu,
        length_scale=length_scale,
        iterations=iters,
        residual=residual,
        mu_summary=summary_arr,
        holdout_idx=holdout_idx,
    )
    print(
        f"Saved best fusion with mu={best_mu:.3f} (iterations={iters}, residual={residual:.2e})"
    )


if __name__ == "__main__":
    main()
