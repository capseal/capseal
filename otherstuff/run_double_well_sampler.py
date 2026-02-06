#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np

A_BOUND = 0.9
B_BOUND = -0.9


def drift(state: np.ndarray) -> np.ndarray:
    x, y = state
    return np.array([-4.0 * x * (x * x - 1.0), -2.0 * y], dtype=np.float64)


def simulate_committor_batch(
    start: np.ndarray,
    n_paths: int,
    dt: float,
    t_max: float,
    sigma: float,
    rng: np.random.Generator,
) -> int:
    sqrt_dt = np.sqrt(dt)
    steps = int(np.ceil(t_max / dt))
    hits = 0
    for _ in range(n_paths):
        state = start.copy().astype(np.float64)
        for _ in range(steps):
            if state[0] > A_BOUND:
                hits += 1
                break
            if state[0] < B_BOUND:
                break
            dw = rng.normal(size=2) * sqrt_dt
            state = state + drift(state) * dt + sigma * dw
    return hits


def beta_ci(alpha: float, beta: float, z: float) -> Tuple[float, float]:
    mean = alpha / (alpha + beta)
    var = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    std = np.sqrt(var)
    return max(0.0, mean - z * std), min(1.0, mean + z * std)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate double-well committor for plan indices")
    parser.add_argument("--plan", default="artifacts/latest_bicep/active_sampling_plan.json")
    parser.add_argument("--grid", default="artifacts/latest_bicep/grid.npz")
    parser.add_argument("--out", default="artifacts/latest_bicep/new_hits_real.csv")
    parser.add_argument("--beta", default="artifacts/latest_bicep/beta_posteriors.npz")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--t-max", type=float, default=6.0)
    parser.add_argument("--sigma", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--min-trials", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--ci-half-width", type=float, default=0.1)
    parser.add_argument("--ci-z", type=float, default=1.96)
    args = parser.parse_args()

    with open(args.plan) as fh:
        plan = json.load(fh)
    grid = np.load(args.grid)
    xs = grid["x"]
    ys = grid["y"]
    beta_path = Path(args.beta)
    if beta_path.exists():
        beta_data = np.load(beta_path)
        alpha = beta_data["alpha"].astype(np.float64)
        beta_vals = beta_data["beta"].astype(np.float64)
    else:
        n = xs.shape[0]
        alpha = np.ones(n, dtype=np.float64)
        beta_vals = np.ones(n, dtype=np.float64)

    rng = np.random.default_rng(args.seed)
    rows = [("index", "successes", "trials")]
    for idx, budget in zip(plan["selected_indices"], plan["budget"]):
        start = np.array([xs[idx], ys[idx]], dtype=np.float64)
        base_alpha = alpha[idx]
        base_beta = beta_vals[idx]
        trials = 0
        hits = 0
        target_max = budget
        min_trials = min(args.min_trials, target_max)
        chunk = max(1, args.chunk_size)
        threshold_low = 0.5 - args.ci_half_width
        threshold_high = 0.5 + args.ci_half_width
        stop_reason = "budget"
        while trials < target_max:
            remaining = target_max - trials
            if trials < min_trials:
                batch = min(min_trials - trials, remaining)
            else:
                batch = min(chunk, remaining)
            if batch <= 0:
                break
            hits += simulate_committor_batch(start, batch, args.dt, args.t_max, args.sigma, rng)
            trials += batch
            alpha_upd = base_alpha + hits
            beta_upd = base_beta + (trials - hits)
            ci_low, ci_high = beta_ci(alpha_upd, beta_upd, args.ci_z)
            if trials >= min_trials:
                if ci_high < threshold_low or ci_low > threshold_high:
                    stop_reason = "ci"
                    break
        rows.append((idx, hits, trials))
        print(f"Sim {idx}: hits={hits}/{trials} (max {budget}, stop={stop_reason})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for row in rows:
            fh.write(",".join(map(str, row)) + "\n")
    print(f"Wrote {len(rows) - 1} rows to {out_path}")


if __name__ == "__main__":
    main()
