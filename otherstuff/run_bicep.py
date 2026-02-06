#!/usr/bin/env python3
"""Run BICEP Rust trajectories for the sampling plan points."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import List, Tuple


# Path to per-point BICEP binary
BICEP_POINT_BINARY = (
    Path(__file__).parent / "BICEPsrc/BICEPrust/bicep/target/release/double_well_point"
)


def run_bicep_for_point(
    x: float,
    y: float,
    n_trajectories: int,
    steps: int = 10000,
    dt: float = 1e-3,
    temperature: float = 0.5,
    left_threshold: float = -0.9,
    right_threshold: float = 0.9,
    seed: int = 42,
) -> Tuple[int, int]:
    """Invoke the Rust `double_well_point` binary to simulate a single plan point."""

    if not BICEP_POINT_BINARY.exists():
        raise FileNotFoundError(
            f"BICEP binary not found at {BICEP_POINT_BINARY}. Build with cargo first."
        )

    cmd: List[str] = [
        str(BICEP_POINT_BINARY),
        f"--x={x}",
        f"--y={y}",
        f"--paths={n_trajectories}",
        f"--dt={dt}",
        f"--t-max={steps * dt}",
        f"--temperature={temperature}",
        f"--left-threshold={left_threshold}",
        f"--right-threshold={right_threshold}",
        f"--seed={seed}",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout = result.stdout.strip().splitlines()
    if not stdout:
        raise RuntimeError(f"BICEP binary produced no output for point ({x}, {y})")

    data = json.loads(stdout[-1])
    return int(data["successes"]), int(data["trials"])


def run_bicep_batch(
    plan_path: Path,
    output_path: Path,
    steps: int = 10000,
    dt: float = 1e-3,
    temperature: float = 0.5,
    left_threshold: float = -0.9,
    right_threshold: float = 0.9,
    seed_base: int = 12345,
) -> dict:
    """Run BICEP for all points in the sampling plan."""

    with open(plan_path) as f:
        plan = json.load(f)

    indices = plan["selected_indices"]
    points = plan["points"]
    budgets = plan["budget"]

    if not indices:
        print("[BICEP] No points in plan!")
        return {"total_trajectories": 0, "points": 0}

    print(f"[BICEP] Running {len(indices)} points, {sum(budgets)} total trajectories")
    print(f"[BICEP] Steps={steps}, dt={dt}, T={temperature}")

    results = []
    total_successes = 0
    total_trajectories = 0

    for i, (idx, pt, budget) in enumerate(zip(indices, points, budgets)):
        x, y = pt

        # Run simulation
        successes, trials = run_bicep_for_point(
            x,
            y,
            budget,
            steps=steps,
            dt=dt,
            temperature=temperature,
            left_threshold=left_threshold,
            right_threshold=right_threshold,
            seed=seed_base + i,
        )

        q_empirical = successes / trials if trials > 0 else 0.5

        results.append({
            "index": idx,
            "successes": successes,
            "trials": trials,
        })

        total_successes += successes
        total_trajectories += trials

        if (i + 1) % 10 == 0 or i == len(indices) - 1:
            print(f"[BICEP] Progress: {i+1}/{len(indices)} points, "
                  f"{total_trajectories} trajectories, "
                  f"avg q={total_successes/total_trajectories:.3f}")

    # Write CSV
    with open(output_path, "w") as f:
        f.write("index,successes,trials\n")
        for r in results:
            f.write(f"{r['index']},{r['successes']},{r['trials']}\n")

    print(f"[BICEP] Saved results to {output_path}")
    print(f"[BICEP] Total: {len(results)} points, {total_trajectories} trajectories")

    return {
        "total_trajectories": total_trajectories,
        "points": len(results),
        "mean_q": total_successes / total_trajectories if total_trajectories > 0 else 0.5,
    }


def main():
    parser = argparse.ArgumentParser(description="Run BICEP simulations for sampling plan")
    parser.add_argument("--plan", required=True, help="Path to sampling plan JSON")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--steps", type=int, default=10000, help="Steps per trajectory")
    parser.add_argument("--dt", type=float, default=1e-3, help="Time step")
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature")
    parser.add_argument("--left", type=float, default=-0.9, help="Left basin threshold")
    parser.add_argument("--right", type=float, default=0.9, help="Right basin threshold")
    parser.add_argument("--seed-base", type=int, default=12345, help="Base seed for RNG")
    args = parser.parse_args()

    run_bicep_batch(
        Path(args.plan),
        Path(args.output),
        steps=args.steps,
        dt=args.dt,
        temperature=args.temperature,
        left_threshold=args.left,
        right_threshold=args.right,
        seed_base=args.seed_base,
    )


if __name__ == "__main__":
    main()
