#!/usr/bin/env python3
"""Fully automated adaptive sampling loop.

This script runs the complete BICEP → ENN → Fusion → Acquisition loop
automatically for N rounds, either with real BICEP or a synthetic simulator.

Usage:
    # Run 5 rounds with synthetic BICEP
    python run_loop.py --rounds 5 --synthetic

    # Run with real BICEP results directory
    python run_loop.py --rounds 5 --bicep-dir /path/to/bicep/results

    # Just show current state without running
    python run_loop.py --status
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


def run_cmd(cmd: list[str], cwd: Path, description: str = "") -> bool:
    """Run a command and return success status."""
    print(f"\n{'─' * 60}")
    print(f"▶ {description or ' '.join(cmd[:3])}")
    print(f"{'─' * 60}")
    result = subprocess.run(cmd, cwd=str(cwd))
    return result.returncode == 0


def run_real_bicep(
    plan_path: Path,
    output_path: Path,
    steps: int = 10000,
    dt: float = 1e-3,
    temperature: float = 0.5,
) -> int:
    """Run real BICEP simulations using physics-based dynamics."""
    from run_bicep import run_bicep_batch

    result = run_bicep_batch(
        plan_path, output_path,
        steps=steps, dt=dt, temperature=temperature
    )
    return result.get("total_trajectories", 0)


def synthetic_bicep(
    plan_path: Path,
    output_path: Path,
    noise_std: float = 0.05,
) -> int:
    """Fast synthetic BICEP using simplified committor (for quick testing)."""
    with open(plan_path) as f:
        plan = json.load(f)

    points = np.array(plan["points"])
    budgets = np.array(plan["budget"])
    indices = plan["selected_indices"]

    if len(points) == 0:
        print("[SyntheticBICEP] No points in plan!")
        return 0

    # True committor (simplified double-well)
    def true_committor(x, y):
        # Transition at x ≈ 0, width controls sharpness
        width = 0.8
        q = 0.5 * (1 + np.tanh(x / width))
        return np.clip(q, 0.001, 0.999)

    results = []
    total_trajectories = 0

    for i, (idx, pt, budget) in enumerate(zip(indices, points, budgets)):
        x, y = pt
        q_true = true_committor(x, y)

        # Add some noise to simulate stochasticity
        q_noisy = np.clip(q_true + np.random.randn() * noise_std, 0.001, 0.999)

        # Simulate binomial outcomes
        successes = np.random.binomial(budget, q_noisy)

        results.append({
            "index": idx,
            "successes": int(successes),
            "trials": int(budget),
        })
        total_trajectories += budget

    # Write CSV
    with open(output_path, "w") as f:
        f.write("index,successes,trials\n")
        for r in results:
            f.write(f"{r['index']},{r['successes']},{r['trials']}\n")

    print(f"[SyntheticBICEP] Generated {len(results)} points, {total_trajectories} trajectories")
    print(f"[SyntheticBICEP] Saved to {output_path}")

    return total_trajectories


@dataclass
class LoopState:
    """Current state of the adaptive loop."""
    run_dir: Path
    round_num: int
    total_trajectories: int
    tube_var: float
    tube_coverage: float
    tube_sampled: int
    tube_total: int
    selected_points: int

    @classmethod
    def load(cls, run_dir: Path) -> Optional["LoopState"]:
        run_dir = Path(run_dir)

        # Load progress history
        history_path = run_dir / "progress_history.json"
        if not history_path.exists():
            return None

        with open(history_path) as f:
            history = json.load(f)

        if not history:
            return None

        latest = history[-1]

        # Load plan
        plan_path = run_dir / "active_sampling_plan.json"
        selected_points = 0
        if plan_path.exists():
            with open(plan_path) as f:
                plan = json.load(f)
            selected_points = len(plan.get("selected_indices", []))

        tube_sampled = latest.get("tube_sampled", 0)
        tube_total = latest.get("tube_points", 1)

        return cls(
            run_dir=run_dir,
            round_num=len(history),
            total_trajectories=int(latest.get("total_sampled", 0) * 40),  # Estimate
            tube_var=latest.get("tube_var_sum", float("inf")),
            tube_coverage=tube_sampled / max(tube_total, 1),
            tube_sampled=tube_sampled,
            tube_total=tube_total,
            selected_points=selected_points,
        )

    def summary(self) -> str:
        return f"""
╔══════════════════════════════════════════════════════════════════╗
║                    ADAPTIVE LOOP STATUS                          ║
╠══════════════════════════════════════════════════════════════════╣
║  Round:              {self.round_num:<10}                               ║
║  Tube coverage:      {self.tube_sampled}/{self.tube_total} ({100*self.tube_coverage:.1f}%)                          ║
║  Tube variance:      {self.tube_var:<10.4f}                              ║
║  Next targets:       {self.selected_points:<10}                              ║
╚══════════════════════════════════════════════════════════════════╝
"""


def run_one_round(
    repo_root: Path,
    run_dir: Path,
    round_num: int,
    use_synthetic: bool = True,
    bicep_results: Optional[Path] = None,
) -> bool:
    """Run one complete round of the loop."""
    print(f"\n{'═' * 70}")
    print(f"  ROUND {round_num}")
    print(f"{'═' * 70}")

    # Step 1: If we have a plan, run BICEP (real or synthetic)
    plan_path = run_dir / "active_sampling_plan.json"
    results_path = run_dir / f"bicep_results_round{round_num}.csv"

    if plan_path.exists() and round_num > 1:
        if use_synthetic:
            print("\n[Step 1/4] Simulating BICEP trajectories...")
            synthetic_bicep(plan_path, results_path)
            bicep_results = results_path
        elif bicep_results is None:
            print("\n[Step 1/4] Waiting for BICEP results...")
            print(f"  Expected file: {results_path}")
            print(f"  Run BICEP and save results, then re-run this script.")
            return False
    else:
        bicep_results = None
        print("\n[Step 1/4] First round - no BICEP results yet")

    # Step 2: Run active_round.py
    print("\n[Step 2/4] Running ENN training + Fusion + Acquisition...")

    cmd = [
        sys.executable,
        str(repo_root / "active_round.py"),
        "--run-dir", str(run_dir),
    ]
    if bicep_results and bicep_results.exists():
        cmd.extend(["--bicep-results", str(bicep_results)])

    success = run_cmd(cmd, repo_root, "Training ENN, running Fusion, computing plan")

    if not success:
        print("\n[ERROR] Round failed!")
        return False

    # Step 3: Show status
    state = LoopState.load(run_dir)
    if state:
        print(state.summary())

    # Step 4: Generate report
    print("\n[Step 3/4] Generating round report...")
    report_cmd = [sys.executable, str(repo_root / "round_report.py"), "--run-dir", str(run_dir)]
    run_cmd(report_cmd, repo_root, "Generating report")

    print(f"\n[Step 4/4] Round {round_num} complete!")

    return True


def run_loop(
    repo_root: Path,
    run_dir: Path,
    n_rounds: int,
    use_synthetic: bool = True,
    target_tube_var: float = 1.0,
    target_coverage: float = 0.8,
) -> None:
    """Run the full adaptive loop for n_rounds."""
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║           AUTOMATED ADAPTIVE SAMPLING LOOP                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Rounds to run:      {n_rounds:<10}                                    ║
║  BICEP mode:         {'Synthetic' if use_synthetic else 'Real':<15}                         ║
║  Target tube var:    {target_tube_var:<10.4f}                                 ║
║  Target coverage:    {100*target_coverage:.1f}%                                           ║
╚══════════════════════════════════════════════════════════════════╝
""")

    # Get current state
    state = LoopState.load(run_dir)
    start_round = state.round_num + 1 if state else 1

    for round_num in range(start_round, start_round + n_rounds):
        success = run_one_round(
            repo_root=repo_root,
            run_dir=run_dir,
            round_num=round_num,
            use_synthetic=use_synthetic,
        )

        if not success:
            print(f"\n[STOPPED] Loop stopped at round {round_num}")
            break

        # Check convergence
        state = LoopState.load(run_dir)
        if state:
            if state.tube_var < target_tube_var:
                print(f"\n[CONVERGED] Tube variance {state.tube_var:.4f} < target {target_tube_var:.4f}")
                break
            if state.tube_coverage > target_coverage:
                print(f"\n[CONVERGED] Tube coverage {100*state.tube_coverage:.1f}% > target {100*target_coverage:.1f}%")
                break

    # Final summary
    print("\n" + "═" * 70)
    print("  LOOP COMPLETE")
    print("═" * 70)

    state = LoopState.load(run_dir)
    if state:
        print(state.summary())

    # Show progress history
    history_path = run_dir / "progress_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        print("\nProgress over rounds:")
        print(f"{'Round':<8} {'Tube Var':<12} {'Sampled':<10} {'Coverage':<10}")
        print("-" * 45)
        for i, entry in enumerate(history, 1):
            tube_var = entry.get("tube_var_sum", 0)
            sampled = entry.get("total_sampled", 0)
            tube_s = entry.get("tube_sampled", 0)
            tube_t = entry.get("tube_points", 1)
            cov = 100 * tube_s / max(tube_t, 1)
            print(f"{i:<8} {tube_var:<12.4f} {sampled:<10} {cov:<10.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Run automated adaptive sampling loop")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds to run")
    parser.add_argument("--run-dir", type=str, default="artifacts/latest_bicep")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic BICEP simulator")
    parser.add_argument("--status", action="store_true", help="Just show current status")
    parser.add_argument("--target-tube-var", type=float, default=1.0, help="Stop when tube variance below this")
    parser.add_argument("--target-coverage", type=float, default=0.8, help="Stop when tube coverage above this")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    run_dir = (repo_root / args.run_dir).resolve()

    if args.status:
        state = LoopState.load(run_dir)
        if state:
            print(state.summary())
        else:
            print("No loop state found. Run the loop first.")
        return

    run_loop(
        repo_root=repo_root,
        run_dir=run_dir,
        n_rounds=args.rounds,
        use_synthetic=args.synthetic,
        target_tube_var=args.target_tube_var,
        target_coverage=args.target_coverage,
    )


if __name__ == "__main__":
    main()
