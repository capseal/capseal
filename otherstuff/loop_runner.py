#!/usr/bin/env python3
"""Main driver for the adaptive BICEP -> Beta -> ENN -> Fusion -> Plan loop.

This script runs N rounds end-to-end with structured artifact management,
comprehensive metrics, and reproducible seeds.

Usage:
    # Run 10 rounds with incrementing seeds
    python -m loop_runner --run-dir artifacts/latest_bicep --rounds 10 --seed 1234 --seed-mode increment

    # Run 5 rounds with fixed seed (for debugging)
    python -m loop_runner --run-dir artifacts/latest_bicep --rounds 5 --seed 42 --seed-mode fixed

    # Run with random seeds
    python -m loop_runner --run-dir artifacts/latest_bicep --rounds 10 --seed-mode random

    # Show status of existing loop
    python -m loop_runner --run-dir artifacts/latest_bicep --status
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from loop_metrics import (
    RoundMetrics,
    compute_round_metrics,
    compute_tube_metrics,
)
from loop_io import (
    CaptureOutput,
    append_to_summary_csv,
    copy_artifacts_to_round,
    copy_plan_as_input,
    copy_plan_as_output,
    copy_tallies,
    create_round_dir,
    get_next_round_num,
    get_previous_round_info,
    get_run_uuid,
    init_run_dir,
    load_json,
    print_round_summary,
    save_fingerprints,
    save_json,
    save_metrics,
    save_report_md,
)
from rebuild_training_data import rebuild_dataset


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------


def get_seed(
    base_seed: int,
    round_num: int,
    mode: str,
    seed_log: List[int],
) -> int:
    """Get seed for this round based on mode."""
    if mode == "fixed":
        return base_seed
    elif mode == "increment":
        return base_seed + round_num
    elif mode == "random":
        # Use base_seed to seed the RNG for reproducibility
        rng = random.Random(base_seed + round_num * 1000)
        seed = rng.randint(0, 2**31 - 1)
        return seed
    else:
        raise ValueError(f"Unknown seed mode: {mode}")


# ---------------------------------------------------------------------------
# BICEP execution
# ---------------------------------------------------------------------------


def run_bicep(
    plan_path: Path,
    output_path: Path,
    seed: int,
    steps: int = 10000,
    dt: float = 1e-3,
    temperature: float = 0.5,
    use_synthetic: bool = False,
) -> int:
    """Run BICEP simulations for the plan.

    Returns total trajectories run.
    """
    if not plan_path.exists():
        print(f"[BICEP] No plan found at {plan_path}, skipping")
        return 0

    with open(plan_path) as f:
        plan = json.load(f)

    indices = plan.get("selected_indices", [])
    points = plan.get("points", [])
    budgets = plan.get("budget", [])

    if not indices:
        print("[BICEP] No points in plan, skipping")
        return 0

    total_budget = sum(budgets)
    print(f"[BICEP] Running {len(indices)} points, {total_budget} total trajectories")
    print(f"[BICEP] seed={seed}, steps={steps}, dt={dt}, T={temperature}")

    if use_synthetic:
        return _run_synthetic_bicep(indices, points, budgets, output_path, seed)
    else:
        return _run_real_bicep(plan_path, output_path, seed, steps, dt, temperature)


def _run_synthetic_bicep(
    indices: List[int],
    points: List[List[float]],
    budgets: List[int],
    output_path: Path,
    seed: int,
) -> int:
    """Fast synthetic BICEP using analytical committor (for testing)."""
    rng = np.random.default_rng(seed)

    def true_committor(x: float, y: float) -> float:
        # Simplified double-well committor: transition at x ~ 0
        width = 0.8
        q = 0.5 * (1 + np.tanh(x / width))
        return float(np.clip(q, 0.001, 0.999))

    results = []
    total_trajectories = 0

    for idx, pt, budget in zip(indices, points, budgets):
        x, y = pt
        q_true = true_committor(x, y)
        # Add noise to simulate stochasticity
        q_noisy = np.clip(q_true + rng.normal(0, 0.05), 0.001, 0.999)
        successes = rng.binomial(budget, q_noisy)
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

    print(f"[BICEP] Synthetic: {len(results)} points, {total_trajectories} trajectories")
    return total_trajectories


def _run_real_bicep(
    plan_path: Path,
    output_path: Path,
    seed: int,
    steps: int,
    dt: float,
    temperature: float,
) -> int:
    """Run real BICEP using the Rust binary."""
    try:
        from run_bicep import run_bicep_batch
        result = run_bicep_batch(
            plan_path,
            output_path,
            steps=steps,
            dt=dt,
            temperature=temperature,
            seed_base=seed,
        )
        return result.get("total_trajectories", 0)
    except ImportError:
        print("[BICEP] run_bicep module not available, falling back to synthetic")
        with open(plan_path) as f:
            plan = json.load(f)
        return _run_synthetic_bicep(
            plan.get("selected_indices", []),
            plan.get("points", []),
            plan.get("budget", []),
            output_path,
            seed,
        )


# ---------------------------------------------------------------------------
# Active round execution
# ---------------------------------------------------------------------------


def run_active_round(
    repo_root: Path,
    run_dir: Path,
    bicep_results: Optional[Path],
    seed: int,
    round_id: str = None,
) -> bool:
    """Run the active round pipeline (ingest -> train -> fuse -> plan).

    Returns True on success.
    """
    from adaptive_sampling import AdaptiveSampler

    # Step 1: Ingest BICEP results
    if bicep_results and bicep_results.exists():
        print(f"[ActiveRound] Ingesting results from {bicep_results}")
        sampler = AdaptiveSampler(run_dir)
        sampler.update_from_csv(bicep_results)
    else:
        print("[ActiveRound] No BICEP results to ingest")

    # Step 1.5: Rebuild training data
    print("[ActiveRound] Rebuilding training dataset...")
    merged_csv_path = run_dir / "training_data_merged.csv"
    rebuild_dataset(
        run_dir=run_dir,
        static_csv_path=repo_root / "double_well_data.csv",
        beta_path=run_dir / "beta_posteriors.npz",
        output_path=merged_csv_path
    )

    # Step 2: Train ENN
    print("[ActiveRound] Training ENN...")
    train_success = _run_train_enn(repo_root, run_dir, seed, data_path=merged_csv_path)
    if not train_success:
        print("[ActiveRound] WARNING: ENN training may have failed")

    # Step 3: Fuse field
    print("[ActiveRound] Running fusion...")
    fuse_success = _run_fuse_field(repo_root, run_dir, data_path=merged_csv_path)
    if not fuse_success:
        print("[ActiveRound] WARNING: Fusion may have failed")

    # Step 4: Compute new plan
    print("[ActiveRound] Computing new sampling plan...")
    sampler = AdaptiveSampler(run_dir)
    plan = sampler.compute_plan(
        num_select=64,
        min_distance=0.08,
        n_bins=10,
    )
    plan_path = run_dir / "active_sampling_plan.json"
    # Pass provenance info to plan metadata
    sampler.save_plan(plan, plan_path, round_id=round_id, seed=seed)
    print(f"[ActiveRound] Saved plan with {len(plan['selected_indices'])} points")

    return True


def _run_train_enn(repo_root: Path, run_dir: Path, seed: int, data_path: Optional[Path] = None) -> bool:
    """Run ENN training."""
    import subprocess
    train_script = repo_root / "train_enn.py"
    if not train_script.exists():
        train_script = repo_root / "train_simple_enn.py"

    if not train_script.exists():
        print(f"[Train] No training script found!")
        return False

    # Compute relative path from repo_root to run_dir
    try:
        rel_run_dir = run_dir.relative_to(repo_root)
    except ValueError:
        rel_run_dir = run_dir

    cmd = [
        sys.executable,
        str(train_script),
        "--seed", str(seed),
        "--run-dir", str(rel_run_dir),
    ]
    if data_path:
        cmd.extend(["--data-path", str(data_path)])

    result = subprocess.run(cmd, cwd=str(repo_root))
    return result.returncode == 0


def _run_fuse_field(repo_root: Path, run_dir: Path, data_path: Optional[Path] = None) -> bool:
    """Run fusion."""
    import subprocess
    fuse_script = repo_root / "fuse_field.py"

    if not fuse_script.exists():
        print(f"[Fuse] No fusion script found!")
        return False

    cmd = [sys.executable, str(fuse_script)]
    if data_path:
        cmd.extend(["--data-path", str(data_path)])
        
    result = subprocess.run(cmd, cwd=str(repo_root))
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Single round execution
# ---------------------------------------------------------------------------


def run_one_round(
    repo_root: Path,
    run_dir: Path,
    round_num: int,
    seed: int,
    seed_mode: str,
    baseline_tube_var: Optional[float] = None,
    use_synthetic: bool = True,
    verbose: bool = False,
    run_uuid: Optional[str] = None,
) -> Tuple[RoundMetrics, bool]:
    """Run one complete round of the loop.

    Returns (metrics, success).
    """
    # Generate round ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    round_id = f"R{round_num:04d}_{timestamp}"

    print(f"\n{'='*70}")
    print(f" ROUND {round_num} ({round_id}) ".center(70))
    print(f"{'='*70}\n")

    # Get info from previous round BEFORE creating new round directory
    # (to avoid the new empty directory being picked up as "previous")
    # Pass current_round_num to ensure we only get rounds from the SAME series
    # (e.g., R0002 looks for R0001, not R0020 from a previous run)
    prev_info = get_previous_round_info(
        run_dir, 
        exclude_round_id=round_id,
        current_round_num=round_num,
    )
    prev_round_dir = prev_info["round_dir"] if prev_info else None
    prev_tube_var = prev_info["tube_var"] if prev_info else None
    prev_sampled = prev_info["sampled"] if prev_info else 0
    prev_coverage = prev_info["coverage"] if prev_info else 0.0
    
    # Use passed baseline or extract from previous round info (if resuming)
    if baseline_tube_var is None and prev_info:
        baseline_tube_var = prev_info.get("baseline_tube_var")

    # Debug output for delta computation
    if prev_info:
        print(f"[Delta Debug] Previous round: {prev_info['round_id']}")
        print(f"[Delta Debug] Previous tube_var_sum: {prev_tube_var:.6f}")
    else:
        print(f"[Delta Debug] No previous round found (first round or missing metrics)")
    
    if baseline_tube_var is not None:
        print(f"[Delta Debug] Baseline tube_var_sum: {baseline_tube_var:.6f}")

    # Create round directory
    round_dir = create_round_dir(run_dir, round_id)
    commands_run = []

    # Set up stdout capture
    log_path = round_dir / "stdout.log"

    with CaptureOutput(log_path):
        # Step 1: Copy input plan
        copy_plan_as_input(run_dir, round_dir)

        # Step 2: Run BICEP (if we have a plan)
        plan_path = run_dir / "active_sampling_plan.json"
        tallies_path = run_dir / f"tallies_round{round_num}.csv"

        if plan_path.exists() and round_num > 1:
            print(f"[Step 1/4] Running BICEP simulations (seed={seed})...")
            bicep_cmd = f"run_bicep(plan={plan_path}, seed={seed})"
            commands_run.append(bicep_cmd)
            run_bicep(
                plan_path=plan_path,
                output_path=tallies_path,
                seed=seed,
                use_synthetic=use_synthetic,
            )
            copy_tallies(tallies_path, round_dir)
        else:
            tallies_path = None
            print("[Step 1/4] First round - no BICEP results yet")

        # Step 3: Run active round
        print(f"[Step 2/4] Running ENN training + Fusion...")
        active_cmd = f"active_round(bicep={tallies_path}, seed={seed})"
        commands_run.append(active_cmd)
        run_active_round(
            repo_root=repo_root,
            run_dir=run_dir,
            bicep_results=tallies_path,
            seed=seed,
            round_id=round_id,
        )

        # Step 4: Copy artifacts to round directory
        print(f"[Step 3/4] Archiving artifacts...")
        artifacts_to_copy = [
            "beta_posteriors.npz",
            "enn.npz",
            "enn.pt",
            "fusion.npz",
            "training_data_merged.csv",
        ]
        copy_artifacts_to_round(run_dir, round_dir, artifacts_to_copy)
        copy_plan_as_output(run_dir, round_dir)

        # Step 5: Compute metrics
        print(f"[Step 4/4] Computing metrics...")
        metrics = compute_round_metrics(
            round_dir=round_dir,
            round_id=round_id,
            seed=seed,
            prev_round_dir=prev_round_dir,
            prev_tube_var=prev_tube_var,
            baseline_tube_var=baseline_tube_var,
            prev_sampled=prev_sampled,
            prev_coverage=prev_coverage,
        )
        
        # Invariant check: Delta calculation
        if prev_tube_var is not None:
             computed_delta = prev_tube_var - metrics.tube.tube_var_sum
             if abs(computed_delta - metrics.tube.tube_var_delta_prev) > 1e-6:
                 print(f"[Delta Debug] CRITICAL WARNING: Delta calculation mismatch! {computed_delta} vs {metrics.tube.tube_var_delta_prev}")
        
        # Sanity check (Task A): Detect state reset
        if round_num > 1 and metrics.counts.sampled_points_total == metrics.counts.sampled_points_new:
            print(f"[WARNING] State reset detected! Round {round_num} has "
                  f"sampled_points_total == sampled_points_new ({metrics.counts.sampled_points_total}). "
                  f"Prior sampling state may not have been loaded correctly.")

        # Save metrics and fingerprints
        save_metrics(metrics, round_dir)
        save_fingerprints(round_dir)
        save_report_md(metrics, round_dir, seed_mode, commands_run)

        # Append to summary CSV (with run_uuid for series tracking)
        append_to_summary_csv(run_dir, metrics, run_uuid=run_uuid)

    # Print summary (outside capture so it goes to console)
    print_round_summary(metrics, verbose=verbose)

    return metrics, True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_loop(
    repo_root: Path,
    run_dir: Path,
    n_rounds: int,
    base_seed: int,
    seed_mode: str,
    use_synthetic: bool = True,
    verbose: bool = False,
    early_stop_patience: int = 0,
    early_stop_min_improve: float = 0.001,
) -> None:
    """Run the full adaptive loop for n_rounds."""
    # Initialize run directory (creates beta_posteriors.npz if missing, etc.)
    init_info = init_run_dir(run_dir, base_seed, seed_mode)
    run_uuid = init_info["run_uuid"]
    
    # Determine starting round (resume from where we left off)
    start_round = get_next_round_num(run_dir)
    end_round = start_round + n_rounds - 1
    
    # Load baseline from previous rounds if resuming
    baseline_tube_var: Optional[float] = None
    if start_round > 1:
        # Find baseline from first complete round
        prev_info = get_previous_round_info(run_dir, current_round_num=start_round)
        if prev_info:
            baseline_tube_var = prev_info.get("baseline_tube_var")
            print(f"[Resume] Continuing from round {start_round} (found {start_round - 1} complete rounds)")
            if baseline_tube_var:
                print(f"[Resume] Loaded baseline tube_var: {baseline_tube_var:.6f}")

    print(f"""
{'='*70}
{'ADAPTIVE SAMPLING LOOP'.center(70)}
{'='*70}
  Run UUID:         {run_uuid}
  Rounds to run:    {n_rounds} (R{start_round:04d} -> R{end_round:04d})
  Seed mode:        {seed_mode}
  Base seed:        {base_seed}
  BICEP mode:       {'Synthetic' if use_synthetic else 'Real'}
  Early stop:       {f'patience={early_stop_patience}' if early_stop_patience > 0 else 'disabled'}
{'='*70}
""")

    # Track seeds and status
    seed_log: List[int] = []
    status_history: List[str] = []
    no_improve_count = 0

    for round_num in range(start_round, end_round + 1):
        # Get seed for this round
        seed = get_seed(base_seed, round_num, seed_mode, seed_log)
        seed_log.append(seed)

        # Run the round
        metrics, success = run_one_round(
            repo_root=repo_root,
            run_dir=run_dir,
            round_num=round_num,
            seed=seed,
            seed_mode=seed_mode,
            baseline_tube_var=baseline_tube_var,
            use_synthetic=use_synthetic,
            verbose=verbose,
            run_uuid=run_uuid,
        )

        if not success:
            print(f"\n[STOPPED] Round {round_num} failed!")
            break
            
        # Set baseline if not set (from first valid round)
        if baseline_tube_var is None:
            baseline_tube_var = metrics.tube.tube_var_baseline
            
        status_history.append(metrics.status)

        # Check early stopping
        if early_stop_patience > 0:
            if metrics.status == "IMPROVED":
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= early_stop_patience:
                print(f"\n[EARLY STOP] No improvement for {early_stop_patience} rounds")
                break

    # Final summary
    print_final_summary(run_dir, seed_log, status_history)


def print_final_summary(
    run_dir: Path,
    seed_log: List[int],
    status_history: List[str],
) -> None:
    """Print final loop summary."""
    print(f"""
{'='*70}
{'LOOP COMPLETE'.center(70)}
{'='*70}
""")

    # Load and print progress
    csv_path = run_dir / "summary.csv"
    if csv_path.exists():
        print("Progress over rounds:")
        # Format headers with consistent width
        headers = (
            f"{'Round':<18} {'Seed':<12} {'Coverage':<10} {'TubeVar':<12} "
            f"{'Delta(Prev)':<12} {'Delta(Base)':<12} {'Status':<10}"
        )
        print(headers)
        print("-" * len(headers))

        from loop_io import load_summary_csv
        rows = load_summary_csv(run_dir)
        for row in rows[-10:]:  # Show last 10 rounds
            # Handle empty strings (first round or missing data)
            delta_prev_raw = row.get('tube_var_delta_prev', '')
            if not delta_prev_raw: # Fallback to old 'tube_var_delta' if new col missing
                delta_prev_raw = row.get('tube_var_delta', '')

            delta_base_raw = row.get('tube_var_delta_baseline', '')

            d_prev_str = f"{float(delta_prev_raw):+.4f}".ljust(12) if delta_prev_raw else "N/A".ljust(12)
            d_base_str = f"{float(delta_base_raw):+.4f}".ljust(12) if delta_base_raw else "N/A".ljust(12)

            print(
                f"{row.get('round_id', '?'):<18} "
                f"{row.get('seed', '?'):<12} "
                f"{float(row.get('tube_coverage', 0)):<10.3f} "
                f"{float(row.get('tube_var', 0)):<12.4f} "
                f"{d_prev_str} "
                f"{d_base_str} "
                f"{row.get('status', '?'):<10}"
            )

    print(f"\n  Artifacts: {run_dir}")
    print(f"  Summary CSV: {run_dir / 'summary.csv'}")
    print(f"  Rounds dir: {run_dir / 'rounds'}")


# ---------------------------------------------------------------------------
# Status command
# ---------------------------------------------------------------------------


def show_status(run_dir: Path) -> None:
    """Show current loop status."""
    print(f"\n{'='*70}")
    print(f" LOOP STATUS ".center(70))
    print(f"{'='*70}\n")

    # Count rounds
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.exists():
        print("No rounds directory found. Loop has not started.")
        return

    round_dirs = sorted([d for d in rounds_dir.iterdir() if d.is_dir()])
    print(f"  Total rounds: {len(round_dirs)}")

    # Load latest metrics
    if round_dirs:
        latest = round_dirs[-1]
        metrics_path = latest / "metrics.json"
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            print(f"  Latest round: {latest.name}")
            print(f"  Tube coverage: {metrics.get('tube', {}).get('tube_coverage', 0):.3f}")
            print(f"  Tube variance: {metrics.get('tube', {}).get('tube_var_sum', 0):.6f}")
            print(f"  Status: {metrics.get('status', '?')}")

    # Show summary stats
    csv_path = run_dir / "summary.csv"
    if csv_path.exists():
        from loop_io import load_summary_csv
        rows = load_summary_csv(run_dir)
        if rows:
            improved = sum(1 for r in rows if r.get("status") == "IMPROVED")
            regressed = sum(1 for r in rows if r.get("status") == "REGRESSED")
            no_change = sum(1 for r in rows if r.get("status") == "NO_CHANGE")
            print(f"\n  Status breakdown:")
            print(f"    IMPROVED:  {improved}")
            print(f"    NO_CHANGE: {no_change}")
            print(f"    REGRESSED: {regressed}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the adaptive BICEP -> ENN -> Fusion -> Plan loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run 10 rounds with incrementing seeds
    python -m loop_runner --run-dir artifacts/latest_bicep --rounds 10 --seed 1234 --seed-mode increment

    # Run with fixed seed for debugging
    python -m loop_runner --run-dir artifacts/latest_bicep --rounds 3 --seed 42 --seed-mode fixed

    # Run with random seeds (logged for reproducibility)
    python -m loop_runner --run-dir artifacts/latest_bicep --rounds 10 --seed-mode random --seed 9999

    # Show current status
    python -m loop_runner --run-dir artifacts/latest_bicep --status

    # Run with early stopping
    python -m loop_runner --rounds 20 --early-stop patience=5 min_improve=0.001

    # Run agent evaluation mode (AgentEvalBench)
    python -m loop_runner --mode agent_eval --agent-bench toy_v1 --run-dir artifacts/agent_test --rounds 5 --seed 12345
""",
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["bicep", "agent_eval"],
        default="bicep",
        help="Loop mode: 'bicep' for BICEP simulations, 'agent_eval' for AgentEvalBench",
    )
    parser.add_argument(
        "--agent-bench",
        type=str,
        default="toy_v1",
        help="Agent benchmark to use (only for --mode agent_eval): toy_v1 or refactor",
    )
    parser.add_argument(
        "--backend",
        choices=["synthetic", "refactor"],
        default="synthetic",
        help="Evaluation backend: 'synthetic' for toy simulation, 'refactor' for real code refactors",
    )
    parser.add_argument(
        "--phase",
        choices=["auto", "plan", "ingest"],
        default="auto",
        help="Execution phase: 'auto' runs full loop, 'plan' creates episodes, 'ingest' processes results",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default="",
        help="Path to target repo (required for --backend refactor)",
    )
    parser.add_argument(
        "--episodes-per-budget-unit",
        type=int,
        default=1,
        help="Episodes per budget unit for agent evaluation",
    )
    parser.add_argument(
        "--targets-per-round",
        type=int,
        default=64,
        help="Number of grid points to evaluate per round (K)",
    )

    parser.add_argument(
        "--run-dir",
        type=str,
        default="artifacts/latest_bicep",
        help="Directory containing grid.npz and artifacts",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of rounds to run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Base seed for RNG",
    )
    parser.add_argument(
        "--seed-mode",
        type=str,
        choices=["fixed", "increment", "random"],
        default="increment",
        help="Seed mode: fixed (same each round), increment (seed+i), random",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic BICEP simulator instead of real Rust binary",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Just show current loop status",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--receipts",
        action="store_true",
        help="Emit round_receipt.json and run_receipt.json for verifiable audit",
    )
    parser.add_argument(
        "--early-stop",
        type=str,
        default="",
        help="Early stop config: 'patience=K min_improve=eps'",
    )

    return parser


def parse_early_stop(config: str) -> Tuple[int, float]:
    """Parse early stop config string."""
    patience = 0
    min_improve = 0.001

    if not config:
        return patience, min_improve

    parts = config.split()
    for part in parts:
        if part.startswith("patience="):
            patience = int(part.split("=")[1])
        elif part.startswith("min_improve="):
            min_improve = float(part.split("=")[1])

    return patience, min_improve


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    run_dir = (repo_root / args.run_dir).resolve()

    # For agent_eval mode, create run_dir if it doesn't exist
    if args.mode == "agent_eval" and not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Init] Created run directory: {run_dir}")

    if not run_dir.exists():
        print(f"Error: run-dir does not exist: {run_dir}")
        sys.exit(1)

    if args.status:
        show_status(run_dir)
        return

    # Branch on mode
    if args.mode == "agent_eval":
        # Check for refactor backend
        if args.backend == "refactor" or args.agent_bench == "refactor":
            from agent_bench.refactor_backend import run_refactor_eval_loop

            # Validate repo path for refactor backend
            if not args.repo:
                print("Error: --repo is required for --backend refactor")
                sys.exit(1)

            repo_path = Path(args.repo).expanduser().resolve()
            if not repo_path.exists():
                print(f"Error: repo does not exist: {repo_path}")
                sys.exit(1)

            # Default to fewer targets for refactor (heavy episodes)
            targets = args.targets_per_round
            if targets == 64:  # Default was not overridden
                targets = 8  # More reasonable default for manual refactors
                print(f"[Info] Using default targets_per_round={targets} for refactor backend")

            run_refactor_eval_loop(
                run_dir=run_dir,
                n_rounds=args.rounds,
                base_seed=args.seed,
                seed_mode=args.seed_mode,
                repo_path=repo_path,
                targets_per_round=targets,
                phase=args.phase if args.phase != "auto" else "plan",
                verbose=args.verbose,
            )
        else:
            # Toy benchmark (synthetic or real)
            from agent_bench.runner import run_agent_eval_loop
            run_agent_eval_loop(
                run_dir=run_dir,
                n_rounds=args.rounds,
                base_seed=args.seed,
                seed_mode=args.seed_mode,
                agent_bench=args.agent_bench,
                episodes_per_budget_unit=args.episodes_per_budget_unit,
                targets_per_round=args.targets_per_round,
                use_synthetic=args.synthetic,
                verbose=args.verbose,
                emit_receipts=args.receipts,
            )
    else:
        # BICEP mode (existing flow)
        early_patience, early_min = parse_early_stop(args.early_stop)

        run_loop(
            repo_root=repo_root,
            run_dir=run_dir,
            n_rounds=args.rounds,
            base_seed=args.seed,
            seed_mode=args.seed_mode,
            use_synthetic=args.synthetic,
            verbose=args.verbose,
            early_stop_patience=early_patience,
            early_stop_min_improve=early_min,
        )


if __name__ == "__main__":
    main()
