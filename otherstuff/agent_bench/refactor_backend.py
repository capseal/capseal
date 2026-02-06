#!/usr/bin/env python3
"""Refactor Backend for AgentEvalBench.

Integrates the refactor harness with the active learning loop.
Handles grid mapping, PLAN phase (create episodes), and INGEST phase (update posteriors).

Grid mapping (5 knobs × 4 values = 1024 configs):
  d0: spec_level (0..3)
  d1: task_bucket (0..3)
  d2: tool_noise_level (0..3)
  d3: verify_flip_level (0..3)
  d4: budget_level (0..3)
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent_bench.refactor_tasks_v1 import (
    ALL_TASKS,
    RefactorTask,
    get_task,
)


# =============================================================================
# Grid mapping constants
# =============================================================================

# Verify flip rate mapping (level -> probability)
VERIFY_FLIP_RATE_MAP = [0.0, 0.01, 0.05, 0.20]

# Budget level mapping (level -> budget dict)
BUDGET_LEVEL_MAP = [
    {},  # Level 0: no enforced budget
    {"max_diff_lines": 200, "max_files_touched": 8},  # Level 1: generous
    {"max_diff_lines": 80, "max_files_touched": 4},   # Level 2: moderate
    {"max_diff_lines": 30, "max_files_touched": 2},   # Level 3: tight
]

# Task buckets by difficulty/category
# Bucket 0: easy (rename, small extract, dead code)
TASK_BUCKET_EASY = [
    "rename_provider_status",
    "remove_pseudocode",
    "minimal_diff_rename",
    "make_deterministic",
]

# Bucket 1: medium (extract helper, tighten errors)
TASK_BUCKET_MEDIUM = [
    "extract_base_notification_channel",
    "tighten_provider_error_handling",
    "extract_http_provider_mixin",
    "extract_update_scheduler",
    "extract_base_provider_settings",
    "standardize_logging",
]

# Bucket 2: large (split module, API compat migration)
TASK_BUCKET_LARGE = [
    "split_cache_module",
    "refactor_get_quotes_signature",
    "deprecate_old_api",
    "thread_safety_cleanup",
    "add_type_hints_models",
    "split_ui_app",
]

# Bucket 3: adversarial (no-LSP resilience, minimal-diff constraint, clarification)
TASK_BUCKET_ADVERSARIAL = [
    "resilience_no_lsp",
    "forbidden_path_refactor",
    "fix_on2_lookup",
    "ambiguous_cleanup",
]

TASK_BUCKETS = [
    TASK_BUCKET_EASY,
    TASK_BUCKET_MEDIUM,
    TASK_BUCKET_LARGE,
    TASK_BUCKET_ADVERSARIAL,
]

# Grid version for refactor backend
REFACTOR_GRID_VERSION = "refactor_grid_v1"


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class RefactorCase:
    """A refactor evaluation case derived from a grid point."""
    grid_idx: int
    task_id: str
    spec_level: int
    tool_noise_level: int
    verify_flip_rate: float
    budget: Dict[str, int]

    def to_knobs_json(self) -> str:
        """Convert to harness knobs JSON string."""
        knobs = {
            "spec_level": self.spec_level,
            "tool_noise_level": self.tool_noise_level,
            "verify_flip_rate": self.verify_flip_rate,
            **self.budget,
        }
        return json.dumps(knobs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RoundPlan:
    """Plan for a round of refactor episodes."""
    round_id: str
    round_num: int
    run_uuid: str
    created_at: str
    harness_run_dir: str
    targets: List[Dict[str, Any]]  # [{grid_idx, task_id, episode_id, knobs}, ...]
    status: str = "planned"  # planned, in_progress, completed

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RoundPlan":
        return cls(**d)


# =============================================================================
# Grid generation and mapping
# =============================================================================

def generate_refactor_grid() -> Dict[str, np.ndarray]:
    """Generate the refactor parameter grid (1024 points = 4^5).

    Returns:
        Dict with per-param arrays and metadata:
        - spec_level: int array (0-3)
        - task_bucket: int array (0-3)
        - tool_noise_level: int array (0-3)
        - verify_flip_level: int array (0-3)
        - budget_level: int array (0-3)
        - grid_version: str
        - n_points: int
    """
    from itertools import product

    all_combos = list(product(
        range(4),  # spec_level
        range(4),  # task_bucket
        range(4),  # tool_noise_level
        range(4),  # verify_flip_level
        range(4),  # budget_level
    ))

    n_points = len(all_combos)
    assert n_points == 4 ** 5 == 1024

    spec_level = np.array([c[0] for c in all_combos], dtype=np.int32)
    task_bucket = np.array([c[1] for c in all_combos], dtype=np.int32)
    tool_noise_level = np.array([c[2] for c in all_combos], dtype=np.int32)
    verify_flip_level = np.array([c[3] for c in all_combos], dtype=np.int32)
    budget_level = np.array([c[4] for c in all_combos], dtype=np.int32)

    return {
        "spec_level": spec_level,
        "task_bucket": task_bucket,
        "tool_noise_level": tool_noise_level,
        "verify_flip_level": verify_flip_level,
        "budget_level": budget_level,
        "grid_version": np.array(REFACTOR_GRID_VERSION),
        "n_points": np.array(n_points),
    }


def save_refactor_grid(path: Path) -> None:
    """Generate and save the refactor grid to an NPZ file."""
    grid = generate_refactor_grid()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **grid)
    print(f"[Grid] Saved {grid['n_points']} refactor configs to {path}")
    print(f"[Grid] Version: {REFACTOR_GRID_VERSION}")


def load_refactor_grid(path: Path) -> Dict[str, np.ndarray]:
    """Load a refactor grid from an NPZ file."""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def grid_point_to_refactor_case(idx: int, grid: Dict[str, np.ndarray]) -> RefactorCase:
    """Map a grid point index to a RefactorCase.

    Deterministic task choice per config:
        task_id = bucket_list[(spec_level*16 + tool_noise_level*4 + budget_level) % len(bucket_list)]
    """
    spec_level = int(grid["spec_level"][idx])
    task_bucket = int(grid["task_bucket"][idx])
    tool_noise_level = int(grid["tool_noise_level"][idx])
    verify_flip_level = int(grid["verify_flip_level"][idx])
    budget_level = int(grid["budget_level"][idx])

    # Get the bucket task list
    bucket_list = TASK_BUCKETS[task_bucket]

    # Deterministic task selection within bucket
    task_index = (spec_level * 16 + tool_noise_level * 4 + budget_level) % len(bucket_list)
    task_id = bucket_list[task_index]

    # Map levels to actual values
    verify_flip_rate = VERIFY_FLIP_RATE_MAP[verify_flip_level]
    budget = BUDGET_LEVEL_MAP[budget_level].copy()

    return RefactorCase(
        grid_idx=idx,
        task_id=task_id,
        spec_level=spec_level,
        tool_noise_level=tool_noise_level,
        verify_flip_rate=verify_flip_rate,
        budget=budget,
    )


# =============================================================================
# PLAN phase
# =============================================================================

def run_plan_phase(
    run_dir: Path,
    round_num: int,
    run_uuid: str,
    grid: Dict[str, np.ndarray],
    selected_targets: np.ndarray,
    harness_run_dir: Path,
    repo_path: Path,
    verbose: bool = False,
) -> RoundPlan:
    """Run the PLAN phase: create episodes for selected targets.

    Args:
        run_dir: Main loop run directory
        round_num: Current round number
        run_uuid: Run UUID
        grid: Refactor grid
        selected_targets: Array of selected grid indices
        harness_run_dir: Directory for refactor harness artifacts
        repo_path: Path to the target repository
        verbose: Verbose output

    Returns:
        RoundPlan with episode mappings
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    round_id = f"R{round_num:04d}_{timestamp}"

    print(f"\n{'='*70}")
    print(f" REFACTOR PLAN PHASE - ROUND {round_num} ".center(70))
    print(f"{'='*70}\n")

    # Initialize harness run directory if needed
    harness_run_dir = Path(harness_run_dir)
    if not (harness_run_dir / "run_meta.json").exists():
        print(f"[Plan] Initializing harness run directory: {harness_run_dir}")
        result = subprocess.run(
            [
                sys.executable, "-m", "agent_bench.refactor_harness_v1", "init",
                "--repo", str(repo_path),
                "--run-dir", str(harness_run_dir),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[Plan] ERROR initializing harness: {result.stderr}")
            raise RuntimeError("Failed to initialize harness run directory")

    # Create round directory
    round_dir = run_dir / "rounds" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)

    targets_info = []

    print(f"[Plan] Creating {len(selected_targets)} episodes...")

    for i, grid_idx in enumerate(selected_targets):
        grid_idx = int(grid_idx)
        case = grid_point_to_refactor_case(grid_idx, grid)

        if verbose:
            print(f"  [{i+1}/{len(selected_targets)}] grid_idx={grid_idx} -> {case.task_id} (spec={case.spec_level})")

        # Create episode via harness
        result = subprocess.run(
            [
                sys.executable, "-m", "agent_bench.refactor_harness_v1", "episode",
                "--run-dir", str(harness_run_dir),
                "--task-id", case.task_id,
                "--spec-level", str(case.spec_level),
                "--knobs", case.to_knobs_json(),
            ],
            capture_output=True,
            text=True,
        )

        # Parse episode_id from output
        episode_id = None
        for line in result.stdout.split("\n"):
            if line.startswith("Episode ID:"):
                episode_id = line.split(":")[1].strip()
                break

        if not episode_id:
            print(f"[Plan] WARNING: Could not get episode_id for grid_idx={grid_idx}")
            print(f"[Plan] stdout: {result.stdout[:500]}")
            print(f"[Plan] stderr: {result.stderr[:500]}")
            continue

        targets_info.append({
            "grid_idx": grid_idx,
            "task_id": case.task_id,
            "episode_id": episode_id,
            "spec_level": case.spec_level,
            "tool_noise_level": case.tool_noise_level,
            "verify_flip_rate": case.verify_flip_rate,
            "budget": case.budget,
        })

    # Create round plan
    plan = RoundPlan(
        round_id=round_id,
        round_num=round_num,
        run_uuid=run_uuid,
        created_at=datetime.now().isoformat(),
        harness_run_dir=str(harness_run_dir),
        targets=targets_info,
        status="planned",
    )

    # Save plan
    plan_path = round_dir / "round_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan.to_dict(), f, indent=2)

    print(f"\n[Plan] Created {len(targets_info)} episodes")
    print(f"[Plan] Plan saved to: {plan_path}")

    # Print instructions
    print(f"""
{'='*70}
MANUAL EPISODE EXECUTION REQUIRED
{'='*70}

For each episode, perform the refactor and then verify:

Episodes created:""")

    for t in targets_info[:5]:  # Show first 5
        episode_dir = harness_run_dir / "episodes" / t["episode_id"]
        print(f"""
  Episode: {t['episode_id']}
    Task: {t['task_id']} (spec_level={t['spec_level']})
    Prompt: {episode_dir}/prompt.txt
    Worktree: See {episode_dir}/worktree_path.txt""")

    if len(targets_info) > 5:
        print(f"\n  ... and {len(targets_info) - 5} more episodes")

    print(f"""
After completing refactors, run verification for each:
    python -m agent_bench.refactor_harness_v1 verify \\
        --run-dir {harness_run_dir} \\
        --episode <episode_id>

Then run INGEST phase:
    python -m loop_runner --mode agent_eval --agent-bench refactor \\
        --run-dir {run_dir} \\
        --phase ingest

{'='*70}
""")

    return plan


# =============================================================================
# INGEST phase
# =============================================================================

def run_ingest_phase(
    run_dir: Path,
    run_uuid: str,
    grid: Dict[str, np.ndarray],
    verbose: bool = False,
) -> Tuple[Dict[str, Any], bool]:
    """Run the INGEST phase: read completed episodes and update posteriors.

    Args:
        run_dir: Main loop run directory
        run_uuid: Run UUID
        grid: Refactor grid
        verbose: Verbose output

    Returns:
        Tuple of (metrics dict, success bool)
    """
    from loop_io import get_next_round_num, save_json, append_to_summary_csv
    from agent_bench.runner import (
        load_posteriors, save_posteriors,
        compute_acquisition_score, select_targets,
    )
    from agent_bench.metrics import compute_agent_tube_metrics

    print(f"\n{'='*70}")
    print(" REFACTOR INGEST PHASE ".center(70))
    print(f"{'='*70}\n")

    # Find the most recent round plan
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.exists():
        print("[Ingest] No rounds directory found")
        return {}, False

    round_dirs = sorted([d for d in rounds_dir.iterdir() if d.is_dir()])
    if not round_dirs:
        print("[Ingest] No round directories found")
        return {}, False

    # Find round with plan but no metrics (incomplete round)
    target_round_dir = None
    plan = None

    for rd in reversed(round_dirs):
        plan_path = rd / "round_plan.json"
        metrics_path = rd / "metrics.json"

        if plan_path.exists() and not metrics_path.exists():
            target_round_dir = rd
            with open(plan_path) as f:
                plan = RoundPlan.from_dict(json.load(f))
            break

    if not plan:
        print("[Ingest] No incomplete round plan found")
        print("[Ingest] All rounds appear to be complete. Run --phase plan to start a new round.")
        return {}, False

    print(f"[Ingest] Processing round: {plan.round_id}")
    print(f"[Ingest] Harness run dir: {plan.harness_run_dir}")

    harness_run_dir = Path(plan.harness_run_dir)

    # Load current posteriors
    alpha, beta, _ = load_posteriors(run_dir)

    # Process each episode
    completed = 0
    skipped = 0
    results = []

    for target in plan.targets:
        episode_id = target["episode_id"]
        grid_idx = target["grid_idx"]

        result_path = harness_run_dir / "episodes" / episode_id / "result.json"

        if not result_path.exists():
            if verbose:
                print(f"  [Skip] {episode_id}: no result.json")
            skipped += 1
            continue

        with open(result_path) as f:
            result = json.load(f)

        passed = result.get("passed", False)
        fail = 0 if passed else 1
        success = 1 if passed else 0

        # Update posteriors
        alpha[grid_idx] += fail
        beta[grid_idx] += success

        results.append({
            "grid_idx": grid_idx,
            "episode_id": episode_id,
            "task_id": target["task_id"],
            "passed": passed,
            "reason": result.get("reason", "unknown"),
        })

        completed += 1

        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {episode_id}: {target['task_id']} -> {result.get('reason', '')}")

    print(f"\n[Ingest] Completed: {completed}, Skipped: {skipped}")

    if completed == 0:
        print("[Ingest] No completed episodes to ingest")
        print("[Ingest] Complete the episodes and run verify, then try again.")
        return {}, False

    # Save updated posteriors
    save_posteriors(run_dir, alpha, beta, run_uuid)
    print(f"[Ingest] Updated posteriors saved")

    # Compute metrics
    # Get previous round info for delta
    prev_tube_var = None
    baseline_tube_var = None

    if plan.round_num > 1:
        from loop_io import get_previous_round_info
        prev_info = get_previous_round_info(run_dir, current_round_num=plan.round_num)
        if prev_info:
            prev_tube_var = prev_info.get("tube_var")
            baseline_tube_var = prev_info.get("baseline_tube_var")

    selected = np.array([t["grid_idx"] for t in plan.targets])

    metrics = compute_agent_tube_metrics(
        alpha=alpha,
        beta=beta,
        round_id=plan.round_id,
        round_num=plan.round_num,
        prev_tube_var=prev_tube_var,
        baseline_tube_var=baseline_tube_var,
        selected=selected,
        episodes_per_target=1,  # One episode per target for refactor
    )

    # Add refactor-specific info
    metrics["refactor"] = {
        "completed_episodes": completed,
        "skipped_episodes": skipped,
        "pass_count": sum(1 for r in results if r["passed"]),
        "fail_count": sum(1 for r in results if not r["passed"]),
    }

    # Save metrics (commit marker)
    save_json(metrics, target_round_dir / "metrics.json")

    # Save episode results
    with open(target_round_dir / "episode_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Update plan status
    plan.status = "completed"
    with open(target_round_dir / "round_plan.json", "w") as f:
        json.dump(plan.to_dict(), f, indent=2)

    # Append to summary CSV
    from agent_bench.runner import _metrics_dict_to_round_metrics
    metrics_obj = _metrics_dict_to_round_metrics(metrics)
    append_to_summary_csv(run_dir, metrics_obj, run_uuid=run_uuid)

    print(f"""
{'='*70}
INGEST COMPLETE
{'='*70}
  Round: {plan.round_id}
  Completed: {completed} episodes
  Passed: {metrics['refactor']['pass_count']}
  Failed: {metrics['refactor']['fail_count']}

  tube_var_sum: {metrics['tube']['tube_var_sum']:.6f}
  tube_coverage: {metrics['tube']['tube_coverage']:.3f}
  status: {metrics['status']}

  Metrics saved to: {target_round_dir / 'metrics.json'}
{'='*70}
""")

    return metrics, True


# =============================================================================
# Main entry point for refactor backend
# =============================================================================

def run_refactor_eval_loop(
    run_dir: Path,
    n_rounds: int,
    base_seed: int,
    seed_mode: str,
    repo_path: Path,
    harness_run_dir: Optional[Path] = None,
    targets_per_round: int = 8,
    phase: str = "plan",  # "plan" or "ingest"
    verbose: bool = False,
) -> None:
    """Run the refactor evaluation loop.

    Args:
        run_dir: Main loop run directory
        n_rounds: Number of rounds to run (for plan phase)
        base_seed: Base seed for RNG
        seed_mode: Seed mode (fixed, increment, random)
        repo_path: Path to target repository
        harness_run_dir: Directory for harness artifacts (default: run_dir/harness)
        targets_per_round: Number of grid points per round
        phase: "plan" to create episodes, "ingest" to process results
        verbose: Verbose output
    """
    import uuid as uuid_mod
    from agent_bench.runner import (
        load_posteriors, save_posteriors,
        compute_acquisition_score, select_targets,
    )
    from loop_io import get_next_round_num

    run_dir = Path(run_dir)
    repo_path = Path(repo_path).expanduser().resolve()

    if harness_run_dir is None:
        harness_run_dir = run_dir / "harness"
    else:
        harness_run_dir = Path(harness_run_dir)

    # Load or create run metadata
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        run_uuid = metadata.get("run_uuid", str(uuid_mod.uuid4())[:8])
        print(f"[Init] Resuming run: {run_uuid}")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        run_uuid = str(uuid_mod.uuid4())[:8]
        metadata = {
            "run_uuid": run_uuid,
            "base_seed": base_seed,
            "seed_mode": seed_mode,
            "mode": "agent_eval",
            "backend": "refactor",
            "repo_path": str(repo_path),
            "harness_run_dir": str(harness_run_dir),
            "created_at": datetime.now().isoformat(),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[Init] New run created: {run_uuid}")

    # Load or create grid
    grid_path = run_dir / "grid.npz"
    if not grid_path.exists():
        print(f"[Init] Generating refactor grid: {grid_path}")
        save_refactor_grid(grid_path)

    grid = load_refactor_grid(grid_path)
    n_points = int(grid["n_points"])
    print(f"[Init] Loaded grid with {n_points} points")

    # Initialize posteriors if needed
    beta_path = run_dir / "beta_posteriors.npz"
    if not beta_path.exists():
        alpha = np.ones(n_points, dtype=np.int64)
        beta = np.ones(n_points, dtype=np.int64)
        save_posteriors(run_dir, alpha, beta, run_uuid)
        print(f"[Init] Created beta_posteriors.npz with {n_points} points (Beta(1,1) priors)")

    (run_dir / "rounds").mkdir(exist_ok=True)

    # Branch on phase
    if phase == "ingest":
        metrics, success = run_ingest_phase(
            run_dir=run_dir,
            run_uuid=run_uuid,
            grid=grid,
            verbose=verbose,
        )
        return

    # PLAN phase
    # Determine round number
    round_num = get_next_round_num(run_dir)

    print(f"""
{'='*70}
{'REFACTOR EVALUATION - PLAN PHASE'.center(70)}
{'='*70}
  Run UUID:         {run_uuid}
  Round:            {round_num}
  Grid points:      {n_points}
  Targets/round:    {targets_per_round}
  Repo:             {repo_path}
  Phase:            {phase}
{'='*70}
""")

    # Load posteriors and select targets
    alpha, beta, _ = load_posteriors(run_dir)
    scores = compute_acquisition_score(alpha, beta)
    selected = select_targets(scores, targets_per_round)

    print(f"[Plan] Selected {len(selected)} targets based on acquisition scores")

    # Run plan phase
    plan = run_plan_phase(
        run_dir=run_dir,
        round_num=round_num,
        run_uuid=run_uuid,
        grid=grid,
        selected_targets=selected,
        harness_run_dir=harness_run_dir,
        repo_path=repo_path,
        verbose=verbose,
    )


# =============================================================================
# CLI for grid generation
# =============================================================================

def main():
    """CLI for refactor grid generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Refactor Backend Utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # grid subcommand
    p_grid = subparsers.add_parser("grid", help="Generate refactor grid")
    p_grid.add_argument("--out", required=True, help="Output path for grid.npz")

    # show-case subcommand
    p_show = subparsers.add_parser("show-case", help="Show refactor case for grid index")
    p_show.add_argument("--grid", required=True, help="Path to grid.npz")
    p_show.add_argument("--idx", type=int, required=True, help="Grid index")

    args = parser.parse_args()

    if args.command == "grid":
        save_refactor_grid(Path(args.out))

    elif args.command == "show-case":
        grid = load_refactor_grid(Path(args.grid))
        case = grid_point_to_refactor_case(args.idx, grid)
        print(f"Grid index: {args.idx}")
        print(f"  task_id: {case.task_id}")
        print(f"  spec_level: {case.spec_level}")
        print(f"  tool_noise_level: {case.tool_noise_level}")
        print(f"  verify_flip_rate: {case.verify_flip_rate}")
        print(f"  budget: {case.budget}")
        print(f"  knobs_json: {case.to_knobs_json()}")


if __name__ == "__main__":
    main()
