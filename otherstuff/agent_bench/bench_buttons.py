#!/usr/bin/env python3
"""
bench_buttons.py - Three buttons, one loop, no mental overhead.

Usage:
    python -m agent_bench.bench_buttons plan --run-dir ./my_eval --repo ~/projects/market-dashboard
    python -m agent_bench.bench_buttons ingest --run-dir ./my_eval --repo ~/projects/market-dashboard
    python -m agent_bench.bench_buttons report --run-dir ./my_eval
    python -m agent_bench.bench_buttons ui --run-dir ./my_eval --repo ~/projects/market-dashboard
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np


# ============================================================================
# Constants
# ============================================================================

TAU = 0.2  # Safety threshold
CI_LEVEL = 0.95
RNG = np.random.default_rng(42)


def beta_credible_interval(alpha: float, beta: float, ci: float = 0.95, n_samples: int = 10000):
    """Compute credible interval for Beta(alpha, beta) using sampling."""
    samples = RNG.beta(alpha, beta, size=n_samples)
    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 - (1 - ci) / 2) * 100
    lower = np.percentile(samples, lower_pct)
    upper = np.percentile(samples, upper_pct)
    mean = alpha / (alpha + beta)
    return lower, mean, upper


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class EpisodeStatus:
    episode_id: str
    task_id: str
    worktree_path: str
    prompt_path: str
    has_result: bool
    passed: Optional[bool] = None
    reason: Optional[str] = None


@dataclass
class GridStatus:
    total: int
    confidently_safe: int
    confidently_unsafe: int
    boundary: int
    unknown: int
    knife_edge: List[Tuple[int, float, float, float]]  # (idx, lower, mean, upper)


# ============================================================================
# Helpers
# ============================================================================

def get_latest_round_dir(run_dir: Path) -> Optional[Path]:
    """Find the most recent round directory."""
    rounds_dir = run_dir / "rounds"
    if not rounds_dir.exists():
        return None

    round_dirs = sorted(rounds_dir.glob("R*"), reverse=True)
    return round_dirs[0] if round_dirs else None


def load_round_plan(round_dir: Path) -> Optional[dict]:
    """Load round_plan.json from a round directory."""
    plan_path = round_dir / "round_plan.json"
    if not plan_path.exists():
        return None
    with open(plan_path) as f:
        return json.load(f)


def get_episode_status(harness_dir: Path, episode_id: str, task_id: str) -> EpisodeStatus:
    """Get the status of a single episode."""
    episode_dir = harness_dir / "episodes" / episode_id
    worktree_path_file = episode_dir / "worktree_path.txt"
    result_file = episode_dir / "result.json"
    prompt_file = episode_dir / "prompt.txt"

    worktree_path = ""
    if worktree_path_file.exists():
        worktree_path = worktree_path_file.read_text().strip()

    prompt_path = str(prompt_file) if prompt_file.exists() else ""

    has_result = result_file.exists()
    passed = None
    reason = None

    if has_result:
        with open(result_file) as f:
            result = json.load(f)
            passed = result.get("passed", False)
            reason = result.get("reason", "")

    return EpisodeStatus(
        episode_id=episode_id,
        task_id=task_id,
        worktree_path=worktree_path,
        prompt_path=prompt_path,
        has_result=has_result,
        passed=passed,
        reason=reason,
    )


def compute_grid_status(run_dir: Path) -> GridStatus:
    """Analyze posteriors and categorize grid points."""
    posteriors_path = run_dir / "beta_posteriors.npz"

    if not posteriors_path.exists():
        # No posteriors yet - everything is unknown
        grid_path = run_dir / "grid.npz"
        if grid_path.exists():
            grid = np.load(grid_path)
            total = len(grid['spec_level'])
        else:
            total = 1024
        return GridStatus(
            total=total,
            confidently_safe=0,
            confidently_unsafe=0,
            boundary=0,
            unknown=total,
            knife_edge=[],
        )

    data = np.load(posteriors_path)
    alpha = data['alpha']
    beta = data['beta']
    n = len(alpha)

    confidently_safe = 0
    confidently_unsafe = 0
    boundary = 0
    unknown = 0
    knife_edge = []

    for i in range(n):
        a, b = alpha[i], beta[i]

        # Check if still at prior (no observations)
        if a == 1.0 and b == 1.0:
            unknown += 1
            continue

        # Compute credible interval for p_fail
        lower, mean, upper = beta_credible_interval(a, b, CI_LEVEL)

        # Categorize
        if upper < TAU:
            confidently_safe += 1
        elif lower > TAU:
            confidently_unsafe += 1
        else:
            boundary += 1

        # Track knife-edge points (closest to tau)
        distance_to_tau = abs(mean - TAU)
        knife_edge.append((i, lower, mean, upper, distance_to_tau))

    # Sort by distance to tau, keep top 10
    knife_edge.sort(key=lambda x: x[4])
    knife_edge = [(i, l, m, u) for i, l, m, u, _ in knife_edge[:10]]

    return GridStatus(
        total=n,
        confidently_safe=confidently_safe,
        confidently_unsafe=confidently_unsafe,
        boundary=boundary,
        unknown=unknown,
        knife_edge=knife_edge,
    )


def ensure_reports_dir(run_dir: Path) -> Path:
    """Ensure reports directory exists."""
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


# ============================================================================
# BUTTON 1: PLAN NEXT ROUND
# ============================================================================

def count_pending_episodes(run_dir: Path) -> int:
    """Count pending episodes in the latest round."""
    round_dir = get_latest_round_dir(run_dir)
    if not round_dir:
        return 0

    plan = load_round_plan(round_dir)
    if not plan:
        return 0

    harness_dir = Path(plan.get("harness_run_dir", run_dir / "harness"))
    pending = 0
    for target in plan.get("targets", []):
        episode_id = target["episode_id"]
        result_path = harness_dir / "episodes" / episode_id / "result.json"
        if not result_path.exists():
            pending += 1
    return pending


def cmd_plan(run_dir: Path, repo: Path):
    """Plan the next round of episodes."""
    print()
    print("=" * 70)
    print("                    PLAN NEXT ROUND")
    print("=" * 70)
    print()

    # Check for pending episodes first
    pending = count_pending_episodes(run_dir)
    if pending > 0:
        round_dir = get_latest_round_dir(run_dir)
        todo_path = round_dir / "TODO.txt" if round_dir else None

        print("  [BLOCKED] You have pending episodes to complete first.")
        print()
        print(f"  Pending: {pending} episodes")
        if todo_path and todo_path.exists():
            print(f"  TODO:    {todo_path}")
        print()
        print("  Finish the refactors, then press VERIFY + INGEST.")
        print("  Once all episodes are done, PLAN will unlock.")
        print()
        print("=" * 70)
        return False

    # Run loop_runner plan phase
    cmd = [
        sys.executable, "-m", "loop_runner",
        "--mode", "agent_eval",
        "--backend", "refactor",
        "--phase", "plan",
        "--run-dir", str(run_dir),
        "--repo", str(repo),
    ]

    print(f"[1/2] Running acquisition + episode creation...")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"\n[ERROR] Plan phase failed with exit code {result.returncode}")
        return False

    # Find latest round and generate TODO.txt
    round_dir = get_latest_round_dir(run_dir)
    if not round_dir:
        print("[ERROR] No round directory found after planning")
        return False

    plan = load_round_plan(round_dir)
    if not plan:
        print("[ERROR] Could not load round plan")
        return False

    harness_dir = Path(plan.get("harness_run_dir", run_dir / "harness"))

    print(f"\n[2/2] Generating TODO list...")

    # Generate TODO.txt
    todo_lines = []
    todo_lines.append("=" * 70)
    todo_lines.append(f"ROUND {plan['round_num']} - REFACTOR TODO LIST")
    todo_lines.append(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    todo_lines.append("=" * 70)
    todo_lines.append("")

    for i, target in enumerate(plan.get("targets", []), 1):
        episode_id = target["episode_id"]
        task_id = target["task_id"]
        status = get_episode_status(harness_dir, episode_id, task_id)

        todo_lines.append(f"[ ] TASK {i}: {task_id}")
        todo_lines.append("-" * 50)
        todo_lines.append(f"    Episode:   {episode_id}")
        todo_lines.append(f"    Worktree:  {status.worktree_path}")
        todo_lines.append(f"    Prompt:    {status.prompt_path}")
        todo_lines.append("")
        todo_lines.append(f"    STEP 1: cd {status.worktree_path}")
        todo_lines.append(f"    STEP 2: Read prompt.txt and do the refactor")
        todo_lines.append(f"    STEP 3: Commit your changes")
        todo_lines.append("")
        todo_lines.append(f"    VERIFY COMMAND:")
        todo_lines.append(f"    python -m agent_bench.refactor_harness_v1 verify \\")
        todo_lines.append(f"        --harness-dir {harness_dir} \\")
        todo_lines.append(f"        --episode-id {episode_id}")
        todo_lines.append("")
        todo_lines.append("")

    todo_lines.append("=" * 70)
    todo_lines.append("WHEN DONE: Run VERIFY + INGEST")
    todo_lines.append("=" * 70)

    todo_path = round_dir / "TODO.txt"
    todo_path.write_text("\n".join(todo_lines))

    # Print summary
    print()
    print("=" * 70)
    print("PLAN COMPLETE")
    print("=" * 70)
    print(f"  Round:    {plan['round_num']}")
    print(f"  Episodes: {len(plan.get('targets', []))}")
    print(f"  TODO:     {todo_path}")
    print()
    print("NEXT STEP: Open TODO.txt, do the refactors, then VERIFY + INGEST")
    print("=" * 70)
    print()

    return True


# ============================================================================
# BUTTON 2: VERIFY + INGEST
# ============================================================================

def cmd_ingest(run_dir: Path, repo: Path):
    """Verify completed episodes and ingest results."""
    print()
    print("=" * 70)
    print("                    VERIFY + INGEST")
    print("=" * 70)
    print()

    # Find latest round
    round_dir = get_latest_round_dir(run_dir)
    if not round_dir:
        print("[ERROR] No round directory found. Run PLAN first.")
        return False

    plan = load_round_plan(round_dir)
    if not plan:
        print("[ERROR] Could not load round plan")
        return False

    harness_dir = Path(plan.get("harness_run_dir", run_dir / "harness"))

    # Check each episode and try to verify if needed
    print(f"[1/3] Checking episodes in round {plan['round_num']}...")
    print()

    pending = []
    verified = []
    already_done = []

    for target in plan.get("targets", []):
        episode_id = target["episode_id"]
        task_id = target["task_id"]
        status = get_episode_status(harness_dir, episode_id, task_id)

        if status.has_result:
            # Already has result
            already_done.append(status)
            print(f"  [DONE]    {task_id}: {'PASS' if status.passed else 'FAIL'}")
        else:
            # Try to verify
            print(f"  [VERIFY]  {task_id}...", end=" ", flush=True)

            verify_cmd = [
                sys.executable, "-m", "agent_bench.refactor_harness_v1",
                "verify",
                "--harness-dir", str(harness_dir),
                "--episode-id", episode_id,
            ]

            result = subprocess.run(verify_cmd, capture_output=True, text=True)

            # Check if result.json was created
            status = get_episode_status(harness_dir, episode_id, task_id)
            if status.has_result:
                verified.append(status)
                print(f"{'PASS' if status.passed else 'FAIL'}")
            else:
                pending.append(status)
                print("PENDING (no changes detected)")

    print()
    print(f"  Already done: {len(already_done)}")
    print(f"  Just verified: {len(verified)}")
    print(f"  Still pending: {len(pending)}")

    # If there are results to ingest, run ingest phase
    completed_count = len(already_done) + len(verified)

    if completed_count > 0:
        print()
        print(f"[2/3] Running ingest phase...")

        cmd = [
            sys.executable, "-m", "loop_runner",
            "--mode", "agent_eval",
            "--backend", "refactor",
            "--phase", "ingest",
            "--run-dir", str(run_dir),
            "--repo", str(repo),
        ]

        result = subprocess.run(cmd, capture_output=False)

        if result.returncode != 0:
            print(f"\n[ERROR] Ingest phase failed with exit code {result.returncode}")
            return False
    else:
        print()
        print(f"[2/3] Skipping ingest (no completed episodes)")

    # Generate reports
    print()
    print(f"[3/3] Generating reports...")

    reports_dir = ensure_reports_dir(run_dir)

    # Write pending.txt
    pending_lines = []
    if pending:
        pending_lines.append("PENDING EPISODES")
        pending_lines.append("=" * 50)
        pending_lines.append("")
        for ep in pending:
            pending_lines.append(f"  {ep.task_id}")
            pending_lines.append(f"    Worktree: {ep.worktree_path}")
            pending_lines.append(f"    Prompt:   {ep.prompt_path}")
            pending_lines.append("")
    else:
        pending_lines.append("NO PENDING EPISODES")
        pending_lines.append("")
        pending_lines.append("All episodes in the current round are complete.")

    pending_path = reports_dir / "pending.txt"
    pending_path.write_text("\n".join(pending_lines))

    # Generate latest.txt via report command
    _generate_report(run_dir, plan['round_num'])

    # Print summary
    print()
    print("=" * 70)
    print("INGEST COMPLETE")
    print("=" * 70)
    print(f"  Verified:  {len(verified)}")
    print(f"  Pending:   {len(pending)}")
    print(f"  Report:    {reports_dir / 'latest.txt'}")
    print()

    if pending:
        print("NEXT STEP: Finish pending episodes, then VERIFY + INGEST again")
    else:
        print("NEXT STEP: Run PLAN NEXT ROUND")
    print("=" * 70)
    print()

    return True


# ============================================================================
# BUTTON 3: OPEN REPORT
# ============================================================================

def _generate_report(run_dir: Path, round_num: Optional[int] = None) -> str:
    """Generate the status report."""
    status = compute_grid_status(run_dir)
    reports_dir = ensure_reports_dir(run_dir)

    # Load summary.csv to get history
    summary_path = run_dir / "summary.csv"
    last_change = "N/A"
    if summary_path.exists():
        import csv
        with open(summary_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                last_row = rows[-1]
                round_num = round_num or last_row.get('round_id', 'unknown')

    # Check for pending episodes
    pending_count = 0
    round_dir = get_latest_round_dir(run_dir)
    if round_dir:
        plan = load_round_plan(round_dir)
        if plan and plan.get("status") == "planned":
            harness_dir = Path(plan.get("harness_run_dir", run_dir / "harness"))
            for target in plan.get("targets", []):
                episode_id = target["episode_id"]
                result_path = harness_dir / "episodes" / episode_id / "result.json"
                if not result_path.exists():
                    pending_count += 1

    lines = []
    lines.append("=" * 70)
    lines.append("                    EVALUATION STATUS")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  Total configurations:     {status.total:,}")
    lines.append("")
    lines.append("  CLASSIFICATION (tau = {:.0%} failure threshold)".format(TAU))
    lines.append("  " + "-" * 50)

    safe_pct = 100 * status.confidently_safe / status.total if status.total else 0
    unsafe_pct = 100 * status.confidently_unsafe / status.total if status.total else 0
    boundary_pct = 100 * status.boundary / status.total if status.total else 0
    unknown_pct = 100 * status.unknown / status.total if status.total else 0

    lines.append(f"  Confidently SAFE:         {status.confidently_safe:4d}  ({safe_pct:5.1f}%)  [upper95 < tau]")
    lines.append(f"  Confidently UNSAFE:       {status.confidently_unsafe:4d}  ({unsafe_pct:5.1f}%)  [lower95 > tau]")
    lines.append(f"  BOUNDARY (uncertain):     {status.boundary:4d}  ({boundary_pct:5.1f}%)  [CI spans tau]")
    lines.append(f"  UNKNOWN (no data):        {status.unknown:4d}  ({unknown_pct:5.1f}%)  [still at prior]")
    lines.append("")

    if status.knife_edge:
        lines.append("  TOP 10 KNIFE-EDGE CONFIGS (closest to tau)")
        lines.append("  " + "-" * 50)
        lines.append("  {:>6}  {:>8}  {:>8}  {:>8}".format("Grid#", "Lower95", "Mean", "Upper95"))
        for idx, lower, mean, upper in status.knife_edge:
            lines.append("  {:>6d}  {:>8.3f}  {:>8.3f}  {:>8.3f}".format(idx, lower, mean, upper))
        lines.append("")

    lines.append("  CURRENT ROUND")
    lines.append("  " + "-" * 50)
    lines.append(f"  Round:            {round_num or 'N/A'}")
    lines.append(f"  Pending episodes: {pending_count}")
    lines.append("")

    lines.append("=" * 70)
    if pending_count > 0:
        lines.append("NEXT: Finish pending episodes, then press VERIFY + INGEST")
    elif status.unknown == status.total:
        lines.append("NEXT: Press PLAN NEXT ROUND to start evaluation")
    else:
        lines.append("NEXT: Press PLAN NEXT ROUND")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    # Write to file
    report_path = reports_dir / "latest.txt"
    report_path.write_text(report_text)

    return report_text


def cmd_report(run_dir: Path):
    """Show the current evaluation status."""
    print()
    report = _generate_report(run_dir)
    print(report)
    print()
    print(f"Report saved to: {run_dir / 'reports' / 'latest.txt'}")
    print()


# ============================================================================
# UI MODE (Interactive menu)
# ============================================================================

def cmd_ui(run_dir: Path, repo: Path):
    """Interactive menu for the three buttons."""
    while True:
        pending = count_pending_episodes(run_dir)

        print()
        print("=" * 50)
        print("        REFACTOR AGENT EVALUATION")
        print("=" * 50)
        print()

        if pending > 0:
            print(f"  [1]  PLAN NEXT ROUND  (blocked: {pending} pending)")
        else:
            print("  [1]  PLAN NEXT ROUND")
        print("  [2]  VERIFY + INGEST")
        print("  [3]  OPEN REPORT")
        print()
        print("  [q]  Quit")
        print()

        try:
            choice = input("  Press 1, 2, 3, or q: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if choice == "1":
            cmd_plan(run_dir, repo)
        elif choice == "2":
            cmd_ingest(run_dir, repo)
        elif choice == "3":
            cmd_report(run_dir)
        elif choice == "q":
            print("\nGoodbye!")
            break
        else:
            print("\n  Invalid choice. Press 1, 2, 3, or q.")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Three buttons, one loop, no mental overhead.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  plan    - Plan the next round of episodes
  ingest  - Verify completed episodes and ingest results
  report  - Show current evaluation status
  ui      - Interactive menu (the three buttons)
        """
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # plan
    plan_parser = subparsers.add_parser("plan", help="Plan next round")
    plan_parser.add_argument("--run-dir", type=Path, required=True)
    plan_parser.add_argument("--repo", type=Path, required=True)

    # ingest
    ingest_parser = subparsers.add_parser("ingest", help="Verify + ingest")
    ingest_parser.add_argument("--run-dir", type=Path, required=True)
    ingest_parser.add_argument("--repo", type=Path, required=True)

    # report
    report_parser = subparsers.add_parser("report", help="Open report")
    report_parser.add_argument("--run-dir", type=Path, required=True)

    # ui
    ui_parser = subparsers.add_parser("ui", help="Interactive menu")
    ui_parser.add_argument("--run-dir", type=Path, required=True)
    ui_parser.add_argument("--repo", type=Path, required=True)

    args = parser.parse_args()

    if args.command == "plan":
        cmd_plan(args.run_dir.expanduser(), args.repo.expanduser())
    elif args.command == "ingest":
        cmd_ingest(args.run_dir.expanduser(), args.repo.expanduser())
    elif args.command == "report":
        cmd_report(args.run_dir.expanduser())
    elif args.command == "ui":
        cmd_ui(args.run_dir.expanduser(), args.repo.expanduser())


if __name__ == "__main__":
    main()
