"""capseal learn - Learn which patches fail on your codebase.

This is the friendly, product-ready interface for real-mode evaluation.
Runs actual patches against your code to build a failure prediction model.

Usage:
    capseal learn src/ --budget 10      # Spend at most $10
    capseal learn src/ --rounds 5       # Or cap by rounds
    capseal learn src/ --time 30m       # Or cap by time
"""
from __future__ import annotations

import click
from pathlib import Path


@click.command("learn")
@click.argument("path", type=click.Path(exists=True), required=True)
@click.option("--budget", "-b", type=float, default=5.0, help="Max spend in dollars (default: $5)")
@click.option("--rounds", "-n", type=int, default=5, help="Max evaluation rounds (default: 5)")
@click.option("--time", "-t", type=str, default=None, help="Max runtime (e.g., '30m', '2h')")
@click.option("--targets-per-round", "-k", type=int, default=16, help="Targets per round (default: 16)")
@click.option("--seed", "-s", type=int, default=None, help="Random seed for reproducibility")
@click.option("--prove", is_flag=True, help="Generate cryptographic proof")
@click.option("--pricing", type=str, default="claude-sonnet", help="Pricing preset (default: claude-sonnet)")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
def learn_command(
    path: str,
    budget: float,
    rounds: int,
    time: str | None,
    targets_per_round: int,
    seed: int | None,
    prove: bool,
    pricing: str,
    quiet: bool,
) -> None:
    """Learn which patches fail on your codebase.

    Runs real patches against Semgrep findings to build a failure prediction
    model. The learned model is then used by `capseal review --gate` to filter
    risky patches before they waste CI time.

    \b
    Examples:
        capseal learn src/ --budget 10       # Spend up to $10 learning
        capseal learn . --rounds 3           # Run 3 rounds
        capseal learn src/ --time 30m        # Learn for 30 minutes
        capseal learn src/ --prove           # With cryptographic proof

    \b
    After learning:
        capseal review src/ --gate           # Uses learned model
        capseal report .capseal/runs/latest  # View what was learned
    """
    import datetime
    import json
    import subprocess
    import uuid

    import numpy as np

    from ..budget import BudgetTracker, estimate_learning_cost, PRICING_PRESETS
    from ..episode_runner import EpisodeRunner, EpisodeRunnerConfig

    # ANSI colors
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    target_path = Path(path).expanduser().resolve()

    # Parse time limit
    time_limit_seconds = None
    if time:
        time_limit_seconds = _parse_time(time)

    # Validate pricing preset
    if pricing not in PRICING_PRESETS:
        click.echo(f"{RED}Unknown pricing preset: {pricing}{RESET}")
        click.echo(f"Available: {', '.join(PRICING_PRESETS.keys())}")
        raise SystemExit(1)

    if seed is None:
        import time as time_module
        seed = int(time_module.time()) % 100000

    # Estimate cost
    estimated_cost = estimate_learning_cost(
        rounds=rounds,
        targets_per_round=targets_per_round,
        episodes_per_target=1,
        pricing_preset=pricing,
    )

    # Create run directory
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%dT%H%M%S")
    run_uuid = str(uuid.uuid4())[:8]
    run_path = target_path / ".capseal" / "runs" / f"{timestamp}-learn"
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "rounds").mkdir(exist_ok=True)

    if not quiet:
        click.echo(f"\n{CYAN}{'═' * 65}{RESET}")
        click.echo(f"{CYAN}  LEARNING{RESET}")
        click.echo(f"{CYAN}{'═' * 65}{RESET}")
        click.echo(f"  Target:      {target_path}")
        click.echo(f"  Budget:      ${budget:.2f}")
        click.echo(f"  Rounds:      {rounds} (max)")
        click.echo(f"  Targets:     {targets_per_round} per round")
        if time_limit_seconds:
            click.echo(f"  Time limit:  {time}")
        click.echo(f"  Est. cost:   ${estimated_cost:.2f}")
        click.echo(f"{CYAN}{'═' * 65}{RESET}\n")

    try:
        from bef_zk.shared.scoring import compute_acquisition_score, select_targets, compute_tube_metrics
        from bef_zk.shared.features import extract_patch_features, discretize_features, features_to_grid_idx, grid_idx_to_features
        from bef_zk.shared.receipts import build_round_receipt, build_run_receipt

        # Step 1: Scan with Semgrep
        if not quiet:
            click.echo(f"{DIM}[1/4] Scanning codebase with Semgrep...{RESET}")

        result = subprocess.run(
            ["semgrep", "--config", "auto", "--json", "--exclude", "node_modules", "--exclude", ".venv", "--exclude", "vendor", str(target_path)],
            capture_output=True,
            timeout=600,  # 10 minutes for large repos
        )

        findings = []
        try:
            output = json.loads(result.stdout.decode())
            findings = output.get("results", [])
        except json.JSONDecodeError:
            pass

        if not quiet:
            click.echo(f"      Found {len(findings)} potential issues")

        if not findings:
            click.echo(f"{YELLOW}No Semgrep findings - nothing to learn from{RESET}")
            click.echo(f"{DIM}Add some code or run 'semgrep --config auto' manually to check{RESET}")
            raise SystemExit(0)

        # Map findings to grid
        grid_idx_to_findings = {}
        for finding in findings:
            file_path = finding.get("path", "")
            severity = finding.get("extra", {}).get("severity", "warning")
            start_line = finding.get("start", {}).get("line", 1)
            end_line = finding.get("end", {}).get("line", start_line + 5)

            diff_preview = f"diff --git a/{file_path} b/{file_path}\n"
            diff_preview += f"+++ b/{file_path}\n"
            lines_changed = max(5, end_line - start_line + 1)
            diff_preview += f"@@ -{start_line},{lines_changed} @@\n"

            raw_features = extract_patch_features(diff_preview, [{"severity": severity}])
            levels = discretize_features(raw_features)
            grid_idx = features_to_grid_idx(levels)

            if grid_idx not in grid_idx_to_findings:
                grid_idx_to_findings[grid_idx] = []
            grid_idx_to_findings[grid_idx].append(finding)

        if not quiet:
            click.echo(f"      Mapped to {len(grid_idx_to_findings)} unique risk profiles")

        # Step 2: Initialize
        if not quiet:
            click.echo(f"{DIM}[2/4] Initializing model...{RESET}")

        n_points = 1024
        alpha = np.ones(n_points, dtype=np.int64)
        beta = np.ones(n_points, dtype=np.int64)

        # Initialize budget tracker
        budget_tracker = BudgetTracker.create(
            budget=budget,
            pricing_preset=pricing,
            storage_path=run_path / "budget.json",
        )

        # Initialize episode runner
        runner_config = EpisodeRunnerConfig(
            timeout_seconds=60,
            max_retries=1,
            pricing_preset=pricing,
            budget_limit=budget,
            log_path=run_path / "episodes.jsonl",
        )
        runner = EpisodeRunner(target_path, runner_config, budget_tracker)

        # Save metadata
        metadata = {
            "run_uuid": run_uuid,
            "seed": seed,
            "mode": "real",
            "n_rounds": rounds,
            "targets_per_round": targets_per_round,
            "budget": budget,
            "pricing_preset": pricing,
            "created_at": now.isoformat(),
        }
        (run_path / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

        # Step 3: Learning rounds
        if not quiet:
            click.echo(f"\n{DIM}[3/4] Running learning rounds...{RESET}\n")

        rng = np.random.default_rng(seed)
        round_receipts = []
        all_metrics = []
        start_time = datetime.datetime.now()
        total_successes = 0
        total_failures = 0

        for round_num in range(1, rounds + 1):
            # Check budget
            if budget_tracker.budget_exhausted:
                if not quiet:
                    click.echo(f"  {YELLOW}Budget exhausted - stopping early{RESET}")
                break

            # Check time limit
            if time_limit_seconds:
                elapsed = (datetime.datetime.now() - start_time).total_seconds()
                if elapsed >= time_limit_seconds:
                    if not quiet:
                        click.echo(f"  {YELLOW}Time limit reached - stopping early{RESET}")
                    break

            round_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            round_id = f"R{round_num:04d}_{round_timestamp}"
            round_dir = run_path / "rounds" / round_id
            round_dir.mkdir(parents=True, exist_ok=True)

            if not quiet:
                remaining = budget_tracker.remaining_budget
                budget_str = f"${remaining:.2f} remaining" if remaining else ""
                click.echo(f"  {CYAN}Round {round_num}/{rounds}{RESET} {DIM}{budget_str}{RESET}")

            # Select targets using acquisition function
            scores = compute_acquisition_score(alpha, beta)
            K = min(targets_per_round, len(grid_idx_to_findings))

            # Only select from indices that have findings
            available_indices = np.array(list(grid_idx_to_findings.keys()))
            if len(available_indices) == 0:
                break

            available_scores = scores[available_indices]
            sorted_indices = np.argsort(-available_scores)[:K]
            selected = available_indices[sorted_indices]

            # Save plan
            plan = {
                "round_id": round_id,
                "selected": selected.tolist(),
                "episodes_per_target": 1,
            }
            (round_dir / "active_sampling_plan.json").write_text(json.dumps(plan, indent=2))

            # Run episodes
            results = []
            round_successes = 0
            round_failures = 0

            for target_idx, grid_idx in enumerate(selected):
                grid_idx = int(grid_idx)

                # Check budget before each episode
                if budget_tracker.budget_exhausted:
                    break

                finding_list = grid_idx_to_findings.get(grid_idx, [])
                if not finding_list:
                    continue

                finding = finding_list[0]  # Take first finding for this grid point
                episode_id = f"{round_id}_{target_idx:04d}"

                episode_result = runner.run_episode(episode_id, grid_idx, finding)

                results.append({
                    "round_id": round_id,
                    "grid_idx": grid_idx,
                    "episode_idx": 0,
                    "success": 1 if episode_result.success else 0,
                    "cost": episode_result.cost,
                })

                if episode_result.success:
                    round_successes += 1
                    alpha[grid_idx] += 1  # Alpha = successes
                else:
                    round_failures += 1
                    beta[grid_idx] += 1  # Beta = failures

            total_successes += round_successes
            total_failures += round_failures

            # Save results
            import csv
            results_path = round_dir / "agent_results.csv"
            with open(results_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["round_id", "grid_idx", "episode_idx", "success", "cost"])
                writer.writeheader()
                writer.writerows(results)

            # Save posteriors
            np.savez(
                run_path / "beta_posteriors.npz",
                alpha=alpha,
                beta=beta,
                grid_version="semgrep_scan",
                run_uuid=run_uuid,
            )

            # Compute metrics
            tube = compute_tube_metrics(alpha, beta)
            metrics = {
                "round_id": round_id,
                "round_num": round_num,
                "tube": tube,
                "counts": {"successes": round_successes, "failures": round_failures},
            }
            (round_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            all_metrics.append(metrics)

            # Build round receipt
            round_config = {"grid_version": "semgrep_scan", "targets_per_round": targets_per_round}
            receipt = build_round_receipt(round_dir, round_config)
            (round_dir / "round_receipt.json").write_text(json.dumps(receipt, indent=2))
            round_receipts.append(receipt)

            if not quiet:
                click.echo(f"    {GREEN}✓ {round_successes}{RESET} success  {RED}✗ {round_failures}{RESET} fail  cost: ${budget_tracker.total_cost:.2f}")

        # Close runner
        runner.close()

        # Build run receipt
        run_receipt = build_run_receipt(run_path, round_receipts)
        (run_path / "run_receipt.json").write_text(json.dumps(run_receipt, indent=2))

        # Copy final posteriors to models directory
        import shutil
        models_dir = target_path / ".capseal" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_path / "beta_posteriors.npz", models_dir / "beta_posteriors.npz")

        # Create "latest" symlink
        latest_link = target_path / ".capseal" / "runs" / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_path.name)

        # Generate proof if requested
        capsule = None
        if prove and round_receipts:
            if not quiet:
                click.echo(f"\n{DIM}[4/4] Generating cryptographic proof...{RESET}")
            try:
                from bef_zk.capsule.eval_adapter import EvalAdapter, build_eval_capsule, save_eval_capsule

                adapter = EvalAdapter()
                round_results = []
                for round_dir in sorted((run_path / "rounds").glob("R*")):
                    metrics_file = round_dir / "metrics.json"
                    if not metrics_file.exists():
                        continue
                    m = json.loads(metrics_file.read_text())
                    round_results.append({
                        "round_dir": round_dir,
                        "round_id": round_dir.name,
                        "metrics": m.get("tube", m),
                        "status": "IMPROVING",
                        "n_successes": m.get("counts", {}).get("successes", 0),
                        "n_failures": m.get("counts", {}).get("failures", 0),
                        "posteriors_path": run_path / "beta_posteriors.npz",
                        "plan_path": round_dir / "active_sampling_plan.json",
                        "results_path": round_dir / "agent_results.csv",
                    })

                eval_config = {"grid_version": "semgrep_scan", "targets_per_round": targets_per_round}
                trace = adapter.simulate_trace(round_results, eval_config)
                commitment = adapter.commit_to_trace(trace, row_archive_dir=run_path / "row_archive")
                proof = adapter.generate_proof(trace, commitment)
                capsule = build_eval_capsule(trace, commitment, proof, run_path)
                save_eval_capsule(capsule, run_path / "eval_capsule.json")

                if not quiet:
                    click.echo(f"      Capsule: {capsule['capsule_hash'][:16]}...")
            except Exception as e:
                if not quiet:
                    click.echo(f"{YELLOW}Proof generation failed: {e}{RESET}")

        # Analyze what was learned
        learned_insights = _analyze_posteriors(alpha, beta)

        # Final summary
        click.echo(f"\n{CYAN}{'═' * 65}{RESET}")
        click.echo(f"{CYAN}  LEARNING COMPLETE{RESET}")
        click.echo(f"{CYAN}{'═' * 65}{RESET}")
        click.echo(f"  Rounds:          {len(round_receipts)}")
        click.echo(f"  Episodes run:    {total_successes + total_failures}")
        click.echo(f"  Successes:       {total_successes} ({100*total_successes/max(1, total_successes+total_failures):.0f}%)")
        click.echo(f"  Failures:        {total_failures} ({100*total_failures/max(1, total_successes+total_failures):.0f}%)")
        click.echo(f"  Cost:            ${budget_tracker.total_cost:.2f}")
        click.echo()
        click.echo(f"  {BOLD}What was learned:{RESET}")
        for insight in learned_insights[:6]:
            color = RED if "High risk" in insight else (GREEN if "Low risk" in insight else YELLOW)
            click.echo(f"  {color}•{RESET} {insight}")
        click.echo()
        click.echo(f"  Model saved to {models_dir / 'beta_posteriors.npz'}")
        click.echo(f"  {DIM}Future 'capseal review --gate' will use this model.{RESET}")
        click.echo(f"{CYAN}{'═' * 65}{RESET}\n")

    except FileNotFoundError:
        click.echo(f"{RED}Semgrep not found. Install with: pip install semgrep{RESET}")
        raise SystemExit(1)
    except subprocess.TimeoutExpired:
        click.echo(f"{RED}Semgrep timed out{RESET}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"{RED}Error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


def _parse_time(time_str: str) -> int:
    """Parse time string like '30m', '2h', '1h30m' to seconds."""
    import re

    total_seconds = 0
    patterns = [
        (r"(\d+)h", 3600),
        (r"(\d+)m", 60),
        (r"(\d+)s", 1),
    ]

    for pattern, multiplier in patterns:
        match = re.search(pattern, time_str)
        if match:
            total_seconds += int(match.group(1)) * multiplier

    if total_seconds == 0:
        # Try parsing as plain minutes
        try:
            total_seconds = int(time_str) * 60
        except ValueError:
            pass

    return total_seconds if total_seconds > 0 else 1800  # Default 30m


def _analyze_posteriors(alpha: "np.ndarray", beta: "np.ndarray") -> list[str]:
    """Analyze posteriors to generate human-readable insights."""
    from bef_zk.shared.features import grid_idx_to_features

    insights = []

    # Find high-risk and low-risk regions with sufficient data
    high_risk = []
    low_risk = []
    uncertain = []

    for idx in range(len(alpha)):
        total = alpha[idx] + beta[idx]
        if total <= 2:  # Just prior, no data
            continue

        p_fail = alpha[idx] / total

        if p_fail > 0.5 and total > 3:
            high_risk.append((idx, p_fail, total))
        elif p_fail < 0.2 and total > 3:
            low_risk.append((idx, p_fail, total))
        elif 0.2 <= p_fail <= 0.5:
            uncertain.append((idx, p_fail, total))

    # Sort by confidence (total observations)
    high_risk.sort(key=lambda x: -x[2])
    low_risk.sort(key=lambda x: -x[2])

    # Generate insights for high-risk
    for idx, p_fail, total in high_risk[:3]:
        features = grid_idx_to_features(idx)
        desc = _describe_features(features)
        insights.append(f"High risk: {desc} ({p_fail*100:.0f}% failure rate)")

    # Generate insights for low-risk
    for idx, p_fail, total in low_risk[:3]:
        features = grid_idx_to_features(idx)
        desc = _describe_features(features)
        insights.append(f"Low risk: {desc} ({(1-p_fail)*100:.0f}% success rate)")

    # Add uncertain count
    if uncertain:
        insights.append(f"Uncertain: {len(uncertain)} regions need more data")

    if not insights:
        insights.append("Not enough data yet - run more rounds")

    return insights


def _describe_features(features: list[int]) -> str:
    """Convert feature levels to human-readable description."""
    descriptions = []

    # Feature names and level descriptions
    feature_info = [
        ("complexity", ["simple", "moderate", "complex", "very complex"]),
        ("files", ["single file", "few files", "many files", "cross-cutting"]),
        ("severity", ["style", "warning", "error", "security"]),
        ("size", ["tiny", "small", "medium", "large"]),
        ("coverage", ["untested", "partial", "good", "excellent"]),
    ]

    for i, (name, levels) in enumerate(feature_info):
        if i < len(features):
            level_idx = min(features[i], len(levels) - 1)
            descriptions.append(levels[level_idx])

    # Pick most distinctive features
    key_features = []
    if features[2] >= 2:  # High severity
        key_features.append(descriptions[2])
    if features[0] >= 2:  # High complexity
        key_features.append(descriptions[0])
    if features[1] >= 2:  # Many files
        key_features.append(descriptions[1])
    if features[3] >= 2:  # Large change
        key_features.append(descriptions[3])

    if not key_features:
        key_features = [descriptions[0], descriptions[2]]  # complexity + severity

    return " + ".join(key_features[:3])


__all__ = ["learn_command"]
