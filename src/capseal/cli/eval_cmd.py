"""Risk learning loop - learns which patches fail on your codebase.

This command runs a learning loop that:
1. Scans the codebase with Semgrep to find potential issues
2. Maps findings to a 5-dimensional feature grid (1024 points)
3. Uses smart sampling to select the most informative targets
4. Runs episodes (synthetic or real) to estimate failure probability
5. Updates the risk model based on results
6. Generates cryptographic proofs for each round

After evaluation, the learned model is saved to .capseal/models/beta_posteriors.npz
for use by `capseal review --gate`.
"""
from __future__ import annotations

import click
from pathlib import Path


@click.command("eval")
@click.argument("path", type=click.Path(exists=True), required=True)
@click.option("--rounds", "-n", default=5, help="Number of evaluation rounds")
@click.option("--seed", "-s", default=None, type=int, help="Random seed")
@click.option("--synthetic", is_flag=True, help="Use synthetic episodes (no LLM calls)")
@click.option("--targets-per-round", "-k", default=64, help="Targets to select per round")
@click.option("--episodes-per-target", "-e", default=1, help="Episodes per target")
@click.option("--prove", is_flag=True, help="Generate cryptographic proof for the learning run")
def eval_command(
    path: str,
    rounds: int,
    seed: int | None,
    synthetic: bool,
    targets_per_round: int,
    episodes_per_target: int,
    prove: bool,
) -> None:
    """Learn which patches fail on your codebase.

    Runs a learning loop that estimates failure probabilities for different
    patch characteristics. The learned model can then be used by
    `capseal review --gate` to filter risky patches.

    \b
    Examples:
        capseal eval src/ --rounds 5 --synthetic        # Quick synthetic eval
        capseal eval . --rounds 3                       # Real pipeline eval
        capseal eval src/ --seed 42 --rounds 10         # Reproducible long run
        capseal eval src/ --rounds 5 --synthetic --prove # With ZK proof

    \b
    After eval completes:
        capseal review src/ --gate    # Uses learned posteriors
        capseal verify-capsule .capseal/runs/latest/eval_capsule.json  # Verify proof
    """
    import datetime
    import json
    import subprocess
    import time
    import uuid

    import numpy as np

    # ANSI colors
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    RESET = "\033[0m"

    target_path = Path(path).expanduser().resolve()
    if not target_path.exists():
        click.echo(f"{RED}Path not found: {target_path}{RESET}")
        raise SystemExit(1)

    if seed is None:
        seed = int(time.time()) % 100000

    try:
        from capseal.shared.scoring import (
            compute_acquisition_score, select_targets, compute_tube_metrics,
        )
        from capseal.shared.features import (
            extract_patch_features, discretize_features, features_to_grid_idx,
            grid_idx_to_features,
        )
        from capseal.shared.receipts import (
            build_round_receipt, build_run_receipt, collect_round_dirs,
        )

        # Create run directory
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%dT%H%M%S")
        run_uuid = str(uuid.uuid4())[:8]
        run_path = target_path / ".capseal" / "runs" / f"{timestamp}-eval"
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "rounds").mkdir(exist_ok=True)

        click.echo(f"\n{CYAN}{'═' * 65}{RESET}")
        click.echo(f"{CYAN}  RISK LEARNING{RESET}")
        click.echo(f"{CYAN}{'═' * 65}{RESET}")
        click.echo(f"  Run ID:    {run_uuid}")
        click.echo(f"  Target:    {target_path}")
        click.echo(f"  Rounds:    {rounds}")
        click.echo(f"  Mode:      {'Synthetic' if synthetic else 'Real pipeline'}")
        click.echo(f"  Seed:      {seed}")
        click.echo(f"{CYAN}{'═' * 65}{RESET}\n")

        # Step 1: Grid generation
        click.echo(f"{DIM}[1/3] Generating feature grid...{RESET}")

        if synthetic:
            # Synthetic mode: generate full 1024-point grid
            n_points = 1024
            grid = {
                "n_points": n_points,
                "version": "synthetic_1024",
                "mode": "synthetic",
            }
            grid_idx_to_findings = {}
            click.echo(f"      Synthetic grid: {n_points} points")
        else:
            # Real mode: scan with Semgrep and build grid from actual findings
            click.echo(f"      Scanning with Semgrep...")
            result = subprocess.run(
                ["semgrep", "--config", "auto", "--json", str(target_path)],
                capture_output=True,
                timeout=120,
            )

            findings = []
            try:
                output = json.loads(result.stdout.decode())
                findings = output.get("results", [])
            except json.JSONDecodeError:
                pass

            click.echo(f"      Found {len(findings)} Semgrep findings")

            if not findings:
                click.echo(f"{YELLOW}⚠ No findings - cannot run real eval without findings{RESET}")
                click.echo(f"{DIM}Use --synthetic for synthetic evaluation{RESET}")
                raise SystemExit(1)

            # Map findings to grid indices
            grid_idx_to_findings = {}
            finding_grid_indices = set()

            for finding in findings:
                file_path = finding.get("path", "")
                severity = finding.get("extra", {}).get("severity", "warning")
                start_line = finding.get("start", {}).get("line", 1)
                end_line = finding.get("end", {}).get("line", start_line + 5)

                diff_preview = f"diff --git a/{file_path} b/{file_path}\n"
                diff_preview += f"--- a/{file_path}\n"
                diff_preview += f"+++ b/{file_path}\n"
                lines_changed = max(5, end_line - start_line + 1)
                diff_preview += f"@@ -{start_line},{lines_changed} +{start_line},{lines_changed} @@\n"
                diff_preview += "+ # patch placeholder\n" * min(lines_changed, 10)

                raw_features = extract_patch_features(diff_preview, [{"severity": severity}])
                levels = discretize_features(raw_features)
                grid_idx = features_to_grid_idx(levels)

                finding_grid_indices.add(grid_idx)
                if grid_idx not in grid_idx_to_findings:
                    grid_idx_to_findings[grid_idx] = []
                grid_idx_to_findings[grid_idx].append(finding)

            n_points = 1024
            grid = {
                "n_points": n_points,
                "version": "semgrep_scan",
                "mode": "real",
                "unique_grid_indices": len(finding_grid_indices),
            }
            click.echo(f"      Mapped to {len(finding_grid_indices)} unique grid points")

        # Save grid
        np.savez(
            run_path / "grid.npz",
            n_points=np.array(n_points),
            version=np.array(grid.get("version", "unknown")),
        )

        # Step 2: Initialize posteriors
        click.echo(f"{DIM}[2/3] Initializing posteriors...{RESET}")
        alpha = np.ones(n_points, dtype=np.int64)
        beta = np.ones(n_points, dtype=np.int64)

        def save_posteriors(path, alpha, beta, run_uuid):
            np.savez(
                path,
                alpha=alpha.astype(np.int64),
                beta=beta.astype(np.int64),
                grid_version=np.array(grid.get("version", "unknown")),
                run_uuid=np.array(run_uuid),
                posterior_semantics=np.array("Beta over p_fail; alpha=fails+1, beta=successes+1"),
            )

        save_posteriors(run_path / "beta_posteriors.npz", alpha, beta, run_uuid)
        click.echo(f"      Beta(1,1) priors for {n_points} grid points")

        # Save run metadata
        metadata = {
            "run_uuid": run_uuid,
            "seed": seed,
            "mode": "synthetic" if synthetic else "real",
            "n_rounds": rounds,
            "targets_per_round": targets_per_round,
            "episodes_per_target": episodes_per_target,
            "created_at": now.isoformat(),
        }
        (run_path / "run_metadata.json").write_text(json.dumps(metadata, indent=2))

        # Step 3: Round loop
        click.echo(f"\n{DIM}[3/3] Running evaluation rounds...{RESET}\n")

        prev_tube_var = None
        baseline_tube_var = None
        round_receipts = []
        all_metrics = []
        rng = np.random.default_rng(seed)
        tube_coverage = 0.0

        for round_num in range(1, rounds + 1):
            round_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            round_id = f"R{round_num:04d}_{round_timestamp}"
            round_dir = run_path / "rounds" / round_id
            round_dir.mkdir(parents=True, exist_ok=True)

            click.echo(f"  {CYAN}Round {round_num}/{rounds} ({round_id}){RESET}")

            # Compute acquisition scores
            scores = compute_acquisition_score(alpha, beta)
            K = min(targets_per_round, n_points)
            selected = select_targets(scores, K)

            click.echo(f"    Selected {len(selected)} targets")

            # Save plan
            plan = {
                "round_id": round_id,
                "selected": selected.tolist(),
                "episodes_per_target": episodes_per_target,
                "tau": 0.2,
                "sigma": 0.05,
                "w1": 1.0,
                "w2": 0.5,
            }
            (round_dir / "active_sampling_plan.json").write_text(json.dumps(plan, indent=2))

            # Run episodes
            results = []
            successes = 0
            failures = 0

            for target_idx, grid_idx in enumerate(selected):
                grid_idx = int(grid_idx)

                for episode_idx in range(episodes_per_target):
                    episode_seed = hash(f"{run_uuid}:{round_num}:{grid_idx}:{episode_idx}") % (2**32)

                    if synthetic:
                        # Synthetic mode: use closed-form p_fail
                        levels = grid_idx_to_features(grid_idx)
                        a, b, c, d, e = 0.9, 0.3, 0.3, 0.2, 0.2
                        p_fail = float(np.clip(
                            a * (levels[3] / 3) +
                            b * (levels[1] / 3) +
                            c * (levels[0] / 10) +
                            d * (levels[2] / 3) +
                            e * (1.0 / max(1, levels[4] + 1)),
                            0.0, 1.0
                        ))
                        ep_rng = np.random.default_rng(episode_seed)
                        success = ep_rng.random() >= p_fail
                    else:
                        # Real mode: attempt pipeline on a finding
                        if grid_idx in grid_idx_to_findings and grid_idx_to_findings[grid_idx]:
                            finding_list = grid_idx_to_findings[grid_idx]
                            finding = finding_list[episode_idx % len(finding_list)]
                            success = _run_eval_episode(target_path, finding, run_path, rng)
                        else:
                            success = True

                    results.append({
                        "round_id": round_id,
                        "grid_idx": grid_idx,
                        "episode_idx": episode_idx,
                        "episode_seed": episode_seed,
                        "success": 1 if success else 0,
                    })

                    if success:
                        successes += 1
                    else:
                        failures += 1

            # Update posteriors
            for r in results:
                idx = r["grid_idx"]
                if r["success"]:
                    beta[idx] += 1
                else:
                    alpha[idx] += 1

            save_posteriors(run_path / "beta_posteriors.npz", alpha, beta, run_uuid)

            # Save results
            import csv
            results_path = round_dir / "agent_results.csv"
            with open(results_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["round_id", "grid_idx", "episode_idx", "episode_seed", "success"])
                writer.writeheader()
                writer.writerows(results)

            # Compute tube metrics
            tube = compute_tube_metrics(alpha, beta)
            tube_var = tube["tube_var_sum"]
            tube_coverage = tube["tube_coverage"]

            # Compute deltas
            tube_var_delta = None
            tube_var_delta_baseline = None
            if prev_tube_var is not None:
                tube_var_delta = tube_var - prev_tube_var
            if baseline_tube_var is not None:
                tube_var_delta_baseline = tube_var - baseline_tube_var

            # Determine status
            if round_num == 1:
                status = "FIRST_ROUND"
            elif tube_var_delta == 0:
                status = "NO_CHANGE"
            elif tube_var_delta < 0:
                status = "IMPROVING"
            else:
                status = "WORSENING"

            metrics = {
                "round_id": round_id,
                "round_num": round_num,
                "status": status,
                "tube": {
                    "tube_var_sum": tube_var,
                    "tube_coverage": tube_coverage,
                    "tube_var_delta_prev": tube_var_delta,
                    "tube_var_delta_baseline": tube_var_delta_baseline,
                    **tube,
                },
                "counts": {
                    "episodes": len(results),
                    "successes": successes,
                    "failures": failures,
                },
                "plan": {
                    "targets_selected": len(selected),
                    "episodes_per_target": episodes_per_target,
                },
            }

            (round_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
            all_metrics.append(metrics)

            # Generate round receipt
            round_config = {
                "grid_version": grid.get("version", "unknown"),
                "targets_per_round": targets_per_round,
                "episodes_per_budget_unit": episodes_per_target,
                "seed": seed,
                "use_synthetic": synthetic,
            }
            receipt = build_round_receipt(round_dir, round_config)
            (round_dir / "round_receipt.json").write_text(json.dumps(receipt, indent=2))
            round_receipts.append(receipt)

            # Update for next round
            prev_tube_var = tube_var
            if baseline_tube_var is None:
                baseline_tube_var = tube_var

            # Print round summary
            delta_str = f"Δ={tube_var_delta:+.6f}" if tube_var_delta is not None else ""
            status_color = GREEN if status == "IMPROVING" else (YELLOW if status in ["NO_CHANGE", "FIRST_ROUND"] else RED)
            click.echo(f"    uncertainty: {tube_var:.6f} {delta_str}  coverage: {tube_coverage:.3f}  {status_color}{status}{RESET}")
            click.echo(f"    {GREEN}✓ {successes}{RESET} success  {RED}✗ {failures}{RESET} fail")
            click.echo()

        # Build run receipt
        run_receipt = build_run_receipt(run_path, round_receipts)
        (run_path / "run_receipt.json").write_text(json.dumps(run_receipt, indent=2))

        # Generate proof if --prove flag is set
        capsule = None
        if prove:
            click.echo(f"\n{DIM}[4/4] Generating cryptographic proof...{RESET}")
            try:
                from capseal.eval_adapter import (
                    EvalAdapter, build_eval_capsule, save_eval_capsule, verify_eval_capsule,
                )

                adapter = EvalAdapter()

                # Collect round results from the completed eval
                round_results = []
                for round_dir in sorted((run_path / "rounds").glob("R*")):
                    metrics_file = round_dir / "metrics.json"
                    if not metrics_file.exists():
                        continue
                    metrics = json.loads(metrics_file.read_text())
                    results_csv = round_dir / "agent_results.csv"

                    # Count successes and failures from the metrics
                    n_success = metrics.get("counts", {}).get("successes", 0)
                    n_fail = metrics.get("counts", {}).get("failures", 0)

                    # Use round-specific posteriors if they exist, otherwise use global
                    posteriors_path = round_dir / "beta_posteriors.npz"
                    if not posteriors_path.exists():
                        posteriors_path = run_path / "beta_posteriors.npz"

                    round_results.append({
                        "round_dir": round_dir,
                        "round_id": round_dir.name,
                        "metrics": metrics.get("tube", metrics),
                        "status": metrics.get("status", "FIRST_ROUND"),
                        "n_successes": n_success,
                        "n_failures": n_fail,
                        "posteriors_path": posteriors_path,
                        "plan_path": round_dir / "active_sampling_plan.json",
                        "results_path": results_csv,
                    })

                # Build eval config
                eval_config = {
                    "grid_version": grid.get("version", "unknown"),
                    "targets_per_round": targets_per_round,
                    "episodes_per_target": episodes_per_target,
                    "seed": seed,
                }

                # Generate the proof
                trace_artifacts = adapter.simulate_trace(round_results, eval_config)
                commitment = adapter.commit_to_trace(
                    trace_artifacts,
                    row_archive_dir=run_path / "row_archive",
                )
                proof_artifacts = adapter.generate_proof(trace_artifacts, commitment)
                capsule = build_eval_capsule(trace_artifacts, commitment, proof_artifacts, run_path)
                save_eval_capsule(capsule, run_path / "eval_capsule.json")

                click.echo(f"      Capsule saved: {run_path / 'eval_capsule.json'}")

            except Exception as e:
                click.echo(f"{RED}Proof generation failed: {e}{RESET}")
                import traceback
                traceback.print_exc()

        # Write summary.csv
        import csv
        summary_path = run_path / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "run_uuid", "round_id", "seed", "tube_coverage", "tube_var",
                "tube_var_delta", "tube_var_delta_prev", "tube_var_delta_baseline",
                "targets_selected", "total_episodes", "status"
            ])
            for m in all_metrics:
                writer.writerow([
                    run_uuid,
                    m["round_id"],
                    seed,
                    m["tube"]["tube_coverage"],
                    m["tube"]["tube_var_sum"],
                    m["tube"].get("tube_var_delta_prev", ""),
                    m["tube"].get("tube_var_delta_prev", ""),
                    m["tube"].get("tube_var_delta_baseline", ""),
                    m["plan"]["targets_selected"],
                    m["counts"]["episodes"],
                    m["status"],
                ])

        # Copy final posteriors to .capseal/models/
        import shutil
        models_dir = target_path / ".capseal" / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        final_posteriors_path = models_dir / "beta_posteriors.npz"
        shutil.copy2(run_path / "beta_posteriors.npz", final_posteriors_path)

        # Final summary
        click.echo(f"{CYAN}{'═' * 65}{RESET}")
        if prove and capsule:
            click.echo(f"{CYAN}  EVALUATION COMPLETE (PROOF-CARRYING){RESET}")
        else:
            click.echo(f"{CYAN}  EVALUATION COMPLETE{RESET}")
        click.echo(f"{CYAN}{'═' * 65}{RESET}")
        click.echo(f"  Run ID:         {run_uuid}")
        click.echo(f"  Total rounds:   {rounds}")
        click.echo(f"  Uncertainty:    {prev_tube_var:.6f}")
        click.echo(f"  Coverage:       {tube_coverage:.3f}")
        click.echo(f"  Chain hash:     {run_receipt['chain_hash'][:16]}...")
        if prove and capsule:
            click.echo(f"  Capsule hash:   {capsule['capsule_hash'][:16]}...")
            verified = capsule.get("verification", {}).get("constraints_valid", False)
            verified_str = f"{GREEN}Yes{RESET}" if verified else f"{RED}No{RESET}"
            click.echo(f"  Proof verified: {verified_str}")
            click.echo()
            click.echo(f"  {DIM}The cryptographic proof attests that:{RESET}")
            click.echo(f"  {DIM}- All {rounds} rounds executed in sequence{RESET}")
            click.echo(f"  {DIM}- Model updates were correctly chained (no tampering){RESET}")
            click.echo(f"  {DIM}- Episode counts match declared totals{RESET}")
            click.echo(f"  {DIM}- Final model matches the declared hash{RESET}")
        click.echo(f"\n  {GREEN}Model saved to:{RESET}")
        click.echo(f"    {final_posteriors_path}")
        click.echo(f"\n  {DIM}Future 'capseal review --gate' will use this model.{RESET}")
        click.echo(f"{CYAN}{'═' * 65}{RESET}\n")

    except ImportError as e:
        click.echo(f"{RED}Module not available: {e}{RESET}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
    except subprocess.TimeoutExpired:
        click.echo(f"{RED}Semgrep timed out{RESET}")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"{RED}Error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)


def _run_eval_episode(
    target_path: Path,
    finding: dict,
    run_path: Path,
    rng: "np.random.Generator",
) -> bool:
    """Run a single evaluation episode on a finding.

    Attempts to generate a patch for the finding and verify it.
    Returns True if patch succeeds, False if it fails.
    """
    import os

    try:
        from capseal.refactor_engine import (
            generate_refactor_plan, run_multi_agent_patches,
        )

        file_path = finding.get("path", "")
        if not file_path:
            return True

        full_path = target_path / file_path
        if not full_path.exists():
            return True

        # Auto-detect provider/model
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider, model = "anthropic", "claude-sonnet-4-20250514"
        elif os.environ.get("OPENAI_API_KEY"):
            provider, model = "openai", "gpt-4o-mini"
        else:
            # No API key available - assume failure
            return False

        single_finding = [finding]

        try:
            plan = generate_refactor_plan(
                findings=single_finding,
                trace_root=f"eval-{finding.get('check_id', 'unknown')[:8]}",
                aggregate_hash="eval",
                provider=provider,
                model=model,
            )

            if not plan.items:
                return True

            results = run_multi_agent_patches(
                plan=plan,
                project_dir=target_path,
                provider=provider,
                model=model,
                enable_repair=True,
                enable_suppression_memos=True,
                enable_ast_validation=True,
            )

            for r in results:
                if r.final_status == "VALID":
                    return True
                elif r.final_status == "FAIL":
                    return False

            return True

        except Exception:
            return False

    except Exception:
        return False
