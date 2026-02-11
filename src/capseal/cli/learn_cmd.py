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


def _load_capseal_env(target_path: Path) -> None:
    """Load API keys from .capseal/.env if not already in environment.

    This allows users to run capseal commands without manually exporting
    API keys every time they open a new terminal.
    """
    import os

    env_file = target_path / ".capseal" / ".env"
    if not env_file.exists():
        return

    try:
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Only set if not already in environment
                    if key and not os.environ.get(key):
                        os.environ[key] = value
    except Exception:
        pass  # Silently ignore any errors reading the file


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
@click.option("--profile", type=click.Choice(["security", "quality", "bugs", "all", "custom"]),
              default=None, help="Scan profile (default: from config or 'auto')")
@click.option("--rules", type=click.Path(exists=True), default=None, help="Custom semgrep rules path")
@click.option("--test-cmd", type=str, default=None, help="Shell command to validate patches (e.g. 'pytest')")
@click.option("--from-git", is_flag=True, help="Learn from git history (free, instant — no LLM calls)")
@click.option("--parallel", "-j", type=int, default=1, help="Max concurrent episodes per round (default: 1)")
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
    profile: str | None = None,
    rules: str | None = None,
    test_cmd: str | None = None,
    from_git: bool = False,
    parallel: int = 1,
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
    import os
    import subprocess
    import uuid

    # ANSI colors
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    target_path = Path(path).expanduser().resolve()

    # ── Git history learning (no LLM needed) ──────────────────────
    if from_git:
        from .git_learner import learn_from_git
        _run_git_learn(target_path, quiet, CYAN, GREEN, YELLOW, RED, DIM, BOLD, RESET)
        return

    # Check for API key - first try environment, then .capseal/.env fallback
    _load_capseal_env(target_path)

    # Detect auth mode and CLI binary for subscription users
    import shutil as _shutil
    has_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    config_path = target_path / ".capseal" / "config.json"
    config_json_early = None
    cli_binary = None
    if config_path.exists():
        try:
            config_json_early = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    if not has_api_key:
        # Try to find a CLI binary for subscription mode
        provider = (config_json_early or {}).get("provider", "")
        cli_map = {
            "anthropic": "claude",
            "openai": "codex",
            "google": "gemini",
        }
        preferred = cli_map.get(provider)
        if preferred and _shutil.which(preferred):
            cli_binary = preferred
        else:
            # Try any available CLI
            for binary in ("claude", "codex", "gemini"):
                if _shutil.which(binary):
                    cli_binary = binary
                    break

        if not cli_binary:
            click.echo(f"{RED}Error: No API key or provider CLI found.{RESET}", err=True)
            click.echo("", err=True)
            click.echo("Training requires either:", err=True)
            click.echo("  • A provider CLI (claude, codex, gemini) — uses your subscription", err=True)
            click.echo("  • An API key (ANTHROPIC_API_KEY, OPENAI_API_KEY)", err=True)
            click.echo("", err=True)
            click.echo("Install your provider's CLI or set an API key, then try again.", err=True)
            raise SystemExit(1)

        if not quiet:
            click.echo(f"  {DIM}Using {cli_binary} CLI (subscription mode){RESET}")

    import numpy as np

    from ..budget import BudgetTracker, estimate_learning_cost, PRICING_PRESETS
    from ..episode_runner import EpisodeRunner, EpisodeRunnerConfig

    # target_path already set above for env loading

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
        from capseal.shared.scoring import compute_acquisition_score, select_targets, compute_tube_metrics
        from capseal.shared.features import extract_patch_features, discretize_features, features_to_grid_idx, grid_idx_to_features
        from capseal.shared.receipts import build_round_receipt, build_run_receipt

        # Step 1: Scan with Semgrep
        from .scan_profiles import build_semgrep_args, PROFILE_DISPLAY

        # Load config for default profile
        config_json = None
        config_path = target_path / ".capseal" / "config.json"
        if config_path.exists():
            try:
                config_json = json.loads(config_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        effective_profile = profile or (config_json or {}).get("scan_profile")

        # Resolve test_cmd from config if not set via CLI
        effective_test_cmd = test_cmd or (config_json or {}).get("test_cmd")
        if effective_test_cmd and not quiet:
            click.echo(f"{DIM}  Test command: {effective_test_cmd}{RESET}")

        if not quiet:
            profile_label = PROFILE_DISPLAY.get(effective_profile, effective_profile or "auto")
            click.echo(f"{DIM}[1/4] Scanning codebase with Semgrep ({profile_label})...{RESET}")

        semgrep_cmd = build_semgrep_args(target_path, profile=profile, custom_rules=rules, config_json=config_json)
        result = subprocess.run(
            semgrep_cmd,
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

        # Build profile name map for display
        grid_idx_to_name = {}
        for gidx in grid_idx_to_findings:
            feats = grid_idx_to_features(gidx)
            grid_idx_to_name[gidx] = _describe_features(feats)

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
            test_cmd=effective_test_cmd,
            cli_binary=cli_binary,
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

        # ── Live training display ─────────────────────────────────────
        profile_stats = {}  # name -> {passes, fails}
        for gidx in grid_idx_to_findings:
            pname = grid_idx_to_name.get(gidx, f"grid-{gidx}")
            if pname not in profile_stats:
                profile_stats[pname] = {"passes": 0, "fails": 0}

        _live_round = [0]
        _live_latest = [""]
        _live_stop_reason = [""]

        def _build_live_panel():
            from rich.table import Table as RTable
            from rich.panel import Panel as RPanel
            from rich.console import Group as RGroup
            from rich.text import Text as RText
            from rich.box import ROUNDED as RBOX

            # Progress bar
            pct = _live_round[0] / rounds if rounds > 0 else 0
            bar_w = 30
            filled = int(pct * bar_w)
            bar = "━" * filled + "─" * (bar_w - filled)
            remaining = budget_tracker.remaining_budget
            header = RText()
            header.append(f"Round {_live_round[0]}/{rounds} ", style="bold")
            header.append(bar + " ", style="cyan")
            header.append(f"${remaining:.2f} remaining", style="dim")

            # Risk profile table
            table = RTable(box=RBOX, expand=True, show_header=True, header_style="bold")
            table.add_column("Risk Profile", style="bold", min_width=18)
            table.add_column("Pass", justify="center", style="green", min_width=5)
            table.add_column("Fail", justify="center", style="red", min_width=5)
            table.add_column("p_fail", justify="center", min_width=7)
            table.add_column("Status", justify="center", min_width=12)

            for pname, data in profile_stats.items():
                total = data["passes"] + data["fails"]
                if total == 0:
                    pf_str = "—"
                    status = "[dim]⏳ waiting[/dim]"
                else:
                    pf = (data["fails"] + 1) / (total + 2)  # Beta mean with prior
                    pf_str = f"{pf:.2f}"
                    if pf > 0.6:
                        status = "[red]✗ risky[/red]"
                    elif pf > 0.3:
                        status = "[yellow]⚠ learning[/yellow]"
                    else:
                        status = "[green]✓ low risk[/green]"
                table.add_row(pname, str(data["passes"]), str(data["fails"]), pf_str, status)

            latest = RText(f"\nLatest: {_live_latest[0]}", style="dim") if _live_latest[0] else RText("")
            stop = RText(f"\n{_live_stop_reason[0]}", style="yellow") if _live_stop_reason[0] else RText("")

            return RPanel(
                RGroup(header, "", table, latest, stop),
                title="[bold cyan]═══ TRAINING ═══[/bold cyan]",
                border_style="cyan",
                expand=False,
            )

        # Import Live display
        from rich.live import Live
        from rich.console import Console as RConsole
        from .tui_compat import is_inside_tui, emit_tui_event

        _in_tui = is_inside_tui()
        _live_console = RConsole()

        # Skip Rich Live panels when inside the Rust TUI — sidebar handles progress
        use_live = not quiet and not _in_tui
        live_ctx = Live(_build_live_panel(), refresh_per_second=4, console=_live_console) if use_live else None
        if live_ctx:
            live_ctx.__enter__()

        emit_tui_event("train_start", f"round 0/{rounds}")

        try:
            for round_num in range(1, rounds + 1):
                # Check budget
                if budget_tracker.budget_exhausted:
                    _live_stop_reason[0] = "Budget exhausted — stopping early"
                    if quiet:
                        click.echo(f"  {YELLOW}Budget exhausted - stopping early{RESET}")
                    break

                # Check time limit
                if time_limit_seconds:
                    elapsed = (datetime.datetime.now() - start_time).total_seconds()
                    if elapsed >= time_limit_seconds:
                        _live_stop_reason[0] = "Time limit reached — stopping early"
                        if quiet:
                            click.echo(f"  {YELLOW}Time limit reached - stopping early{RESET}")
                        break

                _live_round[0] = round_num

                # Build per-profile summary for enriched events
                _profile_parts = []
                for _pn, _pd in profile_stats.items():
                    _pt = _pd["passes"] + _pd["fails"]
                    if _pt > 0:
                        _pf = (_pd["fails"] + 1) / (_pt + 2)
                        _profile_parts.append(f"{_pn}={_pf:.2f}")
                _profile_str = " ".join(_profile_parts)
                emit_tui_event("train_progress", f"round {round_num}/{rounds} {_profile_str}")

                if live_ctx:
                    live_ctx.update(_build_live_panel())

                round_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                round_id = f"R{round_num:04d}_{round_timestamp}"
                round_dir = run_path / "rounds" / round_id
                round_dir.mkdir(parents=True, exist_ok=True)

                if _in_tui and not quiet:
                    # Compact plain-text output for TUI embedded terminal
                    parts = " | ".join(f"{_pn}: p={(_pd['fails']+1)/(_pd['passes']+_pd['fails']+2):.2f}" for _pn, _pd in profile_stats.items() if _pd["passes"] + _pd["fails"] > 0)
                    click.echo(f"  [{round_num}/{rounds}] {parts}" if parts else f"  [{round_num}/{rounds}] starting...")
                elif quiet:
                    remaining = budget_tracker.remaining_budget
                    budget_str = f"${remaining:.2f} remaining" if remaining else ""
                    click.echo(f"  {CYAN}Round {round_num}/{rounds}{RESET} {DIM}{budget_str}{RESET}")

                # Select targets using acquisition function
                scores = compute_acquisition_score(alpha, beta)
                K = min(targets_per_round, len(grid_idx_to_findings))

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

                # Run episodes (parallel if --parallel > 1)
                results = []
                round_successes = 0
                round_failures = 0
                _display_lock = __import__("threading").Lock()

                def _run_one(target_idx, grid_idx):
                    """Run a single episode — safe to call from threads."""
                    grid_idx = int(grid_idx)
                    finding_list = grid_idx_to_findings.get(grid_idx, [])
                    if not finding_list:
                        return None
                    finding = finding_list[0]
                    episode_id = f"{round_id}_{target_idx:04d}"
                    return (grid_idx, finding, runner.run_episode(episode_id, grid_idx, finding))

                if parallel > 1:
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    episode_futures = []
                    with ThreadPoolExecutor(max_workers=parallel) as executor:
                        for target_idx, grid_idx in enumerate(selected):
                            if budget_tracker.budget_exhausted:
                                break
                            episode_futures.append(
                                executor.submit(_run_one, target_idx, int(grid_idx))
                            )
                        for future in as_completed(episode_futures):
                            try:
                                result_tuple = future.result(timeout=120)
                            except Exception:
                                continue
                            if result_tuple is None:
                                continue
                            gidx, finding, episode_result = result_tuple
                            results.append({
                                "round_id": round_id,
                                "grid_idx": gidx,
                                "episode_idx": 0,
                                "success": 1 if episode_result.success else 0,
                                "cost": episode_result.cost,
                            })
                            pname = grid_idx_to_name.get(gidx, f"grid-{gidx}")
                            with _display_lock:
                                if episode_result.success:
                                    round_successes += 1
                                    beta[gidx] += 1
                                    profile_stats.setdefault(pname, {"passes": 0, "fails": 0})["passes"] += 1
                                    icon = "✓"
                                else:
                                    round_failures += 1
                                    alpha[gidx] += 1
                                    profile_stats.setdefault(pname, {"passes": 0, "fails": 0})["fails"] += 1
                                    icon = "✗"
                                check_id = finding.get("check_id", "")
                                short_check = check_id.split(".")[-1] if check_id else pname
                                file_name = Path(finding.get("path", "")).name
                                _live_latest[0] = f"{icon} {short_check} in {file_name}"
                            if live_ctx:
                                live_ctx.update(_build_live_panel())
                else:
                    # Sequential (default)
                    for target_idx, grid_idx in enumerate(selected):
                        grid_idx = int(grid_idx)
                        if budget_tracker.budget_exhausted:
                            break
                        result_tuple = _run_one(target_idx, grid_idx)
                        if result_tuple is None:
                            continue
                        gidx, finding, episode_result = result_tuple
                        results.append({
                            "round_id": round_id,
                            "grid_idx": gidx,
                            "episode_idx": 0,
                            "success": 1 if episode_result.success else 0,
                            "cost": episode_result.cost,
                        })
                        pname = grid_idx_to_name.get(gidx, f"grid-{gidx}")
                        if episode_result.success:
                            round_successes += 1
                            beta[gidx] += 1
                            profile_stats.setdefault(pname, {"passes": 0, "fails": 0})["passes"] += 1
                            icon = "✓"
                        else:
                            round_failures += 1
                            alpha[gidx] += 1
                            profile_stats.setdefault(pname, {"passes": 0, "fails": 0})["fails"] += 1
                            icon = "✗"
                        check_id = finding.get("check_id", "")
                        short_check = check_id.split(".")[-1] if check_id else pname
                        file_name = Path(finding.get("path", "")).name
                        _live_latest[0] = f"{icon} {short_check} in {file_name}"
                        if live_ctx:
                            live_ctx.update(_build_live_panel())

                total_successes += round_successes
                total_failures += round_failures

                # Append to episode_history.jsonl for instant risk preview
                history_path = target_path / ".capseal" / "models" / "episode_history.jsonl"
                history_path.parent.mkdir(parents=True, exist_ok=True)
                with open(history_path, "a") as hf:
                    for r in results:
                        gidx = r["grid_idx"]
                        finding_list = grid_idx_to_findings.get(gidx, [])
                        finding = finding_list[0] if finding_list else {}
                        pname = grid_idx_to_name.get(gidx, f"grid-{gidx}")
                        hf.write(json.dumps({
                            "grid_idx": gidx,
                            "profile_name": pname,
                            "success": bool(r["success"]),
                            "cost": r["cost"],
                            "description": f"{finding.get('check_id', 'unknown')} in {Path(finding.get('path', 'unknown')).name}",
                            "finding_rule": finding.get("check_id", ""),
                            "finding_file": finding.get("path", ""),
                            "round_num": round_num,
                        }) + "\n")

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
                    n_episodes=total_successes + total_failures,
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

                if quiet:
                    click.echo(f"    {GREEN}✓ {round_successes}{RESET} success  {RED}✗ {round_failures}{RESET} fail  cost: ${budget_tracker.total_cost:.2f}")

        finally:
            if live_ctx:
                live_ctx.__exit__(None, None, None)
            total_episodes = total_successes + total_failures
            emit_tui_event("train_complete", f"{_live_round[0]} rounds, {total_episodes} episodes")

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

        # Package into .cap file
        from .cap_format import create_run_cap_file

        cap_path = run_path.parent / f"{run_path.name}.cap"
        cap_manifest = create_run_cap_file(
            run_dir=run_path,
            output_path=cap_path,
            run_type="learn",
            extras={
                "rounds": len(round_receipts),
                "episodes": total_successes + total_failures,
                "successes": total_successes,
                "failures": total_failures,
                "cost": budget_tracker.total_cost,
            },
        )

        # Create "latest.cap" symlink
        latest_cap_link = target_path / ".capseal" / "runs" / "latest.cap"
        if latest_cap_link.is_symlink() or latest_cap_link.exists():
            latest_cap_link.unlink()
        latest_cap_link.symlink_to(cap_path.name)

        # Generate proof if requested
        capsule = None
        if prove and round_receipts:
            if not quiet:
                click.echo(f"\n{DIM}[4/4] Generating cryptographic proof...{RESET}")
            try:
                from capseal.eval_adapter import EvalAdapter, build_eval_capsule, save_eval_capsule

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
        click.echo()
        click.echo(f"  {GREEN}Sealed:{RESET} {cap_path.relative_to(target_path)}")
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
    from capseal.shared.features import grid_idx_to_features

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


def _run_git_learn(
    target_path: Path,
    quiet: bool,
    CYAN: str, GREEN: str, YELLOW: str, RED: str, DIM: str, BOLD: str, RESET: str,
) -> None:
    """Run git-history-based learning (free, instant)."""
    import json
    import datetime
    import numpy as np
    from .git_learner import learn_from_git

    if not quiet:
        click.echo(f"\n{CYAN}{'═' * 65}{RESET}")
        click.echo(f"{CYAN}  LEARNING FROM GIT HISTORY{RESET}")
        click.echo(f"{CYAN}{'═' * 65}{RESET}")
        click.echo(f"  Target:   {target_path}")
        click.echo(f"  Cost:     $0.00 (no LLM calls)")
        click.echo(f"{CYAN}{'═' * 65}{RESET}\n")

    results = learn_from_git(str(target_path), quiet=quiet)

    if not results:
        click.echo(f"  {YELLOW}No learnable commits found in git history.{RESET}")
        click.echo(f"  {DIM}Try 'capseal learn .' for LLM-based training instead.{RESET}")
        return

    # Convert results to Beta posteriors
    n_points = 1024
    alpha = np.ones(n_points, dtype=np.int64)
    beta = np.ones(n_points, dtype=np.int64)

    total_passes = 0
    total_fails = 0
    for profile_name, data in results.items():
        grid_idx = data.get("grid_idx", 0)
        if 0 <= grid_idx < n_points:
            alpha[grid_idx] += data["fails"]  # alpha = failures (p_fail)
            beta[grid_idx] += data["passes"]  # beta = successes
        total_passes += data["passes"]
        total_fails += data["fails"]

    n_episodes = total_passes + total_fails

    # Save posteriors
    models_dir = target_path / ".capseal" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        models_dir / "beta_posteriors.npz",
        alpha=alpha,
        beta=beta,
        grid_version="semgrep_scan",
        run_uuid="git-history",
        n_episodes=n_episodes,
    )

    # Summary
    if not quiet:
        click.echo(f"\n{CYAN}{'═' * 65}{RESET}")
        click.echo(f"{CYAN}  LEARNING COMPLETE{RESET}")
        click.echo(f"{CYAN}{'═' * 65}{RESET}")
        click.echo(f"  Commits analyzed: {n_episodes}")
        click.echo(f"  Clean commits:   {total_passes}")
        click.echo(f"  Risky commits:   {total_fails}")
        click.echo()

        click.echo(f"  {BOLD}Risk profiles learned:{RESET}")
        for profile_name, data in results.items():
            total = data["passes"] + data["fails"]
            if total == 0:
                continue
            pf = (data["fails"] + 1) / (total + 2)
            if pf > 0.6:
                color = RED
                label = "risky"
            elif pf > 0.3:
                color = YELLOW
                label = "moderate"
            else:
                color = GREEN
                label = "low risk"
            click.echo(f"  {color}•{RESET} {profile_name}: {data['passes']} pass, {data['fails']} fail ({label})")

        click.echo()
        click.echo(f"  Model saved to {models_dir / 'beta_posteriors.npz'}")
        click.echo(f"  {DIM}Future sessions will use this model for gating.{RESET}")
        click.echo(f"{CYAN}{'═' * 65}{RESET}\n")


__all__ = ["learn_command"]
