"""capseal autopilot - Full pipeline with zero configuration.

One command, zero decisions:
    capseal autopilot .                    # interactive (asks before applying)
    capseal autopilot . --apply            # non-interactive (applies automatically)
    capseal autopilot . --budget 10        # higher budget for more learning
    capseal autopilot . --ci               # CI mode: exit 1 if issues, 0 if clean
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click


def _capseal_bin() -> str:
    """Find the capseal binary, preferring the one next to our Python."""
    # Check same directory as the current Python interpreter
    bin_dir = Path(sys.executable).parent
    candidate = bin_dir / "capseal"
    if candidate.exists():
        return str(candidate)
    # Fall back to PATH
    found = shutil.which("capseal")
    if found:
        return found
    return "capseal"


# ANSI
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Provider detection order and defaults
PROVIDERS = [
    ("anthropic", "ANTHROPIC_API_KEY", "claude-opus-4-6"),
    ("openai", "OPENAI_API_KEY", "chatgpt-5.2"),
    ("google", "GOOGLE_API_KEY", "gemini-3-flash"),
]


def _detect_provider() -> tuple[str, str, str] | None:
    """Detect provider from environment variables.

    Returns (provider_name, env_var, default_model) or None.
    """
    for name, env_var, default_model in PROVIDERS:
        if os.environ.get(env_var):
            return name, env_var, default_model
    return None


def _auto_init(target: Path, provider: str, env_var: str, model: str, scan_profile: str | None = None) -> None:
    """Non-interactive workspace init."""
    capseal_dir = target / ".capseal"
    for subdir in ["models", "runs", "policies"]:
        (capseal_dir / subdir).mkdir(parents=True, exist_ok=True)

    config = {
        "version": "0.2.0",
        "provider": provider,
        "model": model,
        "auth_method": "api_key",
        "api_key_env": env_var,
        "plan": "free",
        "agents": ["cli-only"],
        "integrations": {},
        "gate": {"threshold": 0.6, "uncertainty_threshold": 0.15},
        "learn": {"default_rounds": 3, "default_budget": 3.0},
    }
    if scan_profile:
        config["scan_profile"] = scan_profile
    (capseal_dir / "config.json").write_text(json.dumps(config, indent=2))

    policy = {
        "name": "default",
        "gate_threshold": 0.6,
        "uncertainty_threshold": 0.15,
        "auto_apply": False,
        "require_verification": True,
    }
    (capseal_dir / "policies" / "default.json").write_text(json.dumps(policy, indent=2))


def _run_learn(target: Path, rounds: int, budget: float, profile: str | None = None, rules: str | None = None) -> dict:
    """Run capseal learn and return summary."""
    cmd = [_capseal_bin(), "learn", str(target),
           "--rounds", str(rounds), "--budget", str(budget)]
    if profile:
        cmd.extend(["--profile", profile])
    if rules:
        cmd.extend(["--rules", rules])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Parse success/failure counts from output
    successes = 0
    failures = 0
    for line in result.stdout.split("\n"):
        if "✓" in line and "success" in line.lower():
            # Parse "    ✓ 2 success  ✗ 1 fail" pattern
            parts = line.split()
            for i, part in enumerate(parts):
                if "success" in part.lower() and i > 0:
                    try:
                        successes += int(parts[i - 1].strip("✓").strip())
                    except (ValueError, IndexError):
                        pass
                if "fail" in part.lower() and i > 0:
                    try:
                        failures += int(parts[i - 1].strip("✗").strip())
                    except (ValueError, IndexError):
                        pass

    return {
        "returncode": result.returncode,
        "successes": successes,
        "failures": failures,
        "total": successes + failures,
        "success_rate": successes / max(1, successes + failures),
    }


def _run_fix_dryrun(target: Path, profile: str | None = None, rules: str | None = None) -> dict:
    """Run capseal fix --dry-run --json and return parsed result."""
    cmd = [_capseal_bin(), "fix", str(target), "--dry-run", "--json"]
    if profile:
        cmd.extend(["--profile", profile])
    if rules:
        cmd.extend(["--rules", rules])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Parse JSON output (may have multiple lines, take last valid JSON)
    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return {"status": "error", "error": result.stderr or "No output"}


def _run_fix_apply(target: Path, profile: str | None = None, rules: str | None = None) -> dict:
    """Run capseal fix --apply --json and return parsed result."""
    cmd = [_capseal_bin(), "fix", str(target), "--apply", "--json"]
    if profile:
        cmd.extend(["--profile", profile])
    if rules:
        cmd.extend(["--rules", rules])
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    return {"status": "error", "error": result.stderr or "No output"}


def _run_verify(target: Path) -> dict:
    """Verify the latest .cap file."""
    latest_cap = target / ".capseal" / "runs" / "latest.cap"
    if not latest_cap.exists():
        return {"verified": False, "error": "No .cap file found"}

    # Resolve symlink
    cap_path = latest_cap.resolve()

    result = subprocess.run(
        [_capseal_bin(), "verify", str(cap_path), "--json"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    for line in reversed(result.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                data = json.loads(line)
                data["verified"] = data.get("status") == "VERIFIED"
                data["cap_path"] = str(cap_path.relative_to(target))
                return data
            except json.JSONDecodeError:
                continue

    return {
        "verified": result.returncode == 0,
        "cap_path": str(cap_path.relative_to(target)) if cap_path.exists() else "",
    }


@click.command("autopilot")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--apply", "auto_apply", is_flag=True, help="Apply fixes without asking")
@click.option("--budget", type=float, default=3.0, help="Learning budget in dollars (default: $3)")
@click.option("--rounds", type=int, default=3, help="Learning rounds (default: 3)")
@click.option("--ci", is_flag=True, help="CI mode: exit 1 if issues found, 0 if clean")
@click.option("--json", "output_json", is_flag=True, help="Output JSON summary")
@click.option("--profile", type=click.Choice(["security", "quality", "bugs", "all", "custom"]),
              default=None, help="Scan profile (default: from config or 'auto')")
@click.option("--rules", type=click.Path(exists=True), default=None, help="Custom semgrep rules path")
def autopilot_command(
    path: str,
    auto_apply: bool,
    budget: float,
    rounds: int,
    ci: bool,
    output_json: bool,
    profile: str | None = None,
    rules: str | None = None,
) -> None:
    """Run the full CapSeal pipeline with zero configuration.

    Detects your LLM provider from environment variables, initializes
    a workspace, learns your codebase, scans for issues, gates risky
    patches, and generates a cryptographic receipt.

    \b
    Examples:
        capseal autopilot .                    # interactive
        capseal autopilot . --apply            # auto-apply safe fixes
        capseal autopilot . --budget 10        # more learning budget
        capseal autopilot . --ci               # CI mode (exit codes)

    \b
    Requires one of:
        export ANTHROPIC_API_KEY=sk-ant-...
        export OPENAI_API_KEY=sk-...
        export GOOGLE_API_KEY=AI...
    """
    target = Path(path).expanduser().resolve()

    # ── Detect provider ───────────────────────────────────────────
    detected = _detect_provider()

    if not detected:
        # Check for subscription mode — load config and look for CLI binary
        config_path = target / ".capseal" / "config.json"
        cfg = None
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        cli_map = {"anthropic": "claude", "openai": "codex", "google": "gemini"}
        provider_from_config = (cfg or {}).get("provider", "")
        preferred_cli = cli_map.get(provider_from_config)

        cli_binary = None
        if preferred_cli and shutil.which(preferred_cli):
            cli_binary = preferred_cli
        else:
            for binary in ("claude", "codex", "gemini"):
                if shutil.which(binary):
                    cli_binary = binary
                    break

        if cli_binary:
            # Subscription mode — derive provider info from CLI binary
            cli_to_provider = {"claude": "anthropic", "codex": "openai", "gemini": "google"}
            provider = cli_to_provider.get(cli_binary, provider_from_config or "anthropic")
            env_var = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}[provider]
            model = (cfg or {}).get("model", {"anthropic": "claude-opus-4-6", "openai": "chatgpt-5.2", "google": "gemini-3-flash"}[provider])
        else:
            click.echo(f"{RED}Error: No API key or provider CLI found.{RESET}", err=True)
            click.echo("", err=True)
            click.echo("Autopilot requires either:", err=True)
            click.echo("  • A provider CLI (claude, codex, gemini) — uses your subscription", err=True)
            click.echo("  • An API key (ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY)", err=True)
            click.echo("", err=True)
            click.echo("Install your provider's CLI or set an API key, then try again.", err=True)
            raise SystemExit(1)
    else:
        provider, env_var, model = detected
    provider_display = {
        "anthropic": "Anthropic",
        "openai": "OpenAI",
        "google": "Google",
    }

    # ── Banner ────────────────────────────────────────────────────
    if not output_json:
        click.echo()
        click.echo(f"{CYAN}{'═' * 55}{RESET}")
        click.echo(f"{CYAN}  CAPSEAL AUTOPILOT{RESET}")
        click.echo(f"{CYAN}{'═' * 55}{RESET}")
        click.echo(f"  Target:   {target}")
        click.echo(f"  Provider: {provider_display.get(provider, provider)} (auto-detected)")
        click.echo(f"  Model:    {model}")
        if ci:
            click.echo(f"  Mode:     {YELLOW}CI{RESET} (non-interactive)")
        elif auto_apply:
            click.echo(f"  Mode:     {GREEN}Auto-apply{RESET}")
        click.echo(f"{CYAN}{'═' * 55}{RESET}")
        click.echo()

    # Track results for final summary
    summary = {
        "target": str(target),
        "provider": provider,
        "model": model,
        "steps": {},
    }

    # ── Step 1: Init ──────────────────────────────────────────────
    capseal_dir = target / ".capseal"
    if capseal_dir.exists():
        if not output_json:
            click.echo(f"[1/4] Workspace exists             {GREEN}✓{RESET}")
        summary["steps"]["init"] = "skipped"
    else:
        if not output_json:
            click.echo(f"[1/4] Initializing workspace...    ", nl=False)
        _auto_init(target, provider, env_var, model, scan_profile=profile)
        if not output_json:
            click.echo(f"{GREEN}✓{RESET}")
        summary["steps"]["init"] = "created"

    # ── Resolve profile from config ──────────────────────────────
    effective_profile = profile
    if not effective_profile:
        config_path = capseal_dir / "config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
                effective_profile = cfg.get("scan_profile")
            except (json.JSONDecodeError, OSError):
                pass

    # ── Step 2: Learn ─────────────────────────────────────────────
    posteriors_path = capseal_dir / "models" / "beta_posteriors.npz"
    if posteriors_path.exists():
        if not output_json:
            click.echo(f"[2/4] Risk model exists             {GREEN}✓{RESET}")
        summary["steps"]["learn"] = "skipped"
    else:
        if not output_json:
            click.echo(f"[2/4] Learning risk model ({rounds} rounds)...")

        learn_result = _run_learn(target, rounds=rounds, budget=budget, profile=effective_profile, rules=rules)

        if learn_result["returncode"] != 0 and not posteriors_path.exists():
            if not output_json:
                click.echo(f"      {RED}✗ Learning failed{RESET}")
                click.echo(f"      Continuing without risk model...")
            summary["steps"]["learn"] = "failed"
        else:
            total = learn_result["total"]
            rate = learn_result["success_rate"]
            if not output_json:
                click.echo(
                    f"      {GREEN}✓ {learn_result['successes']}{RESET} success  "
                    f"{RED}✗ {learn_result['failures']}{RESET} fail  "
                    f"({rate:.0%} success rate)"
                )
            summary["steps"]["learn"] = {
                "successes": learn_result["successes"],
                "failures": learn_result["failures"],
                "success_rate": rate,
            }

    # ── Step 3: Scan and gate ─────────────────────────────────────
    if not output_json:
        click.echo(f"[3/4] Scanning and gating...")

    dryrun = _run_fix_dryrun(target, profile=effective_profile, rules=rules)

    if dryrun.get("status") == "no_findings":
        if not output_json:
            click.echo(f"      {GREEN}No issues found. Codebase is clean.{RESET}")
        summary["steps"]["scan"] = {"total": 0, "approved": 0, "gated": 0}

        if ci:
            if output_json:
                click.echo(json.dumps({"status": "clean", **summary}))
            else:
                click.echo()
                click.echo(f"{CYAN}{'═' * 55}{RESET}")
                click.echo(f"  {GREEN}CLEAN{RESET} — No issues found")
                click.echo(f"{CYAN}{'═' * 55}{RESET}")
            raise SystemExit(0)

        if not output_json:
            click.echo()
            click.echo(f"{CYAN}{'═' * 55}{RESET}")
            click.echo(f"  {GREEN}AUTOPILOT COMPLETE — No issues found{RESET}")
            click.echo(f"{CYAN}{'═' * 55}{RESET}")
        else:
            click.echo(json.dumps({"status": "clean", **summary}))
        return

    if dryrun.get("status") == "error":
        if not output_json:
            click.echo(f"      {RED}✗ Scan failed: {dryrun.get('error', 'unknown')}{RESET}")
        if ci:
            raise SystemExit(1)
        return

    total = dryrun.get("approved", 0) + dryrun.get("gated", 0) + dryrun.get("flagged", 0)
    approved = dryrun.get("approved", 0)
    gated = dryrun.get("gated", 0)
    flagged = dryrun.get("flagged", 0)

    if not output_json:
        click.echo(f"      Found {total} issues")
        click.echo(f"      {GREEN}Approved:{RESET} {approved}  {YELLOW}Flagged:{RESET} {flagged}  {RED}Gated:{RESET} {gated}")

    summary["steps"]["scan"] = {
        "total": total,
        "approved": approved,
        "gated": gated,
        "flagged": flagged,
    }

    if approved == 0:
        if not output_json:
            click.echo(f"      No safe fixes to apply (all gated as high-risk)")
            click.echo()
            click.echo(f"{CYAN}{'═' * 55}{RESET}")
            click.echo(f"  AUTOPILOT COMPLETE")
            click.echo(f"  {RED}All {gated} issues gated as too risky to auto-fix{RESET}")
            click.echo(f"{CYAN}{'═' * 55}{RESET}")
        if ci:
            raise SystemExit(1)
        return

    # ── Step 4: Apply fixes ───────────────────────────────────────
    should_apply = auto_apply or ci

    if not should_apply and not output_json:
        click.echo()
        try:
            response = click.confirm(
                f"  Apply {approved} safe patches?",
                default=True,
            )
            should_apply = response
        except (click.Abort, EOFError):
            should_apply = False

    if not should_apply:
        if not output_json:
            click.echo()
            click.echo(f"  Skipped. Run {CYAN}capseal fix . --apply{RESET} to apply later.")
            click.echo()
            click.echo(f"{CYAN}{'═' * 55}{RESET}")
            click.echo(f"  AUTOPILOT COMPLETE (patches not applied)")
            click.echo(f"{CYAN}{'═' * 55}{RESET}")
        return

    if not output_json:
        click.echo(f"[4/4] Generating and applying patches...")

    fix_result = _run_fix_apply(target, profile=effective_profile, rules=rules)

    patches_valid = fix_result.get("patches_valid", 0)
    patches_applied = fix_result.get("applied", False)
    cap_file = fix_result.get("cap_file", "")

    if not output_json:
        if patches_valid > 0:
            click.echo(f"      {GREEN}✓ {patches_valid} patches generated{RESET}")
        if gated > 0:
            click.echo(f"      {RED}✗ {gated} gated (too risky){RESET}")

    summary["steps"]["fix"] = {
        "patches_generated": patches_valid,
        "applied": patches_applied,
        "cap_file": cap_file,
    }

    # ── Verify ────────────────────────────────────────────────────
    verify_result = _run_verify(target)
    verified = verify_result.get("verified", False)
    cap_path = verify_result.get("cap_path", "")
    chain_hash = verify_result.get("chain_hash", "")

    summary["steps"]["verify"] = {
        "verified": verified,
        "cap_path": cap_path,
        "chain_hash": chain_hash,
    }

    # ── Final summary ─────────────────────────────────────────────
    if output_json:
        summary["status"] = "complete"
        click.echo(json.dumps(summary, indent=2))
        return

    click.echo()
    click.echo(f"{CYAN}{'═' * 55}{RESET}")
    click.echo(f"{CYAN}  AUTOPILOT COMPLETE{RESET}")
    click.echo(f"{CYAN}{'═' * 55}{RESET}")
    click.echo(f"  Fixed {GREEN}{patches_valid}{RESET} issues, blocked {RED}{gated}{RESET} risky changes")
    if verified:
        click.echo(f"  {GREEN}✓ Verified:{RESET} {cap_path}")
    else:
        click.echo(f"  {YELLOW}⚠ Verification:{RESET} {verify_result.get('error', 'could not verify')}")
    if chain_hash:
        click.echo(f"  Receipt:  {chain_hash[:16]}...")
    click.echo(f"{CYAN}{'═' * 55}{RESET}")
    click.echo()

    if ci and (gated > 0 or flagged > 0):
        raise SystemExit(1)


__all__ = ["autopilot_command"]
