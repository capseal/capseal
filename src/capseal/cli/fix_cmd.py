"""capseal fix - Generate verified patches, gated by learned risk model.

This is the main command that does real work:
    scan → gate → plan → patches → verify → .cap

Usage:
    capseal fix .                    # Full pipeline
    capseal fix . --dry-run          # Show plan without generating patches
    capseal fix . --apply            # Apply patches to working tree
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import click
from capseal.risk_engine import (
    THRESHOLD_APPROVE,
    THRESHOLD_DENY,
    evaluate_risk_for_finding,
)


# ANSI colors
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"


def _load_capseal_env(target_path: Path) -> None:
    """Load API keys from .capseal/.env if not already in environment."""
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
                    if key and not os.environ.get(key):
                        os.environ[key] = value
    except Exception:
        pass


@click.command("fix")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--dry-run", is_flag=True, help="Show plan and gate decisions without generating patches")
@click.option("--apply", "apply_patches", is_flag=True, help="Apply verified patches to working tree")
@click.option(
    "--threshold",
    type=float,
    default=THRESHOLD_DENY,
    help=f"Failure probability threshold (default: {THRESHOLD_DENY})",
)
@click.option("--provider", default="openai", help="LLM provider")
@click.option("--model", default="gpt-4o-mini", help="LLM model")
@click.option("--max-workers", default=4, type=int, help="Max parallel agents")
@click.option("--json", "output_json", is_flag=True, help="Output JSON for CI")
@click.option("--profile", type=click.Choice(["security", "quality", "bugs", "all", "custom"]),
              default=None, help="Scan profile (default: from config or 'auto')")
@click.option("--rules", type=click.Path(exists=True), default=None, help="Custom semgrep rules path")
def fix_command(
    path: str,
    dry_run: bool,
    apply_patches: bool,
    threshold: float,
    provider: str,
    model: str,
    max_workers: int,
    output_json: bool,
    profile: str | None = None,
    rules: str | None = None,
) -> None:
    """Generate verified patches, gated by learned risk model.

    Runs the full pipeline: scan → gate → plan → patches → verify → .cap

    The gate step uses the learned model from `capseal learn` to skip
    high-risk patches that are likely to fail. If no model exists,
    proceeds without gating (with a warning).

    \b
    Examples:
        capseal fix .                    # Full pipeline
        capseal fix . --dry-run          # Preview without generating
        capseal fix . --apply            # Apply patches after verifying
        capseal fix . --threshold 0.5    # Custom risk threshold

    \b
    After fixing:
        capseal verify .capseal/runs/latest.cap    # Verify the run
        capseal report .capseal/runs/latest        # View summary
    """
    target_path = Path(path).expanduser().resolve()

    # Load API keys from .capseal/.env if available
    _load_capseal_env(target_path)

    # Check for API key or CLI proxy
    import shutil as _shutil
    has_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    cli_binary = None

    if not has_api_key:
        # Look for CLI binary for subscription mode
        config_path_early = target_path / ".capseal" / "config.json"
        cfg_provider = ""
        if config_path_early.exists():
            try:
                _cfg = json.loads(config_path_early.read_text())
                cfg_provider = _cfg.get("provider", "")
            except (json.JSONDecodeError, OSError):
                pass

        cli_map = {"anthropic": "claude", "openai": "codex", "google": "gemini"}
        preferred = cli_map.get(cfg_provider)
        if preferred and _shutil.which(preferred):
            cli_binary = preferred
        else:
            for binary in ("claude", "codex", "gemini"):
                if _shutil.which(binary):
                    cli_binary = binary
                    break

        if not dry_run and not cli_binary:
            click.echo(f"{RED}Error: No API key or provider CLI found.{RESET}", err=True)
            click.echo("", err=True)
            click.echo("Fix requires either:", err=True)
            click.echo("  • A provider CLI (claude, codex, gemini) — uses your subscription", err=True)
            click.echo("  • An API key (ANTHROPIC_API_KEY, OPENAI_API_KEY)", err=True)
            click.echo("", err=True)
            click.echo("Install your provider's CLI or set an API key, then try again.", err=True)
            raise SystemExit(1)

    # target_path already set above for env loading
    capseal_dir = target_path / ".capseal"
    model_path = capseal_dir / "models" / "beta_posteriors.npz"
    runs_dir = capseal_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Load config for default profile
    config_json = None
    config_file = capseal_dir / "config.json"
    if config_file.exists():
        try:
            config_json = json.loads(config_file.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    # Create run directory
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = runs_dir / f"{timestamp}-fix"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not output_json:
        click.echo()
        click.echo("═" * 65)
        click.echo("  CAPSEAL FIX")
        click.echo("═" * 65)
        click.echo(f"  Target:    {target_path}")
        click.echo(f"  Threshold: {threshold:.0%} failure rate")
        if dry_run:
            click.echo(f"  Mode:      {YELLOW}DRY RUN{RESET} (no patches will be generated)")
        elif apply_patches:
            click.echo(f"  Mode:      {GREEN}APPLY{RESET} (patches will be applied)")
        else:
            click.echo(f"  Mode:      Generate patches (use --apply to apply)")
        click.echo("═" * 65)
        click.echo()

    # ─────────────────────────────────────────────────────────────────
    # Step 1: Scan with Semgrep
    # ─────────────────────────────────────────────────────────────────
    if not output_json:
        click.echo("[1/5] Scanning with Semgrep...")

    try:
        from .scan_profiles import build_semgrep_args
        semgrep_cmd = build_semgrep_args(target_path, profile=profile, custom_rules=rules, config_json=config_json)
        result = subprocess.run(
            semgrep_cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        semgrep_output = json.loads(result.stdout) if result.stdout else {"results": []}
        findings = semgrep_output.get("results", [])
    except subprocess.TimeoutExpired:
        click.echo("Error: Semgrep scan timed out", err=True)
        raise SystemExit(1)
    except FileNotFoundError:
        click.echo("Error: Semgrep not installed. Run: pip install semgrep", err=True)
        raise SystemExit(1)
    except json.JSONDecodeError:
        findings = []

    if not findings:
        if output_json:
            click.echo(json.dumps({"status": "no_findings", "findings": 0}))
        else:
            click.echo("      No findings detected. Nothing to fix.")
        return

    if not output_json:
        click.echo(f"      Found {len(findings)} issues")

    # Save scan results
    (run_dir / "semgrep_findings.json").write_text(json.dumps(findings, indent=2))

    # ─────────────────────────────────────────────────────────────────
    # Step 2: Load model and gate findings
    # ─────────────────────────────────────────────────────────────────
    if not output_json:
        click.echo("[2/5] Gating findings...")

    has_model = model_path.exists()
    if has_model:
        if not output_json:
            click.echo(f"      Loaded model from {model_path.relative_to(target_path)}")
    else:
        if not output_json:
            click.echo(f"      {YELLOW}No learned model found.{RESET}")
            click.echo(f"      Tip: Run 'capseal learn .' first for smarter gating.")
            click.echo(f"      Proceeding without gating...")

    # Gate findings
    approved = []
    gated = []
    flagged = []

    for finding in findings:
        risk = evaluate_risk_for_finding(
            finding,
            workspace=target_path,
            approve_threshold=THRESHOLD_APPROVE,
            deny_threshold=threshold,
        )
        finding["_failure_prob"] = float(risk.p_fail)
        finding["_grid_idx"] = risk.grid_cell
        finding["_label"] = risk.label
        finding["_confidence"] = float(risk.confidence)

        if risk.model_loaded:
            has_model = True

        if risk.decision == "deny":
            gated.append(finding)
        elif risk.decision == "flag":
            flagged.append(finding)
        else:
            approved.append(finding)

    if not output_json:
        click.echo(f"      {GREEN}Approved:{RESET} {len(approved)}  {YELLOW}Flagged:{RESET} {len(flagged)}  {RED}Gated:{RESET} {len(gated)}")

    # Save gate results
    gate_result = {
        "threshold": threshold,
        "has_model": has_model,
        "summary": {
            "total": len(findings),
            "approved": len(approved),
            "flagged": len(flagged),
            "gated": len(gated),
        },
        "approved": [
            {
                "check_id": f.get("check_id"),
                "path": f.get("path"),
                "line": f.get("start", {}).get("line"),
                "failure_prob": f.get("_failure_prob"),
                "label": f.get("_label"),
            }
            for f in approved
        ],
        "flagged": [
            {
                "check_id": f.get("check_id"),
                "path": f.get("path"),
                "line": f.get("start", {}).get("line"),
                "failure_prob": f.get("_failure_prob"),
                "label": f.get("_label"),
            }
            for f in flagged
        ],
        "gated": [
            {
                "check_id": f.get("check_id"),
                "path": f.get("path"),
                "line": f.get("start", {}).get("line"),
                "failure_prob": f.get("_failure_prob"),
                "label": f.get("_label"),
            }
            for f in gated
        ],
    }
    (run_dir / "gate_result.json").write_text(json.dumps(gate_result, indent=2))

    # ─────────────────────────────────────────────────────────────────
    # Step 3: Dry run - show plan and stop
    # ─────────────────────────────────────────────────────────────────
    if dry_run:
        if not output_json:
            click.echo()
            click.echo("[3/5] Planning patches (dry run)...")
            click.echo()

            if gated:
                click.echo(f"  {RED}Gated (will skip):{RESET}")
                for f in gated[:5]:
                    prob = f.get("_failure_prob", 0)
                    click.echo(f"    ✗ {f.get('check_id', 'unknown').split('.')[-1]}")
                    click.echo(f"      {f.get('path')}:{f.get('start', {}).get('line', 0)}")
                    click.echo(f"      Predicted failure: {prob:.0%}")
                    if f.get("_label"):
                        click.echo(f"      Label: {f.get('_label')}")
                if len(gated) > 5:
                    click.echo(f"    ... and {len(gated) - 5} more")
                click.echo()

            if flagged:
                click.echo(f"  {YELLOW}Flagged (review):{RESET}")
                for f in flagged[:5]:
                    prob = f.get("_failure_prob", 0)
                    click.echo(f"    ! {f.get('check_id', 'unknown').split('.')[-1]}")
                    click.echo(f"      {f.get('path')}:{f.get('start', {}).get('line', 0)}")
                    click.echo(f"      Predicted failure: {prob:.0%}")
                    if f.get("_label"):
                        click.echo(f"      Label: {f.get('_label')}")
                if len(flagged) > 5:
                    click.echo(f"    ... and {len(flagged) - 5} more")
                click.echo()

            if approved:
                click.echo(f"  {GREEN}Approved (will fix):{RESET}")
                for f in approved[:5]:
                    prob = f.get("_failure_prob", 0)
                    click.echo(f"    ✓ {f.get('check_id', 'unknown').split('.')[-1]}")
                    click.echo(f"      {f.get('path')}:{f.get('start', {}).get('line', 0)}")
                    click.echo(f"      Predicted failure: {prob:.0%}")
                    if f.get("_label"):
                        click.echo(f"      Label: {f.get('_label')}")
                if len(approved) > 5:
                    click.echo(f"    ... and {len(approved) - 5} more")
                click.echo()

            click.echo("═" * 65)
            click.echo("  DRY RUN COMPLETE")
            click.echo("═" * 65)
            click.echo(f"  Would generate patches for {len(approved)} findings")
            click.echo(f"  Would skip {len(gated)} high-risk findings")
            click.echo()
            click.echo(f"  Run without --dry-run to generate patches.")
            click.echo("═" * 65)
        else:
            click.echo(json.dumps({
                "status": "dry_run",
                "approved": len(approved),
                "gated": len(gated),
                "flagged": len(flagged),
            }))
        return

    # ─────────────────────────────────────────────────────────────────
    # Step 4: Generate patches for approved findings
    # ─────────────────────────────────────────────────────────────────
    if not approved:
        if not output_json:
            click.echo()
            click.echo("[3/5] No approved findings to fix.")
            click.echo("      All findings were gated as high-risk.")
        else:
            click.echo(json.dumps({"status": "all_gated", "gated": len(gated)}))
        return

    if not output_json:
        click.echo()
        click.echo(f"[3/5] Generating patches for {len(approved)} findings...")

    # Generate patches directly using LLM
    patches_dir = run_dir / "patches"
    patches_dir.mkdir(exist_ok=True)

    patches_generated = []
    patches_failed = []

    for i, finding in enumerate(approved):
        file_path = finding.get("path", "")
        check_id = finding.get("check_id", "unknown")
        start_line = finding.get("start", {}).get("line", 1)
        end_line = finding.get("end", {}).get("line", start_line + 5)
        message = finding.get("extra", {}).get("message", "")

        if not output_json:
            short_id = check_id.split(".")[-1]
            click.echo(f"      [{i+1}/{len(approved)}] {short_id} at {Path(file_path).name}:{start_line}")

        # Read the file content
        full_path = target_path / file_path
        if not full_path.exists():
            patches_failed.append({"finding": check_id, "error": "file not found"})
            continue

        try:
            file_content = full_path.read_text()
            lines = file_content.split("\n")

            # Get context around the finding
            context_start = max(0, start_line - 10)
            context_end = min(len(lines), end_line + 10)
            context_lines = lines[context_start:context_end]
            context = "\n".join(f"{context_start + j + 1}: {line}" for j, line in enumerate(context_lines))

            # Generate patch with LLM (or CLI proxy)
            patch_content = _generate_patch(
                file_path=file_path,
                file_content=file_content,
                context=context,
                finding=finding,
                provider=provider,
                model=model,
                cli_binary=cli_binary,
            )

            if patch_content:
                patch_id = f"patch_{i:03d}"
                patch_file = patches_dir / f"{patch_id}.diff"
                patch_file.write_text(patch_content)
                patches_generated.append({
                    "patch_id": patch_id,
                    "file_path": file_path,
                    "check_id": check_id,
                    "line": start_line,
                    "patch_file": str(patch_file.relative_to(run_dir)),
                })
                if not output_json:
                    click.echo(f"            {GREEN}✓{RESET} Generated patch")
            else:
                patches_failed.append({"finding": check_id, "error": "no patch generated"})
                if not output_json:
                    click.echo(f"            {YELLOW}⊘{RESET} No patch needed")

        except Exception as e:
            patches_failed.append({"finding": check_id, "error": str(e)})
            if not output_json:
                click.echo(f"            {RED}✗{RESET} Error: {e}")

    # Save patches manifest
    patches_manifest = {
        "generated": patches_generated,
        "failed": patches_failed,
    }
    (run_dir / "patches_manifest.json").write_text(json.dumps(patches_manifest, indent=2))

    # ─────────────────────────────────────────────────────────────────
    # Step 5: Verify and show results
    # ─────────────────────────────────────────────────────────────────
    patches_valid = len(patches_generated)
    patches_failed_count = len(patches_failed)

    if not output_json:
        click.echo()
        click.echo(f"[4/5] Patches: {GREEN}{patches_valid} generated{RESET}, {patches_failed_count} failed")

    # Show diffs
    if patches_generated and not output_json:
        click.echo()
        click.echo("  Generated patches:")
        for patch in patches_generated:
            patch_file = run_dir / patch["patch_file"]
            if patch_file.exists():
                click.echo(f"    {DIM}─── {patch['file_path']}:{patch['line']} ───{RESET}")
                diff_content = patch_file.read_text()
                # Show first few lines of diff
                diff_lines = diff_content.split("\n")[:15]
                for line in diff_lines:
                    if line.startswith("+") and not line.startswith("+++"):
                        click.echo(f"    {GREEN}{line}{RESET}")
                    elif line.startswith("-") and not line.startswith("---"):
                        click.echo(f"    {RED}{line}{RESET}")
                    else:
                        click.echo(f"    {line}")
                if len(diff_content.split("\n")) > 15:
                    click.echo(f"    {DIM}... ({len(diff_content.split(chr(10))) - 15} more lines){RESET}")
        click.echo()

    # ─────────────────────────────────────────────────────────────────
    # Step 6: Apply patches if requested
    # ─────────────────────────────────────────────────────────────────
    applied = False
    if apply_patches and patches_generated:
        if not output_json:
            click.echo("[5/5] Applying patches...")

        for patch in patches_generated:
            patch_file = run_dir / patch["patch_file"]
            if patch_file.exists():
                result = subprocess.run(
                    ["git", "apply", str(patch_file)],
                    capture_output=True,
                    cwd=target_path,
                )
                if result.returncode == 0:
                    if not output_json:
                        click.echo(f"      {GREEN}✓{RESET} Applied: {patch['file_path']}")
                    applied = True
                else:
                    if not output_json:
                        click.echo(f"      {RED}✗{RESET} Failed: {result.stderr.decode()[:50]}")
    elif not output_json:
        click.echo("[5/5] Patches ready (use --apply to apply)")

    # ─────────────────────────────────────────────────────────────────
    # Seal the run
    # ─────────────────────────────────────────────────────────────────
    # Build run receipt
    run_metadata = {
        "run_type": "fix",
        "timestamp": timestamp,
        "target": str(target_path),
        "threshold": threshold,
        "has_model": has_model,
        "findings_total": len(findings),
        "findings_approved": len(approved),
        "findings_gated": len(gated),
        "patches_generated": len(patches_generated),
        "patches_valid": patches_valid,
        "patches_failed": patches_failed_count,
        "applied": applied,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

    # Create receipt
    receipt_hash = hashlib.sha256(json.dumps(run_metadata, sort_keys=True).encode()).hexdigest()
    run_receipt = {
        "schema": "run_receipt_v1",
        "run_type": "fix",
        "chain_hash": receipt_hash,
        "total_rounds": 1,
        "statements": [
            {"round_id": "fix", "statement_hash": receipt_hash}
        ],
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (run_dir / "run_receipt.json").write_text(json.dumps(run_receipt, indent=2))

    # Create .cap file
    from .cap_format import create_run_cap_file
    cap_path = runs_dir / f"{timestamp}-fix.cap"
    create_run_cap_file(
        run_dir=run_dir,
        output_path=cap_path,
        run_type="fix",
        extras={
            "findings_approved": len(approved),
            "findings_gated": len(gated),
            "patches_valid": patches_valid,
        },
    )

    # Create "latest.cap" symlink
    latest_cap_link = runs_dir / "latest.cap"
    if latest_cap_link.is_symlink() or latest_cap_link.exists():
        latest_cap_link.unlink()
    latest_cap_link.symlink_to(cap_path.name)

    # Update latest symlink
    latest_link = runs_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(run_dir.name)

    # ─────────────────────────────────────────────────────────────────
    # Output summary
    # ─────────────────────────────────────────────────────────────────
    if output_json:
        click.echo(json.dumps({
            "status": "complete",
            "approved": len(approved),
            "gated": len(gated),
            "patches_valid": patches_valid,
            "patches_failed": patches_failed,
            "applied": apply_patches,
            "cap_file": str(cap_path.relative_to(target_path)),
        }))
    else:
        click.echo()
        click.echo("═" * 65)
        click.echo("  FIX COMPLETE")
        click.echo("═" * 65)
        click.echo(f"  Findings:  {len(findings)} total")
        click.echo(f"  Gated:     {len(gated)} (skipped as high-risk)")
        click.echo(f"  Fixed:     {len(approved)} attempted")
        click.echo(f"  Patches:   {patches_valid} valid, {patches_failed_count} failed")
        if applied:
            click.echo(f"  Applied:   {GREEN}Yes{RESET}")
        else:
            click.echo(f"  Applied:   No (use --apply)")
        click.echo()
        click.echo(f"  {GREEN}Sealed:{RESET} {cap_path.relative_to(target_path)}")
        click.echo("═" * 65)

def _generate_patch(
    file_path: str,
    file_content: str,
    context: str,
    finding: dict,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    cli_binary: str | None = None,
) -> str | None:
    """Generate a patch for a finding using LLM or CLI proxy."""
    check_id = finding.get("check_id", "unknown")
    message = finding.get("extra", {}).get("message", "")
    start_line = finding.get("start", {}).get("line", 1)

    prompt = f"""You are a code security expert. Fix the following security issue.

## Issue
- Check ID: {check_id}
- Message: {message}
- Location: {file_path}:{start_line}

## Code Context (line numbers shown)
```
{context}
```

## Task
Generate a unified diff that fixes this security issue. The fix should:
1. Address the security concern directly
2. Maintain the original functionality
3. Be minimal - only change what's necessary

Output ONLY the unified diff, starting with --- and +++. No explanation.

Example format:
--- a/{file_path}
+++ b/{file_path}
@@ -10,5 +10,6 @@
 unchanged line
-old problematic line
+new fixed line
 unchanged line
"""

    try:
        if cli_binary:
            # Subscription mode: use CLI proxy
            result = subprocess.run(
                [cli_binary, "--print", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                return None
            diff = result.stdout.strip()
        else:
            # API key mode: use OpenAI-compatible client
            import openai
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
            )
            diff = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if diff.startswith("```"):
            lines = diff.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            diff = "\n".join(lines)

        # Validate it looks like a diff
        if diff.startswith("---") or diff.startswith("diff "):
            return diff
        elif "---" in diff:
            idx = diff.find("---")
            return diff[idx:]
        else:
            return None

    except Exception:
        return None


__all__ = ["fix_command"]
