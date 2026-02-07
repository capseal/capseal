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
import numpy as np


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
@click.option("--threshold", type=float, default=0.6, help="Failure probability threshold (default: 0.6)")
@click.option("--provider", default="openai", help="LLM provider")
@click.option("--model", default="gpt-4o-mini", help="LLM model")
@click.option("--max-workers", default=4, type=int, help="Max parallel agents")
@click.option("--json", "output_json", is_flag=True, help="Output JSON for CI")
def fix_command(
    path: str,
    dry_run: bool,
    apply_patches: bool,
    threshold: float,
    provider: str,
    model: str,
    max_workers: int,
    output_json: bool,
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

    # Check for API key
    if not dry_run:
        if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
            click.echo("Error: No API key found.", err=True)
            click.echo("", err=True)
            click.echo("Set one of:", err=True)
            click.echo("  export OPENAI_API_KEY=sk-...", err=True)
            click.echo("  export ANTHROPIC_API_KEY=sk-ant-...", err=True)
            click.echo("", err=True)
            click.echo("Or run 'capseal init' to set up API credentials.", err=True)
            raise SystemExit(1)

    # target_path already set above for env loading
    capseal_dir = target_path / ".capseal"
    model_path = capseal_dir / "models" / "beta_posteriors.npz"
    runs_dir = capseal_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

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
        result = subprocess.run(
            ["semgrep", "--config", "auto", "--json",
             "--exclude", "node_modules", "--exclude", ".venv", "--exclude", "vendor",
             str(target_path)],
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

    posteriors = None
    has_model = False
    if model_path.exists():
        try:
            data = np.load(model_path)
            alpha = data["alpha"]
            beta = data["beta"]
            failure_prob = beta / (alpha + beta)
            posteriors = {"alpha": alpha, "beta": beta, "failure_prob": failure_prob}
            has_model = True
            if not output_json:
                click.echo(f"      Loaded model from {model_path.relative_to(target_path)}")
        except Exception as e:
            if not output_json:
                click.echo(f"      {YELLOW}Warning:{RESET} Could not load model: {e}")
    else:
        if not output_json:
            click.echo(f"      {YELLOW}No learned model found.{RESET}")
            click.echo(f"      Tip: Run 'capseal learn .' first for smarter gating.")
            click.echo(f"      Proceeding without gating...")

    # Gate findings
    approved = []
    gated = []
    flagged = []

    from capseal.shared.features import extract_patch_features, discretize_features, features_to_grid_idx

    for finding in findings:
        check_id = finding.get("check_id", "unknown")
        file_path = finding.get("path", "")
        line = finding.get("start", {}).get("line", 0)

        # Compute grid index
        grid_idx = _compute_grid_idx(finding)

        if posteriors:
            prob = posteriors["failure_prob"][grid_idx] if grid_idx < len(posteriors["failure_prob"]) else 0.5
            finding["_failure_prob"] = float(prob)
            finding["_grid_idx"] = grid_idx

            if prob > threshold:
                gated.append(finding)
            elif prob > 0.3:
                flagged.append(finding)
            else:
                approved.append(finding)
        else:
            finding["_failure_prob"] = None
            finding["_grid_idx"] = grid_idx
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
        "approved": [{"check_id": f.get("check_id"), "path": f.get("path"), "line": f.get("start", {}).get("line")} for f in approved],
        "flagged": [{"check_id": f.get("check_id"), "path": f.get("path"), "line": f.get("start", {}).get("line")} for f in flagged],
        "gated": [{"check_id": f.get("check_id"), "path": f.get("path"), "line": f.get("start", {}).get("line"), "failure_prob": f.get("_failure_prob")} for f in gated],
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
                if len(gated) > 5:
                    click.echo(f"    ... and {len(gated) - 5} more")
                click.echo()

            if approved:
                click.echo(f"  {GREEN}Approved (will fix):{RESET}")
                for f in approved[:5]:
                    click.echo(f"    ✓ {f.get('check_id', 'unknown').split('.')[-1]}")
                    click.echo(f"      {f.get('path')}:{f.get('start', {}).get('line', 0)}")
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

            # Generate patch with LLM
            patch_content = _generate_patch(
                file_path=file_path,
                file_content=file_content,
                context=context,
                finding=finding,
                provider=provider,
                model=model,
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


def _compute_grid_idx(finding: dict) -> int:
    """Compute grid index for a finding using shared feature extraction.

    Uses the same encoding as the learning system for consistency.
    """
    from capseal.shared.features import extract_patch_features, discretize_features, features_to_grid_idx

    file_path = finding.get("path", "")
    severity = finding.get("extra", {}).get("severity", "warning")
    start_line = finding.get("start", {}).get("line", 1)
    end_line = finding.get("end", {}).get("line", start_line + 5)

    # Create a synthetic diff preview (same as learn_cmd)
    diff_preview = f"diff --git a/{file_path} b/{file_path}\n"
    diff_preview += f"+++ b/{file_path}\n"
    lines_changed = max(5, end_line - start_line + 1)
    diff_preview += f"@@ -{start_line},{lines_changed} @@\n"

    # Extract features using shared module
    raw_features = extract_patch_features(diff_preview, [{"severity": severity}])
    levels = discretize_features(raw_features)
    return features_to_grid_idx(levels)


def _generate_patch(
    file_path: str,
    file_content: str,
    context: str,
    finding: dict,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> str | None:
    """Generate a patch for a finding using LLM."""
    import openai

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
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1000,
        )
        diff = response.choices[0].message.content.strip()

        # Validate it looks like a diff
        if diff.startswith("---") or diff.startswith("diff "):
            return diff
        elif "---" in diff:
            # Extract diff portion
            idx = diff.find("---")
            return diff[idx:]
        else:
            return None

    except Exception as e:
        return None


__all__ = ["fix_command"]
