"""Scan command - the simple path to gated code review.

Usage:
    capseal scan .              # Scan current directory
    capseal scan . --gate       # Scan and gate based on learned model
    capseal scan . --json       # Output JSON for CI
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import click
from capseal.risk_engine import (
    THRESHOLD_APPROVE,
    evaluate_risk,
    evaluate_risk_for_finding,
)


@click.command("scan")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--gate", is_flag=True, help="Gate findings based on learned model")
@click.option("--json", "output_json", is_flag=True, help="Output JSON instead of human-readable")
@click.option("--threshold", type=float, default=0.6, help="Failure probability threshold for gating (default: 0.6)")
@click.option("--diff", "diff_path", type=click.Path(exists=True), default=None, help="Evaluate a unified diff file directly")
@click.option("--profile", type=click.Choice(["security", "quality", "bugs", "all", "custom"]),
              default=None, help="Scan profile (default: from config or 'auto')")
@click.option("--rules", type=click.Path(exists=True), default=None, help="Custom semgrep rules path (use with --profile custom)")
def scan_command(
    path: str,
    gate: bool,
    output_json: bool,
    threshold: float,
    diff_path: str | None,
    profile: str | None,
    rules: str | None,
) -> None:
    """Scan codebase for issues using Semgrep.

    \b
    Examples:
        capseal scan .                         # Find all issues
        capseal scan . --profile security      # Security-focused scan
        capseal scan . --profile quality        # Code quality scan
        capseal scan . --profile all            # Everything
        capseal scan . --json                  # CI-friendly JSON output
    """
    from .scan_profiles import build_semgrep_args, PROFILE_DISPLAY

    target_path = Path(path).resolve()
    capseal_dir = target_path / ".capseal"

    # Direct diff scoring mode (used by integrations and consistency checks)
    if diff_path:
        diff_text = Path(diff_path).read_text()
        risk = evaluate_risk(
            diff_text,
            workspace=target_path,
            approve_threshold=THRESHOLD_APPROVE,
            deny_threshold=threshold,
        )
        payload = {
            "decision": risk.decision,
            "p_fail": round(risk.p_fail, 6),
            "confidence": round(risk.confidence, 6),
            "uncertainty": round(risk.uncertainty, 6),
            "grid_cell": risk.grid_cell,
            "label": risk.label,
            "features": risk.features,
            "model_loaded": risk.model_loaded,
            "thresholds": {
                "approve_below": THRESHOLD_APPROVE,
                "deny_at_or_above": threshold,
            },
        }
        if output_json:
            click.echo(json.dumps(payload, indent=2))
        else:
            click.echo("CAPSEAL DIFF RISK")
            click.echo(f"  Decision:    {risk.decision}")
            click.echo(f"  p_fail:      {risk.p_fail:.2f}")
            click.echo(f"  Label:       {risk.label}")
            click.echo(f"  Grid cell:   {risk.grid_cell}")
            click.echo(f"  Confidence:  {risk.confidence:.2f}")
            click.echo(f"  Model:       {'trained' if risk.model_loaded else 'untrained'}")
        return

    # Load config for default profile
    config_json = None
    config_path = capseal_dir / "config.json"
    if config_path.exists():
        try:
            config_json = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    effective_profile = profile or (config_json or {}).get("scan_profile")

    # Step 1: Scan with Semgrep
    if not output_json:
        profile_label = PROFILE_DISPLAY.get(effective_profile, effective_profile or "auto")
        click.echo(f"Scanning with Semgrep ({profile_label})...")

    try:
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
        sys.exit(1)
    except FileNotFoundError:
        click.echo("Error: Semgrep not installed. Run: pip install semgrep", err=True)
        sys.exit(1)
    except json.JSONDecodeError:
        findings = []

    if not findings:
        if output_json:
            click.echo(json.dumps({"findings": [], "gated": [], "approved": []}))
        else:
            click.echo("No findings detected.")
        return

    # Step 2: Classify findings
    approved = []
    gated = []
    flagged = []
    model_loaded = False

    for finding in findings:
        check_id = finding.get("check_id", "unknown")
        file_path = finding.get("path", "")
        line = finding.get("start", {}).get("line", 0)
        message = finding.get("extra", {}).get("message", "")

        # Determine risk
        if gate:
            risk = evaluate_risk_for_finding(
                finding,
                workspace=target_path,
                approve_threshold=THRESHOLD_APPROVE,
                deny_threshold=threshold,
            )
            model_loaded = model_loaded or risk.model_loaded
            finding_entry = {
                "check_id": check_id,
                "path": file_path,
                "line": line,
                "message": message,
                "failure_prob": float(risk.p_fail),
                "grid_idx": risk.grid_cell,
                "label": risk.label,
                "confidence": float(risk.confidence),
            }
            if risk.decision == "deny":
                gated.append({
                    **finding_entry,
                })
            elif risk.decision == "flag":
                flagged.append({
                    **finding_entry,
                })
            else:
                approved.append({
                    **finding_entry,
                })
        else:
            # No model - just list findings
            approved.append({
                "check_id": check_id,
                "path": file_path,
                "line": line,
                "message": message,
            })

    if gate and not model_loaded and not output_json:
        click.echo("No learned model found. Run 'capseal learn .' first for personalized gating.")
        click.echo("Proceeding with allow-by-default decisions.\n")

    # Step 4: Output
    if output_json:
        click.echo(json.dumps({
            "total_findings": len(findings),
            "approved": approved,
            "gated": gated,
            "flagged": flagged,
            "gate_enabled": gate,
            "model_loaded": model_loaded,
        }, indent=2))
    else:
        click.echo(f"\n{'═' * 60}")
        click.echo(f"  CAPSEAL SCAN RESULTS")
        click.echo(f"{'═' * 60}")
        click.echo(f"  Target: {target_path}")
        click.echo(f"  Findings: {len(findings)}")

        if gate:
            click.echo(f"\n  Gate Results (threshold: {threshold:.0%} failure rate):")
            click.echo(f"    ✓ Approved:  {len(approved)}")
            click.echo(f"    ⚠ Flagged:   {len(flagged)}")
            click.echo(f"    ✗ Gated:     {len(gated)}")

            if gated:
                click.echo(f"\n  Gated (high-risk, skip these patches):")
                for g in gated:
                    click.echo(f"    • {g['check_id']}")
                    click.echo(f"      {g['path']}:{g['line']}")
                    click.echo(f"      Predicted failure: {g['failure_prob']:.0%}")
                    if g.get("label"):
                        click.echo(f"      Label: {g['label']}")

            if flagged:
                click.echo(f"\n  Flagged (review suggested):")
                for f in flagged[:5]:
                    click.echo(f"    • {f['check_id']}")
                    click.echo(f"      {f['path']}:{f['line']}")
                    click.echo(f"      Predicted failure: {f['failure_prob']:.0%}")
                    if f.get("label"):
                        click.echo(f"      Label: {f['label']}")
                if len(flagged) > 5:
                    click.echo(f"    ... and {len(flagged) - 5} more")

            if approved:
                click.echo(f"\n  Approved (low-risk, proceed with patches):")
                for a in approved[:5]:  # Show first 5
                    click.echo(f"    • {a['check_id']}")
                    click.echo(f"      {a['path']}:{a['line']}")
                    click.echo(f"      Predicted failure: {a['failure_prob']:.0%}")
                    if a.get("label"):
                        click.echo(f"      Label: {a['label']}")
                if len(approved) > 5:
                    click.echo(f"    ... and {len(approved) - 5} more")
        else:
            click.echo(f"\n  Findings:")
            for f in findings[:10]:
                check_id = f.get("check_id", "unknown")
                file_path = f.get("path", "")
                line = f.get("start", {}).get("line", 0)
                click.echo(f"    • {check_id}")
                click.echo(f"      {file_path}:{line}")
            if len(findings) > 10:
                click.echo(f"    ... and {len(findings) - 10} more")

            if not gate:
                click.echo(f"\n  Tip: Run with --gate to filter by learned risk model")

        click.echo(f"{'═' * 60}\n")

    # Step 5: Persist gate results if gating was used
    if gate:
        import hashlib
        import time
        from datetime import datetime, timezone

        timestamp = time.strftime("%Y%m%dT%H%M%S")
        run_dir = capseal_dir / "runs" / f"{timestamp}-review"
        run_dir.mkdir(parents=True, exist_ok=True)

        gate_result = {
            "timestamp": timestamp,
            "target": str(target_path),
            "threshold": threshold,
            "model_path": str(capseal_dir / "models" / "beta_posteriors.npz"),
            "model_loaded": model_loaded,
            "summary": {
                "total": len(findings),
                "approved": len(approved),
                "flagged": len(flagged),
                "gated": len(gated),
            },
            "decisions": [
                {"finding": g["check_id"], "path": g["path"], "line": g["line"],
                 "decision": "gate", "failure_prob": g["failure_prob"], "label": g.get("label")}
                for g in gated
            ] + [
                {"finding": f["check_id"], "path": f["path"], "line": f["line"],
                 "decision": "flag", "failure_prob": f["failure_prob"], "label": f.get("label")}
                for f in flagged
            ] + [
                {"finding": a["check_id"], "path": a["path"], "line": a["line"],
                 "decision": "approve", "failure_prob": a.get("failure_prob", 0), "label": a.get("label")}
                for a in approved
            ],
        }

        (run_dir / "gate_result.json").write_text(json.dumps(gate_result, indent=2))

        # Write run metadata
        run_metadata = {
            "run_type": "review",
            "mode": "gate",
            "timestamp": timestamp,
            "target": str(target_path),
            "threshold": threshold,
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2))

        # Build run receipt (hash chain for review run)
        gate_result_hash = hashlib.sha256(
            json.dumps(gate_result, sort_keys=True).encode()
        ).hexdigest()
        run_receipt = {
            "schema": "run_receipt_v1",
            "run_type": "review",
            "chain_hash": gate_result_hash,
            "total_rounds": 1,
            "statements": [
                {
                    "round_id": "gate",
                    "statement_hash": gate_result_hash,
                }
            ],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (run_dir / "run_receipt.json").write_text(json.dumps(run_receipt, indent=2))

        # Package into .cap file
        from .cap_format import create_run_cap_file

        cap_path = run_dir.parent / f"{run_dir.name}.cap"
        create_run_cap_file(
            run_dir=run_dir,
            output_path=cap_path,
            run_type="review",
        )

        # Create "latest.cap" symlink
        latest_cap_link = capseal_dir / "runs" / "latest.cap"
        if latest_cap_link.is_symlink() or latest_cap_link.exists():
            latest_cap_link.unlink()
        latest_cap_link.symlink_to(cap_path.name)

        # Update latest symlink
        latest_link = capseal_dir / "runs" / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)

        if not output_json:
            click.echo(f"  Sealed: {cap_path.relative_to(target_path)}")


__all__ = ["scan_command"]
