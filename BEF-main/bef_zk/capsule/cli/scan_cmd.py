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
import numpy as np

from bef_zk.shared.features import (
    extract_patch_features,
    discretize_features,
    features_to_grid_idx,
)


@click.command("scan")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--gate", is_flag=True, help="Gate findings based on learned model")
@click.option("--json", "output_json", is_flag=True, help="Output JSON instead of human-readable")
@click.option("--threshold", type=float, default=0.6, help="Failure probability threshold for gating (default: 0.6)")
def scan_command(path: str, gate: bool, output_json: bool, threshold: float) -> None:
    """Scan codebase for issues and optionally gate based on learned model.

    Without --gate: Just scans and shows findings.
    With --gate: Loads learned model and filters high-risk findings.

    Examples:
        capseal scan .              # Show all findings
        capseal scan . --gate       # Gate high-risk findings
        capseal scan . --json       # CI-friendly output
    """
    target_path = Path(path).resolve()
    capseal_dir = target_path / ".capseal"
    model_path = capseal_dir / "models" / "beta_posteriors.npz"

    # Step 1: Scan with Semgrep
    if not output_json:
        click.echo("Scanning with Semgrep...")

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

    # Step 2: Load model if gating
    model = None
    if gate:
        if model_path.exists():
            try:
                data = np.load(model_path)
                alpha = data["alpha"]
                beta = data["beta"]
                # Compute failure probabilities
                failure_prob = beta / (alpha + beta)
                model = {"alpha": alpha, "beta": beta, "failure_prob": failure_prob}
            except Exception as e:
                if not output_json:
                    click.echo(f"Warning: Could not load model: {e}", err=True)
        else:
            if not output_json:
                click.echo("No learned model found. Run 'capseal learn .' first to enable gating.")
                click.echo("Showing all findings without gating.\n")

    # Step 3: Classify findings
    approved = []
    gated = []
    flagged = []

    for finding in findings:
        check_id = finding.get("check_id", "unknown")
        file_path = finding.get("path", "")
        line = finding.get("start", {}).get("line", 0)
        message = finding.get("extra", {}).get("message", "")

        # Compute grid index for this finding (simplified feature extraction)
        grid_idx = _compute_grid_idx(finding)

        # Determine risk
        if model and gate:
            prob = model["failure_prob"][grid_idx] if grid_idx < len(model["failure_prob"]) else 0.5
            if prob > threshold:
                risk = "HIGH"
                gated.append({
                    "check_id": check_id,
                    "path": file_path,
                    "line": line,
                    "message": message,
                    "failure_prob": float(prob),
                    "grid_idx": grid_idx,
                })
            elif prob > 0.3:
                risk = "MEDIUM"
                flagged.append({
                    "check_id": check_id,
                    "path": file_path,
                    "line": line,
                    "message": message,
                    "failure_prob": float(prob),
                    "grid_idx": grid_idx,
                })
            else:
                risk = "LOW"
                approved.append({
                    "check_id": check_id,
                    "path": file_path,
                    "line": line,
                    "message": message,
                    "failure_prob": float(prob),
                    "grid_idx": grid_idx,
                })
        else:
            # No model - just list findings
            approved.append({
                "check_id": check_id,
                "path": file_path,
                "line": line,
                "message": message,
            })

    # Step 4: Output
    if output_json:
        click.echo(json.dumps({
            "total_findings": len(findings),
            "approved": approved,
            "gated": gated,
            "flagged": flagged,
            "gate_enabled": gate,
            "model_loaded": model is not None,
        }, indent=2))
    else:
        click.echo(f"\n{'═' * 60}")
        click.echo(f"  CAPSEAL SCAN RESULTS")
        click.echo(f"{'═' * 60}")
        click.echo(f"  Target: {target_path}")
        click.echo(f"  Findings: {len(findings)}")

        if gate and model:
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

            if approved:
                click.echo(f"\n  Approved (low-risk, proceed with patches):")
                for a in approved[:5]:  # Show first 5
                    click.echo(f"    • {a['check_id']}")
                    click.echo(f"      {a['path']}:{a['line']}")
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
    if gate and model:
        import time
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        run_dir = capseal_dir / "runs" / f"{timestamp}-review"
        run_dir.mkdir(parents=True, exist_ok=True)

        gate_result = {
            "timestamp": timestamp,
            "target": str(target_path),
            "threshold": threshold,
            "model_path": str(model_path),
            "summary": {
                "total": len(findings),
                "approved": len(approved),
                "flagged": len(flagged),
                "gated": len(gated),
            },
            "decisions": [
                {"finding": g["check_id"], "path": g["path"], "line": g["line"],
                 "decision": "gate", "failure_prob": g["failure_prob"]}
                for g in gated
            ] + [
                {"finding": f["check_id"], "path": f["path"], "line": f["line"],
                 "decision": "flag", "failure_prob": f["failure_prob"]}
                for f in flagged
            ] + [
                {"finding": a["check_id"], "path": a["path"], "line": a["line"],
                 "decision": "approve", "failure_prob": a.get("failure_prob", 0)}
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

        # Update latest symlink
        latest_link = capseal_dir / "runs" / "latest"
        if latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(run_dir.name)


def _compute_grid_idx(finding: dict) -> int:
    """Compute grid index for a finding using shared feature extraction.

    Uses the same encoding as the learning system for consistency.
    """
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


__all__ = ["scan_command"]
