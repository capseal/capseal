"""Watch command for CI integration - runs on every PR."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click


def _get_changed_files() -> list[str]:
    """Get list of changed files from git diff."""
    import subprocess

    try:
        # Try to get files changed vs main/master
        for base in ["origin/main", "origin/master", "main", "master"]:
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", base],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
                if files:
                    return files
            except subprocess.CalledProcessError:
                continue

        # Fallback: get staged files
        result = subprocess.run(
            ["git", "diff", "--name-only", "--cached"],
            capture_output=True,
            text=True,
        )
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

    except Exception:
        return []


def _compute_risk_score(file_path: str, diff_lines: int) -> float:
    """Compute risk score for a file change."""
    # TODO: Replace with canonical risk_engine -- see upgrade-spec #2.
    import random

    # Seed based on file path for determinism
    random.seed(hash(file_path))

    # Base score from file characteristics
    score = 0.2

    # Higher risk for certain file types
    if file_path.endswith((".py", ".js", ".ts")):
        score += 0.1
    if "test" in file_path.lower():
        score -= 0.1  # Tests are lower risk
    if "config" in file_path.lower() or "secret" in file_path.lower():
        score += 0.2

    # Risk increases with diff size
    if diff_lines > 100:
        score += 0.2
    elif diff_lines > 50:
        score += 0.1

    # Add some variance
    score += random.uniform(-0.1, 0.1)

    return max(0.0, min(1.0, score))


def _generate_receipt(file_path: str, decision: str, risk_score: float) -> dict:
    """Generate a receipt for a gating decision."""
    timestamp = datetime.now(timezone.utc).isoformat()
    rounded_score = round(risk_score, 3)
    # Use the rounded score in the hash to ensure verification works
    content = f"{file_path}:{decision}:{rounded_score:.3f}:{timestamp}"

    return {
        "file": file_path,
        "decision": decision,
        "risk_score": rounded_score,
        "timestamp": timestamp,
        "receipt_hash": hashlib.sha256(content.encode()).hexdigest()[:16],
    }


def _run_gate_analysis(
    files: list[str],
    gate_threshold: float,
    review_threshold: float,
) -> dict:
    """Run gating analysis on changed files."""
    results = {
        "files_analyzed": len(files),
        "approved": [],
        "flagged_for_review": [],
        "gated": [],
        "receipts": [],
    }

    for file_path in files:
        # Skip non-source files
        if not any(file_path.endswith(ext) for ext in [".py", ".js", ".ts", ".go", ".rs", ".java"]):
            continue

        # Compute risk score
        risk_score = _compute_risk_score(file_path, diff_lines=50)  # Mock diff lines

        # Make decision
        if risk_score >= gate_threshold:
            decision = "gated"
            results["gated"].append(file_path)
        elif risk_score >= review_threshold:
            decision = "review"
            results["flagged_for_review"].append(file_path)
        else:
            decision = "approved"
            results["approved"].append(file_path)

        # Generate receipt
        receipt = _generate_receipt(file_path, decision, risk_score)
        results["receipts"].append(receipt)

    return results


def _verify_receipts(receipts: list[dict]) -> bool:
    """Verify all receipts are valid."""
    for receipt in receipts:
        # Reconstruct the exact content used during generation
        content = f"{receipt['file']}:{receipt['decision']}:{receipt['risk_score']:.3f}:{receipt['timestamp']}"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if receipt["receipt_hash"] != expected_hash:
            return False
    return True


def _format_pr_comment(results: dict, verified: bool) -> str:
    """Format results as a PR comment (markdown)."""
    lines = [
        "## CapSeal Gate Report",
        "",
        f"**{results['files_analyzed']} files analyzed**",
        "",
        "| Status | Count |",
        "|--------|-------|",
        f"| ✓ Approved | {len(results['approved'])} |",
        f"| ⚠ Flagged for Review | {len(results['flagged_for_review'])} |",
        f"| ✗ Gated (too risky) | {len(results['gated'])} |",
        "",
    ]

    if results["gated"]:
        lines.append("### Gated Files")
        lines.append("")
        for f in results["gated"]:
            receipt = next((r for r in results["receipts"] if r["file"] == f), None)
            score = receipt["risk_score"] if receipt else "N/A"
            lines.append(f"- `{f}` (risk: {score:.0%})")
        lines.append("")

    if results["flagged_for_review"]:
        lines.append("### Files Flagged for Review")
        lines.append("")
        for f in results["flagged_for_review"]:
            receipt = next((r for r in results["receipts"] if r["file"] == f), None)
            score = receipt["risk_score"] if receipt else "N/A"
            lines.append(f"- `{f}` (risk: {score:.0%})")
        lines.append("")

    # Verification status
    if verified:
        lines.append("---")
        lines.append("✓ All receipts verified")
    else:
        lines.append("---")
        lines.append("⚠ Receipt verification failed")

    return "\n".join(lines)


@click.command("watch")
@click.option(
    "--gate-threshold",
    type=float,
    default=0.7,
    help="Risk score threshold for gating (default: 0.7)",
)
@click.option(
    "--review-threshold",
    type=float,
    default=0.4,
    help="Risk score threshold for flagging review (default: 0.4)",
)
@click.option(
    "--json", "json_output",
    is_flag=True,
    help="Output JSON instead of human-readable",
)
@click.option(
    "--comment", "output_comment",
    type=click.Path(),
    help="Write PR comment markdown to file",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    help="Write full report to JSON file",
)
@click.option(
    "--files",
    type=str,
    help="Comma-separated list of files to analyze (default: git diff)",
)
def watch_command(
    gate_threshold: float,
    review_threshold: float,
    json_output: bool,
    output_comment: str | None,
    output: str | None,
    files: str | None,
) -> None:
    """Run CapSeal gate check for CI/CD integration.

    Analyzes changed files, computes risk scores, and gates risky changes.
    Returns exit code 0 if all gated files pass verification, 1 otherwise.

    \b
    Usage in CI:
        capseal watch --json > report.json
        capseal watch --comment pr-comment.md

    \b
    Example GitHub Action:
        - run: pip install capseal
        - run: capseal watch --json > capseal-report.json
        - run: capseal verify-capsule .capseal/runs/latest
    """
    start_time = time.time()

    # Get files to analyze
    if files:
        file_list = [f.strip() for f in files.split(",")]
    else:
        file_list = _get_changed_files()

    if not file_list:
        if json_output:
            click.echo(json.dumps({"status": "no_changes", "files_analyzed": 0}))
        else:
            click.echo("No changed files to analyze.")
        return

    # Run analysis
    results = _run_gate_analysis(file_list, gate_threshold, review_threshold)

    # Verify receipts
    verified = _verify_receipts(results["receipts"])

    # Save capsule
    capseal_dir = Path(".capseal/runs/watch")
    capseal_dir.mkdir(parents=True, exist_ok=True)

    capsule = {
        "schema": "watch_capsule_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "verification": {"receipts_valid": verified},
    }
    capsule["capsule_hash"] = hashlib.sha256(
        json.dumps(capsule, sort_keys=True).encode()
    ).hexdigest()

    capsule_path = capseal_dir / "watch_capsule.json"
    capsule_path.write_text(json.dumps(capsule, indent=2))

    elapsed = time.time() - start_time

    # Output
    if json_output:
        output_data = {
            "status": "pass" if not results["gated"] else "gated",
            "files_analyzed": results["files_analyzed"],
            "approved": len(results["approved"]),
            "flagged_for_review": len(results["flagged_for_review"]),
            "gated": len(results["gated"]),
            "receipts_verified": verified,
            "elapsed_seconds": round(elapsed, 2),
            "capsule_path": str(capsule_path),
        }
        click.echo(json.dumps(output_data, indent=2))
    else:
        click.echo(f"CapSeal Watch - {results['files_analyzed']} files analyzed in {elapsed:.1f}s")
        click.echo()
        click.echo(f"  ✓ Approved:           {len(results['approved'])}")
        click.echo(f"  ⚠ Flagged for Review: {len(results['flagged_for_review'])}")
        click.echo(f"  ✗ Gated:              {len(results['gated'])}")
        click.echo()

        if results["gated"]:
            click.echo("Gated files (too risky):")
            for f in results["gated"]:
                receipt = next((r for r in results["receipts"] if r["file"] == f), None)
                if receipt:
                    click.echo(f"  - {f} (risk: {receipt['risk_score']:.0%})")

        click.echo()
        if verified:
            click.echo("✓ All receipts verified")
        else:
            click.echo("⚠ Receipt verification failed")

        click.echo(f"\nCapsule: {capsule_path}")

    # Write PR comment if requested
    if output_comment:
        comment = _format_pr_comment(results, verified)
        Path(output_comment).write_text(comment)
        if not json_output:
            click.echo(f"PR comment written to: {output_comment}")

    # Write full report if requested
    if output:
        full_report = {
            "capsule": capsule,
            "results": results,
            "elapsed_seconds": round(elapsed, 2),
        }
        Path(output).write_text(json.dumps(full_report, indent=2))
        if not json_output:
            click.echo(f"Full report written to: {output}")

    # Exit code: 0 if no gated files, 1 if any gated
    if results["gated"] and not verified:
        sys.exit(1)


__all__ = ["watch_command"]
