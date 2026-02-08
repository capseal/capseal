"""capseal ci-report — CI-friendly report from the latest session.

Generates markdown or JSON suitable for GitHub PR comments, Actions
step summaries, or any CI pipeline.

Usage:
    capseal ci-report .                          # Markdown to stdout
    capseal ci-report . --format json            # JSON to stdout
    capseal ci-report . -o capseal-report.md     # Write to file
"""
from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path

import click


@click.command("ci-report")
@click.argument("workspace", default=".", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["markdown", "json", "sarif"]), default="markdown",
              help="Output format (default: markdown)")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Write to file instead of stdout")
def ci_report_command(workspace: str, fmt: str, output: str | None) -> None:
    """Generate a CI-friendly report from the latest CapSeal session.

    Reads the most recent .cap file and produces a summary table suitable
    for GitHub PR comments or Actions step summaries.

    \b
    Examples:
        capseal ci-report .
        capseal ci-report . --format json
        capseal ci-report . -o capseal-report.md
    """
    ws = Path(workspace).resolve()
    runs_dir = ws / ".capseal" / "runs"

    if not runs_dir.exists():
        click.echo("No CapSeal sessions found.", err=True)
        raise SystemExit(1)

    cap_files = sorted(
        [f for f in runs_dir.glob("*.cap") if not f.is_symlink()],
        key=lambda p: p.stat().st_mtime,
    )
    if not cap_files:
        click.echo("No .cap files found.", err=True)
        raise SystemExit(1)

    latest = cap_files[-1]

    # Load actions
    actions = _load_actions(latest)
    manifest = _load_manifest(latest)
    verified = _verify_chain(actions)

    # Count stats
    total = len(actions)
    approved = sum(1 for a in actions if _map_decision(a.get("gate_decision")) == "approve")
    denied = sum(1 for a in actions if _map_decision(a.get("gate_decision")) == "deny")
    flagged = sum(1 for a in actions if _map_decision(a.get("gate_decision")) == "flag")

    # Build per-file summary
    file_stats: dict[str, dict] = {}
    for action in actions:
        metadata = action.get("metadata") or {}
        for f in metadata.get("files_affected", []):
            if f not in file_stats:
                file_stats[f] = {"findings": 0, "p_fail": 0.0, "decision": "auto-fixed"}
            file_stats[f]["findings"] += 1
            p = action.get("gate_score") or 0.0
            file_stats[f]["p_fail"] = max(file_stats[f]["p_fail"], p)
            dec = _map_decision(action.get("gate_decision"))
            if dec == "deny":
                file_stats[f]["decision"] = "blocked"
            elif dec == "flag":
                file_stats[f]["decision"] = "flagged for review"

    if fmt == "sarif":
        from ..sarif import build_sarif_log
        sarif = build_sarif_log(actions, manifest, latest, workspace=ws)
        text = json.dumps(sarif, indent=2)
    elif fmt == "markdown":
        text = _build_markdown(manifest, file_stats, total, approved, denied, flagged, verified, latest)
    else:
        text = json.dumps({
            "session": manifest.get("session_name", ""),
            "total_actions": total,
            "approved": approved,
            "denied": denied,
            "flagged": flagged,
            "verified": verified,
            "files": file_stats,
            "cap_file": str(latest),
        }, indent=2)

    if output:
        Path(output).write_text(text)
        click.echo(f"Report written to {output}")
    else:
        click.echo(text)

    # Write to GitHub Step Summary if available
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path and fmt == "markdown":
        with open(summary_path, "a") as f:
            f.write(text + "\n")


def _build_markdown(
    manifest: dict, file_stats: dict, total: int,
    approved: int, denied: int, flagged: int,
    verified: bool, cap_path: Path,
) -> str:
    """Build markdown report."""
    icon = "\u2713" if verified else "\u2717"
    lines = [
        f"## CapSeal Security Gate {icon}",
        "",
    ]

    if file_stats:
        lines.extend([
            "| File | Findings | Risk | Decision |",
            "|------|----------|------|----------|",
        ])
        for f, stats in sorted(file_stats.items()):
            p = stats["p_fail"]
            risk = f"p_fail={p:.2f}"
            dec = stats["decision"]
            dec_icon = "\u2713" if dec == "auto-fixed" else "\u26a0"
            lines.append(f"| {f} | {stats['findings']} | {risk} | {dec_icon} {dec} |")
        lines.append("")

    lines.append(f"**{total} actions** gated and sealed. {approved} approved, {denied} denied, {flagged} flagged.")
    chain_str = "\u2713 verified" if verified else "\u2717 FAILED"
    lines.append(f"Chain integrity: {chain_str}")

    return "\n".join(lines)


def _map_decision(decision: str | None) -> str:
    """Map internal gate decision to normalized name."""
    return {
        "pass": "approve",
        "skip": "deny",
        "human_review": "flag",
    }.get(decision or "", "approve")


def _verify_chain(actions: list[dict]) -> bool:
    """Quick chain integrity check."""
    try:
        from capseal.agent_protocol import AgentAction

        prev_hash = None
        for raw in actions:
            aa = AgentAction.from_dict(raw)
            receipt_hash = aa.compute_receipt_hash()
            expected_parent = raw.get("parent_receipt_hash")
            if prev_hash is not None and expected_parent != prev_hash:
                return False
            prev_hash = receipt_hash
        return True
    except Exception:
        return False


def _load_manifest(cap_path: Path) -> dict:
    """Load manifest from .cap tarball."""
    try:
        with tarfile.open(cap_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith("manifest.json"):
                    f = tar.extractfile(member)
                    if f:
                        return json.loads(f.read().decode("utf-8"))
    except Exception:
        pass
    return {}


def _load_actions(cap_path: Path) -> list[dict]:
    """Load actions from .cap tarball or run directory."""
    actions: list[dict] = []

    try:
        with tarfile.open(cap_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith("actions.jsonl"):
                    f = tar.extractfile(member)
                    if f:
                        for line in f.read().decode("utf-8").strip().split("\n"):
                            if line.strip():
                                try:
                                    actions.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                        return actions
    except Exception:
        pass

    # Fallback: run directory
    run_dir = cap_path.parent / cap_path.stem
    actions_file = run_dir / "actions.jsonl"
    if actions_file.exists():
        for line in actions_file.read_text().strip().split("\n"):
            if line.strip():
                try:
                    actions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return actions


__all__ = ["ci_report_command"]
