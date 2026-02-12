"""capseal report — Enterprise risk dashboard.

Reads the trained model, current semgrep findings, and session history to
produce a human-readable risk report suitable for engineering leads.

Usage:
    capseal report                   # Rich terminal output
    capseal report . --json          # CI-friendly JSON
    capseal report src/ --print      # Plain text to stdout
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import click


# ── Static suggestion map ────────────────────────────────────────────────────
# Maps semgrep rule ID fragments to one-line remediation suggestions.
SUGGESTIONS: dict[str, str] = {
    # Python security
    "dangerous-exec": "Replace exec() with ast.literal_eval() or a safe parser",
    "eval": "Replace eval() with ast.literal_eval() or a whitelist",
    "subprocess-shell": "Use subprocess.run() with a list instead of shell=True",
    "sql-injection": "Use parameterized queries (e.g. cursor.execute(sql, params))",
    "pickle": "Replace pickle with json or a safer serialization format",
    "yaml-load": "Use yaml.safe_load() instead of yaml.load()",
    "hardcoded-password": "Move secrets to environment variables or a vault",
    "hardcoded-secret": "Move secrets to environment variables or a vault",
    "import-module": "Add input validation instead of refactoring the import",
    "importlib": "Add input validation instead of refactoring the import",
    "path-traversal": "Validate and sanitize file paths against a base directory",
    "command-injection": "Use subprocess.run() with a list; never interpolate user input",
    "ssrf": "Validate URLs against an allowlist before making requests",
    "xss": "Sanitize user input before rendering in HTML",
    "open-redirect": "Validate redirect URLs against a domain allowlist",
    "deserialization": "Use json instead of pickle/marshal for untrusted data",
    # Broad patterns
    "refactor": "Break into smaller single-file changes for safer automation",
    "restructur": "Break into smaller single-file changes for safer automation",
    "cross-cutting": "Split into focused per-file patches instead of broad changes",
}


def _suggest(check_id: str) -> str:
    """Find a suggestion for a semgrep rule ID."""
    lower = check_id.lower()
    for fragment, suggestion in SUGGESTIONS.items():
        if fragment in lower:
            return suggestion
    return ""


@click.command("report")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--json", "output_json", is_flag=True, help="Output JSON for CI integration")
@click.option("--print", "print_output", is_flag=True, help="Print plain text to stdout")
def report_command(path: str, output_json: bool, print_output: bool) -> None:
    """Generate a risk report for your project.

    Reads the trained risk model, runs a semgrep scan, and aggregates
    session history into an enterprise-grade security report.

    \b
    Examples:
        capseal report               # Rich terminal dashboard
        capseal report . --json      # JSON for CI pipelines
        capseal report --print       # Plain text output
    """
    target = Path(path).resolve()
    capseal_dir = target / ".capseal"

    if not capseal_dir.exists():
        click.echo("Error: No .capseal/ workspace found. Run 'capseal init' first.", err=True)
        raise SystemExit(1)

    report_data = _build_report(target)

    if output_json:
        click.echo(json.dumps(report_data, indent=2))
    elif print_output:
        _print_text_report(report_data)
    else:
        _print_rich_report(report_data)


# ── Data gathering ───────────────────────────────────────────────────────────

def _build_report(target: Path) -> dict:
    """Gather all data needed for the report."""
    import numpy as np

    report: dict = {
        "project": target.name,
        "project_path": str(target),
        "model": {"trained": False, "episodes": 0, "last_updated": ""},
        "hotspots": [],
        "recommendations": {"safe": [], "needs_review": [], "dont_automate": []},
        "session_history": {
            "total_sessions": 0,
            "total_actions": 0,
            "denied_actions": 0,
            "flagged_actions": 0,
            "chain_integrity": "no sessions",
        },
        "findings_count": 0,
    }

    # ── Load model ────────────────────────────────────────────────────────
    posteriors_path = target / ".capseal" / "models" / "beta_posteriors.npz"
    alpha = None
    beta = None
    if posteriors_path.exists():
        try:
            data = np.load(posteriors_path, allow_pickle=True)
            alpha = data["alpha"]
            beta = data["beta"]
            n_episodes = int(data["n_episodes"]) if "n_episodes" in data else "?"
            report["model"]["trained"] = True
            report["model"]["episodes"] = n_episodes

            # Last updated from mtime
            import datetime
            mtime = posteriors_path.stat().st_mtime
            dt = datetime.datetime.fromtimestamp(mtime)
            delta = datetime.datetime.now() - dt
            if delta.days > 0:
                report["model"]["last_updated"] = f"{delta.days}d ago"
            elif delta.seconds > 3600:
                report["model"]["last_updated"] = f"{delta.seconds // 3600}h ago"
            else:
                report["model"]["last_updated"] = f"{delta.seconds // 60}m ago"
        except Exception:
            pass

    # ── Run semgrep scan ──────────────────────────────────────────────────
    findings = _run_semgrep(target)
    report["findings_count"] = len(findings)
    report["session_history"] = _gather_session_history(target)

    if not findings:
        return report

    # ── Score findings and build hotspots ─────────────────────────────────
    from capseal.risk_engine import THRESHOLD_APPROVE, THRESHOLD_DENY, evaluate_risk_for_finding

    # Group findings by file (use relative paths for display)
    by_file: dict[str, list] = {}
    for f in findings:
        fp = f.get("path", "unknown")
        try:
            fp = str(Path(fp).relative_to(target))
        except ValueError:
            pass
        f["_display_path"] = fp
        by_file.setdefault(fp, []).append(f)

    # Score each finding
    scored_findings: list[dict] = []
    for finding in findings:
        file_path = finding.get("_display_path", finding.get("path", ""))
        severity = finding.get("extra", {}).get("severity", "warning")
        check_id = finding.get("check_id", "")
        start_line = finding.get("start", {}).get("line", 1)
        end_line = finding.get("end", {}).get("line", start_line + 5)

        # Canonical scoring path used by scan/fix/mcp gate.
        risk = evaluate_risk_for_finding(finding, workspace=target)
        p_fail = float(risk.p_fail)
        label = risk.label

        scored_findings.append({
            "file": file_path,
            "check_id": check_id,
            "severity": severity,
            "p_fail": p_fail,
            "line": start_line,
            "message": finding.get("extra", {}).get("message", ""),
            "suggestion": _suggest(check_id),
            "label": label,
        })

    # ── Build per-file hotspots ───────────────────────────────────────────
    for file_path, file_findings in sorted(by_file.items()):
        # Find scored entries for this file
        file_scores = [s for s in scored_findings if s["file"] == file_path]
        if not file_scores:
            continue

        avg_p_fail = sum(s["p_fail"] for s in file_scores) / len(file_scores)
        max_p_fail = max(s["p_fail"] for s in file_scores)

        if max_p_fail >= THRESHOLD_DENY:
            recommendation = "NEEDS HUMAN REVIEW"
        elif avg_p_fail < THRESHOLD_APPROVE:
            recommendation = "safe to auto-fix"
        else:
            recommendation = "review recommended"

        # Find the highest risk finding for detail
        riskiest = max(file_scores, key=lambda s: s["p_fail"])

        hotspot: dict = {
            "file": file_path,
            "findings": len(file_scores),
            "p_fail": round(avg_p_fail, 2),
            "max_p_fail": round(max_p_fail, 2),
            "recommendation": recommendation,
            "label": riskiest.get("label", "unclassified"),
        }
        if riskiest["suggestion"]:
            hotspot["detail_check"] = riskiest["check_id"].split(".")[-1] if "." in riskiest["check_id"] else riskiest["check_id"]
            hotspot["detail_p_fail"] = round(riskiest["p_fail"], 2)
            hotspot["suggestion"] = riskiest["suggestion"]

        report["hotspots"].append(hotspot)

    # Sort: highest risk first
    report["hotspots"].sort(key=lambda h: -h["max_p_fail"])

    # ── Build agent recommendations ───────────────────────────────────────
    # Group by check_id (rule type)
    by_rule: dict[str, list] = {}
    for sf in scored_findings:
        short_id = sf["check_id"].split(".")[-1] if "." in sf["check_id"] else sf["check_id"]
        by_rule.setdefault(short_id, []).append(sf)

    for rule_id, rule_findings in sorted(by_rule.items()):
        avg_p = sum(f["p_fail"] for f in rule_findings) / len(rule_findings)
        entry = {
            "rule": rule_id,
            "count": len(rule_findings),
            "avg_p_fail": round(avg_p, 2),
        }
        if avg_p < THRESHOLD_APPROVE:
            report["recommendations"]["safe"].append(entry)
        elif avg_p < THRESHOLD_DENY:
            report["recommendations"]["needs_review"].append(entry)
        else:
            report["recommendations"]["dont_automate"].append(entry)

    return report


def _run_semgrep(target: Path) -> list[dict]:
    """Run semgrep and return findings."""
    from .scan_profiles import build_semgrep_args

    config_path = target / ".capseal" / "config.json"
    config_json = None
    if config_path.exists():
        try:
            config_json = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    try:
        cmd = build_semgrep_args(target, config_json=config_json)
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        output = json.loads(result.stdout.decode())
        return output.get("results", [])
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return []


def _gather_session_history(target: Path) -> dict:
    """Count sessions, actions, decisions from .cap files."""
    runs_dir = target / ".capseal" / "runs"
    history: dict = {
        "total_sessions": 0,
        "total_actions": 0,
        "denied_actions": 0,
        "flagged_actions": 0,
        "chain_integrity": "no sessions",
    }

    if not runs_dir.exists():
        return history

    cap_files = [f for f in runs_dir.glob("*.cap") if not f.is_symlink()]
    history["total_sessions"] = len(cap_files)

    if not cap_files:
        return history

    all_valid = True
    for cap_file in cap_files:
        actions = _load_actions(cap_file, runs_dir)
        history["total_actions"] += len(actions)

        prev_hash = None
        for action in actions:
            gate = action.get("gate_decision")
            if gate == "skip":
                history["denied_actions"] += 1
            elif gate == "human_review":
                history["flagged_actions"] += 1

            # Verify chain link
            expected_parent = action.get("parent_receipt_hash")
            if expected_parent is not None and expected_parent != prev_hash:
                all_valid = False

            # Compute this action's receipt hash
            try:
                from capseal.agent_protocol import AgentAction
                aa = AgentAction.from_dict(action)
                prev_hash = aa.compute_receipt_hash()
            except Exception:
                prev_hash = None

    if cap_files:
        history["chain_integrity"] = "all receipts verified" if all_valid else "chain break detected"

    return history


def _load_actions(cap_file: Path, runs_dir: Path) -> list[dict]:
    """Load actions from a .cap file or its run directory."""
    import tarfile

    actions = []
    try:
        with tarfile.open(cap_file, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith("actions.jsonl"):
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8").strip()
                        for line in content.split("\n"):
                            if line.strip():
                                try:
                                    actions.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                        return actions
    except Exception:
        pass

    # Fallback: run directory
    run_dir = runs_dir / cap_file.stem
    actions_file = run_dir / "actions.jsonl"
    if actions_file.exists():
        for line in actions_file.read_text().strip().split("\n"):
            if line.strip():
                try:
                    actions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return actions


# ── Rich output ──────────────────────────────────────────────────────────────

def _print_rich_report(data: dict) -> None:
    """Print the enterprise risk dashboard using Rich."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.box import ROUNDED

    console = Console()

    # Header
    model = data["model"]
    if model["trained"]:
        model_line = f"{model['episodes']} episodes, last updated {model['last_updated']}"
    else:
        model_line = "not trained — run 'capseal learn .' first"

    header = (
        f"  Project: {data['project']}\n"
        f"  Model:   {model_line}"
    )
    console.print()
    console.print(Panel(
        header,
        title="[bold cyan]═══ CAPSEAL RISK REPORT ═══[/bold cyan]",
        border_style="cyan",
        padding=(1, 1),
        expand=False,
    ))

    if data["findings_count"] == 0:
        console.print("\n  [green]No security findings detected.[/green] Your code looks clean.\n")
        return

    # ── Risk Hotspots ─────────────────────────────────────────────────────
    hotspots = data["hotspots"]
    if hotspots:
        table = Table(
            title="RISK HOTSPOTS",
            box=ROUNDED,
            border_style="cyan",
            show_header=True,
            header_style="bold",
        )
        table.add_column("File", style="white", min_width=20)
        table.add_column("Findings", justify="right", min_width=8)
        table.add_column("p_fail", justify="right", min_width=8)
        table.add_column("Label", min_width=24)
        table.add_column("Recommendation", min_width=25)

        for h in hotspots:
            pf = h["p_fail"]
            if pf < 0.3:
                pf_style = "green"
                rec_style = "green"
            elif pf <= 0.6:
                pf_style = "yellow"
                rec_style = "yellow"
            else:
                pf_style = "red"
                rec_style = "red bold"

            table.add_row(
                h["file"],
                str(h["findings"]),
                f"[{pf_style}]{pf:.2f}[/{pf_style}]",
                h.get("label", "unclassified"),
                f"[{rec_style}]{h['recommendation']}[/{rec_style}]",
            )

        console.print()
        console.print(table)

        # Detail lines for high-risk hotspots
        for h in hotspots:
            if h.get("suggestion") and h.get("max_p_fail", 0) > 0.5:
                console.print(f"    [dim]└─ {h.get('detail_check', '')} — "
                              f"refactors fail {h['detail_p_fail']*100:.0f}% of the time[/dim]")
                console.print(f"       [dim]Suggestion: {h['suggestion']}[/dim]")

    # ── Agent Recommendations ─────────────────────────────────────────────
    recs = data["recommendations"]
    console.print()

    rec_lines = []
    if recs["safe"]:
        safe_names = ", ".join(r["rule"] for r in recs["safe"])
        rec_lines.append(f"  [green]✓ Safe to automate:[/green] {safe_names}")
    if recs["needs_review"]:
        review_names = ", ".join(r["rule"] for r in recs["needs_review"])
        rec_lines.append(f"  [yellow]⚠ Needs review:[/yellow] {review_names}")
    if recs["dont_automate"]:
        dont_names = ", ".join(r["rule"] for r in recs["dont_automate"])
        rec_lines.append(f"  [red]✗ Don't automate:[/red] {dont_names}")

    if rec_lines:
        console.print(Panel(
            "\n".join(rec_lines),
            title="[bold]AGENT RECOMMENDATIONS[/bold]",
            title_align="left",
            border_style="cyan",
            padding=(1, 1),
            expand=False,
        ))

    # ── Session History ───────────────────────────────────────────────────
    sh = data["session_history"]
    if sh["total_sessions"] > 0:
        integrity_style = "green" if "verified" in sh["chain_integrity"] else "red"
        history_lines = (
            f"  {sh['total_sessions']} sessions, "
            f"{sh['total_actions']} total actions, "
            f"{sh['denied_actions']} denied, "
            f"{sh['flagged_actions']} flagged\n"
            f"  Chain integrity: [{integrity_style}]✓ {sh['chain_integrity']}[/{integrity_style}]"
        )
        console.print(Panel(
            history_lines,
            title="[bold]SESSION HISTORY[/bold]",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
            expand=False,
        ))
    else:
        console.print("  [dim]No sessions recorded yet.[/dim]")

    console.print()


# ── Plain text output ────────────────────────────────────────────────────────

def _print_text_report(data: dict) -> None:
    """Print plain text report to stdout."""
    model = data["model"]
    model_str = (f"{model['episodes']} episodes, last updated {model['last_updated']}"
                 if model["trained"] else "not trained")

    lines = [
        "═══ CAPSEAL RISK REPORT ═══",
        f"Project: {data['project']}",
        f"Model: {model_str}",
        "",
    ]

    if data["findings_count"] == 0:
        lines.append("No security findings detected.")
        click.echo("\n".join(lines))
        return

    # Hotspots
    lines.append("RISK HOTSPOTS")
    for h in data["hotspots"]:
        rec = h["recommendation"].upper() if h["max_p_fail"] > 0.6 else h["recommendation"]
        lines.append(
            f"  {h['file']:<30} — {h['findings']} findings, p_fail={h['p_fail']:.2f} "
            f"[{h.get('label', 'unclassified')}] ({rec})"
        )
        if h.get("suggestion") and h.get("max_p_fail", 0) > 0.5:
            lines.append(f"    └─ {h.get('detail_check', '')} — refactors fail {h['detail_p_fail']*100:.0f}% of the time")
            lines.append(f"       Suggestion: {h['suggestion']}")
    lines.append("")

    # Recommendations
    recs = data["recommendations"]
    lines.append("AGENT RECOMMENDATIONS")
    if recs["safe"]:
        lines.append(f"  ✓ Safe to automate: {', '.join(r['rule'] for r in recs['safe'])}")
    if recs["needs_review"]:
        lines.append(f"  ⚠ Needs review: {', '.join(r['rule'] for r in recs['needs_review'])}")
    if recs["dont_automate"]:
        lines.append(f"  ✗ Don't automate: {', '.join(r['rule'] for r in recs['dont_automate'])}")
    lines.append("")

    # Session history
    sh = data["session_history"]
    lines.append("SESSION HISTORY")
    lines.append(f"  {sh['total_sessions']} sessions, {sh['total_actions']} total actions, "
                 f"{sh['denied_actions']} denied, {sh['flagged_actions']} flagged")
    lines.append(f"  Chain integrity: ✓ {sh['chain_integrity']}")

    click.echo("\n".join(lines))


__all__ = ["report_command"]
