"""capseal report - Generate human-readable summary of a CapSeal run.

Creates a markdown report summarizing:
- What was reviewed
- Gate decisions (pass/skip/review)
- Why skips happened
- Verification status
- Cost summary
"""
from __future__ import annotations

import json
from pathlib import Path

import click


@click.command("report")
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output path (default: run_dir/report.md)")
@click.option("--format", "fmt", type=click.Choice(["markdown", "text", "json"]), default="markdown")
@click.option("--print", "print_output", is_flag=True, help="Print to stdout instead of file")
def report_command(run_dir: str, output: str | None, fmt: str, print_output: bool) -> None:
    """Generate a human-readable summary of a CapSeal run.

    Creates a report with:
    - Files and findings reviewed
    - Gate decisions (patches passed, skipped, flagged)
    - Risk explanations for skipped patches
    - Verification status
    - Cost summary

    \b
    Examples:
        capseal report .capseal/runs/latest
        capseal report .capseal/runs/latest --format json
        capseal report .capseal/runs/latest --print
    """
    run_path = Path(run_dir)

    # Gather data from run directory
    data = _gather_run_data(run_path)

    # Generate report
    if fmt == "markdown":
        content = _generate_markdown_report(data, run_path)
    elif fmt == "json":
        content = json.dumps(data, indent=2)
    else:
        content = _generate_text_report(data, run_path)

    # Output
    if print_output:
        click.echo(content)
    else:
        output_path = Path(output) if output else run_path / "report.md"
        output_path.write_text(content)
        click.echo(f"Report saved to: {output_path}")


def _gather_run_data(run_path: Path) -> dict:
    """Gather all data from a run directory."""
    data = {
        "run_dir": str(run_path),
        "run_type": "unknown",
        "metadata": {},
        "files_reviewed": [],
        "findings_count": 0,
        "gate_decisions": {"pass": 0, "skip": 0, "human_review": 0},
        "skipped_reasons": [],
        "actions": [],
        "verification": {"valid": None, "capsule_hash": None},
        "cost": {"total": 0.0, "tokens": 0},
        "verdict": "",
    }

    # Load run metadata
    metadata_path = run_path / "run_metadata.json"
    if metadata_path.exists():
        data["metadata"] = json.loads(metadata_path.read_text())
        data["run_type"] = data["metadata"].get("run_type") or data["metadata"].get("mode", "unknown")

    # Load risk log (from agent runs)
    risk_log_path = run_path / "risk_log.json"
    if risk_log_path.exists():
        risk_log = json.loads(risk_log_path.read_text())
        data["run_type"] = "agent"

        for entry in risk_log:
            decision = entry.get("decision", "pass")
            data["gate_decisions"][decision] = data["gate_decisions"].get(decision, 0) + 1

            if decision in ("skip", "human_review"):
                data["skipped_reasons"].append({
                    "action": entry.get("description", "unknown"),
                    "decision": decision,
                    "risk_score": entry.get("risk_score"),
                    "suggestion": entry.get("suggestion", ""),
                })

    # Load actions (from agent runs)
    actions_path = run_path / "actions.jsonl"
    if actions_path.exists():
        content = actions_path.read_text().strip()
        for line in content.split("\n"):
            if line.strip():
                try:
                    action = json.loads(line)
                    data["actions"].append({
                        "type": action.get("action_type"),
                        "success": action.get("success"),
                        "gate_decision": action.get("gate_decision"),
                    })
                except json.JSONDecodeError:
                    pass

    # Load episodes (from learn runs)
    episodes_path = run_path / "episodes.jsonl"
    if episodes_path.exists():
        data["run_type"] = "learn"
        data["learning"] = {
            "episodes": 0,
            "successes": 0,
            "failures": 0,
            "by_finding": {},
        }
        content = episodes_path.read_text().strip()
        for line in content.split("\n"):
            if line.strip():
                try:
                    ep = json.loads(line)
                    data["cost"]["total"] += ep.get("cost", 0)
                    data["cost"]["tokens"] += ep.get("tokens_used", 0)
                    if ep.get("file_path"):
                        data["files_reviewed"].append(ep["file_path"])

                    # Track learning stats
                    data["learning"]["episodes"] += 1
                    finding_id = ep.get("finding_id", "unknown")
                    success = ep.get("success", False)

                    if success:
                        data["learning"]["successes"] += 1
                    else:
                        data["learning"]["failures"] += 1

                    if finding_id not in data["learning"]["by_finding"]:
                        data["learning"]["by_finding"][finding_id] = {"success": 0, "fail": 0}
                    if success:
                        data["learning"]["by_finding"][finding_id]["success"] += 1
                    else:
                        data["learning"]["by_finding"][finding_id]["fail"] += 1
                except json.JSONDecodeError:
                    pass

    # Load budget
    budget_path = run_path / "budget.json"
    if budget_path.exists():
        budget = json.loads(budget_path.read_text())
        data["cost"]["total"] = budget.get("total_cost", 0)
        data["cost"]["tokens"] = budget.get("total_input_tokens", 0) + budget.get("total_output_tokens", 0)

    # Check for capsules
    for capsule_name in ["agent_capsule.json", "eval_capsule.json", "workflow_capsule.json"]:
        capsule_path = run_path / capsule_name
        if capsule_path.exists():
            capsule = json.loads(capsule_path.read_text())
            data["verification"]["capsule_hash"] = capsule.get("capsule_hash", "")[:32]
            data["verification"]["valid"] = capsule.get("verification", {}).get("constraints_valid")
            break

    # Check for run_receipt.json (learn runs)
    run_receipt_path = run_path / "run_receipt.json"
    if run_receipt_path.exists():
        receipt = json.loads(run_receipt_path.read_text())
        data["verification"]["receipt_file"] = "run_receipt.json"
        data["verification"]["chain_hash"] = receipt.get("chain_hash", "")[:32]
        data["verification"]["total_rounds"] = receipt.get("total_rounds", 0)

    # Count round receipts
    rounds_dir = run_path / "rounds"
    if rounds_dir.exists():
        round_receipts = list(rounds_dir.glob("*/round_receipt.json"))
        data["verification"]["round_receipts"] = len(round_receipts)

    # Load review findings
    findings_path = run_path / "reviews" / "semgrep_report.json"
    if findings_path.exists():
        findings = json.loads(findings_path.read_text())
        data["findings_count"] = len(findings.get("results", []))
        for f in findings.get("results", []):
            if f.get("path"):
                data["files_reviewed"].append(f["path"])

    # Deduplicate files
    data["files_reviewed"] = list(set(data["files_reviewed"]))

    # Load gate_result.json if exists
    gate_result_path = run_path / "gate" / "gate_result.json"
    if not gate_result_path.exists():
        gate_result_path = run_path / "gate_result.json"
    if gate_result_path.exists():
        gate_result = json.loads(gate_result_path.read_text())
        data["gate_result"] = gate_result
        summary = gate_result.get("summary", {})
        data["gate_decisions"]["pass"] = summary.get("approved", 0)
        data["gate_decisions"]["skip"] = summary.get("gated", 0)
        data["gate_decisions"]["human_review"] = summary.get("flagged", 0)

    # Generate verdict
    total_decisions = sum(data["gate_decisions"].values())

    # For learn runs, use learning stats for verdict
    if data["run_type"] == "learn" and "learning" in data:
        learning = data["learning"]
        total = learning["episodes"]
        successes = learning["successes"]
        failures = learning["failures"]
        if total > 0:
            data["verdict"] = f"Learned from {total} episodes: {successes} succeeded ({successes*100//total}%), {failures} failed"
        else:
            data["verdict"] = "No episodes run"
    elif total_decisions == 0:
        data["verdict"] = "No patches evaluated"
    elif data["gate_decisions"]["skip"] == 0 and data["gate_decisions"]["human_review"] == 0:
        data["verdict"] = f"All {data['gate_decisions']['pass']} patches passed verification"
    else:
        skipped = data["gate_decisions"]["skip"] + data["gate_decisions"]["human_review"]
        total_findings = skipped + data["gate_decisions"]["pass"]
        data["verdict"] = f"Gated {skipped} of {total_findings} findings, {data['gate_decisions']['pass']} approved"

    return data


def _generate_markdown_report(data: dict, run_path: Path) -> str:
    """Generate markdown report."""
    lines = [
        "# CapSeal Report",
        "",
        f"**Run:** `{run_path.name}`",
        f"**Type:** {data['run_type']}",
        f"**Verdict:** {data['verdict']}",
        "",
    ]

    # Summary - different for learn vs other runs
    if data["run_type"] == "learn" and "learning" in data:
        learning = data["learning"]
        total = learning["episodes"]
        success_rate = learning["successes"] / total * 100 if total > 0 else 0

        lines.extend([
            "## Learning Summary",
            "",
            f"- **Episodes run:** {total}",
            f"- **Successes:** {learning['successes']} ({success_rate:.0f}%)",
            f"- **Failures:** {learning['failures']} ({100 - success_rate:.0f}%)",
            f"- **Total cost:** ${data['cost']['total']:.2f}",
            "",
        ])

        # Results by finding type
        if learning["by_finding"]:
            lines.extend([
                "## Results by Finding Type",
                "",
                "| Finding | Success | Fail | Rate | Risk |",
                "|---------|---------|------|------|------|",
            ])
            for finding_id, stats in sorted(learning["by_finding"].items()):
                s, f = stats["success"], stats["fail"]
                total_f = s + f
                rate = s / total_f * 100 if total_f > 0 else 0
                risk = "LOW" if rate >= 70 else "HIGH" if rate < 40 else "MEDIUM"
                # Shorten finding ID
                short_id = finding_id.split(".")[-1] if "." in finding_id else finding_id
                lines.append(f"| {short_id} | {s} | {f} | {rate:.0f}% | {risk} |")
            lines.append("")

            # What was learned
            lines.extend([
                "## What Was Learned",
                "",
            ])
            for finding_id, stats in sorted(learning["by_finding"].items()):
                s, f = stats["success"], stats["fail"]
                total_f = s + f
                rate = s / total_f * 100 if total_f > 0 else 0
                short_id = finding_id.split(".")[-1] if "." in finding_id else finding_id
                if rate >= 70:
                    lines.append(f"- ✓ **{short_id}** patches succeed {rate:.0f}% → approve these")
                elif rate < 40:
                    lines.append(f"- ✗ **{short_id}** patches fail {100-rate:.0f}% → gate these")
                else:
                    lines.append(f"- ⚠ **{short_id}** uncertain ({rate:.0f}% success) → flag for review")
            lines.append("")

    else:
        lines.extend([
            "## Summary",
            "",
            f"- **Files reviewed:** {len(data['files_reviewed'])}",
            f"- **Findings:** {data['findings_count']}",
            f"- **Patches passed:** {data['gate_decisions']['pass']}",
            f"- **Patches skipped:** {data['gate_decisions']['skip']}",
            f"- **Flagged for review:** {data['gate_decisions']['human_review']}",
            "",
        ])

    # Verification
    lines.extend([
        "## Verification",
        "",
    ])
    if data["verification"].get("capsule_hash"):
        valid_str = "✅ Valid" if data["verification"]["valid"] else "❌ Invalid"
        lines.extend([
            f"- **Capsule hash:** `{data['verification']['capsule_hash']}...`",
            f"- **Proof status:** {valid_str}",
            "",
        ])
    elif data["verification"].get("chain_hash"):
        lines.extend([
            f"- **Receipt:** {data['verification'].get('receipt_file', 'run_receipt.json')}",
            f"- **Chain hash:** `{data['verification']['chain_hash']}...`",
            f"- **Rounds chained:** {data['verification'].get('total_rounds', 0)}",
        ])
        if data["verification"].get("round_receipts"):
            lines.append(f"- **Round receipts:** {data['verification']['round_receipts']}")
        lines.append("")
    else:
        lines.append("- No cryptographic proof generated")
        lines.append("")

    # Skipped patches
    if data["skipped_reasons"]:
        lines.extend([
            "## Skipped Patches",
            "",
            "| Action | Decision | Risk Score | Reason |",
            "|--------|----------|------------|--------|",
        ])
        for skip in data["skipped_reasons"]:
            risk = f"{skip['risk_score']:.2f}" if skip["risk_score"] is not None else "N/A"
            lines.append(f"| {skip['action']} | {skip['decision']} | {risk} | {skip['suggestion'][:50]} |")
        lines.append("")

    # Cost
    if data["cost"]["total"] > 0:
        lines.extend([
            "## Cost Summary",
            "",
            f"- **Total cost:** ${data['cost']['total']:.2f}",
            f"- **Tokens used:** {data['cost']['tokens']:,}",
            "",
        ])

    # Files reviewed
    if data["files_reviewed"]:
        lines.extend([
            "## Files Reviewed",
            "",
        ])
        for f in sorted(data["files_reviewed"])[:20]:
            lines.append(f"- `{f}`")
        if len(data["files_reviewed"]) > 20:
            lines.append(f"- ... and {len(data['files_reviewed']) - 20} more")
        lines.append("")

    # Footer
    lines.extend([
        "---",
        "*Generated by [CapSeal](https://capseal.dev)*",
    ])

    return "\n".join(lines)


def _generate_text_report(data: dict, run_path: Path) -> str:
    """Generate plain text report."""
    lines = [
        "=" * 60,
        "CAPSEAL REPORT",
        "=" * 60,
        "",
        f"Run:     {run_path.name}",
        f"Type:    {data['run_type']}",
        f"Verdict: {data['verdict']}",
        "",
        "-" * 60,
        "SUMMARY",
        "-" * 60,
        f"Files reviewed:    {len(data['files_reviewed'])}",
        f"Findings:          {data['findings_count']}",
        f"Patches passed:    {data['gate_decisions']['pass']}",
        f"Patches skipped:   {data['gate_decisions']['skip']}",
        f"Flagged for review: {data['gate_decisions']['human_review']}",
        "",
    ]

    if data["verification"]["capsule_hash"]:
        valid_str = "VALID" if data["verification"]["valid"] else "INVALID"
        lines.extend([
            "-" * 60,
            "VERIFICATION",
            "-" * 60,
            f"Capsule: {data['verification']['capsule_hash']}...",
            f"Status:  {valid_str}",
            "",
        ])

    if data["skipped_reasons"]:
        lines.extend([
            "-" * 60,
            "SKIPPED PATCHES",
            "-" * 60,
        ])
        for skip in data["skipped_reasons"]:
            risk = f"{skip['risk_score']:.2f}" if skip["risk_score"] is not None else "N/A"
            lines.append(f"  [{skip['decision'].upper()}] {skip['action']}")
            lines.append(f"    Risk: {risk} | {skip['suggestion']}")
        lines.append("")

    if data["cost"]["total"] > 0:
        lines.extend([
            "-" * 60,
            "COST",
            "-" * 60,
            f"Total:  ${data['cost']['total']:.2f}",
            f"Tokens: {data['cost']['tokens']:,}",
            "",
        ])

    lines.append("=" * 60)

    return "\n".join(lines)


__all__ = ["report_command"]
