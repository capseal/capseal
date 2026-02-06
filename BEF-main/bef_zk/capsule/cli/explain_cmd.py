"""Human-readable verification report command."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from ..contracts import (
    ExitCode,
    VerificationProfile,
    is_schema_supported,
)
from .exit_codes import (
    EXIT_VERIFIED,
    EXIT_PROOF_INVALID,
    EXIT_POLICY_MISMATCH,
    EXIT_COMMITMENT_FAILED,
    EXIT_DA_AUDIT_FAILED,
    EXIT_REPLAY_DIVERGED,
    EXIT_MALFORMED,
    exit_code_description,
)


def _format_hash(h: str | None, length: int = 16) -> str:
    """Format a hash for display."""
    if not h:
        return "(none)"
    if len(h) > length:
        return f"{h[:length]}..."
    return h


def _check_section(name: str, checks: list[tuple[str, bool, str]]) -> dict:
    """
    Run a section of checks and return results.

    checks: list of (check_name, passed, detail_if_failed)
    """
    results = []
    all_passed = True

    for check_name, passed, detail in checks:
        results.append({
            "check": check_name,
            "status": "PASS" if passed else "FAIL",
            "detail": None if passed else detail,
        })
        if not passed:
            all_passed = False

    return {
        "section": name,
        "status": "PASS" if all_passed else "FAIL",
        "checks": results,
    }


def explain_receipt(receipt: dict) -> dict:
    """
    Generate a detailed verification report for a receipt.

    Returns a structured report with pass/fail for each verification step.
    """
    report = {
        "summary": {
            "capsule_id": receipt.get("capsule_id", "(unknown)"),
            "schema": receipt.get("schema", "(unknown)"),
            "capseal_version": receipt.get("capseal_version", "(unknown)"),
        },
        "sections": [],
        "overall_status": "PASS",
        "exit_code": EXIT_VERIFIED,
        "exit_description": "verified",
    }

    # Section 1: Schema Compatibility
    schema = receipt.get("schema", "")
    schema_checks = [
        (
            "Schema version supported",
            is_schema_supported(schema, "receipt"),
            f"Unsupported schema: {schema}",
        ),
        (
            "Required top-level fields",
            all(k in receipt for k in ["schema", "capsule_id", "header", "payload"]),
            "Missing required fields",
        ),
    ]
    report["sections"].append(_check_section("Schema", schema_checks))

    # Section 2: Header Integrity
    header = receipt.get("header", {})
    header_checks = [
        (
            "Header schema valid",
            header.get("schema", "").startswith("capsule_header_"),
            f"Invalid header schema: {header.get('schema')}",
        ),
        (
            "VM ID present",
            bool(header.get("vm_id")),
            "Missing vm_id",
        ),
        (
            "Trace ID present",
            bool(header.get("trace_id")),
            "Missing trace_id",
        ),
        (
            "Statement hash present",
            bool(header.get("statement_hash")),
            "Missing statement_hash",
        ),
        (
            "Row commitment present",
            bool(header.get("row_commitment")),
            "Missing row_commitment",
        ),
    ]
    report["sections"].append(_check_section("Header Integrity", header_checks))

    # Section 3: Policy Binding
    policy_ref = header.get("policy_ref", {})
    policy_checks = [
        (
            "Policy ID present",
            bool(policy_ref.get("policy_id")),
            "Missing policy_id in policy_ref",
        ),
        (
            "Policy version present",
            bool(policy_ref.get("policy_version")),
            "Missing policy_version",
        ),
        (
            "Policy hash present",
            bool(policy_ref.get("policy_hash")),
            "Missing policy_hash (cannot verify policy binding)",
        ),
    ]
    report["sections"].append(_check_section("Policy Binding", policy_checks))

    # Section 4: Proof Binding
    payload = receipt.get("payload", {})
    proof_checks = [
        (
            "Proof format specified",
            bool(payload.get("proof_format")),
            "Missing proof_format",
        ),
        (
            "Proof data present",
            bool(payload.get("proof_data")),
            "Missing proof_data",
        ),
    ]

    # Check if proof format is demo (not cryptographically verified)
    proof_format = payload.get("proof_format", "")
    if proof_format == "demo_v1":
        proof_checks.append((
            "Demo proof (not cryptographic)",
            True,
            None,
        ))

    report["sections"].append(_check_section("Proof Binding", proof_checks))

    # Section 5: Row Commitment
    row_commitment = header.get("row_commitment", {})
    commitment_checks = [
        (
            "Row root present",
            bool(row_commitment.get("root")),
            "Missing row root",
        ),
        (
            "Tree arity specified",
            row_commitment.get("tree_arity") is not None,
            "Missing tree_arity",
        ),
        (
            "Chunk handles root present",
            bool(row_commitment.get("chunk_handles_root")),
            "Missing chunk_handles_root",
        ),
    ]
    report["sections"].append(_check_section("Row Commitment", commitment_checks))

    # Section 6: Data Availability (if applicable)
    da_ref = header.get("da_ref", {})
    if da_ref:
        da_checks = [
            (
                "DA policy hash present",
                bool(da_ref.get("policy_hash")),
                "Missing DA policy_hash",
            ),
        ]
        report["sections"].append(_check_section("Data Availability", da_checks))

    # Compute overall status
    failed_sections = [s for s in report["sections"] if s["status"] == "FAIL"]
    if failed_sections:
        report["overall_status"] = "FAIL"
        # Determine exit code based on which section failed
        for section in failed_sections:
            if section["section"] == "Schema":
                report["exit_code"] = EXIT_MALFORMED
                break
            elif section["section"] == "Policy Binding":
                report["exit_code"] = EXIT_POLICY_MISMATCH
                break
            elif section["section"] == "Proof Binding":
                report["exit_code"] = EXIT_PROOF_INVALID
                break
            elif section["section"] == "Row Commitment":
                report["exit_code"] = EXIT_COMMITMENT_FAILED
                break
            elif section["section"] == "Data Availability":
                report["exit_code"] = EXIT_DA_AUDIT_FAILED
                break
        report["exit_description"] = exit_code_description(report["exit_code"])

    return report


def format_report_text(report: dict) -> str:
    """Format report as human-readable text."""
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append("CAPSEAL VERIFICATION REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Summary
    summary = report["summary"]
    lines.append(f"Capsule ID: {_format_hash(summary['capsule_id'], 32)}")
    lines.append(f"Schema:     {summary['schema']}")
    lines.append(f"Version:    {summary['capseal_version']}")
    lines.append("")

    # Sections
    for section in report["sections"]:
        status_symbol = "+" if section["status"] == "PASS" else "X"
        lines.append(f"[{status_symbol}] {section['section']}")

        for check in section["checks"]:
            check_symbol = "+" if check["status"] == "PASS" else "X"
            lines.append(f"    [{check_symbol}] {check['check']}")
            if check["detail"]:
                lines.append(f"        -> {check['detail']}")

        lines.append("")

    # Overall result
    lines.append("-" * 70)
    overall = report["overall_status"]
    if overall == "PASS":
        lines.append("RESULT: VERIFIED")
    else:
        lines.append(f"RESULT: FAILED ({report['exit_description']})")
        lines.append(f"Exit Code: {report['exit_code']}")
    lines.append("-" * 70)

    return "\n".join(lines)


@click.command("explain")
@click.argument("receipt_path", type=click.Path(exists=True))
@click.option(
    "--json", "json_output",
    is_flag=True,
    help="Output JSON report",
)
@click.option(
    "--section",
    type=click.Choice([
        "schema", "header", "policy", "proof", "commitment", "da"
    ]),
    help="Show only a specific section",
)
def explain_command(
    receipt_path: str,
    json_output: bool,
    section: str | None,
) -> None:
    """Generate a human-readable verification report.

    Unlike 'verify' which returns a binary pass/fail, 'explain' provides
    detailed information about each verification check, helping diagnose
    why a receipt might be invalid.

    \b
    Example:
        capseal explain receipt.json
        capseal explain receipt.json --json
        capseal explain receipt.json --section proof
    """
    # Load receipt
    try:
        with open(receipt_path) as f:
            receipt = json.load(f)
    except json.JSONDecodeError as e:
        if json_output:
            click.echo(json.dumps({
                "error": "parse_error",
                "message": str(e),
            }))
        else:
            click.echo(f"Error: Failed to parse receipt: {e}")
        raise SystemExit(EXIT_MALFORMED)

    # Generate report
    report = explain_receipt(receipt)

    # Filter to specific section if requested
    if section:
        section_map = {
            "schema": "Schema",
            "header": "Header Integrity",
            "policy": "Policy Binding",
            "proof": "Proof Binding",
            "commitment": "Row Commitment",
            "da": "Data Availability",
        }
        target = section_map.get(section)
        if target:
            report["sections"] = [
                s for s in report["sections"]
                if s["section"] == target
            ]

    # Output
    if json_output:
        click.echo(json.dumps(report, indent=2))
    else:
        click.echo(format_report_text(report))

    # Exit with appropriate code
    raise SystemExit(report["exit_code"])


__all__ = ["explain_command", "explain_receipt", "format_report_text"]
