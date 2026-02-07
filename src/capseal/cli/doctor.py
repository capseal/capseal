"""Doctor command - one-click pipeline verification with derived reports."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from .trace_schema import get_schema, TraceSchema
from .utils import find_repo_root


@dataclass
class CheckResult:
    """Result of a single check."""
    name: str
    status: str  # "pass", "fail", "skip", "warn"
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class PipelineReport:
    """Complete pipeline verification report."""
    capsule_path: str
    capsule_hash: str
    report_timestamp: str
    checks: list[CheckResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    index: dict[str, str] = field(default_factory=dict)  # artifact -> path/hash

    def overall_status(self) -> str:
        if any(c.status == "fail" for c in self.checks):
            return "FAIL"
        if any(c.status == "warn" for c in self.checks):
            return "WARN"
        return "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "capsule_path": self.capsule_path,
            "capsule_hash": self.capsule_hash,
            "report_timestamp": self.report_timestamp,
            "overall_status": self.overall_status(),
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "duration_ms": c.duration_ms,
                    "details": c.details,
                    "error": c.error,
                }
                for c in self.checks
            ],
            "summary": self.summary,
            "index": self.index,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Pipeline Verification Report",
            f"",
            f"**Generated:** {self.report_timestamp}",
            f"**Capsule:** `{self.capsule_path}`",
            f"**Hash:** `{self.capsule_hash[:16]}...`",
            f"**Status:** {'✓' if self.overall_status() == 'PASS' else '✗'} {self.overall_status()}",
            f"",
            f"## Checks",
            f"",
            f"| Check | Status | Duration | Details |",
            f"|-------|--------|----------|---------|",
        ]
        for c in self.checks:
            icon = {"pass": "✓", "fail": "✗", "warn": "⚠", "skip": "○"}[c.status]
            detail_str = ", ".join(f"{k}={v}" for k, v in list(c.details.items())[:3])
            if c.error:
                detail_str = c.error[:50]
            lines.append(f"| {c.name} | {icon} {c.status} | {c.duration_ms:.1f}ms | {detail_str} |")

        lines.extend([
            f"",
            f"## Summary",
            f"",
        ])
        for k, v in self.summary.items():
            lines.append(f"- **{k}:** {v}")

        lines.extend([
            f"",
            f"## Artifact Index",
            f"",
            f"| Artifact | Location/Hash |",
            f"|----------|---------------|",
        ])
        for k, v in self.index.items():
            lines.append(f"| {k} | `{v[:60]}{'...' if len(v) > 60 else ''}` |")

        return "\n".join(lines)


def _run_cli(args: list[str], timeout: float = 30.0) -> tuple[int, str, str]:
    """Run capseal CLI command and capture output."""
    # Find the capseal script using repo root detection
    repo_root = find_repo_root()
    capseal = repo_root / "capseal"
    if not capseal.exists():
        # Fallback: try to find in PATH or as module
        capseal = Path("capseal")

    cmd = [str(capseal)] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


def _hash_file(path: Path) -> str:
    """Compute SHA256 of file."""
    if not path.exists():
        return "NOT_FOUND"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_capsule(path: Path) -> dict[str, Any] | None:
    """Load capsule JSON."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def run_doctor(capsule_path: Path, output_dir: Path | None = None, sample_rows: int = 1) -> PipelineReport:
    """Run full pipeline verification and generate report."""

    report = PipelineReport(
        capsule_path=str(capsule_path),
        capsule_hash=_hash_file(capsule_path),
        report_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    capsule_data = _load_capsule(capsule_path)
    if not capsule_data:
        report.checks.append(CheckResult(
            name="load_capsule",
            status="fail",
            duration_ms=0,
            error="Failed to load capsule JSON",
        ))
        return report

    # Index the capsule artifact
    report.index["capsule_json"] = str(capsule_path)
    report.index["capsule_hash"] = report.capsule_hash

    # Check 1: Inspect
    t0 = time.perf_counter()
    code, stdout, stderr = _run_cli(["inspect", str(capsule_path)])
    duration = (time.perf_counter() - t0) * 1000

    inspect_details = {}
    if code == 0:
        for line in stdout.strip().split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                inspect_details[k.strip().lower().replace(" ", "_")] = v.strip()

    report.checks.append(CheckResult(
        name="inspect",
        status="pass" if code == 0 else "fail",
        duration_ms=duration,
        details=inspect_details,
        error=stderr if code != 0 else None,
    ))

    # Index from inspect
    if "capsule_id" in inspect_details:
        report.index["capsule_id"] = inspect_details["capsule_id"]
    if "trace_id" in inspect_details:
        report.index["trace_id"] = inspect_details["trace_id"]

    # Check 2: Verify
    t0 = time.perf_counter()
    code, stdout, stderr = _run_cli(["verify", str(capsule_path), "--json"])
    duration = (time.perf_counter() - t0) * 1000

    verify_details = {}
    verify_status = "fail"
    if code == 0:
        try:
            vdata = json.loads(stdout)
            verify_details = {
                "status": vdata.get("status"),
                "proof_verified": vdata.get("proof_verified", vdata.get("policy_verified")),
                "backend": vdata.get("backend_id", vdata.get("backend")),
            }
            # Accept both "VERIFIED" (from wrapper) and "PROOF_ONLY" (from raw JSON)
            # PROOF_ONLY means cryptographic proof passed even if policy checks skipped
            passing_statuses = {"VERIFIED", "PROOF_ONLY", "POLICY_ENFORCED"}
            status = vdata.get("status", "")
            verify_status = "pass" if status in passing_statuses or vdata.get("proof_verified") else "fail"
        except json.JSONDecodeError:
            verify_details["raw"] = stdout[:100]

    # Handle specific error codes
    error_msg = None
    if code != 0:
        if "E036" in stderr:
            verify_status = "warn"
            error_msg = "Policy document missing (E036)"
        else:
            error_msg = stderr[:100] if stderr else f"Exit code {code}"

    report.checks.append(CheckResult(
        name="verify",
        status=verify_status,
        duration_ms=duration,
        details=verify_details,
        error=error_msg,
    ))

    # Check 3: Audit
    t0 = time.perf_counter()
    code, stdout, stderr = _run_cli(["audit", str(capsule_path), "--format", "summary"])
    duration = (time.perf_counter() - t0) * 1000

    audit_details = {}
    if code == 0:
        for line in stdout.strip().split("\n"):
            if "Total Events:" in line:
                audit_details["total_events"] = line.split(":")[1].strip()
            if "Hash Chain:" in line:
                audit_details["hash_chain"] = "VALID" if "VALID" in line else "INVALID"
            if "Events Log:" in line:
                events_path = line.split(":")[1].strip()
                report.index["events_log"] = events_path
                if Path(events_path).exists():
                    report.index["events_log_hash"] = _hash_file(Path(events_path))

    report.checks.append(CheckResult(
        name="audit",
        status="pass" if code == 0 and audit_details.get("hash_chain") == "VALID" else "fail",
        duration_ms=duration,
        details=audit_details,
        error=stderr if code != 0 else None,
    ))

    # Check 4: Sample row opening
    for row_idx in range(sample_rows):
        t0 = time.perf_counter()
        code, stdout, stderr = _run_cli(["row", str(capsule_path), "--row", str(row_idx), "--json"])
        duration = (time.perf_counter() - t0) * 1000

        row_details = {}
        if code == 0:
            try:
                rdata = json.loads(stdout)
                row_details = {
                    "row_index": rdata.get("row_index"),
                    "commitment": rdata.get("commitment", "")[:16] + "...",
                    "proof_length": len(rdata.get("proof", [])),
                    "values_count": len(rdata.get("row_values", [])),
                }

                # Try to decode with schema
                schema_id = capsule_data.get("trace_schema_id", inspect_details.get("schema"))
                schema = get_schema(schema_id) if schema_id else None
                if schema and rdata.get("row_values"):
                    decoded = schema.decode_row(rdata["row_values"])
                    row_details["decoded_fields"] = list(decoded.keys())[:5]

                report.index[f"row_{row_idx}_commitment"] = rdata.get("commitment", "")
            except json.JSONDecodeError:
                row_details["raw"] = stdout[:100]

        report.checks.append(CheckResult(
            name=f"row_open_{row_idx}",
            status="pass" if code == 0 else "fail",
            duration_ms=duration,
            details=row_details,
            error=stderr[:100] if code != 0 and stderr else None,
        ))

    # Check 5: Schema binding (if present)
    capsule_schema_id = capsule_data.get("trace_schema_id")
    capsule_schema_hash = capsule_data.get("trace_schema_hash")
    if capsule_schema_id:
        t0 = time.perf_counter()
        schema = get_schema(capsule_schema_id)
        duration = (time.perf_counter() - t0) * 1000

        schema_details = {
            "schema_id": capsule_schema_id,
            "capsule_hash": capsule_schema_hash[:16] + "..." if capsule_schema_hash else None,
        }
        schema_status = "pass"
        schema_error = None

        if not schema:
            schema_status = "warn"
            schema_error = f"Schema '{capsule_schema_id}' not in registry"
        elif capsule_schema_hash:
            computed_hash = schema.hash()
            schema_details["computed_hash"] = computed_hash[:16] + "..."
            if computed_hash != capsule_schema_hash:
                schema_status = "fail"
                schema_error = "Schema hash mismatch"
            else:
                schema_details["hash_match"] = True

        report.checks.append(CheckResult(
            name="schema_binding",
            status=schema_status,
            duration_ms=duration,
            details=schema_details,
            error=schema_error,
        ))
        report.index["trace_schema_id"] = capsule_schema_id
        if capsule_schema_hash:
            report.index["trace_schema_hash"] = capsule_schema_hash

    # Build summary from capsule data
    report.summary = {
        "capsule_id": capsule_data.get("capsule_id", inspect_details.get("capsule_id", "unknown")),
        "trace_id": capsule_data.get("trace_id", inspect_details.get("trace_id", "unknown")),
        "backend": capsule_data.get("backend_id", inspect_details.get("backend", "unknown")),
        "schema": capsule_data.get("schema", inspect_details.get("schema", "unknown")),
        "trace_schema_id": capsule_schema_id,
        "checks_passed": sum(1 for c in report.checks if c.status == "pass"),
        "checks_total": len(report.checks),
        "total_duration_ms": sum(c.duration_ms for c in report.checks),
    }

    # Index additional artifacts from capsule
    capsule_dir = capsule_path.parent

    # Policy
    policy_path = capsule_dir / "policy.json"
    if policy_path.exists():
        report.index["policy_json"] = str(policy_path)
        report.index["policy_hash"] = _hash_file(policy_path)

    # Row archive
    row_archive = capsule_dir / "row_archive"
    if row_archive.is_dir():
        report.index["row_archive_dir"] = str(row_archive)
        chunks = list(row_archive.glob("chunk_*.json"))
        report.index["row_archive_chunks"] = str(len(chunks))

    # Proofs
    proofs_dir = capsule_dir / "proofs"
    if proofs_dir.is_dir():
        report.index["proofs_dir"] = str(proofs_dir)

    # Manifests
    manifests_dir = capsule_dir / "manifests"
    if manifests_dir.is_dir():
        report.index["manifests_dir"] = str(manifests_dir)
        manifest_index = manifests_dir / "manifest_index.json"
        if manifest_index.exists():
            report.index["manifest_index_hash"] = _hash_file(manifest_index)

    return report


@click.command("doctor")
@click.argument("capsule", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None,
              help="Output directory for reports (default: <capsule_dir>/doctor_report/)")
@click.option("--sample-rows", type=int, default=1, help="Number of rows to sample for opening")
@click.option("--json", "json_output", is_flag=True, help="Output JSON only (no markdown)")
@click.option("--quiet", "-q", is_flag=True, help="Minimal output")
def doctor(capsule: Path, output: Path | None, sample_rows: int, json_output: bool, quiet: bool):
    """One-click pipeline verification with derived reports.

    Runs: inspect → verify → audit → row opening → summary

    Produces:
        report.json  - Machine-readable verification report
        report.md    - Human-readable summary
        index/       - Links to all artifacts with hashes

    Examples:
        capseal doctor out/run/strategy_capsule.json
        capseal doctor capsule.json -o ./reports --sample-rows 3
    """

    if not quiet:
        click.echo(f"Running pipeline doctor on: {capsule}")
        click.echo("")

    report = run_doctor(capsule, output, sample_rows)

    # Determine output directory
    if output is None:
        output = capsule.parent / "doctor_report"
    output.mkdir(parents=True, exist_ok=True)

    # Write JSON report
    json_path = output / "report.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2))

    # Write Markdown report
    md_path = output / "report.md"
    md_path.write_text(report.to_markdown())

    # Write index file
    index_path = output / "index.json"
    index_path.write_text(json.dumps(report.index, indent=2))

    if json_output:
        click.echo(json.dumps(report.to_dict(), indent=2))
    else:
        # Print summary
        status_icon = "✓" if report.overall_status() == "PASS" else ("⚠" if report.overall_status() == "WARN" else "✗")
        click.echo(f"{status_icon} {report.overall_status()}")
        click.echo("")

        if not quiet:
            click.echo("Checks:")
            for c in report.checks:
                icon = {"pass": "✓", "fail": "✗", "warn": "⚠", "skip": "○"}[c.status]
                click.echo(f"  {icon} {c.name}: {c.status} ({c.duration_ms:.1f}ms)")
                if c.error:
                    click.echo(f"      {c.error}")

            click.echo("")
            click.echo(f"Reports written to: {output}/")
            click.echo(f"  report.json  - Machine-readable")
            click.echo(f"  report.md    - Human-readable")
            click.echo(f"  index.json   - Artifact hashes")

    # Exit with appropriate code
    sys.exit(0 if report.overall_status() == "PASS" else 1)
