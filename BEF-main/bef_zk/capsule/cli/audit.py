"""Audit command - export and inspect audit trails from capsules."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import click

from .exit_codes import EXIT_VERIFIED, EXIT_MALFORMED
from .security_assumptions import (
    build_security_assumptions,
    dataset_info_from_capsule,
    policy_info_from_capsule,
    print_security_assumptions,
)
from bef_zk.codec import canonical_decode
from .cap_format import CapExtractionError, extract_cap_file


def _load_capsule_with_base(path: Path) -> tuple[dict[str, Any], Path, tempfile.TemporaryDirectory | None]:
    """Load a capsule and return artifact base dir + optional temp context."""
    if path.suffix == ".cap":
        tmpdir = tempfile.TemporaryDirectory()
        extract_dir = Path(tmpdir.name)
        try:
            extract_cap_file(path, extract_dir)
        except CapExtractionError as exc:
            tmpdir.cleanup()
            raise click.ClickException(f"failed to extract capsule: {exc}")
        capsule_json = extract_dir / "capsule.json"
        if not capsule_json.exists():
            tmpdir.cleanup()
            raise click.ClickException("capsule archive missing capsule.json")
        data = json.loads(capsule_json.read_text())
        return data, extract_dir, tmpdir
    if path.suffix == ".json":
        return json.loads(path.read_text()), path.parent, None
    data = canonical_decode(path.read_bytes())
    return data, path.parent, None


def _find_events_log(capsule: dict[str, Any], base_dir: Path) -> Path | None:
    """Find the events log file referenced by a capsule."""
    artifacts = capsule.get("artifacts", {})
    events_entry = artifacts.get("events_log", {})

    # Try rel_path first
    rel_path = events_entry.get("rel_path") or events_entry.get("path")
    if rel_path:
        # Relative to capsule directory
        candidate = base_dir / rel_path
        if candidate.exists():
            return candidate

    # Try absolute path
    abs_path = events_entry.get("path")
    if abs_path:
        candidate = Path(abs_path)
        if candidate.exists():
            return candidate

    # Look for events.jsonl in same directory
        candidate = base_dir / "events.jsonl"
    if candidate.exists():
        return candidate

    return None


def _verify_event_chain(events: list[dict[str, Any]]) -> tuple[bool, str]:
    """Verify the hash chain of events."""
    prev_hash = ""
    genesis_ok = False
    for i, event in enumerate(events):
        recorded_prev = event.get("prev_event_hash", "")
        if i == 0:
            if recorded_prev in ("", "0" * 64):
                genesis_ok = True
            else:
                return False, "Event 0: prev_hash must be zero"
            prev_for_hash = recorded_prev or ""
        else:
            if recorded_prev != prev_hash:
                return False, f"Event {i}: prev_hash mismatch (expected {prev_hash[:16]}..., got {recorded_prev[:16]}...)"
            prev_for_hash = recorded_prev

        # Recompute event hash
        event_copy = {k: v for k, v in event.items() if k not in {"event_hash", "prev_event_hash"}}
        canonical = json.dumps(event_copy, sort_keys=True, separators=(",", ":")).encode()
        prev_bytes = bytes.fromhex(prev_for_hash) if prev_for_hash else b""
        computed_hash = hashlib.sha256(prev_bytes + canonical).hexdigest()

        recorded_hash = event.get("event_hash", "")
        if recorded_hash and recorded_hash != computed_hash:
            return False, f"Event {i}: hash mismatch (computed {computed_hash[:16]}..., recorded {recorded_hash[:16]}...)"

        prev_hash = recorded_hash or computed_hash

    if genesis_ok:
        return True, "GENESIS_OK"
    return True, ""


def _parse_events(events_path: Path) -> list[dict[str, Any]]:
    """Parse JSONL events file."""
    events = []
    with events_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")
    return events


@click.command("audit")
@click.argument("capsule", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output file (default: stdout)",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["json", "jsonl", "csv", "summary"]),
    default="summary",
    help="Output format",
)
@click.option(
    "--verify/--no-verify",
    default=True,
    help="Verify hash chain integrity",
)
@click.option(
    "--filter-type",
    type=str,
    help="Filter events by type (e.g., 'proof_artifact')",
)
@click.option(
    "--from-seq",
    type=int,
    default=0,
    help="Start from this sequence number",
)
@click.option(
    "--to-seq",
    type=int,
    default=None,
    help="End at this sequence number",
)
def audit_command(
    capsule: Path,
    output: Optional[Path],
    format: str,
    verify: bool,
    filter_type: Optional[str],
    from_seq: int,
    to_seq: Optional[int],
) -> None:
    """Export and inspect audit trail from a capsule.

    Extracts the event log, verifies hash chain integrity, and exports
    in various formats for analysis.

    \b
    Formats:
        summary - Human-readable summary (default)
        json    - Full events as JSON array
        jsonl   - Events as JSON lines
        csv     - Tabular format for spreadsheets

    \b
    Examples:
        # Show audit summary
        capsule audit out/run/strategy_capsule.json

        # Export to CSV
        capsule audit out/run/strategy_capsule.json -f csv -o audit.csv

        # Export specific event types
        capsule audit out/run/strategy_capsule.json -f json --filter-type proof_artifact
    """
    # Load capsule
    try:
        capsule_data, base_dir, tmp_ctx = _load_capsule_with_base(capsule)
    except Exception as e:
        click.echo(f"Error: Failed to load capsule: {e}", err=True)
        sys.exit(EXIT_MALFORMED)

    # Find events log
    events_path = _find_events_log(capsule_data, base_dir)
    if events_path is None:
        click.echo("Error: Events log not found", err=True)
        click.echo("Checked:", err=True)
        click.echo(f"  - artifacts.events_log in capsule", err=True)
        click.echo(f"  - {base_dir / 'events.jsonl'}", err=True)
        if tmp_ctx:
            tmp_ctx.cleanup()
        sys.exit(EXIT_MALFORMED)

    # Parse events
    try:
        events = _parse_events(events_path)
    except Exception as e:
        click.echo(f"Error: Failed to parse events: {e}", err=True)
        sys.exit(EXIT_MALFORMED)

    # Verify hash chain
    chain_valid = True
    chain_error = ""
    if verify:
        chain_valid, chain_error = _verify_event_chain(events)

    warnings = []
    if verify and not chain_valid and chain_error:
        warnings.append(chain_error)

    assumptions = build_security_assumptions(
        operation="audit",
        capsule=capsule_data,
        policy_info=policy_info_from_capsule(capsule_data),
        dataset_info=dataset_info_from_capsule(capsule_data),
        warnings=warnings or None,
    )

    # Filter events
    filtered = events
    if filter_type:
        filtered = [e for e in events if e.get("type") == filter_type]
    if from_seq > 0:
        filtered = [e for e in filtered if e.get("seq", 0) >= from_seq]
    if to_seq is not None:
        filtered = [e for e in filtered if e.get("seq", 0) <= to_seq]

    # Format output
    def write_output(content: str) -> None:
        if output:
            output.write_text(content)
            click.echo(f"Written to {output}")
        else:
            click.echo(content)

    if format == "summary":
        # Build summary
        lines = []
        lines.append(f"Audit Trail: {capsule.name}")
        lines.append(f"Events Log: {events_path}")
        lines.append(f"Total Events: {len(events)}")
        if filtered != events:
            lines.append(f"Filtered Events: {len(filtered)}")
        lines.append("")

        if verify:
            if chain_valid:
                if chain_error == "GENESIS_OK":
                    lines.append("Hash Chain: VALID (genesis prev_hash = 0)")
                else:
                    lines.append("Hash Chain: VALID")
            else:
                lines.append(f"Hash Chain: INVALID - {chain_error}")
        lines.append("")

        # Event type summary
        type_counts: dict[str, int] = {}
        for e in events:
            t = e.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        lines.append("Event Types:")
        for t, count in sorted(type_counts.items()):
            lines.append(f"  {t}: {count}")
        lines.append("")

        # Timeline
        if events:
            first = events[0]
            last = events[-1]
            lines.append("Timeline:")
            lines.append(f"  First: seq={first.get('seq', 0)}, type={first.get('type', '?')}")
            lines.append(f"  Last:  seq={last.get('seq', 0)}, type={last.get('type', '?')}")

            # Check recorded hash in capsule
            anchor = capsule_data.get("anchor", {})
            recorded_hash = anchor.get("events_log_hash", "")
            if recorded_hash:
                lines.append(f"  Capsule events_log_hash: {recorded_hash[:40]}...")

        if assumptions:
            lines.append("")
            lines.append("Security Assumptions:")
            pp = assumptions.get("proof_primitive") or {}
            if pp:
                lines.append(
                    f"  Proof primitive: {(pp.get('backend') or pp.get('pc'))}"
                )
            policy = assumptions.get("policy") or {}
            if policy.get("policy_id"):
                version = policy.get("policy_version") or "unspecified"
                p_hash = policy.get("policy_hash")
                hash_note = f", hash={p_hash[:12]}…" if p_hash else ""
                lines.append(
                    f"  Policy: {policy['policy_id']} (version {version}{hash_note})"
                )
            dataset = assumptions.get("dataset") or {}
            if dataset.get("datasets"):
                lines.append(f"  Datasets committed: {len(dataset['datasets'])}")
            if warnings:
                lines.append("  Warnings: " + ", ".join(warnings))

        write_output("\n".join(lines))

    elif format == "json":
        result = {
            "capsule": str(capsule),
            "events_path": str(events_path),
            "total_events": len(events),
            "filtered_events": len(filtered),
            "chain_valid": chain_valid,
            "chain_error": chain_error if not chain_valid else None,
            "events": filtered,
        }
        if assumptions:
            result["security_assumptions"] = assumptions
        write_output(json.dumps(result, indent=2))

    elif format == "jsonl":
        lines = [json.dumps(e) for e in filtered]
        write_output("\n".join(lines))

    elif format == "csv":
        if not filtered:
            write_output("seq,ts_ms,type,source,data\n")
        else:
            # Flatten events for CSV
            output_io = StringIO()
            fieldnames = ["seq", "ts_ms", "type", "source", "run_id", "trace_id", "data_summary"]
            writer = csv.DictWriter(output_io, fieldnames=fieldnames)
            writer.writeheader()
            for e in filtered:
                data = e.get("data", {})
                data_summary = json.dumps(data)[:100] if data else ""
                writer.writerow({
                    "seq": e.get("seq", ""),
                    "ts_ms": e.get("ts_ms", ""),
                    "type": e.get("type", ""),
                    "source": e.get("source", ""),
                    "run_id": e.get("run_id", ""),
                    "trace_id": e.get("trace_id", ""),
                    "data_summary": data_summary,
                })
            write_output(output_io.getvalue())

    if tmp_ctx:
        tmp_ctx.cleanup()

    # Exit code
    if verify and not chain_valid:
        sys.exit(EXIT_MALFORMED)
    sys.exit(EXIT_VERIFIED)


@click.command("audit-fetch")
@click.argument("run_id", type=str)
@click.option(
    "--relay", "-r",
    type=str,
    required=True,
    help="Relay base URL",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    help="Output file",
)
@click.option(
    "--from-seq",
    type=int,
    default=0,
    help="Start from this sequence number",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["json", "jsonl"]),
    default="jsonl",
    help="Output format",
)
def audit_fetch_command(
    run_id: str,
    relay: str,
    output: Optional[Path],
    from_seq: int,
    format: str,
) -> None:
    """Fetch audit trail from a relay server.

    Retrieves the event log for a specific run from the relay API.

    \b
    Examples:
        # Fetch events
        capsule audit-fetch run_12345 -r https://relay.example.com -o events.jsonl

        # Fetch as JSON
        capsule audit-fetch run_12345 -r https://relay.example.com -f json
    """
    import urllib.request
    import urllib.error

    # Build URL
    url = f"{relay.rstrip('/')}/api/runs/{run_id}/events"
    if from_seq > 0:
        url += f"?last_seq={from_seq}"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        click.echo(f"Error: HTTP {e.code} - {e.reason}", err=True)
        sys.exit(EXIT_MALFORMED)
    except urllib.error.URLError as e:
        click.echo(f"Error: {e.reason}", err=True)
        sys.exit(EXIT_MALFORMED)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_MALFORMED)

    events = data.get("events", [])

    if format == "json":
        content = json.dumps({"run_id": run_id, "events": events}, indent=2)
    else:
        content = "\n".join(json.dumps(e) for e in events)

    if output:
        output.write_text(content)
        click.echo(f"Fetched {len(events)} events to {output}")
    else:
        click.echo(content)

    sys.exit(EXIT_VERIFIED)
