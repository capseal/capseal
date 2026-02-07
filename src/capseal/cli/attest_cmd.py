"""CLI commands for project trace attestations."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from capseal.attest import (
    SEVERITY_ORDER,
    diff_summaries,
    load_profile,
    load_summary,
    run_attestation,
)


def _severity_choice(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = value.lower()
    if value not in SEVERITY_ORDER:
        raise click.BadParameter(f"Unknown severity: {value}")
    return value


@click.command("attest")
@click.option("--run", "run_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--project-dir", type=click.Path(exists=True, file_okay=False), required=True)
@click.option("--profile", "profile_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--profile-id", required=True)
@click.option(
    "--fail-on",
    type=click.Choice(["error", "warning", "info"]),
    default=None,
    help="Fail if violations at or above this severity appear",
)
def attest_command(run_dir, project_dir, profile_path, profile_id, fail_on):
    """Compute semantic attestations over a traced run."""

    run_path = Path(run_dir)
    proj = Path(project_dir)
    profile = load_profile(Path(profile_path), profile_id)

    summary, stats = run_attestation(run_path, proj, profile_id, profile)

    total_files = sum(v["files"] for v in stats.values())
    total_violations = len(summary.get("violations", []))
    click.echo(f"Attestation profile: {profile_id}")
    click.echo(f"  Files processed: {total_files}")
    for tracer_name, counts in stats.items():
        click.echo(
            f"  - {tracer_name}: {counts['files']} files, {counts['violations']} violations"
        )
    click.echo(f"  Violations: {total_violations}")
    summary_path = run_path / "attestations" / profile_id / "summary.json"
    click.echo(f"Summary: {summary_path}")

    if fail_on:
        blocking = [
            v for v in summary.get("violations", [])
            if SEVERITY_ORDER.get(v.get("severity", ""), 0) >= SEVERITY_ORDER[fail_on]
        ]
        if blocking:
            click.echo(
                f"FAIL: {len(blocking)} violations >= {fail_on}. See summary for details.",
                err=True,
            )
            raise SystemExit(1)


@click.command("attest-diff")
@click.option("--base", "base_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--head", "head_dir", required=True, type=click.Path(exists=True, file_okay=False))
@click.option("--profile-id", required=True)
@click.option(
    "--fail-on",
    type=click.Choice(["error", "warning", "info"]),
    default=None,
    help="Fail if new violations meet or exceed this severity",
)
def attest_diff_command(base_dir, head_dir, profile_id, fail_on):
    """Compare two attestation summaries."""

    base_summary = load_summary(Path(base_dir), profile_id)
    head_summary = load_summary(Path(head_dir), profile_id)
    diff = diff_summaries(base_summary, head_summary)

    click.echo(f"Attestation diff (profile={profile_id})")
    click.echo(f"  Base trace: {base_summary.get('trace_root', '')[:32]}...")
    click.echo(f"  Head trace: {head_summary.get('trace_root', '')[:32]}...")
    click.echo(f"  New violations: {len(diff['new'])}")
    click.echo(f"  Resolved violations: {len(diff['resolved'])}")
    click.echo(f"  Unchanged violations: {len(diff['unchanged'])}")

    if fail_on:
        blocking = [
            v for v in diff["new"]
            if SEVERITY_ORDER.get(v.get("severity", ""), 0) >= SEVERITY_ORDER[fail_on]
        ]
        if blocking:
            click.echo(
                f"FAIL: {len(blocking)} new {fail_on}+ violations detected.",
                err=True,
            )
            raise SystemExit(1)

