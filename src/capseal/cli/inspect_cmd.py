"""capsule inspect command - display capsule metadata.

Usage:
    capsule inspect <receipt.cap> [--json]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from .cap_format import read_cap_manifest, read_cap_capsule


def _format_size(size: int) -> str:
    """Format byte size in human-readable form."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"


def _inspect_cap_file(cap_path: Path) -> dict[str, Any]:
    """Extract inspection metadata from a .cap file."""
    import tarfile

    manifest = read_cap_manifest(cap_path)

    # Get archive stats
    stats = {
        "total_size": cap_path.stat().st_size,
        "files": [],
    }

    with tarfile.open(cap_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.isfile():
                stats["files"].append({
                    "name": member.name,
                    "size": member.size,
                })

    return {
        "format": ".cap",
        "manifest": manifest.to_dict(),
        "stats": stats,
    }


def _inspect_capsule_json(capsule_path: Path) -> dict[str, Any]:
    """Extract inspection metadata from a capsule.json file."""
    capsule = json.loads(capsule_path.read_text())
    header = capsule.get("header", {})

    return {
        "format": "capsule.json",
        "capsule_id": capsule.get("capsule_hash", "")[:16],
        "trace_id": capsule.get("trace_id", ""),
        "policy_id": header.get("policy_id", capsule.get("policy_id", "")),
        "policy_hash": header.get("policy_hash", capsule.get("policy_hash", "")),
        "backend": header.get("backend_id", capsule.get("backend", "")),
        "verification_profile": header.get(
            "verification_profile",
            capsule.get("verification_profile", "")
        ),
        "schema": capsule.get("schema", ""),
        "header_schema": header.get("schema", ""),
        "stats": {
            "file_size": capsule_path.stat().st_size,
        },
    }


@click.command("inspect")
@click.argument("capsule", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output as JSON",
)
def inspect_command(capsule: Path, output_json: bool) -> None:
    """Display capsule metadata.

    Shows key information about a capsule without full verification.
    Supports both .cap archives and raw capsule.json files.
    """
    if capsule.suffix == ".cap":
        info = _inspect_cap_file(capsule)
    else:
        info = _inspect_capsule_json(capsule)

    if output_json:
        click.echo(json.dumps(info, indent=2))
    else:
        # Human-readable output
        click.echo(f"Format:     {info.get('format', 'unknown')}")

        if "manifest" in info:
            m = info["manifest"]
            click.echo(f"Capsule ID: {m.get('capsule_id', 'N/A')}")
            click.echo(f"Trace ID:   {m.get('trace_id', 'N/A')}")
            click.echo(f"Policy ID:  {m.get('policy_id', 'N/A')}")
            click.echo(f"Backend:    {m.get('backend', 'N/A')}")
            click.echo(f"Profile:    {m.get('verification_profile', 'N/A')}")
            click.echo(f"Chunks:     {m.get('num_chunks', 0)}")
            click.echo(f"Created:    {m.get('created_at', 'N/A')}")

            if info.get("stats"):
                stats = info["stats"]
                click.echo(f"Archive:    {_format_size(stats.get('total_size', 0))}")
                if stats.get("files"):
                    click.echo("Contents:")
                    for f in stats["files"]:
                        click.echo(f"  {f['name']}: {_format_size(f['size'])}")
        else:
            click.echo(f"Capsule ID: {info.get('capsule_id', 'N/A')}")
            click.echo(f"Trace ID:   {info.get('trace_id', 'N/A')}")
            click.echo(f"Policy ID:  {info.get('policy_id', 'N/A')}")
            click.echo(f"Backend:    {info.get('backend', 'N/A')}")
            click.echo(f"Profile:    {info.get('verification_profile', 'N/A')}")
            click.echo(f"Schema:     {info.get('schema', 'N/A')}")

            if info.get("stats"):
                stats = info["stats"]
                click.echo(f"File size:  {_format_size(stats.get('file_size', 0))}")
