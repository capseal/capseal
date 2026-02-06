"""Capseal diff command - streamlined repo comparison with receipts.

Usage:
    capseal diff ~/projects/CapsuleTech ~/BEF-main
    capseal diff --watch
    capseal diff HEAD~5..HEAD
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import click

# Import MCP tools for receipt generation
try:
    from bef_zk.capsule.mcp_server import (
        tool_diff_bundle,
        tool_verify,
        tool_doctor,
        EVENT_LOG_PATH,
    )
    import bef_zk.capsule.mcp_server as mcp_mod
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


def _clear_receipts():
    """Clear receipt log for fresh session."""
    if os.path.exists(EVENT_LOG_PATH):
        os.remove(EVENT_LOG_PATH)
    if MCP_AVAILABLE:
        mcp_mod._last_hash = None


def _load_receipts() -> list[dict]:
    """Load all receipts."""
    receipts = []
    try:
        with open(EVENT_LOG_PATH, "r") as f:
            for line in f:
                if line.strip():
                    receipts.append(json.loads(line))
    except FileNotFoundError:
        pass
    return receipts


def _print_receipt(r: dict, idx: int):
    """Print a receipt in compact form."""
    tool = r.get("tool", "?")
    ok = "\033[32m✓\033[0m" if r.get("result_ok") else "\033[31m✗\033[0m"
    ts = time.strftime("%H:%M:%S", time.localtime(r.get("ts_ms", 0) / 1000))
    prev = r.get("prev_hash", "")[:8] or "genesis"
    curr = r.get("event_hash", "")[:8]
    click.echo(f"  {ok} [{ts}] {tool:<12} {prev}→{curr}")


def _setup_remote(repo_path: str, remote_path: str, remote_name: str = "compare") -> bool:
    """Add a remote for comparison."""
    try:
        # Check if remote exists
        result = subprocess.run(
            ["git", "-C", repo_path, "remote", "get-url", remote_name],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            # Add the remote
            subprocess.run(
                ["git", "-C", repo_path, "remote", "add", remote_name, remote_path],
                capture_output=True, check=True
            )
        # Fetch
        subprocess.run(
            ["git", "-C", repo_path, "fetch", remote_name],
            capture_output=True, check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


@click.command("diff")
@click.argument("source", required=False)
@click.argument("target", required=False)
@click.option("--ref", "-r", default="HEAD~5..HEAD", help="Git ref range (default: HEAD~5..HEAD)")
@click.option("--watch", "-w", is_flag=True, help="Watch receipts in real-time")
@click.option("--clear", "-c", is_flag=True, help="Clear receipt log before starting")
@click.option("--verify-capsule", "-v", help="Also verify a capsule")
@click.option("--json", "json_out", is_flag=True, help="Output as JSON")
def diff_command(
    source: str | None,
    target: str | None,
    ref: str,
    watch: bool,
    clear: bool,
    verify_capsule: str | None,
    json_out: bool,
):
    """Compare repos/refs and generate hash-chained receipts.

    \b
    Examples:
        capseal diff                              # Diff HEAD~5..HEAD in current repo
        capseal diff ~/projects/CapsuleTech .     # Compare two repos
        capseal diff --ref main..feature          # Compare branches
        capseal diff --watch                      # Watch receipts stream
        capseal diff -v path/to/capsule.json     # Also verify a capsule

    Every operation generates a receipt. View with: capseal logs
    """
    if not MCP_AVAILABLE:
        click.echo("Error: MCP tools not available", err=True)
        raise SystemExit(1)

    if watch:
        _watch_receipts()
        return

    if clear:
        _clear_receipts()
        click.echo("Cleared receipt log")

    # Determine what we're diffing
    workspace = os.environ.get("CAPSEAL_WORKSPACE_ROOT", os.getcwd())

    if source and target:
        # Two paths provided - compare repos
        source_path = os.path.abspath(os.path.expanduser(source))
        target_path = os.path.abspath(os.path.expanduser(target))

        if not os.path.isdir(source_path):
            click.echo(f"Error: {source_path} is not a directory", err=True)
            raise SystemExit(1)
        if not os.path.isdir(target_path):
            click.echo(f"Error: {target_path} is not a directory", err=True)
            raise SystemExit(1)

        # Set up remote and compare
        click.echo(f"Comparing repos:")
        click.echo(f"  Source: {source_path}")
        click.echo(f"  Target: {target_path}")
        click.echo()

        if _setup_remote(target_path, source_path, "capseal_compare"):
            base_ref = "capseal_compare/main"
            head_ref = "HEAD"
            repo_path = target_path
        else:
            click.echo("Error: Could not set up remote for comparison", err=True)
            raise SystemExit(1)

    elif source and not target:
        # One path - diff within that repo
        repo_path = os.path.abspath(os.path.expanduser(source))
        if ".." in ref:
            base_ref, head_ref = ref.split("..", 1)
        else:
            base_ref = f"{ref}~5"
            head_ref = ref
    else:
        # No paths - diff current repo
        repo_path = workspace
        if ".." in ref:
            base_ref, head_ref = ref.split("..", 1)
        else:
            base_ref = f"{ref}~5"
            head_ref = ref

    # Run diff
    click.echo("=" * 60)
    click.echo("  Diff with Receipt Generation")
    click.echo("=" * 60)
    click.echo()
    click.echo(f"[diff_bundle] {base_ref}..{head_ref}")

    result = tool_diff_bundle(repo_path, base_ref, head_ref)

    if not result.get("ok"):
        click.echo(f"  Error: {result.get('stderr', 'unknown error')}", err=True)
        raise SystemExit(1)

    file_count = result.get("file_count", 0)
    files = result.get("files", [])

    click.echo(f"  {file_count} files changed:")
    for f in files[:15]:
        click.echo(f"    • {f}")
    if len(files) > 15:
        click.echo(f"    ... and {len(files) - 15} more")
    click.echo()

    # Optionally verify a capsule
    if verify_capsule:
        capsule_path = os.path.abspath(os.path.expanduser(verify_capsule))
        click.echo(f"[verify] {capsule_path}")
        v_result = tool_verify(capsule_path)
        status = "\033[32m✓ VALID\033[0m" if v_result.get("ok") else "\033[31m✗ INVALID\033[0m"
        click.echo(f"  {status}")
        click.echo()

        click.echo(f"[doctor] Running diagnostics...")
        d_result = tool_doctor(capsule_path, sample_rows=0)
        status = "\033[32m✓ PASS\033[0m" if d_result.get("ok") else "\033[31m✗ FAIL\033[0m"
        click.echo(f"  {status}")
        click.echo()

    # Show receipts
    receipts = _load_receipts()
    click.echo("─" * 60)
    click.echo(f"Receipts ({len(receipts)} generated):")
    for i, r in enumerate(receipts):
        _print_receipt(r, i)

    click.echo("─" * 60)
    click.echo(f"Receipt log: {EVENT_LOG_PATH}")
    click.echo("View live: capseal diff --watch")

    if json_out:
        click.echo()
        click.echo(json.dumps({
            "files": files,
            "file_count": file_count,
            "receipts": receipts,
        }, indent=2))


def _watch_receipts():
    """Watch receipts in real-time."""
    click.echo("Watching receipts... (Ctrl+C to stop)")
    click.echo("─" * 60)

    seen = set()
    try:
        while True:
            receipts = _load_receipts()
            for i, r in enumerate(receipts):
                h = r.get("event_hash", "")
                if h not in seen:
                    seen.add(h)
                    _print_receipt(r, i)
            time.sleep(0.5)
    except KeyboardInterrupt:
        click.echo()
        click.echo(f"Total: {len(seen)} receipts")


@click.command("logs")
@click.option("--follow", "-f", is_flag=True, help="Follow mode (tail -f)")
@click.option("--lines", "-n", default=20, help="Number of lines to show")
@click.option("--json", "json_out", is_flag=True, help="Pretty print JSON")
@click.option("--clear", "-c", is_flag=True, help="Clear the log")
def logs_command(follow: bool, lines: int, json_out: bool, clear: bool):
    """View hash-chained receipt log.

    \b
    Examples:
        capseal logs              # Show last 20 receipts
        capseal logs -f           # Follow mode
        capseal logs --json       # Pretty print
        capseal logs --clear      # Clear log
    """
    if clear:
        _clear_receipts()
        click.echo("Receipt log cleared")
        return

    if follow:
        _watch_receipts()
        return

    receipts = _load_receipts()
    recent = receipts[-lines:] if len(receipts) > lines else receipts

    if not recent:
        click.echo("No receipts yet. Run: capseal diff")
        return

    click.echo(f"Receipts ({len(recent)} of {len(receipts)}):")
    click.echo("─" * 60)

    for i, r in enumerate(recent):
        if json_out:
            tool = r.get("tool", "?")
            ok = "✓" if r.get("result_ok") else "✗"
            ts = time.strftime("%H:%M:%S", time.localtime(r.get("ts_ms", 0) / 1000))
            click.echo(f"{ok} [{ts}] {tool}")
            args = json.dumps(r.get("args", {}))
            if len(args) > 70:
                args = args[:67] + "..."
            click.echo(f"   └─ {args}")
        else:
            _print_receipt(r, i)

    click.echo("─" * 60)
    click.echo(f"Log: {EVENT_LOG_PATH}")
