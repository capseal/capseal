"""Fetch command - governed remote dataset acquisition."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import subprocess
import json
import os

from .exit_codes import EXIT_MALFORMED, EXIT_PROOF_INVALID
from .redact import redact_secrets
from .run import _find_project_root
from bef_zk.sandbox import SandboxRunner, SandboxConfig, is_sandbox_available


@click.command("fetch")
@click.option("--url", required=True, help="Remote URL to download")
@click.option("--dataset-id", "dataset_id", type=str, help="Dataset identifier")
@click.option("--output", "-o", type=click.Path(path_type=Path), help="Output directory")
@click.option("--policy", "-p", type=click.Path(exists=True, path_type=Path), required=True, help="Policy file")
@click.option("--policy-id", required=True, help="Policy identifier")
@click.option("--policy-version", default=None, help="Policy version label")
@click.option("--sandbox/--no-sandbox", default=False, help="Run fetch inside sandbox")
@click.option("--sandbox-memory", type=int, default=2048, help="Sandbox memory limit (MB)")
@click.option("--sandbox-timeout", type=int, default=300, help="Sandbox wall clock timeout (s)")
@click.option("--sandbox-allow-network/--sandbox-deny-network", default=True, help="Allow sandbox network (fetch requires network)")
@click.option("--dataset-tree-arity", type=int, default=16, help="Dataset Merkle arity for manifests")
def fetch_command(
    url: str,
    dataset_id: Optional[str],
    output: Optional[Path],
    policy: Path,
    policy_id: str,
    policy_version: Optional[str],
    sandbox: bool,
    sandbox_memory: int,
    sandbox_timeout: int,
    sandbox_allow_network: bool,
    dataset_tree_arity: int,
) -> None:
    """Fetch a remote dataset under governed network policy."""

    if not sandbox_allow_network and sandbox:
        click.echo("Error: fetch requires network access inside sandbox. Use --sandbox-allow-network.", err=True)
        raise SystemExit(EXIT_MALFORMED)

    root = _find_project_root()
    script = root / "scripts" / "fetch_dataset.py"
    if not script.exists():
        click.echo(f"Error: fetch script missing at {script}", err=True)
        raise SystemExit(EXIT_MALFORMED)

    cmd = [
        "python3",
        str(script),
        "--url",
        url,
        "--policy",
        str(policy.resolve()),
        "--policy-id",
        policy_id,
        "--dataset-tree-arity",
        str(dataset_tree_arity),
    ]
    if dataset_id:
        cmd.extend(["--dataset-id", dataset_id])
    if output:
        cmd.extend(["--output-dir", str(output.resolve())])
    if policy_version:
        cmd.extend(["--policy-version", policy_version])

    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)

    try:
        if sandbox:
            if not is_sandbox_available():
                click.echo("Warning: sandbox backend unavailable; running unsandboxed.", err=True)
                sandbox = False
            else:
                config = SandboxConfig(
                    datasets=[],
                    output_dir=(output or (root / "out")),
                    policy_path=policy.resolve(),
                    memory_mb=sandbox_memory,
                    wall_time_sec=sandbox_timeout,
                    network=True,
                    capseal_root=root,
                )
                runner = SandboxRunner(config)
                result = runner.run(cmd, env=env)
                sandbox_isolation = result.resource_usage.get("isolation", {})
                if result.returncode != 0:
                    click.echo(redact_secrets(result.stderr), err=True)
                    raise SystemExit(EXIT_PROOF_INVALID)
                try:
                    summary = json.loads(result.stdout)
                except json.JSONDecodeError:
                    click.echo(redact_secrets(result.stdout), err=True)
                    raise SystemExit(EXIT_PROOF_INVALID)
                _print_fetch_summary(summary, sandbox_isolation)
                return
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root), env=env)
        if proc.returncode != 0:
            click.echo(redact_secrets(proc.stderr or proc.stdout), err=True)
            raise SystemExit(EXIT_PROOF_INVALID)
        summary = json.loads(proc.stdout)
        _print_fetch_summary(summary, None)
    except json.JSONDecodeError:
        click.echo("Fetch failed: invalid output", err=True)
        raise SystemExit(EXIT_PROOF_INVALID)


def _print_fetch_summary(summary: dict[str, any], isolation: dict[str, any] | None) -> None:
    click.echo("\nDataset fetch completed")
    click.echo(f"  Dataset ID: {summary.get('dataset_id')}")
    click.echo(f"  Dataset root: {summary.get('dataset_root')}")
    click.echo(f"  Materialized at: {summary.get('materialized_path')}")
    click.echo(f"  Receipt: {summary.get('fetch_receipt')}")
