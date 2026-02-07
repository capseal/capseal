"""Initialize capseal workspace."""
from __future__ import annotations

import json
import os
from pathlib import Path

import click

from ..contracts import (
    CAPSEAL_VERSION,
    POLICY_SCHEMA_VERSION,
    WorkspaceContract,
)


DEFAULT_POLICY = {
    "schema": f"bef_benchmark_policy_{POLICY_SCHEMA_VERSION}",
    "policy_id": "default",
    "policy_version": "1.0.0",
    "tracks": [
        {
            "track_id": "default",
            "description": "Default track with minimal restrictions.",
            "rules": {
                "forbid_gpu": False,
                "require_deterministic_build": False,
                "required_public_outputs": [],
            },
        }
    ],
}

DEFAULT_CONFIG = {
    "schema": "capseal_config_v1",
    "capseal_version": CAPSEAL_VERSION,
    "workspace_version": "1.0.0",
    "default_policy": "default",
    "sandbox_backend": "auto",
    "network_policy": {
        "allow_fetch": True,
        "allowed_domains": [],
        "blocked_domains": [],
    },
}


@click.command("init")
@click.option(
    "-p", "--path",
    type=click.Path(),
    default=".",
    help="Project root to initialize (default: current directory)",
)
@click.option(
    "--force", "-f",
    is_flag=True,
    help="Overwrite existing workspace",
)
@click.option(
    "--policy",
    type=click.Path(exists=True),
    help="Copy an existing policy file into the workspace",
)
@click.option(
    "--json", "json_output",
    is_flag=True,
    help="Output JSON summary",
)
@click.option(
    "--tui/--no-tui",
    default=True,
    help="Use interactive TUI (default: on if terminal)",
)
def init_command(path: str, force: bool, policy: str | None, json_output: bool, tui: bool) -> None:
    """Initialize a capseal workspace.

    Creates the .capseal/ directory structure with default configuration
    and policy files.

    \b
    Example:
        capseal init
        capseal init -p ./my-project
        capseal init --policy existing_policy.json
    """
    import sys

    # Use TUI if terminal is interactive and no JSON output
    if tui and not json_output and sys.stdin.isatty() and sys.stdout.isatty():
        try:
            from .init_tui import run_init_tui
            run_init_tui(path)
            return
        except ImportError:
            # Fall through to non-TUI mode if dependencies missing
            click.echo("Note: TUI dependencies not installed. Using basic mode.")
            click.echo("Install with: pip install rich questionary")
        except KeyboardInterrupt:
            click.echo("\nAborted.")
            raise SystemExit(1)

    root = Path(path).resolve()
    workspace = root / WorkspaceContract.PROJECT_WORKSPACE

    # Check if workspace exists
    if workspace.exists() and not force:
        if json_output:
            click.echo(json.dumps({
                "status": "error",
                "message": f"Workspace already exists at {workspace}",
                "hint": "Use --force to overwrite",
            }))
        else:
            click.echo(f"Error: Workspace already exists at {workspace}")
            click.echo("Use --force to overwrite")
        raise SystemExit(1)

    # Create directory structure
    dirs_created = []
    for subdir in [
        WorkspaceContract.RUNS_DIR,
        WorkspaceContract.DATASETS_DIR,
        WorkspaceContract.POLICIES_DIR,
        WorkspaceContract.RECEIPTS_DIR,
    ]:
        dir_path = workspace / subdir
        dir_path.mkdir(parents=True, exist_ok=True)
        dirs_created.append(str(dir_path.relative_to(root)))

    # Write config file
    config_path = workspace / WorkspaceContract.CONFIG_FILE
    with open(config_path, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)

    # Write default policy or copy provided one
    policy_dir = workspace / WorkspaceContract.POLICIES_DIR
    if policy:
        # Copy existing policy
        import shutil
        policy_name = Path(policy).name
        shutil.copy(policy, policy_dir / policy_name)
        policy_created = str((policy_dir / policy_name).relative_to(root))
    else:
        # Write default policy
        default_policy_path = policy_dir / "default.json"
        with open(default_policy_path, "w") as f:
            json.dump(DEFAULT_POLICY, f, indent=2)
        policy_created = str(default_policy_path.relative_to(root))

    # Initialize empty events log
    events_path = workspace / WorkspaceContract.EVENTS_LOG
    events_path.touch()

    # Output result
    result = {
        "status": "initialized",
        "workspace": str(workspace.relative_to(root)),
        "config": str(config_path.relative_to(root)),
        "policy": policy_created,
        "directories": dirs_created,
        "capseal_version": CAPSEAL_VERSION,
    }

    if json_output:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Initialized capseal workspace at {workspace}")
        click.echo(f"  Config: {result['config']}")
        click.echo(f"  Policy: {result['policy']}")
        click.echo(f"  Directories:")
        for d in dirs_created:
            click.echo(f"    - {d}")
        click.echo()
        click.echo("Next steps:")
        click.echo("  1. Edit .capseal/policies/default.json to customize your policy")
        click.echo("  2. Run 'capseal fetch -d <url>' to fetch datasets")
        click.echo("  3. Run 'capseal run -- <command>' to generate a receipt")
        click.echo("  4. Run 'capseal doctor' to verify your environment")


__all__ = ["init_command"]
