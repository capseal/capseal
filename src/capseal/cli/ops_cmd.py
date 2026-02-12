"""`capseal ops` RunPod lifecycle controls for voice operator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from capseal.operator.runpod_ops import (
    clear_pod_id_in_config,
    get_pod_status,
    load_operator_config,
    operator_config_paths,
    resolve_runpod_api_key,
    resolve_runpod_pod_id,
    resume_pod,
    stop_pod,
    terminate_pod,
)


def _runtime(workspace: Path) -> tuple[Path, dict[str, Any], str | None, str | None]:
    config_path, cfg = load_operator_config(workspace)
    pod_id = resolve_runpod_pod_id(cfg)
    api_key = resolve_runpod_api_key(cfg)
    return config_path, cfg, pod_id, api_key


@click.group("ops")
@click.option(
    "--workspace",
    "-w",
    default=".",
    type=click.Path(exists=True, file_okay=False),
    show_default=True,
    help="Workspace containing .capseal/operator.json",
)
@click.pass_context
def ops_group(ctx: click.Context, workspace: str) -> None:
    """Manage the PersonaPlex RunPod used by the operator voice channel."""
    ctx.ensure_object(dict)
    ctx.obj["workspace"] = Path(workspace).resolve()


@ops_group.command("status")
@click.option("--json", "as_json", is_flag=True, help="Print JSON status")
@click.pass_context
def ops_status(ctx: click.Context, as_json: bool) -> None:
    workspace: Path = ctx.obj["workspace"]
    cfg_path, _, pod_id, api_key = _runtime(workspace)
    if not pod_id:
        raise SystemExit(f"No runpod_pod_id configured in {cfg_path}")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY not set (env or config)")
    status = get_pod_status(api_key, pod_id)
    if as_json:
        click.echo(json.dumps(status, indent=2))
        return
    click.echo(f"pod_id: {status['pod_id']}")
    click.echo(f"status: {status['status']}")
    click.echo(f"network_ready: {status['network_ready']}")


@ops_group.command("start")
@click.pass_context
def ops_start(ctx: click.Context) -> None:
    workspace: Path = ctx.obj["workspace"]
    cfg_path, _, pod_id, api_key = _runtime(workspace)
    if not pod_id:
        raise SystemExit(f"No runpod_pod_id configured in {cfg_path}")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY not set (env or config)")
    resume_pod(api_key, pod_id)
    click.echo(f"resume requested for pod {pod_id}")


@ops_group.command("stop")
@click.pass_context
def ops_stop(ctx: click.Context) -> None:
    workspace: Path = ctx.obj["workspace"]
    cfg_path, _, pod_id, api_key = _runtime(workspace)
    if not pod_id:
        raise SystemExit(f"No runpod_pod_id configured in {cfg_path}")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY not set (env or config)")
    stop_pod(api_key, pod_id)
    click.echo(f"stop requested for pod {pod_id}")


@ops_group.command("teardown")
@click.option("--yes", is_flag=True, help="Skip confirmation")
@click.pass_context
def ops_teardown(ctx: click.Context, yes: bool) -> None:
    workspace: Path = ctx.obj["workspace"]
    cfg_path, _, pod_id, api_key = _runtime(workspace)
    if not pod_id:
        raise SystemExit(f"No runpod_pod_id configured in {cfg_path}")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY not set (env or config)")

    if not yes:
        confirmed = click.confirm(f"Terminate pod {pod_id} and remove saved pod id?")
        if not confirmed:
            click.echo("aborted")
            return

    terminate_pod(api_key, pod_id)
    for path in operator_config_paths(workspace):
        if path.exists():
            try:
                clear_pod_id_in_config(path)
            except OSError:
                # Best-effort cleanup; don't fail teardown if a secondary config
                # path is not writable in the current execution environment.
                pass
    click.echo(f"terminated pod {pod_id} and cleared saved pod id")
