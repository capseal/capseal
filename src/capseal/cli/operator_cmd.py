"""capseal operator -- Launch the CapSeal Operator Agent daemon.

Real-time notifications for autonomous AI agent sessions.
Watches events.jsonl, scores significance, sends to Telegram.
"""
from __future__ import annotations

import click
from pathlib import Path


@click.command("operator")
@click.argument("workspace", type=click.Path(exists=True), default=".")
@click.option("--config", "-c", type=click.Path(), default=None,
              help="Config file path (default: ~/.capseal/operator.json)")
@click.option("--setup", type=click.Choice(["telegram"]),
              help="Interactive channel setup")
@click.option("--test", is_flag=True,
              help="Send a test notification and exit")
@click.option("--replay", is_flag=True,
              help="Replay all existing events instead of seeking to end")
@click.option("--bg", is_flag=True,
              help="Run in background as a detached process")
def operator_command(workspace: str, config: str, setup: str,
                     test: bool, replay: bool, bg: bool) -> None:
    """Launch the CapSeal Operator Agent.

    Watches events.jsonl from CapSeal sessions and sends real-time
    notifications through configured channels (Telegram, etc.).

    \b
    Setup:
        capseal operator --setup telegram

    \b
    Usage:
        capseal operator .                    Monitor current directory
        capseal operator /path/to/project     Monitor specific project
        capseal operator . --test             Send test notification
        capseal operator . --replay           Replay existing events
        capseal operator . --bg               Run in background
    """
    import asyncio
    import os
    import sys

    from capseal.operator.daemon import (
        OperatorDaemon,
        interactive_setup_telegram,
        send_test,
    )
    from capseal.operator.config import load_config

    ws = Path(workspace).resolve()

    if setup == "telegram":
        asyncio.run(interactive_setup_telegram())
        return

    config_path = Path(config) if config else None
    cfg = load_config(config_path, workspace=ws)

    if test:
        asyncio.run(send_test(ws, cfg))
        return

    if bg:
        import subprocess
        args = [sys.executable, "-m", "capseal.operator",
                "--workspace", str(ws)]
        if config:
            args.extend(["--config", config])
        if replay:
            args.append("--replay")
        proc = subprocess.Popen(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        click.echo(f"Operator daemon started in background (PID: {proc.pid})")
        return

    # Foreground
    daemon = OperatorDaemon(ws, cfg)
    if replay:
        daemon.replay = True

    import signal
    loop = asyncio.new_event_loop()

    try:
        loop.add_signal_handler(signal.SIGINT, lambda: setattr(daemon, 'running', False))
        loop.add_signal_handler(signal.SIGTERM, lambda: setattr(daemon, 'running', False))
    except NotImplementedError:
        pass  # Windows doesn't support signal handlers on event loops

    try:
        loop.run_until_complete(daemon.run())
    except KeyboardInterrupt:
        pass
    finally:
        click.echo("\nOperator shutting down.")
        loop.close()
