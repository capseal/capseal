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
@click.option("--verify", is_flag=True,
              help="Run operator readiness checks and exit")
@click.option("--provision", is_flag=True,
              help="Create/update .capseal/operator.json in the workspace and exit")
@click.option("--voice", "enable_voice", is_flag=True,
              help="Enable voice output when used with --provision")
@click.option("--voice-provider", type=click.Choice(["openai", "personaplex"]),
              default="openai", show_default=True,
              help="Voice provider to write during --provision")
@click.option("--live-call", is_flag=True,
              help="Enable live voice call mode when used with --provision")
@click.option("--telegram-chat-id", type=str, default=None,
              help="Telegram chat_id to write during --provision")
@click.option("--telegram-bot-token", type=str, default=None,
              help="Telegram bot token to write during --provision (plaintext)")
@click.option("--notify-threshold", type=float, default=0.5, show_default=True,
              help="notify_threshold value to write during --provision")
@click.option("--speak-gate-decisions", multiple=True,
              type=click.Choice(["approve", "flag", "deny"]),
              help="Gate decisions to speak aloud during --provision")
@click.option("--speak-min-score", type=float, default=0.55, show_default=True,
              help="Minimum significance score for spoken gate events during --provision")
@click.option("--test", is_flag=True,
              help="Send a test notification and exit")
@click.option("--replay", is_flag=True,
              help="Replay all existing events instead of seeking to end")
@click.option("--bg", is_flag=True,
              help="Run in background as a detached process")
def operator_command(workspace: str, config: str, setup: str,
                     verify: bool, provision: bool, enable_voice: bool, voice_provider: str,
                     live_call: bool, telegram_chat_id: str | None, telegram_bot_token: str | None,
                     notify_threshold: float, speak_gate_decisions: tuple[str, ...], speak_min_score: float,
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
        capseal operator . --verify           Check operator readiness
        capseal operator . --provision --voice --live-call
        capseal operator . --test             Send test notification
        capseal operator . --replay           Replay existing events
        capseal operator . --bg               Run in background
    """
    import asyncio
    import sys

    from capseal.operator.daemon import (
        OperatorDaemon,
        interactive_setup_telegram,
        send_test,
    )
    from capseal.operator.config import load_config
    from capseal.operator.ops import (
        provision_operator_config,
        render_verify_report,
        verify_operator_setup,
    )

    ws = Path(workspace).resolve()

    if setup == "telegram":
        asyncio.run(interactive_setup_telegram())
        return

    config_path = Path(config) if config else None

    if provision:
        target = provision_operator_config(
            ws,
            config_path=config_path,
            enable_voice=enable_voice,
            voice_provider=voice_provider,
            live_call=live_call,
            notify_threshold=notify_threshold,
            telegram_chat_id=telegram_chat_id,
            telegram_bot_token=telegram_bot_token,
            use_token_env=telegram_bot_token is None,
            speak_gate_decisions=list(speak_gate_decisions) if speak_gate_decisions else None,
            speak_min_score=speak_min_score,
        )
        click.echo(f"Provisioned operator config: {target}")
        click.echo("Next: run `capseal operator . --verify`")
        if not verify:
            return

    cfg = load_config(config_path, workspace=ws)

    if verify:
        ok, checks = verify_operator_setup(ws, cfg)
        click.echo(render_verify_report(checks))
        if ok:
            click.echo("\noperator verify: ready")
            return
        click.echo("\noperator verify: failed")
        raise SystemExit(1)

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
