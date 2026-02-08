"""PTY Shell — full terminal with persistent CapSeal status bar.

Launches the user's shell inside a PTY, paints a status bar on row 1,
and detects agent launches to auto-inject MCP config.

Uses stdlib pty.fork() + select.select() — no external dependencies.
"""
from __future__ import annotations

import fcntl
import json
import os
import select
import signal
import struct
import sys
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Any

import click


class _StatusState:
    """Mutable status bar state shared between threads."""

    __slots__ = (
        "workspace", "action_count", "chain_ok", "risk",
        "latest_event", "event_time",
    )

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.action_count = 0
        self.chain_ok = True
        self.risk: float | None = None
        self.latest_event = ""
        self.event_time = 0.0


class PtyShell:
    """PTY-based shell with CapSeal status bar."""

    def __init__(self, workspace: Path, config: dict) -> None:
        self.workspace = workspace.resolve()
        self.config = config
        self.status = _StatusState(self.workspace)
        self._master_fd: int = -1
        self._child_pid: int = -1
        self._old_termios: list | None = None
        self._old_winsize: bytes | None = None
        self._rows: int = 24
        self._cols: int = 80
        self._input_buf = bytearray()  # Circular buffer for agent detection
        self._running = False
        self._event_thread: threading.Thread | None = None

    def run(self) -> None:
        """Main entry: fork shell, set up scroll region, proxy I/O."""
        if not os.isatty(0):
            print("capseal shell: requires an interactive terminal", file=sys.stderr)
            sys.exit(1)

        # Save original terminal state
        self._old_termios = termios.tcgetattr(sys.stdin)
        self._old_winsize = fcntl.ioctl(
            sys.stdout, termios.TIOCGWINSZ, b"\x00" * 8
        )
        rows, cols = struct.unpack("HHHH", self._old_winsize)[:2]
        self._rows = rows
        self._cols = cols

        # Load initial status from workspace
        self._load_workspace_status()

        # Fork the PTY
        pid, master_fd = os.forkpty()
        if pid == 0:
            # Child process — exec the user's shell
            shell = os.environ.get("SHELL", "/bin/bash")
            env = os.environ.copy()
            env["CAPSEAL_ACTIVE"] = "1"
            env["CAPSEAL_WORKSPACE"] = str(self.workspace)
            os.execve(shell, [shell], env)
            # Should not reach here
            os._exit(1)

        # Parent process
        self._child_pid = pid
        self._master_fd = master_fd
        self._running = True

        # Set child PTY size
        self._set_child_winsize()

        # Set up signal handlers
        old_sigwinch = signal.signal(signal.SIGWINCH, self._handle_sigwinch)
        old_sigint = signal.signal(signal.SIGINT, self._handle_sigint)
        old_sigterm = signal.signal(signal.SIGTERM, self._handle_sigterm)

        # Start event watcher thread
        self._start_event_watcher()

        try:
            # Set stdin to raw mode
            tty.setraw(sys.stdin)

            # Set up scroll region and paint status bar
            self._setup_scroll_region()
            self._paint_status_bar()

            # Main I/O proxy loop
            self._proxy_io()
        finally:
            self._running = False

            # Restore terminal
            self._restore_terminal()

            # Restore signal handlers
            signal.signal(signal.SIGWINCH, old_sigwinch)
            signal.signal(signal.SIGINT, old_sigint)
            signal.signal(signal.SIGTERM, old_sigterm)

            # Wait for child to exit
            try:
                os.waitpid(self._child_pid, 0)
            except ChildProcessError:
                pass

    def _setup_scroll_region(self) -> None:
        """Reserve row 1 for the status bar using ANSI scroll region."""
        # Set scroll region to rows 2..max
        os.write(sys.stdout.fileno(), f"\033[2;{self._rows}r".encode())
        # Move cursor to row 2
        os.write(sys.stdout.fileno(), b"\033[2;1H")

    def _paint_status_bar(self) -> None:
        """Render status bar on row 1 (reverse video)."""
        s = self.status

        # Build status text
        parts = [
            "CAPSEAL",
            f"~/{self.workspace.name}",
            f"{s.action_count} actions",
            f"chain: {'intact' if s.chain_ok else 'BROKEN'}",
        ]
        if s.risk is not None:
            parts.append(f"risk: {s.risk:.2f}")

        # Add latest event if recent (< 5 seconds)
        if s.latest_event and (time.time() - s.event_time) < 5.0:
            parts.append(s.latest_event)

        bar_text = " \u2500 ".join(parts)

        # Pad/truncate to terminal width
        bar_text = bar_text[: self._cols - 4]
        display = f"\u256d\u2500 {bar_text} "
        display += "\u2500" * max(0, self._cols - len(display) - 1)
        display += "\u256e"

        # Save cursor, move to row 1, paint in reverse video, restore cursor
        out = b"\0337"  # Save cursor
        out += b"\033[1;1H"  # Move to row 1, col 1
        out += b"\033[7m"  # Reverse video
        out += display.encode("utf-8", errors="replace")
        out += b"\033[0m"  # Reset attributes
        out += b"\0338"  # Restore cursor
        os.write(sys.stdout.fileno(), out)

    def _handle_sigwinch(self, signum: int, frame: Any) -> None:
        """Terminal resize: update child PTY and repaint scroll region."""
        try:
            self._old_winsize = fcntl.ioctl(
                sys.stdout, termios.TIOCGWINSZ, b"\x00" * 8
            )
            rows, cols = struct.unpack("HHHH", self._old_winsize)[:2]
            self._rows = rows
            self._cols = cols
            self._set_child_winsize()
            self._setup_scroll_region()
            self._paint_status_bar()
        except OSError:
            pass

    def _handle_sigint(self, signum: int, frame: Any) -> None:
        """Forward SIGINT to child."""
        try:
            os.kill(self._child_pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        """Forward SIGTERM to child, then exit."""
        try:
            os.kill(self._child_pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self._running = False

    def _set_child_winsize(self) -> None:
        """Set the child PTY window size (rows-1 for status bar)."""
        child_rows = max(1, self._rows - 1)
        winsize = struct.pack("HHHH", child_rows, self._cols, 0, 0)
        try:
            fcntl.ioctl(self._master_fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

    def _detect_agent_launch(self, data: bytes) -> str | None:
        """Scan input buffer for agent binary names.

        Returns agent key if detected, None otherwise.
        """
        self._input_buf.extend(data)
        # Keep last 256 bytes
        if len(self._input_buf) > 256:
            self._input_buf = self._input_buf[-256:]

        buf_str = self._input_buf.decode("utf-8", errors="ignore")
        for trigger, agent_key in [
            ("claude", "claude-code"),
            ("codex", "codex"),
            ("gemini", "gemini-cli"),
        ]:
            # Check if buffer ends with trigger followed by space/newline/CR
            for sep in (" ", "\n", "\r"):
                pattern = trigger + sep
                if pattern in buf_str:
                    # Clear buffer to avoid re-triggering
                    self._input_buf.clear()
                    return agent_key
        return None

    def _inject_mcp_config(self, agent_key: str) -> None:
        """Write .mcp.json before agent runs."""
        try:
            from .hub import _write_mcp_json, _write_claude_instructions, AGENTS
            agent_info = AGENTS.get(agent_key, {})
            if agent_info.get("mcp_support"):
                _write_mcp_json(self.workspace)
            if agent_key == "claude-code":
                _write_claude_instructions(self.workspace)
        except Exception:
            pass

    def _start_event_watcher(self) -> None:
        """Background thread tailing .capseal/events.jsonl."""
        events_path = self.workspace / ".capseal" / "events.jsonl"

        def _watch() -> None:
            # Wait for events file to appear
            while self._running and not events_path.exists():
                time.sleep(1.0)
            if not self._running:
                return

            try:
                with open(events_path) as f:
                    # Seek to end
                    f.seek(0, 2)
                    while self._running:
                        line = f.readline()
                        if line:
                            try:
                                event = json.loads(line.strip())
                                self.status.latest_event = event.get("summary", "")[:40]
                                self.status.event_time = time.time()

                                # Update action count on record events
                                if event.get("type") == "record":
                                    self.status.action_count += 1

                                self._paint_status_bar()
                            except (json.JSONDecodeError, KeyError):
                                pass
                        else:
                            time.sleep(0.3)
            except OSError:
                pass

        self._event_thread = threading.Thread(target=_watch, daemon=True)
        self._event_thread.start()

    def _proxy_io(self) -> None:
        """select() loop: proxy stdin ↔ master_fd."""
        stdin_fd = sys.stdin.fileno()
        stdout_fd = sys.stdout.fileno()

        while self._running:
            try:
                rlist, _, _ = select.select(
                    [stdin_fd, self._master_fd], [], [], 1.0
                )
            except (OSError, ValueError):
                break

            if stdin_fd in rlist:
                try:
                    data = os.read(stdin_fd, 4096)
                except OSError:
                    break
                if not data:
                    break

                # Agent detection
                agent = self._detect_agent_launch(data)
                if agent:
                    self._inject_mcp_config(agent)

                # Forward to child
                try:
                    os.write(self._master_fd, data)
                except OSError:
                    break

            if self._master_fd in rlist:
                try:
                    data = os.read(self._master_fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                # Forward to stdout
                try:
                    os.write(stdout_fd, data)
                except OSError:
                    break

            # Clear stale events from status bar
            if (
                self.status.latest_event
                and (time.time() - self.status.event_time) > 5.0
            ):
                self.status.latest_event = ""
                self._paint_status_bar()

    def _restore_terminal(self) -> None:
        """Reset scroll region and restore terminal state."""
        try:
            # Reset scroll region
            os.write(sys.stdout.fileno(), b"\033[r")
            # Clear screen and move to top
            os.write(sys.stdout.fileno(), b"\033[H\033[J")
        except OSError:
            pass

        if self._old_termios is not None:
            try:
                termios.tcsetattr(
                    sys.stdin, termios.TCSADRAIN, self._old_termios
                )
            except termios.error:
                pass

    def _load_workspace_status(self) -> None:
        """Load current action count and chain status from workspace."""
        runs_dir = self.workspace / ".capseal" / "runs"
        if not runs_dir.exists():
            return

        # Count actions in latest session
        latest = runs_dir / "latest"
        if latest.exists():
            actions_file = latest / "actions.jsonl" if latest.is_dir() else None
            if latest.is_symlink():
                target = runs_dir / os.readlink(latest)
                if target.is_dir():
                    actions_file = target / "actions.jsonl"

            if actions_file and actions_file.exists():
                try:
                    count = sum(
                        1 for line in actions_file.read_text().strip().split("\n")
                        if line.strip()
                    )
                    self.status.action_count = count
                except OSError:
                    pass


def run_pty_shell(workspace: Path, config: dict) -> None:
    """Callable entry point for launching the PTY shell."""
    shell = PtyShell(workspace, config)
    shell.run()


@click.command("shell")
@click.option(
    "--workspace", "-w", default=".",
    type=click.Path(exists=True, file_okay=False),
    help="Project directory containing .capseal/",
)
@click.option(
    "--dry-run", is_flag=True,
    help="Print config summary without launching PTY",
)
def shell_command(workspace: str, dry_run: bool) -> None:
    """Launch a PTY shell with live CapSeal status bar.

    The status bar shows workspace name, action count, chain integrity,
    and risk score. Agent launches (claude, codex, gemini) are auto-detected
    and MCP config is injected before the agent starts.

    \b
    Examples:
        capseal shell                  Launch in current directory
        capseal shell -w ~/project     Launch for a specific project
        capseal shell --dry-run        Show config without launching
    """
    ws = Path(workspace).resolve()
    capseal_dir = ws / ".capseal"

    # Load config
    config: dict = {}
    config_path = capseal_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    if dry_run:
        click.echo("CapSeal PTY Shell — dry run")
        click.echo(f"  Workspace:  {ws}")
        click.echo(f"  Config:     {config_path}")
        click.echo(f"  Provider:   {config.get('provider', 'not configured')}")
        click.echo(f"  Model:      {config.get('model', 'not configured')}")
        click.echo(f"  .capseal/:  {'exists' if capseal_dir.exists() else 'not found'}")

        posteriors = capseal_dir / "models" / "beta_posteriors.npz"
        click.echo(f"  Risk model: {'trained' if posteriors.exists() else 'not trained'}")
        return

    if not os.isatty(0):
        click.echo("Error: capseal shell requires an interactive terminal", err=True)
        sys.exit(1)

    run_pty_shell(ws, config)
