"""CapSeal Hub — interactive entry point when `capseal` is invoked with no args."""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

from .tui_compat import suppress_cpr
suppress_cpr()

import questionary
from rich.panel import Panel
from rich.table import Table
from rich.box import ROUNDED

from .init_tui import (
    CAPSEAL_STYLE,
    BANNER,
    console,
    print_success,
    print_warn,
    print_box,
)
from .autopilot_cmd import _detect_provider, _auto_init, _capseal_bin

# ── Agent registry ───────────────────────────────────────────────────────────

AGENTS = {
    "claude-code": {
        "name": "Claude Code",
        "binary": "claude",
        "mcp_support": True,
    },
    "codex": {
        "name": "Codex",
        "binary": "codex",
        "mcp_support": True,
    },
    "gemini-cli": {
        "name": "Gemini CLI",
        "binary": "gemini",
        "mcp_support": False,
    },
    "cursor": {
        "name": "Cursor",
        "binary": "cursor",
        "mcp_support": True,
    },
    "terminal": {
        "name": "Terminal (subshell)",
        "binary": None,
        "mcp_support": False,
    },
}

AGENT_DISPLAY = {k: v["name"] for k, v in AGENTS.items()}

# Gating protocol instructions for Claude Code (shared with init_tui.py)
CAPSEAL_INSTRUCTIONS = """## CapSeal Required — MANDATORY

This project uses CapSeal for verified code changes. You MUST follow this protocol:

### At session start or after any conversation summary:
Call `capseal_status` to check session state. This recovers your place if context was compacted.

### Before EACH file change (one gate per file):
Call `capseal_gate` with the SPECIFIC FILE you're about to change. Gate each file INDIVIDUALLY — do not batch multiple files into one gate call.

### Gate decisions are BINDING:
- **"approve"**: You may proceed with the change
- **"deny"**: **DO NOT make the change. Period.** Tell the user: "CapSeal blocked this change (predicted failure rate: X%). I cannot proceed."
- **"flag"**: Ask the user for explicit permission before proceeding. Do not assume approval.

### After each APPROVED change (one record per file):
Call `capseal_record` for EACH file you changed — one record per file, not one summary for all changes. Include what you changed and whether it succeeded.

### When ALL changes are complete:
Call `capseal_seal` to create the cryptographic receipt. Do not skip this step.

Gate decisions from CapSeal are not suggestions. They are requirements.
"""


# ── Config helpers ───────────────────────────────────────────────────────────

def _load_config(target: Path) -> dict:
    """Safely load .capseal/config.json."""
    config_path = target / ".capseal" / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_config(target: Path, config: dict) -> None:
    """Write config back to .capseal/config.json."""
    config_path = target / ".capseal" / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")


# ── Workspace bootstrap ─────────────────────────────────────────────────────

def _ensure_workspace(target: Path) -> dict:
    """Ensure .capseal/ exists. Auto-init on first visit. Returns config."""
    config = _load_config(target)
    if config:
        return config

    # First visit — auto-detect provider and create workspace silently
    capseal_dir = target / ".capseal"
    detected = _detect_provider()

    if detected:
        provider, env_var, model = detected
        _auto_init(target, provider, env_var, model)
        console.print(f"  [green]✓[/green] Workspace initialized (auto-detected {provider})")
    else:
        # No env var found — ask for provider, then auth method
        console.print()
        console.print("  [yellow]No API key detected.[/yellow]")

        provider_choice = questionary.select(
            "Which LLM provider do you use?",
            choices=[
                questionary.Choice("Anthropic (Claude)", value="anthropic"),
                questionary.Choice("OpenAI", value="openai"),
                questionary.Choice("Google (Gemini)", value="google"),
            ],
            style=CAPSEAL_STYLE,
        ).ask()
        if not provider_choice:
            return {}

        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
        }
        model_map = {
            "anthropic": "claude-opus-4-6",
            "openai": "chatgpt-5.2",
            "google": "gemini-3-flash",
        }
        sub_names = {
            "anthropic": "Claude subscription (Pro/Max)",
            "openai": "Codex subscription",
            "google": "Gemini subscription",
        }
        key_names = {
            "anthropic": "Anthropic API key",
            "openai": "OpenAI API key",
            "google": "Google API key",
        }

        auth_choice = questionary.select(
            "How do you authenticate?",
            choices=[
                questionary.Choice(f"{sub_names[provider_choice]} — no API key needed", value="subscription"),
                questionary.Choice(key_names[provider_choice], value="api_key"),
            ],
            style=CAPSEAL_STYLE,
        ).ask()
        if not auth_choice:
            return {}

        env_var = env_map[provider_choice]

        if auth_choice == "subscription":
            _auto_init(target, provider_choice, env_var, model_map[provider_choice])
            config = _load_config(target)
            config["auth_method"] = "subscription"
            config["plan"] = "subscription"
            _save_config(target, config)
            console.print(f"  [green]✓[/green] Workspace initialized ({sub_names[provider_choice]})")
        else:
            api_key = questionary.password(
                f"Enter {env_var}:",
                style=CAPSEAL_STYLE,
            ).ask()
            if api_key:
                os.environ[env_var] = api_key
                capseal_dir.mkdir(parents=True, exist_ok=True)
                env_path = capseal_dir / ".env"
                env_path.write_text(f"{env_var}={api_key}\n")
                env_path.chmod(0o600)

            _auto_init(target, provider_choice, env_var, model_map[provider_choice])
            console.print(f"  [green]✓[/green] Workspace initialized ({provider_choice})")

    return _load_config(target)


# ── Status banner ────────────────────────────────────────────────────────────

def _cap_session_key(cap_path: Path) -> str:
    """Return canonical session key for a .cap file.

    Session identity is based on run-id / file stem, not manifest session_name.
    This keeps counts stable when session_name is customized.
    """
    return cap_path.stem


def _show_status_banner(target: Path, config: dict) -> None:
    """Print the hub header with workspace status."""
    console.print(BANNER)

    provider = config.get("provider", "unknown")
    model_id = config.get("model", "")
    default_agent = config.get("default_agent", "")

    # Friendly model names
    MODEL_NAMES = {
        "claude-opus-4-6": "Claude Opus 4.6",
        "claude-sonnet-4-20250514": "Claude Sonnet 4",
        "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
        "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
        "chatgpt-5.2": "ChatGPT 5.2",
        "gpt-4o-mini": "GPT-4o Mini",
        "gemini-3-flash": "Gemini 3 Flash",
    }
    model = MODEL_NAMES.get(model_id, model_id)

    # Risk model status
    posteriors = target / ".capseal" / "models" / "beta_posteriors.npz"
    if posteriors.exists():
        try:
            import numpy as np
            data = np.load(posteriors)
            n_episodes = int(data["n_episodes"]) if "n_episodes" in data else "?"
            model_status = f"[green]✓ trained ({n_episodes} episodes)[/green]"
        except Exception:
            model_status = "[green]✓ trained[/green]"
    else:
        model_status = "[dim]✗ not trained yet[/dim] [dim](select 'Train risk model' below)[/dim]"

    # Session count (align with TUI semantics):
    # count unique run sessions and show how many are sealed.
    runs_dir = target / ".capseal" / "runs"
    sessions: dict[str, dict[str, bool]] = {}
    if runs_dir.exists():
        for entry in runs_dir.iterdir():
            if entry.name in ("latest", "latest.cap"):
                continue
            if entry.is_dir():
                sessions.setdefault(entry.name, {"sealed": False})
            elif entry.suffix == ".cap" and not entry.is_symlink():
                key = _cap_session_key(entry)
                session = sessions.setdefault(key, {"sealed": False})
                session["sealed"] = True

    session_count = len(sessions)
    sealed_count = sum(1 for s in sessions.values() if s["sealed"])
    last_session = ""
    if sessions:
        newest_name = max(sessions.keys())
        try:
            ts_str = newest_name.split("-")[0]
            dt = datetime.strptime(ts_str, "%Y%m%dT%H%M%S")
            delta = datetime.now() - dt
            if delta.days > 0:
                last_session = f"{delta.days}d ago"
            elif delta.seconds > 3600:
                last_session = f"{delta.seconds // 3600}h ago"
            else:
                last_session = f"{delta.seconds // 60}m ago"
        except (ValueError, IndexError):
            pass

    # Build status lines
    lines = [
        f"  Workspace:  {target}",
        f"  Provider:   {provider} ({model})" if model else f"  Provider:   {provider}",
    ]
    if default_agent:
        agent_name = AGENT_DISPLAY.get(default_agent, default_agent)
        lines.append(f"  Agent:      {agent_name}")
    lines.append(f"  Risk model: {model_status}")
    if session_count > 0:
        sess_str = f"  Sessions:   {session_count} ({sealed_count} sealed)"
        if last_session:
            sess_str += f", last: {last_session}"
        lines.append(sess_str)

    status_text = "\n".join(lines)
    console.print(Panel(
        status_text,
        border_style="cyan",
        padding=(0, 1),
        expand=False,
    ))


# ── Hub menu ─────────────────────────────────────────────────────────────────

def _hub_menu(target: Path, config: dict) -> None:
    """Main menu loop."""
    while True:
        console.print()

        # Check if risk model is trained
        model_path = target / ".capseal" / "models" / "beta_posteriors.npz"
        model_trained = model_path.exists()

        # Build choices dynamically
        choices = [
            questionary.Choice("Start a coding session", value="session"),
            questionary.Choice("Scan & fix this codebase", value="autopilot"),
        ]
        if not model_trained:
            choices.append(questionary.Choice("Train risk model", value="train"))
        choices.extend([
            questionary.Choice("View past sessions", value="sessions"),
            questionary.Choice("View risk report", value="report"),
            questionary.Choice("Run self-test", value="selftest"),
            questionary.Choice("CapSeal shell", value="shell"),
            questionary.Choice("Configure", value="configure"),
            questionary.Choice("Exit", value="exit"),
        ])

        choice = questionary.select(
            "What would you like to do?",
            choices=choices,
            style=CAPSEAL_STYLE,
        ).ask()

        if choice is None or choice == "exit":
            console.print("[dim]Goodbye.[/dim]")
            break
        elif choice == "session":
            _start_session(target, config)
        elif choice == "autopilot":
            _run_autopilot(target)
        elif choice == "train":
            _train_model(target, config)
            # Refresh banner to show updated status
            _show_status_banner(target, config)
        elif choice == "sessions":
            _view_sessions(target)
        elif choice == "report":
            _run_report(target)
        elif choice == "selftest":
            _run_selftest(target)
        elif choice == "shell":
            _run_shell_repl(target, config)
        elif choice == "configure":
            _run_configure(target)
            # Reload config after configure
            config = _load_config(target)


# ── Agent session ────────────────────────────────────────────────────────────

def _resolve_agent(config: dict) -> str | None:
    """Get agent to launch. Uses saved default or prompts for selection.

    Returns agent key (e.g. 'claude-code') or None if cancelled.
    """
    default_agent = config.get("default_agent", "")

    # Check saved default
    if default_agent and default_agent in AGENTS:
        agent_info = AGENTS[default_agent]
        binary = agent_info["binary"]
        # Terminal is always available
        if binary is None or shutil.which(binary):
            console.print(f"  Agent: [cyan]{agent_info['name']}[/cyan]  [dim](change: capseal configure)[/dim]")
            return default_agent
        else:
            console.print(f"  [yellow]{agent_info['name']} is no longer in PATH. Select another agent.[/yellow]")

    # Build choices from installed agents
    choices = []
    for key, info in AGENTS.items():
        binary = info["binary"]
        if binary is None:
            # Terminal — always available
            choices.append(questionary.Choice(info["name"], value=key))
        elif shutil.which(binary):
            choices.append(questionary.Choice(info["name"], value=key))
        else:
            choices.append(questionary.Choice(
                f"{info['name']} (not installed)",
                value=key,
                disabled="not found in PATH",
            ))

    selected = questionary.select(
        "Which agent?",
        choices=choices,
        style=CAPSEAL_STYLE,
    ).ask()

    return selected


def _write_mcp_json(target: Path) -> Path:
    """Write/merge .mcp.json at workspace root with capseal MCP config."""
    capseal_bin = _capseal_bin()
    capseal_entry = {
        "command": capseal_bin,
        "args": ["mcp-serve", "-w", str(target)],
    }

    mcp_path = target / ".mcp.json"
    mcp_config: dict = {"mcpServers": {}}

    # Merge with existing if present
    if mcp_path.exists():
        try:
            existing = json.loads(mcp_path.read_text())
            if isinstance(existing, dict):
                mcp_config = existing
                if "mcpServers" not in mcp_config:
                    mcp_config["mcpServers"] = {}
        except (json.JSONDecodeError, OSError):
            pass

    mcp_config["mcpServers"]["capseal"] = capseal_entry
    mcp_path.write_text(json.dumps(mcp_config, indent=2) + "\n")
    return mcp_path


def _write_claude_instructions(target: Path) -> None:
    """Write .claude/instructions.md with gating protocol if not already present."""
    claude_dir = target / ".claude"
    instructions_file = claude_dir / "instructions.md"
    try:
        claude_dir.mkdir(parents=True, exist_ok=True)
        if instructions_file.exists():
            existing = instructions_file.read_text()
            if "CapSeal Required" not in existing:
                with open(instructions_file, "a") as f:
                    f.write("\n\n" + CAPSEAL_INSTRUCTIONS)
        else:
            instructions_file.write_text(CAPSEAL_INSTRUCTIONS)
    except OSError:
        pass


def _start_session(target: Path, config: dict) -> None:
    """Launch an agent with CapSeal MCP pre-injected."""
    console.print()

    # Resolve which agent to use
    agent_key = _resolve_agent(config)
    if not agent_key:
        return

    agent_info = AGENTS[agent_key]
    agent_name = agent_info["name"]

    # Save as default for next time
    if config.get("default_agent") != agent_key:
        config["default_agent"] = agent_key
        _save_config(target, config)

    # Pre-launch: write MCP config
    if agent_info["mcp_support"]:
        _write_mcp_json(target)

    # Claude Code gets instructions.md too
    if agent_key == "claude-code":
        _write_claude_instructions(target)

    # Risk model status
    posteriors = target / ".capseal" / "models" / "beta_posteriors.npz"
    if posteriors.exists():
        model_line = "loaded (predictions active)"
    else:
        model_line = "not trained (gates will approve all changes)"
        console.print('  [dim]Tip: No risk model yet. Gates will approve all changes.[/dim]')
        console.print('  [dim]Train first with "Train risk model" from the main menu.[/dim]\n')

    # MCP status
    if agent_info["mcp_support"]:
        mcp_line = "injected (5 tools: gate, record, seal, status, context)"
    else:
        mcp_line = "not available for this agent"

    # Session banner
    banner_lines = [
        f"  Agent:      {agent_name}",
        f"  Workspace:  {target}",
        f"  MCP:        {mcp_line}",
        f"  Risk model: {model_line}",
    ]
    if agent_info["mcp_support"]:
        banner_lines.append("")
        banner_lines.append("  All agent actions will be gated and sealed.")

    console.print(Panel(
        "\n".join(banner_lines),
        title="[bold]CapSeal Session[/bold]",
        title_align="left",
        border_style="cyan",
        padding=(1, 1),
        expand=False,
    ))

    if not agent_info["mcp_support"] and agent_key != "terminal":
        console.print(f"  [dim]Note: {agent_name} does not support MCP. Actions won't be gated.[/dim]")
        console.print(f"  [dim]Consider using 'capseal autopilot .' for gated execution.[/dim]")
        console.print()

    # Build launch command
    env = os.environ.copy()
    env["CAPSEAL_ACTIVE"] = "1"
    env["CAPSEAL_WORKSPACE"] = str(target)
    env["CAPSEAL_AGENT"] = agent_name

    if agent_key == "terminal":
        capseal_ps1 = r'\[\033[36m\]capseal\[\033[0m\] \[\033[33m\]\w\[\033[0m\] ❯ '
        env["PS1"] = capseal_ps1
        cmd = [os.environ.get("SHELL", "/bin/bash"), "--norc", "--noprofile"]
        console.print("  [dim]CapSeal subshell. Type 'exit' to end the session.[/dim]")
    elif agent_key == "cursor":
        cmd = [agent_info["binary"], "--folder", str(target)]
    else:
        cmd = [agent_info["binary"]]

    console.print(f"\n  Launching {agent_name}...\n")

    # Launch with try/finally for auto-seal
    start_time = time.time()
    cap_path = None
    launch_failed = False
    try:
        subprocess.run(cmd, cwd=str(target), env=env)
    except KeyboardInterrupt:
        console.print("\n  [dim]Session interrupted.[/dim]")
    except FileNotFoundError:
        console.print(f"\n  [red]Error: '{cmd[0]}' not found in PATH.[/red]")
        launch_failed = True
    finally:
        elapsed = time.time() - start_time

        if not launch_failed:
            # Cursor may return immediately (opens in background)
            if agent_key == "cursor" and elapsed < 2.0:
                console.print()
                console.print("  [dim]Cursor opened in the background.[/dim]")
                console.print("  [dim]When your session is done, run:[/dim]")
                console.print("  [cyan]capseal verify .capseal/runs/latest.cap[/cyan]")
            else:
                # Auto-seal any unfinished sessions
                if agent_info["mcp_support"]:
                    cap_path = _auto_seal_if_needed(target)
                _show_session_summary(target, cap_path, agent_name, elapsed)
                # Auto-learn if model is missing or stale
                if cap_path:
                    _maybe_auto_learn(target, config)


# ── Auto-seal ────────────────────────────────────────────────────────────────

def _auto_seal_if_needed(target: Path) -> str | None:
    """Seal any unfinished MCP sessions. Returns .cap path or None."""
    runs_dir = target / ".capseal" / "runs"
    if not runs_dir.exists():
        return None

    # Find unsealed run dirs (have actions.jsonl but no .cap file)
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        if run_dir.name in ("latest",):
            continue

        actions_file = run_dir / "actions.jsonl"
        if not actions_file.exists():
            continue

        content = actions_file.read_text().strip()
        if not content:
            continue

        # Already has .cap file?
        cap_file = runs_dir / f"{run_dir.name}.cap"
        if cap_file.exists():
            # Session was properly sealed by the agent
            return str(cap_file)

        # Check if already finalized but .cap creation failed
        metadata_path = run_dir / "run_metadata.json"
        already_finalized = False
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text())
                already_finalized = meta.get("proved", False)
            except (json.JSONDecodeError, OSError):
                pass

        # Attempt auto-seal
        try:
            from ..agent_runtime import AgentRuntime
            from .cap_format import create_run_cap_file

            posteriors_path = target / ".capseal" / "models" / "beta_posteriors.npz"
            runtime = AgentRuntime(
                output_dir=run_dir,
                gate_posteriors=posteriors_path if posteriors_path.exists() else None,
            )

            action_count = len(runtime.actions) if hasattr(runtime, 'actions') else 0
            if action_count == 0:
                return None

            if not already_finalized:
                runtime.finalize(prove=True)

            create_run_cap_file(
                run_dir=run_dir,
                output_path=cap_file,
                run_type="mcp",
                extras={"auto_sealed": True, "actions_count": action_count},
            )

            # Update symlinks
            latest_cap = runs_dir / "latest.cap"
            if latest_cap.is_symlink() or latest_cap.exists():
                latest_cap.unlink()
            latest_cap.symlink_to(cap_file.name)

            latest = runs_dir / "latest"
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            latest.symlink_to(run_dir.name)

            console.print("  [green]✓[/green] Session auto-sealed")
            return str(cap_file)

        except Exception as e:
            console.print(f"  [yellow]Auto-seal failed: {e}[/yellow]")
            return None

    return None


# ── Session summary ──────────────────────────────────────────────────────────

def _show_session_summary(
    target: Path,
    cap_path: str | None,
    agent_name: str,
    elapsed: float,
) -> None:
    """Display summary after agent exits."""
    console.print()

    if not cap_path:
        console.print("  [dim]No actions were recorded in this session.[/dim]")
        return

    # Read manifest for details
    actions_count = "?"
    chain_hash = ""
    try:
        from .cap_format import read_cap_manifest
        manifest = read_cap_manifest(Path(cap_path))
        actions_count = manifest.extras.get("actions_count", manifest.extras.get("episodes", "?"))
        chain_hash = manifest.extras.get("chain_hash", manifest.extras.get("run_hash", ""))
    except Exception:
        pass

    # Format duration
    minutes = int(elapsed) // 60
    if minutes > 0:
        duration = f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        duration = f"{int(elapsed)} seconds"

    cap_name = Path(cap_path).name
    hash_display = f"{chain_hash[:16]}..." if chain_hash else "(none)"

    lines = [
        f"  Agent:       {agent_name}",
        f"  Duration:    {duration}",
        f"  Actions:     {actions_count}",
        f"  Receipt:     .capseal/runs/{cap_name}",
        f"  Chain hash:  {hash_display}",
    ]

    console.print(Panel(
        "\n".join(lines),
        title="[bold]Session Complete[/bold]",
        title_align="left",
        border_style="green",
        padding=(1, 1),
        expand=False,
    ))
    console.print(f"  Verify: [cyan]capseal verify .capseal/runs/{cap_name}[/cyan]")


# ── Auto-learn ───────────────────────────────────────────────────────────────

def _maybe_auto_learn(target: Path, config: dict) -> None:
    """Automatically run learn if the risk model is missing or stale."""
    # Subscription users don't have an API key for LLM calls
    if config.get("auth_method") == "subscription":
        return

    posteriors = target / ".capseal" / "models" / "beta_posteriors.npz"
    runs_dir = target / ".capseal" / "runs"
    if not runs_dir.exists():
        return

    cap_files = sorted(
        [f for f in runs_dir.glob("*.cap") if not f.is_symlink()],
        key=lambda f: f.name,
    )
    session_count = len(cap_files)

    needs_learn = False
    reason = ""

    if not posteriors.exists():
        # No model yet — need at least 2 sessions to have meaningful data
        if session_count >= 2:
            needs_learn = True
            reason = "No risk model yet. Building from session history..."
    else:
        # Model exists — check if stale
        last_update = config.get("last_model_update", "")
        if last_update:
            # Count sessions newer than last model update
            new_sessions = 0
            for cap in cap_files:
                try:
                    ts_part = cap.stem[:15]
                    if ts_part > last_update:
                        new_sessions += 1
                except (ValueError, IndexError):
                    pass
            if new_sessions >= 3:
                needs_learn = True
                reason = f"Risk model is stale ({new_sessions} new sessions). Updating..."
        else:
            # No timestamp recorded — treat model as potentially stale
            if session_count >= 4:
                needs_learn = True
                reason = "Risk model may be stale. Updating..."

    if not needs_learn:
        return

    console.print()
    console.print(f"  [cyan]{reason}[/cyan]")

    # Run a quick learn (3 rounds, $3 budget)
    capseal_bin = _capseal_bin()
    learn_cmd = [capseal_bin, "learn", str(target), "--rounds", "3", "--budget", "3"]

    # Pass test_cmd if configured
    test_cmd = config.get("test_cmd")
    if test_cmd:
        learn_cmd.extend(["--test-cmd", test_cmd])

    try:
        subprocess.run(learn_cmd, cwd=str(target))

        # Update last_model_update timestamp in config
        from datetime import datetime
        config["last_model_update"] = datetime.now().strftime("%Y%m%dT%H%M%S")
        _save_config(target, config)
    except Exception as e:
        console.print(f"  [yellow]Auto-learn failed: {e}[/yellow]")


# ── View past sessions ───────────────────────────────────────────────────────

def _view_sessions(target: Path) -> None:
    """List past session receipts in a table."""
    runs_dir = target / ".capseal" / "runs"
    if not runs_dir.exists():
        console.print("  [dim]No sessions found. Start a coding session first.[/dim]")
        return

    cap_files = sorted(
        [f for f in runs_dir.glob("*.cap") if not f.is_symlink()],
        key=lambda f: f.name,
        reverse=True,
    )

    if not cap_files:
        console.print("  [dim]No session receipts found.[/dim]")
        return

    table = Table(title="Past Sessions", box=ROUNDED, border_style="cyan")
    table.add_column("Date", style="white")
    table.add_column("Type", style="cyan")
    table.add_column("Actions", justify="right")
    table.add_column("Hash", style="dim")
    table.add_column("Status", justify="center")

    for cap_file in cap_files[:20]:
        try:
            from .cap_format import read_cap_manifest
            manifest = read_cap_manifest(cap_file)

            # Parse date from filename (YYYYMMDDTHHMMSS-type.cap)
            name = cap_file.stem
            try:
                # Handle names like "20260207T120000-mcp" or "20260207T120000-learn"
                ts_part = name[:15]  # YYYYMMDDTHHMMSS
                dt = datetime.strptime(ts_part, "%Y%m%dT%H%M%S")
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, IndexError):
                date_str = manifest.created_at[:16] if manifest.created_at else "unknown"

            # Run type from filename suffix
            parts = name.split("-", 1)
            run_type = parts[1] if len(parts) > 1 else "unknown"

            actions = str(manifest.extras.get("actions_count",
                         manifest.extras.get("episodes", "?")))
            chain_hash = manifest.extras.get("chain_hash",
                         manifest.extras.get("run_hash", ""))
            hash_str = chain_hash[:12] + "..." if chain_hash else "-"
            status = "[green]sealed[/green]"

            table.add_row(date_str, run_type, actions, hash_str, status)
        except Exception:
            table.add_row(cap_file.name, "?", "?", "-", "[red]error[/red]")

    console.print()
    console.print(table)

    if len(cap_files) > 20:
        console.print(f"  [dim]... and {len(cap_files) - 20} more[/dim]")

    console.print()

    # Let user select a session to view the action chain
    if cap_files:
        session_choices = []
        for cap_file in cap_files[:20]:
            name = cap_file.stem
            try:
                ts_part = name[:15]
                dt = datetime.strptime(ts_part, "%Y%m%dT%H%M%S")
                label = f"{dt.strftime('%Y-%m-%d %H:%M')} ({name.split('-', 1)[1] if '-' in name else 'unknown'})"
            except (ValueError, IndexError):
                label = name
            session_choices.append(questionary.Choice(label, value=str(cap_file)))
        session_choices.append(questionary.Choice("Back to menu", value=""))

        selected = questionary.select(
            "View action chain for a session?",
            choices=session_choices,
            style=CAPSEAL_STYLE,
        ).ask()

        if selected:
            show_action_chain(Path(selected))


# ── Chain visualization ──────────────────────────────────────────────────────

def _load_actions_from_cap(cap_path: Path) -> list[dict]:
    """Extract actions from a .cap file or its run directory."""
    import tarfile

    # Try to find actions.jsonl in the .cap tarball
    actions = []
    try:
        with tarfile.open(cap_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith("actions.jsonl"):
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8").strip()
                        for line in content.split("\n"):
                            if line.strip():
                                try:
                                    actions.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                        return actions
    except Exception:
        pass

    # Fallback: look for the run directory next to the .cap file
    run_dir_name = cap_path.stem  # e.g., "20260207T183338-mcp"
    run_dir = cap_path.parent / run_dir_name
    actions_file = run_dir / "actions.jsonl"
    if actions_file.exists():
        content = actions_file.read_text().strip()
        for line in content.split("\n"):
            if line.strip():
                try:
                    actions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return actions


def show_action_chain(cap_path: Path) -> None:
    """Display the action chain for a .cap file."""
    # Resolve symlinks for better display
    if cap_path.is_symlink():
        cap_path = cap_path.resolve()

    actions = _load_actions_from_cap(cap_path)

    if not actions:
        console.print("  [dim]No actions found in this session.[/dim]")
        return

    # Parse session info from filename
    name = cap_path.stem
    try:
        ts_part = name[:15]
        dt = datetime.strptime(ts_part, "%Y%m%dT%H%M%S")
        date_str = dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, IndexError):
        date_str = "unknown"

    run_type = name.split("-", 1)[1] if "-" in name else "unknown"

    # Header
    console.print()
    console.print(f"  [bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"  [bold]SESSION[/bold]: {name}")
    console.print(f"  Date: {date_str}")
    console.print(f"  Type: {run_type}")
    console.print(f"  Receipt: .capseal/runs/{cap_path.name}")
    console.print(f"  [bold cyan]{'=' * 60}[/bold cyan]")
    console.print()
    console.print(f"  [bold]ACTION CHAIN[/bold] ({len(actions)} actions)")
    console.print()

    # Render each action as a chain link
    for i, action in enumerate(actions):
        action_id = action.get("action_id", f"act_{i:04d}")
        action_type = action.get("action_type", "unknown")
        gate_score = action.get("gate_score")
        gate_decision = action.get("gate_decision")
        success = action.get("success", True)
        parent_hash = action.get("parent_receipt_hash")
        metadata = action.get("metadata") or {}

        # Get description from metadata (stored by MCP server)
        description = metadata.get("description", "")
        files = metadata.get("files_affected", [])
        tool_name = metadata.get("tool_name", "")

        # Compute receipt hash for display
        from ..agent_protocol import AgentAction
        try:
            agent_action = AgentAction.from_dict(action)
            receipt_hash = agent_action.compute_receipt_hash()[:12]
        except Exception:
            receipt_hash = "????????????"

        # Build the display for this action
        id_short = action_id.split("_")[1] if "_" in action_id else action_id

        # Gate line
        if gate_decision:
            decision_display = {"pass": "approve", "skip": "deny", "human_review": "flag"}.get(
                gate_decision, gate_decision
            )
            gate_line = f"gate: {decision_display}"
            if gate_score is not None:
                gate_line += f" (p_fail={gate_score:.2f})"
        else:
            gate_line = "gate: (not gated)"

        # Description line
        if description:
            desc_line = description[:50]
        elif files:
            desc_line = ", ".join(f[:30] for f in files[:2])
        elif tool_name:
            desc_line = f"tool: {tool_name}"
        else:
            desc_line = action_type

        # Success indicator
        status_icon = "[green]✓[/green]" if success else "[red]✗[/red]"

        # Hash line
        hash_line = f"hash: {receipt_hash}..."
        if parent_hash:
            hash_line += f" [dim]← chains to {parent_hash[:12]}[/dim]"

        # Box width
        w = 55

        # Connection line from previous action
        if i > 0:
            console.print(f"  {'':>14}│")

        # Action box
        console.print(f"  ┌─ {action_id} ─{'─' * (w - len(action_id) - 5)}┐")
        console.print(f"  │ {status_icon} {gate_line:<{w - 4}}│")
        console.print(f"  │   {desc_line:<{w - 5}}│")
        console.print(f"  │   {hash_line}")
        console.print(f"  └{'─' * (w - 1)}┘")

    # Chain summary
    console.print()
    all_success = all(a.get("success", True) for a in actions)
    chain_status = "[green]✓ intact[/green]" if all_success else "[yellow]⚠ has failures[/yellow]"
    console.print(f"  CHAIN: {chain_status} ({len(actions)}/{len(actions)} linked)")
    console.print(f"  [bold cyan]{'=' * 60}[/bold cyan]")
    console.print()


# ── Run a command ───────────────────────────────────────────────────────────

_SHELL_COMMANDS = frozenset({
    "echo", "ls", "cat", "cd", "git", "pwd", "grep", "find", "rm", "cp",
    "mv", "mkdir", "touch", "head", "tail", "less", "pip", "python",
    "python3", "pytest", "npm", "node", "make", "curl", "wget", "which",
    "env", "export", "source", "diff", "wc", "sort", "uniq", "xargs",
    "docker", "cargo", "go", "ruby", "java", "sh", "bash", "zsh",
})


def _run_shell_repl(target: Path, config: dict) -> None:
    """Persistent REPL — capseal and shell commands until user exits."""
    console.print()
    console.print('  [dim]CapSeal shell — type "exit" to return to hub[/dim]')
    console.print()

    # State-changing commands that trigger a mini status update
    _state_commands = {"learn", "fix", "autopilot", "scan", "init", "sign"}

    while True:
        try:
            user_input = questionary.text(
                "capseal >",
                style=CAPSEAL_STYLE,
            ).ask()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input or user_input.strip() in ("exit", "quit", "q"):
            break

        cmd_text = user_input.strip()
        _execute_shell_input(target, cmd_text)

        # Show mini status after state-changing commands
        first_word = cmd_text.split()[0] if cmd_text.split() else ""
        # Strip "capseal" prefix for checking
        check_word = first_word
        if cmd_text.startswith("capseal "):
            parts = cmd_text[len("capseal "):].strip().split()
            check_word = parts[0] if parts else ""
        if check_word in _state_commands:
            _show_mini_status(target)


def _execute_shell_input(target: Path, user_input: str) -> None:
    """Execute a single command line (capseal subcommand or shell command)."""
    # Strip leading "capseal " if the user typed the full command
    if user_input.startswith("capseal "):
        user_input = user_input[len("capseal "):].strip()

    if user_input.startswith("export "):
        # Handle export in-process so env vars persist
        assignment = user_input[7:].replace("\n", "").replace("\r", "").strip()
        if "=" in assignment:
            key, value = assignment.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value
            print_success(f"{key} set successfully")
            if "API_KEY" in key:
                console.print("  [dim]API key configured for this session.[/dim]\n")
        else:
            print_warn("Usage: export KEY=VALUE")
        return
    elif user_input.startswith("!"):
        # Explicit shell escape
        subprocess.run(user_input[1:].strip(), shell=True, cwd=str(target))
    elif shlex.split(user_input)[0] in _SHELL_COMMANDS:
        # Known shell command — run directly
        subprocess.run(user_input, shell=True, cwd=str(target))
    else:
        # Capseal subcommand
        capseal_bin = _capseal_bin()
        subprocess.run([capseal_bin] + shlex.split(user_input), cwd=str(target))


def _show_mini_status(target: Path) -> None:
    """Show compact status after state-changing commands."""
    model_path = target / ".capseal" / "models" / "beta_posteriors.npz"
    if model_path.exists():
        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        age = datetime.now() - mtime
        if age.total_seconds() < 60:
            console.print(f'  [dim]✓ Risk model updated {int(age.total_seconds())}s ago[/dim]')


# ── Train risk model ─────────────────────────────────────────────────────────

def _train_model(target: Path, config: dict) -> None:
    """Run capseal learn to train the risk model."""
    console.print()

    # Check if git history is available
    import subprocess as _sp
    has_git = _sp.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=str(target), capture_output=True,
    ).returncode == 0

    # Check for LLM access
    has_api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    has_cli = (
        shutil.which("claude")
        or shutil.which("codex")
        or shutil.which("gemini")
    )
    has_llm = has_api_key or has_cli

    # Build training method choices
    choices = []
    if has_git:
        choices.append(questionary.Choice(
            "Quick learn from git history (free, ~5 seconds)",
            value="git",
        ))
    if has_llm:
        choices.append(questionary.Choice(
            "Full learn with LLM patches ($0.20-5.00, ~2 minutes)",
            value="llm",
        ))

    if not choices:
        console.print(Panel(
            "[yellow]Training requires either git history or LLM access.[/yellow]\n\n"
            "Either:\n"
            "  • Use a git repository (for free git-history learning)\n"
            "  • Install a provider CLI (claude, codex, gemini)\n"
            "  • Set an API key: export ANTHROPIC_API_KEY=sk-ant-...\n\n"
            "Then try again.",
            title="No Training Method Available",
            border_style="yellow",
        ))
        return

    # If only one option, use it directly
    if len(choices) == 1:
        method = choices[0].value
    else:
        method = questionary.select(
            "Training method:",
            choices=choices,
            style=CAPSEAL_STYLE,
        ).ask()
        if not method:
            return

    if method == "git":
        cmd = [_capseal_bin(), "learn", str(target), "--from-git"]
    else:
        cmd = [_capseal_bin(), "learn", str(target), "--rounds", "5"]
        test_cmd = config.get("test_cmd")
        if test_cmd:
            cmd.extend(["--test-cmd", test_cmd])
        budget = config.get("learn_budget", "5.0")
        cmd.extend(["--budget", str(budget)])

    # Run learn with inherited stdio so user sees progress
    result = subprocess.run(cmd, cwd=str(target))

    if result.returncode == 0:
        # Reload config to pick up any changes learn made
        config_path = target / ".capseal" / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config.update(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass

        # Update last_model_update timestamp
        config["last_model_update"] = datetime.now().isoformat()
        _save_config(target, config)

        print_success("Risk model trained successfully!")
        console.print("  Your next coding session will use learned risk gating.\n")
    else:
        print_warn("Training encountered errors. Check the output above.")


# ── Risk report wrapper ───────────────────────────────────────────────────────

def _run_report(target: Path) -> None:
    """Run the risk report dashboard."""
    from .report_cmd import _build_report, _print_rich_report
    console.print()
    console.print("  [cyan]Scanning codebase and building risk report...[/cyan]")
    console.print()
    data = _build_report(target)
    _print_rich_report(data)


# ── Self-test wrapper ────────────────────────────────────────────────────────

def _run_selftest(target: Path) -> None:
    """Run the 9-step self-test."""
    from .test_cmd import test_command
    try:
        ctx = test_command.make_context("test", [str(target)])
        with ctx:
            test_command.invoke(ctx)
    except SystemExit:
        pass  # test_command exits 1 on failure — don't exit the hub


# ── Autopilot wrapper ────────────────────────────────────────────────────────

def _run_autopilot(target: Path) -> None:
    """Launch capseal autopilot."""
    console.print()
    console.print("  [cyan]Running capseal autopilot...[/cyan]")
    console.print()
    capseal_bin = _capseal_bin()
    subprocess.run([capseal_bin, "autopilot", str(target)])


# ── Configure wrapper ────────────────────────────────────────────────────────

def _run_configure(target: Path) -> None:
    """Launch the init TUI for configuration."""
    from .init_tui import run_init_tui
    run_init_tui(str(target))


# ── Entry point ──────────────────────────────────────────────────────────────

def run_hub(workspace: str = ".") -> None:
    """Main hub entry point. Called when user types `capseal` with no args."""
    target = Path(workspace).resolve()

    try:
        config = _ensure_workspace(target)
        if not config:
            # User cancelled during setup
            return
        _show_status_banner(target, config)
        _hub_menu(target, config)
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/dim]")
