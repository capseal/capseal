#!/usr/bin/env python3
"""
CapSeal interactive onboarding TUI.
Drop this into src/capseal/cli/init_tui.py and call run_init_tui() from the init command.
"""

import json
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.box import ROUNDED, HEAVY, DOUBLE
from rich import print as rprint
import questionary
from questionary import Style

console = Console()

# ── Styling ──────────────────────────────────────────────────────────────────

CAPSEAL_STYLE = Style([
    ("qmark", "fg:cyan bold"),
    ("question", "fg:white bold"),
    ("answer", "fg:cyan bold"),
    ("pointer", "fg:cyan bold"),
    ("highlighted", "fg:cyan bold"),
    ("selected", "fg:cyan"),
    ("separator", "fg:#666666"),
    ("instruction", "fg:#888888"),
    ("text", "fg:white"),
    ("disabled", "fg:#666666 italic"),
])

CYAN = "cyan"
DIM = "dim"
GREEN = "green"
YELLOW = "yellow"
RED = "red"
BOLD = "bold"


# ── Banner ───────────────────────────────────────────────────────────────────

BANNER = """[cyan]
 ██████╗ █████╗ ██████╗ ███████╗███████╗ █████╗ ██╗     
██╔════╝██╔══██╗██╔══██╗██╔════╝██╔════╝██╔══██╗██║     
██║     ███████║██████╔╝███████╗█████╗  ███████║██║     
██║     ██╔══██║██╔═══╝ ╚════██║██╔══╝  ██╔══██║██║     
╚██████╗██║  ██║██║     ███████║███████╗██║  ██║███████╗
 ╚═════╝╚═╝  ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝  ╚═╝╚══════╝[/cyan]"""


# ── Helper functions ─────────────────────────────────────────────────────────

def print_step(icon: str, label: str, detail: str = ""):
    """Print a step marker in the onboarding flow."""
    if detail:
        console.print(f"│\n◇  [bold]{label}[/bold]")
        console.print(f"│  [dim]{detail}[/dim]")
    else:
        console.print(f"│\n◇  [bold]{label}[/bold]")


def print_box(title: str, lines: list[str]):
    """Print a bordered info box inline with the flow."""
    content = "\n".join(f"  {line}" for line in lines)
    panel = Panel(
        content,
        title=f"[bold]{title}[/bold]",
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        expand=False,
    )
    console.print("│")
    console.print(panel)


def print_success(msg: str):
    console.print(f"│  [green]✓[/green] {msg}")


def print_warn(msg: str):
    console.print(f"│  [yellow]⚠[/yellow] {msg}")


def mask_key(key: str) -> str:
    """Mask an API key for display."""
    if len(key) < 12:
        return "••••••••"
    return key[:7] + "••••" + key[-4:]


def animate_dots(msg: str, duration: float = 0.8):
    """Quick animated dots for a 'working' feel."""
    for i in range(3):
        console.print(f"\r│  {msg}{'.' * (i + 1)}   ", end="")
        time.sleep(duration / 3)
    console.print()


# ── Main onboarding flow ────────────────────────────────────────────────────

def run_init_tui(target_dir: str = "."):
    target = Path(target_dir).resolve()
    capseal_dir = target / ".capseal"

    # ── Banner ───────────────────────────────────────────────────────────
    console.print()
    console.print(BANNER)
    console.print(f"[dim]  v0.2.0 — Learned risk gating for AI agents[/dim]")
    console.print(f"[dim]  Every AI change, verified.[/dim]\n")

    # Check if already initialized
    if capseal_dir.exists():
        overwrite = questionary.confirm(
            "This directory already has a .capseal workspace. Reinitialize?",
            default=False,
            style=CAPSEAL_STYLE,
        ).ask()
        if not overwrite:
            console.print("\n[dim]Aborted.[/dim]")
            return

    # ── Start flow ───────────────────────────────────────────────────────
    console.print("┌  [bold]CapSeal onboarding[/bold]")

    # ── Security notice ──────────────────────────────────────────────────
    # Tip for home directory
    target_note = f"Target: {target}"
    if target == Path.home():
        target_note += "\n\n[dim]Tip: Run 'capseal init' from a project directory for\nproject-specific risk models.[/dim]"

    print_box("Security", [
        "CapSeal generates and applies AI-powered code patches.",
        "Patches are gated by a learned risk model, but no gate is perfect.",
        "",
        "Recommended:",
        "• Review gated patches before applying (capseal fix . --dry-run)",
        "• Keep backups or use version control",
        "• Run capseal verify after every session",
        "",
        target_note,
    ])

    proceed = questionary.confirm(
        "I understand. Continue?",
        default=True,
        style=CAPSEAL_STYLE,
    ).ask()

    if not proceed:
        console.print("│\n└  [dim]Aborted.[/dim]")
        return

    # ── Mode selection ───────────────────────────────────────────────────
    console.print("│")
    mode = questionary.select(
        "Setup mode",
        choices=[
            questionary.Choice("QuickStart (recommended)", value="quick"),
            questionary.Choice("Advanced (custom thresholds, policy profiles)", value="advanced"),
        ],
        style=CAPSEAL_STYLE,
    ).ask()

    # ── LLM Provider ────────────────────────────────────────────────────
    console.print("│")
    provider = questionary.select(
        "LLM provider (for patch generation)",
        choices=[
            questionary.Choice("Anthropic (Claude)", value="anthropic"),
            questionary.Choice("OpenAI", value="openai"),
            questionary.Choice("Custom endpoint", value="custom"),
        ],
        style=CAPSEAL_STYLE,
    ).ask()

    # ── API Key ──────────────────────────────────────────────────────────
    if provider == "custom":
        endpoint = questionary.text(
            "Custom API endpoint URL",
            style=CAPSEAL_STYLE,
        ).ask()
    else:
        endpoint = None

    env_key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "custom": "CAPSEAL_API_KEY",
    }
    env_var = env_key_map[provider]
    existing_key = os.environ.get(env_var, "")

    if existing_key:
        use_existing = questionary.confirm(
            f"Found {env_var} in environment ({mask_key(existing_key)}). Use it?",
            default=True,
            style=CAPSEAL_STYLE,
        ).ask()
        if use_existing:
            api_key = existing_key
        else:
            api_key = questionary.password(
                f"Enter {env_var}",
                style=CAPSEAL_STYLE,
            ).ask()
    else:
        api_key = questionary.password(
            f"Enter {env_var}",
            style=CAPSEAL_STYLE,
        ).ask()

    if not api_key:
        print_warn("No API key provided. You can add one later in .capseal/config.json")
        print_warn("Without an API key, only capseal scan and capseal review --gate will work.")
        api_key = ""

    if api_key:
        print_box(f"{env_var}", [
            f"Key: {mask_key(api_key)}",
            f"Saved to .capseal/config.json",
        ])

    # ── Model selection ──────────────────────────────────────────────────
    if provider == "anthropic":
        model_choices = [
            questionary.Choice("claude-sonnet-4-20250514 (recommended)", value="claude-sonnet-4-20250514"),
            questionary.Choice("claude-opus-4-20250514", value="claude-opus-4-20250514"),
            questionary.Choice("claude-haiku-4-5-20251001", value="claude-haiku-4-5-20251001"),
        ]
    elif provider == "openai":
        model_choices = [
            questionary.Choice("gpt-4o (recommended)", value="gpt-4o"),
            questionary.Choice("gpt-4o-mini", value="gpt-4o-mini"),
            questionary.Choice("o3", value="o3"),
        ]
    else:
        model_choices = None

    if model_choices:
        console.print("│")
        model = questionary.select(
            "Default model",
            choices=model_choices,
            style=CAPSEAL_STYLE,
        ).ask()
    else:
        console.print("│")
        model = questionary.text(
            "Model name",
            style=CAPSEAL_STYLE,
        ).ask()

    # ── Plan selection ───────────────────────────────────────────────────
    console.print("│")
    print_box("Plans", [
        "[bold]Free[/bold]        scan + review --gate (risk assessment only)",
        "             No patch generation, no .cap receipts",
        "",
        "[bold cyan]Pro $29/mo[/bold cyan]   learn + fix + verify + .cap receipts",
        "             Full pipeline: learned gating + verified patches",
        "             Up to 50 learn runs/month",
        "",
        "[bold yellow]Team $99/mo[/bold yellow]  Everything in Pro + team dashboard",
        "             Shared risk models, CI/CD integration",
        "             Unlimited learn runs, priority support",
        "",
        "[bold red]Enterprise[/bold red]  SSO, audit dashboard, SLAs, on-prem",
        "             Contact sales@capseal.dev",
    ])

    plan = questionary.select(
        "Select plan",
        choices=[
            questionary.Choice("Free (scan + gate only)", value="free"),
            questionary.Choice("Pro $29/mo (full pipeline)", value="pro"),
            questionary.Choice("Team $99/mo (collaboration)", value="team"),
            questionary.Choice("Enterprise (contact sales)", value="enterprise"),
            questionary.Choice("I have a license key", value="license"),
        ],
        style=CAPSEAL_STYLE,
    ).ask()

    license_key = ""
    if plan == "license":
        license_key = questionary.text(
            "Enter license key",
            style=CAPSEAL_STYLE,
        ).ask()
        if license_key:
            print_success(f"License key saved")
            plan = "pro"  # Assume pro for now
    elif plan == "enterprise":
        console.print("│  [dim]Contact sales@capseal.dev for enterprise pricing.[/dim]")
        plan = "free"
    elif plan in ("pro", "team"):
        console.print("│  [dim]Stripe checkout coming soon. Using free tier for now.[/dim]")
        console.print("│  [dim]All features unlocked during beta.[/dim]")
        # During beta, unlock everything

    # ── Agent selection ──────────────────────────────────────────────────
    console.print("│")
    print_box("Agent Integration", [
        "CapSeal is a trust layer that protects AI agents.",
        "Select the agents you use, and we'll show you how to connect them.",
    ])

    agents = questionary.checkbox(
        "Which agents do you use?",
        choices=[
            questionary.Choice("Claude Code", value="claude-code", checked=True),
            questionary.Choice("OpenClaw", value="openclaw"),
            questionary.Choice("Cursor", value="cursor"),
            questionary.Choice("Windsurf", value="windsurf"),
            questionary.Choice("Cline", value="cline"),
            questionary.Choice("Custom MCP client", value="custom-mcp"),
            questionary.Choice("None (CLI only)", value="cli-only"),
        ],
        style=CAPSEAL_STYLE,
    ).ask()

    if not agents:
        agents = ["cli-only"]

    # Store integration status
    integrations = {}
    workspace_path = str(target)

    # Show integration instructions for each selected agent
    for agent in agents:
        if agent == "cli-only":
            continue

        if agent == "claude-code":
            print_box("Claude Code integration", [
                "Run this to connect CapSeal to Claude Code:",
                "",
                f"  [cyan]claude mcp add capseal -- capseal mcp-serve -w {workspace_path}[/cyan]",
                "",
                "Or add to your Claude Code MCP config:",
                '  {',
                '    "capseal": {',
                '      "type": "stdio",',
                '      "command": "capseal",',
                f'      "args": ["mcp-serve", "-w", "{workspace_path}"]',
                '    }',
                '  }',
            ])
            integrations["claude-code"] = {"type": "mcp", "configured": False}

            auto_config = questionary.confirm(
                "Auto-configure Claude Code now?",
                default=True,
                style=CAPSEAL_STYLE,
            ).ask()

            if auto_config:
                import subprocess
                result = subprocess.run(
                    ["claude", "mcp", "add", "capseal", "--", "capseal", "mcp-serve", "-w", workspace_path],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    print_success("Claude Code configured")
                    integrations["claude-code"]["configured"] = True
                elif "already exists" in result.stderr.lower() or "already" in result.stdout.lower():
                    print_success("Claude Code already configured (capseal MCP server found)")
                    integrations["claude-code"]["configured"] = True
                else:
                    print_warn(f"Failed to configure: {result.stderr.strip() or 'Unknown error'}")
                    print_warn("You can run the command manually later.")

                # Write .claude/instructions.md for Claude Code
                claude_dir = target / ".claude"
                instructions_file = claude_dir / "instructions.md"
                capseal_instructions = """## CapSeal Required — MANDATORY

This project uses CapSeal for verified code changes. You MUST follow this protocol:

### At session start or after any conversation summary:
Call `capseal_status` to check session state. This recovers your place if context was compacted.

### Before EACH file change:
Call `capseal_gate` with the SPECIFIC FILE you're about to change. Gate each file INDIVIDUALLY, not multiple files at once.

### Gate decisions are BINDING:
- **"approve"**: You may proceed with the change
- **"deny"**: **DO NOT make the change. Period.** Tell the user: "CapSeal blocked this change (predicted failure rate: X%). I cannot proceed."
- **"flag"**: Ask the user for explicit permission before proceeding. Do not assume approval.

### After each APPROVED change:
Call `capseal_record` with what you changed and whether it succeeded.

### When ALL changes are complete:
Call `capseal_seal` to create the cryptographic receipt. Do not skip this step.

Gate decisions from CapSeal are not suggestions. They are requirements.
"""
                try:
                    claude_dir.mkdir(parents=True, exist_ok=True)
                    if instructions_file.exists():
                        # Check if CapSeal section already exists
                        existing = instructions_file.read_text()
                        if "CapSeal Required" not in existing:
                            # Append CapSeal section
                            with open(instructions_file, "a") as f:
                                f.write("\n\n" + capseal_instructions)
                            print_success("Added CapSeal instructions to .claude/instructions.md")
                        else:
                            print_success(".claude/instructions.md already has CapSeal instructions")
                    else:
                        instructions_file.write_text(capseal_instructions)
                        print_success("Created .claude/instructions.md with CapSeal instructions")
                except Exception as e:
                    print_warn(f"Could not write .claude/instructions.md: {e}")

        elif agent == "openclaw":
            print_box("OpenClaw integration", [
                "Option A: Install as a skill (simple)",
                f"  [cyan]capseal export-skill ~/.openclaw/workspace/skills/capseal[/cyan]",
                "",
                "Option B: Add as MCP server via mcporter (deep)",
                "  Add to mcporter config:",
                '  {',
                '    "capseal": {',
                '      "command": "capseal",',
                f'      "args": ["mcp-serve", "-w", "{workspace_path}"],',
                '      "transport": "stdio"',
                '    }',
                '  }',
            ])
            integrations["openclaw"] = {"type": "skill+mcp", "configured": False}

            auto_config = questionary.confirm(
                "Install OpenClaw skill now?",
                default=True,
                style=CAPSEAL_STYLE,
            ).ask()

            if auto_config:
                skill_dir = Path.home() / ".openclaw" / "workspace" / "skills" / "capseal"
                try:
                    skill_dir.mkdir(parents=True, exist_ok=True)
                    # Export the skill
                    from .skill_export import export_skill
                    export_skill(skill_dir)
                    print_success(f"Skill installed to {skill_dir}")
                    integrations["openclaw"]["configured"] = True
                except Exception as e:
                    print_warn(f"Failed to install skill: {e}")

        elif agent == "cursor":
            print_box("Cursor integration", [
                "Add to Cursor MCP settings (~/.cursor/mcp.json):",
                '  {',
                '    "mcpServers": {',
                '      "capseal": {',
                '        "command": "capseal",',
                f'        "args": ["mcp-serve", "-w", "{workspace_path}"]',
                '      }',
                '    }',
                '  }',
            ])
            integrations["cursor"] = {"type": "mcp", "configured": False}

        elif agent == "windsurf":
            print_box("Windsurf integration", [
                "Add to Windsurf MCP settings:",
                '  {',
                '    "capseal": {',
                '      "command": "capseal",',
                f'      "args": ["mcp-serve", "-w", "{workspace_path}"]',
                '    }',
                '  }',
            ])
            integrations["windsurf"] = {"type": "mcp", "configured": False}

        elif agent == "cline":
            print_box("Cline integration", [
                "Add to Cline MCP settings (~/.cline/mcp.json):",
                '  {',
                '    "mcpServers": {',
                '      "capseal": {',
                '        "command": "capseal",',
                f'        "args": ["mcp-serve", "-w", "{workspace_path}"]',
                '      }',
                '    }',
                '  }',
            ])
            integrations["cline"] = {"type": "mcp", "configured": False}

        elif agent == "custom-mcp":
            print_box("Custom MCP client", [
                "CapSeal exposes 3 MCP tools:",
                "  • capseal_gate   - Gate before execution (approve/deny/flag)",
                "  • capseal_record - Record after execution",
                "  • capseal_seal   - Seal session into .cap receipt",
                "",
                "Start the MCP server:",
                f"  [cyan]capseal mcp-serve -w {workspace_path}[/cyan]",
                "",
                "Or use stdio transport in your MCP config:",
                '  {',
                '    "command": "capseal",',
                f'    "args": ["mcp-serve", "-w", "{workspace_path}"],',
                '    "transport": "stdio"',
                '  }',
            ])
            integrations["custom-mcp"] = {"type": "mcp", "configured": False}

    # ── Advanced options ─────────────────────────────────────────────────
    gate_threshold = 0.6
    uncertainty_threshold = 0.15
    learn_rounds = 5
    learn_budget = 5.0

    if mode == "advanced":
        console.print("│")
        gate_threshold = float(questionary.text(
            "Gate threshold (predicted failure rate to block, default 0.6)",
            default="0.6",
            style=CAPSEAL_STYLE,
        ).ask())

        uncertainty_threshold = float(questionary.text(
            "Uncertainty threshold (flag for human review, default 0.15)",
            default="0.15",
            style=CAPSEAL_STYLE,
        ).ask())

        learn_rounds = int(questionary.text(
            "Default learn rounds (default 5)",
            default="5",
            style=CAPSEAL_STYLE,
        ).ask())

        learn_budget = float(questionary.text(
            "Default learn budget in $ (default 5.00)",
            default="5.00",
            style=CAPSEAL_STYLE,
        ).ask())

    # ── Create workspace ─────────────────────────────────────────────────
    console.print("│")
    animate_dots("│  Initializing workspace")

    # Create directories
    dirs = [
        capseal_dir,
        capseal_dir / "models",
        capseal_dir / "runs",
        capseal_dir / "policies",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    # Write config
    config = {
        "version": "0.2.0",
        "provider": provider,
        "model": model,
        "api_key_env": env_var,
        "plan": plan,
        "license_key": license_key if license_key else None,
        "agents": agents,
        "integrations": integrations,
        "gate": {
            "threshold": gate_threshold,
            "uncertainty_threshold": uncertainty_threshold,
        },
        "learn": {
            "default_rounds": learn_rounds,
            "default_budget": learn_budget,
        },
    }

    if endpoint:
        config["endpoint"] = endpoint

    if api_key:
        # Store key in a separate .env file, not in config.json
        env_path = capseal_dir / ".env"
        with open(env_path, "w") as f:
            f.write(f"{env_var}={api_key}\n")
        env_path.chmod(0o600)
        # Also set in current process so child processes (like capseal learn) inherit it
        os.environ[env_var] = api_key

    config_path = capseal_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Write default policy
    policy = {
        "name": "default",
        "gate_threshold": gate_threshold,
        "uncertainty_threshold": uncertainty_threshold,
        "auto_apply": False,
        "require_verification": True,
    }
    policy_path = capseal_dir / "policies" / "default.json"
    with open(policy_path, "w") as f:
        json.dump(policy, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    provider_display = {
        "anthropic": "Anthropic (Claude)",
        "openai": "OpenAI",
        "custom": "Custom",
    }

    plan_display = {
        "free": "Free",
        "pro": "Pro",
        "team": "Team",
        "enterprise": "Enterprise",
    }

    print_box("Workspace initialized", [
        f"Target:     {target}",
        f"Provider:   {provider_display.get(provider, provider)}",
        f"Model:      {model}",
        f"Plan:       {plan_display.get(plan, plan)} {'[green](all features unlocked during beta)[/green]' if plan != 'free' else ''}",
        f"",
        f"Config:     .capseal/config.json",
        f"Credentials:.capseal/.env",
        f"Models:     .capseal/models/",
        f"Runs:       .capseal/runs/",
        f"Policy:     .capseal/policies/default.json",
        f"",
        f"Gate:       block patches with >{gate_threshold*100:.0f}% predicted failure",
        f"Flag:       human review when uncertainty >{uncertainty_threshold}",
    ])

    # ── Semgrep check ────────────────────────────────────────────────────
    import shutil
    if shutil.which("semgrep"):
        print_success("Semgrep found")
    else:
        print_warn("Semgrep not found. Install with: pip install semgrep")
        print_warn("capseal scan and capseal learn require Semgrep.")

    # ── Quick start ──────────────────────────────────────────────────────
    console.print("│")
    console.print("◇  [bold]Quick start[/bold]")
    console.print("│")
    console.print("│  [cyan]capseal learn . --rounds 5[/cyan]     Build risk model for this codebase")
    console.print("│  [cyan]capseal fix . --dry-run[/cyan]        Preview what would be fixed")
    console.print("│  [cyan]capseal fix .[/cyan]                  Generate verified patches")
    console.print("│  [cyan]capseal verify[/cyan]                 Verify the sealed receipt")
    console.print("│")

    # ── Auto-learn offer ─────────────────────────────────────────────────
    if api_key:
        run_learn = questionary.confirm(
            f"Run capseal learn now? (~${learn_budget:.2f} budget, {learn_rounds} rounds)",
            default=False,
            style=CAPSEAL_STYLE,
        ).ask()

        if run_learn:
            console.print("│")
            console.print(f"│  [cyan]Running: capseal learn . --rounds {learn_rounds} --budget {learn_budget}[/cyan]")
            console.print("│  [dim]This will take 1-2 minutes...[/dim]")
            console.print("│")
            console.print("└  [bold green]Setup complete. Starting learn...[/bold green]\n")
            # Hand off to the actual learn command
            os.system(f"capseal learn . --rounds {learn_rounds} --budget {learn_budget}")
            return

    console.print(f"└  [bold green]Done. Ready to learn your codebase.[/bold green]\n")


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    try:
        run_init_tui(target)
    except KeyboardInterrupt:
        console.print("\n[dim]Aborted.[/dim]")
        sys.exit(1)
