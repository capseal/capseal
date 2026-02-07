"""capseal doctor - Check that everything is wired up correctly.

Usage:
    capseal doctor              # Check current directory
    capseal doctor /path/to/project
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import click
import numpy as np


CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _check(label: str, ok: bool, detail: str = "", fail_detail: str = "") -> bool:
    """Print a check result line and return whether it passed."""
    if ok:
        click.echo(f"  {label + ':':<16}{GREEN}✓{RESET} {detail}")
    else:
        click.echo(f"  {label + ':':<16}{RED}✗{RESET} {fail_detail or detail}")
    return ok


def _check_warn(label: str, detail: str) -> None:
    click.echo(f"  {label + ':':<16}{YELLOW}⚠{RESET} {detail}")


@click.command("doctor")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--json", "output_json", is_flag=True, help="Output JSON")
def doctor_command(path: str, output_json: bool) -> None:
    """Check that your CapSeal workspace is correctly configured.

    Checks workspace, config, API key, Semgrep, risk model,
    agent integrations, and last session receipt.

    \b
    Examples:
        capseal doctor
        capseal doctor /path/to/project
    """
    target = Path(path).expanduser().resolve()
    capseal_dir = target / ".capseal"
    results = {}

    if not output_json:
        click.echo()
        click.echo(f"{CYAN}{'═' * 55}{RESET}")
        click.echo(f"{CYAN}  CAPSEAL DOCTOR{RESET}")
        click.echo(f"{CYAN}{'═' * 55}{RESET}")
        click.echo(f"  Workspace:      {target}")
        click.echo()

    # 1. Workspace
    has_workspace = capseal_dir.exists()
    results["workspace"] = has_workspace
    if not output_json:
        _check("Config", (capseal_dir / "config.json").exists(),
               ".capseal/config.json found",
               "No .capseal/ workspace (run: capseal init)")

    if not has_workspace:
        if output_json:
            click.echo(json.dumps({"status": "no_workspace", "target": str(target)}))
        else:
            click.echo()
            click.echo(f"  Run {CYAN}capseal init{RESET} to set up this directory.")
            click.echo(f"{CYAN}{'═' * 55}{RESET}")
        return

    # 2. Config
    config = {}
    config_path = capseal_dir / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            pass

    # 3. Provider / API key
    provider = config.get("provider", "unknown")
    auth_method = config.get("auth_method", "api_key")
    provider_display = {
        "anthropic": "Anthropic", "openai": "OpenAI",
        "google": "Google", "custom": "Custom",
    }

    env_key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "custom": "CAPSEAL_API_KEY",
    }

    if auth_method == "subscription":
        results["provider"] = True
        if not output_json:
            _check("Provider", True, f"{provider_display.get(provider, provider)} (subscription)")
    else:
        env_var = env_key_map.get(provider, "CAPSEAL_API_KEY")
        # Check environment, then .capseal/.env
        has_key = bool(os.environ.get(env_var))
        if not has_key:
            env_file = capseal_dir / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.startswith(f"{env_var}=") and len(line.split("=", 1)[1].strip()) > 0:
                        has_key = True
                        break
        results["provider"] = has_key
        if not output_json:
            _check("Provider", has_key,
                   f"{provider_display.get(provider, provider)} ({env_var} set)",
                   f"{provider_display.get(provider, provider)} ({env_var} NOT set)")

    # 4. Model
    model = config.get("model", "") or ""
    results["model"] = model
    if not output_json:
        _check("Model", bool(model), model or "configured",
               "Not set (run: capseal init)")

    # 5. Semgrep
    semgrep_path = shutil.which("semgrep")
    semgrep_version = ""
    if semgrep_path:
        try:
            result = subprocess.run(
                ["semgrep", "--version"], capture_output=True, text=True, timeout=10,
            )
            semgrep_version = result.stdout.strip().split("\n")[0]
        except Exception:
            semgrep_version = "installed"
    results["semgrep"] = bool(semgrep_path)
    if not output_json:
        _check("Semgrep", bool(semgrep_path),
               f"installed ({semgrep_version})" if semgrep_version else "installed",
               "NOT installed (run: pip install semgrep)")

    # 6. Risk model
    posteriors_path = capseal_dir / "models" / "beta_posteriors.npz"
    has_model = posteriors_path.exists()
    model_detail = "not trained (run: capseal learn .)"
    if has_model:
        try:
            data = np.load(posteriors_path)
            alpha = data["alpha"]
            beta = data["beta"]
            # Count cells with real data (alpha + beta > 2 means at least one observation)
            episodes = int(np.sum((alpha + beta) > 2))
            model_detail = f"beta_posteriors.npz (trained, {episodes} regions with data)"
        except Exception:
            model_detail = "beta_posteriors.npz (exists, could not read)"
    results["risk_model"] = has_model
    if not output_json:
        _check("Risk model", has_model, model_detail, model_detail)

    # 7. Agent integrations
    agents = config.get("agents", [])
    integrations = config.get("integrations", {})
    if agents and agents != ["cli-only"]:
        if not output_json:
            click.echo()
            click.echo(f"  {BOLD}Agent integrations:{RESET}")

        results["agents"] = {}
        for agent in agents:
            if agent == "cli-only":
                continue

            configured = False
            detail = ""

            if agent == "claude-code":
                # Check if claude mcp list contains capseal
                try:
                    result = subprocess.run(
                        ["claude", "mcp", "list"], capture_output=True, text=True, timeout=10,
                    )
                    configured = "capseal" in result.stdout.lower()
                    detail = "MCP server configured" if configured else "Not configured (run: capseal init)"
                except Exception:
                    detail = "Could not check (claude CLI not found)"

            elif agent == "openclaw":
                skill_dir = Path.home() / ".openclaw" / "workspace" / "skills" / "capseal"
                configured = (skill_dir / "SKILL.md").exists()
                detail = f"Skill installed at {skill_dir}" if configured else "Not configured (run: capseal init)"

            elif agent in ("cursor", "windsurf", "cline"):
                config_paths = {
                    "cursor": Path.home() / ".cursor" / "mcp.json",
                    "windsurf": Path.home() / ".windsurf" / "mcp.json",
                    "cline": Path.home() / ".cline" / "mcp.json",
                }
                mcp_config = config_paths.get(agent)
                if mcp_config and mcp_config.exists():
                    try:
                        data = json.loads(mcp_config.read_text())
                        configured = "capseal" in data.get("mcpServers", {})
                        detail = f"MCP configured in {mcp_config.name}" if configured else f"Not in {mcp_config.name}"
                    except Exception:
                        detail = f"Could not parse {mcp_config}"
                else:
                    detail = "Not configured (run: capseal init)"

            display_names = {
                "claude-code": "Claude Code",
                "openclaw": "OpenClaw",
                "cursor": "Cursor",
                "windsurf": "Windsurf",
                "cline": "Cline",
            }
            results["agents"][agent] = configured
            if not output_json:
                _check(f"  {display_names.get(agent, agent)}", configured, detail, detail)

    # 8. Last session
    latest_cap = capseal_dir / "runs" / "latest.cap"
    if not output_json:
        click.echo()
        click.echo(f"  {BOLD}Last session:{RESET}")

    if latest_cap.exists() or latest_cap.is_symlink():
        cap_path = latest_cap.resolve() if latest_cap.is_symlink() else latest_cap
        cap_name = cap_path.name if cap_path.exists() else "latest.cap"

        results["last_session"] = {"cap": cap_name}
        if not output_json:
            _check("Receipt", True, f".capseal/runs/{cap_name}")

        # Try to verify
        if cap_path.exists():
            try:
                capseal_bin = shutil.which("capseal") or "capseal"
                result = subprocess.run(
                    [capseal_bin, "verify", str(cap_path), "--json"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if line.strip().startswith("{"):
                            try:
                                vdata = json.loads(line.strip())
                                verified = vdata.get("status") == "VERIFIED"
                                results["last_session"]["verified"] = verified
                                actions = vdata.get("num_actions", vdata.get("rounds_verified", "?"))
                                chain_hash = vdata.get("chain_hash", vdata.get("capsule_hash", ""))
                                if not output_json:
                                    _check("Status", verified, "VERIFIED", "REJECTED")
                                    if actions != "?":
                                        click.echo(f"  {'Actions:':<16}{actions}")
                                    if chain_hash:
                                        click.echo(f"  {'Chain:':<16}intact ({chain_hash[:16]}...)")
                                break
                            except json.JSONDecodeError:
                                continue
                else:
                    if not output_json:
                        _check("Status", False, "", "Could not verify")
            except Exception:
                if not output_json:
                    _check("Status", False, "", "Verification error")
    else:
        results["last_session"] = None
        if not output_json:
            click.echo(f"  {'Receipt:':<16}{DIM}No sessions yet{RESET}")

    # Final
    if output_json:
        click.echo(json.dumps(results, indent=2, default=str))
    else:
        click.echo()
        click.echo(f"{CYAN}{'═' * 55}{RESET}")


__all__ = ["doctor_command"]
