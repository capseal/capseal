"""Sandbox status and management commands."""
from __future__ import annotations

import json
import sys

import click

from .redact import redact_secrets


@click.group("sandbox")
def sandbox_group():
    """Manage capseal sandbox execution environment."""
    pass


@sandbox_group.command("status")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
def status_command(json_output: bool) -> None:
    """Check sandbox availability and configuration."""
    from bef_zk.sandbox.detect import get_sandbox_info

    info = get_sandbox_info()

    if json_output:
        click.echo(json.dumps(info, indent=2))
    else:
        click.echo("Capseal Sandbox Status")
        click.echo("=" * 40)
        click.echo(f"System: {info['system']}")
        click.echo(f"Sandbox available: {'Yes' if info['sandbox_available'] else 'No'}")
        click.echo(f"Active backend: {info['detected_backend']}")
        click.echo()
        click.echo("Available backends:")
        for name, available in info["available"].items():
            status = "✓" if available else "✗"
            click.echo(f"  {status} {name}")

        if not info["sandbox_available"]:
            click.echo()
            click.echo("To enable sandboxing:")
            if info["system"] == "Linux":
                click.echo("  sudo apt install bubblewrap  # Debian/Ubuntu")
                click.echo("  sudo dnf install bubblewrap  # Fedora")
                click.echo("  sudo pacman -S bubblewrap    # Arch")
            elif info["system"] == "Darwin":
                click.echo("  sandbox-exec is built into macOS")
                click.echo("  (may require disabling SIP for full functionality)")


@sandbox_group.command("test")
@click.option("--backend", type=str, default=None, help="Force specific backend")
def test_command(backend: str | None) -> None:
    """Run a quick sandbox test to verify it works."""
    from bef_zk.sandbox import SandboxRunner, SandboxConfig, SandboxBackend

    click.echo("Testing sandbox...")

    config = SandboxConfig(
        memory_mb=512,
        wall_time_sec=30,
        network=False,
    )

    if backend:
        try:
            config.backend = SandboxBackend(backend)
        except ValueError:
            click.echo(f"Unknown backend: {backend}", err=True)
            sys.exit(1)

    runner = SandboxRunner(config)

    # Run a simple Python command
    result = runner.run([
        sys.executable, "-c",
        "import os; print('Sandbox OK'); print(f'PID: {os.getpid()}')"
    ])

    click.echo(f"Backend: {result.sandbox_backend}")
    click.echo(f"Exit code: {result.returncode}")
    click.echo(f"Output: {result.stdout.strip()}")

    if result.stderr:
        click.echo(f"Stderr: {redact_secrets(result.stderr.strip())}")

    # Report isolation guarantees
    isolation = result.resource_usage.get("isolation", {})
    if isolation:
            click.echo()
            click.echo("Isolation guarantees:")
            if isolation.get("network_degraded"):
                click.echo("  ⚠ network: DEGRADED (shared with host)")
            elif isolation.get("network"):
                click.echo("  ✓ network: isolated")
            else:
                click.echo("  - network: shared (requested)")
            click.echo(f"  {'✓' if isolation.get('pid_namespace') else '-'} pid_namespace: {'isolated' if isolation.get('pid_namespace') else 'shared'}")
            click.echo(f"  {'✓' if isolation.get('ipc_namespace') else '-'} ipc_namespace: {'isolated' if isolation.get('ipc_namespace') else 'shared'}")
            click.echo(f"  {'✓' if isolation.get('uts_namespace') else '-'} uts_namespace: {'isolated' if isolation.get('uts_namespace') else 'shared'}")
            if isolation.get('filesystem'):
                fs_desc = 'restricted (pivot)' if isolation.get('pivot_root') else 'restricted'
                click.echo(f"  ✓ filesystem: {fs_desc}")
            else:
                click.echo("  - filesystem: full")
            click.echo(f"  {'✓' if isolation.get('memory_limit') else '-'} memory_limit: {'enforced' if isolation.get('memory_limit') else 'none'}")
            click.echo(f"  {'✓' if isolation.get('cpu_limit') else '-'} cpu_limit: {'enforced' if isolation.get('cpu_limit') else 'none'}")
            click.echo(f"  {'✓' if isolation.get('wall_timeout') else '-'} wall_timeout: {'enforced' if isolation.get('wall_timeout') else 'none'}")
            if isolation.get('fallback_from'):
                click.echo(f"  ⚠ fallback: used {isolation['fallback_from']} backend")

    if result.returncode == 0 and "Sandbox OK" in result.stdout:
        click.echo()
        click.echo("✓ Sandbox test passed!")
    else:
        click.echo()
        click.echo("✗ Sandbox test failed")
        sys.exit(1)


@sandbox_group.command("install")
@click.option("--dry-run", is_flag=True, help="Show commands without running")
def install_command(dry_run: bool) -> None:
    """Install sandbox dependencies for this system."""
    import platform
    import shutil
    import subprocess

    system = platform.system()

    if system == "Linux":
        # Detect package manager
        if shutil.which("apt"):
            cmd = ["sudo", "apt", "install", "-y", "bubblewrap"]
        elif shutil.which("dnf"):
            cmd = ["sudo", "dnf", "install", "-y", "bubblewrap"]
        elif shutil.which("pacman"):
            cmd = ["sudo", "pacman", "-S", "--noconfirm", "bubblewrap"]
        elif shutil.which("zypper"):
            cmd = ["sudo", "zypper", "install", "-y", "bubblewrap"]
        else:
            click.echo("Could not detect package manager. Install bubblewrap manually.", err=True)
            sys.exit(1)

        click.echo(f"Installing bubblewrap: {' '.join(cmd)}")
        if not dry_run:
            subprocess.run(cmd, check=True)
            click.echo("✓ Installation complete")
        else:
            click.echo("(dry run - not executing)")

    elif system == "Darwin":
        click.echo("macOS uses built-in sandbox-exec.")
        click.echo("No installation required.")

    else:
        click.echo(f"Unsupported system: {system}", err=True)
        sys.exit(1)


__all__ = ["sandbox_group"]
