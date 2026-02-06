"""Test that all expected CLI commands are exposed on the CLI root.

This test prevents the "forgot to wire it" regression where CLI modules exist
but aren't registered on the root app, making them invisible to users.
"""
from __future__ import annotations

import pytest
from click.testing import CliRunner


def test_cli_exposes_expected_commands():
    """Verify all expected subcommands are available on the CLI root."""
    from bef_zk.capsule.cli import cli as capseal_cli

    runner = CliRunner()
    res = runner.invoke(capseal_cli, ["--help"])

    assert res.exit_code == 0, f"CLI help failed: {res.output}"

    # Core workflow commands (must be present)
    core_commands = [
        "init",
        "run",
        "verify",
        "replay",
        "doctor",
        "inspect",
        "explain",
        "audit",
        "emit",
        "open",
    ]

    # Extended commands (must be present)
    extended_commands = [
        "shell",
        "docs",
        "pipeline",
        "greptile",
        "diff",
        "logs",
        "context",
        "merge",
    ]

    # Trace/profile/workflow commands (must be present)
    trace_commands = [
        "trace",
        "profile",
        "workflow",
        "attest",
        "refactor",
    ]

    all_expected = core_commands + extended_commands + trace_commands

    missing = []
    for name in all_expected:
        if name not in res.output:
            missing.append(name)

    assert not missing, f"Missing CLI commands: {missing}\n\nHelp output:\n{res.output}"


def test_shell_passthrough_forwards_cli_commands():
    """Verify the shell's default() method forwards to CLI root."""
    from bef_zk.capsule.cli.shell import CapsealShell

    shell = CapsealShell()

    # The shell should have CLI passthrough infrastructure
    assert hasattr(shell, "_cli_runner"), "Shell missing _cli_runner"
    assert hasattr(shell, "_cli_root"), "Shell missing _cli_root"
    assert hasattr(shell, "_invoke_cli"), "Shell missing _invoke_cli method"

    # The CLI root should be the Click group
    assert shell._cli_root is not None
    assert hasattr(shell._cli_root, "commands"), "CLI root should be a Click group"


def test_cli_command_help_accessible():
    """Verify individual commands have --help accessible."""
    from bef_zk.capsule.cli import cli as capseal_cli

    runner = CliRunner()

    # Test a few important commands have working help
    commands_to_test = ["profile", "workflow", "trace", "attest", "docs"]

    for cmd in commands_to_test:
        res = runner.invoke(capseal_cli, [cmd, "--help"])
        assert res.exit_code == 0, f"{cmd} --help failed: {res.output}"
        assert "Usage:" in res.output, f"{cmd} --help missing Usage section"
