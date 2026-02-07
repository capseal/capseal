"""CapSeal CLI - Verified AI Code Changes.

Core Commands:
    init    - Initialize workspace
    scan    - Find issues with Semgrep
    learn   - Build risk model from patch outcomes
    fix     - Generate verified patches, gated by risk model
    review  - Gate findings based on learned model
    verify  - Verify receipts and proofs
    report  - Generate human-readable summary
    watch   - CI integration (JSON output)
    demo    - 30-second interactive demo

Advanced commands available via: capseal advanced <command>
"""
from __future__ import annotations

import click

# Core command imports
from .init_cmd import init_command
from .scan_cmd import scan_command
from .learn_cmd import learn_command
from .fix_cmd import fix_command
from .verify import verify_command
from .report_cmd import report_command
from .watch_cmd import watch_command
from .demo_cmd import demo_command
from .workflow_cmd import verify_capsule_command

# Advanced command imports
from .emit import emit_command
from .inspect_cmd import inspect_command
from .run import run_command
from .replay import replay_command
from .open_cmd import open_command
from .replay_row import replay_row_command
from .audit import audit_command, audit_fetch_command
from .sandbox_cmd import sandbox_group
from .shell import shell_command, run_shell
from .fetch import fetch_command
from .doctor import doctor
from .docs_generator import docs_group
from .pipeline import pipeline_group
from .greptile import greptile_group
from .diff import diff_command, logs_command
from .context import context_group
from .merge import merge_command, conflict_bundle_command, merge_apply_command
from .merge_orchestrate import merge_orchestrate_command
from .explain_cmd import explain_command
from .attest_cmd import attest_command, attest_diff_command
from .profile_cmd import profile_command, conformance_command, init_intent_command
from .workflow_cmd import (
    workflow_command,
    verify_workflow_command,
    workflow_status_command,
    compaction_command,
)
from .refactor_cmd import (
    refactor_command,
    show_diff_command,
    apply_refactor_command,
)
from .eval_cmd import eval_command
from .agent_cmd import agent_group
from .trace_cmd import (
    trace_command,
    verify_trace_command,
    trace_open_command,
    prompt_open_command,
    explain_llm_command,
    verify_explain_command,
    verify_review_command,
    agent_review_command,
    review_orchestrator_command,
    dag_command,
    verify_rollup_command,
    demo_review_command,
    explain_review_command,
    review_diff_command,
    pipeline_command,
)


@click.group()
@click.version_option(version="0.2.0", prog_name="capseal")
def cli() -> None:
    """CapSeal — Verified AI Code Changes

    \b
    Quick start:
      capseal scan .             Find issues
      capseal learn .            Build risk model
      capseal fix . --dry-run    Preview gated patches
      capseal fix .              Generate verified patches
      capseal verify             Verify receipts

    \b
    Run 'capseal advanced --help' for power user commands.
    """


# =============================================================================
# CORE COMMANDS (8 user-facing commands)
# =============================================================================

cli.add_command(init_command, name="init")
cli.add_command(scan_command, name="scan")
cli.add_command(learn_command, name="learn")
cli.add_command(fix_command, name="fix")
cli.add_command(verify_command, name="verify")
cli.add_command(report_command, name="report")
cli.add_command(watch_command, name="watch")
cli.add_command(demo_command, name="demo")


@cli.command("mcp-serve")
@click.option(
    "--workspace", "-w",
    type=click.Path(exists=True, file_okay=False),
    help="Project directory containing .capseal/ (defaults to cwd)"
)
def mcp_serve_command(workspace: str | None) -> None:
    """Start the MCP server for agent integration.

    Exposes CapSeal as an MCP server that any agent framework can call.
    Uses stdio transport (what mcporter and most MCP clients expect).

    \b
    Tools exposed:
        capseal_gate   - Gate a proposed action (returns approve/deny/flag)
        capseal_record - Record an executed action
        capseal_seal   - Seal the session into a .cap receipt

    \b
    Usage:
        # From project directory:
        cd ~/projects/my-project
        capseal mcp-serve

        # Or with explicit workspace:
        capseal mcp-serve --workspace ~/projects/my-project

    \b
    Usage with Claude Code:
        claude mcp add capseal -- capseal mcp-serve -w /path/to/project

    \b
    Test with:
        echo '{"jsonrpc":"2.0","method":"tools/list","id":1}' | capseal mcp-serve
    """
    from ..mcp_server import run_mcp_server
    run_mcp_server(workspace=workspace)


@cli.command("export-skill")
@click.argument("destination", type=click.Path())
def export_skill_command(destination: str) -> None:
    """Export CapSeal as an OpenClaw skill.

    Copies SKILL.md and manifest.json to the destination directory.
    Use this to install CapSeal as a skill in OpenClaw or similar agent frameworks.

    \b
    Example:
        capseal export-skill ~/.openclaw/workspace/skills/capseal

    After export, the skill will be available to OpenClaw agents.
    """
    from pathlib import Path
    from .skill_export import export_skill

    dest = Path(destination)
    export_skill(dest)
    click.echo(f"Skill exported to {dest}")
    click.echo(f"  {dest / 'SKILL.md'}")
    click.echo(f"  {dest / 'manifest.json'}")


@cli.command("review")
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--gate", is_flag=True, help="Gate findings based on learned model")
@click.option("--json", "output_json", is_flag=True, help="Output JSON for CI")
@click.option("--threshold", type=float, default=0.6, help="Failure probability threshold (default: 0.6)")
@click.pass_context
def review_command(ctx: click.Context, path: str, gate: bool, output_json: bool, threshold: float) -> None:
    """Review and gate code changes based on learned risk model.

    \b
    Examples:
        capseal review . --gate       Gate high-risk findings
        capseal review . --threshold 0.5   Custom threshold
    """
    ctx.invoke(scan_command, path=path, gate=gate, output_json=output_json, threshold=threshold)


# =============================================================================
# ADVANCED COMMANDS GROUP
# =============================================================================

@cli.group()
def advanced() -> None:
    """Power user commands — shell, trace, merge, refactor, agent internals.

    \b
    Categories:
      Shell & Interactive:  shell
      Tracing & Proofs:     trace, verify-trace, dag, replay, row, emit, run
      Merge Tools:          merge, merge-apply, merge-orchestrate, conflict-bundle
      Refactor Pipeline:    refactor, show-diff, apply-refactor
      Agent Internals:      agent, workflow, workflow-status
      Integrations:         greptile, context, sandbox
      Diagnostics:          doctor, inspect, audit, explain, profile
    """


# Shell & Interactive
advanced.add_command(shell_command, name="shell")

# Tracing & Proofs
advanced.add_command(trace_command, name="trace")
advanced.add_command(verify_trace_command, name="verify-trace")
advanced.add_command(trace_open_command, name="trace-open")
advanced.add_command(dag_command, name="dag")
advanced.add_command(verify_rollup_command, name="verify-rollup")
advanced.add_command(replay_command, name="replay")
advanced.add_command(replay_row_command, name="row")
advanced.add_command(emit_command, name="emit")
advanced.add_command(run_command, name="run")
advanced.add_command(open_command, name="open")

# Merge Tools
advanced.add_command(merge_command, name="merge")
advanced.add_command(merge_apply_command, name="merge-apply")
advanced.add_command(merge_orchestrate_command, name="merge-orchestrate")
advanced.add_command(conflict_bundle_command, name="conflict-bundle")

# Refactor Pipeline
advanced.add_command(refactor_command, name="refactor")
advanced.add_command(show_diff_command, name="show-diff")
advanced.add_command(apply_refactor_command, name="apply-refactor")

# Agent Internals
advanced.add_command(agent_group, name="agent")
advanced.add_command(workflow_command, name="workflow")
advanced.add_command(verify_workflow_command, name="verify-workflow")
advanced.add_command(verify_capsule_command, name="verify-capsule")
advanced.add_command(workflow_status_command, name="workflow-status")
advanced.add_command(agent_review_command, name="agent-review")
advanced.add_command(review_orchestrator_command, name="review-shards")

# Integrations
advanced.add_command(greptile_group, name="greptile")
advanced.add_command(context_group, name="context")
advanced.add_command(sandbox_group, name="sandbox")
advanced.add_command(docs_group, name="docs")
advanced.add_command(pipeline_group, name="pipeline")

# Diagnostics & Inspection
advanced.add_command(doctor, name="doctor")
advanced.add_command(inspect_command, name="inspect")
advanced.add_command(audit_command, name="audit")
advanced.add_command(audit_fetch_command, name="audit-fetch")
advanced.add_command(explain_command, name="explain")
advanced.add_command(explain_review_command, name="explain-review")
advanced.add_command(explain_llm_command, name="explain-llm")
advanced.add_command(verify_explain_command, name="verify-explain")

# Profile & Conformance
advanced.add_command(profile_command, name="profile")
advanced.add_command(conformance_command, name="conformance")
advanced.add_command(init_intent_command, name="init-intent")

# Misc
advanced.add_command(fetch_command, name="fetch")
advanced.add_command(diff_command, name="diff")
advanced.add_command(logs_command, name="logs")
advanced.add_command(attest_command, name="attest")
advanced.add_command(attest_diff_command, name="attest-diff")
advanced.add_command(compaction_command, name="compaction")
advanced.add_command(prompt_open_command, name="prompt-open")
advanced.add_command(verify_review_command, name="verify-review")
advanced.add_command(demo_review_command, name="demo-review")
advanced.add_command(review_diff_command, name="review-diff")
advanced.add_command(pipeline_command, name="pipeline")
advanced.add_command(eval_command, name="eval")


def main() -> None:
    """CLI entry point.

    If no command is given, launches the interactive shell.
    """
    import sys

    # If no args (or just --help/--version), check if we should launch shell
    if len(sys.argv) == 1:
        # No arguments - launch interactive shell
        run_shell()
    else:
        # Has arguments - run normal CLI
        cli(prog_name="capseal")


if __name__ == "__main__":
    main()
