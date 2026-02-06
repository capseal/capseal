"""Capsule CLI - product-grade interface for generating and verifying receipts.

Commands:
    init    - Initialize workspace
    run     - Generate a cryptographic proof (simplified pipeline wrapper)
    verify  - Verify a capsule with stable exit codes for CI
    replay  - Semantic replay verification
    audit   - Export and inspect audit trails
    emit    - Generate a portable verification artifact (.cap file)
    inspect - Display capsule metadata
    explain - Human-readable verification report
    demo    - Self-contained offline demo
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

from .emit import emit_command
from .verify import verify_command
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
from .init_cmd import init_command
from .demo_cmd import demo_command
from .explain_cmd import explain_command
from .attest_cmd import attest_command, attest_diff_command
from .profile_cmd import profile_command, conformance_command, init_intent_command
from .workflow_cmd import (
    workflow_command,
    verify_workflow_command,
    verify_capsule_command,
    workflow_status_command,
    compaction_command,
)
from .refactor_cmd import (
    refactor_command,
    show_diff_command,
    apply_refactor_command,
)
from .eval_cmd import eval_command
from .watch_cmd import watch_command
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
@click.version_option(version="0.2.0", prog_name="capsule")
def cli() -> None:
    """Capsule CLI - cryptographic receipt generation and verification.

    Generate portable verification artifacts from traces and verify them
    with stable exit codes suitable for CI integration.

    \b
    Quick Start:
        capsule run -p policy.json --policy-id demo -d ./data
        capsule verify out/capsule_run/strategy_capsule.json
        capsule audit out/capsule_run/strategy_capsule.json
    """


# Core workflow commands (stable contract)
cli.add_command(init_command, name="init")
cli.add_command(fetch_command, name="fetch")
cli.add_command(run_command, name="run")
cli.add_command(verify_command, name="verify")
cli.add_command(replay_command, name="replay")
cli.add_command(doctor, name="doctor")
cli.add_command(demo_command, name="demo")
cli.add_command(inspect_command, name="inspect")
cli.add_command(explain_command, name="explain")

# Artifact commands
cli.add_command(emit_command, name="emit")
cli.add_command(audit_command, name="audit")
cli.add_command(audit_fetch_command, name="audit-fetch")
cli.add_command(open_command, name="open")
cli.add_command(replay_row_command, name="row")

# Extended commands
cli.add_command(sandbox_group, name="sandbox")
cli.add_command(shell_command, name="shell")
cli.add_command(docs_group, name="docs")
cli.add_command(pipeline_group, name="pipeline")
cli.add_command(greptile_group, name="greptile")
cli.add_command(diff_command, name="diff")
cli.add_command(logs_command, name="logs")
cli.add_command(context_group, name="context")
cli.add_command(merge_command, name="merge")
cli.add_command(conflict_bundle_command, name="conflict-bundle")
cli.add_command(merge_apply_command, name="merge-apply")
cli.add_command(merge_orchestrate_command, name="merge-orchestrate")

# Project trace commands
cli.add_command(trace_command, name="trace")
cli.add_command(verify_trace_command, name="verify-trace")
cli.add_command(trace_open_command, name="trace-open")
cli.add_command(prompt_open_command, name="prompt-open")
cli.add_command(explain_llm_command, name="explain-llm")
cli.add_command(verify_explain_command, name="verify-explain")
cli.add_command(attest_command, name="attest")
cli.add_command(attest_diff_command, name="attest-diff")
cli.add_command(verify_review_command, name="verify-review")
cli.add_command(agent_review_command, name="agent-review")
cli.add_command(review_orchestrator_command, name="review")
cli.add_command(dag_command, name="dag")
cli.add_command(verify_rollup_command, name="verify-rollup")
cli.add_command(demo_review_command, name="demo-review")
cli.add_command(explain_review_command, name="explain-review")
cli.add_command(review_diff_command, name="review-diff")
cli.add_command(pipeline_command, name="pipeline")

# Profile and conformance commands
cli.add_command(profile_command, name="profile")
cli.add_command(conformance_command, name="conformance")
cli.add_command(init_intent_command, name="init-intent")

# Workflow DAG commands
cli.add_command(workflow_command, name="workflow")
cli.add_command(verify_workflow_command, name="verify-workflow")
cli.add_command(verify_capsule_command, name="verify-capsule")
cli.add_command(workflow_status_command, name="workflow-status")
cli.add_command(compaction_command, name="compaction")

# Refactor commands
cli.add_command(refactor_command, name="refactor")
cli.add_command(show_diff_command, name="show-diff")
cli.add_command(apply_refactor_command, name="apply-refactor")

# Eval command (risk learning loop)
cli.add_command(eval_command, name="eval")

# CI integration
cli.add_command(watch_command, name="watch")


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
        cli(prog_name="capsule")


if __name__ == "__main__":
    main()
