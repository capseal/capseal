"""CLI commands for workflow execution.

This module provides commands to run, verify, and manage DAG-based workflows
where every agent task emits a verifiable packet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click


@click.command("workflow")
@click.argument("workflow_file", type=click.Path(exists=True))
@click.option("--project-dir", "-p", type=click.Path(exists=True), required=True, help="Project directory")
@click.option("--run", "-r", type=click.Path(), required=True, help="Output run directory")
@click.option("--cache-dir", type=click.Path(), help="Cache directory for memoization")
@click.option("--no-cache", is_flag=True, help="Disable memoization")
@click.option("--prove", is_flag=True, help="Generate cryptographic proof over workflow execution")
def workflow_command(
    workflow_file: str,
    project_dir: str,
    run: str,
    cache_dir: str | None,
    no_cache: bool,
    prove: bool,
) -> None:
    """Execute a workflow DAG with verifiable agent packets.

    Each node in the workflow emits a receipt-carrying artifact.
    The workflow rollup commits to the entire DAG structure + all receipts.

    \b
    Example workflow file (YAML):
        name: review_pipeline
        tasks:
          - id: trace
            kind: trace
          - id: profile
            kind: agent.profile_extract
            needs: [trace]
          - id: semgrep
            kind: review.semgrep
            needs: [trace]
          - id: explain
            kind: explain_llm
            needs: [semgrep]
    """
    from bef_zk.capsule.workflow_engine import WorkflowSpec, WorkflowRunner

    workflow_path = Path(workflow_file)
    project_path = Path(project_dir).resolve()
    run_dir = Path(run)

    # Load workflow
    if workflow_path.suffix in (".yml", ".yaml"):
        spec = WorkflowSpec.from_yaml(workflow_path)
    else:
        spec = WorkflowSpec.from_dict(json.loads(workflow_path.read_text()))

    click.echo(f"Workflow: {spec.name}")
    click.echo(f"  Tasks: {len(spec.tasks)}")
    click.echo(f"  Order: {' -> '.join(spec.topological_order())}")
    click.echo()

    # Create runner
    cache_path = None if no_cache else (Path(cache_dir) if cache_dir else None)
    runner = WorkflowRunner(project_path, run_dir, cache_path, proof_carrying=prove)

    # Execute
    if prove:
        click.echo("Executing workflow with proof generation...")
    else:
        click.echo("Executing workflow...")
    rollup = runner.run(spec)

    # Report results
    click.echo()
    click.echo("Results:")
    for node_id, result in runner.results.items():
        if result.success:
            status = click.style("PASS", fg="green")
            if result.cached:
                status += click.style(" (cached)", fg="cyan")
        elif result.skipped:
            status = click.style("SKIP", fg="yellow")
        else:
            status = click.style("FAIL", fg="red")

        click.echo(f"  [{status}] {node_id}")
        if result.error:
            click.echo(f"        {result.error}")

    click.echo()
    click.echo(f"Rollup: {run_dir / 'workflow' / 'workflow_rollup.json'}")
    click.echo(f"  merkle_root: {rollup['merkle_root'][:32]}...")
    click.echo(f"  rollup_hash: {rollup['rollup_hash'][:32]}...")

    # Show proof info if generated
    if prove and rollup.get("proof_verified") is not None:
        click.echo()
        click.echo("Proof:")
        click.echo(f"  capsule_hash: {rollup.get('capsule_hash', '')[:32]}...")
        click.echo(f"  statement_hash: {rollup.get('statement_hash', '')[:32]}...")
        click.echo(f"  row_commitment: {rollup.get('row_commitment', '')[:32]}...")
        click.echo(f"  proof_backend: {rollup.get('proof_backend', 'unknown')}")
        if rollup.get("proof_verified"):
            click.echo(f"  proof_verified: {click.style('PASS', fg='green')}")
        else:
            click.echo(f"  proof_verified: {click.style('FAIL', fg='red')}")
        click.echo(f"  capsule: {run_dir / 'workflow_capsule.json'}")

    # Check if all passed
    all_passed = all(r.success or r.skipped for r in runner.results.values())
    if all_passed:
        click.echo()
        click.echo(click.style("WORKFLOW COMPLETE", fg="green", bold=True))
    else:
        click.echo()
        click.echo(click.style("WORKFLOW FAILED", fg="red", bold=True))
        sys.exit(1)


@click.command("verify-workflow")
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--verify-proof", is_flag=True, help="Also verify the workflow capsule proof")
def verify_workflow_command(run_dir: str, verify_proof: bool) -> None:
    """Verify a workflow rollup.

    Checks all vertex receipts and merkle root consistency.
    With --verify-proof, also verifies the cryptographic proof in workflow_capsule.json.
    """
    from bef_zk.capsule.workflow_engine import verify_workflow_rollup

    run_path = Path(run_dir)
    rollup_path = run_path / "workflow" / "workflow_rollup.json"

    if not rollup_path.exists():
        click.echo(f"No workflow rollup found at {rollup_path}")
        sys.exit(1)

    click.echo(f"Verifying: {rollup_path}")
    results = verify_workflow_rollup(rollup_path, run_path)

    for check in results["checks"]:
        click.echo(f"  {check}")

    for error in results["errors"]:
        click.echo(click.style(f"  [FAIL] {error}", fg="red"))

    # Verify capsule if requested
    capsule_valid = True
    if verify_proof:
        from bef_zk.capsule.workflow_adapter import verify_workflow_capsule

        capsule_path = run_path / "workflow_capsule.json"
        if capsule_path.exists():
            click.echo()
            click.echo(f"Verifying capsule: {capsule_path}")
            capsule_valid, capsule_details = verify_workflow_capsule(capsule_path)
            for check in capsule_details.get("checks", []):
                click.echo(f"  [PASS] {check}")
            if capsule_valid:
                click.echo(f"  dag_root: {capsule_details.get('dag_root', 'n/a')}")
                click.echo(f"  num_nodes: {capsule_details.get('num_nodes', 0)}")
            else:
                click.echo(click.style(f"  [FAIL] {capsule_details.get('error', 'unknown error')}", fg="red"))
        else:
            click.echo()
            click.echo(click.style(f"  [WARN] No workflow_capsule.json found (run with --prove to generate)", fg="yellow"))
            capsule_valid = True  # Don't fail just because there's no capsule

    click.echo()
    if results["valid"] and capsule_valid:
        click.echo(click.style("WORKFLOW VERIFIED", fg="green", bold=True))
    else:
        click.echo(click.style("VERIFICATION FAILED", fg="red", bold=True))
        sys.exit(1)


@click.command("workflow-status")
@click.argument("run_dir", type=click.Path(exists=True))
def workflow_status_command(run_dir: str) -> None:
    """Show workflow execution status and node receipts."""
    run_path = Path(run_dir)

    # Load rollup
    rollup_path = run_path / "workflow" / "workflow_rollup.json"
    if not rollup_path.exists():
        click.echo("No workflow rollup found")
        sys.exit(1)

    rollup = json.loads(rollup_path.read_text())

    click.echo(f"Workflow: {rollup.get('workflow_name', 'unknown')}")
    click.echo(f"  rollup_hash: {rollup['rollup_hash'][:32]}...")
    click.echo(f"  merkle_root: {rollup['merkle_root'][:32]}...")
    if rollup.get("trace_root"):
        click.echo(f"  trace_root:  {rollup['trace_root'][:32]}...")
    click.echo()

    # Show vertices
    click.echo("Nodes:")
    for node_id, vertex in rollup.get("vertices", {}).items():
        status = vertex.get("status", "unknown")
        reason = vertex.get("reason")
        required = vertex.get("required", True)
        required_label = "required" if required else "optional"

        if "receipt_hash" in vertex:
            cached = " (cached)" if vertex.get("cached") else ""
            click.echo(f"  {node_id}: {status} [{required_label}]")
            click.echo(f"    type: {vertex.get('agent_type', 'unknown')}")
            click.echo(f"    receipt: {vertex['receipt_hash'][:16]}...{cached}")
            click.echo(f"    output:  {vertex.get('output_hash', 'n/a')[:16]}...")
        else:
            click.echo(f"  {node_id}: {status} [{required_label}]")
            if vertex.get("error"):
                click.echo(f"    error: {vertex['error']}")
        if reason and status != "PASS":
            click.echo(f"    reason: {reason}")

    # Show edges
    click.echo()
    click.echo("Dependencies:")
    for edge in rollup.get("edges", []):
        click.echo(f"  {edge['from']} -> {edge['to']}")


@click.command("verify-capsule")
@click.argument("capsule_file", type=click.Path(exists=True))
def verify_capsule_command(capsule_file: str) -> None:
    """Verify a workflow or eval capsule proof.

    Checks the capsule structure, constraint satisfaction, and hash integrity.
    Automatically detects the capsule type (workflow_capsule_v1 or eval_capsule_v1).
    """
    capsule_path = Path(capsule_file)

    # Load capsule to detect schema
    capsule = json.loads(capsule_path.read_text())
    schema = capsule.get("schema", "")

    click.echo(f"Verifying: {capsule_path}")
    click.echo(f"  schema: {schema}")
    click.echo()

    if schema == "workflow_capsule_v1":
        from bef_zk.capsule.workflow_adapter import verify_workflow_capsule
        valid, details = verify_workflow_capsule(capsule_path)

        for check in details.get("checks", []):
            click.echo(f"  [PASS] {check}")

        if valid:
            click.echo()
            click.echo(f"  dag_root: {details.get('dag_root', 'n/a')}")
            click.echo(f"  num_nodes: {details.get('num_nodes', 0)}")
            click.echo()
            click.echo(click.style("CAPSULE VERIFIED", fg="green", bold=True))
        else:
            click.echo(click.style(f"  [FAIL] {details.get('error', 'unknown error')}", fg="red"))
            click.echo()
            click.echo(click.style("VERIFICATION FAILED", fg="red", bold=True))
            sys.exit(1)

    elif schema == "eval_capsule_v1":
        from bef_zk.capsule.eval_adapter import verify_eval_capsule
        valid, details = verify_eval_capsule(capsule_path)

        for check in details.get("checks", []):
            click.echo(f"  [PASS] {check}")

        if valid:
            click.echo()
            click.echo(f"  final_posteriors: {details.get('final_posteriors', 'n/a')}")
            click.echo(f"  num_rounds: {details.get('num_rounds', 0)}")
            click.echo()
            click.echo(click.style("CAPSULE VERIFIED", fg="green", bold=True))
        else:
            click.echo(click.style(f"  [FAIL] {details.get('error', 'unknown error')}", fg="red"))
            click.echo()
            click.echo(click.style("VERIFICATION FAILED", fg="red", bold=True))
            sys.exit(1)

    elif schema == "agent_capsule_v1":
        from bef_zk.capsule.agent_adapter import verify_agent_capsule
        valid, details = verify_agent_capsule(capsule_path)

        for check in details.get("checks", []):
            click.echo(f"  [PASS] {check}")

        if valid:
            click.echo()
            click.echo(f"  final_receipt: {details.get('final_receipt', 'n/a')[:32]}...")
            click.echo(f"  num_actions: {details.get('num_actions', 0)}")
            click.echo()
            click.echo(click.style("CAPSULE VERIFIED", fg="green", bold=True))
        else:
            click.echo(click.style(f"  [FAIL] {details.get('error', 'unknown error')}", fg="red"))
            click.echo()
            click.echo(click.style("VERIFICATION FAILED", fg="red", bold=True))
            sys.exit(1)

    elif schema == "run_receipt_v1":
        # Use the verify_run_receipt function from shared.receipts
        from bef_zk.shared.receipts import verify_run_receipt

        run_dir = capsule_path.parent
        result = verify_run_receipt(run_dir)

        if result["verified"]:
            click.echo(f"  [PASS] Chain hash verified")
            click.echo(f"  [PASS] {result['rounds_verified']} rounds verified")
            click.echo()
            click.echo(f"  chain_hash: {capsule.get('chain_hash', 'n/a')[:32]}...")
            click.echo(f"  total_rounds: {capsule.get('total_rounds', 0)}")
            click.echo()
            click.echo(click.style("RECEIPT VERIFIED", fg="green", bold=True))
        else:
            for mismatch in result.get("mismatches", []):
                click.echo(click.style(f"  [FAIL] {mismatch}", fg="red"))
            click.echo()
            click.echo(click.style("VERIFICATION FAILED", fg="red", bold=True))
            sys.exit(1)

    else:
        click.echo(click.style(f"  [FAIL] Unknown capsule schema: {schema}", fg="red"))
        click.echo(f"  Supported schemas: workflow_capsule_v1, eval_capsule_v1, agent_capsule_v1, run_receipt_v1")
        click.echo()
        click.echo(click.style("VERIFICATION FAILED", fg="red", bold=True))
        sys.exit(1)


@click.command("compaction")
@click.argument("run_dir", type=click.Path(exists=True))
@click.option("--out", "-o", type=click.Path(), help="Output path for compaction.json")
def compaction_command(run_dir: str, out: str | None) -> None:
    """Build compaction (state_summary + evidence_index) from workflow results.

    This creates a fixed-schema summary that can be carried in-context,
    plus evidence pointers for retrieval.
    """
    from bef_zk.capsule.workflow_engine import (
        build_compaction, NodeResult, AgentPacket
    )

    run_path = Path(run_dir)

    # Load workflow rollup
    rollup_path = run_path / "workflow" / "workflow_rollup.json"
    if not rollup_path.exists():
        click.echo("No workflow rollup found")
        sys.exit(1)

    rollup = json.loads(rollup_path.read_text())

    # Reconstruct node results from receipts
    node_results = {}
    for node_id, vertex in rollup.get("vertices", {}).items():
        receipt_hash = vertex.get("receipt_hash")
        if receipt_hash:
            receipt_path = run_path / "receipts" / f"{node_id}_receipt.json"
            if receipt_path.exists():
                packet = AgentPacket.from_dict(json.loads(receipt_path.read_text()))
                node_results[node_id] = NodeResult(node_id=node_id, success=True, packet=packet)

    # Load findings if available
    findings = []
    aggregate_path = run_path / "reviews" / "aggregate.json"
    if aggregate_path.exists():
        aggregate = json.loads(aggregate_path.read_text())
        findings = aggregate.get("findings", [])

    # Determine gate status
    gate_status = "UNKNOWN"
    diff_receipt_path = run_path / "diff" / "receipt.json"
    if diff_receipt_path.exists():
        diff_receipt = json.loads(diff_receipt_path.read_text())
        gate_status = "PASSED" if diff_receipt.get("gate_pass") else "FAILED"
    elif findings:
        # No diff, just based on findings
        errors = sum(1 for f in findings if f.get("severity") == "error")
        gate_status = "PASSED" if errors == 0 else "FAILED"

    # Build compaction
    compaction = build_compaction(node_results, findings, gate_status, run_path)

    # Write output
    if out:
        out_path = Path(out)
    else:
        out_path = run_path / "workflow" / "compaction.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(compaction.to_dict(), indent=2))

    # Display summary
    click.echo(f"Compaction: {out_path}")
    click.echo()
    click.echo("State Summary (what goes in-context):")
    summary = compaction.state_summary
    click.echo(f"  gate_status: {summary.get('gate_status')}")
    click.echo(f"  total_findings: {summary.get('total_findings')}")
    click.echo(f"  by_severity: {summary.get('findings_by_severity')}")
    click.echo()
    click.echo("  Top risks:")
    for risk in summary.get("top_risks", [])[:5]:
        click.echo(f"    - [{risk['severity'].upper()}] {risk['rule']}: {risk['count']} occurrences")
    click.echo()
    click.echo(f"Evidence Index: {len(compaction.evidence_index)} entries")
    click.echo(f"  (Use evidence_index to retrieve details by hash)")
