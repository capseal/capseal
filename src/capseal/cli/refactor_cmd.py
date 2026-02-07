"""Refactor CLI commands for CapSeal.

Commands:
- capseal refactor: Run full review → plan → patch → verify pipeline
- capseal refactor-plan: Generate refactor plan from review
- capseal refactor-patches: Generate patches from plan
- capseal verify-patches: Verify patches apply correctly
- capseal diff-rollup: Build final verified diff
"""
from __future__ import annotations

import json
from pathlib import Path

import click

from capseal.workflow_engine import (
    WorkflowSpec,
    WorkflowRunner,
    verify_workflow_rollup,
)


@click.command("refactor")
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--run-dir", type=click.Path(), default=None,
              help="Output directory for run artifacts")
@click.option("--provider", default="openai", help="LLM provider")
@click.option("--model", default="gpt-4o-mini", help="LLM model")
@click.option("--max-workers", default=4, type=int, help="Max parallel agents")
@click.option("--skip-explain", is_flag=True, help="Skip LLM explanation step")
@click.option("--verify/--no-verify", default=True, help="Run verification after")
@click.option("--suppress-memos/--no-suppress-memos", default=True,
              help="v5: Reuse cached proofs for repeated findings (saves tokens)")
@click.option("--ast-validate/--no-ast-validate", default=True,
              help="v5: Verify whitelist claims with AST analysis")
def refactor_command(project_dir, run_dir, provider, model, max_workers, skip_explain, verify,
                     suppress_memos, ast_validate):
    """Run full refactor pipeline: review → plan → patches → verify → diff.

    This command:
    1. Traces the codebase
    2. Runs semgrep review
    3. Generates a refactor plan from findings
    4. Multi-agent patch generation
    5. Verifies all patches apply correctly
    6. Produces verified diff rollup with provenance
    """
    import tempfile
    import datetime

    project_path = Path(project_dir).resolve()

    if run_dir:
        run_path = Path(run_dir)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = project_path / ".capseal" / f"refactor_{timestamp}"

    run_path.mkdir(parents=True, exist_ok=True)

    click.echo(f"=== CapSeal Refactor Pipeline ===")
    click.echo(f"Project: {project_path}")
    click.echo(f"Output:  {run_path}")
    click.echo()

    # Build workflow
    tasks = [
        {"id": "trace", "kind": "trace", "required_mode": "required"},
        {"id": "semgrep", "kind": "review.semgrep", "needs": ["trace"], "required_mode": "required"},
        {"id": "plan", "kind": "refactor.plan", "needs": ["semgrep"], "required_mode": "required",
         "params": {"provider": provider, "model": model}},
        {"id": "patches", "kind": "refactor.patches", "needs": ["plan"], "required_mode": "required",
         "params": {"provider": provider, "model": model, "max_workers": max_workers,
                    "enable_suppression_memos": suppress_memos, "enable_ast_validation": ast_validate}},
        {"id": "verify", "kind": "refactor.verify", "needs": ["patches"], "required_mode": "required"},
        {"id": "rollup", "kind": "refactor.rollup", "needs": ["plan", "patches", "verify"], "required_mode": "required"},
    ]

    if not skip_explain:
        tasks.insert(2, {
            "id": "explain",
            "kind": "explain_llm",
            "needs": ["semgrep"],
            "required_mode": "optional",
            "params": {"provider": provider, "model": model},
        })

    spec = WorkflowSpec.from_dict({
        "name": "refactor_pipeline",
        "description": "Full review → refactor → verify pipeline",
        "tasks": tasks,
    })

    # Run workflow
    click.echo("Running workflow...")
    runner = WorkflowRunner(project_path, run_path)
    rollup = runner.run(spec)

    # Print results
    click.echo()
    click.echo("=== Results ===")
    for node_id, result in runner.results.items():
        status_icon = "✓" if result.status == "PASS" else ("⊘" if result.status == "SKIP" else "✗")
        cached_str = " (cached)" if result.cached else ""
        click.echo(f"  {status_icon} {node_id}: {result.status}{cached_str}")
        if result.error and result.status == "FAIL":
            click.echo(f"      Error: {result.error[:100]}...")

    click.echo()

    # Check refactor results
    rollup_result = runner.results.get("rollup")
    if rollup_result and rollup_result.packet:
        diff_rollup_path = run_path / rollup_result.packet.output_path
        if diff_rollup_path.exists():
            diff_data = json.loads(diff_rollup_path.read_text())

            click.echo("=== Refactor Summary ===")

            # v2: Show status_detail breakdown
            status_detail = diff_data.get("status_detail", {})
            if status_detail:
                total = status_detail.get("total_patches", 0)
                valid = status_detail.get("valid_patches", 0)
                skipped = status_detail.get("skipped_patches", 0)
                failed = status_detail.get("failed_patches", 0)
                repaired = status_detail.get("repaired_patches", 0)

                click.echo(f"  Patch Status:")
                click.echo(f"    Total:    {total}")
                click.echo(f"    ✓ Valid:  {valid}" + (f" ({repaired} repaired)" if repaired else ""))
                click.echo(f"    ⊘ Skip:   {skipped}")
                click.echo(f"    ✗ Failed: {failed}")

                # Show skip reasons if any
                skip_reasons = status_detail.get("skip_reasons", {})
                if skip_reasons:
                    reason_counts = {}
                    for reason in skip_reasons.values():
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    click.echo(f"  Skip reasons: {reason_counts}")

                # Show failed patch IDs
                failed_ids = status_detail.get("failed_patch_ids", [])
                if failed_ids:
                    click.echo(f"  Failed patches: {failed_ids}")

                # v3: Show yield metrics if available
                yield_metrics = status_detail.get("yield_metrics", {})
                if yield_metrics:
                    click.echo()
                    click.echo("  Yield Metrics:")
                    yield_rate = yield_metrics.get("yield_rate", 0)
                    resolution_rate = yield_metrics.get("resolution_rate", 0)
                    click.echo(f"    Yield rate:      {yield_rate:.1%} (plan items → valid patches)")
                    click.echo(f"    Resolution rate: {resolution_rate:.1%} (targeted findings → resolved)")

                    findings_targeted = yield_metrics.get("findings_targeted", 0)
                    findings_resolved = yield_metrics.get("findings_resolved", 0)
                    if findings_targeted > 0:
                        click.echo(f"    Findings:        {findings_resolved}/{findings_targeted} resolved")

                    new_warnings = yield_metrics.get("new_warnings_introduced", 0)
                    if new_warnings > 0:
                        click.echo(f"    ⚠ New warnings:  {new_warnings} introduced")

                    timeouts = yield_metrics.get("timeouts", 0)
                    if timeouts > 0:
                        click.echo(f"    ⏱ Timeouts:      {timeouts}")

                    # v5 metrics
                    no_change_proven = yield_metrics.get("no_change_proven", 0)
                    if no_change_proven > 0:
                        click.echo(f"    📋 NO_CHANGE proven: {no_change_proven}")

                    importability_failures = yield_metrics.get("importability_failures", 0)
                    if importability_failures > 0:
                        click.echo(f"    ⚠ Syntax errors: {importability_failures}")

                click.echo()

            click.echo(f"  Files modified: {diff_data.get('total_files_modified', 0)}")
            click.echo(f"  Lines added:    +{diff_data.get('total_lines_added', 0)}")
            click.echo(f"  Lines removed:  -{diff_data.get('total_lines_removed', 0)}")
            click.echo(f"  All verified:   {'✓' if diff_data.get('all_verified') else '✗'}")

            click.echo()
            click.echo(f"  Diff file:   {run_path / 'refactor' / 'combined.diff'}")
            click.echo(f"  Full rollup: {diff_rollup_path}")

            # Show provenance
            provenance = diff_data.get("provenance", {})
            if provenance:
                click.echo()
                click.echo("=== Provenance Chain ===")
                click.echo(f"  trace_root: {provenance.get('trace_root', '')[:32]}...")
                click.echo(f"  plan_hash:  {provenance.get('plan_hash', '')[:32]}...")
                click.echo(f"  patches:    {len(provenance.get('valid_patch_receipts', provenance.get('patch_receipts', {})))}")

    # Verify if requested
    if verify:
        click.echo()
        click.echo("=== Verification ===")
        workflow_rollup_path = run_path / "workflow" / "workflow_rollup.json"
        if workflow_rollup_path.exists():
            results = verify_workflow_rollup(workflow_rollup_path, run_path)
            for check in results["checks"]:
                click.echo(f"  {check}")
            for error in results["errors"]:
                click.echo(f"  [ERROR] {error}")

            if results["valid"]:
                click.echo()
                click.echo("✓ WORKFLOW VERIFIED")
            else:
                click.echo()
                click.echo("✗ VERIFICATION FAILED")
                raise SystemExit(1)


@click.command("show-diff")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--file", "file_filter", default=None, help="Filter to specific file")
@click.option("--patch-id", default=None, help="Show specific patch")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def show_diff_command(run_dir, file_filter, patch_id, as_json):
    """Show the verified diff from a refactor run.

    Displays the patches with their verification status and provenance.
    """
    run_path = Path(run_dir)

    # Load diff rollup
    rollup_path = run_path / "refactor" / "diff_rollup.json"
    if not rollup_path.exists():
        click.echo("No diff_rollup.json found. Run `capseal refactor` first.", err=True)
        raise SystemExit(1)

    rollup = json.loads(rollup_path.read_text())

    if as_json:
        if file_filter:
            patches = [p for p in rollup["patches"] if file_filter in p.get("file_path", "")]
            click.echo(json.dumps(patches, indent=2))
        elif patch_id:
            patch = next((p for p in rollup["patches"] if p.get("patch_id") == patch_id), None)
            if patch:
                click.echo(json.dumps(patch, indent=2))
            else:
                click.echo(f"Patch {patch_id} not found", err=True)
        else:
            click.echo(json.dumps(rollup, indent=2))
        return

    # Text output
    click.echo("=== Verified Diff Rollup ===")
    click.echo(f"Trace root:   {rollup.get('trace_root', '')[:32]}...")
    click.echo(f"Plan hash:    {rollup.get('plan_hash', '')[:32]}...")
    click.echo(f"All verified: {'✓' if rollup.get('all_verified') else '✗'}")
    click.echo()

    # Build verification lookup
    verifications = {v["patch_id"]: v for v in rollup.get("verifications", [])}

    for patch in rollup.get("patches", []):
        if file_filter and file_filter not in patch.get("file_path", ""):
            continue
        if patch_id and patch.get("patch_id") != patch_id:
            continue

        v = verifications.get(patch.get("patch_id"), {})
        verified = v.get("verified", False)
        status = "✓" if verified else "✗"

        click.echo(f"--- {status} {patch.get('file_path')} ---")
        click.echo(f"Patch ID:      {patch.get('patch_id')}")
        click.echo(f"Item ID:       {patch.get('item_id')}")
        click.echo(f"Agent:         {patch.get('agent_type')}")
        click.echo(f"Original hash: {patch.get('original_hash', '')[:16]}...")
        click.echo(f"Expected hash: {patch.get('expected_hash', '')[:16]}...")
        click.echo(f"Changes:       +{patch.get('lines_added', 0)} -{patch.get('lines_removed', 0)} ({patch.get('hunks', 0)} hunks)")

        if not verified and v.get("error"):
            click.echo(f"Error:         {v.get('error')}")

        click.echo()
        click.echo("```diff")
        click.echo(patch.get("patch_content", ""))
        click.echo("```")
        click.echo()


@click.command("apply-refactor")
@click.argument("run_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("project_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--dry-run", is_flag=True, help="Show what would be applied without applying")
@click.option("--force", is_flag=True, help="Apply even if verification failed")
@click.option("--patch-id", multiple=True, help="Apply only specific patches")
@click.option("--combined", is_flag=True, help="Apply combined.diff as single operation")
def apply_refactor_command(run_dir, project_dir, dry_run, force, patch_id, combined):
    """Apply verified patches from a refactor run to the project.

    Uses git apply when possible (more robust than patch).
    Only applies patches that passed verification unless --force is used.
    """
    import subprocess

    run_path = Path(run_dir)
    project_path = Path(project_dir).resolve()

    # Load diff rollup
    rollup_path = run_path / "refactor" / "diff_rollup.json"
    if not rollup_path.exists():
        click.echo("No diff_rollup.json found. Run `capseal refactor` first.", err=True)
        raise SystemExit(1)

    rollup = json.loads(rollup_path.read_text())

    # v2: Check status_detail for failed patches
    status_detail = rollup.get("status_detail", {})
    has_failures = status_detail.get("failed_patches", 0) > 0 if status_detail else not rollup.get("all_verified")

    if has_failures and not force:
        failed_ids = status_detail.get("failed_patch_ids", rollup.get("failed_patches", []))
        click.echo(f"Some patches failed: {failed_ids}", err=True)
        click.echo("Use --force to apply anyway.", err=True)
        raise SystemExit(1)

    # Apply combined.diff as single operation (recommended)
    if combined:
        combined_path = run_path / "refactor" / "combined.diff"
        if not combined_path.exists():
            click.echo("No combined.diff found.", err=True)
            raise SystemExit(1)

        combined_content = combined_path.read_text()
        if not combined_content.strip():
            click.echo("combined.diff is empty (all patches were SKIPs).")
            return

        if dry_run:
            click.echo("=== DRY RUN: Would apply combined.diff ===")
            click.echo(f"  Files: {rollup.get('total_files_modified', 0)}")
            click.echo(f"  Lines: +{rollup.get('total_lines_added', 0)} -{rollup.get('total_lines_removed', 0)}")

            # Verify with git apply --check
            result = subprocess.run(
                ["git", "apply", "--check", "-"],
                input=combined_content.encode(),
                capture_output=True,
                cwd=project_path,
            )
            if result.returncode == 0:
                click.echo("  git apply --check: ✓ would apply cleanly")
            else:
                click.echo(f"  git apply --check: ✗ {result.stderr.decode()[:200]}")
            return

        # Apply with git apply
        click.echo("Applying combined.diff with git apply...")
        result = subprocess.run(
            ["git", "apply", "-"],
            input=combined_content.encode(),
            capture_output=True,
            cwd=project_path,
        )

        if result.returncode == 0:
            click.echo("✓ Applied combined.diff successfully")
        else:
            # Fallback to patch
            click.echo(f"git apply failed, trying patch: {result.stderr.decode()[:100]}")
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(combined_content)
                patch_file = f.name

            try:
                result = subprocess.run(
                    ["patch", "-p1", "-i", patch_file],
                    capture_output=True,
                    text=True,
                    cwd=project_path,
                )
                if result.returncode == 0:
                    click.echo("✓ Applied with patch fallback")
                else:
                    click.echo(f"✗ Failed: {result.stderr}")
            finally:
                Path(patch_file).unlink()
        return

    # Individual patch application
    verifications = {v["patch_id"]: v for v in rollup.get("verifications", [])}
    valid_patch_ids = status_detail.get("valid_patch_ids", []) if status_detail else None

    patches_to_apply = []
    for patch in rollup.get("patches", []):
        pid = patch.get("patch_id")

        # Filter by patch_id if specified
        if patch_id and pid not in patch_id:
            continue

        # v2: Use status_detail to filter
        if valid_patch_ids is not None:
            if pid not in valid_patch_ids and not force:
                click.echo(f"Skipping non-valid patch: {pid}")
                continue
        else:
            v = verifications.get(pid, {})
            if not v.get("verified") and not force:
                click.echo(f"Skipping unverified patch: {pid}")
                continue

        patches_to_apply.append(patch)

    if not patches_to_apply:
        click.echo("No patches to apply.")
        return

    click.echo(f"Applying {len(patches_to_apply)} patches...")

    for patch in patches_to_apply:
        file_path = project_path / patch.get("file_path")
        patch_content = patch.get("patch_content", "")

        if not patch_content.strip():
            click.echo(f"  ⊘ Skip (no-op): {patch.get('file_path')}")
            continue

        if dry_run:
            click.echo(f"  [DRY RUN] Would apply to: {patch.get('file_path')}")
            continue

        # Check original hash
        if file_path.exists():
            from capseal.workflow_engine import sha256_str
            current_hash = sha256_str(file_path.read_text())
            if current_hash != patch.get("original_hash"):
                click.echo(f"  [SKIP] {patch.get('file_path')}: file has changed since patch was generated")
                continue

        # Apply patch with git apply
        result = subprocess.run(
            ["git", "apply", "-"],
            input=patch_content.encode(),
            capture_output=True,
            cwd=project_path,
        )

        if result.returncode == 0:
            click.echo(f"  ✓ Applied: {patch.get('file_path')}")
        else:
            # Fallback to patch command
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(patch_content)
                patch_file = f.name

            try:
                result = subprocess.run(
                    ["patch", "-p1", "-i", patch_file],
                    capture_output=True,
                    text=True,
                    cwd=project_path,
                )
                if result.returncode == 0:
                    click.echo(f"  ✓ Applied (patch fallback): {patch.get('file_path')}")
                else:
                    click.echo(f"  ✗ Failed: {patch.get('file_path')}: {result.stderr}")
            finally:
                Path(patch_file).unlink()

    if dry_run:
        click.echo()
        click.echo("Dry run complete. Use without --dry-run to apply.")


# Export commands
refactor_commands = [
    refactor_command,
    show_diff_command,
    apply_refactor_command,
]
