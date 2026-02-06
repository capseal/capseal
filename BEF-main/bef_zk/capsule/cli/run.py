"""Run command - simplified proof generation wrapper with optional sandboxing."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

import click

from .exit_codes import EXIT_VERIFIED, EXIT_PROOF_INVALID, EXIT_MALFORMED
from .redact import redact_secrets
from .security_assumptions import (
    build_security_assumptions,
    dataset_info_from_capsule,
    policy_info_from_capsule,
    print_security_assumptions,
)
from .utils import find_repo_root
from bef_zk.capsule.bicep_v2_executor import emit_bicep_v2_run


def _find_project_root() -> Path:
    """Find the BEF project root. Delegates to central find_repo_root()."""
    return find_repo_root()


def _is_remote_resource(spec: str | None) -> bool:
    if not spec:
        return False
    lowered = spec.strip().lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


@click.command("run")
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: out/<trace-id>)",
)
@click.option(
    "--trace-id", "-t",
    type=str,
    default="capsule_run",
    help="Trace identifier",
)
@click.option(
    "--policy", "-p",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Policy file path",
)
@click.option(
    "--policy-id",
    type=str,
    required=True,
    help="Policy identifier",
)
@click.option(
    "--dataset", "-d",
    type=str,
    multiple=True,
    help="Dataset path: <path> or <id>=<path> (repeatable)",
)
@click.option(
    "--steps",
    type=int,
    default=64,
    help="Number of trace steps (default: 64)",
)
@click.option(
    "--queries",
    type=int,
    default=8,
    help="Number of FRI queries (default: 8)",
)
@click.option(
    "--challenges",
    type=int,
    default=2,
    help="Number of challenges (default: 2)",
)
@click.option(
    "--backend",
    type=click.Choice(["geom", "risc0"]),
    default="geom",
    help="Proof backend (default: geom)",
)
@click.option(
    "--json", "json_output",
    is_flag=True,
    help="Output JSON summary",
)
@click.option(
    "--sandbox/--no-sandbox",
    default=False,
    help="Run in isolated sandbox (requires bwrap/firejail on Linux or macOS)",
)
@click.option(
    "--sandbox-memory",
    type=int,
    default=4096,
    help="Sandbox memory limit in MB (default: 4096)",
)
@click.option(
    "--sandbox-timeout",
    type=int,
    default=600,
    help="Sandbox wall clock timeout in seconds (default: 600)",
)
@click.option(
    "--sandbox-allow-network/--sandbox-deny-network",
    default=False,
    help="Allow sandboxed runs to access the network (default: deny)",
)
@click.option(
    "--profile",
    type=click.Choice(["default", "train", "eval"]),
    default="default",
    help="Execution profile tag recorded in the capsule",
)
@click.option(
    "--transition-spec",
    type=click.Choice(["bicep_v1", "bicep_v2", "keyed_hash_v1", "identity_v1"]),
    default="bicep_v1",
    help="Transition spec identifier (controls artifact emission pathways)",
)
@click.option(
    "--emit-bicep-v2/--no-emit-bicep-v2",
    default=True,
    help="Emit canonical bicep_v2 run_dir artifacts alongside the capsule run",
)
@click.option(
    "--bicep-run-dir",
    type=click.Path(path_type=Path),
    help="Destination for the emitted bicep_v2 run directory (default: <output>/bicep_v2_run)",
)
@click.option(
    "--bicep-seed",
    type=str,
    default="cafebabe" * 8,
    help="Seed hex for the bicep_v2 run emitter",
)
@click.option(
    "--bicep-steps",
    type=int,
    default=10,
    help="Number of steps for the emitted bicep_v2 run",
)
@click.option(
    "--bicep-paths",
    type=int,
    default=4,
    help="Number of paths in the emitted bicep_v2 run",
)
@click.option(
    "--bicep-channels",
    type=int,
    default=4,
    help="Number of channels in the emitted bicep_v2 run",
)
@click.option(
    "--bicep-audit-k",
    type=int,
    default=8,
    help="Audit sample size k for the emitted bicep_v2 run",
)
@click.option(
    "--verify-bicep/--no-verify-bicep",
    default=True,
    help="Automatically verify the emitted bicep_v2 run via the independent verifier",
)
def run_command(
    output: Optional[Path],
    trace_id: str,
    policy: Path,
    policy_id: str,
    dataset: tuple[str, ...],
    steps: int,
    queries: int,
    challenges: int,
    backend: str,
    json_output: bool,
    sandbox: bool,
    sandbox_memory: int,
    sandbox_timeout: int,
    sandbox_allow_network: bool,
    profile: str,
    transition_spec: str,
    emit_bicep_v2: bool,
    bicep_run_dir: Optional[Path],
    bicep_seed: str,
    bicep_steps: int,
    bicep_paths: int,
    bicep_channels: int,
    bicep_audit_k: int,
    verify_bicep: bool,
) -> None:
    """Generate a cryptographic proof from a trace.

    Simple wrapper around the pipeline for common use cases.

    Examples:

        # Basic run
        capsule run -p policy.json --policy-id demo

        # With datasets
        capsule run -p policy.json --policy-id demo -d ./data -d archive=./archive

        # Custom output
        capsule run -p policy.json --policy-id demo -o ./my_output -t my_trace
    """
    root = _find_project_root()
    script = root / "scripts" / "run_pipeline.py"

    if not script.exists():
        click.echo(f"Error: Pipeline script not found at {script}", err=True)
        sys.exit(EXIT_MALFORMED)

    # Load policy metadata for automatic version wiring
    try:
        policy_data = json.loads(policy.read_text())
    except json.JSONDecodeError as exc:
        click.echo(f"Error: policy file is not valid JSON ({policy}): {exc}", err=True)
        sys.exit(EXIT_MALFORMED)
    detected_policy_version = str(policy_data.get("policy_version") or "unspecified")
    detected_policy_id = policy_data.get("policy_id")
    if detected_policy_id and detected_policy_id != policy_id:
        click.echo(
            f"Warning: --policy-id '{policy_id}' differs from policy file id '{detected_policy_id}'.",
            err=True,
        )

    # Build output directory
    out_dir = output or (root / "out" / trace_id)

    # Build command for run_pipeline.py
    # Note: steps/challenges/queries are FRI prover settings stored in capsule metadata
    # but not directly passed to run_pipeline.py which uses its own defaults
    cmd = [
        sys.executable,
        str(script),
        "--backend", backend,
        "--output-dir", str(out_dir),
        "--trace-id", trace_id,
        "--policy", str(policy.resolve()),
        "--policy-id", policy_id,
        "--policy-version", detected_policy_version,
        "--allow-insecure-da-challenge",
        "--verification-profile", "proof_only",
    ]

    # Set up environment
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)

    # Store proof parameters in environment for capsule metadata
    env["CAPSEAL_FRI_STEPS"] = str(steps)
    env["CAPSEAL_FRI_CHALLENGES"] = str(challenges)
    env["CAPSEAL_FRI_QUERIES"] = str(queries)
    env["CAPSEAL_PROFILE"] = profile

    # Process datasets (passed via environment, not command line)
    dataset_paths: list[Path] = []
    remote_datasets: list[str] = []
    dataset_env_list = []
    for ds in dataset:
        raw_spec = ds.split("=", 1)[1] if "=" in ds else ds
        if _is_remote_resource(raw_spec):
            remote_datasets.append(raw_spec)
            continue
        dataset_paths.append(Path(raw_spec).expanduser().resolve())
        dataset_env_list.append(ds)

    # Pass datasets via environment variable (comma-separated)
    if dataset_env_list:
        env["CAPSEAL_DATASETS"] = ",".join(dataset_env_list)

    if not json_output:
        click.echo(f"Running pipeline: {trace_id}")
        click.echo(f"Output: {out_dir}")
        if sandbox:
            click.echo(f"Sandbox: enabled")
            if remote_datasets and not sandbox_allow_network:
                click.echo(
                    "Error: remote datasets require --sandbox-allow-network when sandboxed.",
                    err=True,
                )
                sys.exit(EXIT_MALFORMED)

    sandbox_isolation = None
    try:
        if sandbox:
            # Run in sandbox
            from bef_zk.sandbox import SandboxRunner, SandboxConfig, is_sandbox_available

            if not is_sandbox_available():
                click.echo(
                    "Warning: No sandbox backend available. "
                    "Install bubblewrap (Linux) or use macOS for sandboxing.",
                    err=True,
                )
                click.echo("Falling back to unsandboxed execution.", err=True)

            config = SandboxConfig(
                datasets=dataset_paths,
                output_dir=out_dir,
                policy_path=policy.resolve(),
                memory_mb=sandbox_memory,
                wall_time_sec=sandbox_timeout,
                network=sandbox_allow_network,
                capseal_root=root,
            )
            runner = SandboxRunner(config)
            sandbox_result = runner.run(cmd)
            sandbox_isolation = sandbox_result.resource_usage.get("isolation", {})

            if not json_output and sandbox_result.sandbox_backend != "none":
                click.echo(f"Sandbox backend: {sandbox_result.sandbox_backend}")
                if sandbox_allow_network:
                    click.echo("  Network: allowed (per --sandbox-allow-network)")
                # Surface isolation guarantees
                isolation = sandbox_result.resource_usage.get("isolation", {})
                if isolation.get("network_degraded"):
                    click.echo(
                        "  ⚠ Network isolation unavailable on this host (shared net in use)",
                        err=True,
                    )
                else:
                    guarantees = []
                    if sandbox_allow_network:
                        guarantees.append("net=allowed")
                    elif isolation.get("network"):
                        guarantees.append("net=isolated")
                    if isolation.get("pid_namespace"):
                        guarantees.append("pid=isolated")
                    if isolation.get("filesystem"):
                        guarantees.append("fs=pivot" if isolation.get("pivot_root") else "fs=restricted")
                    if isolation.get("memory_limit"):
                        guarantees.append("mem=limited")
                    if isolation.get("fallback_from"):
                        guarantees.append(f"fallback={isolation['fallback_from']}")
                    if guarantees:
                        click.echo(f"  Isolation: {', '.join(guarantees)}")

            # Create a result-like object
            class Result:
                def __init__(self, sr):
                    self.returncode = sr.returncode
                    self.stdout = sr.stdout
                    self.stderr = sr.stderr

            result = Result(sandbox_result)
        else:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                cwd=str(root),
            )

        if result.returncode != 0:
            if json_output:
                click.echo(json.dumps({
                    "status": "ERROR",
                    "error": redact_secrets(result.stderr.strip()) or "Pipeline failed",
                }))
            else:
                click.echo(f"Error: {redact_secrets(result.stderr)}", err=True)
            sys.exit(EXIT_PROOF_INVALID)

        # Read capsule to get summary
        capsule_path = out_dir / "strategy_capsule.json"
        if capsule_path.exists():
            capsule = json.loads(capsule_path.read_text())
            summary = {
                "status": "OK",
                "capsule_hash": capsule.get("capsule_hash"),
                "trace_id": trace_id,
                "profile": capsule.get("header", {}).get("profile"),
                "output_dir": str(out_dir),
                "capsule_path": str(capsule_path),
                "proof_path": str(out_dir / "adapter_proof.json"),
            }

            # Add dataset info if present
            if capsule.get("dataset_ref"):
                summary["datasets"] = {
                    ds["dataset_id"]: ds["root"]
                    for ds in capsule["dataset_ref"]["datasets"]
                }

            assumptions = build_security_assumptions(
                operation="run",
                capsule=capsule,
                sandbox_isolation=sandbox_isolation,
                policy_info=policy_info_from_capsule(capsule),
                dataset_info=dataset_info_from_capsule(capsule),
            )
            summary["security_assumptions"] = assumptions

            if json_output:
                click.echo(json.dumps(summary, indent=2))
            else:
                click.echo(f"\nCapsule generated successfully!")
                click.echo(f"  Hash: {summary['capsule_hash']}")
                click.echo(f"  Path: {summary['capsule_path']}")
                if summary.get("profile") and summary.get("profile") != "default":
                    click.echo(f"  Profile: {summary['profile']}")
                if summary.get("datasets"):
                    click.echo(f"  Datasets: {len(summary['datasets'])} bound")
                print_security_assumptions(assumptions)
        else:
            if json_output:
                click.echo(json.dumps({"status": "OK", "output_dir": str(out_dir)}))
            else:
                click.echo(f"Pipeline completed. Output: {out_dir}")

    except Exception as e:
        if json_output:
            click.echo(json.dumps({"status": "ERROR", "error": str(e)}))
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_MALFORMED)
    else:
        if transition_spec == "bicep_v2" and emit_bicep_v2:
            _emit_and_verify_bicep_run(
                root=root,
                out_dir=out_dir,
                run_dir_override=bicep_run_dir,
                seed_hex=bicep_seed,
                steps=bicep_steps,
                num_paths=bicep_paths,
                num_channels=bicep_channels,
                audit_k=bicep_audit_k,
                verify=verify_bicep,
                json_output=json_output,
                transition_spec=transition_spec,
            )


def _emit_and_verify_bicep_run(
    *,
    root: Path,
    out_dir: Path,
    run_dir_override: Optional[Path],
    seed_hex: str,
    steps: int,
    num_paths: int,
    num_channels: int,
    audit_k: int,
    verify: bool,
    json_output: bool,
    transition_spec: str = "bicep_v2",
) -> None:
    run_dir = run_dir_override or (out_dir / "bicep_v2_run")
    try:
        emit_bicep_v2_run(
            run_dir,
            seed_hex=seed_hex,
            num_steps=steps,
            num_paths=num_paths,
            num_channels=num_channels,
            audit_k=audit_k,
        )
        if not json_output:
            click.echo(f"Emitted bicep_v2 run directory: {run_dir}")
        if verify:
            _verify_bicep_run(
                run_dir=run_dir,
                seed_hex=seed_hex,
                transition_spec=transition_spec,
                json_output=json_output,
            )
    except Exception as exc:
        msg = f"Failed to emit bicep_v2 run: {exc}"
        if json_output:
            click.echo(json.dumps({"status": "ERROR", "error": msg}))
        else:
            click.echo(msg, err=True)
            click.echo("Use --no-emit-bicep-v2 to disable emission if this is unexpected.", err=True)


def _verify_bicep_run(*, run_dir: Path, seed_hex: str, transition_spec: str, json_output: bool) -> None:
    trace = run_dir / "trace.jsonl"
    commitments = run_dir / "commitments.json"
    openings = run_dir / "audit_openings"
    manifest_path = run_dir / "manifest.json"

    try:
        from verifier import verify_trace, verify_trace_correctness  # type: ignore
    except ImportError:
        if not json_output:
            click.echo("Independent verifier module missing; skipping verification", err=True)
        return

    ok, msg = verify_trace(trace, commitments)
    if not ok:
        if not json_output:
            click.echo(f"bicep_v2 trace verification failed: {msg}", err=True)
        return

    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    ok, msg = verify_trace_correctness(
        trace,
        seed_hex,
        transition_spec_id=transition_spec,
        openings_path=openings,
        manifest=manifest,
    )
    if ok:
        if not json_output:
            click.echo("  bicep_v2 run verified (trace + correctness)")
    else:
        if not json_output:
            click.echo(f"bicep_v2 correctness verification failed: {msg}", err=True)
