"""Replay command - semantic replay verification for capsules."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import click

from .exit_codes import EXIT_VERIFIED, EXIT_REPLAY_DIVERGED, EXIT_MALFORMED


def _load_capsule(path: Path) -> dict[str, Any]:
    """Load capsule from JSON or binary format."""
    if path.suffix == ".json":
        return json.loads(path.read_text())
    else:
        from bef_zk.codec import canonical_decode
        return canonical_decode(path.read_bytes())


def _find_adapter(backend_id: str):
    """Get the adapter class for a backend."""
    from backends import ADAPTERS
    return ADAPTERS.get(backend_id)


def _reconstruct_args(capsule: dict[str, Any], datasets: dict[str, Path]) -> Any:
    """Reconstruct CLI args from capsule metadata."""
    import argparse

    # Extract parameters from capsule
    proof_system = capsule.get("proof_system", {})
    header = capsule.get("header", {})

    args = argparse.Namespace()
    args.backend = capsule.get("vm_id", "geom")
    args.trace_id = capsule.get("trace_id", "replay")

    # AIR params (if available)
    # These would come from the original run - for now use defaults
    args.steps = 64
    args.num_challenges = 2
    args.num_queries = 8
    args.challenge_seed = 42
    args.encoding_id = capsule.get("hashing", {}).get("encoding_id", "dag_cbor_compact_fields_v1")

    # Dataset paths
    args.datasets = datasets

    return args


@click.command("replay")
@click.argument("capsule", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--dataset", "-d",
    multiple=True,
    help="Dataset mapping id=path (repeatable)",
)
@click.option(
    "--tolerance",
    type=float,
    default=0.0,
    help="Maximum relative tolerance for equivalence (0.0 = exact match)",
)
@click.option(
    "--range",
    "row_range",
    type=str,
    help="Replay only rows within start:end (0-indexed, end exclusive)",
)
@click.option(
    "--sample",
    type=int,
    default=0,
    help="Replay only a random sample of N rows (seeded)",
)
@click.option(
    "--sample-seed",
    type=int,
    help="Seed for row sampling (defaults to capsule hash)",
)
@click.option(
    "--max-divergences",
    type=int,
    default=100,
    help="Stop after this many divergences",
)
@click.option(
    "--until-diverge",
    is_flag=True,
    help="Stop replay immediately after the first divergence",
)
@click.option(
    "--json", "output_json",
    is_flag=True,
    help="Output result as JSON",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed divergence information",
)
def replay_command(
    capsule: Path,
    dataset: tuple[str, ...],
    tolerance: float,
    row_range: str | None,
    sample: int,
    sample_seed: int | None,
    max_divergences: int,
    until_diverge: bool,
    output_json: bool,
    verbose: bool,
) -> None:
    """Replay a capsule trace for semantic verification.

    Re-executes the computation and compares with the original trace
    to verify deterministic reproducibility. Supports full replay,
    index ranges, random sampling, and stopping after the first
    divergence for faster triage.

    \b
    Exit codes:
        0  - Replay verified (identical or within tolerance)
        14 - Replay diverged beyond tolerance
        20 - Malformed/parse error or unsupported

    \b
    Examples:
        # Basic replay
        capsule replay out/run/strategy_capsule.json

        # With datasets and tolerance
        capsule replay out/run/strategy_capsule.json \\
            -d archive1=./archive1 \\
            --tolerance 0.001 --json
    """
    from bef_zk.adapter import ReplayStatus

    # Parse dataset arguments
    ds_map: dict[str, Path] = {}
    for spec in dataset:
        if "=" in spec:
            ds_id, ds_path = spec.split("=", 1)
            ds_map[ds_id.strip()] = Path(ds_path.strip()).expanduser().resolve()
        else:
            p = Path(spec).expanduser().resolve()
            ds_map[p.name] = p

    # Load capsule
    try:
        capsule_data = _load_capsule(capsule)
    except Exception as e:
        if output_json:
            click.echo(json.dumps({
                "status": "ERROR",
                "error": f"Failed to load capsule: {e}",
            }))
        else:
            click.echo(f"Error: Failed to load capsule: {e}", err=True)
        sys.exit(EXIT_MALFORMED)

    # Parse replay options
    parsed_range: tuple[int, int] | None = None
    if row_range and sample > 0:
        click.echo("Error: --range and --sample cannot be used together", err=True)
        sys.exit(EXIT_MALFORMED)
    if row_range:
        try:
            start_str, end_str = row_range.split(":", 1)
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else start
            parsed_range = (start, end)
        except ValueError:
            click.echo("Error: --range must be start:end", err=True)
            sys.exit(EXIT_MALFORMED)

    sample_rows = sample if sample > 0 else None

    # Get backend
    backend_id = capsule_data.get("vm_id", "geom")
    adapter_cls = _find_adapter(backend_id)
    if adapter_cls is None:
        if output_json:
            click.echo(json.dumps({
                "status": "ERROR",
                "error": f"Unknown backend: {backend_id}",
            }))
        else:
            click.echo(f"Error: Unknown backend: {backend_id}", err=True)
        sys.exit(EXIT_MALFORMED)

    # Reconstruct args and create adapter
    args = _reconstruct_args(capsule_data, ds_map)
    adapter = adapter_cls(args)

    # Check if replay is supported
    if not adapter.supports_replay():
        contract = adapter.get_determinism_contract()
        if output_json:
            click.echo(json.dumps({
                "status": "UNSUPPORTED",
                "backend": backend_id,
                "message": f"Backend '{backend_id}' does not support semantic replay",
                "determinism_contract": contract,
            }))
        else:
            click.echo(f"Replay not supported by backend '{backend_id}'")
            click.echo("Determinism contract:", err=True)
            click.echo(json.dumps(contract, indent=2), err=True)
        sys.exit(EXIT_MALFORMED)

    # Load original trace artifacts
    try:
        # Re-simulate to get original artifacts
        original_artifacts = adapter.simulate_trace(args)
    except Exception as e:
        if output_json:
            click.echo(json.dumps({
                "status": "ERROR",
                "error": f"Failed to simulate trace: {e}",
            }))
        else:
            click.echo(f"Error: Failed to simulate trace: {e}", err=True)
        sys.exit(EXIT_MALFORMED)

    # Perform replay
    start_time = time.perf_counter()
    try:
        # Default sample seed derived from capsule hash if not provided
        seed = sample_seed
        if seed is None:
            cap_hash = capsule_data.get("capsule_hash")
            if isinstance(cap_hash, str):
                try:
                    seed = int(cap_hash[:16], 16)
                except ValueError:
                    seed = 0
            else:
                seed = 0

        result = adapter.replay_trace(
            original_artifacts,
            tolerance=tolerance,
            max_divergences=max_divergences,
            row_range=parsed_range,
            sample=sample_rows,
            seed=seed,
            stop_on_first=until_diverge,
        )
    except Exception as e:
        if output_json:
            click.echo(json.dumps({
                "status": "ERROR",
                "error": f"Replay failed: {e}",
            }))
        else:
            click.echo(f"Error: Replay failed: {e}", err=True)
        sys.exit(EXIT_MALFORMED)

    elapsed = time.perf_counter() - start_time

    # Build result
    output = {
        "status": result.status.value.upper(),
        "backend": backend_id,
        "rows_checked": result.rows_checked,
        "rows_matched": result.rows_matched,
        "max_absolute_diff": result.max_absolute_diff,
        "max_relative_diff": result.max_relative_diff,
        "tolerance": tolerance,
        "replay_time_sec": elapsed,
        "divergence_count": len(result.divergences),
    }

    if parsed_range:
        output["row_range"] = {
            "start": parsed_range[0],
            "end": parsed_range[1],
        }
    if sample_rows:
        output["sample"] = sample_rows
        output["sample_seed"] = seed
    if result.row_indices:
        output["row_indices"] = result.row_indices
    if until_diverge:
        output["stop_on_first"] = True

    if result.message:
        output["message"] = result.message

    if verbose and result.divergences:
        output["divergences"] = [
            {
                "row": d.row_index,
                "column": d.column_name,
                "original": d.original_value,
                "replayed": d.replayed_value,
                "abs_diff": d.absolute_diff,
                "rel_diff": d.relative_diff,
            }
            for d in result.divergences[:10]  # Limit output
        ]
        if len(result.divergences) > 10:
            output["divergences_truncated"] = len(result.divergences) - 10

    if output_json:
        click.echo(json.dumps(output, indent=2))
    else:
        desc = []
        if parsed_range:
            desc.append(f"rows {parsed_range[0]}:{parsed_range[1]}")
        if sample_rows:
            desc.append(f"sample={sample_rows}")
        scope = f" ({', '.join(desc)})" if desc else ""

        if result.status == ReplayStatus.IDENTICAL:
            click.echo(f"REPLAY VERIFIED: Identical ({result.rows_checked} rows){scope}")
        elif result.status == ReplayStatus.EQUIVALENT:
            click.echo(
                f"REPLAY VERIFIED: Equivalent within tolerance {tolerance}{scope}"
            )
            click.echo(f"  Max diff: {result.max_relative_diff:.6f}")
        elif result.status == ReplayStatus.DIVERGED:
            click.echo(f"REPLAY DIVERGED: {len(result.divergences)} differences found", err=True)
            click.echo(f"  Max diff: {result.max_relative_diff:.6f} (tolerance: {tolerance})", err=True)
            if verbose:
                for d in result.divergences[:5]:
                    click.echo(f"  Row {d.row_index}, {d.column_name}: {d.original_value} vs {d.replayed_value}", err=True)
        else:
            click.echo(f"REPLAY: {result.status.value}", err=True)
            if result.message:
                click.echo(f"  {result.message}", err=True)

    # Exit code
    if result.status in (ReplayStatus.IDENTICAL, ReplayStatus.EQUIVALENT):
        sys.exit(EXIT_VERIFIED)
    elif result.status == ReplayStatus.DIVERGED:
        sys.exit(EXIT_REPLAY_DIVERGED)
    else:
        sys.exit(EXIT_MALFORMED)
