"""Self-contained demo command - runs offline, fast, deterministically."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import click

from ..contracts import (
    CAPSEAL_VERSION,
    RECEIPT_SCHEMA_VERSION,
    POLICY_SCHEMA_VERSION,
    ExitCode,
    WorkspaceContract,
)
from ..header import (
    HEADER_SCHEMA,
    build_capsule_header,
    compute_header_hash,
    hash_params,
    hash_chunk_meta,
    hash_da_policy,
    hash_row_index_ref,
)
# Note: Uses inline verification for demo mode (no external deps)


DEMO_TRACE = [
    {"row": 0, "action": "init", "value": 0},
    {"row": 1, "action": "add", "value": 10},
    {"row": 2, "action": "add", "value": 20},
    {"row": 3, "action": "multiply", "value": 2},
    {"row": 4, "action": "final", "value": 60},
]

DEMO_POLICY = {
    "schema": f"bef_benchmark_policy_{POLICY_SCHEMA_VERSION}",
    "policy_id": "demo_offline",
    "policy_version": "1.0.0",
    "tracks": [
        {
            "track_id": "demo",
            "description": "Demo track for offline testing.",
            "rules": {
                "forbid_gpu": False,
                "require_deterministic_build": False,
                "required_public_outputs": ["final_value"],
            },
        }
    ],
}


def _hash_trace(trace: list) -> str:
    """Compute deterministic hash of trace data."""
    data = json.dumps(trace, sort_keys=True).encode()
    return hashlib.sha256(b"DEMO_TRACE::" + data).hexdigest()


def _create_demo_receipt() -> dict:
    """Create a minimal valid receipt for demo purposes."""
    trace_hash = _hash_trace(DEMO_TRACE)
    policy_hash = hashlib.sha256(
        json.dumps(DEMO_POLICY, sort_keys=True).encode()
    ).hexdigest()

    # Build deterministic header
    header = build_capsule_header(
        vm_id="demo_vm",
        backend_id="none",
        circuit_id="demo_circuit",
        trace_id=f"demo_{int(time.time())}",
        prev_capsule_hash=None,
        trace_spec_hash=trace_hash,
        statement_hash=hashlib.sha256(b"demo_statement").hexdigest(),
        params_hash=hash_params({"row_width": 5}),
        row_root=trace_hash[:64],
        row_tree_arity=2,
        row_index_ref_hash=hash_row_index_ref({}),
        chunk_meta_hash=hash_chunk_meta({}),
        chunk_handles_root=hashlib.sha256(b"demo_chunks").hexdigest(),
        policy_ref={
            "policy_id": DEMO_POLICY["policy_id"],
            "policy_version": DEMO_POLICY["policy_version"],
            "policy_hash": policy_hash,
        },
        da_policy_hash=hash_da_policy({}),
        anchor={"timestamp": int(time.time())},
        verification_profile="minimal",
    )

    header_hash = compute_header_hash(header)

    # Create minimal proof (no real cryptographic proof in demo mode)
    proof_data = {
        "format": "demo_v1",
        "commitment": trace_hash,
        "public_outputs": {"final_value": 60},
    }

    receipt = {
        "schema": f"capsule_receipt_{RECEIPT_SCHEMA_VERSION}",
        "capsule_id": header_hash,
        "capseal_version": CAPSEAL_VERSION,
        "header": header,
        "payload": {
            "proof_format": "demo_v1",
            "proof_data": json.dumps(proof_data),
            "trace_summary": {
                "row_count": len(DEMO_TRACE),
                "public_outputs": {"final_value": 60},
            },
        },
        "policy": DEMO_POLICY,
        "trace": DEMO_TRACE,  # Include trace in demo mode for transparency
    }

    return receipt


@click.command("demo")
@click.option(
    "-o", "--output",
    type=click.Path(),
    help="Save demo receipt to file",
)
@click.option(
    "--verify/--no-verify",
    default=True,
    help="Verify the demo receipt after creation (default: yes)",
)
@click.option(
    "--json", "json_output",
    is_flag=True,
    help="Output JSON only",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Show detailed trace data",
)
def demo_command(
    output: str | None,
    verify: bool,
    json_output: bool,
    verbose: bool,
) -> None:
    """Run a self-contained demo to verify capseal is working.

    This command:
    1. Creates a minimal trace dataset in memory
    2. Generates a demo receipt (no real proof)
    3. Verifies the receipt structure
    4. Reports success/failure

    The demo runs entirely offline and should complete in under 1 second.
    Use this to verify your capseal installation is working correctly.

    \b
    Example:
        capseal demo              # Quick verification
        capseal demo --verbose    # Show trace details
        capseal demo -o demo.json # Save receipt to file
    """
    start_time = time.time()

    # Create demo receipt
    receipt = _create_demo_receipt()
    creation_time = time.time() - start_time

    result = {
        "status": "created",
        "capseal_version": CAPSEAL_VERSION,
        "receipt_schema": receipt["schema"],
        "capsule_id": receipt["capsule_id"],
        "trace_rows": len(DEMO_TRACE),
        "creation_time_ms": round(creation_time * 1000, 2),
    }

    # Verify if requested
    if verify:
        verify_start = time.time()
        # Simple structural verification (demo mode doesn't have real proofs)
        is_valid = True
        errors = []

        # Check required fields
        for field in ["schema", "capsule_id", "header", "payload"]:
            if field not in receipt:
                is_valid = False
                errors.append(f"Missing required field: {field}")

        # Check header fields
        header = receipt.get("header", {})
        for field in ["schema", "vm_id", "trace_id", "policy_ref"]:
            if field not in header:
                is_valid = False
                errors.append(f"Missing header field: {field}")

        verify_time = time.time() - verify_start
        result["verification"] = {
            "valid": is_valid,
            "errors": errors,
            "verify_time_ms": round(verify_time * 1000, 2),
        }

    total_time = time.time() - start_time
    result["total_time_ms"] = round(total_time * 1000, 2)

    # Save to file if requested
    if output:
        output_path = Path(output)
        with open(output_path, "w") as f:
            json.dump(receipt, f, indent=2)
        result["output_file"] = str(output_path)

    # Output
    if json_output:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo("=== CapSeal Demo ===")
        click.echo(f"Version: {CAPSEAL_VERSION}")
        click.echo(f"Receipt Schema: {receipt['schema']}")
        click.echo(f"Capsule ID: {receipt['capsule_id'][:16]}...")
        click.echo()

        if verbose:
            click.echo("Trace Data:")
            for row in DEMO_TRACE:
                click.echo(f"  Row {row['row']}: {row['action']} -> {row['value']}")
            click.echo()

        click.echo(f"Creation: {result['creation_time_ms']:.2f}ms")

        if verify:
            v = result["verification"]
            status = "PASS" if v["valid"] else "FAIL"
            click.echo(f"Verification: {status} ({v['verify_time_ms']:.2f}ms)")
            if v["errors"]:
                for err in v["errors"]:
                    click.echo(f"  - {err}")

        click.echo(f"Total: {result['total_time_ms']:.2f}ms")

        if output:
            click.echo(f"\nReceipt saved to: {output}")

        # Exit code
        if verify and not result["verification"]["valid"]:
            raise SystemExit(ExitCode.MALFORMED.value)


__all__ = ["demo_command"]
