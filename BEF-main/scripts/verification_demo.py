#!/usr/bin/env python3
"""
Verification Demo - "Try to break it" demonstration

One command that:
1. Generates a run with trace, checkpoints, features, sidecar
2. Verifies full chain (trace -> commitments -> checkpoints)
3. Opens 3 random steps with Merkle proofs
4. Runs 1,000 random mutations and proves all are rejected
5. Prints a final "receipt DAG" summary

This is what convinces skeptics: not "trust me" but "try to break it."

Usage:
    PYTHONPATH=. python scripts/verification_demo.py
    PYTHONPATH=. python scripts/verification_demo.py --mutations 10000
    PYTHONPATH=. python scripts/verification_demo.py --steps 20 --checkpoint-interval 4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bef_zk.capsule.manifest import create_manifest
from bef_zk.capsule.trace_chain import (
    TraceRow,
    TraceEmitter,
    CheckpointReceipt,
    open_row,
    verify_opening,
    verify_trace_against_commitments,
)
from bef_zk.capsule.features_sidecar import (
    create_features_sidecar,
    validate_features_against_sidecar,
)

# Import independent verifier for fuzz testing
sys.path.insert(0, str(Path(__file__).parent.parent / "verifier-independent"))
from verifier import run_fuzz_test


def print_header(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_step(step: int, msg: str):
    print(f"\n[Step {step}] {msg}")


def generate_run(
    output_dir: Path,
    num_steps: int,
    checkpoint_interval: int,
) -> dict:
    """Generate a complete run with all artifacts."""

    # Create manifest and RNG
    seed = bytes.fromhex("de0abcde" * 8)  # Valid hex, 32 bytes
    manifest, rng = create_manifest(
        seed=seed,
        bicep_version="demo-0.1.0",
        checkpoint_interval=checkpoint_interval,
    )

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    manifest.save(manifest_path)

    # Create trace emitter
    emitter = TraceEmitter(
        manifest_hash=manifest.manifest_hash,
        checkpoint_interval=checkpoint_interval,
        output_dir=output_dir,
    )

    # Emit trace rows
    for t in range(num_steps):
        row = TraceRow(
            t=t,
            x_t=[rng.rand("input", t, i) for i in range(4)],
            view_pre={"state": t, "value": rng.rand("state", t, 0)},
            view_post={"state": t + 1, "value": rng.rand("state", t + 1, 0)},
            rand_addrs=[{"tag": "input", "t": t, "i": i} for i in range(4)],
        )
        emitter.emit(row)

    summary = emitter.finalize()

    # Create features CSV
    features_path = output_dir / "features.csv"
    with open(features_path, 'w') as f:
        f.write("f0,f1,f2,f3\n")
        for t in range(num_steps):
            vals = [rng.rand("feature", t, i) for i in range(4)]
            f.write(",".join(f"{v:.8f}" for v in vals) + "\n")

    # Create sidecar
    sidecar = create_features_sidecar(
        features_path=features_path,
        manifest_hash=manifest.manifest_hash,
        head_at_end=emitter.chain.head,
        step_start=0,
        step_end=num_steps,
    )
    sidecar_path = output_dir / "sidecar.json"
    sidecar.save(sidecar_path)

    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "trace_path": output_dir / "trace.jsonl",
        "commitments_path": output_dir / "commitments.json",
        "features_path": features_path,
        "sidecar_path": sidecar_path,
        "sidecar": sidecar,
        "summary": summary,
        "output_dir": output_dir,
        "rng": rng,
    }


def verify_full_chain(run: dict) -> bool:
    """Verify the full chain: trace -> commitments."""
    valid, msg = verify_trace_against_commitments(
        run["trace_path"],
        run["commitments_path"],
    )

    if valid:
        print(f"    Trace verification: PASS ({msg})")
    else:
        print(f"    Trace verification: FAIL ({msg})")

    # Verify features against sidecar
    valid2, msg2 = validate_features_against_sidecar(
        run["features_path"],
        run["sidecar_path"],
    )

    if valid2:
        print(f"    Sidecar verification: PASS ({msg2})")
    else:
        print(f"    Sidecar verification: FAIL ({msg2})")

    return valid and valid2


def open_random_steps(run: dict, num_openings: int) -> list[dict]:
    """Open random steps with Merkle proofs."""
    num_steps = run["summary"]["total_steps"]
    if num_steps == 0:
        return []

    # Select random steps
    steps_to_open = random.sample(range(num_steps), min(num_openings, num_steps))
    steps_to_open.sort()

    openings = []
    for step in steps_to_open:
        opening = open_row(
            step=step,
            trace_path=run["trace_path"],
            checkpoints_dir=run["output_dir"],
        )

        # Verify opening
        valid, msg = verify_opening(opening)

        openings.append({
            "step": step,
            "leaf_index": opening.leaf_index,
            "checkpoint_index": opening.checkpoint_index,
            "chunk_root": opening.chunk_root[:16] + "...",
            "proof_length": len(opening.merkle_proof),
            "valid": valid,
            "msg": msg,
        })

        status = "PASS" if valid else "FAIL"
        print(f"    Step {step}: {status} (checkpoint={opening.checkpoint_index}, leaf={opening.leaf_index}, proof_len={len(opening.merkle_proof)})")

    return openings


def run_mutation_test(run: dict, num_mutations: int) -> dict:
    """Run mutation/fuzz test using independent verifier."""
    results = run_fuzz_test(
        run["trace_path"],
        run["commitments_path"],
        num_mutations,
    )
    return results


def print_receipt_dag(run: dict, openings: list[dict]):
    """Print the receipt DAG summary."""
    manifest = run["manifest"]
    sidecar = run["sidecar"]
    summary = run["summary"]

    # Load checkpoint receipts for DAG
    checkpoints = []
    for i in range(summary["total_checkpoints"]):
        cp_path = run["output_dir"] / f"checkpoint_{i:04d}.json"
        if cp_path.exists():
            cp = CheckpointReceipt.load(cp_path)
            checkpoints.append({
                "index": i,
                "step_range": f"[{cp.step_start}, {cp.step_end})",
                "chunk_root": cp.chunk_root[:16] + "...",
                "head_at_end": cp.head_at_end[:16] + "...",
            })

    print("""
RECEIPT DAG SUMMARY
===================

Level 0: MANIFEST (run configuration)
    """)
    print(f"    manifest_hash: {manifest.manifest_hash}")
    print(f"    seed_commitment: {manifest.config.seed_commitment[:32]}...")
    print(f"    checkpoint_interval: {manifest.config.checkpoint_interval}")

    print("""
        |
        v

Level 1: CHECKPOINTS (trace commitments)
    """)
    for cp in checkpoints:
        print(f"    checkpoint_{cp['index']}: steps {cp['step_range']}")
        print(f"        chunk_root: {cp['chunk_root']}")
        print(f"        head_at_end: {cp['head_at_end']}")

    print("""
        |
        v

Level 2: FEATURES SIDECAR (BICEP -> ENN binding)
    """)
    print(f"    sidecar_hash: {sidecar.compute_sidecar_hash()[:32]}...")
    print(f"    features_hash: {sidecar.features_shard_hash[:32]}...")
    print(f"    trace_anchor: head_at_end={sidecar.head_at_end[:16]}...")
    print(f"    step_range: [{sidecar.step_start}, {sidecar.step_end})")

    print("""
        |
        v

Level 3: SELECTIVE OPENINGS (Merkle proofs)
    """)
    for op in openings:
        print(f"    step {op['step']}: checkpoint={op['checkpoint_index']}, leaf={op['leaf_index']}, verified={op['valid']}")

    print("""
VERIFICATION CHAIN:
    manifest_hash -> checkpoint_roots -> sidecar_hash -> [ENN artifact] -> [Fusion receipt]
                                      ^                 ^
                                      |                 |
                                  trace binding     features binding
    """)


def main():
    parser = argparse.ArgumentParser(description="Verification Demo")
    parser.add_argument("--steps", type=int, default=20, help="Number of trace steps")
    parser.add_argument("--checkpoint-interval", type=int, default=4, help="Checkpoint interval K")
    parser.add_argument("--mutations", type=int, default=1000, help="Number of mutations to test")
    parser.add_argument("--openings", type=int, default=3, help="Number of random steps to open")
    args = parser.parse_args()

    print_header("CAPSEAL VERIFICATION DEMO")
    print("""
This demo shows the tamper-evident verification chain in action.
We generate a run, verify it, open random steps with Merkle proofs,
and prove the system rejects ALL mutations.

"Not trust me. Try to break it."
    """)

    with tempfile.TemporaryDirectory(prefix="verification_demo_") as tmp:
        output_dir = Path(tmp)

        # Step 1: Generate run
        print_step(1, f"Generating run: {args.steps} steps, K={args.checkpoint_interval}")
        start = time.time()
        run = generate_run(output_dir, args.steps, args.checkpoint_interval)
        gen_time = time.time() - start
        print(f"    Generated in {gen_time:.2f}s")
        print(f"    Manifest hash: {run['manifest'].manifest_hash[:32]}...")
        print(f"    Total steps: {run['summary']['total_steps']}")
        print(f"    Total checkpoints: {run['summary']['total_checkpoints']}")
        print(f"    Final head: {run['summary']['head_T'][:32]}...")

        # Step 2: Verify full chain
        print_step(2, "Verifying full chain")
        if not verify_full_chain(run):
            print("\n    [FATAL] Chain verification failed!")
            sys.exit(1)

        # Step 3: Open random steps
        print_step(3, f"Opening {args.openings} random steps with Merkle proofs")
        openings = open_random_steps(run, args.openings)
        if any(not op["valid"] for op in openings):
            print("\n    [FATAL] Some openings failed verification!")
            sys.exit(1)

        # Step 4: Mutation testing
        print_step(4, f"Running {args.mutations} random mutations")
        start = time.time()
        mutation_results = run_mutation_test(run, args.mutations)
        fuzz_time = time.time() - start

        print(f"    Completed in {fuzz_time:.2f}s")
        print(f"    Total mutations: {mutation_results['total']}")
        print(f"    Rejected (correct): {mutation_results['rejected']}")
        print(f"    Accepted (BUG!): {mutation_results['accepted']}")
        print(f"    Reject rate: {100 * mutation_results['rejected'] / mutation_results['total']:.2f}%")

        if mutation_results['accepted'] > 0:
            print(f"\n    [FATAL] {mutation_results['accepted']} mutations were incorrectly accepted!")
            sys.exit(1)

        # Step 5: Print receipt DAG
        print_step(5, "Receipt DAG summary")
        print_receipt_dag(run, openings)

        # Final summary
        print_header("VERIFICATION COMPLETE")
        print(f"""
Results:
    Steps generated: {args.steps}
    Checkpoints: {run['summary']['total_checkpoints']}
    Chain verified: YES
    Selective openings: {len(openings)}/{args.openings} verified
    Mutations tested: {args.mutations}
    Mutations rejected: {mutation_results['rejected']} (100%)

The verification chain is tamper-evident:
    - Every mutation is detected and rejected
    - Selective openings prove membership without revealing full trace
    - Bindings are cryptographically enforced

"This chain is internally consistent AND externally unforgeable."
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
