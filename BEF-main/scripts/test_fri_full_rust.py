#!/usr/bin/env python3
"""Full FRI proof test with Rust acceleration.

This test:
1. Generates a test codeword
2. Generates FRI proof using Python path
3. Generates FRI proof using Rust path
4. Verifies both proofs with Python verifier
5. Compares timing

CRITICAL: Milestone 2 is complete when Rust proof verifies correctly.

Usage:
    PYTHONPATH=. python scripts/test_fri_full_rust.py --size 1024
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bef_zk.fri.config import FRIConfig
from bef_zk.fri.prover import fri_prove, _HAS_RUST
from bef_zk.fri.verifier import fri_verify
from bef_zk.stc.vc import STCVectorCommitment
from bef_zk.stc.aok_cpu import MODULUS


def main():
    parser = argparse.ArgumentParser(description="Full FRI proof Rust test")
    parser.add_argument("--size", type=int, default=1024, help="Domain size (power of 2)")
    parser.add_argument("--queries", type=int, default=8, help="Number of query indices")
    parser.add_argument("--chunk-len", type=int, default=64, help="Chunk length")
    parser.add_argument("--arity", type=int, default=None, help="Chunk tree arity")
    args = parser.parse_args()

    size = args.size
    if size & (size - 1) != 0:
        print(f"Error: size must be power of 2, got {size}")
        sys.exit(1)

    chunk_len = args.chunk_len
    chunk_tree_arity = args.arity or int(os.environ.get("STC_CHUNK_TREE_ARITY", "16"))
    num_rounds = size.bit_length() - 3  # Leave final polynomial
    num_queries = args.queries

    print("FRI Full Proof Test (Rust Milestone 2)")
    print(f"  Domain size: {size}")
    print(f"  FRI rounds: {num_rounds}")
    print(f"  Query count: {num_queries}")
    print(f"  Chunk len: {chunk_len}")
    print(f"  Chunk tree arity: {chunk_tree_arity}")
    print(f"  Rust available: {_HAS_RUST}")
    print()

    # Generate deterministic test codeword
    print("Generating test codeword...")
    base_evals = [(i * 31337 + 12345) % MODULUS for i in range(size)]

    # Generate query indices
    query_indices = [(i * 7919) % size for i in range(num_queries)]
    print(f"Query indices: {query_indices[:5]}..." if len(query_indices) > 5 else f"Query indices: {query_indices}")

    # FRI config
    fri_cfg = FRIConfig(
        field_modulus=MODULUS,
        domain_size=size,
        max_degree=size // 2,
        num_rounds=num_rounds,
        num_queries=num_queries,
    )

    # ========================================
    # Python path
    # ========================================
    print("\n" + "=" * 60)
    print("PYTHON PATH")
    print("=" * 60)

    py_vc = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)
    py_commit = py_vc.commit(base_evals)
    print(f"Python base root: {py_commit.root.hex()[:16]}...")

    print("Generating Python FRI proof...")
    start = time.time()
    py_proof = fri_prove(fri_cfg, py_vc, base_evals, py_commit, query_indices, use_rust=False)
    py_time = time.time() - start
    print(f"  Time: {py_time:.3f}s")
    print(f"  Layers: {len(py_proof.layers)}")
    print(f"  Batches: {len(py_proof.batches)}")

    # Get expected values at query indices for verification
    expected_values = [base_evals[i] for i in query_indices]

    print("Verifying Python FRI proof...")
    py_valid = fri_verify(fri_cfg, py_vc, py_commit, py_proof, query_indices, expected_values)
    print(f"  Valid: {py_valid}")

    if not py_valid:
        print("ERROR: Python proof failed verification!")
        sys.exit(1)

    # ========================================
    # Rust path
    # ========================================
    if not _HAS_RUST:
        print("\nRust not available, skipping Rust test")
        return

    print("\n" + "=" * 60)
    print("RUST PATH")
    print("=" * 60)

    # Create a new VC for Rust path (we don't need Python store for Rust)
    rust_vc = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)
    # We need to commit with Python VC to get base_commitment for API compatibility
    # But the actual Rust path will re-commit internally
    rust_base_commit = rust_vc.commit(base_evals)
    print(f"Rust base root: {rust_base_commit.root.hex()[:16]}...")

    print("Generating Rust FRI proof...")
    start = time.time()
    rust_proof = fri_prove(fri_cfg, rust_vc, base_evals, rust_base_commit, query_indices, use_rust=True)
    rust_time = time.time() - start
    print(f"  Time: {rust_time:.3f}s")
    print(f"  Layers: {len(rust_proof.layers)}")
    print(f"  Batches: {len(rust_proof.batches)}")

    print("Verifying Rust FRI proof...")
    # For verification, we need the base commitment from the proof's first layer
    # The Rust path may produce a different root (it re-commits internally)
    rust_base_commit_from_proof = rust_proof.layers[0].commitment
    rust_valid = fri_verify(fri_cfg, rust_vc, rust_base_commit_from_proof, rust_proof, query_indices, expected_values)
    print(f"  Valid: {rust_valid}")

    # ========================================
    # Compare
    # ========================================
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    # Check layer roots match
    all_ok = True
    if len(py_proof.layers) != len(rust_proof.layers):
        print(f"Layer count mismatch: py={len(py_proof.layers)} rust={len(rust_proof.layers)}")
        all_ok = False
    else:
        print(f"Layer count: MATCH ({len(py_proof.layers)})")
        for i, (py_layer, rust_layer) in enumerate(zip(py_proof.layers, rust_proof.layers)):
            if py_layer.commitment.root != rust_layer.commitment.root:
                print(f"  Layer {i}: ROOT MISMATCH")
                print(f"    py:   {py_layer.commitment.root.hex()[:16]}...")
                print(f"    rust: {rust_layer.commitment.root.hex()[:16]}...")
                all_ok = False
            else:
                print(f"  Layer {i}: root={py_layer.commitment.root.hex()[:12]}... MATCH")

    print()
    print(f"Python verification: {'PASS' if py_valid else 'FAIL'}")
    print(f"Rust verification:   {'PASS' if rust_valid else 'FAIL'}")

    if py_time > 0 and rust_time > 0:
        speedup = py_time / rust_time
        print(f"\nSpeedup: {speedup:.1f}x")

    print("\n" + "=" * 60)
    if all_ok and py_valid and rust_valid:
        print("RESULT: MILESTONE 2 COMPLETE!")
        print(f"  Rust FRI proof (fold + commit + open) verified successfully")
        print(f"  Speedup over Python: {speedup:.1f}x")
    else:
        print("RESULT: TEST FAILED")
        if not rust_valid:
            print("  Rust proof failed verification")
        if not all_ok:
            print("  Layer roots did not match")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
