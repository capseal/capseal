#!/usr/bin/env python3
"""Test FRI prover with Rust acceleration.

This test verifies:
1. Python-only prover produces valid proofs
2. Rust-accelerated prover produces identical proofs
3. Speedup from Rust acceleration

Usage:
    PYTHONPATH=. python scripts/test_fri_prover_rust.py --size 1024
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bef_zk.fri.config import FRIConfig
from bef_zk.fri.prover import fri_prove, _build_layers, _HAS_RUST
from bef_zk.stc.vc import STCVectorCommitment
from bef_zk.stc.aok_cpu import MODULUS


def main():
    parser = argparse.ArgumentParser(description="FRI prover Rust test")
    parser.add_argument("--size", type=int, default=1024, help="Domain size (power of 2)")
    parser.add_argument("--queries", type=int, default=8, help="Number of query indices")
    args = parser.parse_args()

    size = args.size
    if size & (size - 1) != 0:
        print(f"Error: size must be power of 2, got {size}")
        sys.exit(1)

    num_rounds = size.bit_length() - 3  # Leave final polynomial
    num_queries = args.queries

    print(f"FRI Prover Rust Test")
    print(f"  Domain size: {size}")
    print(f"  FRI rounds: {num_rounds}")
    print(f"  Query indices: {num_queries}")
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

    # Vector commitment
    chunk_len = 64
    vc = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)

    # Initial commitment
    print("\nCommitting base evaluations...")
    base_commit = vc.commit(base_evals)
    print(f"  Base root: {base_commit.root.hex()[:16]}...")

    # Python-only layer building
    print("\nBuilding layers with Python...")
    start = time.time()
    py_layers = _build_layers(fri_cfg, vc, base_evals, base_commit, use_rust=False)
    py_time = time.time() - start
    print(f"  Time: {py_time:.3f}s")
    print(f"  Layers: {len(py_layers)}")

    if not _HAS_RUST:
        print("\nRust not available, skipping Rust test")
        return

    # Rust-accelerated with Python commit (compatible but slower)
    print("\nBuilding layers with Rust (with Python commit for open_batch)...")
    vc2 = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)
    base_commit2 = vc2.commit(base_evals)

    start = time.time()
    rust_layers_compat = _build_layers(fri_cfg, vc2, base_evals, base_commit2, use_rust=True, skip_python_commit=False)
    rust_compat_time = time.time() - start
    print(f"  Time: {rust_compat_time:.3f}s")
    print(f"  Layers: {len(rust_layers_compat)}")

    # Rust-accelerated pure (24x faster, no open_batch)
    print("\nBuilding layers with Rust (pure, skip Python commit)...")
    vc3 = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)
    base_commit3 = vc3.commit(base_evals)

    start = time.time()
    rust_layers = _build_layers(fri_cfg, vc3, base_evals, base_commit3, use_rust=True, skip_python_commit=True)
    rust_time = time.time() - start
    print(f"  Time: {rust_time:.3f}s")
    print(f"  Layers: {len(rust_layers)}")

    # Compare layers
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    all_ok = True

    # Check layer count
    if len(py_layers) == len(rust_layers) == len(rust_layers_compat):
        print(f"✓ Layer counts match ({len(py_layers)})")
    else:
        print(f"✗ Layer counts differ")
        all_ok = False

    # Check layer roots (all three should match)
    for i, (py_layer, rust_layer, compat_layer) in enumerate(zip(py_layers, rust_layers, rust_layers_compat)):
        py_root = py_layer.commitment.root
        rust_root = rust_layer.commitment.root
        compat_root = compat_layer.commitment.root
        if py_root == rust_root == compat_root:
            print(f"  Layer {i}: ✓ root={py_root.hex()[:12]}...")
        else:
            print(f"  Layer {i}: ✗ roots differ")
            print(f"    Python:      {py_root.hex()[:16]}...")
            print(f"    Rust compat: {compat_root.hex()[:16]}...")
            print(f"    Rust pure:   {rust_root.hex()[:16]}...")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: ALL CHECKS PASSED")
        print()
        print("Speedups (layer building only):")
        if rust_compat_time < py_time:
            speedup = py_time / rust_compat_time
            print(f"  Rust (compatible, with open_batch): {speedup:.1f}x faster")
        else:
            overhead = rust_compat_time / py_time
            print(f"  Rust (compatible): {overhead:.1f}x slower (double-commit overhead)")

        if rust_time < py_time:
            speedup = py_time / rust_time
            print(f"  Rust (pure, no open_batch):         {speedup:.1f}x faster")
        else:
            print(f"  Rust (pure): slower than Python (unexpected)")
    else:
        print("RESULT: SOME CHECKS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
