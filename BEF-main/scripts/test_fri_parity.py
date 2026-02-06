#!/usr/bin/env python3
"""FRI layer parity test: verify Python and Rust produce identical results.

This test:
1. Generates a test codeword
2. Runs FRI fold+commit with Python VectorCommitment
3. Runs FRI fold+commit with Rust PyFriState
4. Compares layer-by-layer: beta, root, length

CRITICAL: Transcript (beta derivation) stays in Python for BOTH paths.
This ensures we're only testing compute parity, not transcript drift.

Usage:
    PYTHONPATH=. python scripts/test_fri_parity.py --size 1024
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import bef_rust
from bef_zk.fri.domain import fold_codeword
from bef_zk.stc.vc import STCVectorCommitment
from bef_zk.stc.aok_cpu import MODULUS, ROOT_SEED, merkle_from_values

# Transcript derivation (MUST be identical in both paths)
def derive_beta(root: bytes, round_idx: int) -> int:
    """Derive FRI challenge from commitment root (Fiat-Shamir)."""
    h = hashlib.sha256()
    h.update(root)
    h.update(round_idx.to_bytes(4, "big"))
    candidate = int.from_bytes(h.digest(), "big") % MODULUS
    if candidate == 0:
        candidate = 1
    return candidate


def run_python_fri(
    base_evals: list[int],
    num_rounds: int,
    chunk_len: int,
) -> list[tuple[bytes, int, int]]:
    """Run FRI with Python VectorCommitment.

    Returns: List of (root, beta, length) per layer.
    """
    vc = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)

    layers = []
    current = [int(v) % MODULUS for v in base_evals]

    # Initial commitment
    commit = vc.commit(current)

    for round_idx in range(num_rounds):
        beta = derive_beta(commit.root, round_idx)
        layers.append((commit.root, beta, len(current)))

        # Fold
        next_codeword = fold_codeword(current, beta, MODULUS)

        # Commit folded
        commit = vc.commit(next_codeword)
        current = next_codeword

    # Final layer
    layers.append((commit.root, 0, len(current)))

    return layers


def run_rust_fri(
    base_evals: list[int],
    num_rounds: int,
    chunk_len: int,
    chunk_tree_arity: int = 2,
) -> list[tuple[bytes, int, int]]:
    """Run FRI with Rust PyFriState.

    Returns: List of (root, beta, length) per layer.
    """
    # Initialize Rust state
    state = bef_rust.PyFriState(base_evals)

    layers = []

    # Initial commitment (before any folding)
    result = state.commit_current(chunk_len, chunk_tree_arity=chunk_tree_arity)
    current_root = bytes.fromhex(result.root_hex)
    current_len = result.length  # Track length from result, not state

    for round_idx in range(num_rounds):
        # Transcript stays in Python!
        beta = derive_beta(current_root, round_idx)
        layers.append((current_root, beta, current_len))

        # Fold + commit in Rust (one call)
        result = state.fold_and_commit(beta, chunk_len, chunk_tree_arity=chunk_tree_arity)
        current_root = bytes.fromhex(result.root_hex)
        current_len = result.length

    # Final layer
    layers.append((current_root, 0, current_len))

    return layers


def main():
    import os
    parser = argparse.ArgumentParser(description="FRI parity test")
    parser.add_argument("--size", type=int, default=1024, help="Codeword size (power of 2)")
    parser.add_argument("--rounds", type=int, default=None, help="Number of FRI rounds")
    parser.add_argument("--chunk-len", type=int, default=64, help="Chunk length")
    parser.add_argument("--arity", type=int, default=None, help="Chunk tree arity (default: from STC_CHUNK_TREE_ARITY or 16)")
    args = parser.parse_args()

    size = args.size
    if size & (size - 1) != 0:
        print(f"Error: size must be power of 2, got {size}")
        sys.exit(1)

    # Default rounds = log2(size) - 2 (leave some polynomial at the end)
    num_rounds = args.rounds if args.rounds else (size.bit_length() - 3)
    chunk_len = args.chunk_len
    # Get arity from argument, env var, or default
    chunk_tree_arity = args.arity or int(os.environ.get("STC_CHUNK_TREE_ARITY", "16"))

    print(f"FRI Parity Test")
    print(f"  Size: {size}")
    print(f"  Rounds: {num_rounds}")
    print(f"  Chunk len: {chunk_len}")
    print(f"  Chunk tree arity: {chunk_tree_arity}")
    print()

    # Generate deterministic test codeword
    print("Generating test codeword...")
    base_evals = [(i * 31337 + 12345) % MODULUS for i in range(size)]

    # Run Python
    print("\nRunning Python FRI...")
    start = time.time()
    py_layers = run_python_fri(base_evals, num_rounds, chunk_len)
    py_time = time.time() - start
    print(f"  Time: {py_time:.3f}s")
    print(f"  Layers: {len(py_layers)}")

    # Run Rust
    print("\nRunning Rust FRI...")
    start = time.time()
    rust_layers = run_rust_fri(base_evals, num_rounds, chunk_len, chunk_tree_arity)
    rust_time = time.time() - start
    print(f"  Time: {rust_time:.3f}s")
    print(f"  Layers: {len(rust_layers)}")

    # Compare
    print("\n" + "=" * 60)
    print("PARITY CHECKS")
    print("=" * 60)

    all_ok = True

    if len(py_layers) != len(rust_layers):
        print(f"✗ Layer count mismatch: Python={len(py_layers)}, Rust={len(rust_layers)}")
        all_ok = False
    else:
        print(f"✓ Layer counts match ({len(py_layers)})")

    for i, ((py_root, py_beta, py_len), (rust_root, rust_beta, rust_len)) in enumerate(
        zip(py_layers, rust_layers)
    ):
        layer_ok = True
        issues = []

        if py_root != rust_root:
            issues.append(f"root mismatch: py={py_root.hex()[:16]}... rust={rust_root.hex()[:16]}...")
            layer_ok = False

        if py_beta != rust_beta:
            issues.append(f"beta mismatch: py={py_beta} rust={rust_beta}")
            layer_ok = False

        if py_len != rust_len:
            issues.append(f"length mismatch: py={py_len} rust={rust_len}")
            layer_ok = False

        if layer_ok:
            print(f"  Layer {i}: ✓ root={py_root.hex()[:12]}... len={py_len}")
        else:
            print(f"  Layer {i}: ✗ {', '.join(issues)}")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: ALL PARITY CHECKS PASSED")
        speedup = py_time / rust_time if rust_time > 0 else 0
        print(f"Rust is {speedup:.1f}x faster")
    else:
        print("RESULT: PARITY CHECKS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
