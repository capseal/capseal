#!/usr/bin/env python3
"""
STARK Hello World - Interactive FRI Proof Demo

This demonstrates the interactive proof protocol:
1. Prover commits to polynomial evaluations
2. Verifier picks random query points
3. Prover opens evaluations at those points
4. Verifier checks consistency via FRI folding

Usage:
    python scripts/stark_hello.py
    python scripts/stark_hello.py --domain-size 1024 --queries 8
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from typing import List

# Add project root to path
sys.path.insert(0, "/home/ryan/BEF-main")

from bef_zk.fri.config import FRIConfig, MODULUS
from bef_zk.fri.prover import fri_prove
from bef_zk.fri.verifier import fri_verify
from bef_zk.stc.vc import STCVectorCommitment


def generate_polynomial_evals(domain_size: int, degree: int, seed: int = 42) -> List[int]:
    """Generate evaluations of a random polynomial."""
    random.seed(seed)
    # Random coefficients for polynomial of given degree
    coeffs = [random.randint(1, MODULUS - 1) for _ in range(degree + 1)]

    # Evaluate polynomial at domain points (simplified: use indices as points)
    evals = []
    for i in range(domain_size):
        val = 0
        x_power = 1
        for c in coeffs:
            val = (val + c * x_power) % MODULUS
            x_power = (x_power * i) % MODULUS
        evals.append(val)
    return evals


def print_banner():
    print("=" * 60)
    print("  STARK Hello World - FRI Interactive Proof Demo")
    print("=" * 60)
    print()


def main():
    parser = argparse.ArgumentParser(description="STARK Hello World")
    parser.add_argument("--domain-size", type=int, default=512, help="Domain size (power of 2)")
    parser.add_argument("--degree", type=int, default=64, help="Polynomial degree")
    parser.add_argument("--queries", type=int, default=4, help="Number of query points")
    parser.add_argument("--rounds", type=int, default=3, help="FRI folding rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print_banner()

    # Validate domain size is power of 2
    if args.domain_size & (args.domain_size - 1) != 0:
        print(f"ERROR: domain-size must be power of 2, got {args.domain_size}")
        sys.exit(1)

    # Create FRI config
    fri_cfg = FRIConfig(
        field_modulus=MODULUS,
        domain_size=args.domain_size,
        max_degree=args.degree,
        num_rounds=args.rounds,
        num_queries=args.queries,
    )

    print(f"Configuration:")
    print(f"  Field modulus:  2^61 - 1 (Mersenne prime)")
    print(f"  Domain size:    {args.domain_size}")
    print(f"  Poly degree:    {args.degree}")
    print(f"  FRI rounds:     {args.rounds}")
    print(f"  Query points:   {args.queries}")
    print()

    # === PROVER PHASE 1: Commit ===
    print("[Prover] Generating polynomial evaluations...")
    t0 = time.time()
    evals = generate_polynomial_evals(args.domain_size, args.degree, args.seed)
    t_gen = time.time() - t0

    if args.verbose:
        print(f"  First 5 evals: {evals[:5]}")
        print(f"  Last 5 evals:  {evals[-5:]}")

    print(f"[Prover] Committing to {len(evals)} evaluations...")
    t0 = time.time()
    vc = STCVectorCommitment(chunk_len=min(256, args.domain_size))
    commitment = vc.commit(evals)
    t_commit = time.time() - t0

    print(f"  Merkle root: {commitment.root.hex()[:32]}...")
    print(f"  Time:        {t_commit*1000:.1f} ms")
    print()

    # === VERIFIER: Pick random query points ===
    print("[Verifier] Picking random query indices...")
    # Use commitment root as randomness source (Fiat-Shamir heuristic)
    h = hashlib.sha256(commitment.root)
    verifier_seed = int.from_bytes(h.digest(), "big")
    random.seed(verifier_seed)

    query_indices = sorted(random.sample(range(args.domain_size), args.queries))
    query_values = [evals[i] for i in query_indices]

    print(f"  Query indices: {query_indices}")
    if args.verbose:
        print(f"  Query values:  {query_values}")
    print()

    # === PROVER PHASE 2: Generate FRI proof ===
    print("[Prover] Generating FRI proof...")
    t0 = time.time()
    proof = fri_prove(
        fri_cfg=fri_cfg,
        vc=vc,
        base_evals=evals,
        base_commitment=commitment,
        query_indices=query_indices,
        use_rust=False,  # Pure Python for demo
    )
    t_prove = time.time() - t0

    print(f"  FRI layers:  {len(proof.layers)}")
    print(f"  Batch opens: {len(proof.batches)}")
    print(f"  Time:        {t_prove*1000:.1f} ms")

    if args.verbose:
        for i, layer in enumerate(proof.layers):
            print(f"    Layer {i}: len={layer.length}, beta={layer.beta}")
    print()

    # === VERIFIER: Verify FRI proof ===
    print("[Verifier] Verifying FRI proof...")
    t0 = time.time()
    valid = fri_verify(
        fri_cfg=fri_cfg,
        vc=vc,
        base_commitment=commitment,
        proof=proof,
        expected_query_points=query_indices,
        expected_values=query_values,
    )
    t_verify = time.time() - t0

    print(f"  Result:      {'VALID ✓' if valid else 'INVALID ✗'}")
    print(f"  Time:        {t_verify*1000:.1f} ms")
    print()

    # === Summary ===
    print("=" * 60)
    if valid:
        print("  PROOF VERIFIED - Prover knows a low-degree polynomial")
        print("  whose evaluations commit to the Merkle root!")
    else:
        print("  PROOF REJECTED - Verification failed")
    print("=" * 60)
    print()
    print(f"Timing Summary:")
    print(f"  Eval generation: {t_gen*1000:.1f} ms")
    print(f"  Commitment:      {t_commit*1000:.1f} ms")
    print(f"  Prove:           {t_prove*1000:.1f} ms")
    print(f"  Verify:          {t_verify*1000:.1f} ms")
    print(f"  Total:           {(t_gen+t_commit+t_prove+t_verify)*1000:.1f} ms")

    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
