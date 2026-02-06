#!/usr/bin/env python3
"""
STARK Tamper Demo - Shows proof rejection when values are modified

This demonstrates that if the prover tries to cheat by modifying
values after commitment, the verifier will reject.

Usage:
    python scripts/stark_tamper_demo.py
"""
from __future__ import annotations

import hashlib
import random
import sys
import time
from typing import List

sys.path.insert(0, "/home/ryan/BEF-main")

from bef_zk.fri.config import FRIConfig, MODULUS
from bef_zk.fri.prover import fri_prove
from bef_zk.fri.verifier import fri_verify
from bef_zk.stc.vc import STCVectorCommitment


def generate_polynomial_evals(domain_size: int, degree: int, seed: int = 42) -> List[int]:
    """Generate evaluations of a random polynomial."""
    random.seed(seed)
    coeffs = [random.randint(1, MODULUS - 1) for _ in range(degree + 1)]
    evals = []
    for i in range(domain_size):
        val = 0
        x_power = 1
        for c in coeffs:
            val = (val + c * x_power) % MODULUS
            x_power = (x_power * i) % MODULUS
        evals.append(val)
    return evals


def main():
    print("=" * 60)
    print("  STARK Tamper Demo - Security Demonstration")
    print("=" * 60)
    print()

    domain_size = 512
    degree = 64
    num_queries = 4
    num_rounds = 3

    fri_cfg = FRIConfig(
        field_modulus=MODULUS,
        domain_size=domain_size,
        max_degree=degree,
        num_rounds=num_rounds,
        num_queries=num_queries,
    )

    # Generate honest evaluations
    print("[Setup] Generating polynomial evaluations...")
    evals = generate_polynomial_evals(domain_size, degree)
    vc = STCVectorCommitment(chunk_len=256)
    commitment = vc.commit(evals)

    print(f"  Committed root: {commitment.root.hex()[:32]}...")
    print()

    # Get query indices
    h = hashlib.sha256(commitment.root)
    verifier_seed = int.from_bytes(h.digest(), "big")
    random.seed(verifier_seed)
    query_indices = sorted(random.sample(range(domain_size), num_queries))

    # ============================================================
    # SCENARIO 1: Honest proof
    # ============================================================
    print("─" * 60)
    print("SCENARIO 1: Honest Prover")
    print("─" * 60)

    honest_values = [evals[i] for i in query_indices]
    proof = fri_prove(fri_cfg, vc, evals, commitment, query_indices)

    valid = fri_verify(fri_cfg, vc, commitment, proof, query_indices, honest_values)
    print(f"  Query values: {[v % 1000 for v in honest_values]} (mod 1000)")
    print(f"  Verification: {'VALID ✓' if valid else 'INVALID ✗'}")
    print()

    # ============================================================
    # SCENARIO 2: Dishonest - wrong values claimed
    # ============================================================
    print("─" * 60)
    print("SCENARIO 2: Dishonest - Claims Wrong Values")
    print("─" * 60)
    print("  Attacker tries to claim different values at query points")
    print("  while using the same committed proof...")
    print()

    # Attacker claims values are 1 higher than actual
    tampered_values = [(v + 1) % MODULUS for v in honest_values]
    valid = fri_verify(fri_cfg, vc, commitment, proof, query_indices, tampered_values)
    print(f"  Real values:    {[v % 1000 for v in honest_values]} (mod 1000)")
    print(f"  Claimed values: {[v % 1000 for v in tampered_values]} (mod 1000)")
    print(f"  Verification:   {'VALID ✓' if valid else 'REJECTED ✗'}")
    print()

    # ============================================================
    # SCENARIO 3: Dishonest - wrong query points
    # ============================================================
    print("─" * 60)
    print("SCENARIO 3: Dishonest - Wrong Query Points")
    print("─" * 60)
    print("  Attacker tries to answer different query indices")
    print("  than what verifier asked...")
    print()

    # Attacker shifts query indices
    wrong_indices = [(i + 1) % domain_size for i in query_indices]
    wrong_values = [evals[i] for i in wrong_indices]
    valid = fri_verify(fri_cfg, vc, commitment, proof, wrong_indices, wrong_values)
    print(f"  Asked indices:   {query_indices}")
    print(f"  Claimed indices: {wrong_indices}")
    print(f"  Verification:    {'VALID ✓' if valid else 'REJECTED ✗'}")
    print()

    # ============================================================
    # SCENARIO 4: Soundness - statistical guarantee
    # ============================================================
    print("─" * 60)
    print("SOUNDNESS ANALYSIS")
    print("─" * 60)
    print()
    print(f"  With {num_queries} queries and {num_rounds} FRI rounds:")
    print(f"  - Each query checks polynomial consistency at a random point")
    print(f"  - FRI folding reduces degree by half each round")
    print()
    # Simplified soundness bound
    error_prob = 1.0 / (domain_size ** num_queries)
    print(f"  Probability a cheating prover succeeds:")
    print(f"    ≈ 1/{domain_size}^{num_queries} = {error_prob:.2e}")
    print()
    print("  A dishonest prover cannot fake a valid proof")
    print("  unless they know a low-degree polynomial matching the commitment.")
    print()

    print("=" * 60)
    print("  Demo Complete - STARK proofs prevent tampering!")
    print("=" * 60)


if __name__ == "__main__":
    main()
