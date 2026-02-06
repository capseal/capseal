#!/usr/bin/env python3
"""Lightweight fuzzing harness for the geometry AIR + zk proof."""
from __future__ import annotations

import argparse
import copy
import random

from bef_zk.air.geom_air import GeomAIRParams, GeomInitialState
from bef_zk.fri.config import FRIConfig
from bef_zk.stc.vc import STCVectorCommitment, VectorCommitment
from bef_zk.zk_geom.prover import zk_prove_geom
from bef_zk.zk_geom.verifier import zk_verify_geom
from bef_zk.zk_geom.types import GeomProof

PROGRAM = [1, 2, 3, 1, 2, 0]
MODULUS = (1 << 61) - 1


def random_params(steps_pow: int) -> GeomAIRParams:
    steps = 1 << steps_pow
    matrix = [[2, 1], [1, 1]]
    r_challenges = [random.randrange(1, MODULUS) for _ in range(2)]
    return GeomAIRParams(
        steps=steps,
        num_challenges=2,
        r_challenges=r_challenges,
        matrix=matrix,
    )


def random_init_state() -> GeomInitialState:
    gas = random.randrange(1, MODULUS)
    acc = random.randrange(1, MODULUS)
    return GeomInitialState(gas=gas, acc=acc)


def fri_config(steps: int) -> FRIConfig:
    domain_size = 1 << (steps - 1).bit_length()
    num_rounds = max(1, min(6, domain_size.bit_length() - 1))
    return FRIConfig(
        field_modulus=MODULUS,
        domain_size=domain_size,
        max_degree=steps - 1,
        num_rounds=num_rounds,
        num_queries=8,
    )


def fuzz_once(seed: int, steps_pow: int) -> bool:
    random.seed(seed)
    params = random_params(steps_pow)
    init_state = random_init_state()
    fri_cfg = fri_config(params.steps)
    vc: VectorCommitment = STCVectorCommitment(chunk_len=256)

    proof: GeomProof = zk_prove_geom(PROGRAM, params, init_state, fri_cfg, vc)
    ok, _ = zk_verify_geom(PROGRAM, params, init_state, vc, proof)
    if not ok:
        print(f"[FAIL] honest proof rejected (seed={seed}, steps=2^{steps_pow})")
        return False

    corrupted = copy.deepcopy(proof)
    corrupted.statement.final_cnt ^= 1
    ok_bad, _ = zk_verify_geom(PROGRAM, params, init_state, vc, corrupted)
    if ok_bad:
        print(f"[FAIL] corrupted proof accepted (seed={seed}, steps=2^{steps_pow})")
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuzz geometry AIR + zk prover")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--min-pow", type=int, default=4, help="min log2 steps")
    parser.add_argument("--max-pow", type=int, default=8, help="max log2 steps")
    args = parser.parse_args()

    for i in range(args.runs):
        steps_pow = random.randint(args.min_pow, args.max_pow)
        if not fuzz_once(i, steps_pow):
            sys.exit(1)
    print("[OK] fuzzing completed")


if __name__ == "__main__":
    main()
