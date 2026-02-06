#!/usr/bin/env python3
"""Export a real geometry VM trace as bef_trace_v1 for STC/Nova."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from bef_zk.air.geom_air import GeomAIRParams, GeomInitialState, simulate_trace
from bef_zk.air.geom_trace_export import geom_trace_to_bef_trace
from scripts.geom_programs import GEOM_PROGRAM
from scripts.stc_aok import MODULUS


def derive_r_challenges(seed: str, m: int) -> list[int]:
    out: list[int] = []
    counter = 0
    seed_bytes = seed.encode("utf-8")
    while len(out) < m:
        hh = hashlib.sha256(seed_bytes + counter.to_bytes(4, "big")).digest()
        val = int.from_bytes(hh, "big") % MODULUS
        if val != 0:
            out.append(val)
        counter += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export geometry trace as bef_trace_v1")
    parser.add_argument("--steps", type=int, default=64, help="number of VM steps")
    parser.add_argument("--num-challenges", type=int, default=2, help="STC challenges (m)")
    parser.add_argument("--trace-id", type=str, default="geom_demo", help="trace identifier")
    parser.add_argument("--output", type=Path, default=Path("nova_stc/examples/geom_trace.json"))
    parser.add_argument("--challenge-seed", type=str, default="geom-default", help="seed for r_j derivation")
    args = parser.parse_args()

    params = GeomAIRParams(
        steps=args.steps,
        num_challenges=args.num_challenges,
        r_challenges=derive_r_challenges(args.challenge_seed, args.num_challenges),
        matrix=[[2, 1], [1, 1]],
    )
    init = GeomInitialState()
    trace = simulate_trace(GEOM_PROGRAM, params, init)
    bef = geom_trace_to_bef_trace(trace, trace_id=args.trace_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bef, indent=2))
    print(
        f"wrote bef_trace_v1 for steps={args.steps}, m={args.num_challenges} to {args.output}"
    )


if __name__ == "__main__":
    main()
