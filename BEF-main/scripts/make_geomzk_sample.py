#!/usr/bin/env python3
"""Generate toy geomzk proofs (Merkle or STC) for the C++ verifier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bef_zk.zk_geom.backend import get_row_backend

MOD = (1 << 61) - 1


def build_rows(steps: int) -> list[list[int]]:
    rows: list[list[int]] = []
    x1, x2 = 1, 1
    for _ in range(steps):
        rows.append([x1 % MOD, x2 % MOD])
        nx1 = (2 * x1 + x2) % MOD
        nx2 = (x1 + x2) % MOD
        x1, x2 = nx1, nx2
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy proofs for geomzk CLI")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument(
        "--backend",
        choices=["geom_plain_fri", "geom_stc_fri"],
        default="geom_plain_fri",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output path (default: geomzk/examples/<backend>_sample.json)",
    )
    args = parser.parse_args()

    rows = build_rows(args.steps)
    row_width = len(rows[0])
    backend = get_row_backend(args.backend, row_width)
    commitment = backend.commit_rows(rows)

    openings = []
    for idx in range(len(rows)):
        row_values, proof_dict = backend.open_row(commitment, idx)
        openings.append(
            {
                "backend": args.backend,
                "row_index": idx,
                "row_values": [int(v) for v in row_values],
                "proof": proof_dict,
            }
        )

    params = dict(commitment.params)
    row_commit = {
        "backend": args.backend,
        "row_size": row_width,
        "row_width": row_width,
        "n_rows": len(rows),
        "params": params,
    }
    if "root" in params:
        row_commit["root"] = params["root"]
    if "length" in params:
        row_commit["length"] = params["length"]

    proof = {
        "statement": {"steps": args.steps},
        "row_commitment": row_commit,
        "row_openings": openings,
        "fri_params": {
            "domain_size": args.steps,
            "max_degree": args.steps - 1,
            "num_rounds": 1,
            "num_queries": min(args.steps, 8),
        },
        "fri_proof": {"layers": [], "batches": []},
    }

    if args.output is None:
        default_name = "merkle_sample.json" if args.backend == "geom_plain_fri" else "stc_sample.json"
        out_path = ROOT / "geomzk" / "examples" / default_name
    else:
        out_path = args.output
    out_path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
