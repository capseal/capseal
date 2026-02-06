#!/usr/bin/env python3
"""Verify a geometry STC+FRI proof from a JSON transcript."""
from __future__ import annotations

import argparse
import json

from bef_zk.stc.vc import STCVectorCommitment, VectorCommitment
from bef_zk.zk_geom.serialization import proof_from_json
from bef_zk.zk_geom.verifier import verify_proof_only


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify geom proof JSON without rerunning trace")
    parser.add_argument("input", help="Path to proof.json produced by zk_geom_demo")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        proof = proof_from_json(fh.read())

    vc: VectorCommitment = STCVectorCommitment(chunk_len=256)
    ok, stats = verify_proof_only(vc, proof)
    print(json.dumps(stats))
    print("VALID" if ok else "INVALID")


if __name__ == "__main__":
    main()
