"""Convert a BEF trace JSON dump into a GPU sketch JSON file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from stream_accumulator import (
    DEFAULT_MODULUS,
    DEFAULT_NUM_CHALLENGES,
    StreamingAccumulatorCUDA,
)


def load_trace(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text())
    if data.get("schema") != "bef_trace_v1":
        raise ValueError(f"unexpected schema {data.get('schema')}")
    if "chunks" not in data:
        raise ValueError("trace JSON missing 'chunks'")
    return data


def run_sketch(trace_data: Dict[str, Any], *, seed: int, modulus: int, num_challenges: int) -> Dict[str, Any]:
    acc = StreamingAccumulatorCUDA(seed=seed, modulus=modulus, num_challenges=num_challenges)
    for idx, chunk in enumerate(trace_data["chunks"]):
        values = chunk.get("values")
        if not isinstance(values, list):
            raise ValueError(f"chunk {idx} missing 'values' list")
        acc.add_chunk(values)
    proof = acc.prove()

    trace_commitment = proof.get("trace_commitment")

    sketch = {
        "schema": "bef_sketch_v1",
        "trace_id": trace_data.get("trace_id", "unknown"),
        "field_modulus": modulus,
        "seed": seed,
        "length": proof["length"],
        "challenge": proof["r"],
        "global_sketch": proof["global_sketch"],
        "challenges": proof.get("challenges"),
        "global_sketch_vec": proof.get("global_sketch_vec"),
        "trace_commitment": trace_commitment,
        "commitment_root": proof.get("commitment_root"),
        "chunks": [],
        "timing_ms": {
            "cuda_rpow": proof.get("cuda_rpow_ms"),
            "cuda_chunks": proof.get("cuda_chunks_ms"),
            "cuda_fused_global": proof.get("cuda_fused_global_ms"),
            "cuda_fused_chunks": proof.get("cuda_fused_chunks_ms"),
        },
    }

    for idx, (src_chunk, proof_chunk) in enumerate(zip(trace_data["chunks"], proof["chunks"])):
        sketch["chunks"].append(
            {
                "chunk_index": idx,
                "offset": proof_chunk["offset"],
                "length": proof_chunk["length"],
                "root_hex": proof_chunk["root"],
                "sketch_vec": proof_chunk["sketch_vec"],
            }
        )

    return sketch


def main() -> None:
    parser = argparse.ArgumentParser(description="Sketch a BEF trace JSON with the CUDA accumulator.")
    parser.add_argument("trace", type=Path, help="Path to bef_trace_v1 JSON file")
    parser.add_argument("out", type=Path, help="Path to write bef_sketch_v1 JSON output")
    parser.add_argument("--seed", type=lambda s: int(s, 0), default="0xC0FFEE", help="Seed for accumulation challenge")
    parser.add_argument("--modulus", type=int, default=DEFAULT_MODULUS, help="Field modulus (must match compiled CUDA modulus)")
    parser.add_argument("--num-challenges", type=int, default=DEFAULT_NUM_CHALLENGES, help="Number of hash-derived challenges to use")
    args = parser.parse_args()

    trace_data = load_trace(args.trace)
    sketch = run_sketch(
        trace_data,
        seed=args.seed,
        modulus=args.modulus,
        num_challenges=args.num_challenges,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sketch, indent=2))
    print(f"Wrote sketch to {args.out}")


if __name__ == "__main__":
    main()
