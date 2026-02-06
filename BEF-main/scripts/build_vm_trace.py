#!/usr/bin/env python3
"""Create a toy VM trace encoded as bef_trace_v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

PROGRAM = [
    (0x00, 0x10, 3, 1),
    (0x01, 0x20, 2, 3),
    (0x02, 0x11, 5, 4),
    (0x03, 0x30, 1, 4),
    (0x04, 0x40, 2, 3),
    (0x05, 0x21, 3, 2),
    (0x06, 0x50, 1, 2),
    (0x07, 0x60, 1, 1),
]

MODULUS = (1 << 61) - 1


def encode_step(pc: int, opcode: int, gas: int, acc: int) -> int:
    value = (pc & 0xFFFF) << 45
    value |= (opcode & 0xFF) << 37
    value |= (gas & 0xFFF) << 25
    value |= (acc & 0x1FFFFFF)
    assert value < MODULUS
    return value


def build_trace(chunk_len: int) -> dict:
    values = [encode_step(*step) for step in PROGRAM]
    chunks = []
    offset = 0
    idx = 0
    while offset < len(values):
        chunk_values = values[offset : offset + chunk_len]
        chunks.append(
            {
                "chunk_index": len(chunks),
                "offset": offset,
                "values": chunk_values,
            }
        )
        offset += len(chunk_values)
        idx += 1
    return {
        "schema": "bef_trace_v1",
        "trace_id": "vm_demo",
        "field_modulus": MODULUS,
        "num_steps": len(values),
        "vector_length": len(values),
        "chunk_length": chunk_len,
        "chunks": chunks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate toy VM trace")
    parser.add_argument("--chunk-len", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("code/traces/vm_demo.json"))
    args = parser.parse_args()

    trace = build_trace(args.chunk_len)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2))
    print(f"Wrote trace to {args.output}")


if __name__ == "__main__":
    main()
