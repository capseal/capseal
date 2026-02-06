#!/usr/bin/env python3
"""CLI for STC PC commitments (vector commit/open/verify)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from bef_zk.stc.pc_backend import (
    STCCommitment,
    STCIndexProof,
    stc_build_pc_commitment,
    stc_open_index,
    stc_verify_index,
)
from bef_zk.stc.aok_cpu import StreamingAccumulatorCPU


def trace_to_values(trace: dict) -> List[int]:
    values: List[int] = []
    for chunk in trace.get("chunks", []):
        values.extend(int(v) for v in chunk.get("values", []))
    return values


def load_trace(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema") != "bef_trace_v1":
        raise ValueError("unexpected schema")
    return data


def command_commit(args: argparse.Namespace) -> None:
    trace = load_trace(args.trace)
    chunk_len = args.chunk_len or int(trace.get("chunk_length", 1024))
    acc = StreamingAccumulatorCPU(num_challenges=0, chunk_len=chunk_len)
    for chunk in trace["chunks"]:
        acc.add_chunk(chunk["values"])
    commitment = stc_build_pc_commitment(acc)
    args.output.write_text(json.dumps(commitment.to_json(), indent=2))
    print(f"PC commitment written to {args.output}")


def command_open(args: argparse.Namespace) -> None:
    trace = load_trace(args.trace)
    values = trace_to_values(trace)
    commitment = STCCommitment.from_json(json.loads(Path(args.commit).read_text()))
    acc = StreamingAccumulatorCPU(num_challenges=0, chunk_len=commitment.chunk_len)
    for chunk in trace["chunks"]:
        acc.add_chunk(chunk["values"])
    prover_commit = stc_build_pc_commitment(acc)
    if prover_commit.global_root != commitment.global_root:
        raise SystemExit("commitment mismatch between trace and provided commitment")
    proof = stc_open_index(values, prover_commit, args.index)
    args.output.write_text(json.dumps(proof.to_json(), indent=2))
    print(f"Opening for index {args.index} written to {args.output}")


def command_verify(args: argparse.Namespace) -> None:
    commitment = STCCommitment.from_json(json.loads(Path(args.commit).read_text()))
    proof = STCIndexProof.from_json(json.loads(Path(args.proof).read_text()))
    if stc_verify_index(commitment, proof):
        print("PC verify: OK")
    else:
        raise SystemExit("PC verify: FAILED")


def main() -> None:
    parser = argparse.ArgumentParser(description="STC PC backend helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("commit-trace", help="Commit a bef_trace_v1 vector")
    pc.add_argument("trace", type=Path)
    pc.add_argument("output", type=Path)
    pc.add_argument("--chunk-len", type=int, help="Chunk length (defaults to trace chunk_length or 1024)")

    op = sub.add_parser("open", help="Open commitment at an index")
    op.add_argument("trace", type=Path, help="Trace JSON used for commitment")
    op.add_argument("commit", help="PC commitment JSON")
    op.add_argument("index", type=int, help="Global index to open")
    op.add_argument("output", type=Path)

    vr = sub.add_parser("verify", help="Verify index opening")
    vr.add_argument("commit", help="PC commitment JSON")
    vr.add_argument("proof", help="PC opening JSON")

    args = parser.parse_args()
    if args.cmd == "commit-trace":
        command_commit(args)
    elif args.cmd == "open":
        command_open(args)
    elif args.cmd == "verify":
        command_verify(args)


if __name__ == "__main__":
    main()
