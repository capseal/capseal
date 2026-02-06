#!/usr/bin/env python3
"""Mutate fields in a geom proof transcript to ensure verifier rejects corrupted proofs."""
from __future__ import annotations

import argparse
import copy

from bef_zk.zk_geom.serialization import proof_from_json, proof_to_json


def flip_int(value: int) -> int:
    return value ^ 1


def mutate_statement(proof):
    mutated = copy.deepcopy(proof)
    mutated.statement.final_m11 = flip_int(mutated.statement.final_m11)
    return mutated


def mutate_row_commitment(proof):
    mutated = copy.deepcopy(proof)
    row_commit = mutated.row_commitment
    if row_commit is not None:
        root_hex = row_commit.params.get("root") if row_commit.params else None
        if root_hex:
            root_bytes = bytearray(bytes.fromhex(root_hex))
            if root_bytes:
                root_bytes[0] ^= 0x01
                row_commit.params["root"] = bytes(root_bytes).hex()
    return mutated


def mutate_query_index(proof):
    mutated = copy.deepcopy(proof)
    if mutated.query_indices:
        mutated.query_indices[0] ^= 1
    return mutated


def mutate_fri_layer(proof):
    mutated = copy.deepcopy(proof)
    if mutated.fri_proof.layers:
        layer0 = mutated.fri_proof.layers[0]
        root_bytes = bytearray(layer0.commitment.root)
        if root_bytes:
            root_bytes[-1] ^= 0x01
            layer0.commitment.root = bytes(root_bytes)
    return mutated


def mutate_mask_digest(proof):
    mutated = copy.deepcopy(proof)
    digest = bytearray(mutated.mask_digest)
    if digest:
        digest[0] ^= 0x01
        mutated.mask_digest = bytes(digest)
    return mutated

MUTATIONS = {
    "statement": mutate_statement,
    "row_commitment": mutate_row_commitment,
    "query_index": mutate_query_index,
    "fri_layer": mutate_fri_layer,
    "mask_digest": mutate_mask_digest,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mutate proof transcript")
    parser.add_argument("input", help="input proof JSON")
    parser.add_argument("output", help="output corrupted JSON")
    parser.add_argument(
        "--mode",
        choices=MUTATIONS.keys(),
        default="statement",
        help="type of corruption to perform",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        proof = proof_from_json(fh.read())

    mutated = MUTATIONS[args.mode](proof)
    with open(args.output, "w", encoding="utf-8") as fh:
        fh.write(proof_to_json(mutated))


if __name__ == "__main__":
    main()
