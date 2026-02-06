#!/usr/bin/env python3
"""Build a Merkle tree over a directory of policy files."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import json

from bef_zk.stc.merkle import build_kary_levels, root_from_levels, prove_kary


def _compute_hash(data: bytes) -> bytes:
    """Helper used by tests – binary Merkle hash of a payload."""
    return hashlib.sha256(data).digest()


def _build_merkle_tree(leaves: list[bytes]) -> list[list[bytes]]:
    """Deterministic binary tree used by the legacy unit tests."""
    if not leaves:
        return []
    levels: list[list[bytes]] = [list(leaves)]
    current = list(leaves)
    while len(current) > 1:
        next_level: list[bytes] = []
        for i in range(0, len(current), 2):
            left = current[i]
            right = current[i + 1] if i + 1 < len(current) else left
            next_level.append(_compute_hash(left + right))
        levels.append(next_level)
        current = next_level
    return levels


def _merkle_proof(levels: list[list[bytes]], index: int) -> list[dict[str, str]]:
    """Return a binary proof compatible with the tests."""
    if not levels:
        return []
    proof: list[dict[str, str]] = []
    pos = index
    for level in levels[:-1]:  # skip root level
        sibling_idx = pos ^ 1
        if sibling_idx >= len(level):
            sibling_idx = pos  # duplicate when no sibling
        sibling = level[sibling_idx]
        position = "right" if (pos % 2 == 0) else "left"
        proof.append({"hash": sibling.hex(), "position": position})
        pos //= 2
    return proof

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and verify a policy registry")
    parser.add_argument("policy_dir", type=Path, help="Directory containing policy files")
    parser.add_argument("--arity", type=int, default=16, help="Arity of the Merkle tree")
    args = parser.parse_args()

    if not args.policy_dir.is_dir():
        raise NotADirectoryError(f"Policy directory not found: {args.policy_dir}")

    policy_files = sorted([p for p in args.policy_dir.iterdir() if p.is_file() and not p.name.endswith(".proof")])
    if not policy_files:
        raise FileNotFoundError("No policy files found in directory")

    print(f"Found {len(policy_files)} policy files...")
    leaf_hashes = [hashlib.sha256(p.read_bytes()).digest() for p in policy_files]

    levels = build_kary_levels(leaf_hashes, args.arity)
    root = root_from_levels(levels)

    print(f"\nPolicy Registry Merkle Root (Arity={args.arity}):")
    print(root.hex())

    for i, policy_path in enumerate(policy_files):
        proof_siblings = prove_kary(levels, i, args.arity)
        proof_path = policy_path.with_suffix(policy_path.suffix + ".proof")
        proof_data = {
            "leaf_hash": leaf_hashes[i].hex(),
            "leaf_index": i,
            "total_leaves": len(leaf_hashes),
            "arity": args.arity,
            "siblings_by_level": [
                [p.hex() for p in level] for level in proof_siblings
            ]
        }
        proof_path.write_text(json.dumps(proof_data, indent=2))
        print(f"  - Wrote inclusion proof for {policy_path.name} to {proof_path.name}")

if __name__ == "__main__":
    main()
