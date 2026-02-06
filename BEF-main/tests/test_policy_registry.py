import json
from pathlib import Path

from scripts.build_policy_registry import _build_merkle_tree, _compute_hash, _merkle_proof

def test_policy_registry_merkle_roundtrip(tmp_path: Path) -> None:
    files = []
    for i in range(3):
        path = tmp_path / f"policy_{i}.txt"
        path.write_text(f"policy {i}\n")
        files.append(path)
    leaves = [_compute_hash(path.read_bytes()) for path in files]
    levels = _build_merkle_tree(leaves)
    root = levels[-1][0]
    for idx, leaf in enumerate(leaves):
        proof = _merkle_proof(levels, idx)
        computed = leaf
        pos_idx = idx
        for entry in proof:
            sibling = bytes.fromhex(entry["hash"])
            if entry["position"] == "right":
                computed = _compute_hash(computed + sibling)
            else:
                computed = _compute_hash(sibling + computed)
            pos_idx //= 2
        assert computed == root
