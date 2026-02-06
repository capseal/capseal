#!/usr/bin/env python3
"""Test FRI opening parity: verify Rust open_batch matches Python.

This test:
1. Creates a test codeword
2. Commits using both Python (STCVectorCommitment) and Rust (PyFriState)
3. Opens the same indices with both
4. Compares proof structures field-by-field
5. Verifies Rust proofs with Python verifier

CRITICAL: Milestone 2 is complete when this test passes.

Usage:
    PYTHONPATH=. python scripts/test_opening_parity.py --size 256
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import bef_rust
from bef_zk.stc.vc import STCVectorCommitment, VCCommitment, VCBatchProof, VCBatchEntry, ChunkLeafProof
from bef_zk.stc.merkle import MerkleMultiProof, verify_multiproof
from bef_zk.stc.aok_cpu import MODULUS


def rust_proof_to_python(rust_proof: bef_rust.PyBatchProof, commitment: VCCommitment) -> VCBatchProof:
    """Convert Rust PyBatchProof to Python VCBatchProof for verification."""
    entries = [
        VCBatchEntry(
            index=e.index,
            value=e.value,
            chunk_index=e.chunk_index,
            chunk_offset=e.chunk_offset,
            leaf_pos=e.leaf_pos,
            leaf_path=None,  # We use chunk_leaf_proofs instead
        )
        for e in rust_proof.entries
    ]

    # Convert chunk_proof (k-ary multiproof over chunk roots)
    chunk_proof = MerkleMultiProof(
        tree_size=rust_proof.chunk_proof.tree_size,
        arity=rust_proof.chunk_proof.arity,
        sibling_levels=[
            [bytes.fromhex(h) for h in level]
            for level in rust_proof.chunk_proof.sibling_levels
        ],
    )

    # Convert chunk_leaf_proofs (binary multiproofs within chunks)
    chunk_leaf_proofs = [
        ChunkLeafProof(
            chunk_index=clp.chunk_index,
            chunk_offset=clp.chunk_offset,
            leaf_positions=clp.leaf_positions,
            proof=MerkleMultiProof(
                tree_size=clp.proof.tree_size,
                arity=clp.proof.arity,
                sibling_levels=[
                    [bytes.fromhex(h) for h in level]
                    for level in clp.proof.sibling_levels
                ],
            ),
        )
        for clp in rust_proof.chunk_leaf_proofs
    ]

    chunk_roots = [bytes.fromhex(h) for h in rust_proof.chunk_roots]

    return VCBatchProof(
        entries=entries,
        chunk_positions=rust_proof.chunk_positions,
        chunk_roots=chunk_roots,
        chunk_proof=chunk_proof,
        chunk_leaf_proofs=chunk_leaf_proofs,
    )


def compare_multiproofs(name: str, py_proof: MerkleMultiProof, rust_proof: bef_rust.PyMerkleMultiProof) -> bool:
    """Compare Python and Rust multiproofs."""
    all_ok = True

    if py_proof.tree_size != rust_proof.tree_size:
        print(f"  {name}: tree_size mismatch: py={py_proof.tree_size} rust={rust_proof.tree_size}")
        all_ok = False

    if py_proof.arity != rust_proof.arity:
        print(f"  {name}: arity mismatch: py={py_proof.arity} rust={rust_proof.arity}")
        all_ok = False

    if len(py_proof.sibling_levels) != len(rust_proof.sibling_levels):
        print(f"  {name}: sibling_levels length mismatch: py={len(py_proof.sibling_levels)} rust={len(rust_proof.sibling_levels)}")
        all_ok = False
    else:
        for level_idx, (py_level, rust_level) in enumerate(zip(py_proof.sibling_levels, rust_proof.sibling_levels)):
            py_level_hex = [h.hex() for h in py_level]
            if py_level_hex != rust_level:
                print(f"  {name}: sibling_levels[{level_idx}] mismatch:")
                print(f"    py:   {py_level_hex[:3]}...")
                print(f"    rust: {rust_level[:3]}...")
                all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="FRI opening parity test")
    parser.add_argument("--size", type=int, default=256, help="Codeword size (power of 2)")
    parser.add_argument("--chunk-len", type=int, default=64, help="Chunk length")
    parser.add_argument("--arity", type=int, default=None, help="Chunk tree arity (default: from env or 16)")
    args = parser.parse_args()

    size = args.size
    if size & (size - 1) != 0:
        print(f"Error: size must be power of 2, got {size}")
        sys.exit(1)

    chunk_len = args.chunk_len
    chunk_tree_arity = args.arity or int(os.environ.get("STC_CHUNK_TREE_ARITY", "16"))

    print("FRI Opening Parity Test")
    print(f"  Size: {size}")
    print(f"  Chunk len: {chunk_len}")
    print(f"  Chunk tree arity: {chunk_tree_arity}")
    print()

    # Generate deterministic test codeword
    print("Generating test codeword...")
    base_evals = [(i * 31337 + 12345) % MODULUS for i in range(size)]

    # Test indices (spread across multiple chunks)
    num_chunks = (size + chunk_len - 1) // chunk_len
    test_indices = [0, chunk_len - 1, chunk_len, size - 1]  # First chunk, last of first, second chunk, last
    test_indices = [i for i in test_indices if i < size]
    test_indices = sorted(set(test_indices))
    print(f"Test indices: {test_indices}")

    # ========================================
    # Python path
    # ========================================
    print("\n--- Python Path ---")
    py_vc = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)
    py_commit = py_vc.commit(base_evals)
    print(f"Python root: {py_commit.root.hex()[:16]}...")

    py_batch = py_vc.open_batch(py_commit, test_indices)
    print(f"Python batch: {len(py_batch.entries)} entries, {len(py_batch.chunk_positions)} chunks")

    # ========================================
    # Rust path
    # ========================================
    print("\n--- Rust Path ---")
    rust_state = bef_rust.PyFriState(base_evals)
    rust_result = rust_state.commit_and_cache(chunk_len, chunk_tree_arity=chunk_tree_arity)
    rust_root = bytes.fromhex(rust_result.root_hex)
    print(f"Rust root: {rust_root.hex()[:16]}...")

    rust_batch = rust_state.open_batch(0, test_indices)
    print(f"Rust batch: {len(rust_batch.entries)} entries, {len(rust_batch.chunk_positions)} chunks")

    # ========================================
    # Compare roots
    # ========================================
    print("\n" + "=" * 60)
    print("ROOT COMPARISON")
    print("=" * 60)

    all_ok = True

    if py_commit.root == rust_root:
        print(f"  Root: MATCH ({py_commit.root.hex()[:16]}...)")
    else:
        print(f"  Root: MISMATCH")
        print(f"    Python: {py_commit.root.hex()[:16]}...")
        print(f"    Rust:   {rust_root.hex()[:16]}...")
        all_ok = False

    # ========================================
    # Compare batch structure
    # ========================================
    print("\n" + "=" * 60)
    print("BATCH STRUCTURE COMPARISON")
    print("=" * 60)

    # Entries
    if len(py_batch.entries) != len(rust_batch.entries):
        print(f"  Entry count: MISMATCH py={len(py_batch.entries)} rust={len(rust_batch.entries)}")
        all_ok = False
    else:
        print(f"  Entry count: MATCH ({len(py_batch.entries)})")
        for i, (py_e, rust_e) in enumerate(zip(py_batch.entries, rust_batch.entries)):
            if (py_e.index, py_e.value, py_e.chunk_index, py_e.leaf_pos) != (rust_e.index, rust_e.value, rust_e.chunk_index, rust_e.leaf_pos):
                print(f"    Entry {i}: MISMATCH")
                print(f"      py:   idx={py_e.index} val={py_e.value} chunk={py_e.chunk_index} leaf={py_e.leaf_pos}")
                print(f"      rust: idx={rust_e.index} val={rust_e.value} chunk={rust_e.chunk_index} leaf={rust_e.leaf_pos}")
                all_ok = False

    # Chunk positions
    if py_batch.chunk_positions != rust_batch.chunk_positions:
        print(f"  Chunk positions: MISMATCH py={py_batch.chunk_positions} rust={rust_batch.chunk_positions}")
        all_ok = False
    else:
        print(f"  Chunk positions: MATCH ({py_batch.chunk_positions})")

    # Chunk roots
    py_chunk_roots_hex = [r.hex() for r in py_batch.chunk_roots]
    if py_chunk_roots_hex != rust_batch.chunk_roots:
        print(f"  Chunk roots: MISMATCH")
        for i, (py_r, rust_r) in enumerate(zip(py_chunk_roots_hex, rust_batch.chunk_roots)):
            if py_r != rust_r:
                print(f"    {i}: py={py_r[:16]}... rust={rust_r[:16]}...")
        all_ok = False
    else:
        print(f"  Chunk roots: MATCH ({len(py_batch.chunk_roots)} roots)")

    # ========================================
    # Compare chunk proof (k-ary multiproof)
    # ========================================
    print("\n" + "=" * 60)
    print("CHUNK PROOF COMPARISON (k-ary)")
    print("=" * 60)

    if not compare_multiproofs("chunk_proof", py_batch.chunk_proof, rust_batch.chunk_proof):
        all_ok = False
    else:
        print("  chunk_proof: MATCH")

    # ========================================
    # Compare chunk leaf proofs (binary multiproofs)
    # ========================================
    print("\n" + "=" * 60)
    print("CHUNK LEAF PROOFS COMPARISON (binary)")
    print("=" * 60)

    if len(py_batch.chunk_leaf_proofs) != len(rust_batch.chunk_leaf_proofs):
        print(f"  Count: MISMATCH py={len(py_batch.chunk_leaf_proofs)} rust={len(rust_batch.chunk_leaf_proofs)}")
        all_ok = False
    else:
        print(f"  Count: MATCH ({len(py_batch.chunk_leaf_proofs)})")
        for i, (py_clp, rust_clp) in enumerate(zip(py_batch.chunk_leaf_proofs, rust_batch.chunk_leaf_proofs)):
            clp_ok = True
            if py_clp.chunk_index != rust_clp.chunk_index:
                print(f"  CLP {i}: chunk_index mismatch py={py_clp.chunk_index} rust={rust_clp.chunk_index}")
                clp_ok = False
            if py_clp.chunk_offset != rust_clp.chunk_offset:
                print(f"  CLP {i}: chunk_offset mismatch py={py_clp.chunk_offset} rust={rust_clp.chunk_offset}")
                clp_ok = False
            if py_clp.leaf_positions != rust_clp.leaf_positions:
                print(f"  CLP {i}: leaf_positions mismatch py={py_clp.leaf_positions} rust={rust_clp.leaf_positions}")
                clp_ok = False
            if not compare_multiproofs(f"CLP {i}", py_clp.proof, rust_clp.proof):
                clp_ok = False
            if clp_ok:
                print(f"  CLP {i}: MATCH (chunk={py_clp.chunk_index}, positions={py_clp.leaf_positions})")
            else:
                all_ok = False

    # ========================================
    # Verify Rust proof with Python verifier
    # ========================================
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)

    # Convert Rust proof to Python format and verify
    if all_ok:
        rust_as_python = rust_proof_to_python(rust_batch, py_commit)

        # Create a commitment matching the Rust root
        rust_commit = VCCommitment(
            root=rust_root,
            length=rust_result.length,
            chunk_len=chunk_len,
            num_chunks=rust_result.num_chunks,
            challenges=[],
            sketches=[],
            powers=[],
            chunk_tree_arity=chunk_tree_arity,
        )

        try:
            verified = py_vc.verify_batch(rust_commit, rust_as_python)
            if verified:
                print("  Rust proof verified by Python: PASS")
            else:
                print("  Rust proof verified by Python: FAIL")
                all_ok = False
        except Exception as e:
            print(f"  Rust proof verification error: {e}")
            all_ok = False
    else:
        print("  Skipping verification (structural mismatches found)")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 60)
    if all_ok:
        print("RESULT: ALL PARITY CHECKS PASSED")
    else:
        print("RESULT: PARITY CHECKS FAILED")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
