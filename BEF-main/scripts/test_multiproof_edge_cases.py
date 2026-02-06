#!/usr/bin/env python3
"""Multiproof edge case torture tests.

Tests the landmines identified in code review:
1. tree_size semantics (un-padded count)
2. Padding rules for k-ary last groups
3. Multiproof sibling ordering
4. Duplicate/unsorted query handling
5. Value normalization (mod p)

Usage:
    PYTHONPATH=. python scripts/test_multiproof_edge_cases.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import bef_rust
from bef_zk.stc.vc import STCVectorCommitment, VCCommitment
from bef_zk.stc.merkle import MerkleMultiProof, verify_multiproof, multiproof, build_kary_levels
from bef_zk.stc.aok_cpu import MODULUS, chunk_leaf_hash


def test_tree_sizes():
    """Test tree_size = #real leaves (un-padded) across awkward sizes."""
    print("\n=== TEST: tree_size semantics ===")

    # Sizes that stress padding logic
    sizes = [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257]
    chunk_len = 64
    arity = 16

    all_ok = True
    for size in sizes:
        values = [(i * 31337) % MODULUS for i in range(size)]

        # Python commit
        py_vc = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)
        py_commit = py_vc.commit(values)

        # Rust commit
        rust_state = bef_rust.PyFriState(values)
        rust_result = rust_state.commit_and_cache(chunk_len, chunk_tree_arity=arity)
        rust_root = bytes.fromhex(rust_result.root_hex)

        if py_commit.root == rust_root:
            print(f"  size={size:3d}: root MATCH")
        else:
            print(f"  size={size:3d}: ROOT MISMATCH py={py_commit.root.hex()[:12]} rust={rust_root.hex()[:12]}")
            all_ok = False

    return all_ok


def test_padding_edge_cases():
    """Test last-group padding for various arity/size combos."""
    print("\n=== TEST: padding edge cases ===")

    test_cases = [
        # (size, chunk_len, arity, description)
        (1, 64, 2, "single element"),
        (2, 64, 2, "exactly 2 (binary full)"),
        (3, 64, 2, "3 elements (binary odd)"),
        (15, 64, 16, "15 elements (k-ary almost full)"),
        (16, 64, 16, "16 elements (k-ary exactly full)"),
        (17, 64, 16, "17 elements (k-ary overflow by 1)"),
        (31, 64, 16, "31 elements (k-ary 2 groups, last has 15)"),
        (33, 64, 16, "33 elements (k-ary 3 groups, last has 1)"),
        (65, 64, 16, "65 elements (> 1 chunk, first chunk full)"),
        (127, 64, 16, "127 elements (2 chunks, last nearly full)"),
    ]

    all_ok = True
    for size, chunk_len, arity, desc in test_cases:
        values = [(i * 31337) % MODULUS for i in range(size)]

        # Rust commit
        rust_state = bef_rust.PyFriState(values)
        rust_result = rust_state.commit_and_cache(chunk_len, chunk_tree_arity=arity)

        # Open various indices including edge positions
        indices = [0]  # first
        if size > 1:
            indices.append(size - 1)  # last
        if size > 2:
            indices.append(size // 2)  # middle

        rust_proof = rust_state.open_batch(0, indices)

        # Convert and verify with Python
        from bef_zk.fri.prover import _rust_proof_to_python
        py_proof = _rust_proof_to_python(rust_proof)

        rust_commit = VCCommitment(
            root=bytes.fromhex(rust_result.root_hex),
            length=rust_result.length,
            chunk_len=chunk_len,
            num_chunks=rust_result.num_chunks,
            challenges=[],
            sketches=[],
            powers=[],
            chunk_tree_arity=arity,
        )

        py_vc = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)
        verified = py_vc.verify_batch(rust_commit, py_proof)

        if verified:
            print(f"  {desc}: PASS")
        else:
            print(f"  {desc}: FAIL (size={size}, indices={indices})")
            all_ok = False

    return all_ok


def test_multiproof_ordering():
    """Test sibling ordering matches exactly between Rust and Python."""
    print("\n=== TEST: multiproof ordering ===")

    all_ok = True
    chunk_len = 64

    # Test cases: (size, indices, description)
    test_cases = [
        (64, [0], "single index"),
        (64, [0, 63], "first and last"),
        (64, [0, 1], "adjacent pair"),
        (64, [0, 2], "same group, gap"),
        (64, [0, 32], "different halves"),
        (64, [0, 1, 2, 3], "first 4 (same group in k-ary)"),
        (64, [60, 61, 62, 63], "last 4"),
        (128, [0, 64], "across chunks"),
        (128, [63, 64], "chunk boundary"),
        (256, [0, 127, 128, 255], "all boundaries"),
    ]

    for size, indices, desc in test_cases:
        values = [(i * 31337) % MODULUS for i in range(size)]

        # Python path
        py_vc = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)
        py_commit = py_vc.commit(values)
        py_batch = py_vc.open_batch(py_commit, indices)

        # Rust path
        rust_state = bef_rust.PyFriState(values)
        rust_state.commit_and_cache(chunk_len, chunk_tree_arity=16)
        rust_batch = rust_state.open_batch(0, indices)

        # Compare sibling_levels byte-by-byte
        match = True

        # Compare chunk_proof siblings
        py_chunk_siblings = py_batch.chunk_proof.sibling_levels
        rust_chunk_siblings = [[bytes.fromhex(h) for h in level] for level in rust_batch.chunk_proof.sibling_levels]

        if len(py_chunk_siblings) != len(rust_chunk_siblings):
            match = False
        else:
            for i, (py_level, rust_level) in enumerate(zip(py_chunk_siblings, rust_chunk_siblings)):
                if py_level != rust_level:
                    match = False
                    break

        # Compare chunk_leaf_proofs siblings
        if len(py_batch.chunk_leaf_proofs) != len(rust_batch.chunk_leaf_proofs):
            match = False
        else:
            for py_clp, rust_clp in zip(py_batch.chunk_leaf_proofs, rust_batch.chunk_leaf_proofs):
                py_siblings = py_clp.proof.sibling_levels
                rust_siblings = [[bytes.fromhex(h) for h in level] for level in rust_clp.proof.sibling_levels]
                if py_siblings != rust_siblings:
                    match = False
                    break

        if match:
            print(f"  {desc}: MATCH")
        else:
            print(f"  {desc}: MISMATCH")
            all_ok = False

    return all_ok


def test_duplicate_unsorted_indices():
    """Test handling of duplicate and unsorted query indices."""
    print("\n=== TEST: duplicate/unsorted indices ===")

    all_ok = True
    size = 128
    chunk_len = 64
    values = [(i * 31337) % MODULUS for i in range(size)]

    test_cases = [
        ([5, 3, 1, 7], "unsorted"),
        ([0, 0, 0], "all duplicates"),
        ([0, 5, 0, 10, 5], "mixed duplicates"),
        ([127, 0, 64, 32, 64, 0], "unsorted with dups"),
    ]

    for indices, desc in test_cases:
        # Python path (should handle gracefully)
        py_vc = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)
        py_commit = py_vc.commit(values)
        py_batch = py_vc.open_batch(py_commit, indices)

        # Rust path
        rust_state = bef_rust.PyFriState(values)
        rust_state.commit_and_cache(chunk_len, chunk_tree_arity=16)
        rust_batch = rust_state.open_batch(0, indices)

        # Both should produce same unique sorted entries
        py_unique = sorted(set(e.index for e in py_batch.entries))
        rust_unique = sorted(set(e.index for e in rust_batch.entries))

        # Verify the Rust proof with Python
        from bef_zk.fri.prover import _rust_proof_to_python
        rust_as_py = _rust_proof_to_python(rust_batch)

        rust_commit = VCCommitment(
            root=py_commit.root,  # Should match
            length=size,
            chunk_len=chunk_len,
            num_chunks=(size + chunk_len - 1) // chunk_len,
            challenges=[],
            sketches=[],
            powers=[],
            chunk_tree_arity=16,
        )

        verified = py_vc.verify_batch(rust_commit, rust_as_py)

        if py_unique == rust_unique and verified:
            print(f"  {desc}: PASS (unique={py_unique})")
        else:
            print(f"  {desc}: FAIL")
            print(f"    py_unique={py_unique}, rust_unique={rust_unique}, verified={verified}")
            all_ok = False

    return all_ok


def test_value_normalization():
    """Test that values are normalized mod p consistently."""
    print("\n=== TEST: value normalization ===")

    all_ok = True
    chunk_len = 64

    # Values that might cause normalization issues
    test_values = [
        0,
        1,
        MODULUS - 1,
        MODULUS,  # Should wrap to 0
        MODULUS + 1,  # Should wrap to 1
        2 * MODULUS,
        2 * MODULUS + 12345,
    ]

    # Pad to chunk_len
    while len(test_values) < chunk_len:
        test_values.append(len(test_values) * 12345)

    # Python commit
    py_vc = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)
    py_commit = py_vc.commit(test_values)

    # Rust commit
    rust_state = bef_rust.PyFriState(test_values)
    rust_result = rust_state.commit_and_cache(chunk_len, chunk_tree_arity=16)
    rust_root = bytes.fromhex(rust_result.root_hex)

    if py_commit.root == rust_root:
        print(f"  Root match: PASS")
    else:
        print(f"  Root match: FAIL")
        all_ok = False

    # Open and check values
    indices = [0, 1, 2, 3, 4, 5, 6]
    rust_batch = rust_state.open_batch(0, indices)

    for i, idx in enumerate(indices):
        expected = test_values[idx] % MODULUS
        actual = rust_batch.entries[i].value
        if expected == actual:
            print(f"  value[{idx}]: {test_values[idx]} -> {actual} MATCH")
        else:
            print(f"  value[{idx}]: expected {expected}, got {actual} MISMATCH")
            all_ok = False

    return all_ok


def test_adversarial_query_patterns():
    """Test adversarial query patterns that might break assumptions."""
    print("\n=== TEST: adversarial query patterns ===")

    all_ok = True
    size = 256
    chunk_len = 64
    values = [(i * 31337) % MODULUS for i in range(size)]

    test_cases = [
        (list(range(16)), "all indices in first k-ary group"),
        (list(range(48, 64)), "all indices in last positions of first chunk"),
        ([0, 64, 128, 192], "one per chunk"),
        (list(range(0, 256, 16)), "every 16th"),
        ([255], "only last"),
        (list(range(252, 256)), "last 4"),
    ]

    for indices, desc in test_cases:
        py_vc = STCVectorCommitment(chunk_len=chunk_len, num_challenges=2)
        py_commit = py_vc.commit(values)

        rust_state = bef_rust.PyFriState(values)
        rust_state.commit_and_cache(chunk_len, chunk_tree_arity=16)

        rust_batch = rust_state.open_batch(0, indices)

        from bef_zk.fri.prover import _rust_proof_to_python
        rust_as_py = _rust_proof_to_python(rust_batch)

        rust_commit = VCCommitment(
            root=py_commit.root,
            length=size,
            chunk_len=chunk_len,
            num_chunks=(size + chunk_len - 1) // chunk_len,
            challenges=[],
            sketches=[],
            powers=[],
            chunk_tree_arity=16,
        )

        verified = py_vc.verify_batch(rust_commit, rust_as_py)

        if verified:
            print(f"  {desc}: PASS")
        else:
            print(f"  {desc}: FAIL")
            all_ok = False

    return all_ok


def main():
    print("=" * 60)
    print("MULTIPROOF EDGE CASE TORTURE TESTS")
    print("=" * 60)

    results = []

    results.append(("tree_size semantics", test_tree_sizes()))
    results.append(("padding edge cases", test_padding_edge_cases()))
    results.append(("multiproof ordering", test_multiproof_ordering()))
    results.append(("duplicate/unsorted indices", test_duplicate_unsorted_indices()))
    results.append(("value normalization", test_value_normalization()))
    results.append(("adversarial patterns", test_adversarial_query_patterns()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print("=" * 60)
    if all_pass:
        print("ALL EDGE CASE TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
