#!/usr/bin/env python3
"""Archive parity test: verify JSON and Binary archives produce identical results.

This test:
1. Generates a trace with configurable steps
2. Commits with JSON archive
3. Commits with Binary archive
4. Verifies chunk values, roots, and global root are identical

Usage:
    PYTHONPATH=. python scripts/test_archive_parity.py --steps 8192
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bef_zk.stc.archive import ChunkArchive, BinaryChunkArchive
from bef_zk.stc.aok_cpu import (
    StreamingAccumulatorCPU,
    ROOT_SEED,
    merkle_from_values,
)


def generate_trace(num_steps: int, row_width: int = 4) -> list[list[int]]:
    """Generate a deterministic test trace."""
    rows = []
    for i in range(num_steps):
        row = [(i * row_width + j) % (2**61 - 1) for j in range(row_width)]
        rows.append(row)
    return rows


def commit_with_json_archive(
    rows: list[list[int]],
    chunk_len: int,
    archive_dir: Path,
) -> tuple[bytes, list[bytes], list[list[int]]]:
    """Commit trace using JSON archive, return (root, chunk_roots, chunk_values)."""
    archive = ChunkArchive(root_dir=archive_dir)
    acc = StreamingAccumulatorCPU(
        num_challenges=2,
        chunk_len=chunk_len,
        challenge_seed=ROOT_SEED,
        archive=archive,
    )

    # Flatten rows and feed as chunks
    flat = []
    for row in rows:
        flat.extend(row)

    for i in range(0, len(flat), chunk_len):
        acc.add_chunk(flat[i : i + chunk_len])

    # Collect results
    root = acc.root
    chunk_roots = [c.root for c in acc.chunks]

    # Load chunk values back from archive
    chunk_values = []
    for i in range(len(acc.chunks)):
        vals = archive.load_chunk_by_index(i)
        chunk_values.append(vals)

    return root, chunk_roots, chunk_values


def commit_with_binary_archive(
    rows: list[list[int]],
    chunk_len: int,
    archive_dir: Path,
) -> tuple[bytes, list[bytes], list[list[int]]]:
    """Commit trace using Binary archive, return (root, chunk_roots, chunk_values)."""
    archive = BinaryChunkArchive(root_dir=archive_dir)

    # We need to manually track chunks since BinaryChunkArchive doesn't
    # integrate with StreamingAccumulatorCPU yet
    flat = []
    for row in rows:
        flat.extend(row)

    # Use a separate accumulator that writes to binary archive
    acc = StreamingAccumulatorCPU(
        num_challenges=2,
        chunk_len=chunk_len,
        challenge_seed=ROOT_SEED,
        archive=None,  # We'll manually archive
    )

    chunk_idx = 0
    for i in range(0, len(flat), chunk_len):
        chunk_data = flat[i : i + chunk_len]
        acc.add_chunk(chunk_data)
        archive.store_chunk(chunk_idx, chunk_data)
        chunk_idx += 1

    archive.finalize()

    # Collect results
    root = acc.root
    chunk_roots = [c.root for c in acc.chunks]

    # Reload and read back
    archive2 = BinaryChunkArchive(root_dir=archive_dir)
    chunk_values = []
    for i in range(len(acc.chunks)):
        vals = archive2.load_chunk_by_index(i)
        chunk_values.append(vals)
    archive2.close_reader()

    return root, chunk_roots, chunk_values


def verify_chunk_roots(chunk_values: list[list[int]], chunk_roots: list[bytes]) -> bool:
    """Verify chunk values hash to their declared roots."""
    for i, (values, expected_root) in enumerate(zip(chunk_values, chunk_roots)):
        offset = sum(len(chunk_values[j]) for j in range(i))
        computed_root = merkle_from_values(values, offset)
        if computed_root != expected_root:
            print(f"  ✗ Chunk {i} root mismatch!")
            print(f"    Expected: {expected_root.hex()}")
            print(f"    Got:      {computed_root.hex()}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Archive parity test")
    parser.add_argument("--steps", type=int, default=8192, help="Number of trace steps")
    parser.add_argument("--row-width", type=int, default=4, help="Row width")
    parser.add_argument("--chunk-len", type=int, default=256, help="Chunk length")
    args = parser.parse_args()

    print(f"Archive Parity Test")
    print(f"  Steps: {args.steps}")
    print(f"  Row width: {args.row_width}")
    print(f"  Chunk len: {args.chunk_len}")
    print()

    # Generate trace
    print("Generating trace...")
    start = time.time()
    rows = generate_trace(args.steps, args.row_width)
    print(f"  Generated {len(rows)} rows in {time.time() - start:.3f}s")

    with tempfile.TemporaryDirectory(prefix="parity_json_") as json_dir:
        with tempfile.TemporaryDirectory(prefix="parity_bin_") as bin_dir:
            # Commit with JSON archive
            print("\nCommitting with JSON archive...")
            start = time.time()
            json_root, json_chunk_roots, json_chunk_values = commit_with_json_archive(
                rows, args.chunk_len, Path(json_dir)
            )
            json_time = time.time() - start
            print(f"  Root: {json_root.hex()[:16]}...")
            print(f"  Chunks: {len(json_chunk_roots)}")
            print(f"  Time: {json_time:.3f}s")

            # Commit with Binary archive
            print("\nCommitting with Binary archive...")
            start = time.time()
            bin_root, bin_chunk_roots, bin_chunk_values = commit_with_binary_archive(
                rows, args.chunk_len, Path(bin_dir)
            )
            bin_time = time.time() - start
            print(f"  Root: {bin_root.hex()[:16]}...")
            print(f"  Chunks: {len(bin_chunk_roots)}")
            print(f"  Time: {bin_time:.3f}s")

            # Compare results
            print("\n" + "=" * 60)
            print("PARITY CHECKS")
            print("=" * 60)

            all_ok = True

            # Check 1: Global roots match
            if json_root == bin_root:
                print("✓ Global roots match")
            else:
                print("✗ Global roots DIFFER!")
                print(f"  JSON: {json_root.hex()}")
                print(f"  Binary: {bin_root.hex()}")
                all_ok = False

            # Check 2: Chunk counts match
            if len(json_chunk_roots) == len(bin_chunk_roots):
                print(f"✓ Chunk counts match ({len(json_chunk_roots)})")
            else:
                print(f"✗ Chunk counts DIFFER! JSON={len(json_chunk_roots)}, Bin={len(bin_chunk_roots)}")
                all_ok = False

            # Check 3: All chunk roots match
            roots_match = all(j == b for j, b in zip(json_chunk_roots, bin_chunk_roots))
            if roots_match:
                print("✓ All chunk roots match")
            else:
                print("✗ Some chunk roots DIFFER!")
                for i, (j, b) in enumerate(zip(json_chunk_roots, bin_chunk_roots)):
                    if j != b:
                        print(f"  Chunk {i}: JSON={j.hex()[:16]}... Bin={b.hex()[:16]}...")
                all_ok = False

            # Check 4: All chunk values match
            values_match = all(j == b for j, b in zip(json_chunk_values, bin_chunk_values))
            if values_match:
                print("✓ All chunk values match")
            else:
                print("✗ Some chunk values DIFFER!")
                for i, (j, b) in enumerate(zip(json_chunk_values, bin_chunk_values)):
                    if j != b:
                        print(f"  Chunk {i}: len(JSON)={len(j)}, len(Bin)={len(b)}")
                all_ok = False

            # Check 5: Verify chunk roots from values (JSON)
            print("\nVerifying JSON chunk roots from values...")
            if verify_chunk_roots(json_chunk_values, json_chunk_roots):
                print("✓ JSON chunk roots verified")
            else:
                all_ok = False

            # Check 6: Verify chunk roots from values (Binary)
            print("Verifying Binary chunk roots from values...")
            if verify_chunk_roots(bin_chunk_values, bin_chunk_roots):
                print("✓ Binary chunk roots verified")
            else:
                all_ok = False

            print("\n" + "=" * 60)
            if all_ok:
                print("RESULT: ALL PARITY CHECKS PASSED")
                speedup = json_time / bin_time if bin_time > 0 else 0
                print(f"Binary archive is {speedup:.1f}x faster")
            else:
                print("RESULT: PARITY CHECKS FAILED")
                sys.exit(1)
            print("=" * 60)


if __name__ == "__main__":
    main()
