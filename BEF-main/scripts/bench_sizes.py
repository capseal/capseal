#!/usr/bin/env python3
"""Report JSON vs binary artifact sizes for a pipeline run."""
from __future__ import annotations

import argparse
from pathlib import Path


def _fmt(size: int | None) -> str:
    if size is None:
        return "--"
    return f"{size:,}"


def _ratio(a: int | None, b: int | None) -> str:
    if a is None or b is None or b == 0:
        return "--"
    return f"{a / b:.2f}x"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare artifact sizes")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("out/demo_geom"),
        help="pipeline output directory",
    )
    args = parser.parse_args()
    base = args.dir
    proof_json = base / "geom_proof.json"
    proof_bin = base / "geom_proof.bin"
    capsule_json = base / "strategy_capsule.json"
    capsule_bin = base / "strategy_capsule.bin"
    chunk_json = (base / "row_archive" / "chunk_roots.json").resolve()
    chunk_bin = (base / "row_archive" / "chunk_roots.bin").resolve()

    entries = [
        ("Geom proof", proof_json, proof_bin),
        ("Capsule", capsule_json, capsule_bin),
        ("Chunk roots", chunk_json, chunk_bin),
    ]
    print(f"Artifacts in {base}")
    print(f"{'Name':<15} {'JSON (bytes)':>15} {'BIN (bytes)':>15} {'bin/json':>10}")
    for name, json_path, bin_path in entries:
        json_size = json_path.stat().st_size if json_path.exists() else None
        bin_size = bin_path.stat().st_size if bin_path.exists() else None
        print(
            f"{name:<15} {_fmt(json_size):>15} {_fmt(bin_size):>15} {_ratio(bin_size, json_size):>10}"
        )


if __name__ == "__main__":
    main()
