#!/usr/bin/env python3
"""Sanity check that capsule bin/json payloads agree."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bef_zk.codec import canonical_decode


def main() -> None:
    parser = argparse.ArgumentParser(description="Capsule binary smoke test")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("out/demo_geom"),
        help="pipeline output directory",
    )
    args = parser.parse_args()
    json_path = args.dir / "strategy_capsule.json"
    bin_path = args.dir / "strategy_capsule.bin"
    if not json_path.exists() or not bin_path.exists():
        raise SystemExit("both JSON and BIN capsules are required for this check")
    json_capsule = json.loads(json_path.read_text())
    bin_capsule = canonical_decode(bin_path.read_bytes())
    if json_capsule != bin_capsule:
        raise SystemExit("capsule json != bin payload")
    print("Capsule JSON/BIN payloads match.")


if __name__ == "__main__":
    main()
