#!/usr/bin/env python3
"""Record a DA profile swap for an existing STC sketch."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from stc_aok import build_da_profile, verify_fast


def parse_provider(arg: Optional[str]) -> Optional[dict]:
    if not arg:
        return None
    arg = arg.strip()
    try:
        return json.loads(arg)
    except json.JSONDecodeError:
        return {"hint": arg}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create STC DA swap record")
    parser.add_argument("sketch", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--new-da-mode", required=True)
    parser.add_argument("--new-da-delta", type=float, default=0.1)
    parser.add_argument("--new-da-epsilon", type=float, default=1e-6)
    parser.add_argument("--new-da-provider", help="JSON provider hint")
    parser.add_argument("--attester", help="optional attester id")
    args = parser.parse_args()

    sketch = json.loads(args.sketch.read_text())
    if not verify_fast(sketch):
        raise SystemExit("verify_fast failed; cannot create swap")
    old_profile = sketch.get("da_profile")
    if not old_profile:
        raise SystemExit("sketch missing da_profile")

    provider_hint = parse_provider(args.new_da_provider)
    new_profile = build_da_profile(args.new_da_mode, args.new_da_delta, args.new_da_epsilon, provider_hint)
    record = {
        "schema": "stc_da_swap_v1",
        "stc_root": sketch.get("commitment_root"),
        "timestamp": int(time.time()),
        "old_profile": old_profile,
        "new_profile": new_profile,
    }
    if args.attester:
        record["attester"] = args.attester

    args.output.write_text(json.dumps(record, indent=2))
    print(f"wrote swap record to {args.output}")


if __name__ == "__main__":
    main()
