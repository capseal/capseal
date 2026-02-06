#!/usr/bin/env python3
"""Compute registry hashes for trusted relay and manifest signer configs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _load_mapping(path: Path, section: str) -> dict[str, str]:
    data = json.loads(path.read_text())
    entries = data.get(section)
    if not isinstance(entries, dict):
        raise ValueError(f"expected '{section}' object in {path}")
    mapping: dict[str, str] = {}
    for key, value in entries.items():
        if not isinstance(value, dict):
            continue
        pubkey = (value.get("pubkey") or "").strip()
        status = (value.get("status") or "active").lower()
        if pubkey and status == "active":
            mapping[key] = pubkey
    return mapping


def _registry_hash(mapping: dict[str, str]) -> str:
    ordered = {k: mapping[k] for k in sorted(mapping)}
    blob = json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute trust roots for registry JSON files")
    parser.add_argument("--relays", type=Path, help="Path to trusted_relays.json", default=None)
    parser.add_argument(
        "--manifest-signers",
        type=Path,
        help="Path to manifest_signers.json",
        default=None,
    )
    args = parser.parse_args()
    if args.relays:
        relays = _load_mapping(args.relays, "relays")
        print(f"relay root: {_registry_hash(relays)} ({len(relays)} entries)")
    if args.manifest_signers:
        signers = _load_mapping(args.manifest_signers, "signers")
        print(f"manifest root: {_registry_hash(signers)} ({len(signers)} entries)")
    if not args.relays and not args.manifest_signers:
        parser.error("provide at least --relays or --manifest-signers")


if __name__ == "__main__":
    main()
