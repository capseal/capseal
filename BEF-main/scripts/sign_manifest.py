#!/usr/bin/env python3
"""Sign an existing manifest bundle so policy enforcement can succeed."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capsule_bench.manifests import load_manifest_bundle, write_manifest_signature


def _load_key_material(spec: str) -> str:
    candidate = Path(spec)
    if candidate.exists():
        data = candidate.read_text().strip()
    else:
        data = spec.strip()
    if data.startswith("0x"):
        data = data[2:]
    if not data:
        raise ValueError("manifest signer key is empty")
    # Validate hex input early for better error messages
    bytes.fromhex(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sign an existing manifests/ directory with a trusted secp256k1 key"
    )
    parser.add_argument("manifest_root", type=Path, help="Path to the manifests/ directory")
    parser.add_argument("--signer-id", required=True, help="Trusted manifest signer identifier")
    parser.add_argument(
        "--private-key",
        required=True,
        help="Path to or literal hex of the secp256k1 manifest signing key",
    )
    args = parser.parse_args()
    bundle = load_manifest_bundle(args.manifest_root)
    key_hex = _load_key_material(args.private_key)
    sig_path = write_manifest_signature(
        bundle,
        signer_id=args.signer_id,
        private_key_hex=key_hex,
    )
    print(
        f"signed manifest anchor {bundle.anchor_ref} with signer '{args.signer_id}' -> {sig_path}"
    )


if __name__ == "__main__":
    main()
