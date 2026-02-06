#!/usr/bin/env python3
"""CapSeal release script - generates checksums and signatures for distribution artifacts.

Usage:
    python scripts/release.py build    # Build wheel + sdist
    python scripts/release.py sign     # Generate checksums (and GPG sign if available)
    python scripts/release.py verify   # Verify existing signatures
    python scripts/release.py all      # Build, sign, generate demo receipt

This script ensures releases are trust-aligned by:
1. Generating SHA256 checksums for all artifacts
2. Optionally signing with GPG (if available)
3. Including a demo receipt as a release artifact (proof capseal works)
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DIST_DIR = Path("dist")
CHECKSUMS_FILE = DIST_DIR / "SHA256SUMS"


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> bool:
    """Build wheel and sdist."""
    print("Building distribution artifacts...")

    # Clean old builds
    if DIST_DIR.exists():
        import shutil
        shutil.rmtree(DIST_DIR)

    # Build
    result = subprocess.run(
        [sys.executable, "-m", "build"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return False

    print(f"Built artifacts in {DIST_DIR}/")
    for f in DIST_DIR.iterdir():
        if f.suffix in (".whl", ".gz"):
            print(f"  - {f.name}")

    return True


def generate_checksums() -> dict[str, str]:
    """Generate SHA256 checksums for all distribution artifacts."""
    checksums = {}

    for path in DIST_DIR.iterdir():
        if path.suffix in (".whl", ".gz"):
            checksum = sha256_file(path)
            checksums[path.name] = checksum
            print(f"  {checksum}  {path.name}")

    # Write checksums file
    with open(CHECKSUMS_FILE, "w") as f:
        for name, checksum in sorted(checksums.items()):
            f.write(f"{checksum}  {name}\n")

    return checksums


def gpg_sign() -> bool:
    """Sign checksums file with GPG if available."""
    # Check if gpg is available
    result = subprocess.run(
        ["gpg", "--version"],
        capture_output=True,
    )
    if result.returncode != 0:
        print("GPG not available - skipping signature")
        return False

    # Sign the checksums file
    result = subprocess.run(
        ["gpg", "--armor", "--detach-sign", str(CHECKSUMS_FILE)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"GPG signing failed: {result.stderr}")
        return False

    print(f"Signed: {CHECKSUMS_FILE}.asc")
    return True


def generate_demo_receipt() -> Path:
    """Generate a demo receipt to include with the release."""
    print("Generating demo receipt...")

    receipt_path = DIST_DIR / "release_demo_receipt.json"

    result = subprocess.run(
        [sys.executable, "-m", "bef_zk.capsule.cli", "demo", "--json"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Demo failed: {result.stderr}")
        # Create a minimal receipt indicating failure
        receipt = {
            "status": "demo_failed",
            "error": result.stderr,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    else:
        receipt = json.loads(result.stdout)
        receipt["release_timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)

    print(f"Demo receipt: {receipt_path}")
    return receipt_path


def verify() -> bool:
    """Verify existing checksums and signatures."""
    if not CHECKSUMS_FILE.exists():
        print(f"No checksums file found: {CHECKSUMS_FILE}")
        return False

    print("Verifying checksums...")

    # Read expected checksums
    expected = {}
    with open(CHECKSUMS_FILE) as f:
        for line in f:
            parts = line.strip().split("  ", 1)
            if len(parts) == 2:
                expected[parts[1]] = parts[0]

    # Verify each file
    all_valid = True
    for name, expected_hash in expected.items():
        path = DIST_DIR / name
        if not path.exists():
            print(f"  MISSING: {name}")
            all_valid = False
            continue

        actual_hash = sha256_file(path)
        if actual_hash == expected_hash:
            print(f"  OK: {name}")
        else:
            print(f"  MISMATCH: {name}")
            print(f"    Expected: {expected_hash}")
            print(f"    Actual:   {actual_hash}")
            all_valid = False

    # Verify GPG signature if present
    sig_path = Path(str(CHECKSUMS_FILE) + ".asc")
    if sig_path.exists():
        print("\nVerifying GPG signature...")
        result = subprocess.run(
            ["gpg", "--verify", str(sig_path), str(CHECKSUMS_FILE)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("  GPG signature: VALID")
        else:
            print(f"  GPG signature: INVALID - {result.stderr}")
            all_valid = False

    return all_valid


def sign() -> None:
    """Generate checksums and optionally GPG sign."""
    print("Generating checksums...")
    generate_checksums()
    print()
    gpg_sign()


def all_steps() -> None:
    """Run complete release pipeline."""
    print("=" * 60)
    print("CapSeal Release Pipeline")
    print("=" * 60)
    print()

    # Build
    if not build():
        sys.exit(1)
    print()

    # Sign
    sign()
    print()

    # Demo receipt
    generate_demo_receipt()
    print()

    # Verify
    print("Verifying release...")
    if verify():
        print("\nRelease artifacts ready!")
    else:
        print("\nRelease verification FAILED")
        sys.exit(1)


def main() -> None:
    """Entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "build":
        if not build():
            sys.exit(1)
    elif command == "sign":
        sign()
    elif command == "verify":
        if not verify():
            sys.exit(1)
    elif command == "all":
        all_steps()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
