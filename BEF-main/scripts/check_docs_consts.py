#!/usr/bin/env python3
"""
Fail if docs/spec drifts from code constants (encoding ID + domain tags).

Run from repo root:
  python scripts/check_docs_consts.py
"""
from __future__ import annotations

import pathlib
import re
import sys
from typing import Iterable


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Code locations
CODE_FILES = {
    "header": REPO_ROOT / "bef_zk" / "capsule" / "header.py",
    "canonical_cbor": REPO_ROOT / "bef_zk" / "codec" / "canonical_cbor.py",
    "codec_init": REPO_ROOT / "bef_zk" / "codec" / "__init__.py",
}

# Spec locations
SPEC_FILES = {
    "canonicalization": REPO_ROOT / "docs" / "spec" / "02_canonicalization.md",
    "domain_tags": REPO_ROOT / "docs" / "spec" / "03_domain_tags.md",
    "instance_binding": REPO_ROOT / "docs" / "spec" / "04_instance_binding.md",
    "registries": REPO_ROOT / "docs" / "spec" / "08_registries.md",
}

# Strings that must appear in the domain tags spec (source of truth = code reality)
REQUIRED_TAG_STRINGS = [
    "BEF_CAPSULE_V1",
    "BEF_AUDIT_SEED_V1",
    "CAPSULE_HEADER_V2::",
    "CAPSULE_HEADER_COMMIT_V1::",
    "CAPSULE_ID_V2::",
    "CAPSULE_INSTANCE_V1::",
]

ENCODING_ID_PATTERN = re.compile(r'ENCODING_ID\s*=\s*"([^"]+)"')


def read_text(p: pathlib.Path) -> str:
    if not p.exists():
        raise FileNotFoundError(f"missing file: {p}")
    return p.read_text(encoding="utf-8", errors="replace")


def require_contains(haystack: str, needles: Iterable[str]) -> list[str]:
    missing = []
    for n in needles:
        if n not in haystack:
            missing.append(n)
    return missing


def main() -> int:
    errors: list[str] = []

    # Load code files
    try:
        cbor_src = read_text(CODE_FILES["canonical_cbor"])
    except Exception as e:
        return fail([f"Failed to read canonical_cbor.py: {e}"])

    m = ENCODING_ID_PATTERN.search(cbor_src)
    if not m:
        errors.append("Could not find ENCODING_ID in bef_zk/codec/canonical_cbor.py")
        encoding_id = None
    else:
        encoding_id = m.group(1)

    # Load spec files
    spec_blobs = {}
    for name, path in SPEC_FILES.items():
        try:
            spec_blobs[name] = read_text(path)
        except Exception as e:
            errors.append(f"Failed to read {name}: {path} ({e})")

    if errors:
        return fail(errors)

    # Check encoding id appears in canonicalization spec
    if encoding_id:
        if encoding_id not in spec_blobs["canonicalization"]:
            errors.append(
                f'Encoding drift: code ENCODING_ID="{encoding_id}" '
                f"not present in {SPEC_FILES['canonicalization']}"
            )

    # Check required tag strings appear in domain tags spec
    missing_tags = require_contains(spec_blobs["domain_tags"], REQUIRED_TAG_STRINGS)
    if missing_tags:
        errors.append(
            "Domain-tag drift: missing from docs/spec/03_domain_tags.md:\n  - "
            + "\n  - ".join(missing_tags)
        )

    # Optional: sanity check instance binding prefix appears in instance binding spec
    if "CAPSULE_INSTANCE_V1::" not in spec_blobs["instance_binding"]:
        errors.append("Instance binding spec missing CAPSULE_INSTANCE_V1:: prefix.")

    if errors:
        return fail(errors)

    print("OK: docs/spec constants match code (encoding id + domain tags).")
    return 0


def fail(errors: list[str]) -> int:
    print("\nDOCS/CODE DRIFT CHECK FAILED:\n", file=sys.stderr)
    for e in errors:
        print(f"- {e}", file=sys.stderr)
    print("", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
