\
#!/usr/bin/env python3
"""
Extended docs/code drift check:
- Extract ENCODING_ID from bef_zk/codec/canonical_cbor.py
- Extract *_HASH_PREFIX_* or *_HASH_PREFIX_* constants from bef_zk/capsule/header.py and bef_zk/codec/init.py
- Ensure they appear in docs/spec/02_canonicalization.md and docs/spec/03_domain_tags.md

This is intentionally regex-based (fast + robust across refactors).
"""

from __future__ import annotations

import pathlib
import re
import sys
from typing import Iterable, List, Set


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

ENCODING_ID_RE = re.compile(r'ENCODING_ID\s*=\s*"([^"]+)"')
# matches things like: _HASH_PREFIX_INSTANCE = b"CAPSULE_INSTANCE_V1::"
PREFIX_BYTES_RE = re.compile(r'(_HASH_PREFIX_[A-Z0-9_]+)\s*=\s*b"([^"]+)"')
# matches things like: HASH_PREFIX_CAPSULE = b"BEF_CAPSULE_V1"
PREFIX_BYTES_RE2 = re.compile(r'(HASH_PREFIX_[A-Z0-9_]+)\s*=\s*b"([^"]+)"')


def read(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_prefixes(src: str) -> Set[str]:
    out: Set[str] = set()
    for _, val in PREFIX_BYTES_RE.findall(src):
        out.add(val)
    for _, val in PREFIX_BYTES_RE2.findall(src):
        out.add(val)
    return out


def main() -> int:
    errors: List[str] = []

    cbor = REPO_ROOT / "bef_zk" / "codec" / "canonical_cbor.py"
    header = REPO_ROOT / "bef_zk" / "capsule" / "header.py"
    codec_init = REPO_ROOT / "bef_zk" / "codec" / "init.py"

    spec_canon = REPO_ROOT / "docs" / "spec" / "02_canonicalization.md"
    spec_tags = REPO_ROOT / "docs" / "spec" / "03_domain_tags.md"

    for p in [cbor, header, codec_init, spec_canon, spec_tags]:
        if not p.exists():
            errors.append(f"missing file: {p}")

    if errors:
        return fail(errors)

    cbor_src = read(cbor)
    m = ENCODING_ID_RE.search(cbor_src)
    if not m:
        errors.append("could not parse ENCODING_ID from canonical_cbor.py")
        enc = None
    else:
        enc = m.group(1)

    prefixes = set()
    prefixes |= extract_prefixes(read(header))
    prefixes |= extract_prefixes(read(codec_init))

    canon_txt = read(spec_canon)
    tags_txt = read(spec_tags)

    if enc and enc not in canon_txt:
        errors.append(f'encoding id "{enc}" not found in {spec_canon}')

    missing = sorted([p for p in prefixes if p not in tags_txt and p not in canon_txt])
    if missing:
        errors.append("prefix drift: these prefixes exist in code but not in docs/spec:\n  - " + "\n  - ".join(missing))

    return ok() if not errors else fail(errors)


def ok() -> int:
    print("OK: docs/spec includes ENCODING_ID and all extracted hash prefix constants.")
    return 0


def fail(errors: List[str]) -> int:
    print("\nDOCS/CODE DRIFT CHECK FAILED:\n", file=sys.stderr)
    for e in errors:
        print(f"- {e}", file=sys.stderr)
    print("", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
