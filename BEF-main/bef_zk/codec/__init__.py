"""Canonical encoding helpers (dag-cbor-canonical-v1)."""
from __future__ import annotations

from typing import Any

import hashlib

from .canonical_cbor import (
    ENCODING_ID,
    HASH_PREFIX_CAPSULE,
    HASH_PREFIX_SEED,
    FieldElement,
    canonical_decode,
    canonical_encode,
)
from .bef_compact import MAGIC as BEF_COMPACT_MAGIC


def compute_capsule_hash(payload: Any, encoding_id: str = ENCODING_ID) -> str:
    canonical = canonical_encode(payload, encoding_id=encoding_id)
    return hashlib.sha256(HASH_PREFIX_CAPSULE + canonical).hexdigest()


def derive_capsule_seed(
    capsule_hash: str,
    *,
    anchor_ref: str | None = None,
    policy_id: str | None = None,
    policy_version: str | int | None = None,
) -> int:
    parts = [
        capsule_hash.encode("utf-8"),
        (anchor_ref or "").encode("utf-8"),
        (policy_id or "").encode("utf-8"),
        (str(policy_version) if policy_version is not None else "").encode("utf-8"),
    ]
    digest = hashlib.sha256(HASH_PREFIX_SEED + b"::".join(parts)).digest()
    return int.from_bytes(digest[:8], "big")

__all__ = [
    "ENCODING_ID",
    "HASH_PREFIX_CAPSULE",
    "HASH_PREFIX_SEED",
    "FieldElement",
    "canonical_encode",
    "canonical_decode",
    "compute_capsule_hash",
    "derive_capsule_seed",
    "detect_encoding",
]


def detect_encoding(data: bytes) -> str:
    if len(data) >= 4:
        magic = int.from_bytes(data[:4], "big")
        if magic == BEF_COMPACT_MAGIC:
            return "bef_compact_v1"
    return ENCODING_ID
