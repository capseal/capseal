"""Mask generation helpers for the geometry zk prototype."""
from __future__ import annotations

import hashlib
from typing import Dict, List

import hashlib

from ..air.geom_air import GeomAIRParams

MODULUS = (1 << 61) - 1


def _expand_field_elements(seed: bytes, count: int) -> List[int]:
    out: List[int] = []
    ctr = 0
    while len(out) < count:
        block = hashlib.sha256(seed + ctr.to_bytes(8, "big")).digest()
        for i in range(0, len(block), 8):
            chunk = block[i : i + 8]
            if len(chunk) < 8:
                chunk = chunk.ljust(8, b"\x00")
            out.append(int.from_bytes(chunk, "big") % MODULUS)
            if len(out) >= count:
                break
        ctr += 1
    return out


def _hash_to_field(seed: bytes, label: bytes, idx: int, modulus: int = MODULUS) -> int:
    h = hashlib.sha256()
    h.update(seed)
    h.update(label)
    h.update(idx.to_bytes(8, "big"))
    return int.from_bytes(h.digest(), "big") % modulus


def mask_at_index(mask_digest: bytes, idx: int, modulus: int = MODULUS) -> int:
    return _hash_to_field(mask_digest, b"row_mask", idx, modulus)


def column_mask_at_index(
    alpha_digest: bytes,
    column_name: str,
    idx: int,
    modulus: int = MODULUS,
) -> int:
    label = f"colmask:{column_name}".encode("utf-8")
    return _hash_to_field(alpha_digest, label, idx, modulus)


def derive_column_masks(
    alpha_digest: bytes,
    params: GeomAIRParams,
    domain_size: int,
) -> Dict[str, List[int]]:
    """Derive per-column masking polynomials keyed by column name."""

    column_names = [
        "PC",
        "OP",
        "GAS",
        "ACC",
        "X1",
        "X2",
        "CNT",
        "M11",
        "M12",
        "M22",
    ]
    mask_map: Dict[str, List[int]] = {}
    for name in column_names:
        mask_map[name] = [
            column_mask_at_index(alpha_digest, name, i) for i in range(domain_size)
        ]

    for idx in range(params.num_challenges):
        mask_map[f"sketches_{idx}"] = [
            column_mask_at_index(alpha_digest, f"sketches_{idx}", i)
            for i in range(domain_size)
        ]
        mask_map[f"powers_{idx}"] = [
            column_mask_at_index(alpha_digest, f"powers_{idx}", i)
            for i in range(domain_size)
        ]

    return mask_map


def derive_mask_digest(
    alpha_digest: bytes,
    statement_hash: bytes,
    base_root: bytes,
) -> bytes:
    h = hashlib.sha256()
    h.update(b"geom-comp-mask-v1")
    h.update(alpha_digest)
    h.update(statement_hash)
    h.update(base_root)
    return h.digest()


def derive_mask_vector(mask_digest: bytes, domain_size: int) -> List[int]:
    return [mask_at_index(mask_digest, i) for i in range(domain_size)]


def serialize_statement(statement) -> bytes:
    from .types import GeomStatement

    stmt: GeomStatement = statement
    h = hashlib.sha256()
    h.update(stmt.params.steps.to_bytes(8, "big"))
    for row in stmt.params.matrix:
        for entry in row:
            h.update(int(entry).to_bytes(8, "big", signed=False))
    for challenge in stmt.params.r_challenges:
        h.update(int(challenge).to_bytes(8, "big", signed=False))
    h.update(int(stmt.final_m11).to_bytes(8, "big", signed=False))
    h.update(int(stmt.final_m12).to_bytes(8, "big", signed=False))
    h.update(int(stmt.final_m22).to_bytes(8, "big", signed=False))
    h.update(int(stmt.final_cnt).to_bytes(8, "big", signed=False))
    return h.digest()
