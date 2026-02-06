"""Deterministic payload views used for capsule hashing."""
from __future__ import annotations

from typing import Any, Dict

from bef_zk.codec import ENCODING_ID, compute_capsule_hash


#
# Only the fields listed here participate in the payload commitment. The goal is
# to hash *exactly* the prover-controlled material (trace spec, statement,
# proofs, manifests, etc.) independent of mutable wrapper fields such as
# `header`, `authorship`, or transport metadata. New payload keys must be
# explicitly added here so both the prover and verifier agree on the hashing
# boundary.
#
PAYLOAD_KEYS: tuple[str, ...] = (
    "schema",
    "vm_id",
    "air_id",
    "trace_id",
    "prev_capsule_hash",
    "trace_spec",
    "trace_spec_hash",
    "trace_commitment",
    "policy",
    "params",
    "da_policy",
    "da_profile",
    "chunk_meta",
    "row_index_ref",
    "hashing",
    "proof_system",
    "anchor",
    "proofs",
    "row_archive",
    "artifacts",
    "statement",
    "statement_hash",
    "verification_profile",
    "events_log",
    "extras",
)


def payload_view(capsule: Dict[str, Any]) -> Dict[str, Any]:
    """Return the deterministic view of the capsule payload used for hashing."""

    view: Dict[str, Any] = {}
    for key in PAYLOAD_KEYS:
        if key in capsule:
            view[key] = capsule[key]
    return view


def compute_payload_hash(
    capsule: Dict[str, Any], *, encoding_id: str | None = None
) -> str:
    """Compute the canonical payload hash for a capsule."""

    enc = encoding_id or ENCODING_ID
    return compute_capsule_hash(payload_view(capsule), encoding_id=enc)


__all__ = ["PAYLOAD_KEYS", "payload_view", "compute_payload_hash"]
