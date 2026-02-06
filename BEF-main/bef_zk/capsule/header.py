"""Capsule header construction and hashing helpers."""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Dict, Iterable, List

from bef_zk.codec import ENCODING_ID, canonical_encode


HEADER_SCHEMA = "capsule_header_v2"

_HASH_PREFIX_HEADER = b"CAPSULE_HEADER_V2::"
_HASH_PREFIX_HEADER_COMMIT = b"CAPSULE_HEADER_COMMIT_V1::"
_HASH_PREFIX_PARAMS = b"CAPSULE_PARAMS_V1::"
_HASH_PREFIX_CAPSULE_ID = b"CAPSULE_ID_V2::"
_HASH_PREFIX_CHUNK_META = b"CAPSULE_CHUNK_META_V1::"
_HASH_PREFIX_ROW_INDEX_REF = b"CAPSULE_ROW_INDEX_REF_V1::"
_HASH_PREFIX_DA_POLICY = b"CAPSULE_DA_POLICY_V1::"
_HASH_PREFIX_CHUNK_MANIFEST = b"CAPSULE_CHUNK_MANIFEST_V1::"
_HASH_PREFIX_PROOF_SYSTEM = b"CAPSULE_PROOF_SYSTEM_V1::"
_HASH_PREFIX_MANIFEST = b"CAPSULE_MANIFEST_V1::"
_HASH_PREFIX_AIR_PARAMS = b"CAPSULE_AIR_PARAMS_V1::"
_HASH_PREFIX_FRI_CONFIG = b"CAPSULE_FRI_CONFIG_V1::"
_HASH_PREFIX_PROGRAM = b"CAPSULE_PROGRAM_V1::"
_HASH_PREFIX_INSTANCE = b"CAPSULE_INSTANCE_V1::"


def _canonical_hash(prefix: bytes, payload: Any) -> str:
    encoded = canonical_encode(payload, encoding_id=ENCODING_ID)
    return hashlib.sha256(prefix + encoded).hexdigest()


def normalize_params(params: Dict[str, Any] | None) -> Dict[str, Any]:
    """Only retain the row width parameter for header hashing."""
    params = params or {}
    if "row_width" in params:
        return {"row_width": params["row_width"]}
    return params


def sanitize_row_index_ref(ref: Dict[str, Any] | None) -> Dict[str, Any]:
    clean = copy.deepcopy(ref or {})
    pointer = clean.get("pointer")
    if pointer:
        pointer = {k: v for k, v in pointer.items() if k != "provider_root"}
        if pointer:
            clean["pointer"] = pointer
        else:
            clean.pop("pointer", None)
    return clean


def sanitize_da_policy(policy: Dict[str, Any] | None) -> Dict[str, Any]:
    clean = copy.deepcopy(policy or {})
    provider = clean.get("provider")
    if provider:
        provider = {k: v for k, v in provider.items() if k != "archive_root"}
        if provider:
            clean["provider"] = provider
        else:
            clean.pop("provider", None)
    return clean


def hash_params(params: Dict[str, Any] | None) -> str:
    return _canonical_hash(_HASH_PREFIX_PARAMS, normalize_params(params))


def hash_chunk_meta(chunk_meta: Dict[str, Any] | None) -> str:
    return _canonical_hash(_HASH_PREFIX_CHUNK_META, chunk_meta or {})


def hash_row_index_ref(ref: Dict[str, Any] | None) -> str:
    return _canonical_hash(_HASH_PREFIX_ROW_INDEX_REF, sanitize_row_index_ref(ref))


def hash_da_policy(policy: Dict[str, Any] | None) -> str:
    return _canonical_hash(_HASH_PREFIX_DA_POLICY, sanitize_da_policy(policy))


def _normalize_chunk_manifest(handles: Iterable[Any]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, handle in enumerate(handles):
        if isinstance(handle, dict):
            entry = {
                "id": int(handle.get("id", idx)),
                "uri": str(handle.get("uri", "")),
                "sha256": str(handle.get("sha256", "")),
                "size": int(handle.get("size", 0)),
                "content_type": str(handle.get("content_type", "application/octet-stream")),
            }
        else:
            entry = {
                "id": idx,
                "uri": str(handle),
                "sha256": "",
                "size": 0,
                "content_type": "application/octet-stream",
            }
        normalized.append(entry)
    normalized.sort(key=lambda item: item["id"])
    return normalized


def hash_chunk_handles(handles: Iterable[Any]) -> str:
    manifest = {
        "schema": "chunk_manifest_v1",
        "chunks": _normalize_chunk_manifest(handles),
    }
    return _canonical_hash(_HASH_PREFIX_CHUNK_MANIFEST, manifest)


def hash_proof_system(spec: Dict[str, Any] | None) -> str:
    payload = spec or {}
    return _canonical_hash(_HASH_PREFIX_PROOF_SYSTEM, payload)


def hash_manifest_descriptor(manifest: Dict[str, Any] | None) -> str:
    return _canonical_hash(_HASH_PREFIX_MANIFEST, manifest or {})


def hash_air_params(params: Any | None) -> str | None:
    if params is None:
        return None
    if hasattr(params, "__dict__"):
        params_dict = {
            "steps": getattr(params, "steps", None),
            "num_challenges": getattr(params, "num_challenges", None),
            "r_challenges": list(getattr(params, "r_challenges", []) or []),
            "matrix": list(getattr(params, "matrix", []) or []),
            "modulus": getattr(params, "modulus", None),
        }
    else:
        params_dict = params
    return _canonical_hash(_HASH_PREFIX_AIR_PARAMS, params_dict)


def hash_fri_config(cfg: Any | None) -> str | None:
    if cfg is None:
        return None
    if hasattr(cfg, "__dict__"):
        cfg_dict = {
            "field_modulus": getattr(cfg, "field_modulus", None),
            "domain_size": getattr(cfg, "domain_size", None),
            "max_degree": getattr(cfg, "max_degree", None),
            "num_rounds": getattr(cfg, "num_rounds", None),
            "num_queries": getattr(cfg, "num_queries", None),
        }
    else:
        cfg_dict = cfg
    return _canonical_hash(_HASH_PREFIX_FRI_CONFIG, cfg_dict)


def hash_program_descriptor(program: Any | None) -> str | None:
    if program is None:
        return None
    if isinstance(program, dict):
        payload = program
    else:
        payload = {
            "schema": "geom_program_v1",
            "instructions": list(program),
        }
    return _canonical_hash(_HASH_PREFIX_PROGRAM, payload)


def hash_instance_binding(
    *,
    statement_hash: str,
    row_root: str,
    trace_spec_hash: str,
    vk_hash: str | None,
    params_hash: str | None = None,
    chunk_meta_hash: str | None = None,
    row_tree_arity: int | None = None,
    air_params_hash: str | None = None,
    fri_params_hash: str | None = None,
    program_hash: str | None = None,
) -> str:
    payload = {
        "statement_hash": str(statement_hash or "").lower(),
        "row_root": str(row_root or "").lower(),
        "trace_spec_hash": str(trace_spec_hash or "").lower(),
        "vk_hash": str(vk_hash or "").lower(),
        "params_hash": str(params_hash or "").lower(),
        "chunk_meta_hash": str(chunk_meta_hash or "").lower(),
        "row_tree_arity": int(row_tree_arity) if row_tree_arity is not None else None,
        "air_params_hash": str(air_params_hash or "").lower(),
        "fri_params_hash": str(fri_params_hash or "").lower(),
        "program_hash": str(program_hash or "").lower(),
    }
    return _canonical_hash(_HASH_PREFIX_INSTANCE, payload)


def build_capsule_header(
    *,
    vm_id: str,
    backend_id: str,
    circuit_id: str,
    trace_id: str,
    prev_capsule_hash: str | None,
    trace_spec_hash: str,
    statement_hash: str,
    params_hash: str,
    row_root: str,
    row_tree_arity: int | None,
    row_index_ref_hash: str,
    chunk_meta_hash: str,
    chunk_handles_root: str,
    policy_ref: Dict[str, Any],
    da_policy_hash: str,
    anchor: Dict[str, Any] | None,
    proof_system: Dict[str, Any] | None = None,
    chunk_manifest_hash: str | None = None,
    manifest_hash: str | None = None,
    air_params_hash: str | None = None,
    fri_params_hash: str | None = None,
    program_hash: str | None = None,
    da_challenge_hash: str | None = None,
    payload_hash: str | None = None,
    verification_profile: str | None = None,
) -> Dict[str, Any]:
    proof_meta = dict(proof_system or {})
    if air_params_hash is not None:
        proof_meta.setdefault("air_params_hash", air_params_hash)
    if fri_params_hash is not None:
        proof_meta.setdefault("fri_params_hash", fri_params_hash)
    if program_hash is not None:
        proof_meta.setdefault("program_hash", program_hash)

    header = {
        "schema": HEADER_SCHEMA,
        "verification_profile": verification_profile,
        "vm_id": vm_id,
        "backend_id": backend_id,
        "circuit_id": circuit_id,
        "trace_id": trace_id,
        "prev_capsule_hash": prev_capsule_hash,
        "trace_spec_hash": trace_spec_hash,
        "statement_hash": statement_hash,
        "params_hash": params_hash,
        "row_commitment": {
            "root": row_root,
            "tree_arity": row_tree_arity,
            "row_index_ref_hash": row_index_ref_hash,
            "chunk_meta_hash": chunk_meta_hash,
            "chunk_handles_root": chunk_handles_root,
            "chunk_manifest_hash": chunk_manifest_hash or chunk_handles_root,
        },
        "policy_ref": policy_ref,
        "da_ref": {
            "policy_hash": da_policy_hash,
            "challenge_hash": da_challenge_hash,
        },
        "proof_system": proof_meta,
        "artifact_manifest_hash": manifest_hash,
        "anchor": anchor or {},
    }
    if payload_hash is not None:
        header["payload_hash"] = payload_hash
    return header


def compute_header_hash(header: Dict[str, Any]) -> str:
    return _canonical_hash(_HASH_PREFIX_HEADER, header)


def compute_header_commit_hash(header: Dict[str, Any]) -> str:
    commit_view = copy.deepcopy(header)
    commit_view.setdefault("da_ref", {})["challenge_hash"] = None
    return _canonical_hash(_HASH_PREFIX_HEADER_COMMIT, commit_view)


def hash_capsule_identity(header_commit_hash: str, payload_hash: str) -> str:
    payload = {
        "header_commit_hash": header_commit_hash,
        "payload_hash": payload_hash,
    }
    return _canonical_hash(_HASH_PREFIX_CAPSULE_ID, payload)


__all__ = [
    "HEADER_SCHEMA",
    "build_capsule_header",
    "compute_header_hash",
    "compute_header_commit_hash",
    "hash_capsule_identity",
    "hash_chunk_handles",
    "hash_chunk_meta",
    "hash_da_policy",
    "hash_params",
    "hash_proof_system",
    "hash_manifest_descriptor",
    "hash_air_params",
    "hash_fri_config",
    "hash_program_descriptor",
    "hash_instance_binding",
    "hash_row_index_ref",
    "sanitize_da_policy",
    "sanitize_row_index_ref",
]
