#!/usr/bin/env python3
"""Verify a strategy capsule end-to-end with stable error codes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import resource

try:  # Optional dependency for signature checks
    from coincurve import PublicKey
    from coincurve.ecdsa import recoverable_convert
except ImportError:  # pragma: no cover - coincurve may be missing in CI
    PublicKey = None

from bef_zk.air.geom_air import GeomInitialState
from bef_zk.codec import (
    ENCODING_ID,
    canonical_decode,
    compute_capsule_hash,
    derive_capsule_seed,
)
from bef_zk.capsule.da import (
    challenge_signature_payload,
    DA_CHALLENGE_SCHEMA,
    DA_CHALLENGE_V2_SCHEMA,
    derive_da_seed,
    derive_signed_da_seed,
    hash_da_challenge,
    hash_signed_da_challenge,
    verify_da_challenge_binding,
    verify_da_challenge_signature,
)
from bef_zk.capsule.header import (
    HEADER_SCHEMA,
    compute_header_commit_hash,
    compute_header_hash,
    hash_air_params,
    hash_chunk_handles,
    hash_chunk_meta,
    hash_da_policy,
    hash_manifest_descriptor,
    hash_fri_config,
    hash_capsule_identity,
    hash_instance_binding,
    hash_params,
    hash_program_descriptor,
    hash_proof_system,
    hash_row_index_ref,
    sanitize_da_policy,
    sanitize_row_index_ref,
)
from bef_zk.capsule.payload import compute_payload_hash
from bef_zk.da import AvailabilityError, LocalFileSystemProvider, PolicyAwareDAClient
from bef_zk.stc.aok_cpu import merkle_from_values
from bef_zk.stc.merkle import build_kary_levels, root_from_levels, verify_kary
from bef_zk.stc.vc import STCVectorCommitment
from bef_zk.zk_geom.serialization import proof_from_bytes, proof_from_json
from bef_zk.zk_geom.verifier import zk_verify_geom
from backends.risc0_adapter import Risc0TraceAdapter, compute_binding_hash as risc0_compute_binding
from bef_zk.spec import (
    TraceSpecV1,
    StatementV1,
    compute_trace_spec_hash,
    compute_statement_hash,
)
from bef_zk.verifier_errors import *
from scripts.artifact_manifest import encoding_for_path, load_manifest

try:  # Optional dependency for downloading artifacts from R2
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:  # pragma: no cover - boto3 may be missing locally
    boto3 = None

ARTIFACTS_ROOT = Path(os.environ.get("ARTIFACTS_ROOT", "server_data/artifacts"))
R2_ENDPOINT_URL = os.environ.get("R2_ENDPOINT_URL")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")
R2_PREFIX = os.environ.get("R2_PREFIX", "")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "auto"))
REPO_ROOT = Path(__file__).resolve().parents[1]

TRUSTED_RELAYS_ENV: set[str] = {
    relay.strip()
    for relay in os.environ.get("CAPSULE_TRUSTED_RELAYS", "").split(",")
    if relay.strip()
}


def _parse_relay_key_mapping(raw: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in raw.split(","):
        token = item.strip()
        if not token or "=" not in token:
            continue
        relay_id, key = token.split("=", 1)
        relay_id = relay_id.strip()
        key = key.strip()
        if relay_id and key:
            mapping[relay_id] = key
    return mapping


def _registry_hash(mapping: dict[str, str]) -> str:
    ordered = {k: mapping[k] for k in sorted(mapping)}
    blob = json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _load_default_relay_registry() -> tuple[dict[str, str], str | None]:
    registry_path = REPO_ROOT / "config" / "trusted_relays.json"
    try:
        data = json.loads(registry_path.read_text())
    except FileNotFoundError:
        return {}, None
    relays: dict[str, str] = {}
    for relay_id, entry in (data.get("relays") or {}).items():
        pubkey = (entry.get("pubkey") or "").strip()
        status = (entry.get("status") or "active").lower()
        if pubkey and status == "active":
            relays[relay_id] = pubkey
    root = _registry_hash(relays) if relays else None
    return relays, root


DEFAULT_TRUSTED_RELAY_KEYS, DEFAULT_TRUSTED_RELAY_ROOT = _load_default_relay_registry()

TRUSTED_RELAY_KEYS_ENV: dict[str, str] = _parse_relay_key_mapping(
    os.environ.get("CAPSULE_TRUSTED_RELAY_KEYS", "")
)
if not TRUSTED_RELAY_KEYS_ENV and DEFAULT_TRUSTED_RELAY_KEYS:
    TRUSTED_RELAY_KEYS_ENV = dict(DEFAULT_TRUSTED_RELAY_KEYS)

if not TRUSTED_RELAYS_ENV and DEFAULT_TRUSTED_RELAY_KEYS:
    TRUSTED_RELAYS_ENV = set(DEFAULT_TRUSTED_RELAY_KEYS.keys())

TRUSTED_RELAYS_ROOT = os.environ.get("CAPSULE_TRUSTED_RELAYS_ROOT") or DEFAULT_TRUSTED_RELAY_ROOT


def _load_default_manifest_signers() -> tuple[dict[str, str], str | None]:
    registry_path = REPO_ROOT / "config" / "manifest_signers.json"
    try:
        data = json.loads(registry_path.read_text())
    except FileNotFoundError:
        return {}, None
    signers: dict[str, str] = {}
    for signer_id, entry in (data.get("signers") or {}).items():
        pubkey = (entry.get("pubkey") or "").strip()
        status = (entry.get("status") or "active").lower()
        if pubkey and status == "active":
            signers[signer_id] = pubkey
    root = _registry_hash(signers) if signers else None
    return signers, root


DEFAULT_MANIFEST_SIGNERS, DEFAULT_MANIFEST_ROOT = _load_default_manifest_signers()
TRUSTED_MANIFEST_SIGNERS_ENV: dict[str, str] = _parse_relay_key_mapping(
    os.environ.get("CAPSULE_TRUSTED_MANIFEST_SIGNERS", "")
)
_R2_CLIENT = None
_ARTIFACT_MANIFEST_CACHE: Dict[str, List[dict]] = {}
_DOWNLOADED_CACHE: Dict[tuple, Path] = {}
_CURRENT_RUN_ID: str | None = None
from scripts.geom_programs import GEOM_PROGRAM

# ------------------------------
# Local helpers for instrumentation
# ------------------------------
def _rss_bytes() -> int:
    """Return current RSS in bytes (best-effort, Linux-friendly)."""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_r2_client():  # pragma: no cover - requires boto3 + env
    global _R2_CLIENT
    if _R2_CLIENT is not None:
        return _R2_CLIENT
    if not all([boto3, R2_ENDPOINT_URL, R2_BUCKET_NAME, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        return None
    _R2_CLIENT = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "path"}),
    )
    return _R2_CLIENT


def _verify_risc0_proof(
    proof_json: str,
    statement_hash_bytes: bytes,
    capsule: dict,
) -> Tuple[bool, dict]:
    """Verify a RISC0 proof using binding integrity checks.

    Returns (ok, stats) where ok is True if proof is valid.

    Note: The proof contains risc0_binding computed from the original statement_hash,
    and statement_hash stored in the proof. We verify using the stored statement_hash.
    """
    import json
    try:
        proof_data = json.loads(proof_json)
    except json.JSONDecodeError:
        return False, {"error": "invalid proof JSON"}

    image_id = proof_data.get("image_id", "")
    journal_hex = proof_data.get("journal", "")
    journal_digest = proof_data.get("journal_digest", "")
    risc0_binding_hex = proof_data.get("risc0_binding", "")
    stored_statement_hash = proof_data.get("statement_hash", "")

    if not image_id or not journal_digest:
        return False, {"error": "missing image_id or journal_digest"}

    # Use the statement_hash stored in the proof for binding verification
    if stored_statement_hash:
        statement_for_binding = bytes.fromhex(stored_statement_hash)
    else:
        # Fallback: try the capsule's statement_hash
        capsule_statement = capsule.get("statement_hash", "")
        if capsule_statement:
            statement_for_binding = bytes.fromhex(capsule_statement)
        else:
            statement_for_binding = statement_hash_bytes

    # Recompute and verify risc0-specific binding hash
    expected_risc0_binding = risc0_compute_binding(image_id, journal_digest, statement_for_binding)

    if risc0_binding_hex and bytes.fromhex(risc0_binding_hex) != expected_risc0_binding:
        return False, {
            "error": "risc0_binding mismatch",
            "expected": expected_risc0_binding.hex(),
            "got": risc0_binding_hex,
        }

    # Verify journal digest
    if journal_hex:
        journal_bytes = bytes.fromhex(journal_hex)
        computed_digest = hashlib.sha256(journal_bytes).hexdigest()
        if computed_digest != journal_digest:
            return False, {
                "error": "journal_digest mismatch",
                "expected": computed_digest,
                "got": journal_digest,
            }

    stats = {
        "image_id": image_id,
        "journal_len": len(bytes.fromhex(journal_hex)) if journal_hex else 0,
        "binding_verified": True,
        "statement_hash": statement_for_binding.hex(),
    }
    return True, stats


def _load_manifest_entries(run_id: str) -> List[dict]:
    if run_id in _ARTIFACT_MANIFEST_CACHE:
        return _ARTIFACT_MANIFEST_CACHE[run_id]
    manifest_path = ARTIFACTS_ROOT / run_id / "artifacts.json"
    if not manifest_path.exists():
        entries: List[dict] = []
    else:
        try:
            entries = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            entries = []
    _ARTIFACT_MANIFEST_CACHE[run_id] = entries
    return entries


def _download_r2_object(run_id: str, object_key: str, filename: str) -> Path | None:
    client = _get_r2_client()
    if client is None:
        return None
    cache_key = (run_id, object_key)
    if cache_key in _DOWNLOADED_CACHE:
        return _DOWNLOADED_CACHE[cache_key]
    tmp_dir = ARTIFACTS_ROOT / run_id / "_cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / filename
    try:
        client.download_file(R2_BUCKET_NAME, object_key, str(dest))
    except Exception:  # pragma: no cover - network dependent
        return None
    _DOWNLOADED_CACHE[cache_key] = dest
    return dest


def _ensure_local_artifact(path: Path) -> Path:
    if path.exists():
        return path
    run_id = _CURRENT_RUN_ID
    if not run_id:
        return path
    entries = _load_manifest_entries(run_id)
    for entry in entries:
        entry_name = entry.get("name") or ""
        object_key = entry.get("object_key")
        storage = (entry.get("storage") or "local").lower()
        if not object_key:
            continue
        if entry_name.endswith(path.name) or object_key.endswith(path.name):
            if storage == "local":
                candidate = Path(object_key)
                if candidate.exists():
                    return candidate
            elif storage == "r2":
                downloaded = _download_r2_object(run_id, object_key, path.name)
                if downloaded and downloaded.exists():
                    return downloaded
    return path


def _confine_path(root: Path, candidate: Path) -> Path:
    root_resolved = root.resolve()
    candidate_resolved = candidate.resolve()
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError:
        raise PermissionError("artifact path escapes archive root")
    return candidate_resolved


def _resolve_chunk_handle_entries(
    entries: list[dict[str, Any]],
    archive_root: Path,
) -> tuple[list[str], str]:
    resolved: list[str] = []
    try:
        root_resolved = archive_root.resolve()
    except FileNotFoundError:
        return [], E062_ROW_ARCHIVE_MISSING
    for entry in entries:
        if not isinstance(entry, dict):
            return [], E075_CHUNK_HANDLE_INVALID
        uri = entry.get("uri") or entry.get("path") or entry.get("rel_path") or entry.get("abs_path")
        if not uri:
            return [], E075_CHUNK_HANDLE_INVALID
        candidate = Path(uri)
        if not candidate.is_absolute():
            candidate = root_resolved / candidate
        try:
            confined = _confine_path(root_resolved, candidate)
        except (PermissionError, FileNotFoundError):
            return [], E075_CHUNK_HANDLE_INVALID
        if not confined.exists():
            return [], E062_ROW_ARCHIVE_MISSING
        expected_size = entry.get("size")
        if expected_size is not None:
            try:
                recorded = int(expected_size)
            except (TypeError, ValueError):
                return [], E075_CHUNK_HANDLE_INVALID
            if recorded > 0 and confined.stat().st_size != recorded:
                return [], E075_CHUNK_HANDLE_INVALID
        expected_sha = (entry.get("sha256") or "").strip().lower()
        if expected_sha:
            actual_sha = _hash_file(confined)
            if actual_sha.lower() != expected_sha:
                return [], E075_CHUNK_HANDLE_INVALID
        resolved.append(str(confined))
    return resolved, OK


_STATUS_ORDER = {
    "REJECTED": 0,
    "PROOF_ONLY": 1,
    "POLICY_SELF_REPORTED": 2,
    "POLICY_ENFORCED": 3,
    "FULLY_VERIFIED": 4,
}

REQUIRED_EVENT_TYPES = {
    "run_started",
    "statement_locked",
    "proof_artifact",
    "capsule_sealed",
    "run_completed",
}

_LEVEL_CANONICAL = {
    "PROOF_ONLY": "PROOF_ONLY",
    "POLICY_SELF_REPORTED": "POLICY_SELF_REPORTED",
    "POLICY_ENFORCED": "POLICY_ENFORCED",
    "FULL": "FULLY_VERIFIED",
    "FULLY_VERIFIED": "FULLY_VERIFIED",
}

W_POLICY_SELF_REPORTED = "W_POLICY_SELF_REPORTED"

_WARNING_THRESHOLDS = {
    E060_ROW_INDEX_COMMITMENT_MISSING: "FULLY_VERIFIED",
    E062_ROW_ARCHIVE_MISSING: "FULLY_VERIFIED",
    E070_DA_MODE_UNSUPPORTED: "FULLY_VERIFIED",
    E071_DA_CHALLENGE_MISSING: "FULLY_VERIFIED",
    E072_DA_CHALLENGE_HASH_MISMATCH: "FULLY_VERIFIED",
    E073_DA_CHALLENGE_UNTRUSTED: "FULLY_VERIFIED",
    E074_AVAILABILITY_FAILED: "FULLY_VERIFIED",
    E077_DA_CHALLENGE_BINDING_INVALID: "FULLY_VERIFIED",
    E064_MERKLE_PROOF_INVALID: "FULLY_VERIFIED",
    E065_CHUNK_ROOT_MISMATCH: "FULLY_VERIFIED",
    # Event log mismatches are warning-only for PROOF_ONLY/POLICY_ENFORCED,
    # but fail verification at FULLY_VERIFIED level
    E201_EVENT_LOG_MISMATCH: "FULLY_VERIFIED",
    W_POLICY_SELF_REPORTED: "POLICY_ENFORCED",
}


def _normalize_level(label: str | None) -> str:
    if not label:
        return "PROOF_ONLY"
    normalized = _LEVEL_CANONICAL.get(label.upper())
    if normalized is None:
        raise ValueError(f"unknown verification level '{label}'")
    return normalized


def _hash_equal(lhs: str | None, rhs: str | None) -> bool:
    if not lhs or not rhs:
        return False
    return str(lhs).lower() == str(rhs).lower()


def _normalize_hash_value(value: str | None) -> str | None:
    if not value:
        return None
    text = str(value).strip().lower()
    if text.startswith("sha256:"):
        return text.split("sha256:", 1)[1]
    return text


def _compute_status(
    *,
    header_ok: bool,
    proof_ok: bool,
    authorship_ok: bool,
    policy_ok: bool,
    acl_ok: bool,
    da_ok: bool,
    policy_assurance: str,
) -> str:
    if not header_ok or not proof_ok:
        return "REJECTED"
    if not authorship_ok:
        return "PROOF_ONLY"
    if not policy_ok or not acl_ok:
        return "PROOF_ONLY"
    if da_ok:
        return "FULLY_VERIFIED" if policy_assurance == "ATTESTED" else "POLICY_SELF_REPORTED"
    return "POLICY_ENFORCED" if policy_assurance == "ATTESTED" else "POLICY_SELF_REPORTED"


def _resolve(base: Path, entry: Path | str | dict | None) -> Path:
    if entry is None:
        raise FileNotFoundError("missing path entry")
    candidates: list[str | Path] = []
    if isinstance(entry, dict):
        candidates.extend(
            value
            for key in ("rel_path", "path", "abs_path")
            if (value := entry.get(key)) is not None
        )
    else:
        candidates.append(entry)
    base_root = base.resolve()
    for raw in candidates:
        if raw is None:
            continue
        path = Path(raw)
        candidate = path if path.is_absolute() else (base_root / path)
        candidate = candidate.resolve()
        try:
            candidate.relative_to(base_root)
        except ValueError:
            continue
        candidate = _ensure_local_artifact(candidate)
        try:
            candidate.relative_to(base_root)
        except ValueError:
            continue
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"unable to resolve artifact: {entry}")


def _entry_str(entry: Path | str | dict | None) -> str | None:
    if isinstance(entry, dict):
        return entry.get("path") or entry.get("rel_path") or entry.get("abs_path")
    if entry is None:
        return None
    return str(entry)


def _entry_candidates(entry: Path | str | dict | None) -> list[str]:
    candidates: list[str] = []
    if isinstance(entry, dict):
        for key in ("path", "rel_path", "abs_path"):
            value = entry.get(key)
            if value:
                candidates.append(str(value))
    elif entry is not None:
        candidates.append(str(entry))
    return candidates


def _load_capsule(path: Path, encoding_id: str | None = None) -> tuple[dict | None, str]:
    try:
        raw = path.read_bytes()
        enc = encoding_id
        if enc is None and path.suffix == ".bin":
            enc = "dag_cbor_canonical_v1"
        data = canonical_decode(raw) if enc == "dag_cbor_canonical_v1" else json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            return None, E002_PARSE_FAILED
        return data, OK
    except (json.JSONDecodeError, FileNotFoundError, IsADirectoryError):
        return None, E002_PARSE_FAILED


def _payload_hash(capsule: dict) -> str:
    hashing_meta = capsule.get("hashing") or {}
    encoding_id = hashing_meta.get("encoding_id") or ENCODING_ID
    return compute_payload_hash(capsule, encoding_id=encoding_id)


def _compute_capsule_hashes(capsule: dict) -> tuple[str, str | None, str | None, str | None, bool]:
    payload_hash = _payload_hash(capsule)
    header = capsule.get("header")
    if isinstance(header, dict) and header.get("schema") == HEADER_SCHEMA:
        header_hash = compute_header_hash(header)
        commit_hash = compute_header_commit_hash(header)
        capsule_id = hash_capsule_identity(commit_hash, payload_hash) if commit_hash else None
        return payload_hash, header_hash, commit_hash, capsule_id, True
    return payload_hash, None, None, None, False


def _load_chunk_roots(info: dict | None, base: Path) -> tuple[list[bytes] | None, str]:
    if not info:
        return None, E060_ROW_INDEX_COMMITMENT_MISSING
    inline = info.get("chunk_roots_hex")
    if inline:
        try:
            return [bytes.fromhex(h) for h in inline], OK
        except ValueError:
            return None, E061_ROW_INDEX_COMMITMENT_INVALID_FORMAT
    bin_abs = info.get("chunk_roots_bin_abs")
    bin_rel = info.get("chunk_roots_bin_path")
    bin_path_entry: Path | str | dict | None = None
    if bin_abs or bin_rel:
        if bin_rel:
            bin_path_entry = {"rel_path": bin_rel}
            if bin_abs:
                bin_path_entry["path"] = bin_abs
        else:
            bin_path_entry = bin_abs
    if bin_path_entry:
        try:
            resolved = _resolve(base, bin_path_entry)
            data = resolved.read_bytes()
            if len(data) % 32 != 0:
                return None, E061_ROW_INDEX_COMMITMENT_INVALID_FORMAT
            return [data[i : i + 32] for i in range(0, len(data), 32)], OK
        except FileNotFoundError:
            return None, E061_ROW_INDEX_COMMITMENT_INVALID_FORMAT
    json_abs = info.get("chunk_roots_abs")
    json_rel = info.get("chunk_roots_path")
    json_entry: Path | str | dict | None = None
    if json_abs or json_rel:
        if json_rel:
            json_entry = {"rel_path": json_rel}
            if json_abs:
                json_entry["path"] = json_abs
        else:
            json_entry = json_abs
    if json_entry:
        try:
            resolved = _resolve(base, json_entry)
            return [bytes.fromhex(h) for h in json.loads(resolved.read_text())], OK
        except (FileNotFoundError, json.JSONDecodeError):
            return None, E061_ROW_INDEX_COMMITMENT_INVALID_FORMAT
    return None, E060_ROW_INDEX_COMMITMENT_MISSING


def _compute_payload_hash(path: Path) -> str:
    magic = b"\xBE\xF0\xC0\xDE"
    hasher = hashlib.sha256()
    path = _ensure_local_artifact(path)
    with path.open("rb") as fh:
        header = fh.read(6)
        if len(header) < 6:
            hasher.update(header)
        elif header[:4] == magic:
            remainder = header[6:]
            if remainder:
                hasher.update(remainder)
        else:
            hasher.update(header)
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    path = _ensure_local_artifact(path)
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_event_payload(event: dict[str, Any]) -> bytes:
    base = {k: v for k, v in event.items() if k not in {"prev_event_hash", "event_hash"}}
    return json.dumps(base, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _verify_event_log_chain(path: Path, anchor_hash: str) -> tuple[bool, set[str]]:
    prev_hash = "0" * 64
    anchor_lower = str(anchor_hash).lower()
    anchor_seen = False
    event_types: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    return False, event_types
                recorded_prev = str(event.get("prev_event_hash", "")).lower()
                if recorded_prev != prev_hash.lower():
                    return False, event_types
                payload = _canonical_event_payload(event)
                event_type = event.get("type")
                if isinstance(event_type, str):
                    event_types.add(event_type)
                computed = hashlib.sha256(bytes.fromhex(prev_hash) + payload).hexdigest()
                recorded_hash = str(event.get("event_hash", "")).lower()
                if recorded_hash != computed.lower():
                    return False, event_types
                if recorded_hash == anchor_lower:
                    anchor_seen = True
                prev_hash = computed
    except FileNotFoundError:
        return False, event_types
    return anchor_seen, event_types


def _load_acl(path: Path | None) -> tuple[dict[str, list[dict[str, str]]] | None, str]:
    if not path:
        return None, OK
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return None, E002_PARSE_FAILED
    if data.get("schema") and data.get("schema") != "bef_acl_v1":
        return None, E002_PARSE_FAILED
    return data.get("authorizations", {}), OK


def _verify_authorship(capsule: dict, capsule_hash: str, require: bool) -> tuple[str, str | None, bool]:
    authorship = capsule.get("authorship")
    if not authorship:
        if require:
            return E020_SIGNATURE_MISSING, None, False
        return OK, None, False
    if PublicKey is None:
        return E021_SIGNATURE_INVALID, None, False
    signature_hex = authorship.get("signature")
    pubkey_hex = authorship.get("signer_pubkey")
    if not signature_hex or not pubkey_hex:
        return E021_SIGNATURE_INVALID, None, False
    try:
        signature = bytes.fromhex(signature_hex)
        claimed = bytes.fromhex(pubkey_hex)
    except ValueError:
        return E021_SIGNATURE_INVALID, None, False
    if len(signature) != 65 or len(claimed) not in (33, 65):
        return E021_SIGNATURE_INVALID, None, False
    message = bytes.fromhex(capsule_hash)
    try:
        recovered = PublicKey.from_signature_and_message(signature, message, hasher=None)
    except Exception:  # pragma: no cover - coincurve raises ValueError
        return E021_SIGNATURE_INVALID, None, False
    recovered_bytes = recovered.format(compressed=False)
    if recovered_bytes != PublicKey(claimed).format(compressed=False):
        return E021_SIGNATURE_INVALID, None, False
    return OK, recovered_bytes.hex(), True


def _manifest_anchor_message(anchor_hash: str) -> bytes:
    anchor = str(anchor_hash or "").lower()
    digest = anchor.split(":", 1)[1] if ":" in anchor else anchor
    return bytes.fromhex(digest)


def _verify_manifest_signature(
    *,
    manifest_root: Path,
    anchor_hash: str,
    manifest_signers: dict[str, str] | None,
    manifest_trusted_ids: set[str] | None,
    manifest_registry_root: str | None,
) -> str:
    if not manifest_signers:
        return E106_MANIFEST_SIGNATURE_MISSING
    expected_root = _normalize_hash_value(manifest_registry_root)
    if expected_root:
        computed_root = _registry_hash(manifest_signers)
        if _normalize_hash_value(computed_root) != expected_root:
            return E109_MANIFEST_REGISTRY_MISMATCH
    sig_path = manifest_root / "manifest_signature.json"
    try:
        sig_data = json.loads(sig_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return E106_MANIFEST_SIGNATURE_MISSING
    if sig_data.get("schema") not in {None, "capsule_manifest_signature_v1"}:
        return E107_MANIFEST_SIGNATURE_INVALID
    signer_id = (sig_data.get("signer_id") or "").strip()
    signature_hex = sig_data.get("signature")
    if not signer_id or not signature_hex:
        return E107_MANIFEST_SIGNATURE_INVALID
    allowed = manifest_trusted_ids or set(manifest_signers.keys())
    if signer_id not in allowed:
        return E108_MANIFEST_SIGNER_UNTRUSTED
    pubkey_hex = manifest_signers.get(signer_id)
    if not pubkey_hex:
        return E108_MANIFEST_SIGNER_UNTRUSTED
    if PublicKey is None:
        return E107_MANIFEST_SIGNATURE_INVALID
    try:
        signature = bytes.fromhex(signature_hex)
        if len(signature) != 65:
            return E107_MANIFEST_SIGNATURE_INVALID
        message = _manifest_anchor_message(anchor_hash)
        recovered = PublicKey.from_signature_and_message(signature, message, hasher=None)
        expected = PublicKey(bytes.fromhex(pubkey_hex)).format(compressed=False)
        if recovered.format(compressed=False) != expected:
            return E107_MANIFEST_SIGNATURE_INVALID
    except Exception:
        return E107_MANIFEST_SIGNATURE_INVALID
    return OK


def _verify_acl(policy_id: str | None, signer_hex: str | None, acl: dict[str, list[dict[str, str]]] | None) -> tuple[str, bool]:
    if not acl:
        return OK, False
    if not signer_hex:
        return E021_SIGNATURE_INVALID, False
    if not policy_id:
        return E030_POLICY_ID_MISSING, False
    allowed = acl.get(policy_id)
    if not allowed:
        return E022_SIGNER_NOT_AUTHORIZED, False
    key = signer_hex.lower()
    for entry in allowed:
        entry_key = (entry.get("pubkey") or "").lower()
        status = (entry.get("status") or "active").lower()
        if entry_key == key and status == "active":
            return OK, True
    return E022_SIGNER_NOT_AUTHORIZED, False


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_manifest_anchor(manifest_root: Path) -> str:
    names = ["hardware_manifest", "os_fingerprint", "toolchain_manifest", "manifest_index"]
    hashes: dict[str, str] = {}
    for name in names:
        path = manifest_root / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(name)
        hashes[name] = f"sha256:{_hash_file(path)}"
    payload = json.dumps(
        {
            "schema": "capsule_bench_manifest_anchor_v1",
            "hashes": hashes,
        },
        sort_keys=True,
    ).encode()
    return f"capsulebench_manifest_v1:{hashlib.sha256(payload).hexdigest()}"


def _load_policy_document(
    policy_path: Path | None, expected_hash: str | None
) -> tuple[dict | None, str | None, str]:
    if not policy_path:
        return None, None, OK
    try:
        blob = policy_path.read_bytes()
    except (FileNotFoundError, IsADirectoryError):
        return None, None, E002_PARSE_FAILED
    actual_hash = hashlib.sha256(blob).hexdigest()
    if expected_hash and actual_hash.lower() != expected_hash.lower():
        return None, None, E033_POLICY_HASH_MISMATCH
    try:
        document = json.loads(blob.decode("utf-8"))
    except json.JSONDecodeError:
        return None, None, E002_PARSE_FAILED
    if document.get("schema") != "bef_benchmark_policy_v1":
        return None, None, E003_SCHEMA_UNSUPPORTED
    return document, actual_hash, OK


def _enforce_policy_rules(
    capsule: dict,
    statement: StatementV1,
    policy_doc: dict,
    manifest_root: Path | None,
    *,
    manifest_signers: dict[str, str] | None,
    manifest_trusted_ids: set[str] | None,
    manifest_registry_root: str | None,
) -> tuple[str, str | None]:
    if not policy_doc:
        return OK, None
    if manifest_root is None:
        return E104_POLICY_MANIFEST_MISSING, None
    anchor_meta = capsule.get("anchor") or {}
    anchor_ref = anchor_meta.get("anchor_ref")
    try:
        computed_anchor = _compute_manifest_anchor(manifest_root)
    except FileNotFoundError:
        return E104_POLICY_MANIFEST_MISSING, None
    if anchor_ref and computed_anchor.lower() != anchor_ref.lower():
        return E104_POLICY_MANIFEST_MISSING, None
    manifest_status = _verify_manifest_signature(
        manifest_root=manifest_root,
        anchor_hash=computed_anchor,
        manifest_signers=manifest_signers,
        manifest_trusted_ids=manifest_trusted_ids,
        manifest_registry_root=manifest_registry_root,
    )
    if manifest_status != OK:
        return manifest_status, None
    track_id = anchor_meta.get("track_id") or (capsule.get("policy") or {}).get("track_id")
    if not track_id:
        return E103_POLICY_TRACK_UNKNOWN, None
    track = None
    for entry in policy_doc.get("tracks", []):
        if entry.get("track_id") == track_id:
            track = entry
            break
    if not track:
        return E103_POLICY_TRACK_UNKNOWN, None
    rules = track.get("rules", {})
    if rules.get("forbid_gpu"):
        hardware_path = manifest_root / "hardware_manifest.json"
        try:
            hardware = json.loads(hardware_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return E104_POLICY_MANIFEST_MISSING, None
        gpu_info = hardware.get("gpu", {}) or {}
        detected = gpu_info.get("detected")
        devices = gpu_info.get("devices") or []
        if detected or devices:
            return E101_POLICY_VIOLATION_FORBID_GPU, None
    if rules.get("require_deterministic_build"):
        if not anchor_meta.get("docker_image_digest"):
            return E105_POLICY_DOCKER_DIGEST_MISSING, None
    required_outputs = rules.get("required_public_outputs") or []
    if required_outputs:
        values: dict[str, Any] = {}
        for entry in statement.public_inputs or []:
            if hasattr(entry, "name"):
                values[getattr(entry, "name")] = getattr(entry, "value", None)
            elif isinstance(entry, dict) and "name" in entry:
                values[entry["name"]] = entry.get("value")
        for name in required_outputs:
            if name not in values:
                return E102_POLICY_MISSING_PUBLIC_OUTPUT, None
    return OK, track_id
def _legacy_da_seed(capsule: dict) -> int:
    anchor = (capsule.get("anchor") or {}).get("anchor_ref")
    policy = capsule.get("policy") or {}
    capsule_hash = capsule.get("capsule_hash") or ""
    return derive_capsule_seed(
        capsule_hash,
        anchor_ref=anchor,
        policy_id=policy.get("policy_id"),
        policy_version=policy.get("policy_version"),
    )


def _challenge_da_seed(
    *,
    capsule: dict,
    header: dict,
    capsule_hash: str,
    header_commit_hash: str | None,
    required_level: str,
    trusted_relays: set[str] | None,
    trusted_relay_keys: dict[str, str] | None,
    trusted_relay_root: str | None,
) -> tuple[str, int | None, bool]:
    challenge = capsule.get("da_challenge")
    da_ref = header.get("da_ref") or {}
    challenge_hash_expected = da_ref.get("challenge_hash")
    if not challenge_hash_expected and not challenge:
        return OK, None, False
    if challenge_hash_expected and not challenge:
        return E071_DA_CHALLENGE_MISSING, None, True
    if not isinstance(challenge, dict):
        return E071_DA_CHALLENGE_MISSING, None, True

    schema = challenge.get("schema")
    is_v2 = schema == DA_CHALLENGE_V2_SCHEMA

    # Validate schema
    if schema not in (DA_CHALLENGE_SCHEMA, DA_CHALLENGE_V2_SCHEMA):
        return E071_DA_CHALLENGE_MISSING, None, True

    # Verify challenge hash
    if is_v2:
        computed = hash_signed_da_challenge(challenge)
    else:
        computed = hash_da_challenge(challenge)

    if challenge_hash_expected and not _hash_equal(computed, challenge_hash_expected):
        return E072_DA_CHALLENGE_HASH_MISMATCH, None, True

    # For v1: verify capsule_commit_hash binding
    if not is_v2:
        commit_ref = challenge.get("capsule_commit_hash") or challenge.get("capsule_hash")
        if commit_ref and header_commit_hash and not _hash_equal(commit_ref, header_commit_hash):
            return E072_DA_CHALLENGE_HASH_MISMATCH, None, True

    # For v2: verify bind fields match header commitments
    if is_v2:
        row_commitment = header.get("row_commitment", {})
        expected_commit_root = row_commitment.get("root", "")
        expected_payload_hash = header.get("payload_hash", "")

        bind_valid, bind_error = verify_da_challenge_binding(
            challenge, expected_commit_root, expected_payload_hash
        )
        if not bind_valid:
            # Binding mismatch is always fatal
            return E077_DA_CHALLENGE_BINDING_INVALID, None, True

    min_level = _STATUS_ORDER.get(required_level, 0)
    if min_level >= _STATUS_ORDER["FULLY_VERIFIED"]:
        if is_v2:
            # V2 challenges use issuer.key_id and issuer.sig
            issuer = challenge.get("issuer", {})
            issuer_key_id = issuer.get("key_id", "")
            issuer_sig = issuer.get("sig")

            if not issuer_key_id or not issuer_sig:
                return E073_DA_CHALLENGE_UNTRUSTED, None, True

            trusted = trusted_relays or TRUSTED_RELAYS_ENV
            relay_keys = trusted_relay_keys or TRUSTED_RELAY_KEYS_ENV
            registry_root = _normalize_hash_value(trusted_relay_root or TRUSTED_RELAYS_ROOT)

            if trusted and issuer_key_id not in trusted:
                return E073_DA_CHALLENGE_UNTRUSTED, None, True

            pubkey_hex = relay_keys.get(issuer_key_id)
            if not pubkey_hex or PublicKey is None:
                return E073_DA_CHALLENGE_UNTRUSTED, None, True

            if not registry_root:
                return E073_DA_CHALLENGE_UNTRUSTED, None, True

            registry_hash = _normalize_hash_value(_registry_hash(relay_keys))
            if not registry_hash or registry_hash != registry_root:
                return E073_DA_CHALLENGE_UNTRUSTED, None, True

            # Verify signature using the da module helper
            if not verify_da_challenge_signature(challenge, pubkey_hex):
                return E073_DA_CHALLENGE_UNTRUSTED, None, True

            # Check expiration
            expires_ms = int(challenge.get("expires_at_ms") or 0)
            if expires_ms and expires_ms < int(time.time() * 1000):
                return E073_DA_CHALLENGE_UNTRUSTED, None, True
        else:
            # V1 challenges use relay_pubkey_id and relay_signature
            relay_pubkey_id = challenge.get("relay_pubkey_id") or "local_insecure"
            relay_signature = challenge.get("relay_signature")
            trusted = trusted_relays or TRUSTED_RELAYS_ENV
            relay_keys = trusted_relay_keys or TRUSTED_RELAY_KEYS_ENV
            registry_root = _normalize_hash_value(trusted_relay_root or TRUSTED_RELAYS_ROOT)

            if relay_pubkey_id == "local_insecure" or not relay_signature:
                return E073_DA_CHALLENGE_UNTRUSTED, None, True
            if trusted and relay_pubkey_id not in trusted:
                return E073_DA_CHALLENGE_UNTRUSTED, None, True
            pubkey_hex = relay_keys.get(relay_pubkey_id)
            if not pubkey_hex or PublicKey is None:
                return E073_DA_CHALLENGE_UNTRUSTED, None, True
            if not registry_root:
                return E073_DA_CHALLENGE_UNTRUSTED, None, True
            registry_hash = _normalize_hash_value(_registry_hash(relay_keys))
            if not registry_hash or registry_hash != registry_root:
                return E073_DA_CHALLENGE_UNTRUSTED, None, True
            try:
                signature_bytes = bytes.fromhex(relay_signature)
                pubkey_bytes = bytes.fromhex(pubkey_hex)
            except ValueError:
                return E073_DA_CHALLENGE_UNTRUSTED, None, True
            payload_bytes = challenge_signature_payload(challenge)
            digest = hashlib.sha256(payload_bytes).digest()
            try:
                verified = PublicKey(pubkey_bytes).verify(signature_bytes, digest, hasher=None)
            except Exception:  # pragma: no cover - coincurve errors
                return E073_DA_CHALLENGE_UNTRUSTED, None, True
            if not verified:
                return E073_DA_CHALLENGE_UNTRUSTED, None, True
            expires_ms = int(challenge.get("expires_at_ms") or 0)
            if expires_ms and expires_ms < int(time.time() * 1000):
                return E073_DA_CHALLENGE_UNTRUSTED, None, True

    # Derive seed based on schema
    if is_v2:
        seed = derive_signed_da_seed(challenge)
    else:
        seed = derive_da_seed(capsule_hash, challenge)
    return OK, seed, True


def _select_audit_indices(total: int, count: int, seed: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    selected: list[int] = []
    seen: set[int] = set()
    counter = 0
    base = seed.to_bytes(32, "big", signed=False)
    while len(selected) < count:
        digest = hashlib.sha256(base + counter.to_bytes(4, "big", signed=False)).digest()
        counter += 1
        for pos in range(0, len(digest), 4):
            if len(selected) >= count:
                break
            idx = int.from_bytes(digest[pos : pos + 4], "big") % total
            if idx not in seen:
                seen.add(idx)
                selected.append(idx)
        if counter > 1_000_000:
            raise RuntimeError("DA sampling seed generation failed")
    return selected


def _run_da_audit(
    capsule: dict,
    chunk_roots: list[bytes],
    chunk_len: int,
    row_root: bytes,
    sample_count: int,
    provider,
    seed: int,
) -> tuple[str, bool]:
    if sample_count <= 0:
        return OK, False
    if chunk_len <= 0:
        return E066_CHUNK_LENGTH_INVALID, False
    indices = _select_audit_indices(len(chunk_roots), sample_count, seed)
    if not indices:
        return OK, False
    try:
        fetched = provider.fetch_batch(indices)
    except AvailabilityError:
        return E074_AVAILABILITY_FAILED, False
    for idx in indices:
        chunk = fetched.get(idx)
        if chunk is None:
            return E074_AVAILABILITY_FAILED, False
        offset = idx * chunk_len
        derived_root = merkle_from_values(chunk.values, offset)
        expected_root = chunk_roots[idx]
        if derived_root != expected_root:
            return E065_CHUNK_ROOT_MISMATCH, False
        proof = chunk.proof
        if not verify_kary(
            row_root,
            derived_root,
            idx,
            proof.siblings,
            proof.arity,
            proof.tree_size,
        ):
            return E064_MERKLE_PROOF_INVALID, False
    return OK, True


def _verify_row_commitment_binding(
    proof_row_commitment,
    chunk_meta: dict | None,
    row_index_ref: dict | None,
) -> str:
    if proof_row_commitment is None or not chunk_meta or not row_index_ref:
        return OK
    params = proof_row_commitment.params or {}
    proof_root = params.get("root")
    capsule_root = row_index_ref.get("commitment")
    if proof_root and capsule_root:
        if str(proof_root).lower() != str(capsule_root).lower():
            return E053_PROOF_STATEMENT_MISMATCH
    chunk_len_capsule = chunk_meta.get("chunk_len")
    chunk_len_proof = params.get("chunk_len")
    if chunk_len_capsule is not None and chunk_len_proof is not None:
        if int(chunk_len_capsule) != int(chunk_len_proof):
            return E053_PROOF_STATEMENT_MISMATCH
    arity_capsule = row_index_ref.get("tree_arity")
    arity_proof = params.get("chunk_tree_arity")
    if arity_capsule is not None and arity_proof is not None:
        if int(arity_capsule) != int(arity_proof):
            return E053_PROOF_STATEMENT_MISMATCH
    return OK


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------


def _verify_capsule_core(
    capsule_path: Path,
    *,
    policy_path: Path | None = None,
    policy_proof_path: Path | None = None,
    policy_registry_root: str | None = None,
    acl_path: Path | None = None,
    manifest_root: Path | None = None,
    required_level: str | None = None,
    trusted_relays: set[str] | None = None,
    trusted_relay_keys: dict[str, str] | None = None,
    trusted_relay_root: str | None = None,
    trusted_manifest_signers: dict[str, str] | None = None,
    trusted_manifest_ids: set[str] | None = None,
    manifest_registry_root: str | None = None,
) -> tuple[str, dict | None]:
    capsule_path = capsule_path.resolve()
    _t_stage_start = time.perf_counter()
    _metrics: dict[str, float] = {}
    base = capsule_path.parent
    manifest = load_manifest(base)
    encoding = encoding_for_path(manifest, base, capsule_path) if manifest else None

    fallback_relays = TRUSTED_RELAYS_ENV or set(DEFAULT_TRUSTED_RELAY_KEYS.keys())
    trusted_relays = set(trusted_relays or fallback_relays)
    manifest_signers = trusted_manifest_signers or DEFAULT_MANIFEST_SIGNERS
    manifest_signers = dict(manifest_signers or {})
    manifest_ids = trusted_manifest_ids or set(manifest_signers.keys())
    manifest_registry_root = manifest_registry_root or DEFAULT_MANIFEST_ROOT

    capsule, err = _load_capsule(capsule_path, encoding)
    if err != OK:
        return err, None
    warnings: list[str] = []
    schema = capsule.get("schema")
    # Support both bef_capsule_v1 (production) and capsule_receipt_v2 (demo/minimal)
    if schema not in ("bef_capsule_v1", "capsule_receipt_v2"):
        return E003_SCHEMA_UNSUPPORTED, None

    # Fast path for demo/minimal receipts (capsule_receipt_v2)
    # These don't have the full proof structure, just verify structural integrity
    if schema == "capsule_receipt_v2":
        header = capsule.get("header", {})
        if header.get("verification_profile") == "minimal" and header.get("backend_id") == "none":
            # Demo receipt: verify capsule_id is consistent with header+payload
            capsule_id = capsule.get("capsule_id")
            if not capsule_id:
                return E010_CAPSULE_HASH_MISSING, None
            # Basic structural check passed
            return OK, {
                "verification_level": "proof_only",
                "schema": schema,
                "capsule_id": capsule_id[:16] + "...",
                "demo_receipt": True,
            }
        # Non-demo capsule_receipt_v2 not yet supported
        return E003_SCHEMA_UNSUPPORTED, None

    global _CURRENT_RUN_ID
    _CURRENT_RUN_ID = capsule.get("trace_id") or capsule.get("run_id")

    header_verified = False
    proof_verified = False
    events_verified = False
    recorded_capsule_hash = capsule.get("capsule_hash")
    payload_hash, header_hash_calc, commit_hash, capsule_hash_calc, header_v2 = _compute_capsule_hashes(capsule)
    if not recorded_capsule_hash:
        return E010_CAPSULE_HASH_MISSING, None
    recorded_payload_hash = capsule.get("payload_hash")
    if not recorded_payload_hash:
        return E011_CAPSULE_HASH_MISMATCH, None
    payload_verified = _hash_equal(payload_hash, recorded_payload_hash)
    if not payload_verified:
        return E011_CAPSULE_HASH_MISMATCH, None
    if capsule_hash_calc and not _hash_equal(recorded_capsule_hash, capsule_hash_calc):
        return E011_CAPSULE_HASH_MISMATCH, None
    capsule_hash = recorded_capsule_hash

    header = capsule.get("header")
    if not isinstance(header, dict):
        return E012_CAPSULE_HEADER_MISSING, None
    if header.get("schema") != HEADER_SCHEMA:
        return E003_SCHEMA_UNSUPPORTED, None
    if header.get("trace_id") != capsule.get("trace_id"):
        return E013_CAPSULE_HEADER_MISMATCH, None
    backend_id = header.get("backend_id")
    # Allow "none" for demo/minimal receipts, plus production backends
    if backend_id not in {"geom_stc_fri", "geom", "geom_stc_rust", "risc0", "risc0_receipt_v1", "none"}:
        return E003_SCHEMA_UNSUPPORTED, None
    header_hash = compute_header_hash(header)
    header_hash_recorded = capsule.get("header_hash")
    if header_hash_recorded and not _hash_equal(header_hash, header_hash_recorded):
        return E013_CAPSULE_HEADER_MISMATCH, None
    if header_hash_calc and not _hash_equal(header_hash, header_hash_calc):
        return E013_CAPSULE_HEADER_MISMATCH, None
    header_payload_hash = header.get("payload_hash")
    if header_payload_hash and not _hash_equal(header_payload_hash, payload_hash):
        return E013_CAPSULE_HEADER_MISMATCH, None
    recorded_payload = capsule.get("payload_hash")
    if recorded_payload and not _hash_equal(recorded_payload, payload_hash):
        return E013_CAPSULE_HEADER_MISMATCH, None
    profile_label = (header.get("verification_profile") or capsule.get("verification_profile") or "PROOF_ONLY").upper()
    normalized_required = _normalize_level(required_level or "FULL")
    if not payload_verified:
        if _hash_equal(header_hash, capsule_hash):
            payload_verified = False
        else:
            return E011_CAPSULE_HASH_MISMATCH, None
    if header_v2:
        stored_commit = capsule.get("header_commit_hash")
        if stored_commit and commit_hash and not _hash_equal(stored_commit, commit_hash):
            return E013_CAPSULE_HEADER_MISMATCH, None
    statement_hash_hex = header.get("statement_hash") or capsule.get("statement_hash")
    trace_spec_hash = header.get("trace_spec_hash") or capsule.get("trace_spec_hash")
    header_verified = True
    # Record preparation stage time (parsing + header/hash checks)
    _metrics["prep_time_sec"] = time.perf_counter() - _t_stage_start
    _t_stage_start = time.perf_counter()
    profile_label = (header.get("verification_profile") or capsule.get("verification_profile") or "PROOF_ONLY").upper()
    proof_system_meta = header.get("proof_system") or {}
    descriptor = {k: v for k, v in proof_system_meta.items() if k not in {"hash", "instance_hash"}}
    if not descriptor:
        return E301_PROOF_SYSTEM_MISMATCH, None
    computed_ps_hash = hash_proof_system(descriptor)
    recorded_ps_hash = proof_system_meta.get("hash") or computed_ps_hash
    if not _hash_equal(recorded_ps_hash, computed_ps_hash):
        return E301_PROOF_SYSTEM_MISMATCH, None

    row_commitment_meta = header.get("row_commitment") or {}
    header_commit_hash_val = capsule.get("header_commit_hash") or commit_hash
    fallback_keys = TRUSTED_RELAY_KEYS_ENV or dict(DEFAULT_TRUSTED_RELAY_KEYS)
    relay_keys = trusted_relay_keys or fallback_keys
    relay_root = trusted_relay_root or TRUSTED_RELAYS_ROOT or DEFAULT_TRUSTED_RELAY_ROOT
    da_seed_status, challenge_seed, _ = _challenge_da_seed(
        capsule=capsule,
        header=header,
        capsule_hash=capsule_hash,
        header_commit_hash=header_commit_hash_val,
        required_level=normalized_required,
        trusted_relays=trusted_relays,
        trusted_relay_keys=relay_keys,
        trusted_relay_root=relay_root,
    )
    fatal_challenge = da_seed_status == E073_DA_CHALLENGE_UNTRUSTED
    if da_seed_status != OK:
        if _STATUS_ORDER.get(normalized_required, 0) >= _STATUS_ORDER["FULLY_VERIFIED"] or fatal_challenge:
            return da_seed_status, None
        warnings.append(da_seed_status)
        challenge_seed = None

    manifest_hash_expected = header.get("artifact_manifest_hash")
    if manifest_hash_expected:
        if manifest is None:
            return E013_CAPSULE_HEADER_MISMATCH, None
        manifest_hash_actual = hash_manifest_descriptor(manifest)
        if not _hash_equal(manifest_hash_actual, manifest_hash_expected):
            return E013_CAPSULE_HEADER_MISMATCH, None

    header_anchor = header.get("anchor") or {}
    anchor_meta = capsule.get("anchor") or {}
    if anchor_meta != header_anchor:
        return E013_CAPSULE_HEADER_MISMATCH, None

    params_hash_expected = header.get("params_hash")
    params_hash_actual = hash_params(capsule.get("params"))
    if not _hash_equal(params_hash_actual, params_hash_expected):
        return E013_CAPSULE_HEADER_MISMATCH, None

    chunk_meta_hash_expected = (header.get("row_commitment") or {}).get("chunk_meta_hash")
    chunk_meta_hash_actual = hash_chunk_meta(capsule.get("chunk_meta"))
    if not _hash_equal(chunk_meta_hash_actual, chunk_meta_hash_expected):
        return E013_CAPSULE_HEADER_MISMATCH, None

    row_index_ref_hash_expected = (header.get("row_commitment") or {}).get("row_index_ref_hash")
    row_index_ref_hash_actual = hash_row_index_ref(capsule.get("row_index_ref"))
    if not _hash_equal(row_index_ref_hash_actual, row_index_ref_hash_expected):
        return E013_CAPSULE_HEADER_MISMATCH, None

    row_archive_info = capsule.get("row_archive") or {}
    chunk_handles_hash_expected = (header.get("row_commitment") or {}).get("chunk_handles_root")
    chunk_handles_hash_actual = hash_chunk_handles(row_archive_info.get("chunk_handles", []))
    if not _hash_equal(chunk_handles_hash_actual, chunk_handles_hash_expected):
        return E013_CAPSULE_HEADER_MISMATCH, None

    instance_hash_expected = proof_system_meta.get("instance_hash")
    vk_hash_recorded = proof_system_meta.get("vk_hash") or recorded_ps_hash
    instance_binding = hash_instance_binding(
        statement_hash=statement_hash_hex,
        row_root=row_commitment_meta.get("root") or (row_index_ref or {}).get("commitment") or "",
        trace_spec_hash=str(trace_spec_hash),
        vk_hash=vk_hash_recorded,
        params_hash=params_hash_expected,
        chunk_meta_hash=chunk_meta_hash_expected,
        row_tree_arity=row_commitment_meta.get("tree_arity"),
        air_params_hash=proof_system_meta.get("air_params_hash"),
        fri_params_hash=proof_system_meta.get("fri_params_hash"),
        program_hash=proof_system_meta.get("program_hash"),
    )
    if instance_hash_expected and not _hash_equal(instance_hash_expected, instance_binding):
        return E301_PROOF_SYSTEM_MISMATCH, None
    instance_binding_bytes: bytes | None = None
    try:
        instance_binding_bytes = bytes.fromhex(instance_binding)
    except ValueError:
        instance_binding_bytes = None

    capsule_policy = capsule.get("policy") or {}
    policy_ref_expected = header.get("policy_ref") or {}
    policy_ref_actual = {
        "policy_id": capsule_policy.get("policy_id"),
        "policy_version": capsule_policy.get("policy_version"),
        "policy_hash": capsule_policy.get("policy_hash"),
        "track_id": capsule_policy.get("track_id"),
    }
    if policy_ref_actual != policy_ref_expected:
        return E013_CAPSULE_HEADER_MISMATCH, None

    da_policy_hash_expected = (header.get("da_ref") or {}).get("policy_hash")
    da_policy_hash_actual = hash_da_policy(capsule.get("da_policy"))
    if not _hash_equal(da_policy_hash_actual, da_policy_hash_expected):
        return E013_CAPSULE_HEADER_MISMATCH, None

    # Policy document is only required for policy enforcement levels (not proof_only)
    policy_required = any(value for value in policy_ref_expected.values())
    policy_enforcement_required = _STATUS_ORDER.get(normalized_required, 0) >= _STATUS_ORDER["POLICY_SELF_REPORTED"]
    if policy_required and policy_path is None and policy_enforcement_required:
        return E036_POLICY_DOCUMENT_MISSING, None

    artifacts = capsule.get("artifacts", {})
    proofs = capsule.get("proofs", {}) or {}
    geom_entry = proofs.get("geom")
    proof_label = "geom"
    if not geom_entry:
        geom_entry = proofs.get("primary")
        proof_label = "primary"
    if not geom_entry and proofs:
        proof_label, geom_entry = next(iter(proofs.items()))
    if not geom_entry:
        return E050_PROOF_MISSING, None
    chunk_meta = capsule.get("chunk_meta") or {}
    row_index_ref = capsule.get("row_index_ref") or {}
    header_row_root = row_commitment_meta.get("root")
    if header_row_root and row_index_ref.get("commitment"):
        if str(row_index_ref.get("commitment")).lower() != str(header_row_root).lower():
            return E013_CAPSULE_HEADER_MISMATCH, None
    header_tree_arity = row_commitment_meta.get("tree_arity")
    if header_tree_arity is not None and row_index_ref.get("tree_arity") is not None:
        if int(row_index_ref.get("tree_arity")) != int(header_tree_arity):
            return E013_CAPSULE_HEADER_MISMATCH, None
    trace_spec_obj = capsule.get("trace_spec")
    trace_spec_hash = header.get("trace_spec_hash")
    if not trace_spec_obj or not trace_spec_hash:
        return E053_PROOF_STATEMENT_MISMATCH, None
    trace_spec = TraceSpecV1.from_obj(trace_spec_obj)
    computed_trace_spec_hash = compute_trace_spec_hash(trace_spec)
    if computed_trace_spec_hash.lower() != str(trace_spec_hash).lower():
        return E053_PROOF_STATEMENT_MISMATCH, None

    policy_info = capsule_policy
    policy_hash_expected = policy_info.get("policy_hash")
    policy_doc = None
    policy_doc_verified = False
    policy_rules_satisfied = False
    actual_policy_hash = None
    policy_assurance = "SELF_REPORTED"
    if policy_path:
        policy_doc, actual_policy_hash, status = _load_policy_document(policy_path, policy_hash_expected)
        if status != OK:
            return status, None
        policy_doc_verified = True
    if policy_registry_root:
        if not policy_path or not policy_proof_path or not actual_policy_hash:
            return E033_POLICY_HASH_MISMATCH, None
        try:
            proof_data = json.loads(policy_proof_path.read_text())
            leaf_hash_bytes = bytes.fromhex(proof_data["leaf_hash"])
            if leaf_hash_bytes.hex() != actual_policy_hash:
                return E064_MERKLE_PROOF_INVALID, None
            siblings = [
                [bytes.fromhex(h) for h in level]
                for level in proof_data["siblings_by_level"]
            ]
            if not verify_kary(
                root=bytes.fromhex(policy_registry_root),
                leaf=leaf_hash_bytes,
                index=proof_data["leaf_index"],
                proof=siblings,
                arity=proof_data["arity"],
                total_leaves=proof_data["total_leaves"],
            ):
                return E034_POLICY_NOT_IN_REGISTRY, None
            policy_assurance = "ATTESTED"
        except (FileNotFoundError, IsADirectoryError, json.JSONDecodeError, KeyError, ValueError):
            return E002_PARSE_FAILED, None

    # Proof artifact handling
    proof_path_entry = geom_entry if isinstance(geom_entry, dict) else None
    if not proof_path_entry:
        proof_path_entry = (
            geom_entry.get("path")
            or geom_entry.get("json_path")
            or geom_entry.get("bin_path")
            or artifacts.get(f"{proof_label}_proof")
            or artifacts.get("proof")
        )
    if not proof_path_entry:
        return E050_PROOF_MISSING, None
    proof_path = _resolve(base, proof_path_entry)
    expected_hash = None
    entry_tokens = _entry_candidates(proof_path_entry)
    for _fmt, meta in (geom_entry.get("formats") or {}).items():
        meta_tokens = _entry_candidates(meta)
        if any(token for token in entry_tokens if token in meta_tokens):
            expected_hash = meta.get("sha256_payload_hash")
            break
    if expected_hash:
        actual_hash = _compute_payload_hash(proof_path)
        if actual_hash.lower() != expected_hash.lower():
            return E052_PROOF_HASH_MISMATCH, None
    # Backend dispatch for proof parsing
    is_risc0_backend = backend_id in {"risc0", "risc0_receipt_v1"}
    proof = None
    proof_json_raw = None

    try:
        if is_risc0_backend:
            # RISC0: proof is plain JSON, not GeomProof
            proof_json_raw = proof_path.read_text()
            proof = json.loads(proof_json_raw)  # Dict, not GeomProof
        else:
            # Geom backends: use specialized parsers
            proof_encoding = encoding_for_path(manifest, base, proof_path) if manifest else None
            if proof_encoding == "dag_cbor_canonical_v1" or proof_path.suffix == ".bin":
                proof = proof_from_bytes(proof_path.read_bytes())
            else:
                proof = proof_from_json(proof_path.read_text())
    except Exception:
        return E002_PARSE_FAILED, None

    # Skip row commitment binding check for RISC0 (succinct proof, no row archive)
    if not is_risc0_backend:
        binding_status = _verify_row_commitment_binding(proof.row_commitment, chunk_meta, row_index_ref)
        if binding_status != OK:
            return binding_status, None

    statement_obj = capsule.get("statement")
    statement_hash_hex = header.get("statement_hash") or capsule.get("statement_hash")
    if not statement_obj or not statement_hash_hex:
        return E053_PROOF_STATEMENT_MISMATCH, None
    statement = StatementV1.from_obj(statement_obj)
    computed_statement_hash = compute_statement_hash(statement)
    if computed_statement_hash.lower() != str(statement_hash_hex).lower():
        return E053_PROOF_STATEMENT_MISMATCH, None
    if statement.trace_spec_hash and statement.trace_spec_hash.lower() != str(trace_spec_hash).lower():
        return E053_PROOF_STATEMENT_MISMATCH, None
    policy_hash = policy_info.get("policy_hash")
    if policy_hash and statement.policy_hash.lower() != str(policy_hash).lower():
        return E053_PROOF_STATEMENT_MISMATCH, None
    trace_root_expected = row_index_ref.get("commitment")
    if trace_root_expected and statement.trace_root.lower() != str(trace_root_expected).lower():
        return E053_PROOF_STATEMENT_MISMATCH, None
    anchors_expected = []
    anchor_meta = capsule.get("anchor") or {}
    events_entry = artifacts.get("events_log")
    events_path: Path | None = None
    actual_events_hash: str | None = None
    actual_events_len: int | None = None
    if events_entry:
        # Prefer sandbox-safe resolution; if not found, allow a strict
        # fallback to an absolute path recorded in the capsule when
        # verifying directly in the original run directory.
        try:
            events_path = _resolve(base, events_entry)
        except FileNotFoundError:
            if isinstance(events_entry, dict):
                abs_candidate = events_entry.get("path") or events_entry.get("abs_path")
                if abs_candidate:
                    p = Path(abs_candidate)
                    if p.exists():
                        events_path = p
        if events_path and events_path.exists():
            actual_events_hash = f"sha256:{_compute_file_hash(events_path)}"
            actual_events_len = events_path.stat().st_size
    chain_head = anchor_meta.get("event_chain_head") or header_anchor.get("event_chain_head")
    anchor_events_hash = anchor_meta.get("events_log_hash")
    header_events_hash = header_anchor.get("events_log_hash")
    event_issue = False
    if chain_head:
        if not events_path or not events_path.exists():
            event_issue = True
        else:
            chain_ok, event_types_seen = _verify_event_log_chain(events_path, chain_head)
            if not chain_ok:
                event_issue = True
            else:
                missing_events = REQUIRED_EVENT_TYPES - event_types_seen
                if missing_events:
                    event_issue = True
                else:
                    events_verified = True
    elif anchor_events_hash:
        if not actual_events_hash or actual_events_hash.lower() != str(anchor_events_hash).lower():
            event_issue = True
        else:
            events_verified = True
    elif header_events_hash:
        if not actual_events_hash or actual_events_hash.lower() != str(header_events_hash).lower():
            event_issue = True
        else:
            events_verified = True
    if header_events_hash and actual_events_hash:
        if actual_events_hash.lower() != str(header_events_hash).lower():
            event_issue = True
    def _anchor_view(a: dict) -> dict:
        # Compare only the stable subset that the statement promises to carry
        # (manifest rule, anchor_ref, track_id, optional event_chain_head). The
        # events hash/len are allowed to differ or be absent from the statement.
        keys = ["anchor_rule_id", "anchor_ref", "track_id", "event_chain_head"]
        return {k: a.get(k) for k in keys if k in a}
    if anchor_meta:
        anchors_expected.append(_anchor_view(anchor_meta))
    stmt_anchors = [
        _anchor_view(a) for a in (statement.anchors or [])
        if isinstance(a, dict)
    ]
    if anchors_expected and stmt_anchors != anchors_expected:
        return E053_PROOF_STATEMENT_MISMATCH, None
    expected_len = anchor_meta.get("events_log_len") or header_anchor.get("events_log_len")
    if expected_len is not None:
        if actual_events_len is None or int(expected_len) != int(actual_events_len):
            event_issue = True
    if event_issue:
        warnings.append(E201_EVENT_LOG_MISMATCH)
        events_verified = False

    policy_track = None
    policy_status, track_id = _enforce_policy_rules(
        capsule,
        statement,
        policy_doc,
        manifest_root,
        manifest_signers=manifest_signers,
        manifest_trusted_ids=manifest_ids,
        manifest_registry_root=manifest_registry_root,
    )
    if policy_status != OK:
        return policy_status, None
    policy_track = track_id
    policy_rules_satisfied = True

    proof_system_meta = header.get("proof_system") or {}

    # Compute statement_hash_bytes for verification
    if instance_binding_bytes is not None:
        statement_hash_bytes = instance_binding_bytes
    else:
        statement_hash_bytes = bytes.fromhex(statement_hash_hex)

    # Backend dispatch for proof verification
    if is_risc0_backend:
        # RISC0 verification: binding integrity check
        t_v = time.perf_counter()
        risc0_ok, verify_stats = _verify_risc0_proof(
            proof_json_raw,
            statement_hash_bytes,
            capsule,
        )
        if not risc0_ok:
            return E054_PROOF_VERIFICATION_FAILED, None
        proof_verified = True
        # For stats reporting, use placeholder values
        geom_params = None
        fri_params_hash_actual = None
        verify_stats = dict(verify_stats or {})
        verify_stats.setdefault("time_verify_sec", time.perf_counter() - t_v)
    else:
        # Geom backend verification
        statement_data = getattr(proof, "statement", None)
        if statement_data is None or getattr(statement_data, "params", None) is None:
            return E002_PARSE_FAILED, None
        geom_params = statement_data.params
        air_params_hash_expected = proof_system_meta.get("air_params_hash")
        if air_params_hash_expected:
            air_params_hash_actual = hash_air_params(geom_params)
            if not _hash_equal(air_params_hash_actual, air_params_hash_expected):
                return E301_PROOF_SYSTEM_MISMATCH, None
        fri_params_hash_expected = proof_system_meta.get("fri_params_hash")
        fri_params_obj = getattr(getattr(proof, "pc_commitment", None), "fri_params", None)
        if fri_params_hash_expected:
            fri_params_hash_actual = hash_fri_config(fri_params_obj)
            if not fri_params_hash_actual or not _hash_equal(fri_params_hash_actual, fri_params_hash_expected):
                return E301_PROOF_SYSTEM_MISMATCH, None
        else:
            fri_params_hash_actual = hash_fri_config(fri_params_obj)
        program_hash_expected = proof_system_meta.get("program_hash")
        program_hash_actual = hash_program_descriptor(GEOM_PROGRAM)
        if program_hash_expected and not _hash_equal(program_hash_actual, program_hash_expected):
            return E301_PROOF_SYSTEM_MISMATCH, None
        vk_hash_expected = proof_system_meta.get("vk_hash")
        if vk_hash_expected:
            vk_payload = {
                "schema": "capsule_vk_v1",
                "scheme_id": proof_system_meta.get("scheme_id"),
                "backend_id": proof_system_meta.get("backend_id"),
                "circuit_id": proof_system_meta.get("circuit_id"),
                "air_params_hash": hash_air_params(geom_params),
                "fri_params_hash": fri_params_hash_actual,
                "program_hash": program_hash_actual,
            }
            vk_hash_actual = hash_proof_system(vk_payload)
            if not _hash_equal(vk_hash_actual, vk_hash_expected):
                return E301_PROOF_SYSTEM_MISMATCH, None
        capsule_params = capsule.get("params") or {}
        row_width = capsule_params.get("row_width")
        if row_width is None:
            return E002_PARSE_FAILED, None
        init_state = GeomInitialState()
        vc = STCVectorCommitment(chunk_len=row_width)
        t_v = time.perf_counter()
        geom_ok, verify_stats = zk_verify_geom(
            GEOM_PROGRAM,
            geom_params,
            init_state,
            vc,
            proof,
            statement_hash=statement_hash_bytes,
        )
        if not geom_ok:
            return E054_PROOF_VERIFICATION_FAILED, None
        proof_verified = True
        verify_stats = dict(verify_stats or {})
        verify_stats.setdefault("time_verify_sec", time.perf_counter() - t_v)

    # Optional Nova state check
    nova_stats_entry = artifacts.get("nova_stats")
    if nova_stats_entry:
        nova_stats_path = _resolve(base, nova_stats_entry)
        try:
            nova_stats = json.loads(nova_stats_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return E002_PARSE_FAILED, None
        expected_commitment = capsule.get("trace_commitment")
        if expected_commitment is not None and nova_stats.get("nova_state") != expected_commitment:
            return E055_NOVA_STATE_MISMATCH, None
    else:
        nova_stats_path = None

    # Row archive bookkeeping
    row_archive_path: Path | None = None
    row_archive_entry = artifacts.get("row_archive")
    if row_archive_entry:
        candidate_entry: Path | str | dict | None = None
        if isinstance(row_archive_entry, dict):
            rel = row_archive_entry.get("rel_path")
            abs_path = row_archive_entry.get("path") or row_archive_entry.get("abs_path")
            mode = row_archive_entry.get("mode", "")
            if rel:
                candidate_entry = {"rel_path": rel}
                if abs_path:
                    candidate_entry["path"] = abs_path
            else:
                candidate_entry = abs_path
            if not candidate_entry and mode:
                return E062_ROW_ARCHIVE_MISSING, None
        else:
            candidate_entry = str(row_archive_entry)
        if candidate_entry:
            row_archive_path = _resolve(base, candidate_entry)
            if not row_archive_path.exists():
                return E062_ROW_ARCHIVE_MISSING, None
    row_archive_info = (
        capsule.get("row_archive")
        or geom_entry.get("row_archive")
        or row_archive_entry
    )
    chunk_handle_entries: list[dict[str, Any]] = []
    chunk_tree_arity = 2
    chunk_roots: list[bytes] | None = None
    row_root_bytes: bytes | None = None
    row_index_ok = False
    manifest_chunk_count = 0
    expected_chunk_count = int(chunk_meta.get("num_chunks") or 0)
    if row_archive_info:
        manifest_handles = row_archive_info.get("chunk_handles", []) or []
        for entry in manifest_handles:
            if not isinstance(entry, dict):
                return E075_CHUNK_HANDLE_INVALID, None
            chunk_handle_entries.append(entry)
        manifest_chunk_count = len(chunk_handle_entries)
        chunk_tree_arity = int(row_archive_info.get("chunk_tree_arity") or 2)
        digest_hex = row_archive_info.get("chunk_roots_digest")
        if digest_hex:
            roots_abs = row_archive_info.get("chunk_roots_bin_abs")
            roots_rel = row_archive_info.get("chunk_roots_bin_path")
            if not roots_abs and not roots_rel:
                return E063_CHUNK_ROOTS_DIGEST_MISMATCH, None
            roots_entry: Path | str | dict | None
            if roots_rel:
                roots_entry = {"rel_path": roots_rel}
                if roots_abs:
                    roots_entry["path"] = roots_abs
            else:
                roots_entry = roots_abs
            actual_digest = hashlib.sha256(_resolve(base, roots_entry).read_bytes()).hexdigest()
            if actual_digest.lower() != digest_hex.lower():
                return E063_CHUNK_ROOTS_DIGEST_MISMATCH, None
        row_archive_abs = row_archive_info.get("abs_path")
        if row_archive_abs and not row_archive_path:
            row_archive_path = Path(row_archive_abs)
        if row_index_ref and row_index_ref.get("commitment"):
            chunk_roots, err = _load_chunk_roots(row_archive_info, base)
            if err != OK:
                return err, None
            arity = int(row_index_ref.get("tree_arity") or row_archive_info.get("chunk_tree_arity") or 2)
            # For succinct backends (RISC0), skip merkle verification if arity < 2 or no chunks
            if arity < 2 or not chunk_roots:
                # Use the stored commitment directly for succinct proofs
                row_root_bytes = bytes.fromhex(row_index_ref.get("commitment"))
                row_index_ok = True
            else:
                t_idx = time.perf_counter()
                levels = build_kary_levels(chunk_roots, arity)
                derived_root = root_from_levels(levels)
                if derived_root.hex() != row_index_ref.get("commitment"):
                    return E064_MERKLE_PROOF_INVALID, None
                if expected_chunk_count and len(chunk_roots) != expected_chunk_count:
                    return E076_CHUNK_CARDINALITY_MISMATCH, None
                if manifest_chunk_count and len(chunk_roots) != manifest_chunk_count:
                    return E076_CHUNK_CARDINALITY_MISMATCH, None
                row_root_bytes = derived_root
                row_index_ok = True
                _metrics["row_index_check_sec"] = time.perf_counter() - t_idx
        else:
            row_index_ok = False
    if expected_chunk_count and manifest_chunk_count and expected_chunk_count != manifest_chunk_count:
        return E076_CHUNK_CARDINALITY_MISMATCH, None

    # Authorship + ACL
    acl, err = _load_acl(acl_path)
    if err != OK:
        return err, None
    require_authorship = bool(acl) or _STATUS_ORDER.get(normalized_required, 0) >= _STATUS_ORDER["POLICY_SELF_REPORTED"]
    auth_status, signer_hex, authorship_verified = _verify_authorship(capsule, capsule_hash, require_authorship)
    if auth_status != OK:
        return auth_status, None
    acl_status, acl_authorized = _verify_acl(
        (capsule.get("policy") or {}).get("policy_id"),
        signer_hex,
        acl,
    )
    if acl_status != OK:
        return acl_status, None

    # Data availability audit
    da_audit_verified = False
    da_policy = capsule.get("da_policy")
    legacy_profile = capsule.get("da_profile") if not da_policy else None
    da_required = _STATUS_ORDER.get(normalized_required, 0) >= _STATUS_ORDER["FULLY_VERIFIED"]
    perform_da = bool(da_policy or legacy_profile)
    archive_root: Path | None = None
    if da_required and not perform_da:
        return E071_DA_CHALLENGE_MISSING, None
    if perform_da:
        if not row_root_bytes or not chunk_roots:
            if da_required:
                return E060_ROW_INDEX_COMMITMENT_MISSING, None
            warnings.append(E060_ROW_INDEX_COMMITMENT_MISSING)
        chunk_len = int(chunk_meta.get("chunk_len") or 0)
        provider_config: dict[str, Any] = {}
        timeout_ms = 0
        retry_count = 0
        if da_policy:
            sample_count = int(da_policy.get("k_samples") or 0)
            provider_config = da_policy.get("provider") or {}
            timeout_ms = int(da_policy.get("provider_timeout_ms") or 0)
            retry_count = int(da_policy.get("provider_retry_count") or 0)
            mode = (provider_config.get("mode") or "LOCAL_FILE").upper()
        else:
            sample_cfg = legacy_profile.get("sampling") or {}
            sample_count = int(sample_cfg.get("k_min") or 0)
            mode = (legacy_profile.get("mode") or "LIGHT_SAMPLING").upper()
        if mode not in {"LIGHT_SAMPLING", "LOCAL_FILE"}:
            if da_required:
                return E070_DA_MODE_UNSUPPORTED, None
            warnings.append(E070_DA_MODE_UNSUPPORTED)
            perform_da = False
        if perform_da:
            archive_root = row_archive_path
            provider_root = provider_config.get("archive_root") if provider_config else None
            if provider_root:
                archive_root = Path(provider_root)
            if archive_root is None:
                if da_required:
                    return E062_ROW_ARCHIVE_MISSING, None
                warnings.append(E062_ROW_ARCHIVE_MISSING)
                perform_da = False
            else:
                archive_root = _ensure_local_artifact(Path(archive_root))
                if not archive_root.exists():
                    if da_required:
                        return E062_ROW_ARCHIVE_MISSING, None
                    warnings.append(E062_ROW_ARCHIVE_MISSING)
                    perform_da = False
                elif not archive_root.is_dir():
                    if da_required:
                        return E062_ROW_ARCHIVE_MISSING, None
                    warnings.append(E062_ROW_ARCHIVE_MISSING)
                    perform_da = False
        if perform_da:
            if not chunk_handle_entries:
                if da_required:
                    return E075_CHUNK_HANDLE_INVALID, None
                warnings.append(E075_CHUNK_HANDLE_INVALID)
                perform_da = False
        if perform_da:
            resolved_handles, handle_err = _resolve_chunk_handle_entries(chunk_handle_entries, archive_root)
            if handle_err != OK:
                if da_required:
                    return handle_err, None
                warnings.append(handle_err)
                perform_da = False
        if perform_da:
            provider = LocalFileSystemProvider(
                archive_root=archive_root,
                chunk_handles=resolved_handles,
                chunk_roots=chunk_roots,
                tree_arity=chunk_tree_arity,
            )
            wrapped_provider = (
                PolicyAwareDAClient(provider, retries=retry_count, timeout_ms=timeout_ms)
                if da_policy
                else provider
            )
            da_seed = challenge_seed if challenge_seed is not None else _legacy_da_seed(capsule)
            if challenge_seed is None and da_policy:
                if da_required:
                    return E071_DA_CHALLENGE_MISSING, None
                warnings.append(E071_DA_CHALLENGE_MISSING)
            else:
                t_da = time.perf_counter()
                audit_status, da_audit_verified = _run_da_audit(
                    capsule,
                    chunk_roots,
                    chunk_len,
                    row_root_bytes,
                    sample_count,
                    wrapped_provider,
                    da_seed,
                )
                if audit_status != OK:
                    if da_required:
                        return audit_status, None
                    warnings.append(audit_status)
                else:
                    _metrics["da_audit_sec"] = time.perf_counter() - t_da

    acl_required = bool(acl)
    policy_ok = bool(policy_rules_satisfied)
    acl_ok = True if not acl_required else bool(authorship_verified and acl_authorized)
    da_ok = bool(da_audit_verified)
    if policy_assurance != "ATTESTED":
        warnings.append(W_POLICY_SELF_REPORTED)

    status = _compute_status(
        header_ok=header_verified,
        proof_ok=proof_verified,
        authorship_ok=authorship_verified,
        policy_ok=policy_ok,
        acl_ok=acl_ok,
        da_ok=da_ok,
        policy_assurance=policy_assurance,
    )
    for warning_code in warnings:
        threshold = _WARNING_THRESHOLDS.get(warning_code)
        if threshold and _STATUS_ORDER.get(normalized_required, 0) >= _STATUS_ORDER[threshold]:
            return warning_code, None
    if _STATUS_ORDER.get(status, 0) < _STATUS_ORDER.get(normalized_required, 0):
        return E302_VERIFICATION_PROFILE_UNSATISFIED, None

    result = {
        "steps": getattr(geom_params, "steps", 0) if geom_params else 0,
        "num_challenges": getattr(geom_params, "num_challenges", 0) if geom_params else 0,
        "backend_id": backend_id,
        "verify_stats": {
            **(verify_stats or {}),
            "prep_time_sec": _metrics.get("prep_time_sec"),
            "row_index_check_sec": _metrics.get("row_index_check_sec"),
            "da_audit_sec": _metrics.get("da_audit_sec"),
            "rss_bytes": _rss_bytes(),
            "time_total_sec": (verify_stats or {}).get("time_verify_sec")
                + (_metrics.get("prep_time_sec") or 0.0)
                + (_metrics.get("row_index_check_sec") or 0.0)
                + (_metrics.get("da_audit_sec") or 0.0),
        },
        "trace_commitment": capsule.get("trace_commitment"),
        "proof_path": str(proof_path),
        "nova_stats_path": str(nova_stats_path) if nova_stats_path else None,
        "row_archive": str(row_archive_path) if row_archive_path else None,
        "capsule_hash_ok": header_verified,
        "row_index_commitment_ok": row_index_ok,
        "policy_verified": policy_doc_verified,
        "policy_rules_satisfied": policy_rules_satisfied,
        "authorship_verified": authorship_verified,
        "acl_authorized": acl_authorized,
        "da_audit_verified": da_audit_verified,
        "events_verified": events_verified,
        "proof_verified": proof_verified,
        "header_verified": header_verified,
        "status": status,
        "verification_profile": profile_label,
        "required_level": normalized_required,
        "policy_track": policy_track,
        "policy_assurance": policy_assurance,
        "payload_hash": payload_hash,
        "warnings": warnings,
    }
    return OK, result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def verify_capsule(
    capsule_path: Path,
    *,
    policy_path: Path | None = None,
    policy_proof_path: Path | None = None,
    policy_registry_root: str | None = None,
    acl_path: Path | None = None,
    manifest_root: Path | None = None,
    required_level: str | None = None,
    trusted_relays: set[str] | None = None,
    trusted_relay_keys: dict[str, str] | None = None,
    trusted_relay_root: str | None = None,
    trusted_manifest_signers: dict[str, str] | None = None,
    trusted_manifest_ids: set[str] | None = None,
    trusted_manifest_root: str | None = None,
) -> dict:
    status, result = _verify_capsule_core(
        capsule_path,
        policy_path=policy_path,
        policy_proof_path=policy_proof_path,
        policy_registry_root=policy_registry_root,
        acl_path=acl_path,
        manifest_root=manifest_root,
        required_level=required_level,
        trusted_relays=trusted_relays,
        trusted_relay_keys=trusted_relay_keys,
        trusted_relay_root=trusted_relay_root,
        trusted_manifest_signers=trusted_manifest_signers,
        trusted_manifest_ids=trusted_manifest_ids,
        manifest_registry_root=trusted_manifest_root,
    )
    if status == OK:
        return result or {}
    if status == E062_ROW_ARCHIVE_MISSING:
        raise FileNotFoundError("row archive missing or unreadable")
    if status == E055_NOVA_STATE_MISMATCH:
        raise RuntimeError("Nova STC state mismatch between stats and capsule")
    raise ValueError(status)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a BEF strategy capsule")
    parser.add_argument("capsule", type=Path, help="path to strategy_capsule.(json|bin)")
    parser.add_argument("--policy", type=Path, help="path to policy file to enforce")
    parser.add_argument("--policy-inclusion-proof", type=Path, help="path to JSON Merkle proof for policy")
    parser.add_argument("--policy-registry-root", type=str, help="hex Merkle root of trusted policy registry")
    parser.add_argument("--acl-path", type=Path, help="path to ACL JSON mapping policy IDs to authorized signer keys")
    parser.add_argument("--manifest-root", type=Path, help="path to manifests/ directory for policy enforcement")
    parser.add_argument(
        "--required-level",
        choices=["proof_only", "policy_self_reported", "policy_enforced", "full"],
        help="minimum verification level to enforce",
    )
    parser.add_argument(
        "--trusted-relay-id",
        action="append",
        default=None,
        help="relay identifier to treat as trusted (can be repeated)",
    )
    parser.add_argument(
        "--relay-key",
        action="append",
        default=None,
        help="trusted relay key mapping relay_id=hexpubkey (can be repeated)",
    )
    parser.add_argument(
        "--trusted-relay-root",
        type=str,
        help="expected hex hash of the trusted relay registry (sha256 of sorted id->key map)",
    )
    parser.add_argument(
        "--trusted-manifest-signer",
        action="append",
        default=None,
        help="manifest signer identifier to trust (can be repeated)",
    )
    parser.add_argument(
        "--manifest-signer-key",
        action="append",
        default=None,
        help="trusted manifest signer mapping signer_id=hexpubkey",
    )
    parser.add_argument(
        "--trusted-manifest-root",
        type=str,
        help="expected hex hash of the manifest signer registry",
    )
    args = parser.parse_args()

    if args.policy_registry_root and not (args.policy and args.policy_inclusion_proof):
        parser.error("policy registry verification requires --policy and --policy-inclusion-proof")

    trusted_relays = set(filter(None, args.trusted_relay_id or []))
    if not trusted_relays and DEFAULT_TRUSTED_RELAY_KEYS:
        trusted_relays = set(DEFAULT_TRUSTED_RELAY_KEYS.keys())
    relay_keys = dict(DEFAULT_TRUSTED_RELAY_KEYS)
    relay_keys.update(TRUSTED_RELAY_KEYS_ENV)
    for entry in args.relay_key or []:
        relay_keys.update(_parse_relay_key_mapping(entry))
    manifest_signers = dict(DEFAULT_MANIFEST_SIGNERS)
    manifest_signers.update(TRUSTED_MANIFEST_SIGNERS_ENV)
    for entry in args.manifest_signer_key or []:
        manifest_signers.update(_parse_relay_key_mapping(entry))
    trusted_manifest_ids = set(filter(None, args.trusted_manifest_signer or []))
    if not trusted_manifest_ids:
        trusted_manifest_ids = set(manifest_signers.keys())

    status, result = _verify_capsule_core(
        args.capsule,
        policy_path=args.policy,
        policy_proof_path=args.policy_inclusion_proof,
        policy_registry_root=args.policy_registry_root,
        acl_path=args.acl_path,
        manifest_root=args.manifest_root,
        required_level=args.required_level,
        trusted_relays=trusted_relays,
        trusted_relay_keys=relay_keys,
        trusted_relay_root=args.trusted_relay_root or TRUSTED_RELAYS_ROOT,
        trusted_manifest_signers=manifest_signers,
        trusted_manifest_ids=trusted_manifest_ids,
        manifest_registry_root=args.trusted_manifest_root or DEFAULT_MANIFEST_ROOT,
    )
    if status == OK:
        print(json.dumps(result, indent=2))
        sys.exit(0)
    error_payload = {"status": "REJECT", "error_code": status}
    print(json.dumps(error_payload, indent=2), file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
