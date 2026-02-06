"""Data-availability challenge helpers.

v1 challenges support binding to commit_root and payload_hash to prevent
prover-picked challenges. The verifier checks these bindings when FULL
verification is required.
"""
from __future__ import annotations

import hashlib
import os
import time
import uuid
from typing import Any, Dict, Optional

from bef_zk.codec import ENCODING_ID, canonical_encode


DA_CHALLENGE_SCHEMA = "capsule_da_challenge_v1"
DA_CHALLENGE_V2_SCHEMA = "stc_da_challenge_v1"
_HASH_PREFIX_DA_CHALLENGE = b"CAPSULE_DA_CHALLENGE_V1::"
_HASH_PREFIX_DA_CHALLENGE_V2 = b"STC_DA_CHALLENGE_V1::"


def _canonical_bytes(payload: Dict[str, Any]) -> bytes:
    return canonical_encode(payload, encoding_id=ENCODING_ID)


def build_da_challenge(
    *,
    capsule_commit_hash: str,
    challenge_id: str | None = None,
    nonce: bytes | None = None,
    relay_pubkey_id: str | None = "local_insecure",
    issued_at_ms: int | None = None,
    expires_at_ms: int | None = None,
) -> Dict[str, Any]:
    """Produce a (potentially insecure) DA challenge for development use."""

    challenge_id = challenge_id or str(uuid.uuid4())
    nonce = nonce or os.urandom(16)
    now_ms = int(time.time() * 1000)
    challenge = {
        "schema": DA_CHALLENGE_SCHEMA,
        "challenge_id": challenge_id,
        "capsule_commit_hash": capsule_commit_hash,
        "verifier_nonce": nonce.hex(),
        "relay_pubkey_id": relay_pubkey_id,
        "issued_at_ms": issued_at_ms or now_ms,
        "expires_at_ms": expires_at_ms or (now_ms + 10 * 60 * 1000),
    }
    if relay_pubkey_id == "local_insecure":
        challenge["relay_signature"] = None
    return challenge


def hash_da_challenge(challenge: Dict[str, Any]) -> str:
    payload = _canonical_bytes(challenge)
    return hashlib.sha256(_HASH_PREFIX_DA_CHALLENGE + payload).hexdigest()


def challenge_signature_payload(challenge: Dict[str, Any]) -> bytes:
    """Return the canonical bytes that the relay signs (without signature field)."""

    payload = dict(challenge)
    payload.pop("relay_signature", None)
    return _canonical_bytes(payload)


def derive_da_seed(capsule_hash: str, challenge: Dict[str, Any]) -> int:
    nonce_hex = challenge.get("verifier_nonce") or ""
    try:
        nonce = bytes.fromhex(nonce_hex)
    except ValueError:
        nonce = nonce_hex.encode("utf-8")
    challenge_id = (challenge.get("challenge_id") or "").encode("utf-8")
    material = [
        b"CAPSULE_DA_SEED_V1::",
        bytes.fromhex(capsule_hash),
        nonce,
        challenge_id,
    ]
    digest = hashlib.sha256(b"".join(material)).digest()
    return int.from_bytes(digest, "big")


# =============================================================================
# Signed DA Challenge v1 - with binding to commit_root and payload_hash
# =============================================================================


def build_signed_da_challenge(
    *,
    commit_root: str,
    payload_hash: str,
    k_samples: int,
    chunk_len: int,
    chunk_tree_arity: int,
    seed: bytes | None = None,
    issuer_key_id: str,
    issued_at_ms: int | None = None,
    expires_at_ms: int | None = None,
) -> Dict[str, Any]:
    """Build a signed DA challenge with proper bindings.

    This is the "real" challenge format that prevents prover-picked challenges.
    The `bind` field ties the challenge to specific capsule commitments.

    Args:
        commit_root: The row commitment root (hex) to bind to
        payload_hash: The capsule payload hash (hex) to bind to
        k_samples: Number of chunks to sample
        chunk_len: Length of each chunk
        chunk_tree_arity: Arity of the chunk merkle tree
        seed: Random seed for sampling (generated if not provided)
        issuer_key_id: Identifier of the challenger key
        issued_at_ms: Issuance timestamp (current time if not provided)
        expires_at_ms: Expiration timestamp (10 min from now if not provided)

    Returns:
        Challenge dict (unsigned - call sign_da_challenge to add signature)
    """
    seed = seed or os.urandom(32)
    now_ms = int(time.time() * 1000)

    challenge = {
        "schema": DA_CHALLENGE_V2_SCHEMA,
        "seed": seed.hex(),
        "k": k_samples,
        "chunk_len": chunk_len,
        "chunk_tree_arity": chunk_tree_arity,
        "bind": {
            "commit_root": commit_root,
            "payload_hash": payload_hash,
        },
        "issuer": {
            "key_id": issuer_key_id,
        },
        "issued_at_ms": issued_at_ms or now_ms,
        "expires_at_ms": expires_at_ms or (now_ms + 10 * 60 * 1000),
    }
    return challenge


def hash_signed_da_challenge(challenge: Dict[str, Any]) -> str:
    """Hash a signed DA challenge (excluding the signature)."""
    payload = dict(challenge)
    issuer = dict(payload.get("issuer", {}))
    issuer.pop("sig", None)
    payload["issuer"] = issuer
    encoded = _canonical_bytes(payload)
    return hashlib.sha256(_HASH_PREFIX_DA_CHALLENGE_V2 + encoded).hexdigest()


def sign_da_challenge(
    challenge: Dict[str, Any],
    private_key_hex: str,
) -> Dict[str, Any]:
    """Sign a DA challenge with the issuer's private key.

    Uses secp256k1 ECDSA (same as Ethereum) via coincurve.

    Args:
        challenge: The unsigned challenge dict
        private_key_hex: The issuer's private key (hex)

    Returns:
        Challenge dict with signature added to issuer.sig
    """
    try:
        from coincurve import PrivateKey
    except ImportError:
        raise RuntimeError("coincurve required for signing - pip install coincurve")

    # Hash the challenge (without signature)
    challenge_hash = hash_signed_da_challenge(challenge)
    digest = bytes.fromhex(challenge_hash)

    # Sign with ECDSA
    private_key = PrivateKey(bytes.fromhex(private_key_hex))
    signature = private_key.sign(digest, hasher=None)

    # Add signature to challenge
    signed = dict(challenge)
    signed["issuer"] = dict(signed.get("issuer", {}))
    signed["issuer"]["sig"] = signature.hex()
    return signed


def verify_da_challenge_signature(
    challenge: Dict[str, Any],
    public_key_hex: str,
) -> bool:
    """Verify a DA challenge signature.

    Args:
        challenge: The signed challenge dict
        public_key_hex: The issuer's public key (hex)

    Returns:
        True if signature is valid
    """
    try:
        from coincurve import PublicKey
    except ImportError:
        raise RuntimeError("coincurve required for verification - pip install coincurve")

    issuer = challenge.get("issuer", {})
    sig_hex = issuer.get("sig")
    if not sig_hex:
        return False

    try:
        # Hash the challenge (without signature)
        challenge_hash = hash_signed_da_challenge(challenge)
        digest = bytes.fromhex(challenge_hash)

        # Verify ECDSA signature
        signature = bytes.fromhex(sig_hex)
        public_key = PublicKey(bytes.fromhex(public_key_hex))
        return public_key.verify(signature, digest, hasher=None)
    except Exception:
        return False


def verify_da_challenge_binding(
    challenge: Dict[str, Any],
    expected_commit_root: str,
    expected_payload_hash: str,
) -> tuple[bool, str]:
    """Verify that a DA challenge binds to the expected commitments.

    Args:
        challenge: The signed DA challenge
        expected_commit_root: Expected row commitment root (hex)
        expected_payload_hash: Expected payload hash (hex)

    Returns:
        (is_valid, error_message)
    """
    bind = challenge.get("bind", {})
    if not bind:
        return False, "challenge missing bind field"

    commit_root = bind.get("commit_root", "")
    payload_hash = bind.get("payload_hash", "")

    if not commit_root or not payload_hash:
        return False, "challenge bind missing commit_root or payload_hash"

    if commit_root.lower() != expected_commit_root.lower():
        return False, f"bind.commit_root mismatch: {commit_root} != {expected_commit_root}"

    if payload_hash.lower() != expected_payload_hash.lower():
        return False, f"bind.payload_hash mismatch: {payload_hash} != {expected_payload_hash}"

    return True, ""


def derive_signed_da_seed(challenge: Dict[str, Any]) -> int:
    """Derive the sampling seed from a signed DA challenge.

    The seed is derived from the challenge's seed field, not from
    the capsule hash (which would allow prover-picking).
    """
    seed_hex = challenge.get("seed", "")
    if not seed_hex:
        return 0
    try:
        seed_bytes = bytes.fromhex(seed_hex)
    except ValueError:
        seed_bytes = seed_hex.encode("utf-8")

    # Mix with challenge params to prevent replay
    k = challenge.get("k", 0)
    chunk_len = challenge.get("chunk_len", 0)
    arity = challenge.get("chunk_tree_arity", 0)
    bind = challenge.get("bind", {})
    commit_root = bind.get("commit_root", "")

    material = [
        b"STC_DA_SEED_V1::",
        seed_bytes,
        k.to_bytes(4, "big"),
        chunk_len.to_bytes(4, "big"),
        arity.to_bytes(2, "big"),
        bytes.fromhex(commit_root) if commit_root else b"",
    ]
    digest = hashlib.sha256(b"".join(material)).digest()
    return int.from_bytes(digest, "big")


__all__ = [
    "DA_CHALLENGE_SCHEMA",
    "DA_CHALLENGE_V2_SCHEMA",
    "build_da_challenge",
    "hash_da_challenge",
    "challenge_signature_payload",
    "derive_da_seed",
    # Signed challenge v1
    "build_signed_da_challenge",
    "hash_signed_da_challenge",
    "sign_da_challenge",
    "verify_da_challenge_signature",
    "verify_da_challenge_binding",
    "derive_signed_da_seed",
]
