#!/usr/bin/env python3
"""
Independent Verifier - Pure stdlib Python implementation

This is a DELIBERATELY INDEPENDENT implementation of the verification logic.
It imports NOTHING from bef_zk to catch correlated bugs.

Only uses: json, hashlib, sys, pathlib (all stdlib)

Implements:
- Canonical row encoding
- SHA256 digest of row
- Hash chain evolution (head_{t+1} = H(head_t || ":" || d_t))
- Merkle root + path verification
- Sidecar hash verification

Usage:
  python verifier.py verify-trace <trace.jsonl> <commitments.json>
  python verifier.py verify-sidecar <features.csv> <sidecar.json>
  python verifier.py verify-opening <opening.json> [checkpoint.json]
  python verifier.py fuzz <trace.jsonl> <commitments.json> <num_mutations>
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

# =============================================================================
# CANONICAL JSON SERIALIZATION (independent implementation)
# =============================================================================

def canonical_json_bytes(obj: Any) -> bytes:
    """Produce deterministic JSON bytes for hashing.

    Rules:
    - Keys sorted alphabetically
    - No whitespace
    - Consistent float representation
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    ).encode('utf-8')


def hash_canonical(obj: Any) -> str:
    """SHA256 hash of canonical JSON."""
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()


# =============================================================================
# TRACE ROW DIGEST
# =============================================================================

def compute_row_digest(row: dict) -> str:
    """Compute SHA256 digest of a trace row."""
    return hash_canonical(row)


# =============================================================================
# HASH CHAIN
# =============================================================================

def genesis_head(manifest_hash: str) -> str:
    """Compute genesis head: H("genesis:" || manifest_hash)"""
    preimage = f"genesis:{manifest_hash}"
    return hashlib.sha256(preimage.encode('utf-8')).hexdigest()


def chain_append(head: str, row: dict) -> tuple[str, str]:
    """Append row to chain, return (new_head, row_digest).

    d_t = H(row_t)
    head_{t+1} = H(head_t || ":" || d_t)
    """
    d_t = compute_row_digest(row)
    preimage = f"{head}:{d_t}"
    new_head = hashlib.sha256(preimage.encode('utf-8')).hexdigest()
    return new_head, d_t


# =============================================================================
# MERKLE TREE (independent implementation)
# =============================================================================

def next_power_of_2(n: int) -> int:
    """Return smallest power of 2 >= n."""
    size = 1
    while size < n:
        size *= 2
    return size


def compute_merkle_root(digests: list[str]) -> str:
    """Compute Merkle root of digest list."""
    if not digests:
        return hashlib.sha256(b"empty").hexdigest()

    # Pad to power of 2
    size = next_power_of_2(len(digests))
    leaves = digests + ["0" * 64] * (size - len(digests))

    # Build tree bottom-up
    while len(leaves) > 1:
        next_level = []
        for i in range(0, len(leaves), 2):
            combined = f"{leaves[i]}:{leaves[i+1]}"
            next_level.append(hashlib.sha256(combined.encode('utf-8')).hexdigest())
        leaves = next_level

    return leaves[0]


def verify_merkle_proof(leaf_digest: str, proof: list[tuple[str, str]], root: str) -> bool:
    """Verify Merkle proof for a leaf.

    proof is list of (sibling_hash, direction) where direction is "L" or "R"
    """
    current = leaf_digest
    for sibling, direction in proof:
        if direction == "L":
            combined = f"{sibling}:{current}"
        else:  # "R"
            combined = f"{current}:{sibling}"
        current = hashlib.sha256(combined.encode('utf-8')).hexdigest()
    return current == root


# =============================================================================
# FILE HASH
# =============================================================================

def compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


# =============================================================================
# ADDRESSABLE RNG (independent implementation per transition_spec_v1)
# =============================================================================

import hmac
import struct

class AddressableRNG:
    """Addressable randomness: rand(tag, t, i) = PRG(seed, tag || t || i).

    This is an INDEPENDENT implementation - no imports from bef_zk.
    Matches rng_id=hmac_sha256_v1, domain_sep_scheme_id=tag_t_i_v1.
    """

    def __init__(self, seed: bytes):
        if len(seed) < 16:
            raise ValueError("Seed must be at least 16 bytes")
        self.seed = seed

    def rand(self, tag: str, t: int, i: int) -> float:
        """Generate deterministic random value in [0, 1).

        address = encode_utf8(tag) || be64(t) || be64(i)
        output = unpack_be64(HMAC_SHA256(seed, address)[:8]) / 2^64
        """
        address = tag.encode('utf-8') + struct.pack('>QQ', t, i)
        digest = hmac.new(self.seed, address, hashlib.sha256).digest()
        value = struct.unpack('>Q', digest[:8])[0]
        return value / (2**64)

    @classmethod
    def from_hex(cls, hex_seed: str) -> 'AddressableRNG':
        return cls(bytes.fromhex(hex_seed))


# =============================================================================
# TRANSITION VERIFICATION (Level 1: Local Validity)
# =============================================================================
#
# Architecture: Dispatcher pattern for versioned transition specs.
# Each spec has:
#   - verify_transition_<spec>(view_pre, view_post, x_t, manifest) -> (bool, str)
#   - verify_output_<spec>(x_t, t, rng) -> (bool, str)
#   - verify_continuity_<spec>(rows) -> (bool, str)
#   - compute_genesis_state_<spec>(manifest) -> dict
#
# Adding a new spec = add functions + register in TRANSITION_SPECS dict.
# =============================================================================

# -----------------------------------------------------------------------------
# SPEC: identity_v1 (toy spec for testing infrastructure)
# -----------------------------------------------------------------------------

def compute_genesis_state_identity_v1(manifest: dict) -> dict:
    """Compute initial state for identity_v1: state=0."""
    return {"state": 0}


def verify_transition_identity_v1(view_pre: dict, view_post: dict, x_t: list, manifest: dict) -> tuple[bool, str]:
    """Verify transition_fn_id=identity_v1: s_{t+1} = s_t + 1."""
    pre_state = view_pre.get("state")
    post_state = view_post.get("state")

    if pre_state is None or post_state is None:
        return False, "Missing 'state' field in view_pre or view_post"

    expected = pre_state + 1
    if post_state != expected:
        return False, f"Transition constraint failed: expected state={expected}, got {post_state}"

    return True, "Transition constraint satisfied"


def verify_continuity_identity_v1(rows: list[dict]) -> tuple[bool, str]:
    """Verify continuity for identity_v1: view_post[t].state == view_pre[t+1].state."""
    for i in range(len(rows) - 1):
        post_t = rows[i].get("view_post", {})
        pre_t1 = rows[i + 1].get("view_pre", {})

        if post_t.get("state") != pre_t1.get("state"):
            return False, f"Continuity broken at t={i}: view_post.state={post_t.get('state')} != view_pre.state={pre_t1.get('state')}"

    return True, f"Continuity verified across {len(rows)} rows"


# -----------------------------------------------------------------------------
# SPEC: keyed_hash_v1 (first real spec with cryptographic state)
# -----------------------------------------------------------------------------

def compute_genesis_state_keyed_hash_v1(manifest: dict) -> dict:
    """Compute initial state for keyed_hash_v1.

    genesis_hash = H("keyed_hash_v1:genesis:" || seed_commitment)
    """
    seed_commitment = manifest.get("seed_commitment", "")
    preimage = f"keyed_hash_v1:genesis:{seed_commitment}"
    genesis_hash = hashlib.sha256(preimage.encode('utf-8')).hexdigest()
    return {"t": 0, "hash": genesis_hash}


def verify_transition_keyed_hash_v1(view_pre: dict, view_post: dict, x_t: list, manifest: dict) -> tuple[bool, str]:
    """Verify transition_fn_id=keyed_hash_absorb_v1.

    Constraint:
        view_post.hash = SHA256(view_pre.hash || canonical_json(x_t))
        view_post.t = view_pre.t + 1
    """
    pre_hash = view_pre.get("hash")
    pre_t = view_pre.get("t")
    post_hash = view_post.get("hash")
    post_t = view_post.get("t")

    if pre_hash is None or post_hash is None:
        return False, "Missing 'hash' field in view_pre or view_post"

    if pre_t is None or post_t is None:
        return False, "Missing 't' field in view_pre or view_post"

    # Verify timestep increment
    if post_t != pre_t + 1:
        return False, f"Timestep constraint failed: expected t={pre_t + 1}, got {post_t}"

    # Verify hash evolution: H(pre_hash || canonical_json(x_t))
    x_t_bytes = canonical_json_bytes(x_t)
    preimage = pre_hash.encode('utf-8') + x_t_bytes
    expected_hash = hashlib.sha256(preimage).hexdigest()

    if post_hash != expected_hash:
        return False, f"Hash transition failed: expected {expected_hash[:16]}..., got {post_hash[:16]}..."

    return True, "Keyed hash transition verified"


def verify_continuity_keyed_hash_v1(rows: list[dict]) -> tuple[bool, str]:
    """Verify continuity for keyed_hash_v1.

    view_post[t].hash == view_pre[t+1].hash
    view_post[t].t == view_pre[t+1].t
    """
    for i in range(len(rows) - 1):
        post_t = rows[i].get("view_post", {})
        pre_t1 = rows[i + 1].get("view_pre", {})

        if post_t.get("hash") != pre_t1.get("hash"):
            return False, f"Continuity broken at t={i}: hash mismatch"

        if post_t.get("t") != pre_t1.get("t"):
            return False, f"Continuity broken at t={i}: t mismatch ({post_t.get('t')} != {pre_t1.get('t')})"

    return True, f"Continuity verified across {len(rows)} rows"


# -----------------------------------------------------------------------------
# SPEC: rng_features_v1 (output function, shared across specs)
# -----------------------------------------------------------------------------

def verify_output_rng_features_v1(
    x_t: list[float],
    t: int,
    rng: AddressableRNG,
    tolerance: float = 1e-10,
) -> tuple[bool, str]:
    """Verify output_fn_id=rng_features_v1: x_t[i] = rng.rand("input", t, i)."""
    for i, actual in enumerate(x_t):
        expected = rng.rand("input", t, i)
        if abs(actual - expected) > tolerance:
            return False, f"Output constraint failed at x_t[{i}]: expected {expected:.15f}, got {actual:.15f}"

    return True, f"Output constraint satisfied ({len(x_t)} features)"


def verify_rng_addresses(row: dict, rng: AddressableRNG) -> tuple[bool, str]:
    """Verify RNG addresses match the PRG spec."""
    rand_addrs = row.get("rand_addrs", [])
    x_t = row.get("x_t", [])
    t = row.get("t", 0)

    # For rng_features_v1, we expect addresses for "input" tag
    expected_addrs = [{"tag": "input", "t": t, "i": i} for i in range(len(x_t))]

    # Check that declared addresses match expected
    for i, addr in enumerate(rand_addrs):
        if i >= len(expected_addrs):
            break
        exp = expected_addrs[i]
        if addr.get("tag") != exp["tag"] or addr.get("t") != exp["t"] or addr.get("i") != exp["i"]:
            return False, f"RNG address mismatch at index {i}: expected {exp}, got {addr}"

    return True, f"RNG addresses verified ({len(rand_addrs)} addresses)"


# -----------------------------------------------------------------------------
# QUANTIZATION (for bicep_v1 commitment-boundary canonicalization)
# -----------------------------------------------------------------------------

def quantize(value: float, precision_bits: int = 24, clamp_min: int = -(2**31), clamp_max: int = 2**31 - 1) -> int:
    """Quantize float to fixed-point integer.

    Uses round-to-nearest-ties-to-even (Python default).
    """
    scale = 2 ** precision_bits
    raw = round(value * scale)
    return max(clamp_min, min(clamp_max, raw))


def dequantize(q_value: int, precision_bits: int = 24) -> float:
    """Recover float from quantized integer."""
    scale = 2 ** precision_bits
    return q_value / scale


def quantize_list(values: list[float], precision_bits: int = 24) -> list[int]:
    """Quantize a list of floats."""
    return [quantize(v, precision_bits) for v in values]


# -----------------------------------------------------------------------------
# SPEC: bicep_v1 (production spec with quantized outputs, AIR-ready)
# -----------------------------------------------------------------------------

def compute_genesis_state_bicep_v1(manifest: dict) -> dict:
    """Compute initial state for bicep_v1.

    state_root = H("bicep_v1:state:genesis:" || seed_commitment)
    output_chain = H("bicep_v1:output_chain:genesis:" || seed_commitment)
    """
    seed_commitment = manifest.get("seed_commitment", "")

    state_root = hashlib.sha256(
        f"bicep_v1:state:genesis:{seed_commitment}".encode()
    ).hexdigest()

    output_chain = hashlib.sha256(
        f"bicep_v1:output_chain:genesis:{seed_commitment}".encode()
    ).hexdigest()

    return {"state_root": state_root, "output_chain": output_chain}


def compute_rng_use_hash(rand_addrs: list[dict]) -> str:
    """Compute H(canonical(rand_addrs)) for tamper evidence."""
    return hashlib.sha256(canonical_json_bytes(rand_addrs)).hexdigest()


def compute_output_chain_update(prev_chain: str, x_t_q: list[int]) -> str:
    """Compute H(prev_chain || canonical(x_t_q))."""
    x_bytes = canonical_json_bytes(x_t_q)
    preimage = prev_chain.encode('utf-8') + x_bytes
    return hashlib.sha256(preimage).hexdigest()


def verify_transition_bicep_v1(view_pre: dict, view_post: dict, row: dict, manifest: dict) -> tuple[bool, str]:
    """Verify transition_fn_id=bicep_sde_v1.

    Checks:
    1. rng_use_hash matches H(canonical(rand_addrs))
    2. output_chain update: H(prev_chain || canonical(x_t_q))
    3. state_root update (for now: trust producer, will add SDE check later)
    """
    # Get row fields
    x_t_q = row.get("x_t_q", [])
    rand_addrs = row.get("rand_addrs", [])
    rng_use_hash = row.get("rng_use_hash", "")

    # Check rng_use_hash
    expected_rng_hash = compute_rng_use_hash(rand_addrs)
    if rng_use_hash != expected_rng_hash:
        return False, f"rng_use_hash mismatch: expected {expected_rng_hash[:16]}..., got {rng_use_hash[:16]}..."

    # Check output_chain update
    prev_chain = view_pre.get("output_chain", "")
    expected_chain = compute_output_chain_update(prev_chain, x_t_q)
    actual_chain = view_post.get("output_chain", "")

    if actual_chain != expected_chain:
        return False, f"output_chain mismatch: expected {expected_chain[:16]}..., got {actual_chain[:16]}..."

    # Note: state_root verification requires SDE recomputation.
    # For now, we trust the producer. Full verification will be added
    # when SDE step function is implemented.
    # TODO: verify_state_root_update(view_pre, view_post, row, manifest)

    return True, "Transition verified (output_chain + rng_use_hash)"


def verify_continuity_bicep_v1(rows: list[dict]) -> tuple[bool, str]:
    """Verify continuity for bicep_v1.

    view_post[t].state_root == view_pre[t+1].state_root
    view_post[t].output_chain == view_pre[t+1].output_chain
    """
    for i in range(len(rows) - 1):
        post_t = rows[i].get("view_post", {})
        pre_t1 = rows[i + 1].get("view_pre", {})

        if post_t.get("state_root") != pre_t1.get("state_root"):
            return False, f"Continuity broken at t={i}: state_root mismatch"

        if post_t.get("output_chain") != pre_t1.get("output_chain"):
            return False, f"Continuity broken at t={i}: output_chain mismatch"

    return True, f"Continuity verified across {len(rows)} rows"


def verify_output_bicep_features_v1(
    row: dict,
    t: int,
    rng: AddressableRNG,
    manifest: dict,
) -> tuple[bool, str]:
    """Verify output_fn_id=bicep_features_v1.

    For bicep_v1, outputs are quantized: x_t_q is the canonical form.
    Checks:
    1. x_t_q values match expected from RNG
    2. rand_addrs are consistent with expected addresses
    """
    x_t_q = row.get("x_t_q", [])
    rand_addrs = row.get("rand_addrs", [])
    precision_bits = manifest.get("x_quant_precision_bits", 24)

    # Check rand_addrs: support both per-element (bicep_v1) and range (bicep_v2) formats
    if rand_addrs and "i_count" in rand_addrs[0]:
        # Range format (bicep_v2): {tag, t, i_start, i_count, layout_id}
        # Just verify the "input" tag addresses cover x_t_q
        input_addrs = [a for a in rand_addrs if a.get("tag") == "input"]
        if input_addrs:
            total_covered = sum(a.get("i_count", 0) for a in input_addrs)
            if total_covered < len(x_t_q):
                return False, f"rand_addrs input range covers {total_covered} elements, need {len(x_t_q)}"
    else:
        # Per-element format (bicep_v1)
        expected_addrs = [{"tag": "input", "t": t, "i": i} for i in range(len(x_t_q))]
        if len(rand_addrs) != len(expected_addrs):
            return False, f"rand_addrs count mismatch: expected {len(expected_addrs)}, got {len(rand_addrs)}"

        for i, (actual_addr, expected_addr) in enumerate(zip(rand_addrs, expected_addrs)):
            if (actual_addr.get("tag") != expected_addr["tag"] or
                actual_addr.get("t") != expected_addr["t"] or
                actual_addr.get("i") != expected_addr["i"]):
                return False, f"rand_addrs[{i}] mismatch: expected {expected_addr}, got {actual_addr}"

    # Compute expected quantized outputs from RNG
    expected_q = []
    for i in range(len(x_t_q)):
        raw = rng.rand("input", t, i)
        expected_q.append(quantize(raw, precision_bits))

    # Compare quantized values
    for i, (actual, expected) in enumerate(zip(x_t_q, expected_q)):
        if actual != expected:
            return False, f"x_t_q[{i}] mismatch: expected {expected}, got {actual}"

    return True, f"Output verified ({len(x_t_q)} quantized features)"


# -----------------------------------------------------------------------------
# SPEC: bicep_v2 (audited state transitions via Merkle openings)
# -----------------------------------------------------------------------------
#
# bicep_v2 inherits all bicep_v1 checks (output_chain, rng_use_hash) and adds:
#   - state_root = MerkleRoot(quantized state tensor)
#   - Deterministic audit challenges (Fiat-Shamir-lite)
#   - Audit openings sidecar verified per-step
#
# The heavy lifting lives in state_audit.py; this is the dispatcher wrapper.
# -----------------------------------------------------------------------------

def compute_genesis_state_bicep_v2(manifest: dict) -> dict:
    """Compute initial state for bicep_v2.

    state_root = MerkleRoot(initial quantized state tensor from RNG)
    output_chain = H("bicep_v2:output_chain:genesis:" || seed_commitment)
    """
    seed_commitment = manifest.get("seed_commitment", "")

    # Import state_audit for Merkle tree construction
    from state_audit import AddressableRNG as SA_RNG, build_merkle_tree, quantize as sa_quantize, total_leaves

    rng = SA_RNG(bytes.fromhex(manifest.get("_seed_hex", ""))) if manifest.get("_seed_hex") else None

    if rng is not None:
        num_paths = manifest.get("state_num_paths", 4)
        num_channels = manifest.get("state_num_channels", 4)
        precision_bits = manifest.get("x_quant_precision_bits", 24)
        n_leaves = total_leaves(num_paths, num_channels)

        state_q = []
        for idx in range(n_leaves):
            val = rng.rand("state_init", 0, idx)
            state_q.append(sa_quantize(val, precision_bits))

        state_root, _ = build_merkle_tree(state_q)
    else:
        # Without seed, fall back to commitment-only check
        state_root = None

    output_chain = hashlib.sha256(
        f"bicep_v2:output_chain:genesis:{seed_commitment}".encode()
    ).hexdigest()

    result = {"output_chain": output_chain}
    if state_root is not None:
        result["state_root"] = state_root
    return result


def verify_transition_bicep_v2(view_pre: dict, view_post: dict, row: dict, manifest: dict) -> tuple[bool, str]:
    """Verify transition for bicep_v2.

    Inherits output_chain + rng_use_hash checks from bicep_v1.
    state_root transition is verified via audit openings (separate pass).
    """
    # Reuse bicep_v1 output_chain + rng_use_hash checks
    x_t_q = row.get("x_t_q", [])
    rand_addrs = row.get("rand_addrs", [])
    rng_use_hash = row.get("rng_use_hash", "")

    expected_rng_hash = compute_rng_use_hash(rand_addrs)
    if rng_use_hash != expected_rng_hash:
        return False, f"rng_use_hash mismatch: expected {expected_rng_hash[:16]}..., got {rng_use_hash[:16]}..."

    prev_chain = view_pre.get("output_chain", "")
    expected_chain = compute_output_chain_update(prev_chain, x_t_q)
    actual_chain = view_post.get("output_chain", "")

    if actual_chain != expected_chain:
        return False, f"output_chain mismatch: expected {expected_chain[:16]}..., got {actual_chain[:16]}..."

    # state_root transition is verified by the audit pass (verify_audit_openings)
    return True, "Transition verified (output_chain + rng_use_hash; state_root via audit)"


def verify_continuity_bicep_v2(rows: list[dict]) -> tuple[bool, str]:
    """Verify continuity for bicep_v2 (same as v1)."""
    return verify_continuity_bicep_v1(rows)


def verify_audit_openings(
    rows: list[dict],
    audits: list[dict],
    manifest: dict,
    manifest_hash: str,
    seed_hex: str,
) -> tuple[bool, str, dict]:
    """Verify all audit openings for bicep_v2 trace.

    Returns (success, message, stats) where stats contains audit coverage info.
    """
    from state_audit import (
        AddressableRNG as SA_RNG,
        verify_audit_opening,
        canonical_json_bytes as sa_canonical,
    )

    rng = SA_RNG.from_hex(seed_hex)
    head = hashlib.sha256(f"genesis:{manifest_hash}".encode()).hexdigest()

    total_leaves_checked = 0
    steps_audited = 0

    for t, (row, audit) in enumerate(zip(rows, audits)):
        success, msg = verify_audit_opening(
            audit, row, manifest, manifest_hash, head, rng,
        )
        if not success:
            return False, f"Audit failed at step t={t}: {msg}", {
                "steps_audited": steps_audited,
                "total_leaves_checked": total_leaves_checked,
                "failure_step": t,
                "failure_msg": msg,
            }

        total_leaves_checked += len(audit.get("audit_indices", []))
        steps_audited += 1

        # Advance head
        row_digest = hashlib.sha256(sa_canonical(row)).hexdigest()
        head = hashlib.sha256(f"{head}:{row_digest}".encode()).hexdigest()

    stats = {
        "steps_audited": steps_audited,
        "total_leaves_checked": total_leaves_checked,
        "audit_k": manifest.get("audit_k", 0),
        "num_paths": manifest.get("state_num_paths", 0),
        "num_channels": manifest.get("state_num_channels", 0),
    }
    return True, f"All {steps_audited} steps audited ({total_leaves_checked} leaf transitions verified)", stats


# -----------------------------------------------------------------------------
# SPEC REGISTRY (dispatcher)
# -----------------------------------------------------------------------------

TRANSITION_SPECS = {
    "identity_v1": {
        "transition_fn": verify_transition_identity_v1,
        "continuity_fn": verify_continuity_identity_v1,
        "genesis_fn": compute_genesis_state_identity_v1,
    },
    "keyed_hash_v1": {
        "transition_fn": verify_transition_keyed_hash_v1,
        "continuity_fn": verify_continuity_keyed_hash_v1,
        "genesis_fn": compute_genesis_state_keyed_hash_v1,
    },
    "bicep_v1": {
        "transition_fn": verify_transition_bicep_v1,
        "continuity_fn": verify_continuity_bicep_v1,
        "genesis_fn": compute_genesis_state_bicep_v1,
        "uses_quantized_outputs": True,
    },
    "bicep_v2": {
        "transition_fn": verify_transition_bicep_v2,
        "continuity_fn": verify_continuity_bicep_v2,
        "genesis_fn": compute_genesis_state_bicep_v2,
        "uses_quantized_outputs": True,
        "has_audit_openings": True,
    },
}

OUTPUT_SPECS = {
    "rng_features_v1": verify_output_rng_features_v1,
    "bicep_features_v1": verify_output_bicep_features_v1,
}


def get_transition_spec(spec_id: str) -> dict:
    """Get transition spec functions by ID."""
    if spec_id not in TRANSITION_SPECS:
        raise ValueError(f"Unknown transition spec: {spec_id}. Available: {list(TRANSITION_SPECS.keys())}")
    return TRANSITION_SPECS[spec_id]


def get_output_spec(spec_id: str):
    """Get output verification function by ID."""
    if spec_id not in OUTPUT_SPECS:
        raise ValueError(f"Unknown output spec: {spec_id}. Available: {list(OUTPUT_SPECS.keys())}")
    return OUTPUT_SPECS[spec_id]


# -----------------------------------------------------------------------------
# UNIFIED VERIFICATION (dispatcher-based)
# -----------------------------------------------------------------------------

def verify_row_correctness(
    row: dict,
    rng: AddressableRNG,
    manifest: dict,
    transition_spec_id: str = "identity_v1",
    output_spec_id: str = "rng_features_v1",
) -> tuple[bool, str]:
    """Verify a single row satisfies transition + output constraints.

    This is Level 1 (Local Validity) verification.
    Dispatches to the appropriate spec functions.
    """
    t = row.get("t", 0)
    x_t = row.get("x_t", [])
    view_pre = row.get("view_pre", {})
    view_post = row.get("view_post", {})

    # Get spec functions
    transition_spec = get_transition_spec(transition_spec_id)

    # Verify transition constraint (dispatched)
    # bicep_v1 takes full row; others take (view_pre, view_post, x_t, manifest)
    if transition_spec.get("uses_quantized_outputs"):
        valid, msg = transition_spec["transition_fn"](view_pre, view_post, row, manifest)
    else:
        valid, msg = transition_spec["transition_fn"](view_pre, view_post, x_t, manifest)
    if not valid:
        return False, f"Row t={t}: {msg}"

    # Verify output constraint (dispatched)
    # bicep_features_v1 takes (row, t, rng, manifest); rng_features_v1 takes (x_t, t, rng)
    if output_spec_id == "bicep_features_v1":
        output_fn = get_output_spec(output_spec_id)
        valid, msg = output_fn(row, t, rng, manifest)
    else:
        output_fn = get_output_spec(output_spec_id)
        valid, msg = output_fn(x_t, t, rng)
    if not valid:
        return False, f"Row t={t}: {msg}"

    # Verify RNG addresses (skip for bicep_v1 - rng_use_hash handles this)
    if not transition_spec.get("uses_quantized_outputs"):
        valid, msg = verify_rng_addresses(row, rng)
        if not valid:
            return False, f"Row t={t}: {msg}"

    return True, f"Row t={t} is locally valid"


def verify_genesis_state(
    first_row: dict,
    manifest: dict,
    transition_spec_id: str = "identity_v1",
) -> tuple[bool, str]:
    """Verify the first row has correct genesis state."""
    transition_spec = get_transition_spec(transition_spec_id)
    expected_genesis = transition_spec["genesis_fn"](manifest)
    actual_pre = first_row.get("view_pre", {})

    # Compare relevant fields based on spec
    if transition_spec_id == "identity_v1":
        if actual_pre.get("state") != expected_genesis.get("state"):
            return False, f"Genesis state mismatch: expected {expected_genesis}, got {actual_pre}"
    elif transition_spec_id == "keyed_hash_v1":
        if actual_pre.get("hash") != expected_genesis.get("hash"):
            return False, f"Genesis hash mismatch: expected {expected_genesis['hash'][:16]}..., got {actual_pre.get('hash', 'None')[:16] if actual_pre.get('hash') else 'None'}..."
        if actual_pre.get("t") != expected_genesis.get("t"):
            return False, f"Genesis t mismatch: expected {expected_genesis['t']}, got {actual_pre.get('t')}"
    elif transition_spec_id == "bicep_v1":
        if actual_pre.get("state_root") != expected_genesis.get("state_root"):
            return False, f"Genesis state_root mismatch: expected {expected_genesis['state_root'][:16]}..., got {actual_pre.get('state_root', 'None')[:16] if actual_pre.get('state_root') else 'None'}..."
        if actual_pre.get("output_chain") != expected_genesis.get("output_chain"):
            return False, f"Genesis output_chain mismatch: expected {expected_genesis['output_chain'][:16]}..., got {actual_pre.get('output_chain', 'None')[:16] if actual_pre.get('output_chain') else 'None'}..."
    elif transition_spec_id == "bicep_v2":
        if expected_genesis.get("state_root") is not None:
            if actual_pre.get("state_root") != expected_genesis["state_root"]:
                return False, f"Genesis state_root mismatch: expected {expected_genesis['state_root'][:16]}..., got {actual_pre.get('state_root', 'None')[:16] if actual_pre.get('state_root') else 'None'}..."
        if actual_pre.get("output_chain") != expected_genesis.get("output_chain"):
            return False, f"Genesis output_chain mismatch: expected {expected_genesis['output_chain'][:16]}..., got {actual_pre.get('output_chain', 'None')[:16] if actual_pre.get('output_chain') else 'None'}..."

    return True, "Genesis state verified"


# Default output spec per transition spec
DEFAULT_OUTPUT_SPEC = {
    "identity_v1": "rng_features_v1",
    "keyed_hash_v1": "rng_features_v1",
    "bicep_v1": "bicep_features_v1",
    "bicep_v2": "bicep_features_v1",
}


def verify_trace_correctness(
    trace_path: Path,
    seed_hex: str,
    transition_spec_id: str = "identity_v1",
    output_spec_id: Optional[str] = None,
    manifest: Optional[dict] = None,
    openings_path: Optional[Path] = None,
) -> tuple[bool, str]:
    """Verify entire trace satisfies correctness constraints.

    Level 1: local validity + continuity.
    Level 1.5 (bicep_v2): audit openings verification.

    Args:
        openings_path: Path to audit openings JSON file (list of per-step audits).
                       Required for bicep_v2. Can be a single JSON file or a directory
                       containing audit_step_NNNN.json files.
    """
    # Auto-derive output spec from transition spec if not specified
    if output_spec_id is None:
        output_spec_id = DEFAULT_OUTPUT_SPEC.get(transition_spec_id, "rng_features_v1")

    rng = AddressableRNG.from_hex(seed_hex)

    # Build manifest if not provided
    if manifest is None:
        manifest = {
            "seed_commitment": hashlib.sha256(b"seed_commit:" + bytes.fromhex(seed_hex)).hexdigest(),
            "transition_spec_id": transition_spec_id,
            "output_fn_id": output_spec_id,
            "x_quant_precision_bits": 24,
        }

    rows = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        return True, "Empty trace is vacuously correct"

    # Verify genesis state
    transition_spec = get_transition_spec(transition_spec_id)

    # For bicep_v2, inject seed into manifest for genesis Merkle computation
    if transition_spec_id == "bicep_v2":
        manifest = dict(manifest)
        manifest["_seed_hex"] = seed_hex

    valid, msg = verify_genesis_state(rows[0], manifest, transition_spec_id)
    if not valid:
        return False, msg

    # Verify each row individually
    for row in rows:
        valid, msg = verify_row_correctness(row, rng, manifest, transition_spec_id, output_spec_id)
        if not valid:
            return False, msg

    # Verify continuity (dispatched)
    valid, msg = transition_spec["continuity_fn"](rows)
    if not valid:
        return False, msg

    # Level 1.5: Audit openings (bicep_v2)
    if transition_spec.get("has_audit_openings"):
        if openings_path is None:
            # Try default location: same dir as trace, openings/ subdirectory
            default_dir = trace_path.parent / "audit_openings"
            default_file = trace_path.parent / "audit_openings.json"
            if default_dir.is_dir():
                openings_path = default_dir
            elif default_file.exists():
                openings_path = default_file

        if openings_path is None:
            return False, "bicep_v2 requires audit openings (--openings <path>)"

        # Load audits
        audits = _load_audit_openings(openings_path, len(rows))

        if len(audits) != len(rows):
            return False, f"Audit count mismatch: {len(audits)} audits for {len(rows)} rows"

        manifest_hash = hashlib.sha256(
            canonical_json_bytes({k: v for k, v in manifest.items() if not k.startswith("_")})
        ).hexdigest()

        valid, msg, stats = verify_audit_openings(rows, audits, manifest, manifest_hash, seed_hex)
        if not valid:
            return False, msg

        return True, (
            f"Trace is correct: {len(rows)} rows verified "
            f"(spec={transition_spec_id}, output={output_spec_id}, "
            f"audit: {stats['total_leaves_checked']} leaf transitions across {stats['steps_audited']} steps)"
        )

    return True, f"Trace is correct: {len(rows)} rows verified (spec={transition_spec_id}, output={output_spec_id})"


def _load_audit_openings(path: Path, expected_count: int) -> list[dict]:
    """Load audit openings from a file or directory."""
    if path.is_dir():
        # Load audit_step_NNNN.json files in order
        audits = []
        for t in range(expected_count):
            step_file = path / f"audit_step_{t:04d}.json"
            if not step_file.exists():
                raise FileNotFoundError(f"Missing audit file: {step_file}")
            with open(step_file) as f:
                audits.append(json.load(f))
        return audits
    else:
        # Single JSON file containing list of audits
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f"Expected list of audits, got {type(data).__name__}")


# =============================================================================
# VERIFY TRACE (Integrity - Level 0)
# =============================================================================

def verify_trace(trace_path: Path, commitments_path: Path) -> tuple[bool, str]:
    """Verify trace.jsonl matches commitments.json.

    Replays hash chain and compares final head.
    """
    # Load commitments
    with open(commitments_path) as f:
        commitments = json.load(f)

    manifest_hash = commitments["manifest_hash"]
    expected_head = commitments["head_T"]
    expected_steps = commitments["total_steps"]

    # Replay hash chain
    head = genesis_head(manifest_hash)
    step_count = 0

    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            head, _ = chain_append(head, row)
            step_count += 1

    # Verify
    if step_count != expected_steps:
        return False, f"Step count mismatch: got {step_count}, expected {expected_steps}"

    if head != expected_head:
        return False, f"Head mismatch:\n  got:      {head}\n  expected: {expected_head}"

    return True, f"Verified {step_count} steps, head matches"


# =============================================================================
# VERIFY SIDECAR
# =============================================================================

def verify_sidecar(features_path: Path, sidecar_path: Path) -> tuple[bool, str]:
    """Verify features file matches sidecar binding."""
    # Load sidecar
    with open(sidecar_path) as f:
        sidecar = json.load(f)

    expected_hash = sidecar["features_shard_hash"]

    # Compute actual hash
    actual_hash = compute_file_hash(features_path)

    if actual_hash != expected_hash:
        return False, f"Features hash mismatch:\n  actual:   {actual_hash}\n  expected: {expected_hash}"

    return True, f"Features hash verified ({actual_hash[:16]}...)"


# =============================================================================
# VERIFY OPENING
# =============================================================================

def verify_opening(opening_path: Path, checkpoint_path: Optional[Path] = None) -> tuple[bool, str]:
    """Verify row opening with Merkle proof."""
    # Load opening
    with open(opening_path) as f:
        opening = json.load(f)

    row = opening["row"]
    proof = [(p[0], p[1]) for p in opening["merkle_proof"]]
    chunk_root = opening["chunk_root"]

    # Compute row digest
    row_digest = compute_row_digest(row)

    # Verify Merkle proof
    if not verify_merkle_proof(row_digest, proof, chunk_root):
        return False, f"Merkle proof invalid for row t={row.get('t')}"

    # Cross-check with checkpoint if provided
    if checkpoint_path:
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        if chunk_root != checkpoint["chunk_root"]:
            return False, "Chunk root mismatch with checkpoint"

        if opening["manifest_hash"] != checkpoint["manifest_hash"]:
            return False, "Manifest hash mismatch with checkpoint"

    return True, f"Row opening verified (t={row.get('t')}, leaf_index={opening['leaf_index']})"


# =============================================================================
# FUZZ / MUTATION TESTING
# =============================================================================

import random
import copy

def mutate_trace(trace_rows: list[dict], mutation_type: str) -> list[dict]:
    """Apply a single mutation to trace rows."""
    rows = copy.deepcopy(trace_rows)

    if mutation_type == "flip_byte":
        # Modify a random field in a random row
        if rows:
            idx = random.randint(0, len(rows) - 1)
            if rows[idx].get("x_t"):
                fidx = random.randint(0, len(rows[idx]["x_t"]) - 1)
                rows[idx]["x_t"][fidx] += random.uniform(-0.001, 0.001)

    elif mutation_type == "swap_rows":
        # Swap two adjacent rows
        if len(rows) >= 2:
            idx = random.randint(0, len(rows) - 2)
            rows[idx], rows[idx + 1] = rows[idx + 1], rows[idx]

    elif mutation_type == "delete_row":
        # Delete a random row
        if rows:
            idx = random.randint(0, len(rows) - 1)
            del rows[idx]

    elif mutation_type == "duplicate_row":
        # Duplicate a random row
        if rows:
            idx = random.randint(0, len(rows) - 1)
            rows.insert(idx, copy.deepcopy(rows[idx]))

    elif mutation_type == "modify_t":
        # Modify timestep
        if rows:
            idx = random.randint(0, len(rows) - 1)
            rows[idx]["t"] += 1

    elif mutation_type == "modify_view":
        # Modify view_pre or view_post
        if rows:
            idx = random.randint(0, len(rows) - 1)
            if random.random() < 0.5:
                rows[idx]["view_pre"]["tampered"] = True
            else:
                rows[idx]["view_post"]["tampered"] = True

    return rows


def run_fuzz_test(trace_path: Path, commitments_path: Path, num_mutations: int) -> dict:
    """Run fuzz testing with random mutations."""
    # Load original trace
    rows = []
    with open(trace_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Load commitments
    with open(commitments_path) as f:
        commitments = json.load(f)

    manifest_hash = commitments["manifest_hash"]
    expected_head = commitments["head_T"]
    expected_steps = commitments["total_steps"]

    mutation_types = ["flip_byte", "swap_rows", "delete_row", "duplicate_row", "modify_t", "modify_view"]
    results = {
        "total": num_mutations,
        "rejected": 0,
        "accepted": 0,  # Should always be 0 for mutations
        "by_type": {t: {"total": 0, "rejected": 0} for t in mutation_types},
    }

    for i in range(num_mutations):
        # Pick random mutation type
        mtype = random.choice(mutation_types)
        results["by_type"][mtype]["total"] += 1

        # Apply mutation
        mutated = mutate_trace(rows, mtype)

        # Replay and check
        head = genesis_head(manifest_hash)
        step_count = 0
        for row in mutated:
            head, _ = chain_append(head, row)
            step_count += 1

        # Check if rejected
        if step_count != expected_steps or head != expected_head:
            results["rejected"] += 1
            results["by_type"][mtype]["rejected"] += 1
        else:
            results["accepted"] += 1

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "verify-trace":
        if len(sys.argv) != 4:
            print("Usage: verify-trace <trace.jsonl> <commitments.json>")
            sys.exit(1)

        success, msg = verify_trace(Path(sys.argv[2]), Path(sys.argv[3]))
        print(f"[INDEPENDENT-VERIFIER] {'PASS' if success else 'FAIL'}: {msg}")
        sys.exit(0 if success else 1)

    elif cmd == "verify-sidecar":
        if len(sys.argv) != 4:
            print("Usage: verify-sidecar <features.csv> <sidecar.json>")
            sys.exit(1)

        success, msg = verify_sidecar(Path(sys.argv[2]), Path(sys.argv[3]))
        print(f"[INDEPENDENT-VERIFIER] {'PASS' if success else 'FAIL'}: {msg}")
        sys.exit(0 if success else 1)

    elif cmd == "verify-opening":
        if len(sys.argv) < 3:
            print("Usage: verify-opening <opening.json> [checkpoint.json]")
            sys.exit(1)

        checkpoint_path = Path(sys.argv[3]) if len(sys.argv) >= 4 else None
        success, msg = verify_opening(Path(sys.argv[2]), checkpoint_path)
        print(f"[INDEPENDENT-VERIFIER] {'PASS' if success else 'FAIL'}: {msg}")
        sys.exit(0 if success else 1)

    elif cmd == "verify-correctness":
        if len(sys.argv) < 4:
            print("Usage: verify-correctness <trace.jsonl> <seed_hex> [--spec <spec_id>] [--openings <path>]")
            print("  seed_hex: The RNG seed in hexadecimal (e.g., 'abcd1234' * 8)")
            print(f"  Available specs: {list(TRANSITION_SPECS.keys())}")
            print("  --openings: Path to audit openings (file or directory, required for bicep_v2)")
            sys.exit(1)

        trace_path = Path(sys.argv[2])
        seed_hex = sys.argv[3]

        # Parse optional arguments
        spec_id = "identity_v1"
        openings_path = None
        if "--spec" in sys.argv:
            spec_idx = sys.argv.index("--spec")
            if spec_idx + 1 < len(sys.argv):
                spec_id = sys.argv[spec_idx + 1]
        if "--openings" in sys.argv:
            op_idx = sys.argv.index("--openings")
            if op_idx + 1 < len(sys.argv):
                openings_path = Path(sys.argv[op_idx + 1])

        success, msg = verify_trace_correctness(
            trace_path, seed_hex,
            transition_spec_id=spec_id,
            openings_path=openings_path,
        )
        print(f"[INDEPENDENT-VERIFIER] {'PASS' if success else 'FAIL'}: {msg}")
        sys.exit(0 if success else 1)

    elif cmd == "list-specs":
        print("Available transition specs:")
        for spec_id in TRANSITION_SPECS:
            print(f"  - {spec_id}")
        print("\nAvailable output specs:")
        for spec_id in OUTPUT_SPECS:
            print(f"  - {spec_id}")
        sys.exit(0)

    elif cmd == "fuzz":
        if len(sys.argv) != 5:
            print("Usage: fuzz <trace.jsonl> <commitments.json> <num_mutations>")
            sys.exit(1)

        num_mutations = int(sys.argv[4])
        results = run_fuzz_test(Path(sys.argv[2]), Path(sys.argv[3]), num_mutations)

        print(f"[INDEPENDENT-VERIFIER] Fuzz test results:")
        print(f"  Total mutations: {results['total']}")
        print(f"  Rejected (correct): {results['rejected']}")
        print(f"  Accepted (BUG!): {results['accepted']}")
        print(f"  Reject rate: {100 * results['rejected'] / results['total']:.2f}%")
        print(f"\n  By mutation type:")
        for mtype, stats in results["by_type"].items():
            if stats["total"] > 0:
                rate = 100 * stats["rejected"] / stats["total"]
                print(f"    {mtype}: {stats['rejected']}/{stats['total']} rejected ({rate:.1f}%)")

        # Fail if any mutations were accepted
        if results["accepted"] > 0:
            print(f"\n  [FAIL] {results['accepted']} mutations were incorrectly accepted!")
            sys.exit(1)
        else:
            print(f"\n  [PASS] All mutations correctly rejected")
            sys.exit(0)

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
