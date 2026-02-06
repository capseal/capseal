#!/usr/bin/env python3
"""
Canonical state audit primitives for bicep_v2.

This module defines the frozen verification contract used by BOTH
the executor and the independent verifier: canonical JSON, quantization,
addressable RNG, Merkle tree over quantized state, deterministic audit
challenge derivation, and audit opening (generate/verify).

All logic here must remain consistent with verifier-independent/state_audit.py
and verifier-independent/verifier.py.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import struct
from typing import Any


# =============================================================================
# CANONICAL JSON
# =============================================================================

def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


# =============================================================================
# QUANTIZATION
# =============================================================================

def quantize(
    value: float,
    precision_bits: int = 24,
    clamp_min: int = -(2**31),
    clamp_max: int = 2**31 - 1,
) -> int:
    scale = 2**precision_bits
    raw = round(value * scale)
    return max(clamp_min, min(clamp_max, raw))


def dequantize(q_value: int, precision_bits: int = 24) -> float:
    scale = 2**precision_bits
    return q_value / scale


# =============================================================================
# ADDRESSABLE RNG
# =============================================================================

class AddressableRNG:
    def __init__(self, seed: bytes):
        if len(seed) < 16:
            raise ValueError("Seed must be at least 16 bytes")
        self.seed = seed

    def rand(self, tag: str, t: int, i: int) -> float:
        address = tag.encode("utf-8") + struct.pack(">QQ", t, i)
        digest = hmac.new(self.seed, address, hashlib.sha256).digest()
        value = struct.unpack(">Q", digest[:8])[0]
        return value / (2**64)

    def rand_normal(
        self, tag: str, t: int, i: int, mu: float = 0.0, sigma: float = 1.0
    ) -> float:
        u1 = max(self.rand(tag, t, i * 2), 1e-10)
        u2 = self.rand(tag, t, i * 2 + 1)
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    @property
    def seed_commitment(self) -> str:
        return hashlib.sha256(b"seed_commit:" + self.seed).hexdigest()

    @classmethod
    def from_hex(cls, hex_seed: str) -> "AddressableRNG":
        return cls(bytes.fromhex(hex_seed))


# =============================================================================
# STATE LAYOUT / MERKLE
# =============================================================================

def leaf_index(path_id: int, channel_id: int, num_channels: int) -> int:
    return path_id * num_channels + channel_id


def leaf_to_path_channel(idx: int, num_channels: int) -> tuple[int, int]:
    return idx // num_channels, idx % num_channels


def total_leaves(num_paths: int, num_channels: int) -> int:
    return num_paths * num_channels


def _next_power_of_2(n: int) -> int:
    size = 1
    while size < n:
        size *= 2
    return size


def _leaf_digest(value_q: int) -> str:
    return hashlib.sha256(canonical_json_bytes(value_q)).hexdigest()


def build_merkle_tree(leaves_q: list[int]) -> tuple[str, list[list[str]]]:
    if not leaves_q:
        return hashlib.sha256(b"empty_state").hexdigest(), [[]]

    leaf_hashes = [_leaf_digest(v) for v in leaves_q]
    size = _next_power_of_2(len(leaf_hashes))
    padded = leaf_hashes + ["0" * 64] * (size - len(leaf_hashes))

    levels: list[list[str]] = [padded]
    current = padded
    while len(current) > 1:
        nxt = []
        for i in range(0, len(current), 2):
            combined = f"{current[i]}:{current[i+1]}"
            nxt.append(hashlib.sha256(combined.encode()).hexdigest())
        levels.append(nxt)
        current = nxt

    return current[0], levels


def compute_merkle_proof(levels: list[list[str]], leaf_idx: int) -> list[tuple[str, str]]:
    proof: list[tuple[str, str]] = []
    idx = leaf_idx
    for level in levels[:-1]:
        if idx % 2 == 0:
            sibling_idx = idx + 1
            direction = "R"
        else:
            sibling_idx = idx - 1
            direction = "L"
        if sibling_idx < len(level):
            proof.append((level[sibling_idx], direction))
        else:
            proof.append(("0" * 64, direction))
        idx //= 2
    return proof


def verify_merkle_proof(value_q: int, proof: list[tuple[str, str]], root: str) -> bool:
    current = _leaf_digest(value_q)
    for sibling, direction in proof:
        if direction == "L":
            combined = f"{sibling}:{current}"
        else:
            combined = f"{current}:{sibling}"
        current = hashlib.sha256(combined.encode()).hexdigest()
    return current == root


# =============================================================================
# AUDIT CHALLENGES
# =============================================================================

def compute_challenge_seed(
    manifest_hash: str, head_t: str, state_root_t: str, rng_use_hash_t: str
) -> str:
    preimage = f"{manifest_hash}{head_t}{state_root_t}{rng_use_hash_t}"
    return hashlib.sha256(preimage.encode()).hexdigest()


def sample_audit_indices(challenge_seed: str, num_leaves: int, k: int) -> list[int]:
    seed_bytes = bytes.fromhex(challenge_seed)
    indices: set[int] = set()
    counter = 0
    while len(indices) < k and len(indices) < num_leaves:
        addr = seed_bytes + counter.to_bytes(4, "big")
        h = hashlib.sha256(addr).digest()
        idx = int.from_bytes(h[:4], "big") % num_leaves
        indices.add(idx)
        counter += 1
    return sorted(indices)


# =============================================================================
# SDE STEP (Euler-Maruyama + optional jump)
# =============================================================================

def sde_step_em(
    state_pre_q: int, params: dict, epsilon: float, precision_bits: int = 24
) -> int:
    state_pre = dequantize(state_pre_q, precision_bits)

    theta = params["theta"]
    mu0 = params["mu0"]
    sigma = params["sigma"]
    dt = params["dt"]

    drift = theta * (mu0 - state_pre) * dt
    diffusion = sigma * math.sqrt(dt) * epsilon
    state_post = state_pre + drift + diffusion
    return quantize(state_post, precision_bits)


def sde_step_with_jump(
    state_pre_q: int,
    params: dict,
    epsilon: float,
    jump_flag: float,
    jump_mag: float,
    precision_bits: int = 24,
) -> int:
    state_post_q = sde_step_em(state_pre_q, params, epsilon, precision_bits)
    jump_rate = params.get("jump_rate", 0.0)
    jump_scale = params.get("jump_scale", 0.0)
    if jump_rate > 0 and jump_flag < jump_rate:
        state_post = dequantize(state_post_q, precision_bits)
        state_post += jump_scale * jump_mag
        state_post_q = quantize(state_post, precision_bits)
    return state_post_q


# =============================================================================
# AUDIT OPENINGS (generate/verify)
# =============================================================================

def generate_audit_opening(
    t: int,
    state_pre_q: list[int],
    state_post_q: list[int],
    levels_pre: list[list[str]],
    levels_post: list[list[str]],
    state_root_pre: str,
    state_root_post: str,
    audit_indices: list[int],
    challenge_seed: str,
) -> dict:
    openings_pre = []
    openings_post = []
    for idx in audit_indices:
        proof_pre = compute_merkle_proof(levels_pre, idx)
        openings_pre.append(
            {"leaf_index": idx, "value_q": state_pre_q[idx] if idx < len(state_pre_q) else 0, "merkle_proof": proof_pre}
        )
        proof_post = compute_merkle_proof(levels_post, idx)
        openings_post.append(
            {"leaf_index": idx, "value_q": state_post_q[idx] if idx < len(state_post_q) else 0, "merkle_proof": proof_post}
        )
    return {
        "schema": "bicep_audit_v1",
        "t": t,
        "challenge_seed": challenge_seed,
        "audit_indices": audit_indices,
        "openings_pre": openings_pre,
        "openings_post": openings_post,
    }


def verify_audit_opening(
    audit: dict, row: dict, manifest: dict, manifest_hash: str, head_t: str, rng: AddressableRNG
) -> tuple[bool, str]:
    t = audit["t"]
    state_root_pre = row["view_pre"]["state_root"]
    state_root_post = row["view_post"]["state_root"]
    rng_use_hash = row["rng_use_hash"]

    expected_seed = compute_challenge_seed(manifest_hash, head_t, state_root_pre, rng_use_hash)
    if audit["challenge_seed"] != expected_seed:
        return False, f"Challenge seed mismatch at t={t}: expected {expected_seed[:16]}..., got {audit['challenge_seed'][:16]}..."

    num_paths = manifest.get("state_num_paths", 4)
    num_channels = manifest.get("state_num_channels", 4)
    num_leaves = total_leaves(num_paths, num_channels)
    audit_k = manifest.get("audit_k", 16)

    expected_indices = sample_audit_indices(expected_seed, num_leaves, audit_k)
    if audit["audit_indices"] != expected_indices:
        return False, f"Audit indices mismatch at t={t}: expected {expected_indices[:3]}..., got {audit['audit_indices'][:3]}..."

    precision_bits = manifest.get("x_quant_precision_bits", 24)
    sde_params = manifest.get("sde_params", {})

    for i, idx in enumerate(audit["audit_indices"]):
        opening_pre = audit["openings_pre"][i]
        opening_post = audit["openings_post"][i]
        if opening_pre["leaf_index"] != idx or opening_post["leaf_index"] != idx:
            return False, f"Leaf index mismatch at t={t}, audit[{i}]: expected {idx}"

        if not verify_merkle_proof(opening_pre["value_q"], opening_pre["merkle_proof"], state_root_pre):
            return False, f"Pre-state Merkle proof failed at t={t}, leaf={idx}"
        if not verify_merkle_proof(opening_post["value_q"], opening_post["merkle_proof"], state_root_post):
            return False, f"Post-state Merkle proof failed at t={t}, leaf={idx}"

        epsilon = rng.rand_normal("sde_noise", t, idx)
        jump_rate = sde_params.get("jump_rate", 0.0)
        if jump_rate > 0:
            jump_flag = rng.rand("jump_mask", t, idx)
            jump_mag = rng.rand_normal("jump_mag", t, idx)
            expected_post_q = sde_step_with_jump(
                opening_pre["value_q"], sde_params, epsilon, jump_flag, jump_mag, precision_bits
            )
        else:
            expected_post_q = sde_step_em(opening_pre["value_q"], sde_params, epsilon, precision_bits)

        if opening_post["value_q"] != expected_post_q:
            return False, f"SDE transition failed at t={t}, leaf={idx}: expected {expected_post_q}, got {opening_post['value_q']}"

    return True, f"Audit verified at t={t}: {len(audit['audit_indices'])} leaves checked"

