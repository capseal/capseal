#!/usr/bin/env python3
"""
Trace Generator - Reference implementation for generating valid traces.

This is the "ground truth" trace generator that produces traces matching
the transition spec contracts. Used for:
1. Testing the verifier against known-good traces
2. Generating test fixtures
3. Demonstrating spec semantics

Usage:
  python trace_generator.py generate identity_v1 <seed_hex> <num_steps> <output.jsonl>
  python trace_generator.py generate keyed_hash_v1 <seed_hex> <num_steps> <output.jsonl>
  python trace_generator.py demo <spec_id>
"""
from __future__ import annotations

import hashlib
import hmac
import json
import struct
import sys
from pathlib import Path
from typing import Any


# =============================================================================
# CANONICAL JSON (must match verifier exactly)
# =============================================================================

def canonical_json_bytes(obj: Any) -> bytes:
    """Produce deterministic JSON bytes for hashing."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,
    ).encode('utf-8')


# =============================================================================
# ADDRESSABLE RNG (must match verifier exactly)
# =============================================================================

class AddressableRNG:
    """Addressable randomness: rand(tag, t, i) = PRG(seed, tag || t || i)."""

    def __init__(self, seed: bytes):
        if len(seed) < 16:
            raise ValueError("Seed must be at least 16 bytes")
        self.seed = seed

    def rand(self, tag: str, t: int, i: int) -> float:
        """Generate deterministic random value in [0, 1)."""
        address = tag.encode('utf-8') + struct.pack('>QQ', t, i)
        digest = hmac.new(self.seed, address, hashlib.sha256).digest()
        value = struct.unpack('>Q', digest[:8])[0]
        return value / (2**64)

    @property
    def seed_commitment(self) -> str:
        """Public commitment to seed (hash, not seed itself)."""
        return hashlib.sha256(b"seed_commit:" + self.seed).hexdigest()

    @classmethod
    def from_hex(cls, hex_seed: str) -> 'AddressableRNG':
        return cls(bytes.fromhex(hex_seed))


# =============================================================================
# SPEC: identity_v1
# =============================================================================

def generate_trace_identity_v1(
    seed_hex: str,
    num_steps: int,
    num_features: int = 4,
) -> tuple[list[dict], dict]:
    """Generate a valid trace for identity_v1 spec.

    Returns:
        (rows, manifest) tuple
    """
    rng = AddressableRNG.from_hex(seed_hex)

    manifest = {
        "transition_spec_id": "identity_v1",
        "output_fn_id": "rng_features_v1",
        "seed_commitment": rng.seed_commitment,
        "state_view_schema_id": "minimal_projection_v1",
    }

    rows = []
    for t in range(num_steps):
        # State is just the timestep counter
        view_pre = {"state": t}
        view_post = {"state": t + 1}

        # Output is RNG-derived
        x_t = [rng.rand("input", t, i) for i in range(num_features)]

        # RNG addresses consumed
        rand_addrs = [{"tag": "input", "t": t, "i": i} for i in range(num_features)]

        row = {
            "schema": "bicep_trace_v1",
            "t": t,
            "x_t": x_t,
            "view_pre": view_pre,
            "view_post": view_post,
            "rand_addrs": rand_addrs,
        }
        rows.append(row)

    return rows, manifest


# =============================================================================
# SPEC: keyed_hash_v1
# =============================================================================

def compute_genesis_hash(seed_commitment: str) -> str:
    """Compute genesis hash for keyed_hash_v1."""
    preimage = f"keyed_hash_v1:genesis:{seed_commitment}"
    return hashlib.sha256(preimage.encode('utf-8')).hexdigest()


def compute_next_hash(current_hash: str, x_t: list[float]) -> str:
    """Compute next hash: H(current_hash || canonical_json(x_t))."""
    x_t_bytes = canonical_json_bytes(x_t)
    preimage = current_hash.encode('utf-8') + x_t_bytes
    return hashlib.sha256(preimage).hexdigest()


def generate_trace_keyed_hash_v1(
    seed_hex: str,
    num_steps: int,
    num_features: int = 4,
) -> tuple[list[dict], dict]:
    """Generate a valid trace for keyed_hash_v1 spec.

    Returns:
        (rows, manifest) tuple
    """
    rng = AddressableRNG.from_hex(seed_hex)

    manifest = {
        "transition_spec_id": "keyed_hash_v1",
        "output_fn_id": "rng_features_v1",
        "seed_commitment": rng.seed_commitment,
        "state_view_schema_id": "keyed_hash_state_v1",
        "transition_fn_id": "keyed_hash_absorb_v1",
    }

    # Compute genesis hash
    current_hash = compute_genesis_hash(rng.seed_commitment)

    rows = []
    for t in range(num_steps):
        # Output is RNG-derived
        x_t = [rng.rand("input", t, i) for i in range(num_features)]

        # State is hash that evolves by absorbing outputs
        view_pre = {"t": t, "hash": current_hash}

        # Compute next hash
        next_hash = compute_next_hash(current_hash, x_t)
        view_post = {"t": t + 1, "hash": next_hash}

        # RNG addresses consumed
        rand_addrs = [{"tag": "input", "t": t, "i": i} for i in range(num_features)]

        row = {
            "schema": "bicep_trace_v1",
            "t": t,
            "x_t": x_t,
            "view_pre": view_pre,
            "view_post": view_post,
            "rand_addrs": rand_addrs,
        }
        rows.append(row)

        # Advance state
        current_hash = next_hash

    return rows, manifest


# =============================================================================
# SPEC: bicep_v1 (production spec with quantized outputs)
# =============================================================================

def quantize(value: float, precision_bits: int = 24, clamp_min: int = -(2**31), clamp_max: int = 2**31 - 1) -> int:
    """Quantize float to fixed-point integer."""
    scale = 2 ** precision_bits
    raw = round(value * scale)
    return max(clamp_min, min(clamp_max, raw))


def dequantize(q_value: int, precision_bits: int = 24) -> float:
    """Recover float from quantized integer."""
    scale = 2 ** precision_bits
    return q_value / scale


def compute_genesis_bicep_v1(seed_commitment: str) -> tuple[str, str]:
    """Compute genesis state_root and output_chain for bicep_v1."""
    state_root = hashlib.sha256(
        f"bicep_v1:state:genesis:{seed_commitment}".encode()
    ).hexdigest()

    output_chain = hashlib.sha256(
        f"bicep_v1:output_chain:genesis:{seed_commitment}".encode()
    ).hexdigest()

    return state_root, output_chain


def compute_rng_use_hash(rand_addrs: list[dict]) -> str:
    """Compute H(canonical(rand_addrs)) for tamper evidence."""
    return hashlib.sha256(canonical_json_bytes(rand_addrs)).hexdigest()


def compute_output_chain_update(prev_chain: str, x_t_q: list[int]) -> str:
    """Compute H(prev_chain || canonical(x_t_q))."""
    x_bytes = canonical_json_bytes(x_t_q)
    preimage = prev_chain.encode('utf-8') + x_bytes
    return hashlib.sha256(preimage).hexdigest()


def compute_state_root_update(prev_root: str, x_t_q: list[int], t: int) -> str:
    """Compute new state_root.

    For now, simple hash chain. In real BICEP, this would be
    derived from SDE state evolution.
    """
    # Simple model: state_root chains timestep + quantized outputs
    preimage = f"{prev_root}:{t}:{canonical_json_bytes(x_t_q).decode()}"
    return hashlib.sha256(preimage.encode()).hexdigest()


def generate_trace_bicep_v1(
    seed_hex: str,
    num_steps: int,
    num_features: int = 4,
    precision_bits: int = 24,
) -> tuple[list[dict], dict]:
    """Generate a valid trace for bicep_v1 spec.

    Returns:
        (rows, manifest) tuple
    """
    rng = AddressableRNG.from_hex(seed_hex)

    manifest = {
        "transition_spec_id": "bicep_v1",
        "output_fn_id": "bicep_features_v1",
        "seed_commitment": rng.seed_commitment,
        "state_view_schema_id": "bicep_state_v1",
        "transition_fn_id": "bicep_sde_v1",
        "x_quant_scheme_id": "fixed_point_v1",
        "x_quant_precision_bits": precision_bits,
        "sampling_scheme_id": "standard_v1",
    }

    # Compute genesis state
    state_root, output_chain = compute_genesis_bicep_v1(rng.seed_commitment)

    rows = []
    for t in range(num_steps):
        # Output is RNG-derived, then quantized
        x_t_raw = [rng.rand("input", t, i) for i in range(num_features)]
        x_t_q = [quantize(v, precision_bits) for v in x_t_raw]
        x_t_f = [dequantize(q, precision_bits) for q in x_t_q]  # Convenience float

        # RNG addresses consumed
        rand_addrs = [{"tag": "input", "t": t, "i": i} for i in range(num_features)]
        rng_use_hash = compute_rng_use_hash(rand_addrs)

        # State before step
        view_pre = {
            "state_root": state_root,
            "output_chain": output_chain,
        }

        # Compute new state
        new_output_chain = compute_output_chain_update(output_chain, x_t_q)
        new_state_root = compute_state_root_update(state_root, x_t_q, t)

        # State after step
        view_post = {
            "state_root": new_state_root,
            "output_chain": new_output_chain,
        }

        row = {
            "schema": "bicep_trace_v1",
            "t": t,
            "x_t_q": x_t_q,
            "x_t_f": x_t_f,  # Convenience (not committed)
            "view_pre": view_pre,
            "view_post": view_post,
            "rand_addrs": rand_addrs,
            "rng_use_hash": rng_use_hash,
        }
        rows.append(row)

        # Advance state
        state_root = new_state_root
        output_chain = new_output_chain

    return rows, manifest


# =============================================================================
# SPEC REGISTRY
# =============================================================================

GENERATORS = {
    "identity_v1": generate_trace_identity_v1,
    "keyed_hash_v1": generate_trace_keyed_hash_v1,
    "bicep_v1": generate_trace_bicep_v1,
}


def generate_trace(
    spec_id: str,
    seed_hex: str,
    num_steps: int,
    num_features: int = 4,
) -> tuple[list[dict], dict]:
    """Generate trace for given spec."""
    if spec_id not in GENERATORS:
        raise ValueError(f"Unknown spec: {spec_id}. Available: {list(GENERATORS.keys())}")
    return GENERATORS[spec_id](seed_hex, num_steps, num_features)


def save_trace(rows: list[dict], output_path: Path):
    """Save trace to JSONL file."""
    with open(output_path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')


def save_manifest(manifest: dict, output_path: Path):
    """Save manifest to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


# =============================================================================
# DEMO
# =============================================================================

def demo_spec(spec_id: str):
    """Demo a spec with a small trace."""
    seed_hex = "deadbeef" * 8
    num_steps = 3

    print(f"\n{'='*60}")
    print(f"DEMO: {spec_id}")
    print(f"{'='*60}")
    print(f"Seed (hex): {seed_hex[:16]}...")
    print(f"Steps: {num_steps}")

    rows, manifest = generate_trace(spec_id, seed_hex, num_steps)

    print(f"\nManifest:")
    print(json.dumps(manifest, indent=2))

    print(f"\nTrace rows:")
    for row in rows:
        print(f"\n  t={row['t']}:")
        # Handle both float (x_t) and quantized (x_t_q) output formats
        if "x_t" in row:
            print(f"    x_t = [{', '.join(f'{v:.6f}' for v in row['x_t'][:3])}...]")
        if "x_t_q" in row:
            print(f"    x_t_q = {row['x_t_q'][:3]}...")
            if "x_t_f" in row and row["x_t_f"]:
                print(f"    x_t_f = [{', '.join(f'{v:.6f}' for v in row['x_t_f'][:3])}...]")
        if "rng_use_hash" in row:
            print(f"    rng_use_hash = {row['rng_use_hash'][:16]}...")
        print(f"    view_pre  = {json.dumps(row['view_pre'])[:60]}...")
        print(f"    view_post = {json.dumps(row['view_post'])[:60]}...")

    # Show state evolution for keyed_hash_v1
    if spec_id == "keyed_hash_v1":
        print("\n  Hash chain evolution:")
        for i, row in enumerate(rows):
            pre_hash = row['view_pre']['hash'][:16]
            post_hash = row['view_post']['hash'][:16]
            print(f"    t={i}: {pre_hash}... -> H(hash || x_t) -> {post_hash}...")

    # Show output chain evolution for bicep_v1
    if spec_id == "bicep_v1":
        print("\n  Output chain evolution (quantized):")
        for i, row in enumerate(rows):
            pre_chain = row['view_pre']['output_chain'][:16]
            post_chain = row['view_post']['output_chain'][:16]
            print(f"    t={i}: {pre_chain}... -> H(chain || x_t_q) -> {post_chain}...")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"\nAvailable specs: {list(GENERATORS.keys())}")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "generate":
        if len(sys.argv) != 6:
            print("Usage: generate <spec_id> <seed_hex> <num_steps> <output.jsonl>")
            sys.exit(1)

        spec_id = sys.argv[2]
        seed_hex = sys.argv[3]
        num_steps = int(sys.argv[4])
        output_path = Path(sys.argv[5])

        rows, manifest = generate_trace(spec_id, seed_hex, num_steps)
        save_trace(rows, output_path)

        # Also save manifest
        manifest_path = output_path.with_suffix('.manifest.json')
        save_manifest(manifest, manifest_path)

        print(f"Generated {len(rows)} rows for spec={spec_id}")
        print(f"  Trace: {output_path}")
        print(f"  Manifest: {manifest_path}")

    elif cmd == "demo":
        spec_id = sys.argv[2] if len(sys.argv) > 2 else "identity_v1"
        demo_spec(spec_id)

    elif cmd == "list":
        print("Available specs:")
        for spec_id in GENERATORS:
            print(f"  - {spec_id}")

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
