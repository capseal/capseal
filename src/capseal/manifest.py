"""Manifest and verification infrastructure for tamper-evident BICEP traces.

Phase 0: Lock verification invariants in manifest.json
- Addressable RNG: rand(tag, t, i) = PRG(seed, tag || t || i)
- Numeric model: quantized float (log/commit quantized values)
- Trace schema versioning with deterministic serialization

This module is the foundation for future STARK/IVC verification.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import struct
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# =============================================================================
# VERSION CONSTANTS (immutable once published)
# =============================================================================

MANIFEST_SCHEMA_VERSION = "manifest_v1"
TRACE_ROW_SCHEMA_VERSION = "bicep_trace_v1"
CHECKPOINT_SCHEMA_VERSION = "checkpoint_v1"
FEATURES_SIDECAR_SCHEMA = "features_sidecar_v1"

# =============================================================================
# ADDRESSABLE RNG - The single biggest "future verification killer" crushed now
# =============================================================================

class AddressableRNG:
    """Addressable randomness: rand(tag, t, i) = PRG(seed, tag || t || i).

    CRITICAL: No stream consumption. No call-order dependence.
    This is what makes replay verification deterministic.
    """

    def __init__(self, seed: bytes):
        if len(seed) < 16:
            raise ValueError("Seed must be at least 16 bytes")
        self.seed = seed
        self._seed_hash = hashlib.sha256(seed).hexdigest()[:16]

    def rand(self, tag: str, t: int, i: int) -> float:
        """Generate deterministic random value in [0, 1).

        Args:
            tag: Semantic tag (e.g., "sample", "noise", "dropout")
            t: Timestep index
            i: Within-step index

        Returns:
            Deterministic float in [0, 1)
        """
        # Canonical serialization: tag bytes + big-endian t + big-endian i
        address = tag.encode('utf-8') + struct.pack('>QQ', t, i)
        # HMAC-SHA256 as PRG
        digest = hmac.new(self.seed, address, hashlib.sha256).digest()
        # Take first 8 bytes as uint64, normalize to [0, 1)
        value = struct.unpack('>Q', digest[:8])[0]
        return value / (2**64)

    def rand_int(self, tag: str, t: int, i: int, low: int, high: int) -> int:
        """Generate deterministic random integer in [low, high)."""
        r = self.rand(tag, t, i)
        return int(low + r * (high - low))

    def rand_normal(self, tag: str, t: int, i: int, mu: float = 0.0, sigma: float = 1.0) -> float:
        """Generate deterministic normal sample using Box-Muller."""
        import math
        u1 = max(self.rand(tag, t, i * 2), 1e-10)  # Avoid log(0)
        u2 = self.rand(tag, t, i * 2 + 1)
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def get_address_record(self, tag: str, t: int, i_start: int, i_count: int) -> dict:
        """Return a record of which random addresses were consumed."""
        return {
            "tag": tag,
            "t": t,
            "i_start": i_start,
            "i_count": i_count,
        }

    @property
    def seed_commitment(self) -> str:
        """Public commitment to seed (hash, not seed itself)."""
        return hashlib.sha256(b"seed_commit:" + self.seed).hexdigest()


# =============================================================================
# NUMERIC MODEL - Quantization for verification-friendly floats
# =============================================================================

class NumericModel(Enum):
    """Numeric precision model for trace values."""
    FLOAT64_QUANTIZED = "float64_quantized"  # Prototype-friendly: quantize on commit
    FIXED32 = "fixed32"                       # Verification-friendly: 32-bit fixed point
    FIXED64 = "fixed64"                       # Higher precision fixed point


@dataclass
class QuantizationConfig:
    """Configuration for quantizing floats for commitment."""
    model: NumericModel = NumericModel.FLOAT64_QUANTIZED
    precision_bits: int = 24  # Bits of precision to preserve
    max_exponent: int = 127   # IEEE 754 compatible range

    def quantize(self, value: float) -> int:
        """Quantize float to integer representation."""
        import math
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"Cannot quantize {value}")

        if self.model == NumericModel.FLOAT64_QUANTIZED:
            # Multiply by 2^precision_bits, round to nearest int
            scale = 2 ** self.precision_bits
            return int(round(value * scale))
        else:
            raise NotImplementedError(f"Model {self.model} not implemented")

    def dequantize(self, qvalue: int) -> float:
        """Recover float from quantized representation."""
        if self.model == NumericModel.FLOAT64_QUANTIZED:
            scale = 2 ** self.precision_bits
            return qvalue / scale
        else:
            raise NotImplementedError(f"Model {self.model} not implemented")


# =============================================================================
# CANONICAL SERIALIZATION - Deterministic JSON for hashing
# =============================================================================

def canonical_json(obj: Any) -> bytes:
    """Produce deterministic JSON bytes for hashing.

    Rules:
    - Keys sorted alphabetically
    - No whitespace
    - Numbers: finite floats only, no scientific notation for small values
    - Strings: UTF-8
    - No locale-dependent formatting
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False,
        allow_nan=False,  # Reject NaN/Inf
    ).encode('utf-8')


def hash_canonical(obj: Any) -> str:
    """Hash an object via canonical JSON serialization."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()


# =============================================================================
# MANIFEST SCHEMA
# =============================================================================

@dataclass
class ManifestConfig:
    """Immutable configuration locked at run start.

    This is what makes future verification possible without rewriting BICEP.

    Frozen specs:
    - docs/contracts/transition_spec_v1.md (identity_v1 - toy spec)
    - docs/contracts/keyed_hash_spec_v1.md (keyed_hash_v1 - cryptographic state)

    Adding a new spec:
    1. Create docs/contracts/<spec>_v1.md with frozen definitions
    2. Add spec to verifier-independent/verifier.py TRANSITION_SPECS dict
    3. Add generator to verifier-independent/trace_generator.py
    4. Add tests to verifier-independent/test_<spec>.py
    """
    # Schema identification
    schema: str = MANIFEST_SCHEMA_VERSION
    trace_row_schema: str = TRACE_ROW_SCHEMA_VERSION

    # ==========================================================================
    # TRANSITION SPEC CONTRACT (frozen - see docs/contracts/*.md)
    # ==========================================================================
    #
    # Available specs:
    #   - identity_v1: toy spec, state = counter (for testing infrastructure)
    #   - keyed_hash_v1: cryptographic state, hash chains float outputs
    #   - bicep_v1: production spec, quantized outputs, AIR-ready
    #
    # To add a new spec: freeze the contract first, then implement.
    # ==========================================================================

    # Overall spec version (determines which verifier dispatch to use)
    transition_spec_id: str = "identity_v1"

    # RNG contract (shared across all current specs)
    rng_id: str = "hmac_sha256_v1"  # PRG algorithm
    domain_sep_scheme_id: str = "tag_t_i_v1"  # Domain separation format
    seed_commitment: str = ""  # H(seed), not seed itself

    # Sampling scheme (how RNG outputs expand to noise values)
    # - standard_v1: direct PRG output
    # - antithetic_v1: each epsilon expands to (e, -e)
    sampling_scheme_id: str = "standard_v1"

    # Numeric model (internal computation)
    numeric_model: str = "float64_ieee754_v1"  # IEEE 754 double

    # ==========================================================================
    # QUANTIZATION (for commitment-boundary canonicalization)
    # ==========================================================================
    # Everything committed (hashed/compared) must be quantized integers.
    # Floats are convenience only, derived from quantized form.

    x_quant_scheme_id: str = "fixed_point_v1"
    x_quant_precision_bits: int = 24  # scale = 2^24
    x_quant_clamp_min: int = -(2**31)  # Saturation floor
    x_quant_clamp_max: int = 2**31 - 1  # Saturation ceiling

    # State view schema (what view_pre/view_post mean)
    # - minimal_projection_v1: identity_v1 uses {"state": int}
    # - keyed_hash_state_v1: keyed_hash_v1 uses {"t": int, "hash": hex_string}
    # - bicep_state_v1: bicep_v1 uses {"state_root": hex, "output_chain": hex}
    state_view_schema_id: str = "minimal_projection_v1"

    # Transition and output functions
    # - identity_v1: s_{t+1} = s_t + 1
    # - keyed_hash_absorb_v1: s_{t+1}.hash = H(s_t.hash || x_t)
    # - bicep_sde_v1: SDE state evolution with quantized outputs
    transition_fn_id: str = "identity_v1"
    output_fn_id: str = "rng_features_v1"  # x_t = g(s_t, r_t)

    # ==========================================================================
    # TRACE CONFIGURATION
    # ==========================================================================

    checkpoint_interval: int = 1024  # Emit checkpoint every K steps

    # ==========================================================================
    # CODE/ENVIRONMENT BINDING
    # ==========================================================================

    bicep_version: str = ""
    bicep_code_hash: str = ""  # Hash of relevant source files
    policy_hash: str = ""
    inputs_hash: str = ""  # Dataset + config hash

    # Timestamps (ISO 8601 UTC)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


@dataclass
class Manifest:
    """Complete run manifest with computed hash."""
    config: ManifestConfig
    manifest_hash: str = ""

    def __post_init__(self):
        if not self.manifest_hash:
            self.manifest_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute manifest hash from canonical config."""
        return hash_canonical(asdict(self.config))

    def to_dict(self) -> dict:
        return {
            "manifest_hash": self.manifest_hash,
            **asdict(self.config),
        }

    def save(self, path: Path) -> None:
        """Save manifest to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Manifest] Saved to {path}")
        print(f"[Manifest] manifest_hash = {self.manifest_hash}")

    @classmethod
    def load(cls, path: Path) -> 'Manifest':
        """Load and verify manifest from JSON file."""
        with open(path) as f:
            data = json.load(f)

        stored_hash = data.pop("manifest_hash", "")
        config = ManifestConfig(**data)
        manifest = cls(config=config, manifest_hash=stored_hash)

        # Verify hash
        computed = manifest.compute_hash()
        if computed != stored_hash:
            raise ValueError(
                f"Manifest hash mismatch: stored={stored_hash}, computed={computed}"
            )

        return manifest


# =============================================================================
# MANIFEST BUILDER
# =============================================================================

# Spec presets for convenience
SPEC_PRESETS = {
    "identity_v1": {
        "transition_spec_id": "identity_v1",
        "transition_fn_id": "identity_v1",
        "state_view_schema_id": "minimal_projection_v1",
        "output_fn_id": "rng_features_v1",
    },
    "keyed_hash_v1": {
        "transition_spec_id": "keyed_hash_v1",
        "transition_fn_id": "keyed_hash_absorb_v1",
        "state_view_schema_id": "keyed_hash_state_v1",
        "output_fn_id": "rng_features_v1",
    },
    "bicep_v1": {
        "transition_spec_id": "bicep_v1",
        "transition_fn_id": "bicep_sde_v1",
        "state_view_schema_id": "bicep_state_v1",
        "output_fn_id": "bicep_features_v1",
        "x_quant_scheme_id": "fixed_point_v1",
        "sampling_scheme_id": "standard_v1",
    },
    "bicep_v2": {
        "transition_spec_id": "bicep_v2",
        "transition_fn_id": "bicep_sde_em_v1",
        "state_view_schema_id": "bicep_state_v2",
        "output_fn_id": "bicep_features_v1",
        "x_quant_scheme_id": "fixed_point_v1",
        "sampling_scheme_id": "standard_v1",
    },
}


def create_manifest(
    seed: bytes,
    policy_path: Optional[Path] = None,
    inputs_paths: list[Path] = None,
    bicep_version: str = "0.1.0",
    checkpoint_interval: int = 1024,
    transition_spec_id: str = "identity_v1",
) -> tuple[Manifest, AddressableRNG]:
    """Create manifest and RNG for a new run.

    Args:
        seed: RNG seed (at least 16 bytes)
        policy_path: Optional path to policy file
        inputs_paths: Optional list of input file paths
        bicep_version: Version string for BICEP
        checkpoint_interval: Steps per checkpoint (K)
        transition_spec_id: Which transition spec to use
            - "identity_v1": toy spec (state = counter)
            - "keyed_hash_v1": cryptographic state (hash chains float outputs)
            - "bicep_v1": production spec (quantized outputs, AIR-ready)

    Returns:
        Tuple of (Manifest, AddressableRNG) bound together
    """
    rng = AddressableRNG(seed)

    # Get spec preset
    if transition_spec_id not in SPEC_PRESETS:
        raise ValueError(
            f"Unknown transition_spec_id: {transition_spec_id}. "
            f"Available: {list(SPEC_PRESETS.keys())}"
        )
    spec_config = SPEC_PRESETS[transition_spec_id]

    # Compute input hashes
    policy_hash = ""
    if policy_path and policy_path.exists():
        policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()

    inputs_hash = ""
    if inputs_paths:
        hasher = hashlib.sha256()
        for p in sorted(inputs_paths):  # Deterministic order
            if p.exists():
                hasher.update(p.read_bytes())
        inputs_hash = hasher.hexdigest()

    config = ManifestConfig(
        seed_commitment=rng.seed_commitment,
        policy_hash=policy_hash,
        inputs_hash=inputs_hash,
        bicep_version=bicep_version,
        checkpoint_interval=checkpoint_interval,
        **spec_config,
    )

    manifest = Manifest(config=config)
    return manifest, rng


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "TRACE_ROW_SCHEMA_VERSION",
    "CHECKPOINT_SCHEMA_VERSION",
    "FEATURES_SIDECAR_SCHEMA",
    "AddressableRNG",
    "NumericModel",
    "QuantizationConfig",
    "canonical_json",
    "hash_canonical",
    "ManifestConfig",
    "Manifest",
    "create_manifest",
    "SPEC_PRESETS",
]
