"""Features sidecar binding BICEP trace to downstream consumers (ENN, FusionAlpha).

Phase 2a: Features shard + sidecar binding to checkpoint
- BICEP writes features + sidecar with cryptographic binding
- ENN MUST require and validate sidecar (refuse if missing/mismatch)
- FusionAlpha consumes ENN output with same binding pattern

This is the spine of the whole product:
"ENN only consumes features bound to a BICEP checkpoint receipt"
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .manifest import FEATURES_SIDECAR_SCHEMA, canonical_json, hash_canonical

# =============================================================================
# FEATURES SIDECAR - Cryptographic binding of features to checkpoint
# =============================================================================

@dataclass
class FeaturesSidecar:
    """Sidecar file binding feature shard to BICEP checkpoint.

    When BICEP writes features_shard, it MUST also write this sidecar.
    This is the cryptographic link: "these features came from those committed trace rows."
    """
    schema: str = FEATURES_SIDECAR_SCHEMA

    # Content binding
    features_shard_hash: str = ""   # SHA256 of features CSV/binary
    features_row_count: int = 0     # Number of feature rows
    features_dim: int = 0           # Feature vector dimension

    # Trace binding
    trace_anchor_type: str = "checkpoint"  # "checkpoint" or "full_trace"
    checkpoint_index: int = -1             # -1 if binding to full trace
    step_start: int = 0
    step_end: int = 0
    head_at_end: str = ""           # Chain head at step_end

    # Manifest binding
    manifest_hash: str = ""
    policy_hash: str = ""
    inputs_hash: str = ""

    # Metadata
    created_at: str = ""
    bicep_version: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def compute_sidecar_hash(self) -> str:
        """Hash of the sidecar itself."""
        return hash_canonical(asdict(self))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sidecar_hash"] = self.compute_sidecar_hash()
        return d

    def save(self, path: Path) -> None:
        """Save sidecar to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'FeaturesSidecar':
        """Load and verify sidecar from JSON file."""
        with open(path) as f:
            data = json.load(f)

        stored_hash = data.pop("sidecar_hash", "")
        sidecar = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

        # Verify hash
        computed = sidecar.compute_sidecar_hash()
        if stored_hash and computed != stored_hash:
            raise ValueError(
                f"Sidecar hash mismatch: stored={stored_hash}, computed={computed}"
            )

        return sidecar


def compute_features_hash(features_path: Path) -> str:
    """Compute SHA256 hash of features file."""
    hasher = hashlib.sha256()
    with open(features_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def create_features_sidecar(
    features_path: Path,
    manifest_hash: str,
    head_at_end: str,
    step_start: int,
    step_end: int,
    policy_hash: str = "",
    inputs_hash: str = "",
    checkpoint_index: int = -1,
    bicep_version: str = "0.1.0",
) -> FeaturesSidecar:
    """Create sidecar for a features shard.

    Args:
        features_path: Path to features CSV/binary file
        manifest_hash: Hash of run manifest
        head_at_end: Chain head at step_end
        step_start: First step included in shard
        step_end: Last step (exclusive) in shard
        policy_hash: Hash of policy document
        inputs_hash: Hash of input dataset
        checkpoint_index: Checkpoint index if binding to single checkpoint
        bicep_version: BICEP version string

    Returns:
        FeaturesSidecar with all bindings
    """
    # Compute features hash
    features_hash = compute_features_hash(features_path)

    # Count rows and detect dimension
    row_count = 0
    feature_dim = 0

    # Detect format by extension
    if features_path.suffix == '.csv':
        with open(features_path) as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                # Assuming features are numeric columns
                feature_dim = len(header)
            for _ in reader:
                row_count += 1
    else:
        # Binary format: assume header with dimensions
        with open(features_path, 'rb') as f:
            # Read simple header: row_count (int32), dim (int32)
            import struct
            header = f.read(8)
            if len(header) >= 8:
                row_count, feature_dim = struct.unpack('<II', header)

    sidecar = FeaturesSidecar(
        features_shard_hash=features_hash,
        features_row_count=row_count,
        features_dim=feature_dim,
        trace_anchor_type="checkpoint" if checkpoint_index >= 0 else "full_trace",
        checkpoint_index=checkpoint_index,
        step_start=step_start,
        step_end=step_end,
        head_at_end=head_at_end,
        manifest_hash=manifest_hash,
        policy_hash=policy_hash,
        inputs_hash=inputs_hash,
        bicep_version=bicep_version,
    )

    return sidecar


# =============================================================================
# ENN OUTPUT ARTIFACT - Binding ENN outputs back to BICEP input
# =============================================================================

@dataclass
class ENNOutputArtifact:
    """ENN output artifact with cryptographic binding to input features.

    ENN writes:
    - predictions (and optionally embeddings)
    - input_features_shard_hash: binding to BICEP output
    - enn_config_hash + enn_code_hash: model reproducibility
    - predictions_hash: commitment to outputs
    """
    schema: str = "enn_output_v1"

    # Input binding (to BICEP)
    input_features_shard_hash: str = ""
    input_sidecar_hash: str = ""
    trace_anchor_head: str = ""  # Chain head from sidecar

    # ENN configuration
    enn_config_hash: str = ""
    enn_code_hash: str = ""
    enn_version: str = ""

    # Output commitment
    predictions_hash: str = ""
    predictions_row_count: int = 0
    embeddings_hash: str = ""  # Optional

    # Metadata
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def compute_artifact_hash(self) -> str:
        return hash_canonical(asdict(self))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["artifact_hash"] = self.compute_artifact_hash()
        return d

    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ENNOutputArtifact':
        with open(path) as f:
            data = json.load(f)
        stored_hash = data.pop("artifact_hash", "")
        artifact = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        computed = artifact.compute_artifact_hash()
        if stored_hash and computed != stored_hash:
            raise ValueError(f"Artifact hash mismatch")
        return artifact


# =============================================================================
# FUSION ALPHA DECISION RECEIPT
# =============================================================================

@dataclass
class FusionDecisionReceipt:
    """FusionAlpha decision receipt with cryptographic binding to ENN output.

    FusionAlpha reads ENN artifact and produces decisions.
    This completes the chain: BICEP receipt -> ENN receipt -> FusionAlpha receipt
    """
    schema: str = "fusion_decision_v1"

    # Input binding (to ENN)
    enn_artifact_hash: str = ""
    enn_predictions_hash: str = ""

    # Upstream binding (transitively to BICEP)
    bicep_trace_anchor_head: str = ""
    bicep_manifest_hash: str = ""

    # Decision outputs
    decisions_hash: str = ""
    decisions_count: int = 0
    decision_method: str = ""  # e.g., "threshold_0.5", "calibrated_isotonic"

    # Metadata
    created_at: str = ""
    fusion_version: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def compute_receipt_hash(self) -> str:
        return hash_canonical(asdict(self))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["receipt_hash"] = self.compute_receipt_hash()
        return d

    def save(self, path: Path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

class SidecarValidationError(Exception):
    """Raised when sidecar validation fails."""
    pass


def validate_features_against_sidecar(
    features_path: Path,
    sidecar_path: Path,
) -> tuple[bool, str]:
    """Validate features file matches sidecar binding.

    This is what ENN MUST call before consuming features.
    Refuse to run if validation fails.
    """
    try:
        sidecar = FeaturesSidecar.load(sidecar_path)
    except Exception as e:
        return False, f"Failed to load sidecar: {e}"

    # Verify features hash
    actual_hash = compute_features_hash(features_path)
    if actual_hash != sidecar.features_shard_hash:
        return False, (
            f"Features hash mismatch: "
            f"file={actual_hash[:16]}..., sidecar={sidecar.features_shard_hash[:16]}..."
        )

    return True, f"Validated features against sidecar (hash={actual_hash[:16]}...)"


def require_valid_sidecar(
    features_path: Path,
    sidecar_path: Path,
) -> FeaturesSidecar:
    """Load and validate sidecar, raise if invalid.

    Use this as the entry point in ENN:
        sidecar = require_valid_sidecar(features_path, sidecar_path)
    """
    if not sidecar_path.exists():
        raise SidecarValidationError(
            f"Sidecar not found: {sidecar_path}\n"
            "ENN requires features to be bound to a BICEP checkpoint receipt.\n"
            "Run BICEP with trace emission enabled to generate sidecar."
        )

    if not features_path.exists():
        raise SidecarValidationError(f"Features file not found: {features_path}")

    valid, msg = validate_features_against_sidecar(features_path, sidecar_path)
    if not valid:
        raise SidecarValidationError(f"Sidecar validation failed: {msg}")

    return FeaturesSidecar.load(sidecar_path)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FeaturesSidecar",
    "compute_features_hash",
    "create_features_sidecar",
    "ENNOutputArtifact",
    "FusionDecisionReceipt",
    "SidecarValidationError",
    "validate_features_against_sidecar",
    "require_valid_sidecar",
]
