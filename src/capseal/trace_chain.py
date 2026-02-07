"""Trace rows and hash chain for tamper-evident BICEP execution.

Phase 1: BICEP as witness/trace emitter (hash chain first)
- TraceRow: minimal per-step witness
- Hash chain: d_t = H(row_t), head_{t+1} = H(head_t || d_t)
- Checkpoint receipts every K steps

This is tamper-evidence done: given manifest + commitments + trace,
anyone can recompute head_T and verify it matches.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Optional

from .manifest import (
    TRACE_ROW_SCHEMA_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    canonical_json,
    hash_canonical,
    QuantizationConfig,
)

# =============================================================================
# TRACE ROW - Minimal per-step witness
# =============================================================================

@dataclass
class RandRecord:
    """Record of random addresses consumed in a step."""
    tag: str
    t: int
    i_start: int
    i_count: int


@dataclass
class TraceRow:
    """Minimal trace row for verification.

    Contains exactly what's needed to verify the transition, not "whatever was around."
    - t: timestep
    - rand_addrs: random addresses consumed
    - view_pre: smallest state slice required to check the step
    - view_post: state after step
    - x_t: features emitted to downstream (ENN)
    - aux: optional expensive-to-recompute values (avoid widening trust)
    """
    schema: str = TRACE_ROW_SCHEMA_VERSION
    t: int = 0
    rand_addrs: list[dict] = field(default_factory=list)
    view_pre: dict[str, Any] = field(default_factory=dict)
    view_post: dict[str, Any] = field(default_factory=dict)
    x_t: list[float] = field(default_factory=list)
    aux: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dict for serialization."""
        return asdict(self)

    def compute_digest(self) -> str:
        """Compute d_t = H(row_t) using canonical serialization."""
        return hash_canonical(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict) -> 'TraceRow':
        """Reconstruct from dict."""
        return cls(
            schema=data.get("schema", TRACE_ROW_SCHEMA_VERSION),
            t=data["t"],
            rand_addrs=data.get("rand_addrs", []),
            view_pre=data.get("view_pre", {}),
            view_post=data.get("view_post", {}),
            x_t=data.get("x_t", []),
            aux=data.get("aux", {}),
        )


# =============================================================================
# HASH CHAIN - head_{t+1} = H(head_t || d_t)
# =============================================================================

@dataclass
class HashChainState:
    """Running state of the hash chain."""
    head: str  # Current chain head (hex)
    step: int  # Current step count
    digests: list[str] = field(default_factory=list)  # Recent d_t values (for checkpoint)

    @classmethod
    def genesis(cls, manifest_hash: str) -> 'HashChainState':
        """Create genesis state from manifest hash."""
        # head_0 = H("genesis" || manifest_hash)
        genesis_preimage = b"genesis:" + manifest_hash.encode('utf-8')
        head_0 = hashlib.sha256(genesis_preimage).hexdigest()
        return cls(head=head_0, step=0, digests=[])

    def append(self, row: TraceRow) -> str:
        """Append row to chain, return new head.

        Computes: d_t = H(row_t), head_{t+1} = H(head_t || d_t)
        """
        d_t = row.compute_digest()
        self.digests.append(d_t)

        # head_{t+1} = H(head_t || d_t)
        preimage = self.head.encode('utf-8') + b":" + d_t.encode('utf-8')
        self.head = hashlib.sha256(preimage).hexdigest()
        self.step += 1

        return d_t

    def get_checkpoint_digests(self) -> list[str]:
        """Get digests since last checkpoint, then clear."""
        digests = self.digests.copy()
        self.digests = []
        return digests


# =============================================================================
# CHECKPOINT RECEIPT - Emitted every K steps
# =============================================================================

@dataclass
class CheckpointReceipt:
    """Checkpoint receipt vertex in the verification DAG.

    Every K steps, emit this small receipt object binding:
    - manifest_hash: run configuration
    - policy_hash: governance rules
    - trace_anchor: checkpoint hash + step range
    - outputs_hash: feature shard hash for this chunk
    - verifier_method_id: "replay" now, "stark_v1" later
    """
    schema: str = CHECKPOINT_SCHEMA_VERSION
    manifest_hash: str = ""
    policy_hash: str = ""
    inputs_hash: str = ""  # seed commitment + dataset hash

    # Trace anchor
    checkpoint_index: int = 0
    step_start: int = 0
    step_end: int = 0  # exclusive
    head_at_start: str = ""  # head_{step_start}
    head_at_end: str = ""    # head_{step_end}

    # Merkle commitment to rows in this chunk
    chunk_root: str = ""
    row_digests: list[str] = field(default_factory=list)

    # Output binding
    outputs_hash: str = ""  # hash of features emitted in this chunk

    # Verifier contract
    verifier_method_id: str = "replay_v1"  # "replay_v1" now, "stark_v1" later

    # Timestamp
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def compute_receipt_hash(self) -> str:
        """Hash of the receipt itself."""
        return hash_canonical(asdict(self))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["receipt_hash"] = self.compute_receipt_hash()
        return d

    def save(self, path: Path) -> None:
        """Save checkpoint receipt to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'CheckpointReceipt':
        """Load checkpoint receipt from JSON."""
        with open(path) as f:
            data = json.load(f)

        stored_hash = data.pop("receipt_hash", "")
        receipt = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

        # Verify hash if present
        if stored_hash:
            computed = receipt.compute_receipt_hash()
            if computed != stored_hash:
                raise ValueError(
                    f"Checkpoint receipt hash mismatch: stored={stored_hash[:16]}..., "
                    f"computed={computed[:16]}..."
                )

        return receipt


def compute_merkle_root(digests: list[str]) -> str:
    """Compute Merkle root of digest list.

    Simple binary Merkle tree. Pads with zeros if not power of 2.
    """
    if not digests:
        return hashlib.sha256(b"empty").hexdigest()

    # Pad to power of 2
    n = len(digests)
    size = 1
    while size < n:
        size *= 2

    # Copy and pad
    leaves = digests + ["0" * 64] * (size - n)

    # Build tree bottom-up
    while len(leaves) > 1:
        next_level = []
        for i in range(0, len(leaves), 2):
            combined = leaves[i].encode('utf-8') + b":" + leaves[i + 1].encode('utf-8')
            next_level.append(hashlib.sha256(combined).hexdigest())
        leaves = next_level

    return leaves[0]


def compute_merkle_proof(digests: list[str], index: int) -> list[tuple[str, str]]:
    """Compute Merkle proof for digest at index.

    Returns list of (sibling_hash, direction) where direction is "L" or "R".
    """
    if not digests or index >= len(digests):
        return []

    # Pad to power of 2
    n = len(digests)
    size = 1
    while size < n:
        size *= 2

    leaves = digests + ["0" * 64] * (size - n)
    proof = []
    idx = index

    while len(leaves) > 1:
        next_level = []
        for i in range(0, len(leaves), 2):
            if i == idx - (idx % 2):
                # Include sibling in proof
                if idx % 2 == 0:
                    proof.append((leaves[i + 1], "R"))
                else:
                    proof.append((leaves[i], "L"))
            combined = leaves[i].encode('utf-8') + b":" + leaves[i + 1].encode('utf-8')
            next_level.append(hashlib.sha256(combined).hexdigest())
        leaves = next_level
        idx //= 2

    return proof


def verify_merkle_proof(leaf_digest: str, proof: list[tuple[str, str]], root: str) -> bool:
    """Verify Merkle proof for a leaf."""
    current = leaf_digest
    for sibling, direction in proof:
        if direction == "L":
            combined = sibling.encode('utf-8') + b":" + current.encode('utf-8')
        else:
            combined = current.encode('utf-8') + b":" + sibling.encode('utf-8')
        current = hashlib.sha256(combined).hexdigest()
    return current == root


# =============================================================================
# TRACE EMITTER - Coordinates row emission and checkpointing
# =============================================================================

class TraceEmitter:
    """Coordinates trace row emission, hash chain, and checkpoints.

    Usage:
        emitter = TraceEmitter(manifest, checkpoint_interval=1024)
        for t in range(T):
            row = TraceRow(t=t, ...)
            emitter.emit(row)
        emitter.finalize()
    """

    def __init__(
        self,
        manifest_hash: str,
        policy_hash: str = "",
        inputs_hash: str = "",
        checkpoint_interval: int = 1024,
        output_dir: Optional[Path] = None,
    ):
        self.manifest_hash = manifest_hash
        self.policy_hash = policy_hash
        self.inputs_hash = inputs_hash
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir or Path(".")

        # Hash chain state
        self.chain = HashChainState.genesis(manifest_hash)
        self.checkpoint_index = 0
        self.last_checkpoint_step = 0
        self.last_checkpoint_head = self.chain.head

        # Accumulate features for output binding
        self.chunk_features: list[list[float]] = []

        # Trace file handle (lazy open)
        self._trace_file: Optional[BinaryIO] = None
        self._commitments: list[dict] = []

    def emit(self, row: TraceRow) -> str:
        """Emit a trace row, return its digest."""
        # Append to hash chain
        d_t = self.chain.append(row)

        # Record commitment
        self._commitments.append({
            "t": row.t,
            "d_t": d_t,
            "head": self.chain.head,
        })

        # Accumulate features
        self.chunk_features.append(row.x_t)

        # Write row to trace file
        self._write_row(row)

        # Check for checkpoint
        if self.chain.step - self.last_checkpoint_step >= self.checkpoint_interval:
            self._emit_checkpoint()

        return d_t

    def _write_row(self, row: TraceRow) -> None:
        """Write row to trace.jsonl file."""
        if self._trace_file is None:
            trace_path = self.output_dir / "trace.jsonl"
            self._trace_file = open(trace_path, 'w')

        line = json.dumps(row.to_dict(), sort_keys=True, separators=(',', ':'))
        self._trace_file.write(line + "\n")

    def _emit_checkpoint(self) -> None:
        """Emit checkpoint receipt for current chunk."""
        digests = self.chain.get_checkpoint_digests()
        chunk_root = compute_merkle_root(digests)

        # Hash features for output binding
        features_data = canonical_json(self.chunk_features)
        outputs_hash = hashlib.sha256(features_data).hexdigest()

        receipt = CheckpointReceipt(
            manifest_hash=self.manifest_hash,
            policy_hash=self.policy_hash,
            inputs_hash=self.inputs_hash,
            checkpoint_index=self.checkpoint_index,
            step_start=self.last_checkpoint_step,
            step_end=self.chain.step,
            head_at_start=self.last_checkpoint_head,
            head_at_end=self.chain.head,
            chunk_root=chunk_root,
            row_digests=digests,
            outputs_hash=outputs_hash,
        )

        # Save receipt
        receipt_path = self.output_dir / f"checkpoint_{self.checkpoint_index:04d}.json"
        receipt.save(receipt_path)

        # Update state
        self.checkpoint_index += 1
        self.last_checkpoint_step = self.chain.step
        self.last_checkpoint_head = self.chain.head
        self.chunk_features = []

    def finalize(self) -> dict:
        """Finalize trace emission, emit final checkpoint if needed."""
        # Emit partial checkpoint if any rows remain
        if self.chain.step > self.last_checkpoint_step:
            self._emit_checkpoint()

        # Close trace file
        if self._trace_file:
            self._trace_file.close()
            self._trace_file = None

        # Write commitments summary
        commitments_path = self.output_dir / "commitments.json"
        summary = {
            "manifest_hash": self.manifest_hash,
            "head_0": HashChainState.genesis(self.manifest_hash).head,
            "head_T": self.chain.head,
            "total_steps": self.chain.step,
            "total_checkpoints": self.checkpoint_index,
            "commitments": self._commitments,
        }
        with open(commitments_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary

    def get_final_head(self) -> str:
        """Get current chain head."""
        return self.chain.head


# =============================================================================
# REPLAY VERIFIER - Verify trace matches commitments
# =============================================================================

def verify_trace_against_commitments(
    trace_path: Path,
    commitments_path: Path,
) -> tuple[bool, str]:
    """Verify trace.jsonl matches commitments.json.

    This is the core tamper-evidence check: recompute head_T from trace rows.
    """
    # Load commitments
    with open(commitments_path) as f:
        commitments = json.load(f)

    manifest_hash = commitments["manifest_hash"]
    expected_head = commitments["head_T"]
    expected_steps = commitments["total_steps"]

    # Replay hash chain
    chain = HashChainState.genesis(manifest_hash)
    step_count = 0

    with open(trace_path) as f:
        for line in f:
            if not line.strip():
                continue
            row_dict = json.loads(line)
            row = TraceRow.from_dict(row_dict)
            chain.append(row)
            step_count += 1

    if step_count != expected_steps:
        return False, f"Step count mismatch: got {step_count}, expected {expected_steps}"

    if chain.head != expected_head:
        return False, f"Head mismatch: got {chain.head}, expected {expected_head}"

    return True, f"Verified {step_count} steps, head matches"


# =============================================================================
# SELECTIVE OPENING - Reveal specific rows with Merkle proofs
# =============================================================================

@dataclass
class RowOpening:
    """A selective opening of a single trace row with Merkle proof.

    This is what you return when policy allows revealing a specific step.
    The verifier can confirm the row is part of the committed trace without
    seeing the full trace.
    """
    # The row being revealed
    row: TraceRow

    # Merkle membership proof
    merkle_proof: list[tuple[str, str]]  # [(sibling_hash, direction), ...]
    leaf_index: int  # Index within chunk

    # Checkpoint anchor
    checkpoint_index: int
    chunk_root: str
    step_start: int
    step_end: int

    # Chain continuity (for cross-checkpoint verification)
    head_at_start: str
    head_at_end: str

    # Metadata
    manifest_hash: str

    def to_dict(self) -> dict:
        return {
            "row": self.row.to_dict(),
            "merkle_proof": self.merkle_proof,
            "leaf_index": self.leaf_index,
            "checkpoint_index": self.checkpoint_index,
            "chunk_root": self.chunk_root,
            "step_start": self.step_start,
            "step_end": self.step_end,
            "head_at_start": self.head_at_start,
            "head_at_end": self.head_at_end,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RowOpening':
        return cls(
            row=TraceRow.from_dict(d["row"]),
            merkle_proof=[(s, dir) for s, dir in d["merkle_proof"]],
            leaf_index=d["leaf_index"],
            checkpoint_index=d["checkpoint_index"],
            chunk_root=d["chunk_root"],
            step_start=d["step_start"],
            step_end=d["step_end"],
            head_at_start=d["head_at_start"],
            head_at_end=d["head_at_end"],
            manifest_hash=d["manifest_hash"],
        )


def open_row(
    step: int,
    trace_path: Path,
    checkpoints_dir: Path,
) -> RowOpening:
    """Open a specific row with Merkle membership proof.

    Args:
        step: The timestep to open (0-indexed)
        trace_path: Path to trace.jsonl
        checkpoints_dir: Path to directory containing checkpoint_*.json files

    Returns:
        RowOpening with the row and its Merkle proof

    Raises:
        ValueError: If step not found or checkpoint missing
    """
    # Load all checkpoints
    checkpoints = []
    for cp_path in sorted(checkpoints_dir.glob("checkpoint_*.json")):
        checkpoints.append(CheckpointReceipt.load(cp_path))

    if not checkpoints:
        raise ValueError("No checkpoints found")

    # Find which checkpoint contains this step
    target_cp = None
    for cp in checkpoints:
        if cp.step_start <= step < cp.step_end:
            target_cp = cp
            break

    if target_cp is None:
        raise ValueError(f"Step {step} not found in any checkpoint")

    # Load the specific row from trace
    target_row = None
    with open(trace_path) as f:
        for line in f:
            if not line.strip():
                continue
            row_dict = json.loads(line)
            row = TraceRow.from_dict(row_dict)
            if row.t == step:
                target_row = row
                break

    if target_row is None:
        raise ValueError(f"Row for step {step} not found in trace")

    # Compute leaf index within chunk
    leaf_index = step - target_cp.step_start

    # Compute Merkle proof
    merkle_proof = compute_merkle_proof(target_cp.row_digests, leaf_index)

    return RowOpening(
        row=target_row,
        merkle_proof=merkle_proof,
        leaf_index=leaf_index,
        checkpoint_index=target_cp.checkpoint_index,
        chunk_root=target_cp.chunk_root,
        step_start=target_cp.step_start,
        step_end=target_cp.step_end,
        head_at_start=target_cp.head_at_start,
        head_at_end=target_cp.head_at_end,
        manifest_hash=target_cp.manifest_hash,
    )


def verify_opening(
    opening: RowOpening,
    checkpoint_receipt: Optional[CheckpointReceipt] = None,
) -> tuple[bool, str]:
    """Verify a row opening is valid.

    Checks:
    1. Row digest matches the leaf in the Merkle tree
    2. Merkle proof is valid against chunk_root
    3. If checkpoint provided, chunk_root matches

    Args:
        opening: The RowOpening to verify
        checkpoint_receipt: Optional checkpoint to cross-check

    Returns:
        (success, message) tuple
    """
    # Step 1: Compute row digest
    row_digest = opening.row.compute_digest()

    # Step 2: Verify Merkle proof
    if not verify_merkle_proof(row_digest, opening.merkle_proof, opening.chunk_root):
        return False, f"Merkle proof invalid for row t={opening.row.t}"

    # Step 3: Cross-check against checkpoint if provided
    if checkpoint_receipt is not None:
        if opening.chunk_root != checkpoint_receipt.chunk_root:
            return False, (
                f"Chunk root mismatch: opening={opening.chunk_root[:16]}..., "
                f"checkpoint={checkpoint_receipt.chunk_root[:16]}..."
            )

        if opening.manifest_hash != checkpoint_receipt.manifest_hash:
            return False, "Manifest hash mismatch"

        if opening.checkpoint_index != checkpoint_receipt.checkpoint_index:
            return False, "Checkpoint index mismatch"

    return True, f"Row t={opening.row.t} verified (leaf_index={opening.leaf_index})"


def verify_opening_chain(
    openings: list[RowOpening],
    checkpoints: list[CheckpointReceipt],
) -> tuple[bool, str]:
    """Verify multiple openings form a consistent chain.

    This checks that:
    1. Each opening is individually valid
    2. Checkpoint chain is continuous (head_at_end connects to next head_at_start)
    3. All openings reference the same manifest

    Args:
        openings: List of RowOpening objects (can span multiple checkpoints)
        checkpoints: List of CheckpointReceipt objects

    Returns:
        (success, message) tuple
    """
    if not openings:
        return True, "No openings to verify"

    # Build checkpoint lookup
    cp_by_index = {cp.checkpoint_index: cp for cp in checkpoints}

    # Verify each opening
    manifest_hash = openings[0].manifest_hash
    for opening in openings:
        # Check manifest consistency
        if opening.manifest_hash != manifest_hash:
            return False, f"Manifest mismatch at step {opening.row.t}"

        # Find checkpoint
        cp = cp_by_index.get(opening.checkpoint_index)
        if cp is None:
            return False, f"Checkpoint {opening.checkpoint_index} not found"

        # Verify opening against checkpoint
        valid, msg = verify_opening(opening, cp)
        if not valid:
            return False, msg

    # Verify checkpoint chain continuity
    sorted_cps = sorted(cp_by_index.values(), key=lambda c: c.checkpoint_index)
    for i in range(len(sorted_cps) - 1):
        if sorted_cps[i].head_at_end != sorted_cps[i + 1].head_at_start:
            return False, (
                f"Checkpoint chain broken between {sorted_cps[i].checkpoint_index} "
                f"and {sorted_cps[i + 1].checkpoint_index}"
            )

    return True, f"Verified {len(openings)} openings across {len(checkpoints)} checkpoints"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "RandRecord",
    "TraceRow",
    "HashChainState",
    "CheckpointReceipt",
    "compute_merkle_root",
    "compute_merkle_proof",
    "verify_merkle_proof",
    "TraceEmitter",
    "verify_trace_against_commitments",
    "RowOpening",
    "open_row",
    "verify_opening",
    "verify_opening_chain",
]
