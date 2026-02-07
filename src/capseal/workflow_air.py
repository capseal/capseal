"""WorkflowAIR — 14-element row encoding for DAG node execution traces.

Each row represents one node's execution in the workflow DAG. The 14 Goldilocks
field elements encode provenance, hashes, and status for each node.

Element  Field                    Encoding
──────── ──────────────────────── ─────────────────────────────────
0        node_index               Sequential index in topo order (0, 1, 2, ...)
1        node_kind_hash_lo        Lower 64 bits of SHA256(executor_id), truncated to Goldilocks
2        node_kind_hash_hi        Upper 64 bits of SHA256(executor_id), truncated to Goldilocks
3        input_hash_lo            Lower 64 bits of input_hash
4        input_hash_hi            Upper 64 bits of input_hash
5        output_hash_lo           Lower 64 bits of output_hash
6        output_hash_hi           Upper 64 bits of output_hash
7        policy_hash_lo           Lower 64 bits of policy_hash
8        policy_hash_hi           Upper 64 bits of policy_hash
9        prev_receipt_hash_lo     Lower 64 bits of the previous node's receipt_hash (0 for root)
10       prev_receipt_hash_hi     Upper 64 bits of same
11       receipt_hash_lo          Lower 64 bits of this node's compute_receipt_hash()
12       receipt_hash_hi          Upper 64 bits of same
13       status_flags             Bitfield: bit0=pass/fail, bit1=deterministic, bit2=policy_met,
                                  bit3=gate_approved
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from capseal.workflow_engine import AgentPacket

# Goldilocks prime: p = 2^64 - 2^32 + 1
GOLDILOCKS_P = (1 << 64) - (1 << 32) + 1

# AIR parameters
WORKFLOW_AIR_ROW_WIDTH = 14
WORKFLOW_AIR_ID = "workflow_air_v1"

# Status flag bit positions
STATUS_BIT_PASS = 0          # bit0: 1=pass, 0=fail
STATUS_BIT_DETERMINISTIC = 1  # bit1: 1=deterministic, 0=non-deterministic
STATUS_BIT_POLICY_MET = 2     # bit2: 1=policy constraints met
STATUS_BIT_GATE_APPROVED = 3  # bit3: 1=gate approved


def sha256_to_field_pair(hex_hash: str) -> tuple[int, int]:
    """Convert SHA256 hex string to two Goldilocks field elements.

    Takes the first 8 bytes as little-endian u64, reduces mod p -> lo.
    Takes bytes 8-16 as little-endian u64, reduces mod p -> hi.

    We lose some bits but retain 128 bits of collision resistance per hash,
    which is sufficient for our security requirements.

    Args:
        hex_hash: 64-character hex string (SHA256 output)

    Returns:
        Tuple of (lo, hi) field elements, each < GOLDILOCKS_P
    """
    if not hex_hash:
        return (0, 0)

    # Handle both with and without 0x prefix
    if hex_hash.startswith("0x"):
        hex_hash = hex_hash[2:]

    # Pad if necessary (shouldn't happen for SHA256)
    if len(hex_hash) < 32:
        hex_hash = hex_hash.zfill(32)

    raw = bytes.fromhex(hex_hash)

    # Take first 8 bytes as little-endian u64
    lo = int.from_bytes(raw[0:8], 'little') % GOLDILOCKS_P
    # Take bytes 8-16 as little-endian u64
    hi = int.from_bytes(raw[8:16], 'little') % GOLDILOCKS_P

    return (lo, hi)


def field_pair_to_hex(lo: int, hi: int) -> str:
    """Convert two Goldilocks field elements back to partial hex.

    This is a lossy operation since we truncated during encoding.
    The returned hex is only the first 16 bytes (128 bits).

    Args:
        lo: Lower 64 bits field element
        hi: Upper 64 bits field element

    Returns:
        32-character hex string (first 128 bits)
    """
    lo_bytes = lo.to_bytes(8, 'little')
    hi_bytes = hi.to_bytes(8, 'little')
    return (lo_bytes + hi_bytes).hex()


def compute_status_flags(
    passed: bool,
    deterministic: bool,
    policy_met: bool,
    gate_approved: bool,
) -> int:
    """Compute the status_flags bitfield from individual flags.

    Args:
        passed: Node executed successfully (not skipped or failed)
        deterministic: Node is deterministic (same inputs -> same outputs)
        policy_met: Policy constraints were satisfied
        gate_approved: Gate check approved (if applicable)

    Returns:
        Integer bitfield with flags encoded
    """
    flags = 0
    if passed:
        flags |= (1 << STATUS_BIT_PASS)
    if deterministic:
        flags |= (1 << STATUS_BIT_DETERMINISTIC)
    if policy_met:
        flags |= (1 << STATUS_BIT_POLICY_MET)
    if gate_approved:
        flags |= (1 << STATUS_BIT_GATE_APPROVED)
    return flags


def decode_status_flags(flags: int) -> dict[str, bool]:
    """Decode status_flags bitfield to individual flags.

    Args:
        flags: Integer bitfield

    Returns:
        Dict with boolean flag values
    """
    return {
        "passed": bool(flags & (1 << STATUS_BIT_PASS)),
        "deterministic": bool(flags & (1 << STATUS_BIT_DETERMINISTIC)),
        "policy_met": bool(flags & (1 << STATUS_BIT_POLICY_MET)),
        "gate_approved": bool(flags & (1 << STATUS_BIT_GATE_APPROVED)),
    }


def sha256_str(s: str) -> str:
    """Compute SHA256 of a string and return hex digest."""
    return hashlib.sha256(s.encode()).hexdigest()


def encode_agent_packet_row(
    packet: "AgentPacket",
    node_index: int,
    prev_receipt_hash: str | None,
    policy_met: bool,
    gate_approved: bool,
) -> list[int]:
    """Encode an AgentPacket as a 14-element Goldilocks row.

    Args:
        packet: The AgentPacket from a workflow node execution
        node_index: Sequential index in topological order (0, 1, 2, ...)
        prev_receipt_hash: Receipt hash of the previous node (None for root)
        policy_met: Whether policy constraints were satisfied
        gate_approved: Whether gate check approved this node

    Returns:
        List of 14 integers, each < GOLDILOCKS_P
    """
    # Element 0: node_index
    e0_node_index = node_index % GOLDILOCKS_P

    # Elements 1-2: node_kind_hash (SHA256 of executor_id)
    executor_hash = sha256_str(packet.executor_id) if packet.executor_id else ""
    e1_kind_lo, e2_kind_hi = sha256_to_field_pair(executor_hash)

    # Elements 3-4: input_hash
    e3_input_lo, e4_input_hi = sha256_to_field_pair(packet.input_hash)

    # Elements 5-6: output_hash
    e5_output_lo, e6_output_hi = sha256_to_field_pair(packet.output_hash)

    # Elements 7-8: policy_hash
    e7_policy_lo, e8_policy_hi = sha256_to_field_pair(packet.policy_hash)

    # Elements 9-10: prev_receipt_hash (0 for root nodes)
    if prev_receipt_hash:
        e9_prev_lo, e10_prev_hi = sha256_to_field_pair(prev_receipt_hash)
    else:
        e9_prev_lo, e10_prev_hi = 0, 0

    # Elements 11-12: receipt_hash (this node's receipt)
    receipt_hash = packet.compute_receipt_hash()
    e11_receipt_lo, e12_receipt_hi = sha256_to_field_pair(receipt_hash)

    # Element 13: status_flags
    passed = True  # Packet exists means node passed
    deterministic = packet.determinism == "deterministic"
    e13_flags = compute_status_flags(passed, deterministic, policy_met, gate_approved)

    row = [
        e0_node_index,
        e1_kind_lo,
        e2_kind_hi,
        e3_input_lo,
        e4_input_hi,
        e5_output_lo,
        e6_output_hi,
        e7_policy_lo,
        e8_policy_hi,
        e9_prev_lo,
        e10_prev_hi,
        e11_receipt_lo,
        e12_receipt_hi,
        e13_flags,
    ]

    # Verify all elements are within field bounds
    assert all(0 <= v < GOLDILOCKS_P for v in row), "Row element exceeds field modulus"
    assert len(row) == WORKFLOW_AIR_ROW_WIDTH, f"Row width mismatch: {len(row)} != {WORKFLOW_AIR_ROW_WIDTH}"

    return row


@dataclass
class DecodedRow:
    """Decoded workflow AIR row with human-readable fields."""
    node_index: int
    node_kind_hash_partial: str  # First 128 bits only
    input_hash_partial: str
    output_hash_partial: str
    policy_hash_partial: str
    prev_receipt_hash_partial: str
    receipt_hash_partial: str
    status_flags: dict[str, bool]

    # Raw field elements for constraint checking
    raw: list[int]


def decode_row(row: list[int]) -> DecodedRow:
    """Decode a 14-element Goldilocks row back to readable form.

    Note: Hash values are only partial (128 bits) due to field encoding.
    This is sufficient for verification but not for full reconstruction.

    Args:
        row: List of 14 integers (field elements)

    Returns:
        DecodedRow with human-readable fields
    """
    if len(row) != WORKFLOW_AIR_ROW_WIDTH:
        raise ValueError(f"Row width mismatch: {len(row)} != {WORKFLOW_AIR_ROW_WIDTH}")

    return DecodedRow(
        node_index=row[0],
        node_kind_hash_partial=field_pair_to_hex(row[1], row[2]),
        input_hash_partial=field_pair_to_hex(row[3], row[4]),
        output_hash_partial=field_pair_to_hex(row[5], row[6]),
        policy_hash_partial=field_pair_to_hex(row[7], row[8]),
        prev_receipt_hash_partial=field_pair_to_hex(row[9], row[10]),
        receipt_hash_partial=field_pair_to_hex(row[11], row[12]),
        status_flags=decode_status_flags(row[13]),
        raw=list(row),
    )


def verify_row_field_bounds(row: list[int]) -> bool:
    """Verify all elements in a row are within Goldilocks field bounds.

    Args:
        row: List of field elements

    Returns:
        True if all elements are valid field elements
    """
    return all(0 <= v < GOLDILOCKS_P for v in row)


def build_row_matrix(
    packets: list["AgentPacket"],
    policy_met_flags: list[bool] | None = None,
    gate_approved_flags: list[bool] | None = None,
) -> list[list[int]]:
    """Build the full row matrix from a list of packets in topo order.

    Args:
        packets: List of AgentPackets in topological order
        policy_met_flags: Per-packet policy satisfaction flags (default all True)
        gate_approved_flags: Per-packet gate approval flags (default all True)

    Returns:
        List of 14-element rows, one per packet
    """
    if policy_met_flags is None:
        policy_met_flags = [True] * len(packets)
    if gate_approved_flags is None:
        gate_approved_flags = [True] * len(packets)

    rows = []
    prev_receipt = None

    for i, packet in enumerate(packets):
        row = encode_agent_packet_row(
            packet=packet,
            node_index=i,
            prev_receipt_hash=prev_receipt,
            policy_met=policy_met_flags[i],
            gate_approved=gate_approved_flags[i],
        )
        rows.append(row)
        prev_receipt = packet.compute_receipt_hash()

    return rows
