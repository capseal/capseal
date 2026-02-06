"""AgentAIR -- 14-element row encoding for general-purpose agent action traces.

Each row represents one discrete action taken by an agent. The 14 Goldilocks
field elements encode action type, hashes, chain provenance, and status flags.

Element  Field                    Encoding
-------- ------------------------ ---------------------------------
0        action_index             Sequential (0, 1, 2, ...)
1        action_type_hash_lo      SHA256(action_type) lo -- categorizes the action
2        action_type_hash_hi      SHA256(action_type) hi
3        instruction_hash_lo      What triggered this action (lo)
4        instruction_hash_hi      (hi)
5        input_hash_lo            Action inputs (lo)
6        input_hash_hi            (hi)
7        output_hash_lo           Action outputs (lo)
8        output_hash_hi           (hi)
9        prev_receipt_hash_lo     Parent action's receipt (lo), 0 for first
10       prev_receipt_hash_hi     (hi)
11       receipt_hash_lo          This action's receipt (lo)
12       receipt_hash_hi          (hi)
13       status_flags             Bitfield: bit0=success, bit1=gate_passed, bit2=gate_evaluated,
                                  bit3=policy_met, bit4=is_tool_call, bit5=is_code_gen
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bef_zk.capsule.agent_protocol import AgentAction

# Goldilocks prime: p = 2^64 - 2^32 + 1
GOLDILOCKS_P = (1 << 64) - (1 << 32) + 1

# AIR parameters
AGENT_AIR_ROW_WIDTH = 14
AGENT_AIR_ID = "agent_air_v1"

# Status flag bit positions
STATUS_BIT_SUCCESS = 0        # bit0: 1=success, 0=failure
STATUS_BIT_GATE_PASSED = 1    # bit1: 1=gate passed (decision != "skip")
STATUS_BIT_GATE_EVALUATED = 2 # bit2: 1=gate was evaluated (gate_score is not None)
STATUS_BIT_POLICY_MET = 3     # bit3: 1=policy verdict is not None
STATUS_BIT_IS_TOOL_CALL = 4   # bit4: 1=action_type == "tool_call"
STATUS_BIT_IS_CODE_GEN = 5    # bit5: 1=action_type == "code_gen"


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


def sha256_str(s: str) -> str:
    """Compute SHA256 of a string and return hex digest."""
    return hashlib.sha256(s.encode()).hexdigest()


def compute_status_flags(
    success: bool,
    gate_passed: bool,
    gate_evaluated: bool,
    policy_met: bool,
    is_tool_call: bool,
    is_code_gen: bool,
) -> int:
    """Compute the status_flags bitfield from individual flags.

    Args:
        success: Action executed successfully
        gate_passed: Gate decision was not "skip"
        gate_evaluated: Gate score is not None
        policy_met: Policy verdict is not None
        is_tool_call: action_type == "tool_call"
        is_code_gen: action_type == "code_gen"

    Returns:
        Integer bitfield with flags encoded
    """
    flags = 0
    if success:
        flags |= (1 << STATUS_BIT_SUCCESS)
    if gate_passed:
        flags |= (1 << STATUS_BIT_GATE_PASSED)
    if gate_evaluated:
        flags |= (1 << STATUS_BIT_GATE_EVALUATED)
    if policy_met:
        flags |= (1 << STATUS_BIT_POLICY_MET)
    if is_tool_call:
        flags |= (1 << STATUS_BIT_IS_TOOL_CALL)
    if is_code_gen:
        flags |= (1 << STATUS_BIT_IS_CODE_GEN)
    return flags


def decode_status_flags(flags: int) -> dict[str, bool]:
    """Decode status_flags bitfield to individual flags.

    Args:
        flags: Integer bitfield

    Returns:
        Dict with boolean flag values
    """
    return {
        "success": bool(flags & (1 << STATUS_BIT_SUCCESS)),
        "gate_passed": bool(flags & (1 << STATUS_BIT_GATE_PASSED)),
        "gate_evaluated": bool(flags & (1 << STATUS_BIT_GATE_EVALUATED)),
        "policy_met": bool(flags & (1 << STATUS_BIT_POLICY_MET)),
        "is_tool_call": bool(flags & (1 << STATUS_BIT_IS_TOOL_CALL)),
        "is_code_gen": bool(flags & (1 << STATUS_BIT_IS_CODE_GEN)),
    }


def compute_action_receipt_hash(action: "AgentAction") -> str:
    """Compute the receipt hash for an AgentAction.

    This is a convenience wrapper around action.compute_receipt_hash().

    Args:
        action: The AgentAction to hash

    Returns:
        64-character hex string (SHA256)
    """
    return action.compute_receipt_hash()


def encode_agent_action_row(
    action: "AgentAction",
    action_index: int,
) -> list[int]:
    """Encode an AgentAction as a 14-element Goldilocks row.

    Args:
        action: The AgentAction to encode
        action_index: Sequential index in the trace (0, 1, 2, ...)

    Returns:
        List of 14 integers, each < GOLDILOCKS_P
    """
    # Element 0: action_index
    e0_action_index = action_index % GOLDILOCKS_P

    # Elements 1-2: action_type_hash (SHA256 of action_type)
    action_type_hash = sha256_str(action.action_type)
    e1_type_lo, e2_type_hi = sha256_to_field_pair(action_type_hash)

    # Elements 3-4: instruction_hash
    e3_instruction_lo, e4_instruction_hi = sha256_to_field_pair(action.instruction_hash)

    # Elements 5-6: input_hash
    e5_input_lo, e6_input_hi = sha256_to_field_pair(action.input_hash)

    # Elements 7-8: output_hash
    e7_output_lo, e8_output_hi = sha256_to_field_pair(action.output_hash)

    # Elements 9-10: prev_receipt_hash (0 for first action)
    if action.parent_receipt_hash:
        e9_prev_lo, e10_prev_hi = sha256_to_field_pair(action.parent_receipt_hash)
    else:
        e9_prev_lo, e10_prev_hi = 0, 0

    # Elements 11-12: receipt_hash (this action's receipt)
    receipt_hash = action.compute_receipt_hash()
    e11_receipt_lo, e12_receipt_hi = sha256_to_field_pair(receipt_hash)

    # Element 13: status_flags
    gate_passed = action.gate_decision != "skip" if action.gate_decision else True
    gate_evaluated = action.gate_score is not None
    policy_met = action.policy_verdict is not None
    is_tool_call = action.action_type == "tool_call"
    is_code_gen = action.action_type == "code_gen"

    e13_flags = compute_status_flags(
        success=action.success,
        gate_passed=gate_passed,
        gate_evaluated=gate_evaluated,
        policy_met=policy_met,
        is_tool_call=is_tool_call,
        is_code_gen=is_code_gen,
    )

    row = [
        e0_action_index,
        e1_type_lo,
        e2_type_hi,
        e3_instruction_lo,
        e4_instruction_hi,
        e5_input_lo,
        e6_input_hi,
        e7_output_lo,
        e8_output_hi,
        e9_prev_lo,
        e10_prev_hi,
        e11_receipt_lo,
        e12_receipt_hi,
        e13_flags,
    ]

    # Verify all elements are within field bounds
    assert all(0 <= v < GOLDILOCKS_P for v in row), "Row element exceeds field modulus"
    assert len(row) == AGENT_AIR_ROW_WIDTH, f"Row width mismatch: {len(row)} != {AGENT_AIR_ROW_WIDTH}"

    return row


@dataclass
class DecodedAgentRow:
    """Decoded agent AIR row with human-readable fields."""
    action_index: int
    action_type_hash_partial: str  # First 128 bits only
    instruction_hash_partial: str
    input_hash_partial: str
    output_hash_partial: str
    prev_receipt_hash_partial: str
    receipt_hash_partial: str
    status_flags: dict[str, bool]

    # Raw field elements for constraint checking
    raw: list[int]


def decode_agent_action_row(row: list[int]) -> DecodedAgentRow:
    """Decode a 14-element Goldilocks row back to readable form.

    Note: Hash values are only partial (128 bits) due to field encoding.
    This is sufficient for verification but not for full reconstruction.

    Args:
        row: List of 14 integers (field elements)

    Returns:
        DecodedAgentRow with human-readable fields
    """
    if len(row) != AGENT_AIR_ROW_WIDTH:
        raise ValueError(f"Row width mismatch: {len(row)} != {AGENT_AIR_ROW_WIDTH}")

    return DecodedAgentRow(
        action_index=row[0],
        action_type_hash_partial=field_pair_to_hex(row[1], row[2]),
        instruction_hash_partial=field_pair_to_hex(row[3], row[4]),
        input_hash_partial=field_pair_to_hex(row[5], row[6]),
        output_hash_partial=field_pair_to_hex(row[7], row[8]),
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


def build_agent_row_matrix(
    actions: list["AgentAction"],
) -> list[list[int]]:
    """Build the full row matrix from a list of actions.

    Args:
        actions: List of AgentActions in order of execution

    Returns:
        List of 14-element rows, one per action
    """
    rows = []
    for i, action in enumerate(actions):
        row = encode_agent_action_row(action, action_index=i)
        rows.append(row)
    return rows
