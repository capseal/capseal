"""EvalAIR -- 14-element row encoding for epistemic evaluation loop traces.

Each row represents one round of the eval loop. The 14 Goldilocks field elements
encode posteriors hashes, metrics, and status for each round.

Element  Field                    Encoding
-------- ------------------------ ---------------------------------
0        round_index              Sequential (0, 1, 2, ...)
1        posteriors_hash_lo       SHA256(beta_posteriors.npz after this round) lo
2        posteriors_hash_hi       SHA256(beta_posteriors.npz after this round) hi
3        prev_posteriors_hash_lo  SHA256(beta_posteriors.npz before this round) lo, 0 for round 0
4        prev_posteriors_hash_hi  SHA256(beta_posteriors.npz before this round) hi, 0 for round 0
5        plan_hash_lo             SHA256(plan_out.json -- selected indices + scores) lo
6        plan_hash_hi             SHA256(plan_out.json) hi
7        results_hash_lo          SHA256(agent_results.csv) lo
8        results_hash_hi          SHA256(agent_results.csv) hi
9        n_successes              Count of successful episodes this round (as field element)
10       n_failures               Count of failed episodes this round
11       tube_var_scaled          tube_var_sum * 10^6, rounded to int (preserve 6 decimal places)
12       tube_coverage_scaled     tube_coverage * 10^6, rounded to int
13       status_flags             Bitfield: bit0=improved, bit1=regressed, bit2=first_round, bit3=receipts_valid
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Goldilocks prime: p = 2^64 - 2^32 + 1
GOLDILOCKS_P = (1 << 64) - (1 << 32) + 1

# AIR parameters
EVAL_AIR_ROW_WIDTH = 14
EVAL_AIR_ID = "eval_air_v1"

# Scaling factor for floating point metrics
METRIC_SCALE = 1_000_000  # 10^6

# Status flag bit positions
STATUS_BIT_IMPROVED = 0       # bit0: 1=tube_var improved (decreased)
STATUS_BIT_REGRESSED = 1      # bit1: 1=tube_var regressed (increased)
STATUS_BIT_FIRST_ROUND = 2    # bit2: 1=first round (no previous posteriors)
STATUS_BIT_RECEIPTS_VALID = 3 # bit3: 1=all receipts in this round are valid


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


def hash_file(path: Path | str) -> str:
    """Compute SHA256 hash of a file.

    Args:
        path: Path to the file

    Returns:
        64-character hex string (SHA256)
    """
    path = Path(path)
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_status_flags(
    improved: bool,
    regressed: bool,
    first_round: bool,
    receipts_valid: bool,
) -> int:
    """Compute the status_flags bitfield from individual flags.

    Args:
        improved: tube_var decreased from previous round
        regressed: tube_var increased from previous round
        first_round: This is the first round (no previous posteriors)
        receipts_valid: All receipts in this round are valid

    Returns:
        Integer bitfield with flags encoded
    """
    flags = 0
    if improved:
        flags |= (1 << STATUS_BIT_IMPROVED)
    if regressed:
        flags |= (1 << STATUS_BIT_REGRESSED)
    if first_round:
        flags |= (1 << STATUS_BIT_FIRST_ROUND)
    if receipts_valid:
        flags |= (1 << STATUS_BIT_RECEIPTS_VALID)
    return flags


def decode_status_flags(flags: int) -> dict[str, bool]:
    """Decode status_flags bitfield to individual flags.

    Args:
        flags: Integer bitfield

    Returns:
        Dict with boolean flag values
    """
    return {
        "improved": bool(flags & (1 << STATUS_BIT_IMPROVED)),
        "regressed": bool(flags & (1 << STATUS_BIT_REGRESSED)),
        "first_round": bool(flags & (1 << STATUS_BIT_FIRST_ROUND)),
        "receipts_valid": bool(flags & (1 << STATUS_BIT_RECEIPTS_VALID)),
    }


def encode_eval_round_row(
    round_index: int,
    posteriors_hash: str,
    prev_posteriors_hash: str | None,
    plan_hash: str,
    results_hash: str,
    n_successes: int,
    n_failures: int,
    tube_var_sum: float,
    tube_coverage: float,
    status: str,
    receipts_valid: bool = True,
) -> list[int]:
    """Encode an eval round as a 14-element Goldilocks row.

    Args:
        round_index: Sequential round index (0, 1, 2, ...)
        posteriors_hash: SHA256 hash of beta_posteriors.npz after this round
        prev_posteriors_hash: SHA256 hash of previous round's posteriors (None for round 0)
        plan_hash: SHA256 hash of plan_out.json (active_sampling_plan.json)
        results_hash: SHA256 hash of agent_results.csv
        n_successes: Count of successful episodes this round
        n_failures: Count of failed episodes this round
        tube_var_sum: Sum of tube variances (float)
        tube_coverage: Tube coverage metric (float)
        status: Status string ("FIRST_ROUND", "IMPROVING", "WORSENING", "NO_CHANGE")
        receipts_valid: Whether all receipts in this round are valid

    Returns:
        List of 14 integers, each < GOLDILOCKS_P
    """
    # Element 0: round_index
    e0_round_index = round_index % GOLDILOCKS_P

    # Elements 1-2: posteriors_hash
    e1_posteriors_lo, e2_posteriors_hi = sha256_to_field_pair(posteriors_hash)

    # Elements 3-4: prev_posteriors_hash (0 for first round)
    if prev_posteriors_hash:
        e3_prev_lo, e4_prev_hi = sha256_to_field_pair(prev_posteriors_hash)
    else:
        e3_prev_lo, e4_prev_hi = 0, 0

    # Elements 5-6: plan_hash
    e5_plan_lo, e6_plan_hi = sha256_to_field_pair(plan_hash)

    # Elements 7-8: results_hash
    e7_results_lo, e8_results_hi = sha256_to_field_pair(results_hash)

    # Elements 9-10: n_successes, n_failures
    e9_successes = n_successes % GOLDILOCKS_P
    e10_failures = n_failures % GOLDILOCKS_P

    # Element 11: tube_var_scaled (multiply by 10^6 and round)
    tube_var_scaled = int(round(tube_var_sum * METRIC_SCALE)) % GOLDILOCKS_P
    e11_tube_var = tube_var_scaled

    # Element 12: tube_coverage_scaled (multiply by 10^6 and round)
    tube_coverage_scaled = int(round(tube_coverage * METRIC_SCALE)) % GOLDILOCKS_P
    e12_tube_coverage = tube_coverage_scaled

    # Element 13: status_flags
    first_round = status == "FIRST_ROUND" or round_index == 0
    improved = status == "IMPROVING"
    regressed = status == "WORSENING"
    e13_flags = compute_status_flags(improved, regressed, first_round, receipts_valid)

    row = [
        e0_round_index,
        e1_posteriors_lo,
        e2_posteriors_hi,
        e3_prev_lo,
        e4_prev_hi,
        e5_plan_lo,
        e6_plan_hi,
        e7_results_lo,
        e8_results_hi,
        e9_successes,
        e10_failures,
        e11_tube_var,
        e12_tube_coverage,
        e13_flags,
    ]

    # Verify all elements are within field bounds
    assert all(0 <= v < GOLDILOCKS_P for v in row), "Row element exceeds field modulus"
    assert len(row) == EVAL_AIR_ROW_WIDTH, f"Row width mismatch: {len(row)} != {EVAL_AIR_ROW_WIDTH}"

    return row


@dataclass
class DecodedEvalRow:
    """Decoded eval AIR row with human-readable fields."""
    round_index: int
    posteriors_hash_partial: str  # First 128 bits only
    prev_posteriors_hash_partial: str
    plan_hash_partial: str
    results_hash_partial: str
    n_successes: int
    n_failures: int
    tube_var_sum: float       # Descaled back to float
    tube_coverage: float      # Descaled back to float
    status_flags: dict[str, bool]

    # Raw field elements for constraint checking
    raw: list[int]


def decode_eval_round_row(row: list[int]) -> DecodedEvalRow:
    """Decode a 14-element Goldilocks row back to readable form.

    Note: Hash values are only partial (128 bits) due to field encoding.
    This is sufficient for verification but not for full reconstruction.

    Args:
        row: List of 14 integers (field elements)

    Returns:
        DecodedEvalRow with human-readable fields
    """
    if len(row) != EVAL_AIR_ROW_WIDTH:
        raise ValueError(f"Row width mismatch: {len(row)} != {EVAL_AIR_ROW_WIDTH}")

    # Descale the metrics back to floats
    tube_var_sum = row[11] / METRIC_SCALE
    tube_coverage = row[12] / METRIC_SCALE

    return DecodedEvalRow(
        round_index=row[0],
        posteriors_hash_partial=field_pair_to_hex(row[1], row[2]),
        prev_posteriors_hash_partial=field_pair_to_hex(row[3], row[4]),
        plan_hash_partial=field_pair_to_hex(row[5], row[6]),
        results_hash_partial=field_pair_to_hex(row[7], row[8]),
        n_successes=row[9],
        n_failures=row[10],
        tube_var_sum=tube_var_sum,
        tube_coverage=tube_coverage,
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
