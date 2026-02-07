"""AIR constraints for AgentAIR.

The FRI proof enforces these constraints to prove agent execution integrity.

Constraint Types:
- Transition constraints: Between consecutive actions
- Boundary constraints: On specific rows (first/last)
- Per-row constraints: On each row individually

Transition Constraints:
  1. Chain constraint: Row[i].receipt_hash == Row[i+1].prev_receipt_hash
     Proves the action chain is unbroken.
     Encoded as: row_next[9] - row_curr[11] == 0 (lo)
                 row_next[10] - row_curr[12] == 0 (hi)

  2. Ordering constraint: Row[i+1].action_index == Row[i].action_index + 1
     Proves actions are in sequential order.
     Encoded as: row_next[0] - row_curr[0] - 1 == 0

Boundary Constraints:
  1. First row: prev_receipt_hash == 0 (no previous action for first action)
  2. Last row: receipt_hash must equal the declared final receipt hash

Per-Row Constraints:
  1. All field elements < GOLDILOCKS_P (range check)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from capseal.agent_air import (
    GOLDILOCKS_P,
    AGENT_AIR_ROW_WIDTH,
    sha256_to_field_pair,
)


@dataclass
class ConstraintResult:
    """Result of checking a single constraint."""
    name: str
    satisfied: bool
    expected: Any = None
    actual: Any = None
    message: str = ""


@dataclass
class AgentConstraints:
    """Container for agent AIR constraint definitions.

    These constraints are designed to be fed into the FRI prover.
    The prover will generate a proof that these constraints hold over
    the committed trace.
    """

    # Number of transition constraints (between consecutive rows)
    NUM_TRANSITION_CONSTRAINTS = 3  # chain_lo, chain_hi, ordering

    # Number of boundary constraints
    NUM_BOUNDARY_CONSTRAINTS = 4  # first_prev_lo, first_prev_hi, last_receipt_lo, last_receipt_hi

    # Constraint identifiers for composition
    CONSTRAINT_NAMES = [
        "chain_lo",           # receipt chain continuity (lo)
        "chain_hi",           # receipt chain continuity (hi)
        "ordering",           # sequential action ordering
        "first_prev_lo",      # first row prev_receipt is 0 (lo)
        "first_prev_hi",      # first row prev_receipt is 0 (hi)
        "last_receipt_lo",    # last row receipt matches declared (lo)
        "last_receipt_hi",    # last row receipt matches declared (hi)
    ]

    @staticmethod
    def eval_transition_chain_lo(row_curr: list[int], row_next: list[int]) -> int:
        """Evaluate chain constraint (lo): row_next[9] - row_curr[11] == 0.

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        # row[11] = receipt_hash_lo of current
        # row[9] = prev_receipt_hash_lo of next
        return (row_next[9] - row_curr[11]) % GOLDILOCKS_P

    @staticmethod
    def eval_transition_chain_hi(row_curr: list[int], row_next: list[int]) -> int:
        """Evaluate chain constraint (hi): row_next[10] - row_curr[12] == 0.

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        # row[12] = receipt_hash_hi of current
        # row[10] = prev_receipt_hash_hi of next
        return (row_next[10] - row_curr[12]) % GOLDILOCKS_P

    @staticmethod
    def eval_transition_ordering(row_curr: list[int], row_next: list[int]) -> int:
        """Evaluate ordering constraint: row_next[0] - row_curr[0] - 1 == 0.

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        # row[0] = action_index
        return (row_next[0] - row_curr[0] - 1) % GOLDILOCKS_P

    @staticmethod
    def eval_boundary_first_prev_lo(row_first: list[int]) -> int:
        """Evaluate first row constraint: row[9] == 0 (prev_receipt_lo).

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        return row_first[9] % GOLDILOCKS_P

    @staticmethod
    def eval_boundary_first_prev_hi(row_first: list[int]) -> int:
        """Evaluate first row constraint: row[10] == 0 (prev_receipt_hi).

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        return row_first[10] % GOLDILOCKS_P

    @staticmethod
    def eval_boundary_last_receipt_lo(row_last: list[int], final_receipt_lo: int) -> int:
        """Evaluate last row constraint: row[11] == final_receipt_lo.

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        return (row_last[11] - final_receipt_lo) % GOLDILOCKS_P

    @staticmethod
    def eval_boundary_last_receipt_hi(row_last: list[int], final_receipt_hi: int) -> int:
        """Evaluate last row constraint: row[12] == final_receipt_hi.

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        return (row_last[12] - final_receipt_hi) % GOLDILOCKS_P

    @staticmethod
    def check_field_bounds(row: list[int]) -> bool:
        """Check that all field elements are < GOLDILOCKS_P."""
        return all(0 <= v < GOLDILOCKS_P for v in row)


def verify_agent_trace(
    rows: list[list[int]],
    final_receipt_hash: str,
) -> tuple[bool, list[ConstraintResult]]:
    """Verify that an agent trace satisfies all AIR constraints.

    This is the verification logic that the FRI proof proves.

    Args:
        rows: List of 14-element rows (trace matrix)
        final_receipt_hash: SHA256 hex hash that the last row's receipt must match

    Returns:
        Tuple of (all_satisfied, list of constraint results)
    """
    results: list[ConstraintResult] = []

    if not rows:
        return False, [ConstraintResult(
            name="trace_non_empty",
            satisfied=False,
            message="Trace is empty",
        )]

    # Verify row width
    for i, row in enumerate(rows):
        if len(row) != AGENT_AIR_ROW_WIDTH:
            results.append(ConstraintResult(
                name=f"row_width_{i}",
                satisfied=False,
                expected=AGENT_AIR_ROW_WIDTH,
                actual=len(row),
                message=f"Row {i} has wrong width",
            ))

    # Verify field bounds for all rows
    for i, row in enumerate(rows):
        if not AgentConstraints.check_field_bounds(row):
            results.append(ConstraintResult(
                name=f"field_bounds_{i}",
                satisfied=False,
                message=f"Row {i} has element(s) >= GOLDILOCKS_P",
            ))

    # Boundary constraint: first row prev_receipt == 0
    first_prev_lo = AgentConstraints.eval_boundary_first_prev_lo(rows[0])
    results.append(ConstraintResult(
        name="first_prev_lo",
        satisfied=(first_prev_lo == 0),
        expected=0,
        actual=rows[0][9],
        message="First row prev_receipt_lo must be 0",
    ))

    first_prev_hi = AgentConstraints.eval_boundary_first_prev_hi(rows[0])
    results.append(ConstraintResult(
        name="first_prev_hi",
        satisfied=(first_prev_hi == 0),
        expected=0,
        actual=rows[0][10],
        message="First row prev_receipt_hi must be 0",
    ))

    # Boundary constraint: last row receipt matches final_receipt_hash
    final_receipt_lo, final_receipt_hi = sha256_to_field_pair(final_receipt_hash)

    last_receipt_lo = AgentConstraints.eval_boundary_last_receipt_lo(rows[-1], final_receipt_lo)
    results.append(ConstraintResult(
        name="last_receipt_lo",
        satisfied=(last_receipt_lo == 0),
        expected=final_receipt_lo,
        actual=rows[-1][11],
        message="Last row receipt_hash_lo must match declared final receipt",
    ))

    last_receipt_hi = AgentConstraints.eval_boundary_last_receipt_hi(rows[-1], final_receipt_hi)
    results.append(ConstraintResult(
        name="last_receipt_hi",
        satisfied=(last_receipt_hi == 0),
        expected=final_receipt_hi,
        actual=rows[-1][12],
        message="Last row receipt_hash_hi must match declared final receipt",
    ))

    # Transition constraints: between consecutive rows
    for i in range(len(rows) - 1):
        row_curr = rows[i]
        row_next = rows[i + 1]

        # Chain constraint (lo)
        chain_lo = AgentConstraints.eval_transition_chain_lo(row_curr, row_next)
        results.append(ConstraintResult(
            name=f"chain_lo_{i}",
            satisfied=(chain_lo == 0),
            expected=row_curr[11],
            actual=row_next[9],
            message=f"Row {i+1} prev_receipt_lo must match row {i} receipt_lo",
        ))

        # Chain constraint (hi)
        chain_hi = AgentConstraints.eval_transition_chain_hi(row_curr, row_next)
        results.append(ConstraintResult(
            name=f"chain_hi_{i}",
            satisfied=(chain_hi == 0),
            expected=row_curr[12],
            actual=row_next[10],
            message=f"Row {i+1} prev_receipt_hi must match row {i} receipt_hi",
        ))

        # Ordering constraint
        ordering = AgentConstraints.eval_transition_ordering(row_curr, row_next)
        results.append(ConstraintResult(
            name=f"ordering_{i}",
            satisfied=(ordering == 0),
            expected=row_curr[0] + 1,
            actual=row_next[0],
            message=f"Row {i+1} action_index must be row {i} action_index + 1",
        ))

    # Check if all constraints satisfied
    all_satisfied = all(r.satisfied for r in results)
    return all_satisfied, results


def build_composition_vector(
    rows: list[list[int]],
    final_receipt_lo: int,
    final_receipt_hi: int,
    alphas: dict[str, int] | None = None,
) -> list[int]:
    """Build the composition polynomial vector for FRI proving.

    The composition vector is a linear combination of all constraint
    polynomials evaluated at each row, weighted by random alphas.

    Args:
        rows: Trace matrix (list of rows)
        final_receipt_lo: Expected final receipt hash (lo)
        final_receipt_hi: Expected final receipt hash (hi)
        alphas: Random weights for each constraint (default: all 1s)

    Returns:
        List of composition values, one per row
    """
    if alphas is None:
        alphas = {name: 1 for name in AgentConstraints.CONSTRAINT_NAMES}

    num_rows = len(rows)
    composition = [0] * num_rows

    for i in range(num_rows):
        row = rows[i]
        row_composition = 0

        # First row boundary constraints
        if i == 0:
            first_prev_lo = AgentConstraints.eval_boundary_first_prev_lo(row)
            first_prev_hi = AgentConstraints.eval_boundary_first_prev_hi(row)
            row_composition += alphas.get("first_prev_lo", 1) * first_prev_lo
            row_composition += alphas.get("first_prev_hi", 1) * first_prev_hi

        # Last row boundary constraints
        if i == num_rows - 1:
            last_receipt_lo = AgentConstraints.eval_boundary_last_receipt_lo(row, final_receipt_lo)
            last_receipt_hi = AgentConstraints.eval_boundary_last_receipt_hi(row, final_receipt_hi)
            row_composition += alphas.get("last_receipt_lo", 1) * last_receipt_lo
            row_composition += alphas.get("last_receipt_hi", 1) * last_receipt_hi

        # Transition constraints (apply to all rows except last)
        if i < num_rows - 1:
            row_next = rows[i + 1]
            chain_lo = AgentConstraints.eval_transition_chain_lo(row, row_next)
            chain_hi = AgentConstraints.eval_transition_chain_hi(row, row_next)
            ordering = AgentConstraints.eval_transition_ordering(row, row_next)

            row_composition += alphas.get("chain_lo", 1) * chain_lo
            row_composition += alphas.get("chain_hi", 1) * chain_hi
            row_composition += alphas.get("ordering", 1) * ordering

        composition[i] = row_composition % GOLDILOCKS_P

    return composition


def derive_constraint_alphas(seed: bytes) -> dict[str, int]:
    """Derive random alpha weights for constraint composition from a seed.

    Uses the Fiat-Shamir paradigm to derive verifier challenges.

    Args:
        seed: Random seed bytes (typically derived from transcript)

    Returns:
        Dict mapping constraint name to alpha weight
    """
    import hashlib

    alphas: dict[str, int] = {}
    for i, name in enumerate(AgentConstraints.CONSTRAINT_NAMES):
        h = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        val = int.from_bytes(h, "big") % GOLDILOCKS_P
        if val == 0:
            val = 1  # Avoid zero weights
        alphas[name] = val

    return alphas
