"""AIR constraints for EvalAIR.

The FRI proof enforces these constraints to prove eval loop execution integrity.

Constraint Types:
- Transition constraints: Between consecutive rounds
- Boundary constraints: On specific rows (first/last)
- Per-row constraints: On each row individually

Transition Constraints:
  1. Posterior chain: Row[i].posteriors_hash == Row[i+1].prev_posteriors_hash
     Proves the posteriors were correctly carried forward -- no substitution.
     Encoded as: row_next[3] - row_curr[1] == 0 (lo)
                 row_next[4] - row_curr[2] == 0 (hi)

  2. Round ordering: Row[i+1].round_index == Row[i].round_index + 1
     Proves rounds are in sequential order.
     Encoded as: row_next[0] - row_curr[0] - 1 == 0

  3. Episode count consistency: n_successes + n_failures == episodes_per_round
     This is verified per-row, not as a transition constraint.

Boundary Constraints:
  1. First row: prev_posteriors_hash == 0 (Beta(1,1) priors, no previous round)
  2. First row: status_flags bit2 == 1 (first_round flag set)
  3. Last row: posteriors_hash must match the publicly declared final posteriors hash

Per-Row Constraints:
  1. n_successes + n_failures == episodes_per_round (from trace_spec)
  2. All field elements < GOLDILOCKS_P (range check)
  3. Status flags consistency (improved XOR regressed, not both)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bef_zk.capsule.eval_air import (
    GOLDILOCKS_P,
    EVAL_AIR_ROW_WIDTH,
    STATUS_BIT_IMPROVED,
    STATUS_BIT_REGRESSED,
    STATUS_BIT_FIRST_ROUND,
    STATUS_BIT_RECEIPTS_VALID,
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
class EvalConstraints:
    """Container for eval AIR constraint definitions.

    These constraints are designed to be fed into the FRI prover.
    The prover will generate a proof that these constraints hold over
    the committed trace.
    """

    # Number of transition constraints (between consecutive rows)
    NUM_TRANSITION_CONSTRAINTS = 3  # posterior_chain_lo, posterior_chain_hi, ordering

    # Number of boundary constraints
    NUM_BOUNDARY_CONSTRAINTS = 5  # first_prev_lo, first_prev_hi, first_round_flag, last_posteriors_lo, last_posteriors_hi

    # Constraint identifiers for composition
    CONSTRAINT_NAMES = [
        "posterior_chain_lo",     # posterior chain continuity (lo)
        "posterior_chain_hi",     # posterior chain continuity (hi)
        "ordering",               # sequential round ordering
        "first_prev_lo",          # first row prev_posteriors is 0 (lo)
        "first_prev_hi",          # first row prev_posteriors is 0 (hi)
        "first_round_flag",       # first row has first_round flag set
        "last_posteriors_lo",     # last row posteriors matches declared (lo)
        "last_posteriors_hi",     # last row posteriors matches declared (hi)
        "episode_count",          # n_successes + n_failures == episodes_per_round
    ]

    @staticmethod
    def eval_transition_posterior_chain_lo(row_curr: list[int], row_next: list[int]) -> int:
        """Evaluate posterior chain constraint (lo): row_next[3] - row_curr[1] == 0.

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        # row[1] = posteriors_hash_lo of current
        # row[3] = prev_posteriors_hash_lo of next
        return (row_next[3] - row_curr[1]) % GOLDILOCKS_P

    @staticmethod
    def eval_transition_posterior_chain_hi(row_curr: list[int], row_next: list[int]) -> int:
        """Evaluate posterior chain constraint (hi): row_next[4] - row_curr[2] == 0.

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        # row[2] = posteriors_hash_hi of current
        # row[4] = prev_posteriors_hash_hi of next
        return (row_next[4] - row_curr[2]) % GOLDILOCKS_P

    @staticmethod
    def eval_transition_ordering(row_curr: list[int], row_next: list[int]) -> int:
        """Evaluate ordering constraint: row_next[0] - row_curr[0] - 1 == 0.

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        # row[0] = round_index
        return (row_next[0] - row_curr[0] - 1) % GOLDILOCKS_P

    @staticmethod
    def eval_boundary_first_prev_lo(row_first: list[int]) -> int:
        """Evaluate first row constraint: row[3] == 0 (prev_posteriors_lo).

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        return row_first[3] % GOLDILOCKS_P

    @staticmethod
    def eval_boundary_first_prev_hi(row_first: list[int]) -> int:
        """Evaluate first row constraint: row[4] == 0 (prev_posteriors_hi).

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        return row_first[4] % GOLDILOCKS_P

    @staticmethod
    def eval_boundary_first_round_flag(row_first: list[int]) -> int:
        """Evaluate first row constraint: status_flags bit2 == 1.

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        flags = row_first[13]
        has_first_round = (flags >> STATUS_BIT_FIRST_ROUND) & 1
        # We want has_first_round == 1, so return 1 - has_first_round
        return (1 - has_first_round) % GOLDILOCKS_P

    @staticmethod
    def eval_boundary_last_posteriors_lo(row_last: list[int], final_posteriors_lo: int) -> int:
        """Evaluate last row constraint: row[1] == final_posteriors_lo.

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        return (row_last[1] - final_posteriors_lo) % GOLDILOCKS_P

    @staticmethod
    def eval_boundary_last_posteriors_hi(row_last: list[int], final_posteriors_hi: int) -> int:
        """Evaluate last row constraint: row[2] == final_posteriors_hi.

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        return (row_last[2] - final_posteriors_hi) % GOLDILOCKS_P

    @staticmethod
    def eval_episode_count(row: list[int], episodes_per_round: int) -> int:
        """Evaluate per-row constraint: row[9] + row[10] == episodes_per_round.

        Returns 0 if constraint is satisfied, non-zero otherwise.
        """
        n_successes = row[9]
        n_failures = row[10]
        total = (n_successes + n_failures) % GOLDILOCKS_P
        return (total - episodes_per_round) % GOLDILOCKS_P

    @staticmethod
    def check_field_bounds(row: list[int]) -> bool:
        """Check that all field elements are < GOLDILOCKS_P."""
        return all(0 <= v < GOLDILOCKS_P for v in row)

    @staticmethod
    def check_status_consistency(row: list[int]) -> bool:
        """Check that improved and regressed are not both set (except for first round)."""
        flags = row[13]
        improved = (flags >> STATUS_BIT_IMPROVED) & 1
        regressed = (flags >> STATUS_BIT_REGRESSED) & 1
        first_round = (flags >> STATUS_BIT_FIRST_ROUND) & 1

        # First round can have neither improved nor regressed
        if first_round:
            return not (improved and regressed)

        # Other rounds can have improved, regressed, or neither (no change)
        # but not both
        return not (improved and regressed)


def verify_eval_trace(
    rows: list[list[int]],
    final_posteriors_hash: str,
    episodes_per_round: int,
) -> tuple[bool, list[ConstraintResult]]:
    """Verify that an eval trace satisfies all AIR constraints.

    This is the verification logic that the FRI proof proves.

    Args:
        rows: List of 14-element rows (trace matrix)
        final_posteriors_hash: SHA256 hex hash that the last row's posteriors must match
        episodes_per_round: Expected episode count per round (n_successes + n_failures)

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
        if len(row) != EVAL_AIR_ROW_WIDTH:
            results.append(ConstraintResult(
                name=f"row_width_{i}",
                satisfied=False,
                expected=EVAL_AIR_ROW_WIDTH,
                actual=len(row),
                message=f"Row {i} has wrong width",
            ))

    # Verify field bounds for all rows
    for i, row in enumerate(rows):
        if not EvalConstraints.check_field_bounds(row):
            results.append(ConstraintResult(
                name=f"field_bounds_{i}",
                satisfied=False,
                message=f"Row {i} has element(s) >= GOLDILOCKS_P",
            ))

    # Verify status consistency for all rows
    for i, row in enumerate(rows):
        if not EvalConstraints.check_status_consistency(row):
            results.append(ConstraintResult(
                name=f"status_consistency_{i}",
                satisfied=False,
                message=f"Row {i} has both improved and regressed flags set",
            ))

    # Boundary constraint: first row prev_posteriors == 0
    first_prev_lo = EvalConstraints.eval_boundary_first_prev_lo(rows[0])
    results.append(ConstraintResult(
        name="first_prev_lo",
        satisfied=(first_prev_lo == 0),
        expected=0,
        actual=rows[0][3],
        message="First row prev_posteriors_lo must be 0",
    ))

    first_prev_hi = EvalConstraints.eval_boundary_first_prev_hi(rows[0])
    results.append(ConstraintResult(
        name="first_prev_hi",
        satisfied=(first_prev_hi == 0),
        expected=0,
        actual=rows[0][4],
        message="First row prev_posteriors_hi must be 0",
    ))

    # Boundary constraint: first row has first_round flag
    first_round_flag = EvalConstraints.eval_boundary_first_round_flag(rows[0])
    results.append(ConstraintResult(
        name="first_round_flag",
        satisfied=(first_round_flag == 0),
        expected=1,
        actual=(rows[0][13] >> STATUS_BIT_FIRST_ROUND) & 1,
        message="First row must have first_round flag set",
    ))

    # Boundary constraint: last row posteriors matches final_posteriors_hash
    final_posteriors_lo, final_posteriors_hi = sha256_to_field_pair(final_posteriors_hash)

    last_posteriors_lo = EvalConstraints.eval_boundary_last_posteriors_lo(rows[-1], final_posteriors_lo)
    results.append(ConstraintResult(
        name="last_posteriors_lo",
        satisfied=(last_posteriors_lo == 0),
        expected=final_posteriors_lo,
        actual=rows[-1][1],
        message="Last row posteriors_hash_lo must match declared final posteriors",
    ))

    last_posteriors_hi = EvalConstraints.eval_boundary_last_posteriors_hi(rows[-1], final_posteriors_hi)
    results.append(ConstraintResult(
        name="last_posteriors_hi",
        satisfied=(last_posteriors_hi == 0),
        expected=final_posteriors_hi,
        actual=rows[-1][2],
        message="Last row posteriors_hash_hi must match declared final posteriors",
    ))

    # Per-row constraint: episode count
    for i, row in enumerate(rows):
        episode_count = EvalConstraints.eval_episode_count(row, episodes_per_round)
        results.append(ConstraintResult(
            name=f"episode_count_{i}",
            satisfied=(episode_count == 0),
            expected=episodes_per_round,
            actual=row[9] + row[10],
            message=f"Row {i} episode count must equal {episodes_per_round}",
        ))

    # Transition constraints: between consecutive rows
    for i in range(len(rows) - 1):
        row_curr = rows[i]
        row_next = rows[i + 1]

        # Posterior chain constraint (lo)
        chain_lo = EvalConstraints.eval_transition_posterior_chain_lo(row_curr, row_next)
        results.append(ConstraintResult(
            name=f"posterior_chain_lo_{i}",
            satisfied=(chain_lo == 0),
            expected=row_curr[1],
            actual=row_next[3],
            message=f"Row {i+1} prev_posteriors_lo must match row {i} posteriors_lo",
        ))

        # Posterior chain constraint (hi)
        chain_hi = EvalConstraints.eval_transition_posterior_chain_hi(row_curr, row_next)
        results.append(ConstraintResult(
            name=f"posterior_chain_hi_{i}",
            satisfied=(chain_hi == 0),
            expected=row_curr[2],
            actual=row_next[4],
            message=f"Row {i+1} prev_posteriors_hi must match row {i} posteriors_hi",
        ))

        # Ordering constraint
        ordering = EvalConstraints.eval_transition_ordering(row_curr, row_next)
        results.append(ConstraintResult(
            name=f"ordering_{i}",
            satisfied=(ordering == 0),
            expected=row_curr[0] + 1,
            actual=row_next[0],
            message=f"Row {i+1} round_index must be row {i} round_index + 1",
        ))

    # Check if all constraints satisfied
    all_satisfied = all(r.satisfied for r in results)
    return all_satisfied, results


def build_composition_vector(
    rows: list[list[int]],
    final_posteriors_lo: int,
    final_posteriors_hi: int,
    episodes_per_round: int,
    alphas: dict[str, int] | None = None,
) -> list[int]:
    """Build the composition polynomial vector for FRI proving.

    The composition vector is a linear combination of all constraint
    polynomials evaluated at each row, weighted by random alphas.

    Args:
        rows: Trace matrix (list of rows)
        final_posteriors_lo: Expected final posteriors hash (lo)
        final_posteriors_hi: Expected final posteriors hash (hi)
        episodes_per_round: Expected episode count per round
        alphas: Random weights for each constraint (default: all 1s)

    Returns:
        List of composition values, one per row
    """
    if alphas is None:
        alphas = {name: 1 for name in EvalConstraints.CONSTRAINT_NAMES}

    num_rows = len(rows)
    composition = [0] * num_rows

    for i in range(num_rows):
        row = rows[i]
        row_composition = 0

        # First row boundary constraints
        if i == 0:
            first_prev_lo = EvalConstraints.eval_boundary_first_prev_lo(row)
            first_prev_hi = EvalConstraints.eval_boundary_first_prev_hi(row)
            first_round_flag = EvalConstraints.eval_boundary_first_round_flag(row)
            row_composition += alphas.get("first_prev_lo", 1) * first_prev_lo
            row_composition += alphas.get("first_prev_hi", 1) * first_prev_hi
            row_composition += alphas.get("first_round_flag", 1) * first_round_flag

        # Last row boundary constraints
        if i == num_rows - 1:
            last_posteriors_lo = EvalConstraints.eval_boundary_last_posteriors_lo(row, final_posteriors_lo)
            last_posteriors_hi = EvalConstraints.eval_boundary_last_posteriors_hi(row, final_posteriors_hi)
            row_composition += alphas.get("last_posteriors_lo", 1) * last_posteriors_lo
            row_composition += alphas.get("last_posteriors_hi", 1) * last_posteriors_hi

        # Per-row constraints
        episode_count = EvalConstraints.eval_episode_count(row, episodes_per_round)
        row_composition += alphas.get("episode_count", 1) * episode_count

        # Transition constraints (apply to all rows except last)
        if i < num_rows - 1:
            row_next = rows[i + 1]
            chain_lo = EvalConstraints.eval_transition_posterior_chain_lo(row, row_next)
            chain_hi = EvalConstraints.eval_transition_posterior_chain_hi(row, row_next)
            ordering = EvalConstraints.eval_transition_ordering(row, row_next)

            row_composition += alphas.get("posterior_chain_lo", 1) * chain_lo
            row_composition += alphas.get("posterior_chain_hi", 1) * chain_hi
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
    for i, name in enumerate(EvalConstraints.CONSTRAINT_NAMES):
        h = hashlib.sha256(seed + i.to_bytes(4, "big")).digest()
        val = int.from_bytes(h, "big") % GOLDILOCKS_P
        if val == 0:
            val = 1  # Avoid zero weights
        alphas[name] = val

    return alphas
