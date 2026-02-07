"""
Claim/Witness Schema - The API of Truth

This module defines the normalized objects that every checker can consume
and every policy can gate. This is the foundation of Layer 1.

A Claim is an assertion about code that can be verified.
A Witness is the evidence that supports or refutes a claim.
A Checker is a deterministic function that produces a verdict.

The LLM can propose; it cannot be the judge.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ClaimType(Enum):
    """Standard claim types that checkers can verify."""
    # Security claims
    NO_SQL_INJECTION = "no_sql_injection"
    NO_SHELL_INJECTION = "no_shell_injection"
    NO_PATH_TRAVERSAL = "no_path_traversal"
    NO_XSS = "no_xss"
    NO_HARDCODED_SECRETS = "no_hardcoded_secrets"
    ALLOWLIST_ENFORCED = "allowlist_enforced"
    ALLOWLIST_CONSTANT = "allowlist_constant"  # Dict/mapping has no mutation sites
    INPUT_VALIDATED = "input_validated"

    # Behavioral claims
    REFACTOR_EQUIVALENCE = "refactor_equivalence"
    STATE_MACHINE_INVARIANT = "state_machine_invariant"
    PURE_FUNCTION = "pure_function"
    NO_SIDE_EFFECTS = "no_side_effects"
    IDEMPOTENT = "idempotent"

    # Structural claims
    TYPE_SAFE = "type_safe"
    NO_DEAD_CODE = "no_dead_code"
    NO_UNUSED_IMPORTS = "no_unused_imports"
    TEST_COVERAGE = "test_coverage"

    # Meta claims
    NO_CHANGE_NEEDED = "no_change_needed"
    ALREADY_MITIGATED = "already_mitigated"
    FALSE_POSITIVE = "false_positive"

    # Committor gate claims
    COMMITTOR_GATE = "committor_gate"

    # Custom (for extension)
    CUSTOM = "custom"


class Verdict(Enum):
    """
    The outcome of a checker's evaluation.

    UNKNOWN must be first-class - it means the checker couldn't decide,
    which is different from PASS or FAIL.
    """
    PASS = "pass"           # Claim verified
    FAIL = "fail"           # Claim refuted (with counterexample)
    UNKNOWN = "unknown"     # Checker couldn't decide (need human review)
    TIMEOUT = "timeout"     # Checker timed out
    ERROR = "error"         # Checker failed to run


@dataclass
class Scope:
    """
    The scope of a claim - what code region it applies to.

    Immutably anchored by hashes so we know when it's invalidated.
    """
    file_path: str
    file_hash: str                          # SHA256 of file content at claim time

    # Fine-grained location (optional)
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None

    # Hash of the specific region (for incremental reuse)
    region_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "function_name": self.function_name,
            "class_name": self.class_name,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "region_hash": self.region_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scope":
        return cls(**d)

    def matches_current(self, current_file_hash: str) -> bool:
        """Check if this scope is still valid (file hasn't changed)."""
        return self.file_hash == current_file_hash

    def identity_key(self) -> str:
        """A stable key for caching/lookup."""
        parts = [self.file_path, self.function_name or "", self.class_name or ""]
        if self.start_line:
            parts.append(f"{self.start_line}-{self.end_line}")
        return ":".join(parts)


@dataclass
class Assumption:
    """
    An explicit assumption that the claim depends on.

    Even tiny assumptions must be explicit so they can be tracked
    and invalidated when the assumption no longer holds.
    """
    description: str
    assumption_type: str  # "input_range", "env_config", "dependency_version", etc.
    value: Any = None
    verified: bool = False

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "assumption_type": self.assumption_type,
            "value": self.value,
            "verified": self.verified,
        }


@dataclass
class Witness:
    """
    Evidence that supports or refutes a claim.

    This is a pointer to the checker's artifact plus metadata
    that allows verification and replay.
    """
    witness_type: str  # "semgrep_output", "sat_result", "test_trace", "counterexample", etc.

    # The actual evidence (or pointer to it)
    artifact_hash: str              # Hash of the witness artifact
    artifact_path: Optional[str] = None  # Where it's stored
    artifact_inline: Optional[str] = None  # Small artifacts can be inline

    # Counterexample (if FAIL)
    counterexample: Optional[str] = None
    counterexample_trace: Optional[list[str]] = None

    # Metadata
    produced_at: str = ""
    producer: str = ""  # Which checker produced this

    def to_dict(self) -> dict:
        return {
            "witness_type": self.witness_type,
            "artifact_hash": self.artifact_hash,
            "artifact_path": self.artifact_path,
            "artifact_inline": self.artifact_inline,
            "counterexample": self.counterexample,
            "counterexample_trace": self.counterexample_trace,
            "produced_at": self.produced_at,
            "producer": self.producer,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Witness":
        return cls(**d)


@dataclass
class CheckerInfo:
    """Information about the checker that evaluated a claim."""
    checker_id: str         # e.g., "semgrep", "hypothesis", "z3", "crosshair"
    checker_version: str    # Version string
    env_key: str            # Environment fingerprint for reproducibility
    config_hash: str = ""   # Hash of checker configuration

    def to_dict(self) -> dict:
        return {
            "checker_id": self.checker_id,
            "checker_version": self.checker_version,
            "env_key": self.env_key,
            "config_hash": self.config_hash,
        }


@dataclass
class Claim:
    """
    A verifiable assertion about code.

    This is the normalized object that every checker can consume
    and every policy can gate.
    """
    claim_id: str                           # Unique identifier
    claim_type: ClaimType                   # What kind of claim
    scope: Scope                            # What code it applies to

    # The assertion
    description: str                        # Human-readable description
    custom_type: Optional[str] = None       # For ClaimType.CUSTOM

    # Dependencies
    assumptions: list[Assumption] = field(default_factory=list)

    # Evaluation
    verdict: Verdict = Verdict.UNKNOWN
    witness: Optional[Witness] = None
    checker: Optional[CheckerInfo] = None

    # Metadata
    created_at: str = ""
    evaluated_at: str = ""

    # For claim chaining (claims can depend on other claims)
    depends_on: list[str] = field(default_factory=list)  # claim_ids

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "claim_type": self.claim_type.value,
            "scope": self.scope.to_dict(),
            "description": self.description,
            "custom_type": self.custom_type,
            "assumptions": [a.to_dict() for a in self.assumptions],
            "verdict": self.verdict.value,
            "witness": self.witness.to_dict() if self.witness else None,
            "checker": self.checker.to_dict() if self.checker else None,
            "created_at": self.created_at,
            "evaluated_at": self.evaluated_at,
            "depends_on": self.depends_on,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Claim":
        return cls(
            claim_id=d["claim_id"],
            claim_type=ClaimType(d["claim_type"]),
            scope=Scope.from_dict(d["scope"]),
            description=d["description"],
            custom_type=d.get("custom_type"),
            assumptions=[Assumption(**a) for a in d.get("assumptions", [])],
            verdict=Verdict(d.get("verdict", "unknown")),
            witness=Witness.from_dict(d["witness"]) if d.get("witness") else None,
            checker=CheckerInfo(**d["checker"]) if d.get("checker") else None,
            created_at=d.get("created_at", ""),
            evaluated_at=d.get("evaluated_at", ""),
            depends_on=d.get("depends_on", []),
        )

    def is_valid(self) -> bool:
        """Check if this claim has passed verification."""
        return self.verdict == Verdict.PASS

    def needs_review(self) -> bool:
        """Check if this claim needs human review."""
        return self.verdict in (Verdict.UNKNOWN, Verdict.ERROR, Verdict.TIMEOUT)

    def invalidated_by(self, new_file_hash: str) -> bool:
        """Check if this claim is invalidated by a file change."""
        return not self.scope.matches_current(new_file_hash)


@dataclass
class ProofObligation:
    """
    A requirement that must be proven for a change to be accepted.

    This is specified in policy and gates the pipeline.
    """
    obligation_id: str
    claim_type: ClaimType
    description: str

    # Scope pattern (what files/functions this applies to)
    file_pattern: str = "*"             # glob pattern
    function_pattern: Optional[str] = None

    # What verdict is required
    required_verdict: Verdict = Verdict.PASS

    # Can this be waived?
    waivable: bool = False
    waiver_requires: list[str] = field(default_factory=list)  # e.g., ["justification", "reviewer_approval"]

    # Which checker(s) can evaluate this
    allowed_checkers: list[str] = field(default_factory=list)  # empty = any

    def to_dict(self) -> dict:
        return {
            "obligation_id": self.obligation_id,
            "claim_type": self.claim_type.value,
            "description": self.description,
            "file_pattern": self.file_pattern,
            "function_pattern": self.function_pattern,
            "required_verdict": self.required_verdict.value,
            "waivable": self.waivable,
            "waiver_requires": self.waiver_requires,
            "allowed_checkers": self.allowed_checkers,
        }


@dataclass
class ClaimBundle:
    """
    A collection of claims for a single change/receipt.

    This is what gets attached to a receipt.
    """
    bundle_id: str
    receipt_id: str
    claims: list[Claim] = field(default_factory=list)

    # Summary
    total_claims: int = 0
    passed: int = 0
    failed: int = 0
    unknown: int = 0

    # Obligations
    obligations_met: list[str] = field(default_factory=list)  # obligation_ids
    obligations_failed: list[str] = field(default_factory=list)
    obligations_waived: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "receipt_id": self.receipt_id,
            "claims": [c.to_dict() for c in self.claims],
            "total_claims": self.total_claims,
            "passed": self.passed,
            "failed": self.failed,
            "unknown": self.unknown,
            "obligations_met": self.obligations_met,
            "obligations_failed": self.obligations_failed,
            "obligations_waived": self.obligations_waived,
        }

    def compute_summary(self) -> None:
        """Recompute summary stats from claims."""
        self.total_claims = len(self.claims)
        self.passed = sum(1 for c in self.claims if c.verdict == Verdict.PASS)
        self.failed = sum(1 for c in self.claims if c.verdict == Verdict.FAIL)
        self.unknown = sum(1 for c in self.claims if c.verdict in (Verdict.UNKNOWN, Verdict.TIMEOUT, Verdict.ERROR))

    def all_passed(self) -> bool:
        """Check if all claims passed."""
        return all(c.verdict == Verdict.PASS for c in self.claims)

    def has_failures(self) -> bool:
        """Check if any claims failed."""
        return any(c.verdict == Verdict.FAIL for c in self.claims)


# ─────────────────────────────────────────────────────────────────
# Checker Registry
# ─────────────────────────────────────────────────────────────────

class CheckerRegistry:
    """
    Registry of deterministic checkers.

    Checkers are functions that:
    1. Take a Claim and file content
    2. Return a Verdict and Witness
    3. Are deterministic given the same inputs
    """

    def __init__(self):
        self._checkers: dict[str, callable] = {}
        self._checker_info: dict[str, CheckerInfo] = {}

    def register(self, checker_id: str, checker_fn: callable, info: CheckerInfo) -> None:
        """Register a checker."""
        self._checkers[checker_id] = checker_fn
        self._checker_info[checker_id] = info

    def get(self, checker_id: str) -> tuple[callable, CheckerInfo] | None:
        """Get a checker and its info."""
        if checker_id in self._checkers:
            return self._checkers[checker_id], self._checker_info[checker_id]
        return None

    def list_checkers(self) -> list[str]:
        """List all registered checkers."""
        return list(self._checkers.keys())

    def evaluate(self, checker_id: str, claim: Claim, file_content: str) -> tuple[Verdict, Witness | None]:
        """
        Evaluate a claim using a specific checker.

        Returns (verdict, witness).
        """
        checker_tuple = self.get(checker_id)
        if not checker_tuple:
            return Verdict.ERROR, None

        checker_fn, info = checker_tuple
        try:
            verdict, witness = checker_fn(claim, file_content)
            claim.verdict = verdict
            claim.witness = witness
            claim.checker = info
            claim.evaluated_at = datetime.utcnow().isoformat() + "Z"
            return verdict, witness
        except Exception as e:
            return Verdict.ERROR, Witness(
                witness_type="error",
                artifact_hash="",
                artifact_inline=str(e),
                produced_at=datetime.utcnow().isoformat() + "Z",
                producer=checker_id,
            )


# Global registry
CHECKER_REGISTRY = CheckerRegistry()


# ─────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────

def create_claim(
    claim_type: ClaimType,
    file_path: str,
    file_content: str,
    description: str,
    start_line: int | None = None,
    end_line: int | None = None,
    function_name: str | None = None,
) -> Claim:
    """Create a new claim with proper hashing."""
    file_hash = hashlib.sha256(file_content.encode()).hexdigest()

    region_hash = None
    if start_line and end_line:
        lines = file_content.split('\n')
        region = '\n'.join(lines[start_line-1:end_line])
        region_hash = hashlib.sha256(region.encode()).hexdigest()

    scope = Scope(
        file_path=file_path,
        file_hash=file_hash,
        function_name=function_name,
        start_line=start_line,
        end_line=end_line,
        region_hash=region_hash,
    )

    claim_id = hashlib.sha256(
        f"{claim_type.value}:{scope.identity_key()}:{datetime.utcnow().isoformat()}".encode()
    ).hexdigest()[:16]

    return Claim(
        claim_id=f"CLM-{claim_id}",
        claim_type=claim_type,
        scope=scope,
        description=description,
        created_at=datetime.utcnow().isoformat() + "Z",
    )


def claim_from_no_change_proof(proof: dict, file_path: str, file_content: str) -> Claim:
    """Convert existing NO_CHANGE proof to a Claim."""
    # Extract span info - handle both dict and SpanAnchor object
    span = proof.get("span_anchor", {})
    if hasattr(span, 'start_line'):
        # It's a SpanAnchor object
        start_line = span.start_line
        end_line = span.end_line
    elif isinstance(span, dict):
        # It's a dict
        start_line = span.get("start_line")
        end_line = span.get("end_line")
    else:
        start_line = None
        end_line = None

    claim = create_claim(
        claim_type=ClaimType.ALREADY_MITIGATED if proof.get("disposition") == "already_mitigated" else ClaimType.FALSE_POSITIVE,
        file_path=file_path,
        file_content=file_content,
        description=proof.get("justification", ""),
        start_line=start_line,
        end_line=end_line,
    )

    # Create witness from evidence
    claim.witness = Witness(
        witness_type="no_change_evidence",
        artifact_hash=proof.get("evidence_chunk_hash", ""),
        artifact_inline=proof.get("evidence_snippet", ""),
        produced_at=datetime.utcnow().isoformat() + "Z",
        producer="refactor_engine",
    )

    # Mark as passed (it was already validated)
    claim.verdict = Verdict.PASS

    return claim
