"""CapSeal Diagnosis Contracts - The Glue Spec

This module defines the mapping:
    (exit_code, error_codes, evidence_refs) → (route, action_plan) → graded_outcome

This is the "contextual semantics" layer that enables agents to diagnose verification
failures without mystical understanding - just stable structure → stable actions.

The diagnosis surface is already operational (VerifyReport, AuditReport, EvidenceIndex).
This module adds:
1. DiagnosisRoute: exit_code → failure bucket
2. ActionType: enumerated action types with machine-checkable success criteria
3. ActionTemplate: for each (route, error_code), the valid action sequence
4. GradingRubric: auto-grade whether an action plan is correct
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from capseal.contracts import ExitCode


# =============================================================================
# DIAGNOSIS ROUTES (exit_code → failure bucket)
# =============================================================================

class DiagnosisRoute(Enum):
    """Failure buckets derived from exit codes.

    Each route maps to a category of verification failure with specific
    diagnostic patterns and remediation strategies.
    """
    VERIFIED = "VERIFIED"                     # exit_code = 0
    PROOF_FAILED = "PROOF_FAILED"             # exit_code = 10
    POLICY_MISMATCH = "POLICY_MISMATCH"       # exit_code = 11
    COMMITMENT_FAILED = "COMMITMENT_FAILED"   # exit_code = 12
    DA_AUDIT_FAILED = "DA_AUDIT_FAILED"       # exit_code = 13
    REPLAY_DIVERGED = "REPLAY_DIVERGED"       # exit_code = 14
    PARSE_ERROR = "PARSE_ERROR"               # exit_code = 20
    RUNTIME_ERROR = "RUNTIME_ERROR"           # exit_code = 30


# Stable mapping from exit codes to routes
EXIT_CODE_TO_ROUTE: dict[int, DiagnosisRoute] = {
    0: DiagnosisRoute.VERIFIED,
    10: DiagnosisRoute.PROOF_FAILED,
    11: DiagnosisRoute.POLICY_MISMATCH,
    12: DiagnosisRoute.COMMITMENT_FAILED,
    13: DiagnosisRoute.DA_AUDIT_FAILED,
    14: DiagnosisRoute.REPLAY_DIVERGED,
    20: DiagnosisRoute.PARSE_ERROR,
    30: DiagnosisRoute.RUNTIME_ERROR,
    31: DiagnosisRoute.RUNTIME_ERROR,
}


def route_from_exit_code(exit_code: int) -> DiagnosisRoute:
    """Map exit code to diagnosis route."""
    return EXIT_CODE_TO_ROUTE.get(exit_code, DiagnosisRoute.PARSE_ERROR)


# =============================================================================
# ACTION TYPES (what actions can be taken)
# =============================================================================

class ActionType(Enum):
    """Enumerated action types that agents can propose.

    Each action type has machine-checkable success criteria.
    """
    # Inspection actions
    OPEN_ARTIFACT = "OPEN_ARTIFACT"           # Open an artifact file
    OPEN_ROW = "OPEN_ROW"                     # Open a specific row
    INSPECT_LAYER = "INSPECT_LAYER"           # Inspect a verification layer
    VIEW_TIMELINE = "VIEW_TIMELINE"           # View audit timeline

    # Comparison actions
    COMPARE_HASH = "COMPARE_HASH"             # Compare hashes
    COMPARE_COMMITMENT = "COMPARE_COMMITMENT" # Compare commitments
    DIFF_REPLAY = "DIFF_REPLAY"               # Diff replay vs original

    # Verification actions
    RERUN_VERIFY = "RERUN_VERIFY"             # Re-run verification
    RERUN_AUDIT = "RERUN_AUDIT"               # Re-run audit
    RERUN_REPLAY = "RERUN_REPLAY"             # Re-run replay

    # Decision actions
    DECISION = "DECISION"                     # Make a final decision


class SuccessCriteriaKind(Enum):
    """Kinds of machine-checkable success criteria."""
    ARTIFACT_EXISTS = "artifact_exists_in_evidence_index"
    ROW_EXISTS = "row_exists_in_openable_rows"
    HASH_MATCH = "hash_match"
    HASH_MISMATCH_EXPLAINED = "hash_mismatch_explained"
    LAYER_INSPECTED = "layer_status_retrieved"
    TIMELINE_RETRIEVED = "timeline_retrieved"
    VERIFY_STATUS_CHANGED = "verify_status_changed"
    DECISION_CONSISTENT = "decision_consistent_with_error_code"


@dataclass
class SuccessCriteria:
    """Machine-checkable success criteria for an action."""
    kind: SuccessCriteriaKind
    expected: Any = None  # Expected value (for comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "expected": self.expected,
        }


@dataclass
class Action:
    """A single action in an action plan.

    Every action has machine-checkable success criteria.
    No "look into it" actions allowed.
    """
    action_type: ActionType
    ref: str | None = None             # Artifact name, row index, layer name
    against: str | None = None         # For comparison actions
    why: str | None = None             # Brief explanation
    success_criteria: SuccessCriteria | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.action_type.value,
            "ref": self.ref,
            "against": self.against,
            "why": self.why,
            "success_criteria": self.success_criteria.to_dict() if self.success_criteria else None,
        }


# =============================================================================
# DIAGNOSIS RESULT (agent output contract)
# =============================================================================

class DecisionType(Enum):
    """Final decision types."""
    ACCEPT = "ACCEPT"
    REJECT_AS_TAMPERED = "REJECT_AS_TAMPERED"
    REJECT_POLICY_VIOLATION = "REJECT_POLICY_VIOLATION"
    REGENERATE_PROOF = "REGENERATE_PROOF"
    REGENERATE_AND_RESEAL = "REGENERATE_AND_RESEAL"
    REBUILD_COMMITMENT = "REBUILD_COMMITMENT"
    RESYNC_EVENTS = "RESYNC_EVENTS"
    FIX_AND_RETRY = "FIX_AND_RETRY"


@dataclass
class DiagnosisResult:
    """The structured output contract for diagnosis.

    This is what the agent produces - a structured object, not prose.
    """
    run_id: str
    route: DiagnosisRoute
    primary_error_code: str | None
    layer_failed: str | None
    next_actions: list[Action]
    decision: DecisionType | None = None
    confidence: float = 1.0  # 0.0 - 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "route": self.route.value,
            "primary_error_code": self.primary_error_code,
            "layer_failed": self.layer_failed,
            "next_actions": [a.to_dict() for a in self.next_actions],
            "decision": self.decision.value if self.decision else None,
            "confidence": self.confidence,
        }


# =============================================================================
# ACTION TEMPLATES (error_code → required action pattern)
# =============================================================================

@dataclass
class ActionTemplate:
    """Template for actions required for a specific error code.

    The template defines:
    - Required action types (must appear in plan)
    - Required refs (must reference these evidence_refs)
    - Valid decisions (plan must end with one of these)
    """
    error_code: str
    required_action_types: list[ActionType]
    required_refs_from_evidence: bool = True  # Actions must reference evidence_ref
    valid_decisions: list[DecisionType] = field(default_factory=list)
    description: str = ""


# Stable mapping from error codes to action templates
ERROR_CODE_TEMPLATES: dict[str, ActionTemplate] = {
    # Proof failures (exit_code = 10)
    "E051_PROOF_MISSING": ActionTemplate(
        error_code="E051_PROOF_MISSING",
        required_action_types=[ActionType.INSPECT_LAYER, ActionType.DECISION],
        required_refs_from_evidence=False,  # No artifact to reference
        valid_decisions=[DecisionType.REGENERATE_PROOF, DecisionType.REJECT_AS_TAMPERED],
        description="Proof file is missing - regenerate or reject",
    ),
    "E052_PROOF_HASH_MISMATCH": ActionTemplate(
        error_code="E052_PROOF_HASH_MISMATCH",
        required_action_types=[ActionType.OPEN_ARTIFACT, ActionType.COMPARE_HASH, ActionType.DECISION],
        required_refs_from_evidence=True,
        valid_decisions=[DecisionType.REJECT_AS_TAMPERED, DecisionType.REGENERATE_AND_RESEAL],
        description="Proof hash mismatch - artifact was modified after seal",
    ),
    "E053_PROOF_VERIFICATION_FAILED": ActionTemplate(
        error_code="E053_PROOF_VERIFICATION_FAILED",
        required_action_types=[ActionType.OPEN_ARTIFACT, ActionType.INSPECT_LAYER, ActionType.DECISION],
        required_refs_from_evidence=True,
        valid_decisions=[DecisionType.REJECT_AS_TAMPERED, DecisionType.REGENERATE_PROOF],
        description="Cryptographic verification failed - proof is invalid",
    ),

    # Policy failures (exit_code = 11)
    "E031_POLICY_NOT_FOUND": ActionTemplate(
        error_code="E031_POLICY_NOT_FOUND",
        required_action_types=[ActionType.INSPECT_LAYER, ActionType.DECISION],
        required_refs_from_evidence=False,
        valid_decisions=[DecisionType.FIX_AND_RETRY, DecisionType.REJECT_POLICY_VIOLATION],
        description="Policy file not found",
    ),
    "E032_POLICY_VERSION_MISMATCH": ActionTemplate(
        error_code="E032_POLICY_VERSION_MISMATCH",
        required_action_types=[ActionType.INSPECT_LAYER, ActionType.DECISION],
        required_refs_from_evidence=False,
        valid_decisions=[DecisionType.FIX_AND_RETRY, DecisionType.REJECT_POLICY_VIOLATION],
        description="Policy version doesn't match",
    ),
    "E033_POLICY_HASH_MISMATCH": ActionTemplate(
        error_code="E033_POLICY_HASH_MISMATCH",
        required_action_types=[ActionType.COMPARE_HASH, ActionType.DECISION],
        required_refs_from_evidence=True,
        valid_decisions=[DecisionType.REJECT_AS_TAMPERED, DecisionType.REJECT_POLICY_VIOLATION],
        description="Policy hash mismatch - policy was modified",
    ),

    # Commitment failures (exit_code = 12)
    "E061_ROW_ROOT_MISMATCH": ActionTemplate(
        error_code="E061_ROW_ROOT_MISMATCH",
        required_action_types=[ActionType.COMPARE_COMMITMENT, ActionType.OPEN_ROW, ActionType.DECISION],
        required_refs_from_evidence=True,
        valid_decisions=[DecisionType.REJECT_AS_TAMPERED, DecisionType.REBUILD_COMMITMENT],
        description="Merkle row root doesn't match commitment",
    ),
    "E062_CHUNK_MISSING": ActionTemplate(
        error_code="E062_CHUNK_MISSING",
        required_action_types=[ActionType.INSPECT_LAYER, ActionType.DECISION],
        required_refs_from_evidence=False,
        valid_decisions=[DecisionType.FIX_AND_RETRY, DecisionType.REJECT_AS_TAMPERED],
        description="Data chunk is missing",
    ),
    "E063_CHUNK_HASH_MISMATCH": ActionTemplate(
        error_code="E063_CHUNK_HASH_MISMATCH",
        required_action_types=[ActionType.OPEN_ARTIFACT, ActionType.COMPARE_HASH, ActionType.DECISION],
        required_refs_from_evidence=True,
        valid_decisions=[DecisionType.REJECT_AS_TAMPERED, DecisionType.REBUILD_COMMITMENT],
        description="Chunk hash doesn't match",
    ),
    "E064_MERKLE_PROOF_INVALID": ActionTemplate(
        error_code="E064_MERKLE_PROOF_INVALID",
        required_action_types=[ActionType.INSPECT_LAYER, ActionType.OPEN_ROW, ActionType.DECISION],
        required_refs_from_evidence=True,
        valid_decisions=[DecisionType.REJECT_AS_TAMPERED, DecisionType.REBUILD_COMMITMENT],
        description="Merkle proof verification failed",
    ),

    # DA Audit failures (exit_code = 13)
    "E071_EVENTS_MISSING": ActionTemplate(
        error_code="E071_EVENTS_MISSING",
        required_action_types=[ActionType.VIEW_TIMELINE, ActionType.DECISION],
        required_refs_from_evidence=False,
        valid_decisions=[DecisionType.RESYNC_EVENTS, DecisionType.REJECT_AS_TAMPERED],
        description="Event log is missing",
    ),
    "E072_CHAIN_BROKEN": ActionTemplate(
        error_code="E072_CHAIN_BROKEN",
        required_action_types=[ActionType.VIEW_TIMELINE, ActionType.INSPECT_LAYER, ActionType.DECISION],
        required_refs_from_evidence=True,
        valid_decisions=[DecisionType.REJECT_AS_TAMPERED],
        description="Event chain hash links are broken",
    ),
    "E073_GENESIS_INVALID": ActionTemplate(
        error_code="E073_GENESIS_INVALID",
        required_action_types=[ActionType.VIEW_TIMELINE, ActionType.DECISION],
        required_refs_from_evidence=False,
        valid_decisions=[DecisionType.REJECT_AS_TAMPERED],
        description="Genesis event is invalid",
    ),
    "E074_AVAILABILITY_FAILED": ActionTemplate(
        error_code="E074_AVAILABILITY_FAILED",
        required_action_types=[ActionType.RERUN_AUDIT, ActionType.DECISION],
        required_refs_from_evidence=False,
        valid_decisions=[DecisionType.RESYNC_EVENTS, DecisionType.REJECT_AS_TAMPERED],
        description="Data availability check failed",
    ),

    # Replay failures (exit_code = 14)
    "E081_REPLAY_DIVERGED": ActionTemplate(
        error_code="E081_REPLAY_DIVERGED",
        required_action_types=[ActionType.DIFF_REPLAY, ActionType.DECISION],
        required_refs_from_evidence=True,
        valid_decisions=[DecisionType.REJECT_AS_TAMPERED, DecisionType.FIX_AND_RETRY],
        description="Replay produced different results",
    ),

    # Parse errors (exit_code = 20)
    "E001_PARSE_FAILED": ActionTemplate(
        error_code="E001_PARSE_FAILED",
        required_action_types=[ActionType.INSPECT_LAYER, ActionType.DECISION],
        required_refs_from_evidence=False,
        valid_decisions=[DecisionType.FIX_AND_RETRY],
        description="Failed to parse capsule",
    ),
    "E002_SCHEMA_UNSUPPORTED": ActionTemplate(
        error_code="E002_SCHEMA_UNSUPPORTED",
        required_action_types=[ActionType.INSPECT_LAYER, ActionType.DECISION],
        required_refs_from_evidence=False,
        valid_decisions=[DecisionType.FIX_AND_RETRY],
        description="Schema version not supported",
    ),
    "E011_CAPSULE_HASH_MISMATCH": ActionTemplate(
        error_code="E011_CAPSULE_HASH_MISMATCH",
        required_action_types=[ActionType.COMPARE_HASH, ActionType.DECISION],
        required_refs_from_evidence=True,
        valid_decisions=[DecisionType.REJECT_AS_TAMPERED],
        description="Capsule hash doesn't match claimed ID",
    ),
    "E021_SIGNATURE_INVALID": ActionTemplate(
        error_code="E021_SIGNATURE_INVALID",
        required_action_types=[ActionType.INSPECT_LAYER, ActionType.DECISION],
        required_refs_from_evidence=True,
        valid_decisions=[DecisionType.REJECT_AS_TAMPERED],
        description="Signature verification failed",
    ),
}


def get_template_for_error(error_code: str) -> ActionTemplate | None:
    """Get the action template for an error code."""
    return ERROR_CODE_TEMPLATES.get(error_code)


# =============================================================================
# GRADING RUBRIC (auto-grade action plans)
# =============================================================================

@dataclass
class GradeResult:
    """Result of grading an action plan."""
    valid: bool
    score: float  # 0.0 - 1.0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "score": self.score,
            "issues": self.issues,
        }


class DiagnosisGrader:
    """Auto-grader for diagnosis action plans.

    Grading is deterministic and cheap. Three layers:
    1. Schema validity - output parses, required fields exist
    2. Referential validity - refs match evidence_index
    3. Diagnosis correctness - follows template for error code
    """

    def __init__(self, evidence_index: dict[str, Any] | None = None):
        """Initialize grader with optional evidence index for ref validation."""
        self.evidence_index = evidence_index or {}

    def grade(
        self,
        diagnosis: DiagnosisResult,
        verify_report: dict[str, Any],
    ) -> GradeResult:
        """Grade a diagnosis result against the verify report.

        Args:
            diagnosis: The agent's diagnosis result
            verify_report: The original verification report

        Returns:
            GradeResult with validity, score, and issues
        """
        issues: list[str] = []
        scores: list[float] = []

        # Layer 1: Schema validity
        schema_score, schema_issues = self._grade_schema(diagnosis)
        scores.append(schema_score)
        issues.extend(schema_issues)

        # Layer 2: Referential validity
        ref_score, ref_issues = self._grade_references(diagnosis, verify_report)
        scores.append(ref_score)
        issues.extend(ref_issues)

        # Layer 3: Diagnosis correctness
        correct_score, correct_issues = self._grade_correctness(diagnosis, verify_report)
        scores.append(correct_score)
        issues.extend(correct_issues)

        # Combined score (weighted: schema 20%, refs 30%, correctness 50%)
        final_score = scores[0] * 0.2 + scores[1] * 0.3 + scores[2] * 0.5

        return GradeResult(
            valid=len(issues) == 0,
            score=round(final_score, 3),
            issues=issues,
        )

    def _grade_schema(self, diagnosis: DiagnosisResult) -> tuple[float, list[str]]:
        """Grade schema validity."""
        issues: list[str] = []

        # Required fields
        if not diagnosis.run_id:
            issues.append("Missing run_id")
        if not diagnosis.route:
            issues.append("Missing route")
        if not diagnosis.next_actions:
            issues.append("Missing next_actions (empty list)")

        # Action validity
        for i, action in enumerate(diagnosis.next_actions):
            if not action.action_type:
                issues.append(f"Action {i}: missing action_type")
            if action.success_criteria is None:
                issues.append(f"Action {i}: missing success_criteria")

        # Must end with DECISION
        if diagnosis.next_actions and diagnosis.next_actions[-1].action_type != ActionType.DECISION:
            issues.append("Action plan must end with DECISION")

        score = 1.0 - (len(issues) * 0.2)
        return max(0.0, score), issues

    def _grade_references(
        self,
        diagnosis: DiagnosisResult,
        verify_report: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """Grade referential validity against evidence index."""
        issues: list[str] = []

        # Extract evidence_refs from verify_report
        valid_refs: set[str] = set()

        # From layers
        layers = verify_report.get("layers", {})
        for layer_name, layer_data in layers.items():
            if isinstance(layer_data, dict):
                ref = layer_data.get("evidence_ref")
                if ref:
                    valid_refs.add(ref)

        # From errors
        for error in verify_report.get("errors", []):
            ref = error.get("evidence_ref")
            if ref:
                valid_refs.add(ref)

        # From evidence_index if available
        if self.evidence_index:
            for artifact in self.evidence_index.get("artifacts", []):
                valid_refs.add(artifact.get("name", ""))

        # Check that action refs are valid
        for i, action in enumerate(diagnosis.next_actions):
            if action.ref and action.ref not in valid_refs:
                # Allow layer names as refs
                if action.ref in ["l0_hash", "l1_commitment", "l2_constraint", "l3_proximity", "l4_receipt"]:
                    continue
                # Allow row indices
                if action.ref.isdigit() or action.ref.startswith("row_"):
                    continue
                # Allow decision values as refs for DECISION actions
                if action.action_type == ActionType.DECISION:
                    try:
                        DecisionType(action.ref)
                        continue
                    except ValueError:
                        pass
                # Allow special refs
                if action.ref in ["audit_timeline", "replay_trace", "original_trace", "capsule_commitment", "row_root"]:
                    continue
                issues.append(f"Action {i}: ref '{action.ref}' not in evidence_index")

        score = 1.0 - (len(issues) * 0.25)
        return max(0.0, score), issues

    def _grade_correctness(
        self,
        diagnosis: DiagnosisResult,
        verify_report: dict[str, Any],
    ) -> tuple[float, list[str]]:
        """Grade diagnosis correctness against error code templates."""
        issues: list[str] = []

        # Check route matches exit_code
        exit_code = verify_report.get("exit_code", 0)
        expected_route = route_from_exit_code(exit_code)
        if diagnosis.route != expected_route:
            issues.append(f"Route mismatch: got {diagnosis.route.value}, expected {expected_route.value} for exit_code={exit_code}")

        # Get primary error code
        errors = verify_report.get("errors", [])
        if errors:
            primary_error = errors[0].get("code", "")

            # Check diagnosis identified correct primary error
            if diagnosis.primary_error_code != primary_error:
                issues.append(f"Primary error mismatch: got {diagnosis.primary_error_code}, expected {primary_error}")

            # Check against template
            template = get_template_for_error(primary_error)
            if template:
                # Check required action types
                action_types = [a.action_type for a in diagnosis.next_actions]
                for required_type in template.required_action_types:
                    if required_type not in action_types:
                        issues.append(f"Missing required action type: {required_type.value}")

                # Check decision is valid
                if diagnosis.decision and template.valid_decisions:
                    if diagnosis.decision not in template.valid_decisions:
                        valid_names = [d.value for d in template.valid_decisions]
                        issues.append(f"Invalid decision: {diagnosis.decision.value}. Valid: {valid_names}")

                # Check refs point to evidence_ref
                if template.required_refs_from_evidence:
                    error_evidence_ref = errors[0].get("evidence_ref")
                    if error_evidence_ref:
                        found_ref = any(a.ref == error_evidence_ref for a in diagnosis.next_actions)
                        if not found_ref:
                            issues.append(f"No action references evidence_ref: {error_evidence_ref}")

        # Check layer_failed matches first failing layer
        layers = verify_report.get("layers", {})
        for layer_name in ["l0_hash", "l1_commitment", "l2_constraint", "l3_proximity", "l4_receipt"]:
            layer_data = layers.get(layer_name, {})
            if layer_data.get("status") == "fail":
                if diagnosis.layer_failed != layer_name:
                    issues.append(f"layer_failed mismatch: got {diagnosis.layer_failed}, expected {layer_name}")
                break

        score = 1.0 - (len(issues) * 0.15)
        return max(0.0, score), issues


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def diagnose_from_verify_report(
    verify_report: dict[str, Any],
    evidence_index: dict[str, Any] | None = None,
) -> DiagnosisResult:
    """Create a diagnosis result from a verify report.

    This is the "correct" diagnosis that an agent should produce.
    Use this to generate training data or as a reference implementation.

    Args:
        verify_report: The verification report
        evidence_index: Optional evidence index for richer actions

    Returns:
        DiagnosisResult with correct route, actions, and decision
    """
    run_id = verify_report.get("run_id", "unknown")
    exit_code = verify_report.get("exit_code", 0)
    route = route_from_exit_code(exit_code)

    # Find failing layer
    layer_failed = None
    layers = verify_report.get("layers", {})
    for layer_name in ["l0_hash", "l1_commitment", "l2_constraint", "l3_proximity", "l4_receipt"]:
        layer_data = layers.get(layer_name, {})
        if layer_data.get("status") == "fail":
            layer_failed = layer_name
            break

    # Get primary error
    errors = verify_report.get("errors", [])
    primary_error_code = errors[0].get("code") if errors else None
    evidence_ref = errors[0].get("evidence_ref") if errors else None

    # Build actions from template
    actions: list[Action] = []
    decision = None

    if primary_error_code:
        template = get_template_for_error(primary_error_code)
        if template:
            for action_type in template.required_action_types:
                if action_type == ActionType.DECISION:
                    # Pick first valid decision
                    decision = template.valid_decisions[0] if template.valid_decisions else DecisionType.FIX_AND_RETRY
                    actions.append(Action(
                        action_type=ActionType.DECISION,
                        ref=decision.value,
                        why=template.description,
                        success_criteria=SuccessCriteria(SuccessCriteriaKind.DECISION_CONSISTENT),
                    ))
                elif action_type == ActionType.OPEN_ARTIFACT:
                    actions.append(Action(
                        action_type=ActionType.OPEN_ARTIFACT,
                        ref=evidence_ref,
                        why=f"Evidence ref from {primary_error_code}",
                        success_criteria=SuccessCriteria(SuccessCriteriaKind.ARTIFACT_EXISTS),
                    ))
                elif action_type == ActionType.COMPARE_HASH:
                    actions.append(Action(
                        action_type=ActionType.COMPARE_HASH,
                        ref=evidence_ref,
                        against="capsule_commitment",
                        why="Verify artifact integrity",
                        success_criteria=SuccessCriteria(SuccessCriteriaKind.HASH_MISMATCH_EXPLAINED),
                    ))
                elif action_type == ActionType.INSPECT_LAYER:
                    actions.append(Action(
                        action_type=ActionType.INSPECT_LAYER,
                        ref=layer_failed,
                        why=f"Inspect failing layer",
                        success_criteria=SuccessCriteria(SuccessCriteriaKind.LAYER_INSPECTED),
                    ))
                elif action_type == ActionType.VIEW_TIMELINE:
                    actions.append(Action(
                        action_type=ActionType.VIEW_TIMELINE,
                        ref="audit_timeline",
                        why="Check event chain",
                        success_criteria=SuccessCriteria(SuccessCriteriaKind.TIMELINE_RETRIEVED),
                    ))
                elif action_type == ActionType.OPEN_ROW:
                    actions.append(Action(
                        action_type=ActionType.OPEN_ROW,
                        ref="row_0",  # First row as default
                        why="Inspect row data",
                        success_criteria=SuccessCriteria(SuccessCriteriaKind.ROW_EXISTS),
                    ))
                elif action_type == ActionType.COMPARE_COMMITMENT:
                    actions.append(Action(
                        action_type=ActionType.COMPARE_COMMITMENT,
                        ref=evidence_ref,
                        against="row_root",
                        why="Verify commitment chain",
                        success_criteria=SuccessCriteria(SuccessCriteriaKind.HASH_MISMATCH_EXPLAINED),
                    ))
                elif action_type == ActionType.DIFF_REPLAY:
                    actions.append(Action(
                        action_type=ActionType.DIFF_REPLAY,
                        ref="replay_trace",
                        against="original_trace",
                        why="Compare replay results",
                        success_criteria=SuccessCriteria(SuccessCriteriaKind.HASH_MISMATCH_EXPLAINED),
                    ))
                elif action_type == ActionType.RERUN_AUDIT:
                    actions.append(Action(
                        action_type=ActionType.RERUN_AUDIT,
                        ref=run_id,
                        why="Re-check data availability",
                        success_criteria=SuccessCriteria(SuccessCriteriaKind.VERIFY_STATUS_CHANGED),
                    ))
    else:
        # No specific error - generic inspection
        actions = [
            Action(
                action_type=ActionType.INSPECT_LAYER,
                ref=layer_failed or "l4_receipt",
                why="Inspect for issues",
                success_criteria=SuccessCriteria(SuccessCriteriaKind.LAYER_INSPECTED),
            ),
            Action(
                action_type=ActionType.DECISION,
                ref=DecisionType.FIX_AND_RETRY.value,
                why="No specific error to diagnose",
                success_criteria=SuccessCriteria(SuccessCriteriaKind.DECISION_CONSISTENT),
            ),
        ]
        decision = DecisionType.FIX_AND_RETRY

    return DiagnosisResult(
        run_id=run_id,
        route=route,
        primary_error_code=primary_error_code,
        layer_failed=layer_failed,
        next_actions=actions,
        decision=decision,
        confidence=1.0,
    )


def grade_diagnosis(
    diagnosis: DiagnosisResult | dict[str, Any],
    verify_report: dict[str, Any],
    evidence_index: dict[str, Any] | None = None,
) -> GradeResult:
    """Grade a diagnosis against a verify report.

    This is the primary auto-grading function.

    Args:
        diagnosis: The diagnosis result (or dict)
        verify_report: The original verification report
        evidence_index: Optional evidence index

    Returns:
        GradeResult with validity and score
    """
    # Convert dict to DiagnosisResult if needed
    if isinstance(diagnosis, dict):
        diagnosis = _diagnosis_from_dict(diagnosis)

    grader = DiagnosisGrader(evidence_index)
    return grader.grade(diagnosis, verify_report)


def _diagnosis_from_dict(d: dict[str, Any]) -> DiagnosisResult:
    """Convert a dict to DiagnosisResult."""
    route = DiagnosisRoute(d.get("route", "PARSE_ERROR"))
    decision = DecisionType(d["decision"]) if d.get("decision") else None

    actions = []
    for action_dict in d.get("next_actions", []):
        action_type = ActionType(action_dict.get("type", "DECISION"))
        criteria = None
        if action_dict.get("success_criteria"):
            criteria = SuccessCriteria(
                kind=SuccessCriteriaKind(action_dict["success_criteria"].get("kind", "decision_consistent_with_error_code")),
                expected=action_dict["success_criteria"].get("expected"),
            )
        actions.append(Action(
            action_type=action_type,
            ref=action_dict.get("ref"),
            against=action_dict.get("against"),
            why=action_dict.get("why"),
            success_criteria=criteria,
        ))

    return DiagnosisResult(
        run_id=d.get("run_id", "unknown"),
        route=route,
        primary_error_code=d.get("primary_error_code"),
        layer_failed=d.get("layer_failed"),
        next_actions=actions,
        decision=decision,
        confidence=d.get("confidence", 1.0),
    )


__all__ = [
    "DiagnosisRoute",
    "ActionType",
    "SuccessCriteriaKind",
    "SuccessCriteria",
    "Action",
    "DecisionType",
    "DiagnosisResult",
    "ActionTemplate",
    "DiagnosisGrader",
    "GradeResult",
    "EXIT_CODE_TO_ROUTE",
    "ERROR_CODE_TEMPLATES",
    "route_from_exit_code",
    "get_template_for_error",
    "diagnose_from_verify_report",
    "grade_diagnosis",
]
