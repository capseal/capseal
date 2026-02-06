"""Verified Refactoring Engine for CapSeal.

This module implements a multi-agent refactoring pipeline where:
1. Review findings drive a refactor plan
2. Multiple agents generate patches asymmetrically (different agents for different tasks)
3. Every patch is verified: original_hash + patch = expected_hash
4. Final diff rollup proves the refactor matches the plan

Architecture:
    Review → RefactorPlan → [PatchAgent₁, PatchAgent₂, ...] → PatchValidator → PatchVerifier → DiffRollup

Key Design Principles:
- Patch validity is a first-class invariant: malformed patches are rejected before verification
- LLM outputs full file content; we generate diffs programmatically (never trust LLM-formatted diffs)
- No-op patches are explicit SKIPs with skip_reason=no_change
- Use git apply for validation (more robust than patch)
- Rollup semantics surface partial failures clearly with status_detail
- Repair loop: malformed patches get one retry with strict constraints

Each step produces a receipt that chains to the previous step's output hash.
"""
from __future__ import annotations

import concurrent.futures
import datetime
import difflib
import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from bef_zk.capsule.workflow_engine import (
    AgentPacket,
    NodeResult,
    NodeSpec,
    NodeExecutor,
    MemoizationCache,
    sha256_str,
    sha256_json,
    sha256_file,
    canonical_json,
    get_env_memoization_key,
    get_environment_fingerprint,
    sort_findings,
    NODE_DETERMINISM,
    Determinism,
    SkipReason,
    RequiredMode,
)


# =============================================================================
# Schema Versions
# =============================================================================

REFACTOR_PLAN_SCHEMA = "refactor_plan_v1"
PATCH_PACKET_SCHEMA = "patch_packet_v2"  # v2: machine-generated diffs, validity checks
PATCH_VERIFICATION_SCHEMA = "patch_verification_v1"
DIFF_ROLLUP_SCHEMA = "diff_rollup_v2"  # v2: status_detail, explicit SKIPs


# =============================================================================
# Patch Status (first-class invariants)
# =============================================================================

class PatchStatus(Enum):
    """Patch generation status."""
    VALID = "valid"           # Patch applies cleanly, makes changes
    NO_CHANGE = "no_change"   # Patch is empty / no-op
    MALFORMED = "malformed"   # Patch failed git apply --check
    REPAIR_PENDING = "repair_pending"  # Waiting for repair attempt
    REPAIR_FAILED = "repair_failed"    # Repair also failed
    FILE_NOT_FOUND = "file_not_found"  # Source file doesn't exist


class PatchSkipReason(Enum):
    """Reason a patch was skipped."""
    NO_CHANGE = "no_change"
    FILE_NOT_FOUND = "file_not_found"
    MALFORMED_AFTER_REPAIR = "malformed_after_repair"
    NOT_APPLICABLE = "not_applicable"
    FINDING_FALSE_POSITIVE = "finding_false_positive"  # Proven false positive
    ALREADY_MITIGATED = "already_mitigated"  # Code is already safe
    WAIVER_GRANTED = "waiver_granted"  # Policy-bound waiver


class NoChangeDisposition(Enum):
    """Disposition for NO_CHANGE patches (must be proven, not assumed)."""
    UNPROVEN = "unproven"  # Default - will be treated as FAIL for security_fix
    FALSE_POSITIVE = "false_positive"  # Finding is incorrect
    ALREADY_MITIGATED = "already_mitigated"  # Code already handles the issue
    WAIVER = "waiver"  # Explicit policy waiver
    SUPPRESSION_ADDED = "suppression_added"  # Added semgrep suppression with justification


@dataclass
class SpanAnchor:
    """Immutable anchor for relocating evidence even if lines shift."""
    start_line: int
    end_line: int
    snippet_hash: str  # SHA256 of the snippet content


@dataclass
class NoChangeProof:
    """Proof that a NO_CHANGE is legitimate (required for security_fix category).

    v4: Hash-bound proofs - evidence is immutable and relocatable:
    - file_pre_hash: Hash of the file when proof was generated
    - evidence_chunk_hash: Hash of the evidence snippet (from trace chunk store if available)
    - span_anchor: (start_line, end_line, snippet_hash) for relocation
    """
    disposition: str  # NoChangeDisposition value
    justification: str  # Human-readable explanation

    # v4: Hash-bound evidence (immutable)
    file_pre_hash: str = ""  # Hash of file when proof was generated
    evidence_chunk_hash: str = ""  # Hash of evidence snippet
    span_anchor: Optional[SpanAnchor] = None  # For relocating evidence

    # Legacy fields (kept for compatibility but less trusted)
    evidence_lines: list[int] = field(default_factory=list)  # Line numbers showing mitigation
    evidence_snippet: str = ""  # Code snippet proving safety

    # Waiver/suppression fields
    suppression_comment: str = ""  # If suppression was added
    waiver_policy_ref: str = ""  # Policy reference if waiver
    waiver_artifact_hash: str = ""  # Hash of traced waiver artifact

    def to_dict(self) -> dict:
        return {
            "disposition": self.disposition,
            "justification": self.justification,
            "file_pre_hash": self.file_pre_hash,
            "evidence_chunk_hash": self.evidence_chunk_hash,
            "span_anchor": {
                "start_line": self.span_anchor.start_line,
                "end_line": self.span_anchor.end_line,
                "snippet_hash": self.span_anchor.snippet_hash,
            } if self.span_anchor else None,
            "evidence_lines": self.evidence_lines,
            "evidence_snippet": self.evidence_snippet,
            "suppression_comment": self.suppression_comment,
            "waiver_policy_ref": self.waiver_policy_ref,
            "waiver_artifact_hash": self.waiver_artifact_hash,
        }

    def verify_integrity(self, current_file_content: str) -> tuple[bool, str]:
        """Verify proof is still valid against current file content.

        Returns (is_valid, error_message).
        """
        current_hash = sha256_str(current_file_content)

        # Check file hasn't changed
        if self.file_pre_hash and current_hash != self.file_pre_hash:
            return False, f"File changed: was {self.file_pre_hash[:16]}..., now {current_hash[:16]}..."

        # Check evidence snippet can be found and matches
        if self.span_anchor:
            lines = current_file_content.split('\n')
            start = self.span_anchor.start_line - 1  # Convert to 0-based
            end = self.span_anchor.end_line
            if start >= 0 and end <= len(lines):
                current_snippet = '\n'.join(lines[start:end])
                current_snippet_hash = sha256_str(current_snippet)
                if current_snippet_hash != self.span_anchor.snippet_hash:
                    return False, f"Evidence snippet changed at lines {self.span_anchor.start_line}-{self.span_anchor.end_line}"

        return True, ""


@dataclass
class TargetIdentity:
    """Stable target identity for finding delta checks.

    Fingerprints alone are unreliable because they change when code changes.
    This provides multiple keys for stable matching:
    - rule_id: The semgrep rule that triggered
    - file_path: Normalized file path
    - match_span: (start_line, end_line) of the match
    - matched_snippet_hash: Hash of the matched code region

    Delta success = no finding with same rule_id overlaps span AND
                    no finding matches snippet hash in that region.
    """
    rule_id: str
    file_path: str
    match_span: tuple[int, int]  # (start_line, end_line)
    matched_snippet_hash: str  # Hash of code region that was matched
    fingerprint: str = ""  # Optional: original semgrep fingerprint for compat

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "file_path": self.file_path,
            "match_span": list(self.match_span),
            "matched_snippet_hash": self.matched_snippet_hash,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_finding(cls, finding: dict, file_content: str = "") -> "TargetIdentity":
        """Build target identity from a semgrep finding."""
        rule_id = finding.get("check_id", finding.get("rule_id", ""))
        file_path = finding.get("path", finding.get("file_path", ""))
        start_line = finding.get("start", {}).get("line", finding.get("start_line", 0))
        end_line = finding.get("end", {}).get("line", finding.get("end_line", start_line))

        # Compute snippet hash from file content if available
        snippet_hash = ""
        if file_content and start_line > 0:
            lines = file_content.split('\n')
            if end_line <= len(lines):
                snippet = '\n'.join(lines[start_line-1:end_line])
                snippet_hash = sha256_str(snippet)

        # Compute fingerprint for compat
        fingerprint = sha256_json({
            "rule_id": rule_id,
            "path": file_path,
            "start_line": start_line,
            "message": finding.get("extra", {}).get("message", finding.get("message", "")),
        })[:16]

        return cls(
            rule_id=rule_id,
            file_path=file_path,
            match_span=(start_line, end_line),
            matched_snippet_hash=snippet_hash,
            fingerprint=fingerprint,
        )


# =============================================================================
# Two-Axis Status Model (v4: separate correctness from value)
# =============================================================================

class ApplyStatus(Enum):
    """Did the patch apply correctly? (Pipeline correctness axis)"""
    APPLIES = "applies"  # Patch applies cleanly, produces expected hash
    DOES_NOT_APPLY = "does_not_apply"  # git apply failed
    MALFORMED = "malformed"  # Patch is syntactically invalid
    TIMEOUT = "timeout"  # LLM or git operation timed out
    FILE_NOT_FOUND = "file_not_found"  # Source file doesn't exist


class EffectStatus(Enum):
    """Did the patch achieve the goal? (Pipeline value axis)"""
    RESOLVES_TARGET = "resolves_target"  # All targeted findings resolved, no regression
    PARTIAL = "partial"  # Some findings resolved, but not all
    NO_EFFECT = "no_effect"  # Findings unchanged (including NO_CHANGE)
    REGRESSION = "regression"  # New warnings introduced


@dataclass
class PatchOutcome:
    """Two-axis outcome for a patch (v4).

    This separates:
    - apply_status: Did the machinery work? (receipts verify, patch applies)
    - effect_status: Did we achieve the goal? (findings resolved)

    Pipeline "correctness" = apply_status is good + receipts verify
    Pipeline "value" = effect_status is RESOLVES_TARGET
    """
    apply_status: ApplyStatus
    effect_status: EffectStatus
    apply_error: str = ""
    effect_details: str = ""

    def to_dict(self) -> dict:
        return {
            "apply_status": self.apply_status.value,
            "effect_status": self.effect_status.value,
            "apply_error": self.apply_error,
            "effect_details": self.effect_details,
        }

    @property
    def is_mechanically_correct(self) -> bool:
        """Pipeline correctness: patch applies and verifies."""
        return self.apply_status == ApplyStatus.APPLIES

    @property
    def is_valuable(self) -> bool:
        """Pipeline value: actually fixed something."""
        return self.effect_status == EffectStatus.RESOLVES_TARGET

    @property
    def legacy_status(self) -> str:
        """Map to legacy VALID/SKIP/FAIL for compatibility."""
        if self.apply_status == ApplyStatus.APPLIES:
            if self.effect_status in (EffectStatus.RESOLVES_TARGET, EffectStatus.PARTIAL):
                return "VALID"
            else:
                return "VALID"  # Applied but no effect - still valid mechanically
        elif self.apply_status == ApplyStatus.FILE_NOT_FOUND:
            return "SKIP"
        elif self.effect_status == EffectStatus.NO_EFFECT:
            return "SKIP"
        else:
            return "FAIL"


@dataclass
class FindingDelta:
    """Result of checking if findings were resolved by a patch.

    v4: Uses stable target identities instead of just fingerprints.
    """
    # Target identities (stable)
    targeted_identities: list[TargetIdentity] = field(default_factory=list)
    resolved_identities: list[TargetIdentity] = field(default_factory=list)
    remaining_identities: list[TargetIdentity] = field(default_factory=list)

    # Legacy fingerprint tracking (for compat)
    resolved_fingerprints: list[str] = field(default_factory=list)
    remaining_fingerprints: list[str] = field(default_factory=list)

    # New findings (regressions)
    new_findings: list[dict] = field(default_factory=list)  # New warnings introduced

    # Summary flags
    all_targeted_resolved: bool = False
    no_new_warnings: bool = True

    def to_dict(self) -> dict:
        return {
            "targeted_identities": [t.to_dict() for t in self.targeted_identities],
            "resolved_identities": [t.to_dict() for t in self.resolved_identities],
            "remaining_identities": [t.to_dict() for t in self.remaining_identities],
            "resolved_fingerprints": self.resolved_fingerprints,
            "remaining_fingerprints": self.remaining_fingerprints,
            "new_findings": self.new_findings,
            "all_targeted_resolved": self.all_targeted_resolved,
            "no_new_warnings": self.no_new_warnings,
        }

    def compute_effect_status(self) -> EffectStatus:
        """Compute effect status from delta results."""
        if not self.no_new_warnings:
            return EffectStatus.REGRESSION
        if self.all_targeted_resolved:
            return EffectStatus.RESOLVES_TARGET
        if self.resolved_identities or self.resolved_fingerprints:
            return EffectStatus.PARTIAL
        return EffectStatus.NO_EFFECT


@dataclass
class WaiverArtifact:
    """Policy-bound waiver artifact (must be traced and reviewed).

    For suppression/waiver to count as a valid "fix", it must:
    1. Be defined in a traced policy file (waivers.yml, .security-waivers.json)
    2. Have required fields: owner, expiry, reason, scope, severity
    3. Be hashed into the receipt chain
    """
    waiver_id: str
    owner: str  # Who approved this waiver (email or @handle)
    expiry: str  # ISO date when waiver expires (empty = permanent, discouraged)
    reason: str  # Why this is acceptable (detailed)
    scope: str  # What is waived: file path, rule ID, or specific finding
    severity: str  # original, accepted (e.g., "high->accepted" or "medium->low")
    policy_file: str = ""  # Path to policy file where this is defined
    artifact_hash: str = ""  # Hash of the waiver entry for receipt chain
    reviewed_by: list[str] = field(default_factory=list)  # Reviewers who approved
    created_at: str = ""
    reviewed_at: str = ""

    def to_dict(self) -> dict:
        return {
            "waiver_id": self.waiver_id,
            "owner": self.owner,
            "expiry": self.expiry,
            "reason": self.reason,
            "scope": self.scope,
            "severity": self.severity,
            "policy_file": self.policy_file,
            "artifact_hash": self.artifact_hash,
            "reviewed_by": self.reviewed_by,
            "created_at": self.created_at,
            "reviewed_at": self.reviewed_at,
        }

    def is_valid(self) -> tuple[bool, str]:
        """Check if waiver is valid and not expired."""
        if not self.owner:
            return False, "Waiver missing owner"
        if not self.reason or len(self.reason) < 20:
            return False, "Waiver reason too short (min 20 chars)"
        if not self.scope:
            return False, "Waiver missing scope"

        # Check expiry
        if self.expiry:
            try:
                expiry_date = datetime.datetime.fromisoformat(self.expiry.replace('Z', '+00:00'))
                if expiry_date < datetime.datetime.now(datetime.timezone.utc):
                    return False, f"Waiver expired on {self.expiry}"
            except ValueError:
                return False, f"Invalid expiry date format: {self.expiry}"

        return True, ""


@dataclass
class DiffShapeSanity:
    """Sanity checks on diff shape to catch pathological patches.

    v4: Limits are conditional on item category:
    - security_fix on complex functions: looser limits (allow bigger rewrites)
    - plugin loaders, __init__.py: strict limits
    - Default: moderate limits
    """
    passed: bool = True
    excessive_deletions: bool = False
    out_of_scope_files: list[str] = field(default_factory=list)
    lines_deleted: int = 0
    lines_added: int = 0
    deletion_ratio: float = 0.0  # deleted / (deleted + added)
    errors: list[str] = field(default_factory=list)

    # v4: Category-adjusted limits used
    limits_profile: str = "default"  # "strict", "default", "loose"
    max_deletion_ratio_used: float = 0.8
    max_absolute_deletions_used: int = 500

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "excessive_deletions": self.excessive_deletions,
            "out_of_scope_files": self.out_of_scope_files,
            "lines_deleted": self.lines_deleted,
            "lines_added": self.lines_added,
            "deletion_ratio": self.deletion_ratio,
            "errors": self.errors,
            "limits_profile": self.limits_profile,
            "max_deletion_ratio_used": self.max_deletion_ratio_used,
            "max_absolute_deletions_used": self.max_absolute_deletions_used,
        }


@dataclass
class PatchValidation:
    """Result of patch validity check (runs before verification)."""
    patch_id: str
    is_valid: bool
    status: str  # PatchStatus value
    skip_reason: Optional[str] = None  # PatchSkipReason value if skipped
    git_apply_output: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "is_valid": self.is_valid,
            "status": self.status,
            "skip_reason": self.skip_reason,
            "git_apply_output": self.git_apply_output,
            "error": self.error,
        }


# =============================================================================
# Refactor Plan
# =============================================================================

class RefactorCategory(Enum):
    """Categories of refactoring work."""
    SECURITY_FIX = "security_fix"
    BUG_FIX = "bug_fix"
    CODE_SMELL = "code_smell"
    PERFORMANCE = "performance"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    TYPE_ANNOTATION = "type_annotation"
    DEPENDENCY_UPDATE = "dependency_update"


@dataclass
class RefactorItem:
    """A single refactoring action."""
    item_id: str
    category: str
    priority: int  # 1=critical, 2=high, 3=medium, 4=low
    file_path: str
    description: str
    finding_fingerprints: list[str]  # Links to review findings
    suggested_change: str
    estimated_complexity: str  # "trivial", "simple", "moderate", "complex"
    dependencies: list[str] = field(default_factory=list)  # Other item_ids this depends on

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "priority": self.priority,
            "file_path": self.file_path,
            "description": self.description,
            "finding_fingerprints": self.finding_fingerprints,
            "suggested_change": self.suggested_change,
            "estimated_complexity": self.estimated_complexity,
            "dependencies": self.dependencies,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RefactorItem":
        return cls(
            item_id=data.get("item_id", ""),
            category=data.get("category", ""),
            priority=data.get("priority", 4),
            file_path=data.get("file_path", ""),
            description=data.get("description", ""),
            finding_fingerprints=data.get("finding_fingerprints", []),
            suggested_change=data.get("suggested_change", ""),
            estimated_complexity=data.get("estimated_complexity", "moderate"),
            dependencies=data.get("dependencies", []),
        )


@dataclass
class RefactorPlan:
    """Complete refactoring plan derived from review findings."""
    schema: str = REFACTOR_PLAN_SCHEMA
    trace_root: str = ""
    review_aggregate_hash: str = ""
    items: list[RefactorItem] = field(default_factory=list)
    execution_order: list[str] = field(default_factory=list)  # item_ids in order
    agent_assignments: dict = field(default_factory=dict)  # item_id -> agent_type
    total_files_affected: int = 0
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "trace_root": self.trace_root,
            "review_aggregate_hash": self.review_aggregate_hash,
            "items": [i.to_dict() for i in self.items],
            "execution_order": self.execution_order,
            "agent_assignments": self.agent_assignments,
            "total_files_affected": self.total_files_affected,
            "timestamp": self.timestamp,
            "plan_hash": self.compute_hash(),
        }

    def compute_hash(self) -> str:
        """Compute deterministic hash of plan content."""
        content = {
            "trace_root": self.trace_root,
            "review_aggregate_hash": self.review_aggregate_hash,
            "items": sorted([i.to_dict() for i in self.items], key=lambda x: x["item_id"]),
            "execution_order": self.execution_order,
            "agent_assignments": self.agent_assignments,
        }
        return sha256_json(content)


# =============================================================================
# Patch Packet
# =============================================================================

@dataclass
class PatchPacket:
    """A verified patch for a single file."""
    schema: str = PATCH_PACKET_SCHEMA
    patch_id: str = ""
    item_id: str = ""  # Links to RefactorItem
    file_path: str = ""

    # Original state
    original_hash: str = ""
    original_line_count: int = 0

    # Patch content
    patch_content: str = ""  # Unified diff format
    patch_hash: str = ""

    # Expected result
    expected_hash: str = ""
    expected_line_count: int = 0

    # Metadata
    agent_type: str = ""  # Which agent generated this
    agent_model: str = ""
    generation_prompt_hash: str = ""
    timestamp: str = ""

    # Change statistics
    lines_added: int = 0
    lines_removed: int = 0
    hunks: int = 0

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "patch_id": self.patch_id,
            "item_id": self.item_id,
            "file_path": self.file_path,
            "original_hash": self.original_hash,
            "original_line_count": self.original_line_count,
            "patch_content": self.patch_content,
            "patch_hash": self.patch_hash,
            "expected_hash": self.expected_hash,
            "expected_line_count": self.expected_line_count,
            "agent_type": self.agent_type,
            "agent_model": self.agent_model,
            "generation_prompt_hash": self.generation_prompt_hash,
            "timestamp": self.timestamp,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "hunks": self.hunks,
            "receipt_hash": self.compute_receipt_hash(),
        }

    def compute_receipt_hash(self) -> str:
        """Compute receipt binding original → patch → expected."""
        content = {
            "item_id": self.item_id,
            "file_path": self.file_path,
            "original_hash": self.original_hash,
            "patch_hash": self.patch_hash,
            "expected_hash": self.expected_hash,
            "agent_type": self.agent_type,
        }
        return sha256_json(content)


# =============================================================================
# Patch Verification
# =============================================================================

@dataclass
class PatchVerification:
    """Verification result for a patch."""
    patch_id: str
    verified: bool
    original_hash_match: bool
    patch_applies_cleanly: bool
    result_hash_match: bool
    actual_result_hash: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "patch_id": self.patch_id,
            "verified": self.verified,
            "original_hash_match": self.original_hash_match,
            "patch_applies_cleanly": self.patch_applies_cleanly,
            "result_hash_match": self.result_hash_match,
            "actual_result_hash": self.actual_result_hash,
            "error": self.error,
        }


# =============================================================================
# Diff Rollup
# =============================================================================

@dataclass
class StatusDetail:
    """Detailed status breakdown for rollup (v4: yield tracking + conflicts)."""
    total_patches: int = 0
    valid_patches: int = 0
    skipped_patches: int = 0
    failed_patches: int = 0
    repaired_patches: int = 0
    conflict_patches: int = 0  # v4: Patches with overlapping hunks
    valid_patch_ids: list[str] = field(default_factory=list)
    skipped_patch_ids: list[str] = field(default_factory=list)
    failed_patch_ids: list[str] = field(default_factory=list)
    conflict_patch_ids: list[str] = field(default_factory=list)  # v4
    skip_reasons: dict = field(default_factory=dict)  # patch_id -> reason

    # v3: Yield metrics (measuring actual impact)
    findings_targeted: int = 0  # Total finding fingerprints we tried to fix
    findings_resolved: int = 0  # Finding fingerprints actually resolved
    findings_remaining: int = 0  # Finding fingerprints still present
    new_warnings_introduced: int = 0  # New issues introduced by patches
    no_change_proven: int = 0  # NO_CHANGE with valid proof
    no_change_unproven: int = 0  # NO_CHANGE without proof (= FAIL for security)
    timeouts: int = 0  # LLM call timeouts
    sanity_failures: int = 0  # Diff-shape sanity check failures

    # v4: Additional metrics
    importability_failures: int = 0  # Patches that failed syntax check
    mechanically_correct: int = 0  # Patches where apply_status is APPLIES
    valuable: int = 0  # Patches where effect_status is RESOLVES_TARGET

    @property
    def yield_rate(self) -> float:
        """Percentage of plan items that became VALID patches."""
        if self.total_patches == 0:
            return 0.0
        return self.valid_patches / self.total_patches

    @property
    def resolution_rate(self) -> float:
        """Percentage of targeted findings that were resolved."""
        if self.findings_targeted == 0:
            return 0.0
        return self.findings_resolved / self.findings_targeted

    def to_dict(self) -> dict:
        return {
            "total_patches": self.total_patches,
            "valid_patches": self.valid_patches,
            "skipped_patches": self.skipped_patches,
            "failed_patches": self.failed_patches,
            "repaired_patches": self.repaired_patches,
            "conflict_patches": self.conflict_patches,
            "valid_patch_ids": self.valid_patch_ids,
            "skipped_patch_ids": self.skipped_patch_ids,
            "failed_patch_ids": self.failed_patch_ids,
            "conflict_patch_ids": self.conflict_patch_ids,
            "skip_reasons": self.skip_reasons,
            # v4: Yield metrics with two-axis breakdown
            "yield_metrics": {
                "findings_targeted": self.findings_targeted,
                "findings_resolved": self.findings_resolved,
                "findings_remaining": self.findings_remaining,
                "new_warnings_introduced": self.new_warnings_introduced,
                "no_change_proven": self.no_change_proven,
                "no_change_unproven": self.no_change_unproven,
                "timeouts": self.timeouts,
                "sanity_failures": self.sanity_failures,
                "importability_failures": self.importability_failures,
                "yield_rate": round(self.yield_rate, 3),
                "resolution_rate": round(self.resolution_rate, 3),
                # Two-axis summary
                "mechanically_correct": self.mechanically_correct,
                "valuable": self.valuable,
            },
        }


@dataclass
class DiffRollup:
    """Final verified diff with full provenance.

    v2 Changes:
    - status_detail: Explicit breakdown of VALID/SKIP/FAIL patches
    - combined_diff only includes VALID patches (no noise from no-ops)
    - all_verified means all required patches verified, not all patches
    """
    schema: str = DIFF_ROLLUP_SCHEMA
    trace_root: str = ""
    plan_hash: str = ""

    # Patches (only VALID ones)
    patches: list[dict] = field(default_factory=list)  # PatchPacket dicts
    verifications: list[dict] = field(default_factory=list)  # PatchVerification dicts

    # All patch results (including SKIPs)
    all_patch_results: list[dict] = field(default_factory=list)

    # Combined diff (ONLY VALID patches, no comments)
    combined_diff: str = ""
    combined_diff_hash: str = ""

    # Statistics (from VALID patches only)
    total_files_modified: int = 0
    total_lines_added: int = 0
    total_lines_removed: int = 0

    # Status detail (v2: explicit partial failure tracking)
    status_detail: StatusDetail = field(default_factory=StatusDetail)

    # Verification summary
    all_verified: bool = False  # True if all VALID patches verified
    failed_patches: list[str] = field(default_factory=list)

    # Provenance chain
    provenance: dict = field(default_factory=dict)

    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "trace_root": self.trace_root,
            "plan_hash": self.plan_hash,
            "patches": self.patches,
            "verifications": self.verifications,
            "all_patch_results": self.all_patch_results,
            "combined_diff": self.combined_diff,
            "combined_diff_hash": self.combined_diff_hash,
            "total_files_modified": self.total_files_modified,
            "total_lines_added": self.total_lines_added,
            "total_lines_removed": self.total_lines_removed,
            "status_detail": self.status_detail.to_dict(),
            "all_verified": self.all_verified,
            "failed_patches": self.failed_patches,
            "provenance": self.provenance,
            "timestamp": self.timestamp,
            "rollup_hash": self.compute_hash(),
        }

    def compute_hash(self) -> str:
        content = {
            "trace_root": self.trace_root,
            "plan_hash": self.plan_hash,
            "patch_receipt_hashes": sorted([p.get("receipt_hash", "") for p in self.patches]),
            "combined_diff_hash": self.combined_diff_hash,
            "all_verified": self.all_verified,
            "status_detail": self.status_detail.to_dict(),
        }
        return sha256_json(content)


# =============================================================================
# LLM Integration
# =============================================================================

def _call_llm(
    prompt: str,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 4000,
    timeout: int = 120,
) -> tuple[str, str]:
    """Call LLM and return (response, prompt_hash)."""
    from bef_zk.capsule.review_agent import call_llm_backend

    prompt_hash = sha256_str(prompt)
    response = call_llm_backend(
        provider=provider,
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return response, prompt_hash


# =============================================================================
# Refactor Plan Generator
# =============================================================================

def generate_refactor_plan(
    findings: list[dict],
    trace_root: str,
    aggregate_hash: str,
    project_context: str = "",
    provider: str = "openai",
    model: str = "gpt-4o-mini",
) -> RefactorPlan:
    """Generate a refactor plan from review findings."""

    # Build prompt
    findings_text = json.dumps(findings[:50], indent=2)  # Limit for context

    prompt = f"""You are a code refactoring planner. Given these review findings, create a structured refactoring plan.

## Review Findings
```json
{findings_text}
```

## Project Context
{project_context or "General codebase"}

## Task
Create a JSON refactoring plan with this structure:
{{
    "items": [
        {{
            "item_id": "REF-001",
            "category": "security_fix|bug_fix|code_smell|performance|style|documentation|type_annotation",
            "priority": 1-4 (1=critical),
            "file_path": "path/to/file.py",
            "description": "What needs to change and why",
            "finding_fingerprints": ["fingerprint1", ...],
            "suggested_change": "Specific code change suggestion",
            "estimated_complexity": "trivial|simple|moderate|complex",
            "dependencies": ["REF-002"] // if this must be done after another item
        }}
    ],
    "execution_order": ["REF-001", "REF-002", ...],
    "agent_assignments": {{
        "REF-001": "security_agent",
        "REF-002": "style_agent"
    }}
}}

Agent types available:
- security_agent: Security fixes, input validation, auth issues
- performance_agent: Performance optimizations, caching
- style_agent: Code style, formatting, naming
- type_agent: Type annotations, type fixes
- doc_agent: Documentation, comments
- general_agent: Everything else

Output ONLY the JSON, no other text.
"""

    response, _ = _call_llm(prompt, provider, model, temperature=0.0)

    # Parse response
    try:
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(response)
    except json.JSONDecodeError:
        # Fallback: create minimal plan
        data = {"items": [], "execution_order": [], "agent_assignments": {}}

    # Build plan
    plan = RefactorPlan(
        trace_root=trace_root,
        review_aggregate_hash=aggregate_hash,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )

    seen_files = set()
    for item_data in data.get("items", []):
        item = RefactorItem.from_dict(item_data)
        plan.items.append(item)
        if item.file_path:
            seen_files.add(item.file_path)

    plan.execution_order = data.get("execution_order", [i.item_id for i in plan.items])
    plan.agent_assignments = data.get("agent_assignments", {})
    plan.total_files_affected = len(seen_files)

    return plan


# =============================================================================
# Patch Validity Check (First-Class Invariant)
# =============================================================================

def validate_patch_with_git(
    patch_content: str,
    project_dir: Path,
    file_path: str,
) -> PatchValidation:
    """Validate patch using git apply --check. Returns validation result."""
    if not patch_content or not patch_content.strip():
        return PatchValidation(
            patch_id="",
            is_valid=False,
            status=PatchStatus.NO_CHANGE.value,
            skip_reason=PatchSkipReason.NO_CHANGE.value,
        )

    try:
        # Run git apply --check
        result = subprocess.run(
            ["git", "apply", "--check", "--unsafe-paths", "-"],
            input=patch_content.encode(),
            capture_output=True,
            cwd=project_dir,
            timeout=30,
        )

        if result.returncode == 0:
            return PatchValidation(
                patch_id="",
                is_valid=True,
                status=PatchStatus.VALID.value,
                git_apply_output=result.stdout.decode() if result.stdout else "",
            )
        else:
            return PatchValidation(
                patch_id="",
                is_valid=False,
                status=PatchStatus.MALFORMED.value,
                git_apply_output=result.stderr.decode() if result.stderr else "",
                error=f"git apply --check failed: {result.stderr.decode()[:500]}",
            )
    except subprocess.TimeoutExpired:
        return PatchValidation(
            patch_id="",
            is_valid=False,
            status=PatchStatus.MALFORMED.value,
            error="git apply --check timed out",
        )
    except FileNotFoundError:
        # git not available, fall back to patch --dry-run
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
                f.write(patch_content)
                patch_file = f.name

            result = subprocess.run(
                ["patch", "--dry-run", "-p1", "-i", patch_file],
                capture_output=True,
                cwd=project_dir,
                timeout=30,
            )
            os.unlink(patch_file)

            if result.returncode == 0:
                return PatchValidation(
                    patch_id="",
                    is_valid=True,
                    status=PatchStatus.VALID.value,
                )
            else:
                return PatchValidation(
                    patch_id="",
                    is_valid=False,
                    status=PatchStatus.MALFORMED.value,
                    error=f"patch --dry-run failed: {result.stderr.decode()[:500]}",
                )
        except Exception as e:
            return PatchValidation(
                patch_id="",
                is_valid=False,
                status=PatchStatus.MALFORMED.value,
                error=f"Validation failed: {e}",
            )


# =============================================================================
# Finding-Delta Gate (verify findings are actually resolved)
# =============================================================================

def run_finding_delta_check(
    patch_content: str,
    original_content: str,
    file_path: str,
    targeted_fingerprints: list[str],
    project_dir: Path,
    semgrep_rules: list[str] = None,
) -> FindingDelta:
    """Check if targeted findings are resolved by the patch.

    This is the "did we actually reduce risk" gate:
    1. Apply patch to get new content
    2. Run semgrep on the patched file
    3. Check if targeted finding fingerprints are gone
    4. Check if new warnings appeared
    """
    delta = FindingDelta()

    if not patch_content or not patch_content.strip():
        # No change - findings cannot be resolved
        delta.remaining_fingerprints = targeted_fingerprints.copy()
        delta.all_targeted_resolved = False
        return delta

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create file structure
            patched_file = tmpdir_path / file_path
            patched_file.parent.mkdir(parents=True, exist_ok=True)
            patched_file.write_text(original_content)

            # Apply patch
            result = subprocess.run(
                ["git", "init"],
                cwd=tmpdir,
                capture_output=True,
            )
            subprocess.run(
                ["git", "add", "."],
                cwd=tmpdir,
                capture_output=True,
            )

            result = subprocess.run(
                ["git", "apply", "-"],
                input=patch_content.encode(),
                capture_output=True,
                cwd=tmpdir,
            )

            if result.returncode != 0:
                delta.remaining_fingerprints = targeted_fingerprints.copy()
                return delta

            patched_content = patched_file.read_text()

            # Run semgrep on patched file
            semgrep_result = subprocess.run(
                ["semgrep", "--config=auto", "--json", str(patched_file)],
                capture_output=True,
                cwd=tmpdir,
                timeout=60,
            )

            if semgrep_result.returncode not in (0, 1):  # 1 = findings found
                # Semgrep failed, can't verify
                delta.remaining_fingerprints = targeted_fingerprints.copy()
                return delta

            try:
                findings_data = json.loads(semgrep_result.stdout.decode())
                new_findings = findings_data.get("results", [])
            except json.JSONDecodeError:
                new_findings = []

            # Compute fingerprints of new findings
            new_fingerprints = set()
            for finding in new_findings:
                fp = sha256_json({
                    "rule_id": finding.get("check_id", ""),
                    "path": finding.get("path", ""),
                    "start_line": finding.get("start", {}).get("line", 0),
                    "message": finding.get("extra", {}).get("message", ""),
                })[:16]
                new_fingerprints.add(fp)

            # Check which targeted fingerprints are resolved
            for fp in targeted_fingerprints:
                # Use prefix matching for flexibility
                fp_prefix = fp[:16] if len(fp) > 16 else fp
                if not any(nfp.startswith(fp_prefix) for nfp in new_fingerprints):
                    delta.resolved_fingerprints.append(fp)
                else:
                    delta.remaining_fingerprints.append(fp)

            # Check for new findings (not in original targeted set)
            original_prefixes = {fp[:16] for fp in targeted_fingerprints}
            for finding in new_findings:
                fp = sha256_json({
                    "rule_id": finding.get("check_id", ""),
                    "path": finding.get("path", ""),
                    "start_line": finding.get("start", {}).get("line", 0),
                    "message": finding.get("extra", {}).get("message", ""),
                })[:16]
                if fp not in original_prefixes:
                    delta.new_findings.append({
                        "rule_id": finding.get("check_id", ""),
                        "message": finding.get("extra", {}).get("message", "")[:200],
                        "line": finding.get("start", {}).get("line", 0),
                    })

            delta.all_targeted_resolved = len(delta.remaining_fingerprints) == 0
            delta.no_new_warnings = len(delta.new_findings) == 0

    except subprocess.TimeoutExpired:
        delta.remaining_fingerprints = targeted_fingerprints.copy()
    except FileNotFoundError:
        # semgrep not available
        delta.remaining_fingerprints = targeted_fingerprints.copy()
    except Exception as e:
        delta.remaining_fingerprints = targeted_fingerprints.copy()

    return delta


# =============================================================================
# Diff-Shape Sanity Checks
# =============================================================================

def _get_sanity_limits(file_path: str, category: str, description: str = "") -> tuple[str, float, int]:
    """Get sanity check limits based on file type and item category.

    Returns (profile_name, max_deletion_ratio, max_absolute_deletions).

    Profiles:
    - "strict": __init__.py, plugin loaders, config files (deletion_ratio=0.5, abs=100)
    - "default": normal files (deletion_ratio=0.8, abs=500)
    - "loose": security fixes in complex functions, large refactors (deletion_ratio=0.9, abs=1000)
    """
    file_name = Path(file_path).name.lower()

    # Strict: plugin loaders, __init__, config
    if file_name == "__init__.py":
        return "strict", 0.5, 100
    if "plugin" in file_path.lower() or "loader" in file_path.lower():
        return "strict", 0.5, 150
    if file_name in ("config.py", "settings.py", "constants.py"):
        return "strict", 0.6, 200

    # Loose: security fixes that may need larger rewrites
    if category == "security_fix":
        # Check description for complexity indicators
        complex_indicators = ["sql", "inject", "cache", "auth", "session", "crypto"]
        if any(ind in description.lower() for ind in complex_indicators):
            return "loose", 0.9, 1000

    # Default
    return "default", 0.8, 500


def check_diff_shape_sanity(
    patch_content: str,
    expected_file_path: str,
    category: str = "",
    description: str = "",
    max_deletion_ratio: Optional[float] = None,
    max_absolute_deletions: Optional[int] = None,
) -> DiffShapeSanity:
    """Check diff shape for pathological patterns.

    v4: Limits are conditional on file type and item category:
    - strict: __init__.py, plugin loaders (smaller changes only)
    - default: normal files
    - loose: security fixes in complex functions (allow bigger rewrites)

    Rejects:
    - Excessive deletions (> max_deletion_ratio or > max_absolute_deletions)
    - Changes to files outside expected scope
    - Patches that look like "delete everything"
    """
    # Get category-conditional limits
    if max_deletion_ratio is None or max_absolute_deletions is None:
        profile, ratio, abs_max = _get_sanity_limits(expected_file_path, category, description)
        max_deletion_ratio = max_deletion_ratio or ratio
        max_absolute_deletions = max_absolute_deletions or abs_max
    else:
        profile = "custom"

    sanity = DiffShapeSanity()
    sanity.limits_profile = profile
    sanity.max_deletion_ratio_used = max_deletion_ratio
    sanity.max_absolute_deletions_used = max_absolute_deletions

    if not patch_content or not patch_content.strip():
        return sanity  # Empty patch is "sane" (will be handled as NO_CHANGE)

    lines = patch_content.split('\n')

    # Count additions and deletions
    additions = 0
    deletions = 0
    current_file = None
    touched_files = set()

    for line in lines:
        if line.startswith('--- a/'):
            current_file = line[6:].strip()
        elif line.startswith('+++ b/'):
            current_file = line[6:].strip()
            if current_file:
                touched_files.add(current_file)
        elif line.startswith('+') and not line.startswith('+++'):
            additions += 1
        elif line.startswith('-') and not line.startswith('---'):
            deletions += 1

    sanity.lines_added = additions
    sanity.lines_deleted = deletions

    total = additions + deletions
    if total > 0:
        sanity.deletion_ratio = deletions / total

    # Check for excessive deletions
    if deletions > max_absolute_deletions:
        sanity.passed = False
        sanity.excessive_deletions = True
        sanity.errors.append(f"Excessive deletions: {deletions} lines (max: {max_absolute_deletions} for {profile} profile)")

    if sanity.deletion_ratio > max_deletion_ratio and deletions > 10:
        sanity.passed = False
        sanity.excessive_deletions = True
        sanity.errors.append(f"High deletion ratio: {sanity.deletion_ratio:.1%} (max: {max_deletion_ratio:.0%} for {profile} profile)")

    # Check for out-of-scope files
    expected_normalized = expected_file_path.lstrip('/')
    for touched in touched_files:
        touched_normalized = touched.lstrip('/')
        if touched_normalized != expected_normalized:
            sanity.out_of_scope_files.append(touched)

    if sanity.out_of_scope_files:
        sanity.passed = False
        sanity.errors.append(f"Patch touches out-of-scope files: {sanity.out_of_scope_files}")

    return sanity


# =============================================================================
# NO_CHANGE Proof Validation
# =============================================================================

def validate_no_change_proof(
    item: "RefactorItem",
    proof: Optional[NoChangeProof],
    file_content: str,
) -> tuple[bool, str]:
    """Validate that a NO_CHANGE disposition is legitimate.

    For security_fix category, NO_CHANGE is only allowed if:
    1. FALSE_POSITIVE: Finding is incorrect (with evidence)
    2. ALREADY_MITIGATED: Code already handles it (with evidence)
    3. WAIVER: Explicit policy waiver (with reference)
    4. SUPPRESSION_ADDED: Added semgrep suppression with justification

    Returns (is_valid, error_message)
    """
    # Categories that REQUIRE proof for NO_CHANGE
    strict_categories = {"security_fix", "bug_fix"}

    if item.category not in strict_categories:
        # Non-strict categories can skip without proof
        return True, ""

    if not proof:
        return False, f"NO_CHANGE for {item.category} requires proof but none provided"

    if proof.disposition == NoChangeDisposition.UNPROVEN.value:
        return False, f"NO_CHANGE disposition is UNPROVEN - must provide evidence"

    if not proof.justification or len(proof.justification) < 20:
        return False, f"NO_CHANGE justification too short (min 20 chars)"

    # Validate specific dispositions
    if proof.disposition == NoChangeDisposition.FALSE_POSITIVE.value:
        if not proof.evidence_snippet:
            return False, "FALSE_POSITIVE requires evidence_snippet showing why finding is wrong"

    if proof.disposition == NoChangeDisposition.ALREADY_MITIGATED.value:
        if not proof.evidence_lines and not proof.evidence_snippet:
            return False, "ALREADY_MITIGATED requires evidence_lines or evidence_snippet"

    if proof.disposition == NoChangeDisposition.WAIVER.value:
        if not proof.waiver_policy_ref:
            return False, "WAIVER requires waiver_policy_ref"

    if proof.disposition == NoChangeDisposition.SUPPRESSION_ADDED.value:
        if not proof.suppression_comment:
            return False, "SUPPRESSION_ADDED requires suppression_comment"
        # Verify suppression is in the content
        if "# nosec" not in file_content and "# noqa" not in file_content and "nosemgrep" not in file_content:
            return False, "SUPPRESSION_ADDED but no suppression comment found in file"

    return True, ""


# =============================================================================
# Importability / Type Sanity Gate (v4)
# =============================================================================

@dataclass
class ImportabilitySanity:
    """Result of checking if patched code is importable/compilable."""
    passed: bool = True
    can_compile: bool = True  # py_compile succeeds
    can_import: bool = True  # import succeeds (optional, more strict)
    syntax_error: str = ""
    import_error: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "can_compile": self.can_compile,
            "can_import": self.can_import,
            "syntax_error": self.syntax_error,
            "import_error": self.import_error,
            "errors": self.errors,
        }


def check_importability(
    patched_content: str,
    file_path: str,
    check_import: bool = False,
) -> ImportabilitySanity:
    """Check if patched Python code is valid (compilable and optionally importable).

    This catches "LLM confidently broke syntax" failures cheaply.

    Args:
        patched_content: The patched file content
        file_path: Path to the file (for module name inference)
        check_import: If True, also try to import the module (slower, stricter)

    Returns:
        ImportabilitySanity with results
    """
    sanity = ImportabilitySanity()

    # Only check Python files
    if not file_path.endswith('.py'):
        return sanity

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(patched_content)
            tmp_path = f.name

        try:
            # Run py_compile for syntax check
            result = subprocess.run(
                ["python", "-m", "py_compile", tmp_path],
                capture_output=True,
                timeout=10,
            )

            if result.returncode != 0:
                sanity.passed = False
                sanity.can_compile = False
                error_output = result.stderr.decode()[:500]
                sanity.syntax_error = error_output
                sanity.errors.append(f"Syntax error: {error_output}")
                return sanity

            # Optional: try actual import
            if check_import:
                # Infer module name from path
                module_name = Path(file_path).stem
                result = subprocess.run(
                    ["python", "-c", f"import sys; sys.path.insert(0, '{Path(tmp_path).parent}'); import {module_name}"],
                    capture_output=True,
                    timeout=10,
                )

                if result.returncode != 0:
                    sanity.can_import = False
                    error_output = result.stderr.decode()[:500]
                    sanity.import_error = error_output
                    # Import failure is a warning, not a hard fail
                    sanity.errors.append(f"Import warning: {error_output}")
                    # Still pass if syntax is ok but import fails (may need context)

        finally:
            os.unlink(tmp_path)

    except subprocess.TimeoutExpired:
        sanity.errors.append("py_compile timed out")
    except Exception as e:
        sanity.errors.append(f"Importability check error: {str(e)[:200]}")

    return sanity


def _generate_diff_from_contents(
    original_content: str,
    new_content: str,
    file_path: str,
) -> str:
    """Generate unified diff from two file contents. Machine-generated, always valid."""
    original_lines = original_content.split('\n')
    new_lines = new_content.split('\n')

    diff_lines = list(difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm='',
    ))
    return '\n'.join(diff_lines)


# =============================================================================
# Patch Generator (LLM outputs content, we generate diff)
# =============================================================================

def _estimate_tokens(text: str) -> int:
    """Rough token estimate for prompt budgeting.

    This is a cheap heuristic (actual tokens vary by model):
    - ~4 chars per token for code
    - ~3 chars per token for prose
    - Add overhead for special tokens
    """
    # Simple heuristic: code is ~4 chars per token
    return len(text) // 4


def _find_enclosing_scope(lines: list[str], target_line: int) -> tuple[int, int, str]:
    """Find the enclosing function/class/method for a line.

    Returns (start_line, end_line, scope_type) where lines are 0-indexed.
    scope_type is 'function', 'class', 'method', or 'module'.
    """
    # Find enclosing function or class
    scope_start = 0
    scope_type = "module"

    # Search backwards for def/class
    for i in range(target_line, -1, -1):
        line = lines[i].strip()
        if line.startswith('def ') or line.startswith('async def '):
            scope_start = i
            scope_type = "function"
            break
        elif line.startswith('class '):
            scope_start = i
            scope_type = "class"
            break

    # Find end of scope by indentation
    if scope_type in ("function", "class"):
        base_indent = len(lines[scope_start]) - len(lines[scope_start].lstrip())
        scope_end = scope_start + 1
        for i in range(scope_start + 1, len(lines)):
            line = lines[i]
            if line.strip():  # Non-empty line
                indent = len(line) - len(line.lstrip())
                if indent <= base_indent and not line.strip().startswith(('#', '@')):
                    break
                scope_end = i + 1
            else:
                scope_end = i + 1
    else:
        scope_end = len(lines)

    return scope_start, scope_end, scope_type


def _extract_imports(file_content: str) -> str:
    """Extract all import statements from file."""
    lines = file_content.split('\n')
    imports = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(line)
        elif imports and (stripped.startswith(')') or stripped == ''):
            # Handle multi-line imports
            if not stripped:
                continue
            imports.append(line)
        elif stripped and not stripped.startswith('#') and imports:
            # We've moved past imports
            break
    return '\n'.join(imports)


def _extract_referenced_constants(file_content: str, start_line: int, end_line: int) -> str:
    """Extract module-level constants that might be referenced in the region."""
    lines = file_content.split('\n')
    region = '\n'.join(lines[start_line:end_line])

    constants = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for CONSTANT_NAME = ... patterns
        if '=' in stripped and not stripped.startswith(('def ', 'class ', '#', 'if ', 'for ', 'while ')):
            parts = stripped.split('=')
            if parts[0].strip().isupper() or parts[0].strip().upper() == parts[0].strip():
                # This is probably a constant
                const_name = parts[0].strip()
                if const_name in region and i < start_line:
                    constants.append(line)

    return '\n'.join(constants[:10])  # Limit to 10 constants


def _extract_relevant_context(file_content: str, item: RefactorItem, context_lines: int = 50) -> tuple[str, int, int]:
    """Extract relevant portion of file for the refactor item.

    Returns (context_snippet, start_line, end_line)
    """
    lines = file_content.split('\n')

    # Look for patterns related to the finding
    keywords = []
    if "import_module" in item.description.lower() or "importlib" in item.description.lower():
        keywords = ["importlib", "import_module", "__getattr__"]
    elif "sql" in item.description.lower():
        keywords = ["execute", "cursor", "query", "sql"]
    else:
        keywords = item.description.split()[:3]

    # Find lines containing keywords
    relevant_lines = []
    for i, line in enumerate(lines):
        if any(kw.lower() in line.lower() for kw in keywords):
            relevant_lines.append(i)

    if not relevant_lines:
        # Return first portion if no match
        start = 0
        end = min(context_lines * 2, len(lines))
    else:
        # Get range around relevant lines
        center = relevant_lines[len(relevant_lines) // 2]
        start = max(0, center - context_lines)
        end = min(len(lines), center + context_lines)

    snippet = '\n'.join(f"{i+1:4}: {lines[i]}" for i in range(start, end))
    return snippet, start + 1, end


def _extract_enhanced_context(
    file_content: str,
    item: RefactorItem,
    max_tokens: int = 3000,
) -> tuple[str, int, int, str]:
    """Extract enhanced context for region-focused edits.

    v4: Token-based context budgeting that includes:
    - Relevant imports
    - Full enclosing function/class
    - Referenced constants in that region

    Returns (context_snippet, start_line, end_line, context_type)
    """
    lines = file_content.split('\n')

    # Find the target lines using keywords
    keywords = []
    if "import_module" in item.description.lower() or "importlib" in item.description.lower():
        keywords = ["importlib", "import_module", "__getattr__"]
    elif "sql" in item.description.lower():
        keywords = ["execute", "cursor", "query", "sql"]
    elif "input" in item.description.lower() or "valid" in item.description.lower():
        keywords = ["input", "request", "param", "arg"]
    else:
        keywords = item.description.split()[:3]

    # Find relevant lines
    relevant_lines = []
    for i, line in enumerate(lines):
        if any(kw.lower() in line.lower() for kw in keywords):
            relevant_lines.append(i)

    if not relevant_lines:
        target_line = min(50, len(lines) // 2)
    else:
        target_line = relevant_lines[len(relevant_lines) // 2]

    # Find enclosing scope
    scope_start, scope_end, scope_type = _find_enclosing_scope(lines, target_line)

    # Build context with token budgeting
    context_parts = []
    used_tokens = 0

    # 1. Always include relevant imports
    imports = _extract_imports(file_content)
    import_tokens = _estimate_tokens(imports)
    if import_tokens < max_tokens * 0.15:  # Max 15% for imports
        context_parts.append(("imports", imports))
        used_tokens += import_tokens

    # 2. Include referenced constants
    constants = _extract_referenced_constants(file_content, scope_start, scope_end)
    const_tokens = _estimate_tokens(constants)
    if const_tokens < max_tokens * 0.1:  # Max 10% for constants
        context_parts.append(("constants", constants))
        used_tokens += const_tokens

    # 3. Include enclosing scope (main content)
    remaining_budget = max_tokens - used_tokens
    scope_content = '\n'.join(lines[scope_start:scope_end])
    scope_tokens = _estimate_tokens(scope_content)

    if scope_tokens <= remaining_budget:
        # Full scope fits
        context_parts.append(("scope", scope_content))
        start, end = scope_start, scope_end
    else:
        # Need to trim - focus on target region
        lines_budget = int(remaining_budget / 10)  # Rough: 10 tokens per line
        half = lines_budget // 2
        start = max(scope_start, target_line - half)
        end = min(scope_end, target_line + half)
        trimmed = '\n'.join(lines[start:end])
        context_parts.append(("region", trimmed))

    # Format snippet with line numbers
    snippet_lines = []
    for i in range(start, end):
        if i < len(lines):
            snippet_lines.append(f"{i+1:4}: {lines[i]}")

    snippet = '\n'.join(snippet_lines)

    return snippet, start + 1, end, scope_type


def generate_patch(
    item: RefactorItem,
    file_content: str,
    file_path: str,
    agent_type: str = "general_agent",
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    project_dir: Optional[Path] = None,
) -> tuple[PatchPacket, PatchValidation, Optional[NoChangeProof]]:
    """Generate a patch for a single refactor item.

    v4 Design: Token-based hybrid strategy (not just line counts):
    - Uses estimated prompt tokens to decide strategy
    - Small files (<4K tokens): LLM outputs complete file, we diff
    - Medium files (4K-12K tokens): Structured edits with enhanced context
    - Large files (>12K tokens): Region-focused with imports/constants

    Enhanced context includes:
    - Relevant imports (always)
    - Full enclosing function/class (when budget allows)
    - Referenced constants

    Key constraints:
    - LLM MUST change something or provide NO_CHANGE proof
    - Diffs are always machine-generated from actual content
    - Diff-shape sanity checks catch pathological outputs

    Returns (patch, validation, no_change_proof) tuple.
    """

    original_hash = sha256_str(file_content)
    original_lines = file_content.split('\n')
    no_change_proof = None

    # v5: Token-based strategy with hard threshold for large files
    # Large files MUST use micro-transactional edits - never request full output
    estimated_tokens = _estimate_tokens(file_content)

    SMALL_TOKEN_THRESHOLD = 4000   # ~16KB file - full content OK
    MEDIUM_TOKEN_THRESHOLD = 8000  # ~32KB file - structured edits

    # v5: Files above LARGE_FILE_TOKEN_THRESHOLD use multi-pass loop
    if estimated_tokens > LARGE_FILE_TOKEN_THRESHOLD:
        # LARGE FILE: v5 multi-pass micro-transactional edits
        # Never request full file output - only bounded structured edits
        target_identity = TargetIdentity.from_finding(
            {
                "check_id": item.finding_fingerprints[0] if item.finding_fingerprints else "",
                "path": file_path,
                "start": {"line": 1},
                "end": {"line": min(50, len(original_lines))},
            },
            file_content
        )

        patch, validation, no_change_proof, passes_used = run_multipass_patch_loop(
            item=item,
            file_content=file_content,
            file_path=file_path,
            target_identity=target_identity,
            agent_type=agent_type,
            provider=provider,
            model=model,
            project_dir=project_dir or Path("."),
            max_passes=3,
        )
        # Add agent model info
        patch.agent_model = model
        return patch, validation, no_change_proof

    if estimated_tokens <= SMALL_TOKEN_THRESHOLD:
        # SMALL FILE: Request complete file output
        prompt = _build_small_file_prompt(item, file_content, file_path, agent_type)
        parse_mode = "full_content"
    elif estimated_tokens <= MEDIUM_TOKEN_THRESHOLD:
        # MEDIUM FILE: Request structured edits with enhanced context
        context_snippet, start_line, end_line, scope_type = _extract_enhanced_context(
            file_content, item, max_tokens=3000
        )
        # Include imports in the prompt for context
        imports = _extract_imports(file_content)
        prompt = _build_structured_edit_prompt_v4(
            item, file_content, file_path, context_snippet,
            start_line, end_line, agent_type, imports, scope_type
        )
        parse_mode = "structured_edits"
    else:
        # MEDIUM-LARGE FILE: v4 region focus (still under v5 threshold)
        context_snippet, start_line, end_line, scope_type = _extract_enhanced_context(
            file_content, item, max_tokens=4000
        )
        imports = _extract_imports(file_content)
        constants = _extract_referenced_constants(file_content, start_line - 1, end_line)
        prompt = _build_region_focus_prompt_v4(
            item, file_content, file_path, context_snippet,
            start_line, end_line, agent_type, imports, constants, scope_type
        )
        parse_mode = "structured_edits"

    # Use longer timeout for larger files
    timeout = 180 if len(original_lines) > 200 else 120
    try:
        response, prompt_hash = _call_llm(prompt, provider, model, temperature=0.0, max_tokens=16000, timeout=timeout)
    except Exception as e:
        # Timeout or other error - return validation failure
        return PatchPacket(
            patch_id=f"PATCH-{item.item_id}-{sha256_str(file_path)[:8]}",
            item_id=item.item_id,
            file_path=file_path,
            original_hash=original_hash,
            original_line_count=len(original_lines),
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        ), PatchValidation(
            patch_id=f"PATCH-{item.item_id}-{sha256_str(file_path)[:8]}",
            is_valid=False,
            status=PatchStatus.MALFORMED.value,
            error=f"LLM call failed: {str(e)[:200]}",
        ), None

    # Parse response based on mode
    if parse_mode == "full_content":
        new_content, no_change_proof = _parse_full_content_response(response, file_content, item)
    else:
        new_content, no_change_proof = _parse_structured_edit_response(response, file_content, item)

    # Check for error responses or invalid content
    if not new_content or len(new_content) < 10:
        patch_content = ""
        expected_hash = original_hash
        new_lines = original_lines
        validation = PatchValidation(
            patch_id="",
            is_valid=False,
            status=PatchStatus.NO_CHANGE.value,
            skip_reason=PatchSkipReason.NO_CHANGE.value,
            error="LLM returned error or empty response",
        )
    else:
        # Ensure consistent line endings
        new_content = new_content.replace('\r\n', '\n')
        if not new_content.endswith('\n'):
            new_content += '\n'

        expected_hash = sha256_str(new_content)
        new_lines = new_content.split('\n')

        # MACHINE-GENERATE THE DIFF (never trust LLM diffs)
        patch_content = _generate_diff_from_contents(file_content, new_content, file_path)

        # Check if this is a no-op
        if expected_hash == original_hash or not patch_content.strip():
            # For security_fix category, NO_CHANGE requires proof
            if item.category == "security_fix" and not no_change_proof:
                validation = PatchValidation(
                    patch_id="",
                    is_valid=False,
                    status=PatchStatus.NO_CHANGE.value,
                    skip_reason=PatchSkipReason.NO_CHANGE.value,
                    error="NO_CHANGE for security_fix requires proof",
                )
            else:
                validation = PatchValidation(
                    patch_id="",
                    is_valid=False,
                    status=PatchStatus.NO_CHANGE.value,
                    skip_reason=PatchSkipReason.NO_CHANGE.value,
                )
        else:
            # Run diff-shape sanity check
            sanity = check_diff_shape_sanity(
                patch_content, file_path,
                category=item.category,
                description=item.description,
            )
            if not sanity.passed:
                validation = PatchValidation(
                    patch_id="",
                    is_valid=False,
                    status=PatchStatus.MALFORMED.value,
                    error=f"Diff sanity check failed: {'; '.join(sanity.errors)}",
                )
            elif project_dir:
                # Validate with git apply --check
                validation = validate_patch_with_git(patch_content, project_dir, file_path)
            else:
                validation = PatchValidation(
                    patch_id="",
                    is_valid=True,
                    status=PatchStatus.VALID.value,
                )

    # Count changes from patch content
    patch_lines = patch_content.split('\n') if patch_content else []
    lines_added = sum(1 for line in patch_lines if line.startswith('+') and not line.startswith('+++'))
    lines_removed = sum(1 for line in patch_lines if line.startswith('-') and not line.startswith('---'))
    hunks = sum(1 for line in patch_lines if line.startswith('@@'))

    patch = PatchPacket(
        patch_id=f"PATCH-{item.item_id}-{sha256_str(file_path)[:8]}",
        item_id=item.item_id,
        file_path=file_path,
        original_hash=original_hash,
        original_line_count=len(original_lines),
        patch_content=patch_content,
        patch_hash=sha256_str(patch_content),
        expected_hash=expected_hash,
        expected_line_count=len(new_lines) if new_lines else len(original_lines),
        agent_type=agent_type,
        agent_model=model,
        generation_prompt_hash=prompt_hash,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        lines_added=lines_added,
        lines_removed=lines_removed,
        hunks=hunks,
    )

    validation.patch_id = patch.patch_id
    return patch, validation, no_change_proof


# =============================================================================
# Prompt Builders (Tightened contracts to prevent echoing)
# =============================================================================

def _build_small_file_prompt(item: RefactorItem, file_content: str, file_path: str, agent_type: str) -> str:
    """Build prompt for small files (< 200 lines). Returns complete file."""
    return f"""You are a code refactoring agent ({agent_type}). Your task is to fix a SPECIFIC issue.

## File: {file_path}
```python
{file_content}
```

## Refactor Task
- Category: {item.category}
- Priority: {item.priority}
- Issue: {item.description}
- Fix: {item.suggested_change}

## CRITICAL REQUIREMENTS
1. You MUST make at least one change to fix the issue described above
2. If you believe NO change is needed, output NO_CHANGE with justification (see format below)
3. Do NOT echo the original file unchanged - that is ALWAYS wrong
4. Make ONLY the changes needed for this specific fix
5. Do NOT add extra imports, features, or refactoring
6. Do NOT delete docstrings or comments unless they are the issue
7. Output must be valid Python (importable)

## Output Format

If you make changes:
```python
<complete modified file content>
```

If NO change needed (rare - requires justification):
```json
{{"no_change": true, "disposition": "false_positive|already_mitigated", "justification": "detailed reason why the code is already safe", "evidence_lines": [line_numbers], "evidence_snippet": "relevant code snippet"}}
```

Output ONLY one of the above formats. No explanations.
"""


def _build_structured_edit_prompt(item: RefactorItem, file_content: str, file_path: str,
                                   context_snippet: str, start_line: int, end_line: int,
                                   agent_type: str) -> str:
    """Build prompt for medium files (200-500 lines). Returns structured edits."""
    lines = file_content.split('\n')
    return f"""You are a code refactoring agent ({agent_type}). Your task is to fix a SPECIFIC issue.

## File: {file_path} ({len(lines)} lines total)

### Focus Region (lines {start_line}-{end_line}):
```
{context_snippet}
```

### File Structure:
```
{_summarize_file_structure(file_content)}
```

## Refactor Task
- Category: {item.category}
- Issue: {item.description}
- Fix: {item.suggested_change}

## CRITICAL REQUIREMENTS
1. You MUST make at least one edit to fix the issue
2. Edits must be in the focus region (lines {start_line}-{end_line})
3. Line numbers are 1-based as shown in the context
4. Make MINIMAL edits - only what's needed for the fix
5. Do NOT make edits outside the relevant area

## Output Format

If you make changes:
```json
{{
  "edits": [
    {{"start_line": N, "end_line": M, "replacement": "new content for those lines"}}
  ]
}}
```

If NO change needed (rare):
```json
{{"no_change": true, "disposition": "false_positive|already_mitigated", "justification": "reason", "evidence_lines": [N], "evidence_snippet": "code"}}
```

Output ONLY the JSON. No explanations.
"""


def _build_region_focus_prompt(item: RefactorItem, file_content: str, file_path: str,
                                context_snippet: str, start_line: int, end_line: int,
                                agent_type: str) -> str:
    """Build prompt for large files (> 500 lines). Focus on region only."""
    lines = file_content.split('\n')
    return f"""You are a code refactoring agent ({agent_type}). Fix a SPECIFIC issue in a large file.

## File: {file_path} ({len(lines)} lines - LARGE FILE)

### Region Requiring Fix (lines {start_line}-{end_line}):
```
{context_snippet}
```

## Refactor Task
- Category: {item.category}
- Issue: {item.description}
- Fix: {item.suggested_change}

## CRITICAL REQUIREMENTS
1. You MUST make at least one edit in the shown region
2. Line numbers are 1-based as shown
3. Make ONLY edits within lines {start_line}-{end_line}
4. Make MINIMAL changes - surgical fix only

## Output Format
```json
{{
  "edits": [
    {{"start_line": N, "end_line": M, "replacement": "new content"}}
  ]
}}
```

Or if genuinely no change needed:
```json
{{"no_change": true, "disposition": "false_positive|already_mitigated", "justification": "reason", "evidence_snippet": "code"}}
```

Output ONLY the JSON.
"""


def _build_structured_edit_prompt_v4(
    item: RefactorItem, file_content: str, file_path: str,
    context_snippet: str, start_line: int, end_line: int,
    agent_type: str, imports: str, scope_type: str,
) -> str:
    """Build v4 prompt for medium files with enhanced context.

    Includes imports and scope information to prevent "edit compiles in
    isolation but breaks surrounding code" failures.
    """
    lines = file_content.split('\n')
    return f"""You are a code refactoring agent ({agent_type}). Your task is to fix a SPECIFIC issue.

## File: {file_path} ({len(lines)} lines, ~{_estimate_tokens(file_content)} tokens)

### File Imports (for reference):
```python
{imports}
```

### Focus Region ({scope_type}, lines {start_line}-{end_line}):
```
{context_snippet}
```

### File Structure:
```
{_summarize_file_structure(file_content)}
```

## Refactor Task
- Category: {item.category}
- Issue: {item.description}
- Fix: {item.suggested_change}

## CRITICAL REQUIREMENTS
1. You MUST make at least one edit to fix the issue
2. Edits must be in the focus region (lines {start_line}-{end_line})
3. Line numbers are 1-based as shown in the context
4. Make MINIMAL edits - only what's needed for the fix
5. Do NOT add/change imports unless the fix requires it
6. Ensure your edit is compatible with the shown imports

## Output Format

If you make changes:
```json
{{
  "edits": [
    {{"start_line": N, "end_line": M, "replacement": "new content for those lines"}}
  ]
}}
```

If NO change needed (rare):
```json
{{"no_change": true, "disposition": "false_positive|already_mitigated", "justification": "reason", "evidence_lines": [N], "evidence_snippet": "code"}}
```

Output ONLY the JSON. No explanations.
"""


def _build_region_focus_prompt_v4(
    item: RefactorItem, file_content: str, file_path: str,
    context_snippet: str, start_line: int, end_line: int,
    agent_type: str, imports: str, constants: str, scope_type: str,
) -> str:
    """Build v4 prompt for large files with full context support.

    Includes imports, referenced constants, and scope type to ensure
    edits are compatible with the full file context.
    """
    lines = file_content.split('\n')
    constants_section = f"""
### Referenced Constants:
```python
{constants}
```
""" if constants else ""

    return f"""You are a code refactoring agent ({agent_type}). Fix a SPECIFIC issue in a large file.

## File: {file_path} ({len(lines)} lines - LARGE FILE)

### File Imports:
```python
{imports}
```
{constants_section}
### Region Requiring Fix ({scope_type}, lines {start_line}-{end_line}):
```
{context_snippet}
```

## Refactor Task
- Category: {item.category}
- Issue: {item.description}
- Fix: {item.suggested_change}

## CRITICAL REQUIREMENTS
1. You MUST make at least one edit in the shown region
2. Line numbers are 1-based as shown
3. Make ONLY edits within lines {start_line}-{end_line}
4. Make MINIMAL changes - surgical fix only
5. Your edit must be compatible with the shown imports
6. Do NOT reference undefined names or missing imports

## Output Format
```json
{{
  "edits": [
    {{"start_line": N, "end_line": M, "replacement": "new content"}}
  ]
}}
```

Or if genuinely no change needed:
```json
{{"no_change": true, "disposition": "false_positive|already_mitigated", "justification": "reason", "evidence_snippet": "code"}}
```

Output ONLY the JSON.
"""


# =============================================================================
# Response Parsers
# =============================================================================

def _build_hash_bound_proof(data: dict, original_content: str) -> NoChangeProof:
    """Build a hash-bound NoChangeProof from LLM response data.

    v4: Computes immutable evidence hashes for later verification.
    """
    evidence_lines = data.get("evidence_lines", [])
    evidence_snippet = data.get("evidence_snippet", "")

    # Compute file_pre_hash
    file_pre_hash = sha256_str(original_content)

    # Compute evidence_chunk_hash from snippet
    evidence_chunk_hash = sha256_str(evidence_snippet) if evidence_snippet else ""

    # Compute span_anchor if we have line numbers
    span_anchor = None
    if evidence_lines and len(evidence_lines) >= 1:
        lines = original_content.split('\n')
        start_line = min(evidence_lines)
        end_line = max(evidence_lines)
        if start_line > 0 and end_line <= len(lines):
            snippet_from_lines = '\n'.join(lines[start_line-1:end_line])
            span_anchor = SpanAnchor(
                start_line=start_line,
                end_line=end_line,
                snippet_hash=sha256_str(snippet_from_lines),
            )

    return NoChangeProof(
        disposition=data.get("disposition", "unproven"),
        justification=data.get("justification", ""),
        file_pre_hash=file_pre_hash,
        evidence_chunk_hash=evidence_chunk_hash,
        span_anchor=span_anchor,
        evidence_lines=evidence_lines,
        evidence_snippet=evidence_snippet,
        suppression_comment=data.get("suppression_comment", ""),
        waiver_policy_ref=data.get("waiver_policy_ref", ""),
    )


def _parse_full_content_response(response: str, original_content: str, item: RefactorItem) -> tuple[str, Optional[NoChangeProof]]:
    """Parse response that should contain complete file content."""
    # Check for NO_CHANGE response first
    if '"no_change"' in response and 'true' in response.lower():
        try:
            json_match = re.search(r'```(?:json)?\n([\s\S]*?)```', response)
            if json_match:
                data = json.loads(json_match.group(1))
            else:
                data = json.loads(response)

            if data.get("no_change"):
                # v4: Build hash-bound proof
                proof = _build_hash_bound_proof(data, original_content)
                return original_content, proof
        except json.JSONDecodeError:
            pass

    # Extract code block
    code_match = re.search(r'```(?:python|py)?\n([\s\S]*?)```', response)
    if code_match:
        return code_match.group(1), None

    # Maybe raw code
    if response.strip() and not response.strip().startswith('{'):
        return response.strip(), None

    return "", None


def _parse_structured_edit_response(response: str, original_content: str, item: RefactorItem) -> tuple[str, Optional[NoChangeProof]]:
    """Parse response that contains structured edits."""
    try:
        # Extract JSON
        json_match = re.search(r'```(?:json)?\n([\s\S]*?)```', response)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            data = json.loads(response)

        # Check for NO_CHANGE
        if data.get("no_change"):
            # v4: Build hash-bound proof
            proof = _build_hash_bound_proof(data, original_content)
            return original_content, proof

        # Apply edits
        edits = data.get("edits", [])
        if not edits:
            return original_content, None

        return _apply_structured_edits(original_content, json.dumps(data)), None

    except json.JSONDecodeError:
        # Try to extract code if JSON failed
        code_match = re.search(r'```(?:python|py)?\n([\s\S]*?)```', response)
        if code_match:
            return code_match.group(1), None
        return "", None


def _summarize_file_structure(content: str) -> str:
    """Generate a summary of file structure for context."""
    lines = content.split('\n')
    summary_lines = []

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Include imports, class/function definitions, important comments
        if (stripped.startswith(('import ', 'from ', 'class ', 'def ', 'async def ')) or
            stripped.startswith(('#', '"""', "'''")) or
            'TODO' in stripped or 'FIXME' in stripped):
            summary_lines.append(f"{i:4}: {line}")

    return '\n'.join(summary_lines[:50])  # Limit to 50 lines


def _apply_structured_edits(original_content: str, llm_response: str) -> str:
    """Apply structured edits from LLM JSON response."""
    lines = original_content.split('\n')

    # Parse JSON edits
    try:
        json_match = re.search(r'```(?:json)?\n([\s\S]*?)```', llm_response)
        if json_match:
            edits_data = json.loads(json_match.group(1))
        else:
            edits_data = json.loads(llm_response)

        edits = edits_data.get('edits', [])
    except json.JSONDecodeError:
        # If not JSON, treat as full content
        code_match = re.search(r'```(?:python|javascript|typescript|go|rust|java|py)?\n([\s\S]*?)```', llm_response)
        if code_match:
            return code_match.group(1)
        return original_content

    if not edits:
        return original_content

    # Sort edits by start_line descending (apply from bottom up to preserve line numbers)
    edits = sorted(edits, key=lambda e: e.get('start_line', 0), reverse=True)

    for edit in edits:
        start = edit.get('start_line', 1) - 1  # Convert to 0-based
        end = edit.get('end_line', start + 1)
        replacement = edit.get('replacement', '')

        if start < 0:
            start = 0
        if end > len(lines):
            end = len(lines)

        # Replace the lines
        replacement_lines = replacement.split('\n') if replacement else []
        lines[start:end] = replacement_lines

    return '\n'.join(lines)


# =============================================================================
# Patch Verifier (uses git apply for robustness)
# =============================================================================

def verify_patch(
    patch: PatchPacket,
    original_content: str,
    project_dir: Optional[Path] = None,
) -> PatchVerification:
    """Verify a patch applies correctly and produces expected result.

    DESIGN (v2):
    - Uses git apply when possible (more robust than patch)
    - Falls back to Python difflib if neither git nor patch available
    - Explicitly handles no-op patches as verified (but caller should SKIP them)
    """

    # Check original hash
    actual_original_hash = sha256_str(original_content)
    original_match = actual_original_hash == patch.original_hash

    if not original_match:
        return PatchVerification(
            patch_id=patch.patch_id,
            verified=False,
            original_hash_match=False,
            patch_applies_cleanly=False,
            result_hash_match=False,
            error=f"Original hash mismatch: expected {patch.original_hash[:16]}..., got {actual_original_hash[:16]}...",
        )

    # Handle no-op patches
    if not patch.patch_content or not patch.patch_content.strip():
        return PatchVerification(
            patch_id=patch.patch_id,
            verified=True,  # No-op is technically verified
            original_hash_match=True,
            patch_applies_cleanly=True,
            result_hash_match=True,
            actual_result_hash=actual_original_hash,
        )

    # Apply patch using git apply (preferred)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Initialize a temporary git repo
            subprocess.run(["git", "init"], cwd=tmpdir, capture_output=True, check=True)

            # Create the file structure
            file_path = tmpdir_path / patch.file_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(original_content)

            # Commit original
            subprocess.run(["git", "add", "."], cwd=tmpdir, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.email=test@test.com", "-c", "user.name=Test",
                 "commit", "-m", "original"],
                cwd=tmpdir, capture_output=True
            )

            # Apply patch
            result = subprocess.run(
                ["git", "apply", "--unsafe-paths", "-"],
                input=patch.patch_content.encode(),
                capture_output=True,
                cwd=tmpdir,
                timeout=30,
            )

            if result.returncode != 0:
                # Try with --3way for better conflict handling
                # Reset first
                file_path.write_text(original_content)
                result = subprocess.run(
                    ["git", "apply", "--3way", "-"],
                    input=patch.patch_content.encode(),
                    capture_output=True,
                    cwd=tmpdir,
                    timeout=30,
                )

            if result.returncode != 0:
                return PatchVerification(
                    patch_id=patch.patch_id,
                    verified=False,
                    original_hash_match=True,
                    patch_applies_cleanly=False,
                    result_hash_match=False,
                    error=f"git apply failed: {result.stderr.decode()[:500]}",
                )

            # Read result
            result_content = file_path.read_text()
            actual_result_hash = sha256_str(result_content)

    except FileNotFoundError:
        # git not available, fall back to Python-based approach
        return _verify_patch_python(patch, original_content)
    except subprocess.TimeoutExpired:
        return PatchVerification(
            patch_id=patch.patch_id,
            verified=False,
            original_hash_match=True,
            patch_applies_cleanly=False,
            result_hash_match=False,
            error="git apply timed out",
        )
    except Exception as e:
        return PatchVerification(
            patch_id=patch.patch_id,
            verified=False,
            original_hash_match=True,
            patch_applies_cleanly=False,
            result_hash_match=False,
            error=str(e),
        )

    # Check result hash
    result_match = actual_result_hash == patch.expected_hash

    return PatchVerification(
        patch_id=patch.patch_id,
        verified=original_match and result_match,
        original_hash_match=original_match,
        patch_applies_cleanly=True,
        result_hash_match=result_match,
        actual_result_hash=actual_result_hash,
        error="" if result_match else f"Result hash mismatch: expected {patch.expected_hash[:16]}..., got {actual_result_hash[:16]}...",
    )


def _verify_patch_python(patch: PatchPacket, original_content: str) -> PatchVerification:
    """Fallback verification using Python's patch module."""
    try:
        import patch as pypatch

        pset = pypatch.fromstring(patch.patch_content.encode())
        if pset and pset.apply(root=None, strip=1):
            # Note: pypatch modifies in place, this is a simplified check
            pass
    except ImportError:
        pass

    # If we can't verify with tools, try subprocess patch as last resort
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_path = Path(tmpdir) / "file"
            orig_path.write_text(original_content)

            patch_path = Path(tmpdir) / "changes.patch"
            patch_path.write_text(patch.patch_content)

            result = subprocess.run(
                ["patch", "-p1", "--input", str(patch_path), str(orig_path)],
                capture_output=True,
                text=True,
                cwd=tmpdir,
            )

            if result.returncode == 0:
                result_content = orig_path.read_text()
                actual_hash = sha256_str(result_content)
                return PatchVerification(
                    patch_id=patch.patch_id,
                    verified=actual_hash == patch.expected_hash,
                    original_hash_match=True,
                    patch_applies_cleanly=True,
                    result_hash_match=actual_hash == patch.expected_hash,
                    actual_result_hash=actual_hash,
                )
    except Exception as e:
        pass

    return PatchVerification(
        patch_id=patch.patch_id,
        verified=False,
        original_hash_match=True,
        patch_applies_cleanly=False,
        result_hash_match=False,
        error="No patch tools available for verification",
    )


# =============================================================================
# Patch Repair (one retry for malformed patches)
# =============================================================================

def repair_malformed_patch(
    item: RefactorItem,
    file_content: str,
    file_path: str,
    original_error: str,
    agent_type: str = "general_agent",
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    project_dir: Optional[Path] = None,
) -> tuple[PatchPacket, PatchValidation, Optional[NoChangeProof]]:
    """Attempt to repair a malformed patch with stricter constraints.

    This is the "repair loop" - one budgeted retry with:
    1. Explicit error feedback
    2. Full file output (no structured edits - they're unreliable)
    3. Clearer instructions

    Returns (patch, validation, no_change_proof) tuple.
    """
    original_hash = sha256_str(file_content)
    original_lines = file_content.split('\n')

    # For repair, show the exact context needed plus full file
    context_snippet, start_line, end_line = _extract_relevant_context(file_content, item, context_lines=50)

    prompt = f"""You are a code repair agent. A previous attempt to fix this code failed.

## Previous Error
{original_error[:300]}

## File: {file_path}
Here is the COMPLETE file:
```python
{file_content}
```

## Focus Area (lines {start_line}-{end_line}):
```
{context_snippet}
```

## Task
- {item.description}
- {item.suggested_change}

## CRITICAL INSTRUCTIONS
1. Output the COMPLETE MODIFIED FILE
2. Make ONLY the minimal changes needed
3. Do NOT add new features or extra code
4. Keep ALL other code exactly the same
5. Output ONLY the code in a code block

```python
<complete modified file here>
```
"""

    try:
        response, prompt_hash = _call_llm(prompt, provider, model, temperature=0.0, max_tokens=16000, timeout=180)
    except Exception as e:
        # Repair also failed
        return PatchPacket(
            patch_id=f"PATCH-{item.item_id}-{sha256_str(file_path)[:8]}-repair",
            item_id=item.item_id,
            file_path=file_path,
            original_hash=original_hash,
            original_line_count=len(original_lines),
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        ), PatchValidation(
            patch_id=f"PATCH-{item.item_id}-{sha256_str(file_path)[:8]}-repair",
            is_valid=False,
            status=PatchStatus.REPAIR_FAILED.value,
            skip_reason=PatchSkipReason.MALFORMED_AFTER_REPAIR.value,
            error=f"Repair LLM call failed: {str(e)[:200]}",
        ), None

    # Extract full file content
    code_match = re.search(r'```(?:python|javascript|typescript|go|rust|java|py)?\n([\s\S]*?)```', response)
    if code_match:
        new_content = code_match.group(1)
    else:
        new_content = response.strip()

    # Ensure consistent endings
    new_content = new_content.replace('\r\n', '\n')
    if not new_content.endswith('\n'):
        new_content += '\n'

    expected_hash = sha256_str(new_content)

    # Generate diff programmatically
    patch_content = _generate_diff_from_contents(file_content, new_content, file_path)

    # Validate
    if expected_hash == original_hash or not patch_content.strip():
        validation = PatchValidation(
            patch_id="",
            is_valid=False,
            status=PatchStatus.REPAIR_FAILED.value,
            skip_reason=PatchSkipReason.NO_CHANGE.value,
            error="Repair produced no changes",
        )
    elif project_dir:
        validation = validate_patch_with_git(patch_content, project_dir, file_path)
        if not validation.is_valid:
            validation.status = PatchStatus.REPAIR_FAILED.value
            validation.skip_reason = PatchSkipReason.MALFORMED_AFTER_REPAIR.value
    else:
        validation = PatchValidation(
            patch_id="",
            is_valid=True,
            status=PatchStatus.VALID.value,
        )

    patch_lines = patch_content.split('\n') if patch_content else []
    lines_added = sum(1 for line in patch_lines if line.startswith('+') and not line.startswith('+++'))
    lines_removed = sum(1 for line in patch_lines if line.startswith('-') and not line.startswith('---'))
    hunks = sum(1 for line in patch_lines if line.startswith('@@'))

    patch = PatchPacket(
        patch_id=f"PATCH-{item.item_id}-{sha256_str(file_path)[:8]}-repair",
        item_id=item.item_id,
        file_path=file_path,
        original_hash=original_hash,
        original_line_count=len(original_lines),
        patch_content=patch_content,
        patch_hash=sha256_str(patch_content),
        expected_hash=expected_hash,
        expected_line_count=len(new_content.split('\n')),
        agent_type=f"{agent_type}_repair",
        agent_model=model,
        generation_prompt_hash=prompt_hash,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        lines_added=lines_added,
        lines_removed=lines_removed,
        hunks=hunks,
    )

    validation.patch_id = patch.patch_id
    return patch, validation, None


# =============================================================================
# Patch Interaction Handling (v4)
# =============================================================================

class PatchInteraction(Enum):
    """Type of interaction between patches."""
    NONE = "none"  # Patches are independent
    SEQUENTIAL = "sequential"  # Must apply in order (same file, no overlap)
    CONFLICT = "conflict"  # Overlapping hunks, cannot merge automatically
    MERGED = "merged"  # Successfully merged into single patch


@dataclass
class HunkSpan:
    """Represents a hunk's line span in the original file."""
    start_line: int
    end_line: int
    patch_id: str

    def overlaps(self, other: "HunkSpan") -> bool:
        """Check if this hunk overlaps with another."""
        return not (self.end_line < other.start_line or other.end_line < self.start_line)


@dataclass
class PatchInteractionResult:
    """Result of analyzing patch interactions."""
    interaction_type: PatchInteraction
    patches_in_order: list[str] = field(default_factory=list)  # Ordered patch IDs
    conflicts: list[tuple[str, str, str]] = field(default_factory=list)  # (patch1, patch2, reason)
    merged_patch: Optional[PatchPacket] = None  # If successfully merged

    def to_dict(self) -> dict:
        return {
            "interaction_type": self.interaction_type.value,
            "patches_in_order": self.patches_in_order,
            "conflicts": self.conflicts,
            "has_merged_patch": self.merged_patch is not None,
        }


def _parse_patch_hunks(patch_content: str) -> list[HunkSpan]:
    """Parse patch to extract hunk spans."""
    hunks = []
    patch_id = ""

    for line in patch_content.split('\n'):
        if line.startswith('@@'):
            # Parse hunk header: @@ -start,count +start,count @@
            match = re.search(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
            if match:
                start = int(match.group(1))
                count = int(match.group(2) or 1)
                hunks.append(HunkSpan(
                    start_line=start,
                    end_line=start + count - 1,
                    patch_id=patch_id,
                ))

    return hunks


def detect_patch_conflicts(
    patches: list[PatchPacket],
) -> dict[str, PatchInteractionResult]:
    """Detect conflicts between patches touching the same file.

    Multiple agents touching the same file is where correctness goes to die.
    This function:
    1. Groups patches by file
    2. Detects overlapping hunks within each file
    3. Returns interaction results for conflict handling

    Returns dict mapping file_path -> PatchInteractionResult
    """
    # Group patches by file
    by_file: dict[str, list[PatchPacket]] = {}
    for patch in patches:
        if patch.file_path not in by_file:
            by_file[patch.file_path] = []
        by_file[patch.file_path].append(patch)

    results: dict[str, PatchInteractionResult] = {}

    for file_path, file_patches in by_file.items():
        if len(file_patches) <= 1:
            # Single patch - no interaction
            results[file_path] = PatchInteractionResult(
                interaction_type=PatchInteraction.NONE,
                patches_in_order=[p.patch_id for p in file_patches],
            )
            continue

        # Multiple patches on same file - check for conflicts
        all_hunks = []
        for patch in file_patches:
            hunks = _parse_patch_hunks(patch.patch_content)
            for hunk in hunks:
                hunk.patch_id = patch.patch_id
                all_hunks.append(hunk)

        # Sort hunks by start line
        all_hunks.sort(key=lambda h: h.start_line)

        # Check for overlaps
        conflicts = []
        for i, hunk1 in enumerate(all_hunks):
            for hunk2 in all_hunks[i+1:]:
                if hunk1.patch_id != hunk2.patch_id and hunk1.overlaps(hunk2):
                    conflicts.append((
                        hunk1.patch_id,
                        hunk2.patch_id,
                        f"Overlapping hunks at lines {hunk1.start_line}-{hunk1.end_line} and {hunk2.start_line}-{hunk2.end_line}",
                    ))

        if conflicts:
            results[file_path] = PatchInteractionResult(
                interaction_type=PatchInteraction.CONFLICT,
                patches_in_order=[p.patch_id for p in file_patches],
                conflicts=conflicts,
            )
        else:
            # No conflicts - can apply sequentially (sorted by first hunk)
            sorted_patches = sorted(
                file_patches,
                key=lambda p: _parse_patch_hunks(p.patch_content)[0].start_line if _parse_patch_hunks(p.patch_content) else 0
            )
            results[file_path] = PatchInteractionResult(
                interaction_type=PatchInteraction.SEQUENTIAL,
                patches_in_order=[p.patch_id for p in sorted_patches],
            )

    return results


def build_patch_dag(
    plan: "RefactorPlan",
    patches: list[PatchPacket],
) -> dict[str, list[str]]:
    """Build a DAG of patch dependencies.

    Returns dict mapping patch_id -> list of patch_ids that must be applied first.

    Includes:
    1. Explicit dependencies from RefactorPlan
    2. Implicit dependencies from file conflicts (must apply in order)
    """
    dag: dict[str, list[str]] = {p.patch_id: [] for p in patches}

    # Add explicit dependencies from plan
    item_to_patch = {}
    for patch in patches:
        item_to_patch[patch.item_id] = patch.patch_id

    for item in plan.items:
        if item.item_id not in item_to_patch:
            continue
        patch_id = item_to_patch[item.item_id]
        for dep_item_id in item.dependencies:
            if dep_item_id in item_to_patch:
                dag[patch_id].append(item_to_patch[dep_item_id])

    # Add implicit dependencies from file conflicts
    interactions = detect_patch_conflicts(patches)
    for file_path, result in interactions.items():
        if result.interaction_type == PatchInteraction.SEQUENTIAL:
            # Create chain: each patch depends on the previous
            ordered = result.patches_in_order
            for i in range(1, len(ordered)):
                if ordered[i-1] not in dag[ordered[i]]:
                    dag[ordered[i]].append(ordered[i-1])

    return dag


@dataclass
class PatchConflictResult:
    """Result for a patch that conflicts with others."""
    patch: PatchPacket
    validation: PatchValidation
    conflicting_patches: list[str]  # List of conflicting patch IDs
    conflict_reason: str

    def to_dict(self) -> dict:
        return {
            "patch": self.patch.to_dict(),
            "validation": self.validation.to_dict(),
            "conflicting_patches": self.conflicting_patches,
            "conflict_reason": self.conflict_reason,
        }


# =============================================================================
# v5: Large-File Mode - Micro-Transactional Structured Edits
# =============================================================================

# v5 Thresholds: Files above this MUST use structured-only mode
LARGE_FILE_TOKEN_THRESHOLD = 8000  # ~32KB - never request full content above this
REGION_WINDOW_CAP = 200  # Maximum lines in edit window

# Cache directory for suppression memos
SUPPRESSION_MEMO_DIR = Path(".capseal/suppression_memos")


class EditOp(Enum):
    """Types of structured edit operations."""
    REPLACE = "replace"  # Replace lines [start, end) with new content
    INSERT = "insert"    # Insert new content after line N
    DELETE = "delete"    # Delete lines [start, end)


@dataclass
class StructuredEdit:
    """A single bounded edit anchored to target identity.

    v5: Edits are micro-transactional - small, precise, verifiable.
    The window_hash locks the edit to the exact content it was designed for.
    """
    op: EditOp
    start_line: int  # 1-based, inclusive
    end_line: int    # 1-based, exclusive for REPLACE/DELETE
    new_content: str = ""  # For REPLACE/INSERT

    # Anchoring - immutable binding
    window_hash: str = ""  # Hash of lines [start_line-context, end_line+context]
    target_identity: Optional[TargetIdentity] = None  # Finding being addressed

    def to_dict(self) -> dict:
        return {
            "op": self.op.value,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "new_content": self.new_content,
            "window_hash": self.window_hash,
            "target_identity": self.target_identity.to_dict() if self.target_identity else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StructuredEdit":
        target = None
        if data.get("target_identity"):
            target = TargetIdentity(
                rule_id=data["target_identity"].get("rule_id", ""),
                file_path=data["target_identity"].get("file_path", ""),
                match_span=tuple(data["target_identity"].get("match_span", [0, 0])),
                matched_snippet_hash=data["target_identity"].get("matched_snippet_hash", ""),
                fingerprint=data["target_identity"].get("fingerprint", ""),
            )
        return cls(
            op=EditOp(data.get("op", "replace")),
            start_line=data.get("start_line", 0),
            end_line=data.get("end_line", 0),
            new_content=data.get("new_content", ""),
            window_hash=data.get("window_hash", ""),
            target_identity=target,
        )


@dataclass
class StructuredEditPlan:
    """Plan for micro-transactional edits on a large file.

    v5: Never outputs full file - only bounded edit operations.
    Each edit is independently verifiable and reversible.
    """
    schema: str = "structured_edit_plan_v5"
    file_path: str = ""
    file_hash: str = ""  # Original file hash
    file_line_count: int = 0

    # Edits to apply (in order)
    edits: list[StructuredEdit] = field(default_factory=list)

    # Context used for generation
    region_start: int = 0  # Region shown to LLM
    region_end: int = 0
    region_hash: str = ""  # Hash of region content
    imports_included: bool = False
    constants_included: bool = False

    # Metadata
    target_identities: list[TargetIdentity] = field(default_factory=list)
    pass_number: int = 1  # Which multi-pass iteration this is

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "file_line_count": self.file_line_count,
            "edits": [e.to_dict() for e in self.edits],
            "region_start": self.region_start,
            "region_end": self.region_end,
            "region_hash": self.region_hash,
            "imports_included": self.imports_included,
            "constants_included": self.constants_included,
            "target_identities": [t.to_dict() for t in self.target_identities],
            "pass_number": self.pass_number,
        }


@dataclass
class SuppressionMemo:
    """Memoized NO_CHANGE proof to avoid re-proving same mitigation.

    v5: When a finding is repeatedly "already mitigated", we cache
    the proof so subsequent runs don't burn tokens rediscovering it.
    """
    schema: str = "suppression_memo_v5"
    target_identity: TargetIdentity = field(default_factory=lambda: TargetIdentity("", "", (0, 0), ""))
    proof: NoChangeProof = field(default_factory=lambda: NoChangeProof("", ""))

    # Validity tracking
    file_hash_when_proven: str = ""  # Only valid if file hash matches
    proven_at: str = ""  # ISO timestamp
    times_reused: int = 0
    last_reused: str = ""

    # Policy ref (for upgrade to formal suppression)
    suggested_suppression_config: str = ""  # semgrep nosemgrep comment or config

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "target_identity": self.target_identity.to_dict(),
            "proof": self.proof.to_dict(),
            "file_hash_when_proven": self.file_hash_when_proven,
            "proven_at": self.proven_at,
            "times_reused": self.times_reused,
            "last_reused": self.last_reused,
            "suggested_suppression_config": self.suggested_suppression_config,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SuppressionMemo":
        target_data = data.get("target_identity", {})
        proof_data = data.get("proof", {})

        span_anchor = None
        if proof_data.get("span_anchor"):
            span_anchor = SpanAnchor(**proof_data["span_anchor"])

        return cls(
            target_identity=TargetIdentity(
                rule_id=target_data.get("rule_id", ""),
                file_path=target_data.get("file_path", ""),
                match_span=tuple(target_data.get("match_span", [0, 0])),
                matched_snippet_hash=target_data.get("matched_snippet_hash", ""),
                fingerprint=target_data.get("fingerprint", ""),
            ),
            proof=NoChangeProof(
                disposition=proof_data.get("disposition", ""),
                justification=proof_data.get("justification", ""),
                file_pre_hash=proof_data.get("file_pre_hash", ""),
                evidence_chunk_hash=proof_data.get("evidence_chunk_hash", ""),
                span_anchor=span_anchor,
                evidence_lines=proof_data.get("evidence_lines", []),
                evidence_snippet=proof_data.get("evidence_snippet", ""),
            ),
            file_hash_when_proven=data.get("file_hash_when_proven", ""),
            proven_at=data.get("proven_at", ""),
            times_reused=data.get("times_reused", 0),
            last_reused=data.get("last_reused", ""),
            suggested_suppression_config=data.get("suggested_suppression_config", ""),
        )

    def is_valid_for_file(self, current_file_hash: str) -> bool:
        """Check if memo is still valid for current file state."""
        return self.file_hash_when_proven == current_file_hash


@dataclass
class ASTValidation:
    """Result of AST-based semantic validation for NO_CHANGE claims.

    v5: Don't just trust LLM says "whitelist exists" - verify with AST.
    """
    passed: bool = True
    claim_type: str = ""  # "whitelist_enforced", "sanitized_input", etc.

    # Evidence from AST analysis
    ast_evidence: list[str] = field(default_factory=list)  # AST node descriptions
    enforcement_locations: list[tuple[int, int]] = field(default_factory=list)  # (line, col)

    # Failure reasons
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "claim_type": self.claim_type,
            "ast_evidence": self.ast_evidence,
            "enforcement_locations": self.enforcement_locations,
            "errors": self.errors,
        }


# =============================================================================
# v5: Suppression Memoization Functions
# =============================================================================

def _get_memo_path(target_identity: TargetIdentity, project_dir: Path) -> Path:
    """Get path for suppression memo file."""
    memo_dir = project_dir / SUPPRESSION_MEMO_DIR
    # Key by rule_id + file_path hash
    key = sha256_str(f"{target_identity.rule_id}:{target_identity.file_path}")[:16]
    return memo_dir / f"memo_{key}.json"


def load_suppression_memo(target_identity: TargetIdentity, project_dir: Path) -> Optional[SuppressionMemo]:
    """Load cached suppression memo if valid."""
    memo_path = _get_memo_path(target_identity, project_dir)
    if not memo_path.exists():
        return None

    try:
        data = json.loads(memo_path.read_text())
        memo = SuppressionMemo.from_dict(data)

        # Check if this is for the same target (match by rule_id + span overlap)
        if memo.target_identity.rule_id != target_identity.rule_id:
            return None
        if memo.target_identity.file_path != target_identity.file_path:
            return None

        return memo
    except (json.JSONDecodeError, KeyError):
        return None


def save_suppression_memo(
    memo: SuppressionMemo,
    project_dir: Path,
) -> None:
    """Save suppression memo for future runs."""
    memo_path = _get_memo_path(memo.target_identity, project_dir)
    memo_path.parent.mkdir(parents=True, exist_ok=True)
    memo_path.write_text(json.dumps(memo.to_dict(), indent=2))


def check_and_reuse_memo(
    target_identity: TargetIdentity,
    file_content: str,
    project_dir: Path,
) -> Optional[NoChangeProof]:
    """Check if we have a reusable suppression memo.

    Returns NoChangeProof if memo is valid and file hasn't changed.
    """
    memo = load_suppression_memo(target_identity, project_dir)
    if not memo:
        return None

    current_hash = sha256_str(file_content)
    if not memo.is_valid_for_file(current_hash):
        return None

    # Valid memo - update reuse stats and return proof
    memo.times_reused += 1
    memo.last_reused = datetime.datetime.utcnow().isoformat() + "Z"
    save_suppression_memo(memo, project_dir)

    return memo.proof


def create_suppression_memo(
    target_identity: TargetIdentity,
    proof: NoChangeProof,
    file_content: str,
) -> SuppressionMemo:
    """Create new suppression memo from proof."""
    # Generate suggested suppression config
    suggestion = ""
    if proof.disposition == "already_mitigated":
        suggestion = f"# nosemgrep: {target_identity.rule_id} - {proof.justification[:50]}"

    return SuppressionMemo(
        target_identity=target_identity,
        proof=proof,
        file_hash_when_proven=sha256_str(file_content),
        proven_at=datetime.datetime.utcnow().isoformat() + "Z",
        times_reused=0,
        suggested_suppression_config=suggestion,
    )


# =============================================================================
# v5: AST-Based Semantic Validation
# =============================================================================

def validate_whitelist_ast(
    file_content: str,
    claim: str,
    evidence_lines: list[int],
) -> ASTValidation:
    """Validate that a whitelist/allowlist is actually enforced via AST.

    v5: When LLM claims "whitelist exists", verify with AST analysis:
    1. Find the claimed whitelist definition
    2. Trace usage to confirm it gates the dangerous operation
    3. Check no bypass paths exist
    """
    import ast

    result = ASTValidation(claim_type="whitelist_enforced")

    try:
        tree = ast.parse(file_content)
    except SyntaxError as e:
        result.passed = False
        result.errors.append(f"Syntax error: {e}")
        return result

    # Look for whitelist-like structures and guarded dangerous calls
    whitelists_found = []
    whitelist_names = set()
    dangerous_calls = []
    guarded_functions = set()  # Functions that have whitelist checks

    class WhitelistFinder(ast.NodeVisitor):
        def __init__(self):
            self.current_function = None

        def visit_FunctionDef(self, node):
            old_func = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = old_func

        def visit_AsyncFunctionDef(self, node):
            old_func = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = old_func

        def visit_Assign(self, node):
            # Look for ALLOWED_X = {...} or _WHITELIST = [...]
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id.upper()
                    if any(kw in name for kw in ['ALLOW', 'WHITELIST', 'VALID', 'SAFE']):
                        if isinstance(node.value, (ast.Dict, ast.List, ast.Set, ast.Tuple)):
                            whitelists_found.append((node.lineno, target.id))
                            whitelist_names.add(target.id)
            self.generic_visit(node)

        def visit_If(self, node):
            # Check for "if X in WHITELIST" or "if X not in WHITELIST"
            test = node.test
            if isinstance(test, ast.Compare):
                for op, comparator in zip(test.ops, test.comparators):
                    if isinstance(op, (ast.In, ast.NotIn)):
                        if isinstance(comparator, ast.Name):
                            if comparator.id in whitelist_names or \
                               any(kw in comparator.id.upper() for kw in ['ALLOW', 'WHITELIST', 'VALID', 'SAFE']):
                                if self.current_function:
                                    guarded_functions.add(self.current_function)
            self.generic_visit(node)

        def visit_Call(self, node):
            # Look for dangerous calls: import_module, exec, eval, etc.
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id

            if func_name in ['import_module', 'exec', 'eval', '__import__', 'getattr']:
                dangerous_calls.append((node.lineno, func_name, self.current_function))
            self.generic_visit(node)

    finder = WhitelistFinder()
    finder.visit(tree)

    # Record what we found
    result.ast_evidence = [
        f"Whitelist at line {line}: {name}" for line, name in whitelists_found
    ]
    result.ast_evidence.extend([
        f"Dangerous call at line {line}: {name} (in {func or 'module'})"
        for line, name, func in dangerous_calls
    ])
    if guarded_functions:
        result.ast_evidence.append(f"Guarded functions: {', '.join(guarded_functions)}")

    # Validation logic
    if not whitelists_found:
        result.passed = False
        result.errors.append("No whitelist/allowlist structure found in AST")
        return result

    if not dangerous_calls:
        # No dangerous calls = nothing to whitelist = claim is vacuously true
        result.passed = True
        result.ast_evidence.append("No dangerous calls found - claim is vacuously valid")
        return result

    # Check dangerous calls - focus on those in evidence region
    unguarded_calls = []
    evidence_set = set(evidence_lines) if evidence_lines else set()

    # Compute evidence range
    if evidence_lines:
        ev_min = min(evidence_lines)
        ev_max = max(evidence_lines)
    else:
        ev_min, ev_max = 0, float('inf')

    for danger_line, danger_name, danger_func in dangerous_calls:
        # Check if this call is relevant to the evidence
        is_relevant = (
            not evidence_lines or  # No specific evidence = check all
            (ev_min <= danger_line <= ev_max + 3)  # Within evidence span (small margin)
        )

        if not is_relevant:
            # This dangerous call is outside the evidence scope - skip
            continue

        if danger_func and danger_func in guarded_functions:
            # This call is in a function with a whitelist check - OK
            continue

        # Not guarded and in evidence scope - fail
        unguarded_calls.append((danger_line, danger_name, danger_func))

    if unguarded_calls:
        result.passed = False
        for line, name, func in unguarded_calls:
            location = f"function {func}" if func else "module level"
            result.errors.append(
                f"Dangerous call {name}() at line {line} ({location}) "
                f"is not guarded by a whitelist check"
            )

    # Record enforcement locations
    result.enforcement_locations = [(line, 0) for line, _ in whitelists_found]

    return result


# =============================================================================
# v5: Large-File Structured Edit Functions
# =============================================================================

def _extract_bounded_region(
    file_content: str,
    target_lines: list[int],
    max_lines: int = REGION_WINDOW_CAP,
) -> tuple[str, int, int, str]:
    """Extract a bounded region around target lines.

    v5: Enforces strict window cap to prevent token explosion.
    Returns (region_content, start_line, end_line, region_hash).
    """
    lines = file_content.split('\n')

    if not target_lines:
        # Default to beginning of file
        target_lines = [min(50, len(lines) // 2)]

    center = sum(target_lines) // len(target_lines)
    half_window = max_lines // 2

    # Compute window bounds
    start = max(0, center - half_window)
    end = min(len(lines), center + half_window)

    # Ensure we don't exceed cap
    if end - start > max_lines:
        end = start + max_lines

    region_content = '\n'.join(lines[start:end])
    region_hash = sha256_str(region_content)

    return region_content, start + 1, end, region_hash  # 1-based


def _build_large_file_edit_prompt_v5(
    item: RefactorItem,
    file_content: str,
    file_path: str,
    region_content: str,
    start_line: int,
    end_line: int,
    imports: str,
    target_identity: Optional[TargetIdentity] = None,
    pass_number: int = 1,
) -> str:
    """Build v5 prompt for large file micro-transactional edits.

    Key constraints:
    1. NEVER request full file output
    2. Only bounded structured edits
    3. Strict line range enforcement
    4. Anchored to target identity
    """
    lines = file_content.split('\n')

    # Add line numbers to region
    numbered_region = '\n'.join(
        f"{i:4}: {lines[i-1]}"
        for i in range(start_line, min(end_line + 1, len(lines) + 1))
    )

    target_info = ""
    if target_identity:
        target_info = f"""
## Target Finding
- Rule: {target_identity.rule_id}
- Location: lines {target_identity.match_span[0]}-{target_identity.match_span[1]}
- Snippet hash: {target_identity.matched_snippet_hash[:16]}...
"""

    pass_info = ""
    if pass_number > 1:
        pass_info = f"\n## Multi-Pass Context\nThis is pass #{pass_number}. Previous edit(s) did not fully resolve the finding. Make a different fix.\n"

    return f"""You are a code refactoring agent. Fix a SPECIFIC issue using BOUNDED EDITS ONLY.

## File: {file_path} ({len(lines)} lines - LARGE FILE, structured edits required)
{pass_info}
### File Imports (for context):
```python
{imports}
```
{target_info}
### Editable Region (lines {start_line}-{end_line}):
```
{numbered_region}
```

## Refactor Task
- Category: {item.category}
- Issue: {item.description}
- Fix: {item.suggested_change}

## CRITICAL CONSTRAINTS (v5 Large File Protocol)
1. NEVER output full file content - only structured edits
2. Edits MUST be within lines {start_line}-{end_line}
3. Each edit MUST be small (typically <10 lines changed)
4. Make SURGICAL fixes - minimal changes only
5. Your edit must work with the shown imports

## Output Format (REQUIRED)
```json
{{
  "edits": [
    {{
      "op": "replace",
      "start_line": N,
      "end_line": M,
      "new_content": "replacement lines\\nwith newlines"
    }}
  ]
}}
```

Or if genuinely no change needed:
```json
{{
  "no_change": true,
  "disposition": "false_positive|already_mitigated",
  "justification": "detailed reason (min 20 chars)",
  "evidence_lines": [line_numbers_showing_mitigation],
  "evidence_snippet": "code proving safety"
}}
```

Output ONLY the JSON. No explanations.
"""


def apply_structured_edits(
    file_content: str,
    edits: list[StructuredEdit],
) -> tuple[str, list[str]]:
    """Apply structured edits to file content.

    Returns (new_content, errors).
    Edits are applied in reverse order by line number to preserve indices.
    """
    lines = file_content.split('\n')
    errors = []

    # Sort edits by start_line descending (apply bottom-up)
    sorted_edits = sorted(edits, key=lambda e: e.start_line, reverse=True)

    for edit in sorted_edits:
        try:
            if edit.op == EditOp.REPLACE:
                # Replace lines [start, end) with new_content
                new_lines = edit.new_content.split('\n') if edit.new_content else []
                lines[edit.start_line - 1:edit.end_line - 1] = new_lines

            elif edit.op == EditOp.INSERT:
                # Insert after start_line
                new_lines = edit.new_content.split('\n')
                lines[edit.start_line:edit.start_line] = new_lines

            elif edit.op == EditOp.DELETE:
                # Delete lines [start, end)
                del lines[edit.start_line - 1:edit.end_line - 1]

        except (IndexError, ValueError) as e:
            errors.append(f"Edit failed at line {edit.start_line}: {e}")

    return '\n'.join(lines), errors


def _parse_structured_edit_response_v5(
    response: str,
    original_content: str,
    item: RefactorItem,
    region_start: int,
    region_end: int,
) -> tuple[list[StructuredEdit], Optional[NoChangeProof]]:
    """Parse v5 structured edit response.

    Validates edits are within allowed region.
    """
    edits = []
    no_change_proof = None

    try:
        # Extract JSON from response
        json_match = re.search(r'```(?:json)?\n([\s\S]*?)```', response)
        if json_match:
            data = json.loads(json_match.group(1))
        else:
            data = json.loads(response)
    except json.JSONDecodeError:
        return [], None

    # Check for NO_CHANGE response
    if data.get("no_change"):
        no_change_proof = _build_hash_bound_proof(data, original_content)
        return [], no_change_proof

    # Parse edits
    lines = original_content.split('\n')
    for edit_data in data.get("edits", []):
        op_str = edit_data.get("op", "replace")
        try:
            op = EditOp(op_str)
        except ValueError:
            op = EditOp.REPLACE

        start_line = edit_data.get("start_line", 0)
        end_line = edit_data.get("end_line", start_line + 1)
        new_content = edit_data.get("new_content", edit_data.get("replacement", ""))

        # Validate bounds
        if start_line < region_start or end_line > region_end + 1:
            # Edit outside allowed region - skip it
            continue

        # Compute window hash for anchoring
        context_start = max(0, start_line - 3)
        context_end = min(len(lines), end_line + 2)
        window_content = '\n'.join(lines[context_start:context_end])
        window_hash = sha256_str(window_content)

        edit = StructuredEdit(
            op=op,
            start_line=start_line,
            end_line=end_line,
            new_content=new_content,
            window_hash=window_hash,
        )
        edits.append(edit)

    return edits, no_change_proof


# =============================================================================
# v5: Multi-Pass Delta-Driven Loop
# =============================================================================

def run_multipass_patch_loop(
    item: RefactorItem,
    file_content: str,
    file_path: str,
    target_identity: TargetIdentity,
    agent_type: str,
    provider: str,
    model: str,
    project_dir: Path,
    max_passes: int = 3,
    semgrep_rules: Optional[list[str]] = None,
) -> tuple[PatchPacket, PatchValidation, Optional[NoChangeProof], int]:
    """Run multi-pass edit loop for large files.

    v5: Apply one edit, rerun semgrep, iterate until resolved or budget exhausted.

    Returns (patch, validation, no_change_proof, passes_used).
    """
    original_hash = sha256_str(file_content)
    original_lines = file_content.split('\n')
    current_content = file_content
    accumulated_edits: list[StructuredEdit] = []
    no_change_proof = None

    imports = _extract_imports(file_content)

    for pass_num in range(1, max_passes + 1):
        # Extract bounded region around target
        target_lines = list(range(target_identity.match_span[0], target_identity.match_span[1] + 1))
        region_content, region_start, region_end, region_hash = _extract_bounded_region(
            current_content, target_lines, max_lines=REGION_WINDOW_CAP
        )

        # Build prompt
        prompt = _build_large_file_edit_prompt_v5(
            item=item,
            file_content=current_content,
            file_path=file_path,
            region_content=region_content,
            start_line=region_start,
            end_line=region_end,
            imports=imports,
            target_identity=target_identity,
            pass_number=pass_num,
        )

        # Call LLM
        try:
            response, prompt_hash = _call_llm(prompt, provider, model, temperature=0.0, max_tokens=4000, timeout=60)
        except Exception as e:
            break

        # Parse response
        edits, proof = _parse_structured_edit_response_v5(
            response, current_content, item, region_start, region_end
        )

        if proof:
            no_change_proof = proof
            break

        if not edits:
            # No edits and no proof - unclear response
            break

        # Apply edits
        new_content, errors = apply_structured_edits(current_content, edits)
        if errors:
            break

        accumulated_edits.extend(edits)
        current_content = new_content

        # Check if finding is resolved (run semgrep on just this file)
        if semgrep_rules and project_dir:
            is_resolved = _check_finding_resolved_quick(
                current_content, file_path, target_identity, project_dir
            )
            if is_resolved:
                break

    # Build final patch
    if current_content != file_content:
        # Generate diff from accumulated edits
        patch_content = _generate_diff_from_contents(file_content, current_content, file_path)
        expected_hash = sha256_str(current_content)

        # Validate
        validation = PatchValidation(
            patch_id="",
            is_valid=bool(patch_content.strip()),
            status=PatchStatus.VALID.value if patch_content.strip() else PatchStatus.NO_CHANGE.value,
        )

        if patch_content.strip() and project_dir:
            validation = validate_patch_with_git(patch_content, project_dir, file_path)
    else:
        patch_content = ""
        expected_hash = original_hash
        validation = PatchValidation(
            patch_id="",
            is_valid=False,
            status=PatchStatus.NO_CHANGE.value,
            skip_reason=PatchSkipReason.NO_CHANGE.value,
        )

    # Build patch packet
    patch_lines = patch_content.split('\n') if patch_content else []
    lines_added = sum(1 for line in patch_lines if line.startswith('+') and not line.startswith('+++'))
    lines_removed = sum(1 for line in patch_lines if line.startswith('-') and not line.startswith('---'))
    hunks = sum(1 for line in patch_lines if line.startswith('@@'))

    patch = PatchPacket(
        patch_id=f"PATCH-{item.item_id}-{sha256_str(file_path)[:8]}",
        item_id=item.item_id,
        file_path=file_path,
        original_hash=original_hash,
        original_line_count=len(original_lines),
        patch_content=patch_content,
        patch_hash=sha256_str(patch_content) if patch_content else "",
        expected_hash=expected_hash,
        expected_line_count=len(current_content.split('\n')),
        agent_type=agent_type,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        lines_added=lines_added,
        lines_removed=lines_removed,
        hunks=hunks,
    )

    validation.patch_id = patch.patch_id

    return patch, validation, no_change_proof, pass_num


def _check_finding_resolved_quick(
    content: str,
    file_path: str,
    target_identity: TargetIdentity,
    project_dir: Path,
) -> bool:
    """Quick check if target finding is resolved in modified content.

    Runs semgrep on just this file/rule for speed.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            test_file = tmpdir_path / file_path
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text(content)

            result = subprocess.run(
                ["semgrep", "--config", "auto", "--json", str(test_file)],
                capture_output=True,
                timeout=30,
                cwd=tmpdir,
            )

            if result.returncode != 0:
                return False

            output = json.loads(result.stdout.decode())
            findings = output.get("results", [])

            # Check if target rule still matches
            for finding in findings:
                if finding.get("check_id") == target_identity.rule_id:
                    return False  # Still found

            return True  # Not found = resolved

    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return False  # Assume not resolved on error


# =============================================================================
# Multi-Agent Patch Generation (with repair loop)
# =============================================================================

@dataclass
class PatchResult:
    """Result of patch generation including validation, proof, and finding-delta.

    v4: Two-axis outcome model separates mechanical correctness from value.
    """
    patch: PatchPacket
    validation: PatchValidation
    repaired: bool = False
    repair_patch: Optional[PatchPacket] = None
    repair_validation: Optional[PatchValidation] = None
    final_status: str = ""  # Legacy: VALID, SKIP, FAIL

    # v3 additions
    no_change_proof: Optional[NoChangeProof] = None  # Required for security_fix NO_CHANGE
    finding_delta: Optional[FindingDelta] = None  # Did we actually resolve findings?
    diff_sanity: Optional[DiffShapeSanity] = None  # Did patch pass sanity checks?

    # v4 additions
    importability: Optional[ImportabilitySanity] = None  # Is patched code valid Python?
    outcome: Optional[PatchOutcome] = None  # Two-axis status model

    def to_dict(self) -> dict:
        return {
            "patch": self.patch.to_dict(),
            "validation": self.validation.to_dict(),
            "repaired": self.repaired,
            "repair_patch": self.repair_patch.to_dict() if self.repair_patch else None,
            "repair_validation": self.repair_validation.to_dict() if self.repair_validation else None,
            "final_status": self.final_status,
            "no_change_proof": self.no_change_proof.to_dict() if self.no_change_proof else None,
            "finding_delta": self.finding_delta.to_dict() if self.finding_delta else None,
            "diff_sanity": self.diff_sanity.to_dict() if self.diff_sanity else None,
            "importability": self.importability.to_dict() if self.importability else None,
            "outcome": self.outcome.to_dict() if self.outcome else None,
        }


def run_multi_agent_patches(
    plan: RefactorPlan,
    project_dir: Path,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    max_workers: int = 4,
    enable_repair: bool = True,
    enable_finding_delta: bool = True,
    enable_importability_check: bool = True,
    enable_conflict_detection: bool = True,
    enable_suppression_memos: bool = True,
    enable_ast_validation: bool = True,
) -> list[PatchResult]:
    """Run multiple agents to generate patches in parallel with repair loop.

    v5 Changes:
    - Suppression memos: Reuse proofs for repeatedly mitigated findings (saves tokens)
    - AST validation: Verify whitelist claims with AST analysis
    - Large file routing: Files >8K tokens use multi-pass structured edits

    v4 Changes:
    - Validates NO_CHANGE proof for security_fix category
    - Runs finding-delta check for VALID patches (confirms findings resolved)
    - Runs importability check for Python files (catches syntax errors)
    - Detects patch conflicts (overlapping hunks on same file)
    - Computes two-axis outcome (apply_status + effect_status)

    Returns PatchResult objects that include:
    - Original patch and validation
    - Repair attempt if original was malformed
    - NO_CHANGE proof if applicable
    - Finding-delta if applicable
    - Importability check result
    - Two-axis outcome
    - Final status: VALID, SKIP (no-op), FAIL, or CONFLICT
    """

    results = []

    def generate_for_item(item: RefactorItem) -> Optional[PatchResult]:
        file_path = project_dir / item.file_path
        if not file_path.exists():
            # File not found - create SKIP result
            patch = PatchPacket(
                patch_id=f"PATCH-{item.item_id}-{sha256_str(item.file_path)[:8]}",
                item_id=item.item_id,
                file_path=item.file_path,
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            )
            validation = PatchValidation(
                patch_id=patch.patch_id,
                is_valid=False,
                status=PatchStatus.FILE_NOT_FOUND.value,
                skip_reason=PatchSkipReason.FILE_NOT_FOUND.value,
                error=f"File not found: {item.file_path}",
            )
            return PatchResult(
                patch=patch,
                validation=validation,
                final_status="SKIP",
            )

        content = file_path.read_text()
        agent_type = plan.agent_assignments.get(item.item_id, "general_agent")

        # v5: Check for cached suppression memo before calling LLM
        if enable_suppression_memos and item.finding_fingerprints:
            target_identity = TargetIdentity.from_finding(
                {
                    "check_id": item.finding_fingerprints[0] if item.finding_fingerprints else "",
                    "path": item.file_path,
                },
                content
            )
            cached_proof = check_and_reuse_memo(target_identity, content, project_dir)
            if cached_proof:
                # Reuse cached proof - no LLM call needed
                patch = PatchPacket(
                    patch_id=f"PATCH-{item.item_id}-{sha256_str(item.file_path)[:8]}",
                    item_id=item.item_id,
                    file_path=item.file_path,
                    original_hash=sha256_str(content),
                    original_line_count=len(content.split('\n')),
                    expected_hash=sha256_str(content),
                    expected_line_count=len(content.split('\n')),
                    timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                    agent_type="suppression_memo_cache",
                )
                validation = PatchValidation(
                    patch_id=patch.patch_id,
                    is_valid=False,
                    status=PatchStatus.NO_CHANGE.value,
                    skip_reason=PatchSkipReason.ALREADY_MITIGATED.value,
                )
                return PatchResult(
                    patch=patch,
                    validation=validation,
                    final_status="SKIP",
                    no_change_proof=cached_proof,
                    outcome=PatchOutcome(
                        apply_status=ApplyStatus.APPLIES,
                        effect_status=EffectStatus.NO_EFFECT,
                        effect_details="Reused cached suppression memo",
                    ),
                )

        # Generate initial patch (now returns 3-tuple)
        patch, validation, no_change_proof = generate_patch(
            item=item,
            file_content=content,
            file_path=item.file_path,
            agent_type=agent_type,
            provider=provider,
            model=model,
            project_dir=project_dir,
        )

        # Check if valid - run importability and finding-delta gates
        if validation.is_valid and validation.status == PatchStatus.VALID.value:
            # v4: Run importability check for Python files
            importability = None
            if enable_importability_check and item.file_path.endswith('.py'):
                # Get the patched content
                try:
                    import tempfile
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmpdir_path = Path(tmpdir)
                        test_file = tmpdir_path / item.file_path
                        test_file.parent.mkdir(parents=True, exist_ok=True)
                        test_file.write_text(content)

                        # Apply patch to get patched content
                        result = subprocess.run(
                            ["git", "init"],
                            cwd=tmpdir,
                            capture_output=True,
                        )
                        subprocess.run(
                            ["git", "add", "."],
                            cwd=tmpdir,
                            capture_output=True,
                        )
                        result = subprocess.run(
                            ["git", "apply", "-"],
                            input=patch.patch_content.encode(),
                            capture_output=True,
                            cwd=tmpdir,
                        )

                        if result.returncode == 0:
                            patched_content = test_file.read_text()
                            importability = check_importability(patched_content, item.file_path)

                            if not importability.passed:
                                # Syntax error - this is a FAIL
                                validation.is_valid = False
                                validation.error = f"Importability failed: {importability.syntax_error[:200]}"
                                return PatchResult(
                                    patch=patch,
                                    validation=validation,
                                    final_status="FAIL",
                                    importability=importability,
                                    outcome=PatchOutcome(
                                        apply_status=ApplyStatus.MALFORMED,
                                        effect_status=EffectStatus.NO_EFFECT,
                                        apply_error=importability.syntax_error[:200],
                                    ),
                                )
                except Exception as e:
                    # Importability check failed, continue without it
                    pass

            # Run finding-delta check
            finding_delta = None
            if enable_finding_delta and item.finding_fingerprints:
                finding_delta = run_finding_delta_check(
                    patch_content=patch.patch_content,
                    original_content=content,
                    file_path=item.file_path,
                    targeted_fingerprints=item.finding_fingerprints,
                    project_dir=project_dir,
                )
                # If findings not resolved, it's not really VALID
                if not finding_delta.all_targeted_resolved:
                    validation.error = f"Findings not resolved: {finding_delta.remaining_fingerprints}"
                    # Don't fail, but mark in result

            # Compute two-axis outcome
            effect_status = EffectStatus.NO_EFFECT
            if finding_delta:
                effect_status = finding_delta.compute_effect_status()
            else:
                # If no finding delta, assume RESOLVES_TARGET if VALID
                effect_status = EffectStatus.RESOLVES_TARGET

            outcome = PatchOutcome(
                apply_status=ApplyStatus.APPLIES,
                effect_status=effect_status,
            )

            return PatchResult(
                patch=patch,
                validation=validation,
                final_status="VALID",
                finding_delta=finding_delta,
                importability=importability,
                outcome=outcome,
            )

        # Check if no-op - validate proof for security_fix
        if validation.status == PatchStatus.NO_CHANGE.value:
            # For security_fix, NO_CHANGE without proof is FAIL
            if item.category == "security_fix":
                is_proven, proof_error = validate_no_change_proof(item, no_change_proof, content)
                if not is_proven:
                    # Unproven NO_CHANGE for security_fix = FAIL
                    return PatchResult(
                        patch=patch,
                        validation=PatchValidation(
                            patch_id=patch.patch_id,
                            is_valid=False,
                            status=PatchStatus.NO_CHANGE.value,
                            error=f"NO_CHANGE rejected: {proof_error}",
                        ),
                        final_status="FAIL",
                        no_change_proof=no_change_proof,
                    )

                # v5: AST validation for whitelist claims
                if enable_ast_validation and no_change_proof:
                    justification = no_change_proof.justification.lower()
                    if any(kw in justification for kw in ['whitelist', 'allowlist', 'allowed_', 'valid_']):
                        ast_result = validate_whitelist_ast(
                            content,
                            claim=no_change_proof.justification,
                            evidence_lines=no_change_proof.evidence_lines,
                        )
                        if not ast_result.passed:
                            # AST doesn't confirm whitelist enforcement
                            return PatchResult(
                                patch=patch,
                                validation=PatchValidation(
                                    patch_id=patch.patch_id,
                                    is_valid=False,
                                    status=PatchStatus.NO_CHANGE.value,
                                    error=f"AST validation failed: {'; '.join(ast_result.errors)}",
                                ),
                                final_status="FAIL",
                                no_change_proof=no_change_proof,
                            )

                # v5: Save suppression memo for valid proof
                if enable_suppression_memos and no_change_proof and item.finding_fingerprints:
                    target_identity = TargetIdentity.from_finding(
                        {
                            "check_id": item.finding_fingerprints[0],
                            "path": item.file_path,
                        },
                        content
                    )
                    memo = create_suppression_memo(target_identity, no_change_proof, content)
                    save_suppression_memo(memo, project_dir)

            return PatchResult(
                patch=patch,
                validation=validation,
                final_status="SKIP",
                no_change_proof=no_change_proof,
            )

        # Malformed patch - attempt repair if enabled
        if enable_repair and validation.status == PatchStatus.MALFORMED.value:
            repair_patch, repair_validation, repair_proof = repair_malformed_patch(
                item=item,
                file_content=content,
                file_path=item.file_path,
                original_error=validation.error or validation.git_apply_output,
                agent_type=agent_type,
                provider=provider,
                model=model,
                project_dir=project_dir,
            )

            if repair_validation.is_valid:
                # Run finding-delta on repair patch
                finding_delta = None
                if enable_finding_delta and item.finding_fingerprints:
                    finding_delta = run_finding_delta_check(
                        patch_content=repair_patch.patch_content,
                        original_content=content,
                        file_path=item.file_path,
                        targeted_fingerprints=item.finding_fingerprints,
                        project_dir=project_dir,
                    )

                return PatchResult(
                    patch=patch,
                    validation=validation,
                    repaired=True,
                    repair_patch=repair_patch,
                    repair_validation=repair_validation,
                    final_status="VALID",
                    finding_delta=finding_delta,
                )
            else:
                return PatchResult(
                    patch=patch,
                    validation=validation,
                    repaired=True,
                    repair_patch=repair_patch,
                    repair_validation=repair_validation,
                    final_status="FAIL",
                )

        # Malformed and no repair - FAIL
        return PatchResult(
            patch=patch,
            validation=validation,
            final_status="FAIL",
        )

    # Execute in order but parallelize independent items
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(generate_for_item, item): item for item in plan.items}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # v4: Detect patch conflicts
    if enable_conflict_detection:
        valid_patches = [
            r.repair_patch if r.repaired and r.repair_validation and r.repair_validation.is_valid
            else r.patch
            for r in results
            if r.final_status == "VALID" and r.patch.patch_content
        ]

        if len(valid_patches) > 1:
            interactions = detect_patch_conflicts(valid_patches)

            # Mark conflicting patches
            conflict_patch_ids = set()
            for file_path, interaction in interactions.items():
                if interaction.interaction_type == PatchInteraction.CONFLICT:
                    for patch_id1, patch_id2, reason in interaction.conflicts:
                        conflict_patch_ids.add(patch_id1)
                        conflict_patch_ids.add(patch_id2)

            # Update results for conflicting patches
            for result in results:
                active_patch = (
                    result.repair_patch if result.repaired and result.repair_validation and result.repair_validation.is_valid
                    else result.patch
                )
                if active_patch.patch_id in conflict_patch_ids:
                    result.final_status = "CONFLICT"
                    result.validation.error = f"Patch conflicts with other patches on same file"
                    if result.outcome:
                        result.outcome.apply_error = "Conflicting hunks with other patches"

    return results


# =============================================================================
# Diff Rollup Builder (v2: status_detail, clean combined diff)
# =============================================================================

def build_diff_rollup(
    plan: RefactorPlan,
    patch_results: list[PatchResult],
    verifications: list[PatchVerification],
    trace_root: str,
) -> DiffRollup:
    """Build final diff rollup with full provenance.

    v3 Changes:
    - Uses PatchResult instead of raw PatchPacket
    - Builds status_detail with VALID/SKIP/FAIL breakdown + yield metrics
    - combined_diff is pure unified diff (no comment headers)
    - Only VALID patches go into combined_diff
    - Tracks finding resolution rate
    """

    # Build status detail
    status = StatusDetail(total_patches=len(patch_results))

    valid_patches = []
    all_results = []

    for result in patch_results:
        all_results.append(result.to_dict())

        # Use repair patch if available and valid
        active_patch = result.repair_patch if result.repaired and result.repair_validation and result.repair_validation.is_valid else result.patch

        # Track yield metrics
        if result.finding_delta:
            status.findings_targeted += len(result.finding_delta.resolved_fingerprints) + len(result.finding_delta.remaining_fingerprints)
            status.findings_resolved += len(result.finding_delta.resolved_fingerprints)
            status.findings_remaining += len(result.finding_delta.remaining_fingerprints)
            status.new_warnings_introduced += len(result.finding_delta.new_findings)

        if result.no_change_proof:
            if result.no_change_proof.disposition != NoChangeDisposition.UNPROVEN.value:
                status.no_change_proven += 1
            else:
                status.no_change_unproven += 1

        # Check for specific failure modes
        if result.validation.error and "timeout" in result.validation.error.lower():
            status.timeouts += 1
        if result.validation.error and "sanity" in result.validation.error.lower():
            status.sanity_failures += 1
        if result.importability and not result.importability.passed:
            status.importability_failures += 1

        # v4: Track two-axis metrics
        if result.outcome:
            if result.outcome.is_mechanically_correct:
                status.mechanically_correct += 1
            if result.outcome.is_valuable:
                status.valuable += 1

        if result.final_status == "VALID":
            status.valid_patches += 1
            status.valid_patch_ids.append(active_patch.patch_id)
            valid_patches.append(active_patch)
            if result.repaired:
                status.repaired_patches += 1
        elif result.final_status == "SKIP":
            status.skipped_patches += 1
            status.skipped_patch_ids.append(result.patch.patch_id)
            status.skip_reasons[result.patch.patch_id] = result.validation.skip_reason or "unknown"
        elif result.final_status == "CONFLICT":
            # v4: Handle conflicts separately
            status.conflict_patches += 1
            status.conflict_patch_ids.append(active_patch.patch_id)
        else:  # FAIL
            status.failed_patches += 1
            status.failed_patch_ids.append(result.patch.patch_id)

    # Build combined diff from VALID patches only (pure diff format, no comments)
    combined_parts = []
    for patch in sorted(valid_patches, key=lambda p: p.file_path):
        if patch.patch_content and patch.patch_content.strip():
            combined_parts.append(patch.patch_content)

    # Join with blank lines but no comment headers
    combined_diff = '\n\n'.join(combined_parts)

    # Build provenance chain
    provenance = {
        "trace_root": trace_root,
        "plan_hash": plan.compute_hash(),
        "valid_patch_receipts": {p.patch_id: p.compute_receipt_hash() for p in valid_patches},
        "verification_summary": {v.patch_id: v.verified for v in verifications},
        "status_breakdown": {
            "valid": status.valid_patches,
            "skipped": status.skipped_patches,
            "failed": status.failed_patches,
            "conflict": status.conflict_patches,
        },
    }

    # all_verified means all VALID patches verified (SKIPs don't count against)
    verification_map = {v.patch_id: v.verified for v in verifications}
    failed_verifications = [
        p.patch_id for p in valid_patches
        if not verification_map.get(p.patch_id, False)
    ]
    all_verified = len(failed_verifications) == 0 and status.failed_patches == 0

    # Statistics from VALID patches only
    total_added = sum(p.lines_added for p in valid_patches)
    total_removed = sum(p.lines_removed for p in valid_patches)
    files_modified = len(set(p.file_path for p in valid_patches))

    return DiffRollup(
        trace_root=trace_root,
        plan_hash=plan.compute_hash(),
        patches=[p.to_dict() for p in valid_patches],
        verifications=[v.to_dict() for v in verifications],
        all_patch_results=all_results,
        combined_diff=combined_diff,
        combined_diff_hash=sha256_str(combined_diff),
        total_files_modified=files_modified,
        total_lines_added=total_added,
        total_lines_removed=total_removed,
        status_detail=status,
        all_verified=all_verified,
        failed_patches=failed_verifications + status.failed_patch_ids,
        provenance=provenance,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
    )


# Legacy compatibility wrapper
def build_diff_rollup_legacy(
    plan: RefactorPlan,
    patches: list[PatchPacket],
    verifications: list[PatchVerification],
    trace_root: str,
) -> DiffRollup:
    """Legacy build function for backward compatibility."""
    # Convert to PatchResults
    verification_map = {v.patch_id: v for v in verifications}
    results = []
    for patch in patches:
        v = verification_map.get(patch.patch_id)
        if v and v.verified:
            status = "VALID"
        elif patch.lines_added == 0 and patch.lines_removed == 0:
            status = "SKIP"
        else:
            status = "FAIL"

        results.append(PatchResult(
            patch=patch,
            validation=PatchValidation(
                patch_id=patch.patch_id,
                is_valid=v.verified if v else False,
                status=PatchStatus.VALID.value if v and v.verified else PatchStatus.MALFORMED.value,
            ),
            final_status=status,
        ))

    return build_diff_rollup(plan, results, verifications, trace_root)


# =============================================================================
# Workflow Executors
# =============================================================================

class RefactorPlanExecutor(NodeExecutor):
    """Generate refactor plan from review findings."""

    def execute(self, spec: NodeSpec, context: dict) -> NodeResult:
        import time
        start_time = time.time()

        # Get review result
        review_result = context.get("semgrep") or context.get("review")
        if not review_result or not review_result.packet:
            return NodeResult(node_id=spec.id, success=False, error="Missing review dependency")

        aggregate_hash = review_result.packet.output_hash

        # Load findings
        aggregate_path = self.run_dir / review_result.packet.output_path
        if not aggregate_path.exists():
            return NodeResult(node_id=spec.id, success=False, error="Aggregate file not found")

        aggregate = json.loads(aggregate_path.read_text())
        findings = aggregate.get("findings", [])
        trace_root = aggregate.get("trace_root", "")

        # Get params
        provider = spec.params.get("provider", "openai")
        model = spec.params.get("model", "gpt-4o-mini")

        try:
            plan = generate_refactor_plan(
                findings=findings,
                trace_root=trace_root,
                aggregate_hash=aggregate_hash,
                provider=provider,
                model=model,
            )

            # Write plan
            out_dir = self.run_dir / "refactor"
            out_dir.mkdir(parents=True, exist_ok=True)
            plan_path = out_dir / "refactor_plan.json"
            plan_path.write_text(json.dumps(plan.to_dict(), indent=2))

            duration_ms = int((time.time() - start_time) * 1000)

            packet = AgentPacket(
                node_id=spec.id,
                agent_type="refactor_plan",
                input_hash=sha256_json({"aggregate_hash": aggregate_hash}),
                executor_id=f"{provider}:{model}",
                output_hash=sha256_file(plan_path),
                output_path="refactor/refactor_plan.json",
                artifacts=[{"path": "refactor/refactor_plan.json", "hash": sha256_file(plan_path), "type": "plan"}],
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                duration_ms=duration_ms,
                determinism="non_deterministic",
            )

            return NodeResult(node_id=spec.id, success=True, packet=packet)

        except Exception as e:
            import traceback
            return NodeResult(node_id=spec.id, success=False, error=f"{e}\n{traceback.format_exc()}")


class PatchGeneratorExecutor(NodeExecutor):
    """Multi-agent patch generation with repair loop."""

    def execute(self, spec: NodeSpec, context: dict) -> NodeResult:
        import time
        start_time = time.time()

        # Get plan
        plan_result = context.get("plan") or context.get("refactor_plan")
        if not plan_result or not plan_result.packet:
            return NodeResult(node_id=spec.id, success=False, error="Missing plan dependency")

        plan_path = self.run_dir / plan_result.packet.output_path
        plan_data = json.loads(plan_path.read_text())

        plan = RefactorPlan(
            trace_root=plan_data.get("trace_root", ""),
            review_aggregate_hash=plan_data.get("review_aggregate_hash", ""),
            items=[RefactorItem.from_dict(i) for i in plan_data.get("items", [])],
            execution_order=plan_data.get("execution_order", []),
            agent_assignments=plan_data.get("agent_assignments", {}),
        )

        # Get params
        provider = spec.params.get("provider", "openai")
        model = spec.params.get("model", "gpt-4o-mini")
        max_workers = spec.params.get("max_workers", 4)
        enable_repair = spec.params.get("enable_repair", True)
        # v5 params (enabled by default)
        enable_suppression_memos = spec.params.get("enable_suppression_memos", True)
        enable_ast_validation = spec.params.get("enable_ast_validation", True)

        try:
            patch_results = run_multi_agent_patches(
                plan=plan,
                project_dir=self.project_dir,
                provider=provider,
                model=model,
                max_workers=max_workers,
                enable_repair=enable_repair,
                enable_suppression_memos=enable_suppression_memos,
                enable_ast_validation=enable_ast_validation,
            )

            # Write patches and results
            out_dir = self.run_dir / "refactor" / "patches"
            out_dir.mkdir(parents=True, exist_ok=True)

            artifacts = []
            patch_ids = []
            status_summary = {"valid": 0, "skip": 0, "fail": 0}

            for result in patch_results:
                # Use repair patch if valid, otherwise original
                active_patch = result.repair_patch if result.repaired and result.repair_validation and result.repair_validation.is_valid else result.patch

                patch_file = out_dir / f"{active_patch.patch_id}.json"
                patch_file.write_text(json.dumps(active_patch.to_dict(), indent=2))
                patch_ids.append(active_patch.patch_id)
                artifacts.append({
                    "path": f"refactor/patches/{active_patch.patch_id}.json",
                    "hash": sha256_file(patch_file),
                    "type": "patch",
                    "status": result.final_status,
                })

                # Also write full result
                result_file = out_dir / f"{active_patch.patch_id}_result.json"
                result_file.write_text(json.dumps(result.to_dict(), indent=2))

                if result.final_status == "VALID":
                    status_summary["valid"] += 1
                elif result.final_status == "SKIP":
                    status_summary["skip"] += 1
                else:
                    status_summary["fail"] += 1

            # Write manifest (v2: includes status breakdown)
            manifest = {
                "schema": "patch_manifest_v2",
                "patch_ids": patch_ids,
                "total_patches": len(patch_results),
                "status_summary": status_summary,
                "valid_patch_ids": [
                    r.repair_patch.patch_id if r.repaired and r.repair_validation and r.repair_validation.is_valid else r.patch.patch_id
                    for r in patch_results if r.final_status == "VALID"
                ],
                "skipped_patch_ids": [r.patch.patch_id for r in patch_results if r.final_status == "SKIP"],
                "failed_patch_ids": [r.patch.patch_id for r in patch_results if r.final_status == "FAIL"],
            }
            manifest_path = out_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2))

            duration_ms = int((time.time() - start_time) * 1000)

            packet = AgentPacket(
                node_id=spec.id,
                agent_type="patch_generator_v2",
                input_hash=sha256_json({"plan_hash": plan.compute_hash()}),
                executor_id=f"multi_agent:{provider}:{model}",
                output_hash=sha256_file(manifest_path),
                output_path="refactor/patches/manifest.json",
                artifacts=artifacts,
                timestamp=datetime.datetime.utcnow().isoformat() + "Z",
                duration_ms=duration_ms,
                determinism="non_deterministic",
            )

            return NodeResult(node_id=spec.id, success=True, packet=packet)

        except Exception as e:
            import traceback
            return NodeResult(node_id=spec.id, success=False, error=f"{e}\n{traceback.format_exc()}")


class PatchVerifierExecutor(NodeExecutor):
    """Verify all patches using git apply."""

    def execute(self, spec: NodeSpec, context: dict) -> NodeResult:
        import time
        start_time = time.time()

        # Get patches
        patches_result = context.get("patches") or context.get("patch_generator")
        if not patches_result or not patches_result.packet:
            return NodeResult(node_id=spec.id, success=False, error="Missing patches dependency")

        # Load patch manifest (v2 format)
        manifest_path = self.run_dir / patches_result.packet.output_path
        manifest = json.loads(manifest_path.read_text())

        patches_dir = manifest_path.parent

        verifications = []
        valid_count = 0
        skipped_count = 0
        failed_count = 0

        # Only verify VALID patches (SKIPs are already handled)
        valid_patch_ids = manifest.get("valid_patch_ids", manifest.get("patch_ids", []))
        skipped_patch_ids = manifest.get("skipped_patch_ids", [])

        for patch_id in valid_patch_ids:
            patch_file = patches_dir / f"{patch_id}.json"
            if not patch_file.exists():
                verifications.append(PatchVerification(
                    patch_id=patch_id,
                    verified=False,
                    original_hash_match=False,
                    patch_applies_cleanly=False,
                    result_hash_match=False,
                    error=f"Patch file not found: {patch_id}",
                ))
                failed_count += 1
                continue

            patch_data = json.loads(patch_file.read_text())
            patch = PatchPacket(**{k: v for k, v in patch_data.items() if k not in ("receipt_hash", "schema")})

            # Load original file
            original_path = self.project_dir / patch.file_path
            if not original_path.exists():
                verifications.append(PatchVerification(
                    patch_id=patch_id,
                    verified=False,
                    original_hash_match=False,
                    patch_applies_cleanly=False,
                    result_hash_match=False,
                    error=f"Original file not found: {patch.file_path}",
                ))
                failed_count += 1
                continue

            original_content = original_path.read_text()
            verification = verify_patch(patch, original_content, self.project_dir)
            verifications.append(verification)

            if verification.verified:
                valid_count += 1
            else:
                failed_count += 1

        # Add SKIP verifications (they're "verified" but skipped)
        for patch_id in skipped_patch_ids:
            verifications.append(PatchVerification(
                patch_id=patch_id,
                verified=True,  # SKIPs are considered verified (no change to verify)
                original_hash_match=True,
                patch_applies_cleanly=True,
                result_hash_match=True,
                actual_result_hash="",
                error="",
            ))
            skipped_count += 1

        # Write verifications
        out_dir = self.run_dir / "refactor" / "verifications"
        out_dir.mkdir(parents=True, exist_ok=True)

        # v2 summary with status breakdown
        summary = {
            "schema": "verification_summary_v2",
            "all_passed": failed_count == 0,
            "total": len(verifications),
            "passed": valid_count + skipped_count,  # SKIPs count as passed
            "failed": failed_count,
            "status_breakdown": {
                "verified_valid": valid_count,
                "verified_skip": skipped_count,
                "failed": failed_count,
            },
            "verifications": [v.to_dict() for v in verifications],
        }
        summary_path = out_dir / "verification_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        duration_ms = int((time.time() - start_time) * 1000)

        packet = AgentPacket(
            node_id=spec.id,
            agent_type="patch_verifier_v2",
            input_hash=sha256_json({"valid_patch_ids": valid_patch_ids, "skipped_patch_ids": skipped_patch_ids}),
            executor_id="patch_verifier_v2",
            output_hash=sha256_file(summary_path),
            output_path="refactor/verifications/verification_summary.json",
            artifacts=[{"path": "refactor/verifications/verification_summary.json", "hash": sha256_file(summary_path), "type": "verification"}],
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            duration_ms=duration_ms,
            determinism="deterministic",
        )

        return NodeResult(node_id=spec.id, success=True, packet=packet)


class DiffRollupExecutor(NodeExecutor):
    """Build final diff rollup with status_detail."""

    def execute(self, spec: NodeSpec, context: dict) -> NodeResult:
        import time
        start_time = time.time()

        # Get plan, patches, verifications
        plan_result = context.get("plan") or context.get("refactor_plan")
        patches_result = context.get("patches") or context.get("patch_generator")
        verify_result = context.get("verify") or context.get("patch_verifier")

        if not all([plan_result, patches_result, verify_result]):
            missing = []
            if not plan_result: missing.append("plan")
            if not patches_result: missing.append("patches")
            if not verify_result: missing.append("verify")
            return NodeResult(node_id=spec.id, success=False, error=f"Missing dependencies: {missing}")

        # Check all have packets
        if not plan_result.packet:
            return NodeResult(node_id=spec.id, success=False, error="Plan result has no packet")
        if not patches_result.packet:
            return NodeResult(node_id=spec.id, success=False, error="Patches result has no packet")
        if not verify_result.packet:
            return NodeResult(node_id=spec.id, success=False, error="Verify result has no packet")

        # Load plan
        plan_path = self.run_dir / plan_result.packet.output_path
        plan_data = json.loads(plan_path.read_text())
        plan = RefactorPlan(
            trace_root=plan_data.get("trace_root", ""),
            review_aggregate_hash=plan_data.get("review_aggregate_hash", ""),
        )

        # Load patches and results (v2 format)
        patches_dir = (self.run_dir / patches_result.packet.output_path).parent
        manifest = json.loads((patches_dir / "manifest.json").read_text())

        # Build PatchResult list from files
        patch_results = []
        for patch_id in manifest.get("patch_ids", []):
            patch_file = patches_dir / f"{patch_id}.json"
            result_file = patches_dir / f"{patch_id}_result.json"

            if result_file.exists():
                # Load full result
                result_data = json.loads(result_file.read_text())
                patch_data = result_data.get("patch", {})
                validation_data = result_data.get("validation", {})

                patch = PatchPacket(**{k: v for k, v in patch_data.items() if k not in ("receipt_hash", "schema")})
                validation = PatchValidation(**{k: v for k, v in validation_data.items()})

                repair_patch = None
                repair_validation = None
                if result_data.get("repair_patch"):
                    repair_patch = PatchPacket(**{k: v for k, v in result_data["repair_patch"].items() if k not in ("receipt_hash", "schema")})
                if result_data.get("repair_validation"):
                    repair_validation = PatchValidation(**result_data["repair_validation"])

                # v3/v4: Load additional fields
                no_change_proof = None
                if result_data.get("no_change_proof"):
                    proof_data = result_data["no_change_proof"]
                    span_anchor = None
                    if proof_data.get("span_anchor"):
                        span_anchor = SpanAnchor(**proof_data["span_anchor"])
                    no_change_proof = NoChangeProof(
                        disposition=proof_data.get("disposition", "unproven"),
                        justification=proof_data.get("justification", ""),
                        file_pre_hash=proof_data.get("file_pre_hash", ""),
                        evidence_chunk_hash=proof_data.get("evidence_chunk_hash", ""),
                        span_anchor=span_anchor,
                        evidence_lines=proof_data.get("evidence_lines", []),
                        evidence_snippet=proof_data.get("evidence_snippet", ""),
                        suppression_comment=proof_data.get("suppression_comment", ""),
                        waiver_policy_ref=proof_data.get("waiver_policy_ref", ""),
                        waiver_artifact_hash=proof_data.get("waiver_artifact_hash", ""),
                    )

                finding_delta = None
                if result_data.get("finding_delta"):
                    delta_data = result_data["finding_delta"]
                    finding_delta = FindingDelta(
                        resolved_fingerprints=delta_data.get("resolved_fingerprints", []),
                        remaining_fingerprints=delta_data.get("remaining_fingerprints", []),
                        new_findings=delta_data.get("new_findings", []),
                        all_targeted_resolved=delta_data.get("all_targeted_resolved", False),
                        no_new_warnings=delta_data.get("no_new_warnings", True),
                    )

                importability = None
                if result_data.get("importability"):
                    imp_data = result_data["importability"]
                    importability = ImportabilitySanity(
                        passed=imp_data.get("passed", True),
                        can_compile=imp_data.get("can_compile", True),
                        can_import=imp_data.get("can_import", True),
                        syntax_error=imp_data.get("syntax_error", ""),
                        import_error=imp_data.get("import_error", ""),
                        errors=imp_data.get("errors", []),
                    )

                outcome = None
                if result_data.get("outcome"):
                    out_data = result_data["outcome"]
                    outcome = PatchOutcome(
                        apply_status=ApplyStatus(out_data.get("apply_status", "applies")),
                        effect_status=EffectStatus(out_data.get("effect_status", "no_effect")),
                        apply_error=out_data.get("apply_error", ""),
                        effect_details=out_data.get("effect_details", ""),
                    )

                patch_results.append(PatchResult(
                    patch=patch,
                    validation=validation,
                    repaired=result_data.get("repaired", False),
                    repair_patch=repair_patch,
                    repair_validation=repair_validation,
                    final_status=result_data.get("final_status", "VALID"),
                    no_change_proof=no_change_proof,
                    finding_delta=finding_delta,
                    importability=importability,
                    outcome=outcome,
                ))
            elif patch_file.exists():
                # Legacy: load just the patch
                patch_data = json.loads(patch_file.read_text())
                patch = PatchPacket(**{k: v for k, v in patch_data.items() if k not in ("receipt_hash", "schema")})

                # Determine status from manifest
                if patch_id in manifest.get("valid_patch_ids", []):
                    final_status = "VALID"
                elif patch_id in manifest.get("skipped_patch_ids", []):
                    final_status = "SKIP"
                else:
                    final_status = "FAIL"

                patch_results.append(PatchResult(
                    patch=patch,
                    validation=PatchValidation(
                        patch_id=patch_id,
                        is_valid=final_status == "VALID",
                        status=PatchStatus.VALID.value if final_status == "VALID" else PatchStatus.NO_CHANGE.value if final_status == "SKIP" else PatchStatus.MALFORMED.value,
                    ),
                    final_status=final_status,
                ))

        # Load verifications
        verify_path = self.run_dir / verify_result.packet.output_path
        verify_data = json.loads(verify_path.read_text())
        verifications = [PatchVerification(**v) for v in verify_data.get("verifications", [])]

        # Build rollup with status_detail
        rollup = build_diff_rollup(
            plan=plan,
            patch_results=patch_results,
            verifications=verifications,
            trace_root=plan.trace_root,
        )

        # Write rollup
        out_dir = self.run_dir / "refactor"
        rollup_path = out_dir / "diff_rollup.json"
        rollup_path.write_text(json.dumps(rollup.to_dict(), indent=2))

        # Write combined diff as separate file (pure diff format)
        diff_path = out_dir / "combined.diff"
        diff_path.write_text(rollup.combined_diff)

        duration_ms = int((time.time() - start_time) * 1000)

        # Node fails if there are failed patches
        node_success = rollup.status_detail.failed_patches == 0

        packet = AgentPacket(
            node_id=spec.id,
            agent_type="diff_rollup_v2",
            input_hash=sha256_json({
                "plan_hash": plan.compute_hash(),
                "patch_count": len(patch_results),
            }),
            executor_id="diff_rollup_v2",
            output_hash=sha256_file(rollup_path),
            output_path="refactor/diff_rollup.json",
            artifacts=[
                {"path": "refactor/diff_rollup.json", "hash": sha256_file(rollup_path), "type": "rollup"},
                {"path": "refactor/combined.diff", "hash": sha256_file(diff_path), "type": "diff"},
            ],
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            duration_ms=duration_ms,
            determinism="deterministic",
        )

        return NodeResult(
            node_id=spec.id,
            success=node_success,
            packet=packet,
            error="" if node_success else f"Failed patches: {rollup.status_detail.failed_patch_ids}",
        )


# Register executors
REFACTOR_EXECUTORS = {
    "refactor.plan": RefactorPlanExecutor,
    "refactor.patches": PatchGeneratorExecutor,
    "refactor.verify": PatchVerifierExecutor,
    "refactor.rollup": DiffRollupExecutor,
}

# Add to NODE_DETERMINISM
NODE_DETERMINISM.update({
    "refactor.plan": Determinism.NON_DETERMINISTIC,
    "refactor.patches": Determinism.NON_DETERMINISTIC,
    "refactor.verify": Determinism.DETERMINISTIC,
    "refactor.rollup": Determinism.DETERMINISTIC,
})
