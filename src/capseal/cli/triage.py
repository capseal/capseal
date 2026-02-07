"""Triage Gate - controls agent spawning based on task signals.

The triage gate implements a "start small, expand if needed" strategy:
1. Start with ONE generalist agent
2. Only spawn specialists when specific signals are detected
3. Share context via MWS (Minimal Working Set) to avoid redundant indexing

This prevents the "parallel agent explosion" problem where 5 agents all
try to index the same repo simultaneously, wasting tokens and API calls.

Signals that trigger specialist spawning:
- SECURITY: Patterns suggesting security-critical changes
- CRYPTO: Cryptographic code modifications
- SCHEMA: Database/trace schema changes
- API: Public API surface changes
- PERFORMANCE: Performance-critical paths
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Specialist(Enum):
    """Specialist agent types that can be spawned."""
    SECURITY = "security"
    CRYPTO = "crypto"
    SCHEMA = "schema"
    API = "api"
    PERFORMANCE = "performance"
    TESTING = "testing"


@dataclass
class TriageSignal:
    """A signal that may trigger specialist spawning."""
    specialist: Specialist
    pattern: str  # Regex pattern
    description: str
    confidence: float  # 0.0 - 1.0
    file_pattern: str | None = None  # Optional file path pattern


# Default signals for specialist triggering
DEFAULT_SIGNALS: list[TriageSignal] = [
    # Security signals
    TriageSignal(
        specialist=Specialist.SECURITY,
        pattern=r"\b(auth|password|token|secret|credential|jwt|oauth|session)\b",
        description="Authentication/authorization code",
        confidence=0.7,
    ),
    TriageSignal(
        specialist=Specialist.SECURITY,
        pattern=r"\b(sql|query|execute|cursor|inject)\b.*\b(user|input|param|arg)\b",
        description="Potential SQL injection vectors",
        confidence=0.8,
    ),
    TriageSignal(
        specialist=Specialist.SECURITY,
        pattern=r"\b(eval|exec|subprocess|os\.system|shell)\b",
        description="Code execution patterns",
        confidence=0.9,
    ),
    TriageSignal(
        specialist=Specialist.SECURITY,
        pattern=r"\b(cors|csp|xss|csrf|sanitize|escape|validate)\b",
        description="Web security mechanisms",
        confidence=0.7,
    ),

    # Crypto signals
    TriageSignal(
        specialist=Specialist.CRYPTO,
        pattern=r"\b(aes|rsa|sha|hmac|encrypt|decrypt|cipher|hash|digest)\b",
        description="Cryptographic operations",
        confidence=0.8,
    ),
    TriageSignal(
        specialist=Specialist.CRYPTO,
        pattern=r"\b(merkle|commitment|proof|witness|snark|stark|fri)\b",
        description="Zero-knowledge/proof systems",
        confidence=0.9,
    ),
    TriageSignal(
        specialist=Specialist.CRYPTO,
        pattern=r"\b(sign|verify|signature|keypair|pubkey|privkey)\b",
        description="Digital signatures",
        confidence=0.8,
    ),

    # Schema signals
    TriageSignal(
        specialist=Specialist.SCHEMA,
        pattern=r"\b(schema|migration|alter|create table|drop|column)\b",
        description="Database schema changes",
        confidence=0.8,
    ),
    TriageSignal(
        specialist=Specialist.SCHEMA,
        pattern=r"\b(TraceSchema|trace_schema|schema_id|field_map)\b",
        description="Trace schema definitions",
        confidence=0.9,
        file_pattern="*.py",
    ),

    # API signals
    TriageSignal(
        specialist=Specialist.API,
        pattern=r"@(app|router)\.(get|post|put|delete|patch)",
        description="HTTP endpoint definitions",
        confidence=0.7,
    ),
    TriageSignal(
        specialist=Specialist.API,
        pattern=r"\b(openapi|swagger|graphql|grpc|protobuf)\b",
        description="API specification formats",
        confidence=0.8,
    ),
    TriageSignal(
        specialist=Specialist.API,
        pattern=r"class\s+\w+(Request|Response|DTO|Schema)\b",
        description="API data structures",
        confidence=0.7,
    ),

    # Performance signals
    TriageSignal(
        specialist=Specialist.PERFORMANCE,
        pattern=r"\b(cache|memoize|lru_cache|redis|memcache)\b",
        description="Caching mechanisms",
        confidence=0.6,
    ),
    TriageSignal(
        specialist=Specialist.PERFORMANCE,
        pattern=r"\b(async|await|concurrent|parallel|thread|pool)\b",
        description="Concurrency patterns",
        confidence=0.5,
    ),
    TriageSignal(
        specialist=Specialist.PERFORMANCE,
        pattern=r"\b(index|optimize|query plan|explain|n\+1)\b",
        description="Query optimization",
        confidence=0.7,
    ),

    # Testing signals
    TriageSignal(
        specialist=Specialist.TESTING,
        pattern=r"\b(test_|_test|spec_|_spec|mock|stub|fixture)\b",
        description="Test code patterns",
        confidence=0.6,
        file_pattern="*test*.py",
    ),
]


@dataclass
class TriageResult:
    """Result of triage analysis."""
    needs_specialists: bool
    triggered_specialists: dict[Specialist, list[str]]  # specialist -> reasons
    confidence_scores: dict[Specialist, float]
    recommended_order: list[Specialist]  # Order to spawn specialists
    context_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "needs_specialists": self.needs_specialists,
            "triggered_specialists": {
                k.value: v for k, v in self.triggered_specialists.items()
            },
            "confidence_scores": {
                k.value: round(v, 2) for k, v in self.confidence_scores.items()
            },
            "recommended_order": [s.value for s in self.recommended_order],
            "context_summary": self.context_summary,
        }


class TriageGate:
    """Analyzes diffs/context to determine if specialists should be spawned.

    Usage:
        gate = TriageGate()
        result = gate.analyze(diff_text, changed_files)

        if result.needs_specialists:
            for specialist in result.recommended_order:
                spawn_agent(specialist)
    """

    def __init__(
        self,
        signals: list[TriageSignal] | None = None,
        confidence_threshold: float = 0.7,
        max_specialists: int = 3,
    ):
        self.signals = signals or DEFAULT_SIGNALS
        self.confidence_threshold = confidence_threshold
        self.max_specialists = max_specialists

    def analyze(
        self,
        diff_text: str,
        changed_files: list[str] | None = None,
    ) -> TriageResult:
        """Analyze diff/context to determine specialist needs.

        Args:
            diff_text: The unified diff or code content
            changed_files: List of changed file paths

        Returns:
            TriageResult with specialist recommendations
        """
        triggered: dict[Specialist, list[str]] = {}
        scores: dict[Specialist, float] = {}

        # Analyze diff against all signals
        for signal in self.signals:
            # Check file pattern if specified
            if signal.file_pattern and changed_files:
                import fnmatch
                if not any(fnmatch.fnmatch(f, signal.file_pattern) for f in changed_files):
                    continue

            # Check content pattern
            matches = re.findall(signal.pattern, diff_text, re.IGNORECASE)
            if matches:
                specialist = signal.specialist
                if specialist not in triggered:
                    triggered[specialist] = []
                    scores[specialist] = 0.0

                triggered[specialist].append(signal.description)
                # Accumulate confidence (capped at 1.0)
                scores[specialist] = min(1.0, scores[specialist] + signal.confidence * 0.3)

        # Filter by confidence threshold
        qualified = {
            s: reasons
            for s, reasons in triggered.items()
            if scores.get(s, 0) >= self.confidence_threshold
        }

        # Determine spawn order (highest confidence first)
        ordered = sorted(
            qualified.keys(),
            key=lambda s: scores.get(s, 0),
            reverse=True,
        )[:self.max_specialists]

        # Build context summary
        if qualified:
            summary_parts = []
            for s in ordered:
                reasons = qualified[s][:3]  # Top 3 reasons
                summary_parts.append(f"{s.value}: {', '.join(reasons)}")
            context_summary = "Specialist signals detected: " + "; ".join(summary_parts)
        else:
            context_summary = "No specialist signals detected. Generalist review sufficient."

        return TriageResult(
            needs_specialists=len(qualified) > 0,
            triggered_specialists=qualified,
            confidence_scores=scores,
            recommended_order=ordered,
            context_summary=context_summary,
        )

    def quick_check(self, diff_text: str) -> bool:
        """Quick check if ANY specialist might be needed.

        Faster than full analyze() - use for early filtering.
        """
        for signal in self.signals:
            if signal.confidence >= 0.8:  # Only check high-confidence signals
                if re.search(signal.pattern, diff_text, re.IGNORECASE):
                    return True
        return False


@dataclass
class AgentPlan:
    """Plan for agent execution."""
    generalist_prompt: str
    specialists_to_spawn: list[Specialist]
    shared_context_path: Path | None
    max_parallel: int = 1  # Start sequential, enable parallel if needed
    timeout_sec: int = 75


class TriageOrchestrator:
    """Orchestrates agent spawning based on triage results.

    This implements the "one agent first" strategy:
    1. Run generalist agent with full MWS context
    2. If generalist flags issues, spawn specialists
    3. Specialists share the same MWS (no re-indexing)
    """

    def __init__(
        self,
        gate: TriageGate | None = None,
    ):
        self.gate = gate or TriageGate()

    def plan(
        self,
        diff_text: str,
        changed_files: list[str],
        mws_path: Path | None = None,
    ) -> AgentPlan:
        """Create an agent execution plan.

        Args:
            diff_text: The unified diff
            changed_files: List of changed file paths
            mws_path: Path to pre-built MWS context (optional)

        Returns:
            AgentPlan specifying what agents to run
        """
        result = self.gate.analyze(diff_text, changed_files)

        # Build generalist prompt
        generalist_prompt = self._build_generalist_prompt(
            changed_files=changed_files,
            triage_hints=result.context_summary,
        )

        return AgentPlan(
            generalist_prompt=generalist_prompt,
            specialists_to_spawn=result.recommended_order,
            shared_context_path=mws_path,
            max_parallel=1 if result.needs_specialists else 1,  # Sequential for now
            timeout_sec=75,
        )

    def _build_generalist_prompt(
        self,
        changed_files: list[str],
        triage_hints: str,
    ) -> str:
        """Build the generalist agent's review prompt."""
        files_summary = "\n".join(f"- {f}" for f in changed_files[:20])
        if len(changed_files) > 20:
            files_summary += f"\n... and {len(changed_files) - 20} more"

        return f"""Review the following code changes. Focus on:
1. Correctness: Logic errors, edge cases, null handling
2. Security: Input validation, auth checks, injection risks
3. Maintainability: Code clarity, naming, documentation

Changed files:
{files_summary}

{triage_hints}

Constraints:
- Do not propose new systems; extend existing primitives
- Focus on the specific changes in this diff
- Cite file paths and line numbers in findings
- Be concise: bullet points preferred over paragraphs
"""


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def triage_diff(diff_text: str, changed_files: list[str] | None = None) -> TriageResult:
    """Quick triage of a diff to determine specialist needs.

    Args:
        diff_text: The unified diff
        changed_files: Optional list of changed file paths

    Returns:
        TriageResult with recommendations
    """
    gate = TriageGate()
    return gate.analyze(diff_text, changed_files)


def needs_security_review(diff_text: str) -> bool:
    """Quick check if diff needs security specialist."""
    gate = TriageGate()
    result = gate.analyze(diff_text)
    return Specialist.SECURITY in result.triggered_specialists


def needs_crypto_review(diff_text: str) -> bool:
    """Quick check if diff needs crypto specialist."""
    gate = TriageGate()
    result = gate.analyze(diff_text)
    return Specialist.CRYPTO in result.triggered_specialists
