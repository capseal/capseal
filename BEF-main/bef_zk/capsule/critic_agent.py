"""
Critic Agent - Adversarial Review for Asymmetric Code Review

This module implements the "critic" role in asymmetric code review:

1. Proposer agents generate patches/claims (untrusted)
2. Critic agent challenges the proposals (adversarial)
3. Deterministic checkers make final verdicts (trusted)

The critic's job is to:
- Find security holes the proposer missed
- Identify edge cases not covered
- Challenge NO_CHANGE justifications
- Spot logical errors in patches

The key insight: LLMs are better at finding problems than proving absence
of problems. Use this adversarially.

IMPORTANT: Critic output is NOT truth - it's a list of attack hypotheses.
Each challenge should be converted into a checkable obligation that
deterministic checkers can verify.
"""
from __future__ import annotations

import json
import os
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from canonical.project_trace import sha256_bytes


@dataclass
class SpanAnchor:
    """Anchor to a specific code span for verification."""
    file_path: str
    start_line: int
    end_line: int
    content_hash: str  # Hash of the span content

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "content_hash": self.content_hash,
        }


@dataclass
class Challenge:
    """A challenge raised by the critic against a proposer's output."""
    challenge_id: str
    challenge_hash: str  # For memoization of repeated critiques

    # Target binding - what exactly is being challenged
    target_type: str  # "patch", "no_change_proof", "claim"
    target_id: str  # ID of the thing being challenged (must be specific!)
    target_fingerprint: str  # Hash of target content for linking

    # Location
    file_path: str

    # Classification
    severity: str  # "critical", "high", "medium", "low"
    category: str  # "security", "logic", "edge_case", "incomplete", "assumption"

    # Details
    description: str

    # Optional fields with defaults
    evidence_span: Optional[SpanAnchor] = None  # Where the issue is
    counterexample: Optional[str] = None  # Specific input that breaks the code
    suggested_test: Optional[str] = None  # Test that would expose the issue

    # Derived checkable claim (if applicable)
    derived_claim_type: Optional[str] = None  # e.g., "ALLOWLIST_CONSTANT"
    derived_claim_description: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "challenge_id": self.challenge_id,
            "challenge_hash": self.challenge_hash,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "target_fingerprint": self.target_fingerprint,
            "file_path": self.file_path,
            "evidence_span": self.evidence_span.to_dict() if self.evidence_span else None,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "counterexample": self.counterexample,
            "suggested_test": self.suggested_test,
            "derived_claim_type": self.derived_claim_type,
            "derived_claim_description": self.derived_claim_description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Challenge":
        span_data = d.get("evidence_span")
        return cls(
            challenge_id=d["challenge_id"],
            challenge_hash=d["challenge_hash"],
            target_type=d["target_type"],
            target_id=d["target_id"],
            target_fingerprint=d.get("target_fingerprint", ""),
            file_path=d["file_path"],
            evidence_span=SpanAnchor(**span_data) if span_data else None,
            severity=d["severity"],
            category=d["category"],
            description=d["description"],
            counterexample=d.get("counterexample"),
            suggested_test=d.get("suggested_test"),
            derived_claim_type=d.get("derived_claim_type"),
            derived_claim_description=d.get("derived_claim_description"),
        )


@dataclass
class CriticResult:
    """Result of critic review."""
    challenges: list[Challenge] = field(default_factory=list)
    summary: str = ""
    confidence: float = 0.0  # 0-1, how confident critic is in challenges
    raw_output: str = ""
    duration_ms: int = 0

    # Stats
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    # Derived obligations (checkable claims from challenges)
    derived_obligations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "challenges": [c.to_dict() for c in self.challenges],
            "summary": self.summary,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "stats": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
            },
            "derived_obligations": self.derived_obligations,
        }

    def compute_stats(self) -> None:
        """Recompute stats from challenges."""
        self.critical_count = sum(1 for c in self.challenges if c.severity == "critical")
        self.high_count = sum(1 for c in self.challenges if c.severity == "high")
        self.medium_count = sum(1 for c in self.challenges if c.severity == "medium")
        self.low_count = sum(1 for c in self.challenges if c.severity == "low")

    def has_blockers(self) -> bool:
        """Check if there are critical/high challenges that should block."""
        return self.critical_count > 0 or self.high_count > 0

    def derive_obligations(self) -> None:
        """Convert challenges into checkable obligations."""
        self.derived_obligations = []

        for c in self.challenges:
            # Map challenge categories to claim types
            if "whitelist" in c.description.lower() or "allowlist" in c.description.lower():
                if "mutable" in c.description.lower() or "modified" in c.description.lower():
                    self.derived_obligations.append({
                        "claim_type": "ALLOWLIST_CONSTANT",
                        "file_path": c.file_path,
                        "description": "Verify allowlist mapping is constant (no mutation sites)",
                        "source_challenge_id": c.challenge_id,
                        "checker": "ast",
                    })
                self.derived_obligations.append({
                    "claim_type": "ALLOWLIST_ENFORCED",
                    "file_path": c.file_path,
                    "description": "Verify allowlist is properly enforced (unknown keys raise error)",
                    "source_challenge_id": c.challenge_id,
                    "checker": "ast",
                })

            if "sql" in c.description.lower() or "injection" in c.description.lower():
                self.derived_obligations.append({
                    "claim_type": "NO_SQL_INJECTION",
                    "file_path": c.file_path,
                    "description": "Verify no SQL injection vulnerabilities",
                    "source_challenge_id": c.challenge_id,
                    "checker": "semgrep",
                })

            if c.suggested_test:
                self.derived_obligations.append({
                    "claim_type": "TEST_COVERAGE",
                    "file_path": c.file_path,
                    "description": f"Run test: {c.suggested_test[:100]}",
                    "source_challenge_id": c.challenge_id,
                    "checker": "pytest",
                    "test_code": c.suggested_test,
                })


CRITIC_PROMPT_VERSION = "critic_v2"

CRITIC_SYSTEM_PROMPT = """You are an ADVERSARIAL CODE REVIEWER. Your job is to CHALLENGE proposed changes.

Your goal: Find problems the proposer missed. Be skeptical. Be thorough.

You are NOT trying to be helpful to the proposer. You are trying to BREAK their code.

CRITICAL: Your challenges must be SPECIFIC and ACTIONABLE. Vague concerns are useless.
Each challenge should suggest a concrete test or checkable property."""


def render_critic_prompt(
    patches: list[dict],
    no_change_proofs: list[dict],
    file_contents: dict[str, str],
    findings: list[dict],
) -> str:
    """Render the critic's review prompt."""

    prompt_parts = [
        textwrap.dedent("""
        ## YOUR ROLE: ADVERSARIAL CRITIC

        A proposer agent has reviewed security findings and produced:
        1. Patches to fix issues
        2. NO_CHANGE justifications for issues they claim are false positives

        YOUR JOB: Find problems with their work. Challenge everything.

        ## WHAT TO LOOK FOR:

        **For PATCHES:**
        - Does the fix actually address the root cause, or just the symptom?
        - Are there edge cases the patch doesn't handle?
        - Does the patch introduce NEW vulnerabilities?
        - Is the patch overly complex when a simpler fix exists?
        - Does the patch break existing functionality?

        **For NO_CHANGE justifications:**
        - Is the justification actually valid?
        - Are there input vectors the proposer didn't consider?
        - Is the "mitigation" they cite actually effective?
        - Could an attacker bypass the supposed protection?
        - Does the proof SHOW the enforcement code, or just CLAIM it exists?

        ## THREAT MODEL CONSIDERATIONS:
        - "Dict is mutable at runtime" only matters if attacker has code execution
        - Focus on realistic attack surfaces: user input, external data, plugins
        - Don't flag theoretical issues that require already having pwned the process

        ## OUTPUT FORMAT

        Output STRICT JSON matching this schema:
        ```json
        {
            "challenges": [
                {
                    "target_type": "patch|no_change_proof",
                    "target_id": "EXACT ID from the input (e.g., 'NCP-001' or 'PATCH-services/__init__.py')",
                    "file_path": "path/to/file.py",
                    "line_range": [start_line, end_line],
                    "severity": "critical|high|medium|low",
                    "category": "security|logic|edge_case|incomplete|assumption",
                    "description": "What's wrong and why it matters",
                    "counterexample": "Specific input/scenario that breaks this",
                    "suggested_test": "pytest code that would expose the issue",
                    "checkable_claim": "What deterministic property should be verified (e.g., 'ALLOWLIST_CONSTANT: mapping has no mutation sites')"
                }
            ],
            "summary": "Overall assessment",
            "confidence": 0.0-1.0,
            "threat_model_notes": "What threat model assumptions affect these challenges"
        }
        ```

        RULES:
        1. **target_id MUST match exactly** - use the ID provided in the input
        2. Be specific. Vague concerns are useless.
        3. Provide counterexamples when possible - specific inputs that break things
        4. suggested_test should be runnable pytest code if possible
        5. checkable_claim should be a property a deterministic checker can verify
        6. severity=critical means "exploitable security vulnerability"
        7. Consider threat model - don't flag issues that require already having code execution
        """).strip(),
        "",
        "## ORIGINAL FINDINGS (what the proposer was asked to fix):",
        "",
    ]

    for i, finding in enumerate(findings, 1):
        fingerprint = finding.get('fingerprint', f"finding-{i}")
        prompt_parts.append(f"{i}. ID={fingerprint} [{finding.get('severity', 'warning')}] {finding.get('file_path', 'unknown')}")
        prompt_parts.append(f"   Rule: {finding.get('rule_id', 'unknown')}")
        prompt_parts.append(f"   Message: {finding.get('message', '')}")
        if finding.get('snippet'):
            prompt_parts.append(f"   ```\n{finding.get('snippet')[:500]}\n   ```")
        prompt_parts.append("")

    prompt_parts.append("## PROPOSER'S PATCHES:")
    prompt_parts.append("")

    if not patches:
        prompt_parts.append("(No patches - proposer claims all findings are false positives)")
        prompt_parts.append("")
    else:
        for idx, patch in enumerate(patches):
            patch_id = patch.get('item_id') or patch.get('patch_id') or f"PATCH-{idx}"
            prompt_parts.append(f"### ID={patch_id}")
            prompt_parts.append(f"File: {patch.get('file_path', 'unknown')}")
            prompt_parts.append(f"Status: {patch.get('status', 'unknown')}")
            if patch.get('diff'):
                prompt_parts.append(f"```diff\n{patch.get('diff')[:1500]}\n```")
            prompt_parts.append("")

    prompt_parts.append("## PROPOSER'S NO_CHANGE JUSTIFICATIONS:")
    prompt_parts.append("")

    if not no_change_proofs:
        prompt_parts.append("(No NO_CHANGE proofs)")
        prompt_parts.append("")
    else:
        for idx, proof in enumerate(no_change_proofs):
            proof_id = proof.get('item_id') or proof.get('proof_id') or f"NCP-{idx:03d}"
            prompt_parts.append(f"### ID={proof_id}")
            prompt_parts.append(f"File: {proof.get('file_path', 'unknown')}")
            prompt_parts.append(f"Disposition: {proof.get('disposition', 'unknown')}")
            prompt_parts.append(f"Justification: {proof.get('justification', '')}")
            if proof.get('evidence_snippet'):
                prompt_parts.append(f"Evidence snippet:\n```\n{proof.get('evidence_snippet')[:800]}\n```")
            else:
                prompt_parts.append("Evidence snippet: (NOT PROVIDED - this is suspicious!)")
            prompt_parts.append("")

    # Add relevant file contents for context
    if file_contents:
        prompt_parts.append("## FILE CONTENTS (for reference):")
        prompt_parts.append("")
        for path, content in file_contents.items():
            # Truncate long files
            lines = content.split('\n')
            if len(lines) > 150:
                content = '\n'.join(lines[:150]) + f"\n... ({len(lines) - 150} more lines)"
            prompt_parts.append(f"### {path}")
            prompt_parts.append(f"```python\n{content}\n```")
            prompt_parts.append("")

    return "\n".join(prompt_parts)


def _call_openai_critic(prompt: str, model: str, temperature: float, max_tokens: int, timeout: int) -> str:
    """Call OpenAI API for critic review."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    data = json.dumps({
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"OpenAI API error {e.code}: {e.read().decode()[:200]}")

    choices = payload.get("choices", [])
    if choices:
        return choices[0].get("message", {}).get("content", "")
    raise RuntimeError("OpenAI API returned no choices")


def _call_anthropic_critic(prompt: str, model: str, temperature: float, max_tokens: int, timeout: int) -> str:
    """Call Anthropic API for critic review."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    data = json.dumps({
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": CRITIC_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Anthropic API error {e.code}: {e.read().decode()[:200]}")

    content = payload.get("content", [])
    if content:
        return content[0].get("text", "")
    raise RuntimeError("Anthropic API returned empty content")


def _parse_critic_output(
    raw_text: str,
    patches: list[dict],
    no_change_proofs: list[dict],
) -> tuple[list[Challenge], str, float, str]:
    """Parse critic's JSON output into challenges with proper target binding."""
    # Strip code fences if present
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    # Try to find JSON object
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        stripped = stripped[start:end + 1]

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ValueError(f"Critic output not valid JSON: {e}")

    # Build lookup for target fingerprints
    target_fingerprints = {}
    for idx, p in enumerate(patches):
        pid = p.get('item_id') or p.get('patch_id') or f"PATCH-{idx}"
        target_fingerprints[pid] = sha256_bytes(json.dumps(p, sort_keys=True).encode())[:16]
    for idx, ncp in enumerate(no_change_proofs):
        nid = ncp.get('item_id') or ncp.get('proof_id') or f"NCP-{idx:03d}"
        target_fingerprints[nid] = sha256_bytes(json.dumps(ncp, sort_keys=True).encode())[:16]

    challenges = []
    rejected_challenges = []  # Track challenges with bad target_ids

    for idx, c in enumerate(parsed.get("challenges", [])):
        target_id = c.get("target_id", "unknown")

        # STRICT ENFORCEMENT: Reject challenges with unknown/empty target_ids
        if not target_id or target_id == "unknown" or target_id.lower() == "unknown":
            rejected_challenges.append({
                "index": idx,
                "reason": "unknown_target_id",
                "description": c.get("description", "")[:50],
            })
            continue  # Skip this challenge

        target_fingerprint = target_fingerprints.get(target_id, "unlinked")

        # If target_id doesn't match any known target, try to repair
        if target_fingerprint == "unlinked":
            # Try fuzzy match on file path
            file_path = c.get("file_path", "")
            for known_id, fp in target_fingerprints.items():
                # Check if the file path matches
                if file_path and any(file_path in str(p.get("file_path", "")) for p in patches + no_change_proofs):
                    target_fingerprint = fp
                    break

        # Compute challenge hash for memoization
        challenge_content = json.dumps({
            "target_id": target_id,
            "file_path": c.get("file_path"),
            "category": c.get("category"),
            "description": c.get("description"),
        }, sort_keys=True)
        challenge_hash = sha256_bytes(challenge_content.encode())[:16]
        challenge_id = f"CRIT-{idx:03d}-{challenge_hash}"

        # Build span anchor if line range provided
        evidence_span = None
        line_range = c.get("line_range")
        if line_range and isinstance(line_range, list) and len(line_range) == 2:
            evidence_span = SpanAnchor(
                file_path=c.get("file_path", ""),
                start_line=line_range[0],
                end_line=line_range[1],
                content_hash="",  # Would need file content to compute
            )

        challenges.append(Challenge(
            challenge_id=challenge_id,
            challenge_hash=challenge_hash,
            target_type=c.get("target_type", "unknown"),
            target_id=target_id,
            target_fingerprint=target_fingerprint,
            file_path=c.get("file_path", ""),
            evidence_span=evidence_span,
            severity=c.get("severity", "medium"),
            category=c.get("category", "logic"),
            description=c.get("description", ""),
            counterexample=c.get("counterexample"),
            suggested_test=c.get("suggested_test"),
            derived_claim_type=c.get("checkable_claim", "").split(":")[0] if c.get("checkable_claim") else None,
            derived_claim_description=c.get("checkable_claim"),
        ))

    summary = parsed.get("summary", "")
    confidence = float(parsed.get("confidence", 0.5))
    threat_model_notes = parsed.get("threat_model_notes", "")

    # Log rejected challenges for debugging
    if rejected_challenges:
        summary += f"\n\n[REJECTED {len(rejected_challenges)} challenges with unknown target_ids]"

    return challenges, summary, confidence, threat_model_notes


def run_critic_review(
    patches: list[dict],
    no_change_proofs: list[dict],
    file_contents: dict[str, str],
    findings: list[dict],
    *,
    provider: str = "openai",
    model: str = "gpt-4o",  # Use stronger model for critic
    temperature: float = 0.3,  # Slightly creative for finding issues
    max_tokens: int = 4000,
    timeout: int = 120,
) -> CriticResult:
    """
    Run the critic agent to challenge proposer's patches/proofs.

    Args:
        patches: Patches produced by proposer (must have item_id/patch_id)
        no_change_proofs: NO_CHANGE proofs from proposer (must have item_id/proof_id)
        file_contents: Dict of file_path -> content for context
        findings: Original findings that were analyzed
        provider: LLM provider ("openai", "anthropic")
        model: Model to use (default gpt-4o for better reasoning)
        temperature: Sampling temperature
        max_tokens: Max output tokens
        timeout: API timeout in seconds

    Returns:
        CriticResult with challenges, derived obligations, and summary
    """
    import time
    start_time = time.time()

    prompt = render_critic_prompt(patches, no_change_proofs, file_contents, findings)

    # Call LLM
    if provider.lower() == "openai":
        raw_output = _call_openai_critic(prompt, model, temperature, max_tokens, timeout)
    elif provider.lower() == "anthropic":
        raw_output = _call_anthropic_critic(prompt, model, temperature, max_tokens, timeout)
    else:
        raise ValueError(f"Unsupported critic provider: {provider}")

    # Parse output with target binding
    challenges, summary, confidence, threat_notes = _parse_critic_output(
        raw_output, patches, no_change_proofs
    )

    if threat_notes:
        summary = f"{summary}\n\nThreat model notes: {threat_notes}"

    duration_ms = int((time.time() - start_time) * 1000)

    result = CriticResult(
        challenges=challenges,
        summary=summary,
        confidence=confidence,
        raw_output=raw_output,
        duration_ms=duration_ms,
    )
    result.compute_stats()
    result.derive_obligations()

    return result


def save_critic_result(result: CriticResult, run_path: Path) -> Path:
    """Save critic result to run directory."""
    critic_path = run_path / "critic_review.json"
    with open(critic_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    return critic_path


def format_critic_challenges(result: CriticResult) -> str:
    """Format critic challenges for display."""
    lines = []

    if not result.challenges:
        lines.append("  No challenges raised - proposer's work looks solid")
        return "\n".join(lines)

    # Group by severity
    by_severity = {"critical": [], "high": [], "medium": [], "low": []}
    for c in result.challenges:
        by_severity.get(c.severity, by_severity["medium"]).append(c)

    severity_symbols = {
        "critical": "\033[91m✗\033[0m",  # Red
        "high": "\033[93m!\033[0m",      # Yellow
        "medium": "\033[96m?\033[0m",    # Cyan
        "low": "\033[2m·\033[0m",        # Dim
    }

    for severity in ["critical", "high", "medium", "low"]:
        challenges = by_severity[severity]
        if not challenges:
            continue

        for c in challenges:
            symbol = severity_symbols[severity]
            target_info = f"[{c.target_id}]" if c.target_id != "unknown" else ""
            lines.append(f"  {symbol} [{severity.upper()}] {target_info} {c.file_path}")
            lines.append(f"      {c.description[:80]}...")
            if c.counterexample:
                lines.append(f"      Counterexample: {c.counterexample[:60]}...")
            if c.derived_claim_type:
                lines.append(f"      → Check: {c.derived_claim_type}")

    lines.append("")
    lines.append(f"  Summary: {result.summary[:200]}...")
    lines.append(f"  Confidence: {result.confidence:.0%}")

    if result.derived_obligations:
        lines.append(f"\n  Derived obligations: {len(result.derived_obligations)} checkable claims")

    return "\n".join(lines)


def generate_critic_tests(result: CriticResult, output_dir: Path) -> list[Path]:
    """Generate pytest files from critic suggestions."""
    test_files = []

    tests_dir = output_dir / "critic_tests"
    tests_dir.mkdir(exist_ok=True)

    for idx, challenge in enumerate(result.challenges):
        if not challenge.suggested_test:
            continue

        test_code = challenge.suggested_test

        # Wrap in pytest if not already
        if "def test_" not in test_code:
            test_code = f'''"""Auto-generated test from critic challenge {challenge.challenge_id}"""
import pytest

def test_critic_challenge_{idx}():
    """
    Challenge: {challenge.description[:100]}
    Counterexample: {challenge.counterexample or 'N/A'}
    """
    {test_code}
'''

        test_file = tests_dir / f"test_critic_{idx:03d}.py"
        test_file.write_text(test_code)
        test_files.append(test_file)

    return test_files


# ─────────────────────────────────────────────────────────────────
# Receipt Binding - Make critic a first-class DAG vertex
# ─────────────────────────────────────────────────────────────────

@dataclass
class CriticReceipt:
    """Receipt for critic review - binds into the verification DAG."""
    receipt_id: str
    inputs_hash: str  # Hash of what critic saw (claims + patches + plan)
    output_hash: str  # Hash of critic_review.json
    challenges_count: int
    high_critical_count: int  # Blocking challenges
    duration_ms: int
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "receipt_id": self.receipt_id,
            "inputs_hash": self.inputs_hash,
            "output_hash": self.output_hash,
            "challenges_count": self.challenges_count,
            "high_critical_count": self.high_critical_count,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }


def build_critic_inputs_hash(
    plan_json: dict,
    claims_json: dict,
    patches_index: dict,
) -> str:
    """Build deterministic hash of critic inputs for receipt binding."""
    blob = {
        "plan_hash": plan_json.get("plan_hash") or plan_json.get("hash", ""),
        "claims_count": len(claims_json.get("claims", [])),
        "patches_count": len(patches_index.get("patches", [])),
        # Include fingerprints, not full content
        "claim_fingerprints": sorted([
            c.get("claim_id", "")[:16] for c in claims_json.get("claims", [])
        ]),
        "patch_fingerprints": sorted([
            p.get("patch_id", p.get("item_id", ""))[:16]
            for p in patches_index.get("patches", [])
        ]),
    }
    return sha256_bytes(json.dumps(blob, sort_keys=True).encode())[:32]


def generate_critic_receipt(
    result: CriticResult,
    inputs_hash: str,
    run_path: Path,
) -> CriticReceipt:
    """Generate a receipt binding the critic review into the DAG."""
    from datetime import datetime

    # Hash the critic output
    output_content = json.dumps(result.to_dict(), sort_keys=True)
    output_hash = sha256_bytes(output_content.encode())[:32]

    receipt_id = f"critic-{output_hash[:8]}"

    receipt = CriticReceipt(
        receipt_id=receipt_id,
        inputs_hash=inputs_hash,
        output_hash=output_hash,
        challenges_count=len(result.challenges),
        high_critical_count=result.critical_count + result.high_count,
        duration_ms=result.duration_ms,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )

    # Save receipt
    receipt_path = run_path / "critic_receipt.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt.to_dict(), f, indent=2)

    return receipt


# ─────────────────────────────────────────────────────────────────
# Mode C: Wire critic challenges into obligation system
# ─────────────────────────────────────────────────────────────────

@dataclass
class CriticObligation:
    """An obligation derived from a critic challenge."""
    obligation_id: str
    source_challenge_id: str
    claim_type: str  # Maps to ClaimType enum
    file_path: str
    description: str
    severity: str
    resolution: str  # "unresolved", "checker_passed", "checker_failed", "waived"
    resolver: Optional[str] = None  # Which checker resolved it, or "manual_waiver"
    waiver_justification: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "obligation_id": self.obligation_id,
            "source_challenge_id": self.source_challenge_id,
            "claim_type": self.claim_type,
            "file_path": self.file_path,
            "description": self.description,
            "severity": self.severity,
            "resolution": self.resolution,
            "resolver": self.resolver,
            "waiver_justification": self.waiver_justification,
        }


def challenges_to_obligations(challenges: list[Challenge]) -> list[CriticObligation]:
    """
    Convert critic challenges to policy obligations.

    Mode C: High/critical challenges MUST be either:
    1. Resolved by a deterministic checker passing
    2. Manually waived with justification
    """
    obligations = []

    # Map challenge categories/descriptions to claim types
    CLAIM_TYPE_MAPPING = {
        # Keywords in description -> claim type
        "allowlist": "ALLOWLIST_CONSTANT",
        "whitelist": "ALLOWLIST_CONSTANT",
        "immutable": "ALLOWLIST_CONSTANT",
        "mutable": "ALLOWLIST_CONSTANT",
        "predefined": "ALLOWLIST_CONSTANT",  # "predefined dictionary" patterns
        "dynamic import": "ALLOWLIST_ENFORCED",  # dynamic import concerns
        "_class_modules": "ALLOWLIST_CONSTANT",
        "_func_modules": "ALLOWLIST_CONSTANT",
        "sql": "NO_SQL_INJECTION",
        "injection": "NO_SQL_INJECTION",
        "shell": "NO_SHELL_INJECTION",
        "subprocess": "NO_SHELL_INJECTION",
        "command": "NO_SHELL_INJECTION",
        "secret": "NO_HARDCODED_SECRETS",
        "credential": "NO_HARDCODED_SECRETS",
        "password": "NO_HARDCODED_SECRETS",
        "xss": "NO_XSS",
        "path traversal": "NO_PATH_TRAVERSAL",
    }

    # Known standard claim types that we can match
    STANDARD_CLAIM_TYPES = {
        "ALLOWLIST_CONSTANT", "ALLOWLIST_ENFORCED",
        "NO_SQL_INJECTION", "NO_SHELL_INJECTION", "NO_HARDCODED_SECRETS",
        "NO_XSS", "NO_PATH_TRAVERSAL", "INPUT_VALIDATED",
        "_CLASS_MODULES", "_FUNC_MODULES", "_SUBMODULES",
        "PROVIDER_REGISTRY", "WHITELIST_ENFORCEMENT", "WHITELIST_DICTIONARY",
    }

    for challenge in challenges:
        # Only create obligations for high/critical challenges
        if challenge.severity not in ("high", "critical"):
            continue

        # Determine claim type from description keywords
        desc_lower = challenge.description.lower()
        claim_type = "CUSTOM"  # Default

        for keyword, ctype in CLAIM_TYPE_MAPPING.items():
            if keyword in desc_lower:
                claim_type = ctype
                break

        # Only use derived_claim_type if it's a known standard type
        # (LLMs often output free-form descriptions which aren't useful)
        if challenge.derived_claim_type:
            normalized = challenge.derived_claim_type.upper().replace(" ", "_")
            # Check if it's short enough to be a real type (not a sentence)
            if normalized in STANDARD_CLAIM_TYPES or len(normalized) < 30:
                claim_type = normalized

        obligation_id = f"OBL-{challenge.challenge_id}"

        obligations.append(CriticObligation(
            obligation_id=obligation_id,
            source_challenge_id=challenge.challenge_id,
            claim_type=claim_type,
            file_path=challenge.file_path,
            description=challenge.description,
            severity=challenge.severity,
            resolution="unresolved",
        ))

    return obligations


def resolve_obligations_with_claims(
    obligations: list[CriticObligation],
    claims: list[dict],
) -> list[CriticObligation]:
    """
    Attempt to resolve critic obligations using existing claims.

    If a claim of the required type PASSED for the same file, the obligation is resolved.

    Resolution strategy:
    1. Exact match: file_path + normalized_claim_type
    2. Equivalent match: file_path + any of the equivalent claim types
    3. Security claim match: file_path + already_mitigated (if security category)
    """
    # Normalize claim types to standard values
    CLAIM_TYPE_NORMALIZATION = {
        # Critic-derived types -> standard ClaimType values
        "_class_modules": ["allowlist_constant", "allowlist_enforced"],
        "_func_modules": ["allowlist_constant", "allowlist_enforced"],
        "_submodules": ["allowlist_constant", "allowlist_enforced"],
        "provider_registry": ["allowlist_constant", "allowlist_enforced"],
        "whitelist_enforcement": ["allowlist_enforced", "allowlist_constant"],
        "whitelist_dictionary": ["allowlist_constant", "allowlist_enforced"],
        "ui_module_whitelist": ["allowlist_constant", "allowlist_enforced"],
        "utils_import_whitelist": ["allowlist_constant", "allowlist_enforced"],
        "allowlist_constant": ["allowlist_constant", "allowlist_enforced"],
        "allowlist_enforced": ["allowlist_enforced", "allowlist_constant"],
        "no_sql_injection": ["no_sql_injection"],
        "no_shell_injection": ["no_shell_injection"],
        "no_hardcoded_secrets": ["no_hardcoded_secrets"],
        "no_xss": ["no_xss"],
        "no_path_traversal": ["no_path_traversal"],
    }

    # Security-related claim types that can be satisfied by already_mitigated
    SECURITY_CLAIM_TYPES = {
        "_class_modules", "whitelist_enforcement", "whitelist_dictionary",
        "ui_module_whitelist", "utils_import_whitelist",
        "allowlist_constant", "allowlist_enforced",
        "no_sql_injection", "no_shell_injection", "no_hardcoded_secrets",
        "no_xss", "no_path_traversal",
    }

    # Index claims by file_path for multiple lookups
    claims_by_file = {}
    for claim in claims:
        file_path = claim.get("scope", {}).get("file_path", "")
        claim_type = claim.get("claim_type", "").lower()
        verdict = claim.get("verdict", "unknown")

        if file_path not in claims_by_file:
            claims_by_file[file_path] = {}

        # Track best verdict per claim type (pass > fail > unknown)
        if claim_type not in claims_by_file[file_path] or verdict == "pass":
            claims_by_file[file_path][claim_type] = {
                "verdict": verdict,
                "claim_id": claim.get("claim_id", ""),
                "description": claim.get("description", ""),
            }

    resolved = []
    for obl in obligations:
        norm_type = obl.claim_type.lower().replace(" ", "_")
        file_claims = claims_by_file.get(obl.file_path, {})

        # Strategy 1: Check for equivalent claim types
        equivalent_types = CLAIM_TYPE_NORMALIZATION.get(norm_type, [norm_type])

        for eq_type in equivalent_types:
            if eq_type in file_claims:
                claim_info = file_claims[eq_type]
                if claim_info["verdict"] == "pass":
                    obl.resolution = "checker_passed"
                    obl.resolver = f"claim:{claim_info['claim_id']}"
                    break
                elif claim_info["verdict"] == "fail":
                    obl.resolution = "checker_failed"
                    obl.resolver = f"claim:{claim_info['claim_id']}"
                    break

        # Strategy 2: If security-related and unresolved, check already_mitigated
        # BUT: ALLOWLIST_CONSTANT obligations CANNOT be satisfied by already_mitigated
        # because the critic is specifically challenging whether the allowlist is enforced,
        # and "the LLM said it's fine" is not sufficient evidence.
        strict_checker_required = {
            "allowlist_constant", "allowlist_enforced",
            "_class_modules", "_func_modules", "_submodules",
            "provider_registry", "whitelist_enforcement", "whitelist_dictionary",
        }

        if obl.resolution == "unresolved" and norm_type in SECURITY_CLAIM_TYPES:
            # Don't allow already_mitigated to satisfy strict obligations
            if norm_type in strict_checker_required:
                # These obligations MUST be satisfied by actual checker pass
                # already_mitigated is not sufficient
                pass
            elif "already_mitigated" in file_claims:
                claim_info = file_claims["already_mitigated"]
                if claim_info["verdict"] == "pass":
                    # Check if the mitigation description relates to the obligation
                    desc_lower = claim_info["description"].lower()
                    if any(kw in desc_lower for kw in ["whitelist", "allowlist", "predefined", "limited"]):
                        obl.resolution = "checker_passed"
                        obl.resolver = f"claim:{claim_info['claim_id']} (already_mitigated)"

        resolved.append(obl)

    return resolved


def save_critic_obligations(
    obligations: list[CriticObligation],
    run_path: Path,
) -> Path:
    """Save critic obligations to run directory."""
    obls_path = run_path / "critic_obligations.json"
    with open(obls_path, "w") as f:
        json.dump({
            "obligations": [o.to_dict() for o in obligations],
            "total": len(obligations),
            "unresolved": sum(1 for o in obligations if o.resolution == "unresolved"),
            "resolved": sum(1 for o in obligations if o.resolution in ("checker_passed", "waived")),
            "failed": sum(1 for o in obligations if o.resolution == "checker_failed"),
        }, f, indent=2)
    return obls_path


# ─────────────────────────────────────────────────────────────────
# Test Templates - Make critic suggestions runnable
# ─────────────────────────────────────────────────────────────────

TEST_TEMPLATES = {
    "ALLOWLIST_CONSTANT": '''"""Test that allowlist mapping is immutable."""
import pytest

def test_{safe_name}_allowlist_immutable():
    """
    Challenge: {description}
    Source: {challenge_id}
    """
    import sys
    sys.path.insert(0, "{project_dir}")

    # Import the module
    from {module_path} import {mapping_name}

    # Attempt mutation - should fail or have no effect
    original_keys = set({mapping_name}.keys())

    try:
        {mapping_name}["__MALICIOUS__"] = "evil.module"
    except (TypeError, KeyError, AttributeError):
        pass  # Good - mutation blocked

    # Verify no mutation occurred
    assert "__MALICIOUS__" not in {mapping_name}, "Allowlist was mutated!"
    assert set({mapping_name}.keys()) == original_keys, "Allowlist keys changed!"
''',

    "NO_SHELL_INJECTION": '''"""Test that shell injection is not possible."""
import pytest
import subprocess

def test_{safe_name}_no_shell_injection():
    """
    Challenge: {description}
    Source: {challenge_id}
    """
    # This test verifies that user input cannot escape to shell
    malicious_input = "; rm -rf /"

    # The function under test should either:
    # 1. Use shell=False (safe)
    # 2. Properly sanitize input
    # 3. Reject dangerous characters

    # TODO: Import and call the function with malicious_input
    # Example:
    # from {module_path} import run_command
    # result = run_command(malicious_input)
    # assert "rm" not in result.args  # Verify command wasn't executed

    pass  # Replace with actual test
''',

    "NO_SQL_INJECTION": '''"""Test that SQL injection is not possible."""
import pytest

def test_{safe_name}_no_sql_injection():
    """
    Challenge: {description}
    Source: {challenge_id}
    """
    malicious_input = "'; DROP TABLE users; --"

    # The function under test should use parameterized queries
    # TODO: Import and call the function with malicious_input
    # Example:
    # from {module_path} import query_user
    # result = query_user(malicious_input)
    # Verify the query was parameterized, not string-concatenated

    pass  # Replace with actual test
''',
}


def generate_test_from_template(
    challenge: Challenge,
    project_dir: str,
) -> Optional[str]:
    """Generate a test file from a challenge using templates."""
    claim_type = challenge.derived_claim_type or "CUSTOM"
    claim_type = claim_type.upper().replace(" ", "_")

    template = TEST_TEMPLATES.get(claim_type)
    if not template:
        return None

    # Extract module path from file path
    file_path = challenge.file_path
    module_path = file_path.replace(project_dir, "").strip("/").replace("/", ".").replace(".py", "")

    # Safe name for function
    safe_name = challenge.challenge_id.replace("-", "_").lower()

    # Try to extract mapping name from description
    mapping_name = "_CLASS_MODULES"  # Default
    desc_lower = challenge.description.lower()
    for candidate in ["_class_modules", "_submodules", "_func_modules", "provider_registry", "allowlist", "whitelist"]:
        if candidate in desc_lower:
            mapping_name = candidate.upper()
            break

    return template.format(
        safe_name=safe_name,
        description=challenge.description[:200],
        challenge_id=challenge.challenge_id,
        project_dir=project_dir,
        module_path=module_path,
        mapping_name=mapping_name,
    )


def generate_all_critic_tests(
    result: CriticResult,
    project_dir: str,
    output_dir: Path,
) -> list[Path]:
    """Generate all test files from critic challenges."""
    test_files = []

    tests_dir = output_dir / "critic_tests"
    tests_dir.mkdir(exist_ok=True)

    for idx, challenge in enumerate(result.challenges):
        # Try template first
        test_code = generate_test_from_template(challenge, project_dir)

        # Fall back to suggested_test if no template
        if not test_code and challenge.suggested_test:
            test_code = challenge.suggested_test
            if "def test_" not in test_code:
                test_code = f'''"""Auto-generated test from critic challenge {challenge.challenge_id}"""
import pytest

def test_critic_challenge_{idx}():
    """
    Challenge: {challenge.description[:100]}
    Counterexample: {challenge.counterexample or 'N/A'}
    """
    {test_code}
'''

        if test_code:
            test_file = tests_dir / f"test_critic_{idx:03d}_{challenge.challenge_id[:8]}.py"
            test_file.write_text(test_code)
            test_files.append(test_file)

    return test_files
