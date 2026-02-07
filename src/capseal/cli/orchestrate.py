"""Multi-agent orchestration with scoped context splitting.

Patterns:
1. Split context into coherent scopes
2. Spawn agents with scoped context (async)
3. Collect results
4. Synthesize with lead agent
5. Create deterministic receipt artifact
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field


# Output contract that all agents must follow
OUTPUT_CONTRACT = """\
You MUST output ONLY a JSON block in this exact format (no other text):

```json
{
  "scope": "your_scope_name",
  "summary": "one sentence assessment",
  "findings": [
    {
      "severity": "HIGH|MEDIUM|LOW",
      "issue": "description of the problem",
      "file": "path/to/file.py",
      "line": 123,
      "evidence": "the problematic code snippet",
      "recommendation": "how to fix it"
    }
  ],
  "patches": []
}
```

Rules:
- Output ONLY the JSON block, nothing else
- severity must be exactly HIGH, MEDIUM, or LOW
- If no findings, use empty array: "findings": []
- Do NOT use any tools - just analyze and respond
"""


@dataclass
class AgentScope:
    """Defines what an agent should focus on."""
    name: str
    description: str
    file_patterns: list[str] = field(default_factory=list)  # glob patterns
    concerns: list[str] = field(default_factory=list)  # what to look for
    context_slice: dict = field(default_factory=dict)  # subset of full context


# Predefined specialist scopes
SPECIALIST_SCOPES = {
    "security": AgentScope(
        name="security",
        description="Analyze for security vulnerabilities",
        concerns=[
            "injection vulnerabilities (SQL, command, XSS)",
            "authentication/authorization flaws",
            "secrets/credentials in code",
            "unsafe deserialization",
            "path traversal",
            "cryptographic weaknesses",
        ],
    ),
    "performance": AgentScope(
        name="performance",
        description="Analyze for performance issues",
        concerns=[
            "O(n²) or worse algorithms",
            "unnecessary allocations",
            "missing caching opportunities",
            "N+1 query patterns",
            "blocking I/O in async context",
            "memory leaks",
        ],
    ),
    "correctness": AgentScope(
        name="correctness",
        description="Analyze for bugs and logic errors",
        concerns=[
            "off-by-one errors",
            "null/undefined handling",
            "race conditions",
            "error handling gaps",
            "type mismatches",
            "edge cases",
        ],
    ),
    "style": AgentScope(
        name="style",
        description="Analyze code style and maintainability",
        concerns=[
            "naming conventions",
            "code duplication",
            "function complexity",
            "missing documentation",
            "inconsistent patterns",
        ],
    ),
}

# Module-based scopes (auto-generated from directory structure)
def generate_module_scopes(files: list[str]) -> dict[str, AgentScope]:
    """Generate scopes based on top-level directories."""
    modules: dict[str, list[str]] = {}

    for f in files:
        parts = Path(f).parts
        if len(parts) > 1:
            module = parts[0]
        else:
            module = "_root"

        if module not in modules:
            modules[module] = []
        modules[module].append(f)

    scopes = {}
    for module, module_files in modules.items():
        scopes[module] = AgentScope(
            name=module,
            description=f"Analyze changes in {module}/",
            file_patterns=[f"{module}/**"] if module != "_root" else ["*"],
        )
        scopes[module].context_slice = {"files": module_files}

    return scopes


def split_context_by_module(context: dict) -> dict[str, dict]:
    """Split context checkpoint by module/directory."""
    files = [f.get("path", f) if isinstance(f, dict) else f for f in context.get("files", [])]
    diffs = context.get("diffs", [])

    # Group by top-level directory
    modules: dict[str, dict] = {}

    for diff_obj in diffs:
        if not isinstance(diff_obj, dict):
            continue
        filepath = diff_obj.get("path", "")
        parts = Path(filepath).parts
        module = parts[0] if len(parts) > 1 else "_root"

        if module not in modules:
            modules[module] = {
                "files": [],
                "diffs": [],
                "summary": {
                    "module": module,
                    "parent_checkpoint": context.get("checkpoint_id"),
                },
            }

        modules[module]["files"].append(filepath)
        modules[module]["diffs"].append(diff_obj)

    return modules


def split_context_by_concern(context: dict, concerns: list[str]) -> dict[str, dict]:
    """Split context by concern (security, perf, etc).

    Each concern gets the FULL context but with focused instructions.
    This is for parallel specialist review, not data partitioning.
    """
    base_context = {
        "files": context.get("files", []),
        "diffs": context.get("diffs", []),
        "summary": context.get("summary", {}),
    }

    splits = {}
    for concern in concerns:
        scope = SPECIALIST_SCOPES.get(concern)
        if scope:
            splits[concern] = {
                **base_context,
                "scope": {
                    "name": scope.name,
                    "description": scope.description,
                    "concerns": scope.concerns,
                },
            }

    return splits


def format_scoped_prompt(scope_name: str, scope_context: dict) -> str:
    """Format a prompt for a scoped agent with strict output contract."""
    scope_info = scope_context.get("scope", {})
    files = scope_context.get("files", [])
    diffs = scope_context.get("diffs", [])

    # Sort files for determinism (files may be dicts or strings)
    if files:
        if isinstance(files[0], dict):
            files = sorted([f.get("path", str(f)) for f in files])
        else:
            files = sorted(files)

    prompt = f"""You are a specialist code review agent.

SCOPE: {scope_name}
FOCUS: {scope_info.get('description', scope_name)}
CONCERNS: {', '.join(scope_info.get('concerns', [scope_name]))}

FILES ({len(files)}):
{chr(10).join(f'- {f}' for f in files[:20])}
{'...' if len(files) > 20 else ''}

CHANGES:
"""

    # Sort diffs by path for determinism
    sorted_diffs = sorted(diffs, key=lambda d: d.get("path", "") if isinstance(d, dict) else "")

    for diff_obj in sorted_diffs[:10]:
        if isinstance(diff_obj, dict):
            path = diff_obj.get("path", "unknown")
            patch = diff_obj.get("patch", "")[:1500]
            prompt += f"\n### {path}\n```diff\n{patch}\n```\n"

    prompt += f"""

{OUTPUT_CONTRACT}
"""

    return prompt


def parse_agent_output(raw_output: str, agent_id: str, scope: str) -> dict:
    """Parse agent output, enforcing the output contract.

    Returns a normalized result dict even if parsing fails.
    """
    # Try to extract JSON from output
    try:
        # Look for fenced JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_output, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(1))
            # Normalize the result
            return {
                "agent_id": agent_id,
                "scope": parsed.get("scope", scope),
                "summary": parsed.get("summary", ""),
                "findings": parsed.get("findings", []),
                "patches": parsed.get("patches", []),
                "parse_ok": True,
                "raw_length": len(raw_output),
            }

        # Try parsing entire output as JSON
        parsed = json.loads(raw_output.strip())
        return {
            "agent_id": agent_id,
            "scope": parsed.get("scope", scope),
            "summary": parsed.get("summary", ""),
            "findings": parsed.get("findings", []),
            "patches": parsed.get("patches", []),
            "parse_ok": True,
            "raw_length": len(raw_output),
        }
    except (json.JSONDecodeError, Exception):
        pass

    # Failed to parse - return failure object with raw output
    return {
        "agent_id": agent_id,
        "scope": scope,
        "summary": f"Agent output could not be parsed as JSON",
        "findings": [],
        "patches": [],
        "parse_ok": False,
        "raw_output": raw_output[:2000] if raw_output else "",
        "raw_length": len(raw_output) if raw_output else 0,
    }


def compute_context_hash(context: dict) -> str:
    """Compute deterministic hash of context for receipts."""
    # Normalize files (may be dicts or strings)
    files = context.get("files", [])
    if files and isinstance(files[0], dict):
        file_paths = sorted(f.get("path", str(f)) for f in files)
    else:
        file_paths = sorted(files) if files else []

    # Extract hashable content
    hashable = {
        "summary": context.get("summary", {}),
        "files": file_paths,
        "diffs_count": len(context.get("diffs", [])),
    }
    content = json.dumps(hashable, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class Orchestrator:
    """Coordinates multi-agent workflows with deterministic outputs."""

    def __init__(self, context: dict):
        self.context = context
        self.context_hash = compute_context_hash(context)
        self.results: dict[str, Any] = {}

    def plan_specialist_review(self) -> list[dict]:
        """Plan a specialist review (security, perf, correctness).

        Tasks are sorted by agent_id for determinism.
        """
        concerns = ["security", "performance", "correctness"]
        splits = split_context_by_concern(self.context, concerns)

        tasks = []
        for concern in sorted(concerns):  # Deterministic order
            scoped_ctx = splits.get(concern)
            if scoped_ctx:
                tasks.append({
                    "agent_id": f"specialist_{concern}",
                    "scope": concern,
                    "prompt": format_scoped_prompt(concern, scoped_ctx),
                    "context": scoped_ctx,
                    "timeout": 180,  # 3 min per specialist
                })

        return tasks

    def plan_module_review(self) -> list[dict]:
        """Plan a module-based review (one agent per directory).

        Tasks are sorted by module name for determinism.
        """
        splits = split_context_by_module(self.context)

        tasks = []
        for module in sorted(splits.keys()):  # Deterministic order
            scoped_ctx = splits[module]
            # Skip tiny modules
            if len(scoped_ctx.get("diffs", [])) < 1:
                continue

            tasks.append({
                "agent_id": f"module_{module}",
                "scope": module,
                "prompt": format_scoped_prompt(module, {
                    **scoped_ctx,
                    "scope": {
                        "name": module,
                        "description": f"Review changes in {module}/",
                        "concerns": ["correctness", "consistency with existing code"],
                    },
                }),
                "context": scoped_ctx,
                "timeout": 180,
            })

        return tasks

    def synthesize_results(self, results: list[dict]) -> dict:
        """Combine results from multiple agents into deterministic summary.

        Output is sorted for reproducibility.
        """
        all_findings = []
        summaries = []
        agent_outputs = []

        for r in results:
            if not isinstance(r, dict):
                continue

            agent_outputs.append({
                "agent_id": r.get("agent_id", "?"),
                "scope": r.get("scope", "?"),
                "parse_ok": r.get("parse_ok", False),
                "findings_count": len(r.get("findings", [])),
            })

            findings = r.get("findings", [])
            # Tag each finding with its source agent
            for f in findings:
                f["_source_agent"] = r.get("agent_id", "?")
            all_findings.extend(findings)

            if r.get("summary"):
                summaries.append(f"[{r.get('scope', '?')}] {r['summary']}")

        # Sort findings deterministically: severity -> file -> line
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        all_findings.sort(key=lambda x: (
            severity_order.get(x.get("severity", "LOW"), 3),
            x.get("file", ""),
            x.get("line", 0),
        ))

        # Compute findings hash for receipt
        findings_hash = hashlib.sha256(
            json.dumps(all_findings, sort_keys=True).encode()
        ).hexdigest()[:16]

        return {
            "version": "1.0",
            "type": "review_synthesis",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "context_hash": self.context_hash,
            "findings_hash": findings_hash,
            "agent_count": len(results),
            "agent_outputs": agent_outputs,
            "total_findings": len(all_findings),
            "high_severity": len([f for f in all_findings if f.get("severity") == "HIGH"]),
            "medium_severity": len([f for f in all_findings if f.get("severity") == "MEDIUM"]),
            "low_severity": len([f for f in all_findings if f.get("severity") == "LOW"]),
            "findings": all_findings,
            "agent_summaries": summaries,
            "verdict": self._generate_verdict(all_findings),
        }

    def _generate_verdict(self, findings: list) -> dict:
        """Generate structured verdict."""
        high = len([f for f in findings if f.get("severity") == "HIGH"])
        med = len([f for f in findings if f.get("severity") == "MEDIUM"])
        low = len([f for f in findings if f.get("severity") == "LOW"])

        if high > 0:
            status = "NEEDS_ATTENTION"
            message = f"{high} high severity issues require immediate attention"
        elif med > 3:
            status = "REVIEW_RECOMMENDED"
            message = f"{med} medium severity issues should be reviewed"
        elif med > 0:
            status = "MINOR_ISSUES"
            message = f"{med} medium, {low} low severity issues found"
        elif low > 0:
            status = "LOOKS_OK"
            message = f"Only {low} low severity issues"
        else:
            status = "CLEAN"
            message = "No issues found"

        return {
            "status": status,
            "message": message,
            "counts": {"high": high, "medium": med, "low": low},
        }
