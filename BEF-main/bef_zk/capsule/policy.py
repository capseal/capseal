"""
Policy DSL - Proof Obligations

This module defines the policy language for specifying what must be proven
for a change to be accepted.

Example policy.yaml:

```yaml
version: "1.0"
name: "security_strict"

obligations:
  - id: no_sql_injection
    claim_type: no_sql_injection
    description: "No SQL injection vulnerabilities"
    file_pattern: "**/*.py"
    required_verdict: pass
    checkers: ["semgrep", "ast"]

  - id: allowlist_enforced
    claim_type: allowlist_enforced
    description: "Dynamic imports must use allowlist"
    file_pattern: "**/services/*.py"
    required_verdict: pass
    waivable: true
    waiver_requires: ["justification", "reviewer_approval"]

profiles:
  security:
    obligations: [no_sql_injection, allowlist_enforced, no_shell_injection]
  refactor:
    obligations: [refactor_equivalence]
  perf:
    obligations: [no_unnecessary_io]
```
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .claims import (
    Claim, ClaimType, Verdict, ProofObligation, ClaimBundle,
    CHECKER_REGISTRY,
)


def glob_match(pattern: str, path: str) -> bool:
    """
    Match a glob pattern against a path, supporting ** for recursive matching.

    Handles ** as "zero or more directories" (more intuitive than PurePath.match).

    Examples:
        src/**/*.py matches src/main.py (0 dirs) and src/foo/main.py (1 dir)
        **/*.py matches any .py file at any depth
    """
    import re

    # Normalize path separators
    path = path.replace("\\", "/")
    pattern = pattern.replace("\\", "/")

    # Convert glob pattern to regex
    # ** = any path (including empty)
    # * = any single path component (no slashes)
    # ? = any single character

    regex_parts = []
    i = 0
    while i < len(pattern):
        if pattern[i:i+2] == "**":
            # ** matches zero or more directories
            regex_parts.append(".*")
            i += 2
            # Skip following / if present
            if i < len(pattern) and pattern[i] == "/":
                regex_parts.append("/?")
                i += 1
        elif pattern[i] == "*":
            # * matches anything except /
            regex_parts.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            # ? matches any single character except /
            regex_parts.append("[^/]")
            i += 1
        elif pattern[i] in ".^$+{}[]|()":
            # Escape regex special characters
            regex_parts.append("\\" + pattern[i])
            i += 1
        else:
            regex_parts.append(pattern[i])
            i += 1

    regex = "^" + "".join(regex_parts) + "$"
    return bool(re.match(regex, path))


# ─────────────────────────────────────────────────────────────────
# Canonical JSON for stable hashing
# ─────────────────────────────────────────────────────────────────

def canonical_json(obj: Any) -> str:
    """
    Produce canonical JSON for stable hashing.

    - Keys sorted recursively
    - No whitespace
    - Unicode escaped consistently
    - Deterministic float representation

    This ensures the same policy produces the same hash regardless of
    YAML key order, whitespace, or formatting.
    """
    def normalize(x):
        if isinstance(x, dict):
            return {k: normalize(v) for k, v in sorted(x.items())}
        elif isinstance(x, list):
            return [normalize(v) for v in x]
        elif isinstance(x, float):
            # Normalize floats to avoid precision issues
            return float(f"{x:.10g}")
        else:
            return x

    normalized = normalize(obj)
    return json.dumps(normalized, separators=(',', ':'), ensure_ascii=True, sort_keys=True)


@dataclass
class PolicySource:
    """Tracks where a policy came from for auditability."""
    source_type: str  # "file", "builtin", "explicit"
    source_path: Optional[str] = None  # Path if loaded from file
    source_uri: Optional[str] = None   # URI like "policy_builtin://security_strict@1.0.0"
    content_hash: Optional[str] = None  # Hash of raw source content (before parsing)

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "source_path": self.source_path,
            "source_uri": self.source_uri,
            "content_hash": self.content_hash,
        }


@dataclass
class Waiver:
    """A waiver for a proof obligation that couldn't be met."""
    obligation_id: str
    reason: str
    justification: str
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    evidence_hash: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "obligation_id": self.obligation_id,
            "reason": self.reason,
            "justification": self.justification,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "evidence_hash": self.evidence_hash,
        }


@dataclass
class PolicyProfile:
    """A named collection of obligations."""
    name: str
    description: str = ""
    obligations: list[str] = field(default_factory=list)  # obligation_ids

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "obligations": self.obligations,
        }


@dataclass
class Policy:
    """
    A complete policy specification.

    Defines what must be proven for changes to be accepted.
    """
    version: str
    name: str
    description: str = ""

    # All available obligations
    obligations: list[ProofObligation] = field(default_factory=list)

    # Named profiles (subsets of obligations)
    profiles: dict[str, PolicyProfile] = field(default_factory=dict)

    # Global settings
    default_profile: str = "security"
    fail_on_unknown: bool = False  # Should UNKNOWN verdicts fail?
    require_all_checkers: bool = False  # Must all checkers pass, or just one?

    # Source tracking (set by loader)
    source: Optional[PolicySource] = None

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "obligations": [o.to_dict() for o in self.obligations],
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()},
            "default_profile": self.default_profile,
            "fail_on_unknown": self.fail_on_unknown,
            "require_all_checkers": self.require_all_checkers,
        }

    def to_dict_with_source(self) -> dict:
        """Full dict including source metadata."""
        d = self.to_dict()
        if self.source:
            d["_source"] = self.source.to_dict()
        d["_hash"] = self.hash()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        obligations = []
        for o in d.get("obligations", []):
            obligations.append(ProofObligation(
                obligation_id=o["id"],
                claim_type=ClaimType(o["claim_type"]),
                description=o.get("description", ""),
                file_pattern=o.get("file_pattern", "*"),
                function_pattern=o.get("function_pattern"),
                required_verdict=Verdict(o.get("required_verdict", "pass")),
                waivable=o.get("waivable", False),
                waiver_requires=o.get("waiver_requires", []),
                allowed_checkers=o.get("checkers", []),
            ))

        profiles = {}
        for name, p in d.get("profiles", {}).items():
            profiles[name] = PolicyProfile(
                name=name,
                description=p.get("description", ""),
                obligations=p.get("obligations", []),
            )

        return cls(
            version=d.get("version", "1.0"),
            name=d.get("name", "default"),
            description=d.get("description", ""),
            obligations=obligations,
            profiles=profiles,
            default_profile=d.get("default_profile", "security"),
            fail_on_unknown=d.get("fail_on_unknown", False),
            require_all_checkers=d.get("require_all_checkers", False),
        )

    @classmethod
    def from_yaml(cls, yaml_content: str, source: PolicySource | None = None) -> "Policy":
        d = yaml.safe_load(yaml_content)
        policy = cls.from_dict(d)
        policy.source = source
        return policy

    @classmethod
    def from_file(cls, path: Path) -> "Policy":
        """Load policy from file with source tracking."""
        path = Path(path).resolve()
        content = path.read_text()
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        source = PolicySource(
            source_type="file",
            source_path=str(path),
            content_hash=content_hash,
        )

        if path.suffix in (".yaml", ".yml"):
            return cls.from_yaml(content, source=source)
        else:
            policy = cls.from_dict(json.loads(content))
            policy.source = source
            return policy

    def get_profile(self, name: str) -> PolicyProfile | None:
        return self.profiles.get(name)

    def get_obligations_for_profile(self, profile_name: str) -> list[ProofObligation]:
        """Get all obligations for a named profile."""
        profile = self.get_profile(profile_name)
        if not profile:
            return []

        return [
            o for o in self.obligations
            if o.obligation_id in profile.obligations
        ]

    def get_obligations_for_file(self, file_path: str, profile_name: str | None = None) -> list[ProofObligation]:
        """Get obligations that apply to a specific file."""
        if profile_name:
            obligations = self.get_obligations_for_profile(profile_name)
        else:
            obligations = self.obligations

        matching = []
        for o in obligations:
            if glob_match(o.file_pattern, file_path):
                matching.append(o)

        return matching

    def hash(self) -> str:
        """
        Get a canonical hash of this policy for reproducibility.

        Uses canonical JSON serialization to ensure:
        - Same policy with reordered keys = same hash
        - Whitespace changes = same hash
        - Consistent across platforms
        """
        return hashlib.sha256(canonical_json(self.to_dict()).encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────
# Default Policies
# ─────────────────────────────────────────────────────────────────

DEFAULT_SECURITY_POLICY = Policy(
    version="1.0",
    name="security_strict",
    description="Strict security policy - all security claims must pass",
    obligations=[
        ProofObligation(
            obligation_id="no_sql_injection",
            claim_type=ClaimType.NO_SQL_INJECTION,
            description="No SQL injection vulnerabilities",
            file_pattern="**/*.py",
            required_verdict=Verdict.PASS,
            allowed_checkers=["semgrep", "ast"],
        ),
        ProofObligation(
            obligation_id="no_shell_injection",
            claim_type=ClaimType.NO_SHELL_INJECTION,
            description="No shell injection (shell=True)",
            file_pattern="**/*.py",
            required_verdict=Verdict.PASS,
            allowed_checkers=["semgrep", "ast"],
        ),
        ProofObligation(
            obligation_id="allowlist_enforced",
            claim_type=ClaimType.ALLOWLIST_ENFORCED,
            description="Dynamic imports/evals must use allowlist",
            file_pattern="**/*.py",
            required_verdict=Verdict.PASS,
            waivable=True,
            waiver_requires=["justification"],
            allowed_checkers=["ast"],
        ),
        ProofObligation(
            obligation_id="no_hardcoded_secrets",
            claim_type=ClaimType.NO_HARDCODED_SECRETS,
            description="No hardcoded secrets or API keys",
            file_pattern="**/*",
            required_verdict=Verdict.PASS,
            allowed_checkers=["semgrep"],
        ),
    ],
    profiles={
        "security": PolicyProfile(
            name="security",
            description="Full security checks",
            obligations=["no_sql_injection", "no_shell_injection", "allowlist_enforced", "no_hardcoded_secrets"],
        ),
        "minimal": PolicyProfile(
            name="minimal",
            description="Just injection checks",
            obligations=["no_sql_injection", "no_shell_injection"],
        ),
    },
    default_profile="security",
)

DEFAULT_REFACTOR_POLICY = Policy(
    version="1.0",
    name="refactor_safe",
    description="Safe refactoring - behavior preservation required",
    obligations=[
        ProofObligation(
            obligation_id="refactor_equivalence",
            claim_type=ClaimType.REFACTOR_EQUIVALENCE,
            description="Refactored code must be behaviorally equivalent",
            file_pattern="**/*.py",
            required_verdict=Verdict.PASS,
            allowed_checkers=["hypothesis"],
        ),
        ProofObligation(
            obligation_id="type_safe",
            claim_type=ClaimType.TYPE_SAFE,
            description="Type annotations must be valid",
            file_pattern="**/*.py",
            required_verdict=Verdict.PASS,
            waivable=True,
            allowed_checkers=["mypy"],
        ),
    ],
    profiles={
        "refactor": PolicyProfile(
            name="refactor",
            description="Behavioral equivalence checks",
            obligations=["refactor_equivalence"],
        ),
        "strict": PolicyProfile(
            name="strict",
            description="Full refactor checks including types",
            obligations=["refactor_equivalence", "type_safe"],
        ),
    },
    default_profile="refactor",
    source=PolicySource(
        source_type="builtin",
        source_uri="policy_builtin://refactor_safe@1.0.0",
    ),
)

# Add source tracking to security policy
DEFAULT_SECURITY_POLICY.source = PolicySource(
    source_type="builtin",
    source_uri="policy_builtin://security_strict@1.0.0",
)

# Registry of built-in policies
BUILTIN_POLICIES = {
    "security_strict": DEFAULT_SECURITY_POLICY,
    "refactor_safe": DEFAULT_REFACTOR_POLICY,
}


# ─────────────────────────────────────────────────────────────────
# Policy Loader
# ─────────────────────────────────────────────────────────────────

@dataclass
class PolicyLoadResult:
    """Result of loading a policy with metadata."""
    policy: Policy
    resolution_path: list[str]  # What was checked in order
    resolved_from: str  # Where it was found

    def to_dict(self) -> dict:
        return {
            "policy_name": self.policy.name,
            "policy_hash": self.policy.hash(),
            "resolution_path": self.resolution_path,
            "resolved_from": self.resolved_from,
            "source": self.policy.source.to_dict() if self.policy.source else None,
        }


def load_policy(
    explicit_path: str | Path | None = None,
    project_root: str | Path | None = None,
    builtin_name: str | None = None,
) -> PolicyLoadResult:
    """
    Load policy with precedence order:

    1. explicit_path (--policy flag) - wins if provided
    2. project_root/.capseal/policy.yaml - if exists
    3. builtin_name or "security_strict" - fallback

    Returns PolicyLoadResult with resolution metadata.
    """
    resolution_path = []

    # 1. Explicit path
    if explicit_path:
        path = Path(explicit_path).resolve()
        resolution_path.append(f"explicit:{path}")
        if path.exists():
            policy = Policy.from_file(path)
            return PolicyLoadResult(
                policy=policy,
                resolution_path=resolution_path,
                resolved_from=f"explicit:{path}",
            )
        else:
            raise FileNotFoundError(f"Explicit policy not found: {path}")

    # 2. Project root .capseal/policy.yaml
    if project_root:
        project_path = Path(project_root).resolve()
        policy_candidates = [
            project_path / ".capseal" / "policy.yaml",
            project_path / ".capseal" / "policy.yml",
            project_path / ".capseal" / "policy.json",
        ]
        for candidate in policy_candidates:
            resolution_path.append(f"project:{candidate}")
            if candidate.exists():
                policy = Policy.from_file(candidate)
                return PolicyLoadResult(
                    policy=policy,
                    resolution_path=resolution_path,
                    resolved_from=f"project:{candidate}",
                )

    # 3. Built-in fallback
    name = builtin_name or "security_strict"
    resolution_path.append(f"builtin:{name}")

    if name in BUILTIN_POLICIES:
        policy = BUILTIN_POLICIES[name]
        return PolicyLoadResult(
            policy=policy,
            resolution_path=resolution_path,
            resolved_from=f"builtin:{name}",
        )
    else:
        available = list(BUILTIN_POLICIES.keys())
        raise ValueError(f"Unknown builtin policy: {name}. Available: {available}")


def resolve_policy_for_review(
    explicit_policy: str | Path | None = None,
    target_path: str | Path | None = None,
    profile_name: str | None = None,
) -> tuple[Policy, str, PolicyLoadResult]:
    """
    Convenience function for review command.

    Returns (policy, profile_name, load_result).
    """
    # Determine project root
    project_root = None
    if target_path:
        project_root = Path(target_path).resolve()
        if project_root.is_file():
            project_root = project_root.parent

    # Load policy
    result = load_policy(
        explicit_path=explicit_policy,
        project_root=project_root,
    )

    # Determine profile
    if not profile_name:
        profile_name = result.policy.default_profile

    return result.policy, profile_name, result


# ─────────────────────────────────────────────────────────────────
# Policy Evaluation
# ─────────────────────────────────────────────────────────────────

@dataclass
class ObligationResult:
    """Result of evaluating a single obligation."""
    obligation: ProofObligation
    claims: list[Claim]
    verdict: Verdict
    waiver: Optional[Waiver] = None

    @property
    def met(self) -> bool:
        return self.verdict == Verdict.PASS or self.waiver is not None

    def to_dict(self) -> dict:
        return {
            "obligation_id": self.obligation.obligation_id,
            "verdict": self.verdict.value,
            "claims": [c.to_dict() for c in self.claims],
            "met": self.met,
            "waiver": self.waiver.to_dict() if self.waiver else None,
        }


@dataclass
class PolicyEvaluation:
    """Complete evaluation of a policy against a set of claims."""
    policy: Policy
    profile_name: str
    results: list[ObligationResult] = field(default_factory=list)

    @property
    def all_met(self) -> bool:
        return all(r.met for r in self.results)

    @property
    def failed_obligations(self) -> list[ObligationResult]:
        return [r for r in self.results if not r.met]

    @property
    def waived_obligations(self) -> list[ObligationResult]:
        return [r for r in self.results if r.waiver is not None]

    def to_dict(self) -> dict:
        return {
            "policy": self.policy.name,
            "policy_hash": self.policy.hash()[:16],
            "profile": self.profile_name,
            "all_met": self.all_met,
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.verdict == Verdict.PASS),
                "failed": sum(1 for r in self.results if r.verdict == Verdict.FAIL),
                "waived": len(self.waived_obligations),
            },
        }


def evaluate_policy(
    policy: Policy,
    profile_name: str,
    claims: list[Claim],
    waivers: list[Waiver] | None = None,
) -> PolicyEvaluation:
    """
    Evaluate a policy against a set of claims.

    Returns a complete evaluation with results for each obligation.
    """
    waivers = waivers or []
    waiver_map = {w.obligation_id: w for w in waivers}

    obligations = policy.get_obligations_for_profile(profile_name)
    results = []

    for obligation in obligations:
        # Find claims that match this obligation
        matching_claims = [
            c for c in claims
            if c.claim_type == obligation.claim_type
            and fnmatch.fnmatch(c.scope.file_path, obligation.file_pattern)
        ]

        # Determine overall verdict
        if not matching_claims:
            # No claims = unknown (need to generate claims first)
            verdict = Verdict.UNKNOWN
        elif all(c.verdict == Verdict.PASS for c in matching_claims):
            verdict = Verdict.PASS
        elif any(c.verdict == Verdict.FAIL for c in matching_claims):
            verdict = Verdict.FAIL
        else:
            verdict = Verdict.UNKNOWN

        # Check for waiver
        waiver = waiver_map.get(obligation.obligation_id)

        results.append(ObligationResult(
            obligation=obligation,
            claims=matching_claims,
            verdict=verdict,
            waiver=waiver,
        ))

    return PolicyEvaluation(
        policy=policy,
        profile_name=profile_name,
        results=results,
    )


def get_required_claims(
    policy: Policy,
    profile_name: str,
    file_path: str,
) -> list[tuple[ClaimType, str]]:
    """
    Get the claims that need to be generated for a file.

    Returns list of (claim_type, checker_id) tuples.
    """
    obligations = policy.get_obligations_for_file(file_path, profile_name)
    required = []

    for o in obligations:
        checkers = o.allowed_checkers or CHECKER_REGISTRY.list_checkers()
        for checker_id in checkers:
            required.append((o.claim_type, checker_id))

    return required
