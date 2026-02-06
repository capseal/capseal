"""
Claim Cache - Incremental Proof Reuse

Caches claim verdicts to avoid re-proving unchanged code.

A cached verdict is only reusable if:
1. policy_hash matches (same policy)
2. checker_id/version matches (same tool)
3. file_hash or scope_hash matches (code unchanged)
4. claim_type matches (same assertion)

Cache key:
    H(policy_hash || checker_id || checker_version || claim_type || scope_hash)
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .claims import Claim, Verdict, Witness, ClaimType, CheckerInfo


@dataclass
class CachedVerdict:
    """A cached verdict with all metadata needed for validation."""
    # Cache key components
    cache_key: str
    policy_hash: str
    checker_id: str
    checker_version: str
    claim_type: str
    scope_hash: str  # file_hash or region_hash

    # The cached result
    verdict: str  # "pass", "fail", "unknown"
    witness_hash: Optional[str] = None  # Hash of witness artifact

    # Metadata
    created_at: str = ""
    file_path: str = ""
    hits: int = 0  # Number of times reused

    def to_dict(self) -> dict:
        return {
            "cache_key": self.cache_key,
            "policy_hash": self.policy_hash,
            "checker_id": self.checker_id,
            "checker_version": self.checker_version,
            "claim_type": self.claim_type,
            "scope_hash": self.scope_hash,
            "verdict": self.verdict,
            "witness_hash": self.witness_hash,
            "created_at": self.created_at,
            "file_path": self.file_path,
            "hits": self.hits,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CachedVerdict":
        return cls(**d)


def compute_cache_key(
    policy_hash: str,
    checker_id: str,
    checker_version: str,
    claim_type: ClaimType | str,
    scope_hash: str,
) -> str:
    """
    Compute cache key for a claim verdict.

    The key uniquely identifies a verification result under specific conditions.
    If any component changes, the key changes, invalidating the cache.
    """
    claim_type_str = claim_type.value if isinstance(claim_type, ClaimType) else claim_type

    key_parts = [
        policy_hash[:16],  # First 16 chars of policy hash
        checker_id,
        checker_version,
        claim_type_str,
        scope_hash[:16],  # First 16 chars of scope hash
    ]
    combined = "||".join(key_parts)
    return hashlib.sha256(combined.encode()).hexdigest()[:24]


class ClaimCache:
    """
    In-memory and on-disk cache for claim verdicts.

    Storage: .capseal/claims_cache.json
    """

    def __init__(self, cache_path: Path | None = None):
        self.cache_path = cache_path
        self._cache: dict[str, CachedVerdict] = {}
        self._modified = False

        if cache_path and cache_path.exists():
            self._load()

    def _load(self) -> None:
        """Load cache from disk."""
        if not self.cache_path or not self.cache_path.exists():
            return

        try:
            data = json.loads(self.cache_path.read_text())
            for entry in data.get("entries", []):
                cached = CachedVerdict.from_dict(entry)
                self._cache[cached.cache_key] = cached
        except Exception:
            # Corrupted cache - start fresh
            self._cache = {}

    def save(self) -> None:
        """Save cache to disk."""
        if not self.cache_path or not self._modified:
            return

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": "1.0",
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "entries": [v.to_dict() for v in self._cache.values()],
            "stats": {
                "total_entries": len(self._cache),
                "total_hits": sum(v.hits for v in self._cache.values()),
            },
        }

        self.cache_path.write_text(json.dumps(data, indent=2))
        self._modified = False

    def lookup(
        self,
        policy_hash: str,
        checker_id: str,
        checker_version: str,
        claim_type: ClaimType | str,
        scope_hash: str,
    ) -> CachedVerdict | None:
        """
        Look up a cached verdict.

        Returns CachedVerdict if found and valid, None otherwise.
        """
        cache_key = compute_cache_key(
            policy_hash, checker_id, checker_version, claim_type, scope_hash
        )

        cached = self._cache.get(cache_key)
        if cached:
            # Validate all components match (defense in depth)
            if (
                cached.policy_hash[:16] == policy_hash[:16] and
                cached.checker_id == checker_id and
                cached.checker_version == checker_version and
                cached.scope_hash[:16] == scope_hash[:16]
            ):
                cached.hits += 1
                self._modified = True
                return cached

        return None

    def store(
        self,
        policy_hash: str,
        checker_id: str,
        checker_version: str,
        claim: Claim,
        verdict: Verdict,
        witness: Witness | None = None,
    ) -> str:
        """
        Store a verdict in the cache.

        Returns the cache key.
        """
        # Determine scope hash
        if claim.scope.region_hash:
            scope_hash = claim.scope.region_hash
        else:
            scope_hash = claim.scope.file_hash

        cache_key = compute_cache_key(
            policy_hash, checker_id, checker_version, claim.claim_type, scope_hash
        )

        cached = CachedVerdict(
            cache_key=cache_key,
            policy_hash=policy_hash,
            checker_id=checker_id,
            checker_version=checker_version,
            claim_type=claim.claim_type.value,
            scope_hash=scope_hash,
            verdict=verdict.value,
            witness_hash=witness.artifact_hash if witness else None,
            created_at=datetime.utcnow().isoformat() + "Z",
            file_path=claim.scope.file_path,
            hits=0,
        )

        self._cache[cache_key] = cached
        self._modified = True

        return cache_key

    def invalidate_file(self, file_path: str) -> int:
        """
        Invalidate all cached verdicts for a file.

        Returns number of entries removed.
        """
        to_remove = [
            key for key, v in self._cache.items()
            if v.file_path == file_path
        ]

        for key in to_remove:
            del self._cache[key]

        if to_remove:
            self._modified = True

        return len(to_remove)

    def clear(self) -> None:
        """Clear all cached verdicts."""
        self._cache = {}
        self._modified = True

    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            "total_entries": len(self._cache),
            "total_hits": sum(v.hits for v in self._cache.values()),
            "by_checker": self._stats_by_field("checker_id"),
            "by_verdict": self._stats_by_field("verdict"),
        }

    def _stats_by_field(self, field: str) -> dict[str, int]:
        """Count entries by a field."""
        counts: dict[str, int] = {}
        for v in self._cache.values():
            key = getattr(v, field, "unknown")
            counts[key] = counts.get(key, 0) + 1
        return counts


def get_cache_for_project(project_root: Path) -> ClaimCache:
    """Get or create a claim cache for a project."""
    cache_path = project_root / ".capseal" / "claims_cache.json"
    return ClaimCache(cache_path)


# ─────────────────────────────────────────────────────────────────
# Cache-aware checker execution
# ─────────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    """Result of checking a claim (possibly from cache)."""
    verdict: Verdict
    witness: Witness | None
    from_cache: bool
    cache_key: str | None = None
    checker_id: str = ""
    checker_version: str = ""

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "from_cache": self.from_cache,
            "cache_key": self.cache_key,
            "checker_id": self.checker_id,
            "checker_version": self.checker_version,
        }


def check_claim_with_cache(
    claim: Claim,
    file_content: str,
    checker_id: str,
    policy_hash: str,
    cache: ClaimCache,
    checker_registry,  # CHECKER_REGISTRY from claims.py
) -> CheckResult:
    """
    Check a claim, using cache if available.

    1. Compute scope hash from current file content
    2. Look up in cache
    3. If hit: return cached verdict
    4. If miss: run checker, store result, return
    """
    import hashlib as _hashlib

    # Get checker info
    checker_tuple = checker_registry.get(checker_id)
    if not checker_tuple:
        return CheckResult(
            verdict=Verdict.ERROR,
            witness=None,
            from_cache=False,
            checker_id=checker_id,
        )

    checker_fn, checker_info = checker_tuple

    # Compute current scope hash
    if claim.scope.start_line and claim.scope.end_line:
        lines = file_content.split('\n')
        region = '\n'.join(lines[claim.scope.start_line-1:claim.scope.end_line])
        current_scope_hash = _hashlib.sha256(region.encode()).hexdigest()
    else:
        current_scope_hash = _hashlib.sha256(file_content.encode()).hexdigest()

    # Try cache lookup
    cached = cache.lookup(
        policy_hash=policy_hash,
        checker_id=checker_id,
        checker_version=checker_info.checker_version,
        claim_type=claim.claim_type,
        scope_hash=current_scope_hash,
    )

    if cached:
        # Cache hit - reconstruct verdict
        verdict = Verdict(cached.verdict)
        witness = None  # Witness would need to be loaded from artifact store
        return CheckResult(
            verdict=verdict,
            witness=witness,
            from_cache=True,
            cache_key=cached.cache_key,
            checker_id=checker_id,
            checker_version=checker_info.checker_version,
        )

    # Cache miss - run checker
    verdict, witness = checker_fn(claim, file_content)

    # Store in cache
    cache_key = cache.store(
        policy_hash=policy_hash,
        checker_id=checker_id,
        checker_version=checker_info.checker_version,
        claim=claim,
        verdict=verdict,
        witness=witness,
    )

    return CheckResult(
        verdict=verdict,
        witness=witness,
        from_cache=False,
        cache_key=cache_key,
        checker_id=checker_id,
        checker_version=checker_info.checker_version,
    )
