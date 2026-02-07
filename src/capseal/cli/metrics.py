"""Hard target metrics - enforces strict resource limits for agents.

This module provides hard enforcement of token/file/snippet limits to prevent
context explosion and ensure predictable agent behavior.

Hard Limits (not negotiable):
- max_files: 25 files in context
- max_snippets: 80 code snippets
- max_total_chars: 120,000 characters (~30-40k tokens)
- max_query_chars: 15,000 characters per query
- max_response_wait: 75 seconds for external APIs

These limits are derived from production experience with Greptile and Cline
where unbounded context leads to:
1. Token exhaustion (context window overflow)
2. Degraded response quality (model distraction)
3. Cost explosion (unnecessary API calls)
4. Latency spikes (processing large contexts)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# Environment-overridable limits
MAX_FILES = int(os.environ.get("CAPSEAL_MAX_FILES", "25"))
MAX_SNIPPETS = int(os.environ.get("CAPSEAL_MAX_SNIPPETS", "80"))
MAX_TOTAL_CHARS = int(os.environ.get("CAPSEAL_MAX_CHARS", "120000"))
MAX_QUERY_CHARS = int(os.environ.get("CAPSEAL_MAX_QUERY_CHARS", "15000"))
MAX_RESPONSE_WAIT_SEC = int(os.environ.get("CAPSEAL_MAX_WAIT_SEC", "75"))

# Token estimation (rough: 1 token ≈ 4 chars for code)
CHARS_PER_TOKEN = 4


@dataclass
class ResourceUsage:
    """Current resource usage tracking."""
    files: int = 0
    snippets: int = 0
    total_chars: int = 0
    query_chars: int = 0
    api_calls: int = 0
    elapsed_sec: float = 0.0

    @property
    def estimated_tokens(self) -> int:
        """Estimate token count from characters."""
        return self.total_chars // CHARS_PER_TOKEN

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "snippets": self.snippets,
            "total_chars": self.total_chars,
            "estimated_tokens": self.estimated_tokens,
            "query_chars": self.query_chars,
            "api_calls": self.api_calls,
            "elapsed_sec": round(self.elapsed_sec, 2),
        }


@dataclass
class Limits:
    """Resource limits configuration."""
    max_files: int = MAX_FILES
    max_snippets: int = MAX_SNIPPETS
    max_total_chars: int = MAX_TOTAL_CHARS
    max_query_chars: int = MAX_QUERY_CHARS
    max_response_wait_sec: int = MAX_RESPONSE_WAIT_SEC

    def to_dict(self) -> dict[str, int]:
        return {
            "max_files": self.max_files,
            "max_snippets": self.max_snippets,
            "max_total_chars": self.max_total_chars,
            "max_query_chars": self.max_query_chars,
            "max_response_wait_sec": self.max_response_wait_sec,
        }


@dataclass
class LimitViolation:
    """Record of a limit violation."""
    resource: str  # files, snippets, chars, etc.
    current: int
    limit: int
    action: str  # truncated, rejected, warned
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "resource": self.resource,
            "current": self.current,
            "limit": self.limit,
            "action": self.action,
            "timestamp": self.timestamp,
        }


class MetricsEnforcer:
    """Enforces hard resource limits with tracking and auditing.

    Usage:
        enforcer = MetricsEnforcer()

        # Check before adding files
        if enforcer.can_add_files(5):
            enforcer.add_files(5)

        # Check remaining budget
        print(f"Files remaining: {enforcer.files_remaining}")

        # Enforce limits (raises on violation)
        enforcer.enforce_files(30)  # Raises LimitExceeded
    """

    def __init__(self, limits: Limits | None = None):
        self.limits = limits or Limits()
        self.usage = ResourceUsage()
        self.violations: list[LimitViolation] = []
        self._start_time = time.time()

    # ─────────────────────────────────────────────────────────────────
    # Remaining budget queries
    # ─────────────────────────────────────────────────────────────────

    @property
    def files_remaining(self) -> int:
        """Number of files that can still be added."""
        return max(0, self.limits.max_files - self.usage.files)

    @property
    def snippets_remaining(self) -> int:
        """Number of snippets that can still be added."""
        return max(0, self.limits.max_snippets - self.usage.snippets)

    @property
    def chars_remaining(self) -> int:
        """Characters that can still be added."""
        return max(0, self.limits.max_total_chars - self.usage.total_chars)

    @property
    def time_remaining(self) -> float:
        """Seconds remaining before timeout."""
        elapsed = time.time() - self._start_time
        return max(0.0, self.limits.max_response_wait_sec - elapsed)

    # ─────────────────────────────────────────────────────────────────
    # Budget checks (non-throwing)
    # ─────────────────────────────────────────────────────────────────

    def can_add_files(self, count: int = 1) -> bool:
        """Check if files can be added without exceeding limit."""
        return self.usage.files + count <= self.limits.max_files

    def can_add_snippets(self, count: int = 1) -> bool:
        """Check if snippets can be added without exceeding limit."""
        return self.usage.snippets + count <= self.limits.max_snippets

    def can_add_chars(self, count: int) -> bool:
        """Check if characters can be added without exceeding limit."""
        return self.usage.total_chars + count <= self.limits.max_total_chars

    def has_time(self) -> bool:
        """Check if there's time remaining."""
        return self.time_remaining > 0

    # ─────────────────────────────────────────────────────────────────
    # Budget consumption (tracking)
    # ─────────────────────────────────────────────────────────────────

    def add_files(self, count: int = 1) -> None:
        """Record files added to context."""
        self.usage.files += count

    def add_snippets(self, count: int = 1) -> None:
        """Record snippets added to context."""
        self.usage.snippets += count

    def add_chars(self, count: int) -> None:
        """Record characters added to context."""
        self.usage.total_chars += count

    def add_query(self, query: str) -> str:
        """Record and potentially truncate a query.

        Returns the (possibly truncated) query.
        """
        self.usage.query_chars += len(query)
        self.usage.api_calls += 1

        if len(query) > self.limits.max_query_chars:
            self.violations.append(LimitViolation(
                resource="query_chars",
                current=len(query),
                limit=self.limits.max_query_chars,
                action="truncated",
            ))
            return query[:self.limits.max_query_chars]
        return query

    # ─────────────────────────────────────────────────────────────────
    # Enforcement (throwing)
    # ─────────────────────────────────────────────────────────────────

    def enforce_files(self, count: int = 1) -> None:
        """Enforce file limit, raising on violation."""
        if not self.can_add_files(count):
            self.violations.append(LimitViolation(
                resource="files",
                current=self.usage.files + count,
                limit=self.limits.max_files,
                action="rejected",
            ))
            raise LimitExceeded(
                f"File limit exceeded: {self.usage.files + count} > {self.limits.max_files}"
            )
        self.add_files(count)

    def enforce_snippets(self, count: int = 1) -> None:
        """Enforce snippet limit, raising on violation."""
        if not self.can_add_snippets(count):
            self.violations.append(LimitViolation(
                resource="snippets",
                current=self.usage.snippets + count,
                limit=self.limits.max_snippets,
                action="rejected",
            ))
            raise LimitExceeded(
                f"Snippet limit exceeded: {self.usage.snippets + count} > {self.limits.max_snippets}"
            )
        self.add_snippets(count)

    def enforce_chars(self, count: int) -> None:
        """Enforce character limit, raising on violation."""
        if not self.can_add_chars(count):
            self.violations.append(LimitViolation(
                resource="chars",
                current=self.usage.total_chars + count,
                limit=self.limits.max_total_chars,
                action="rejected",
            ))
            raise LimitExceeded(
                f"Character limit exceeded: {self.usage.total_chars + count} > {self.limits.max_total_chars}"
            )
        self.add_chars(count)

    def enforce_timeout(self) -> None:
        """Enforce timeout, raising on violation."""
        if not self.has_time():
            elapsed = time.time() - self._start_time
            self.violations.append(LimitViolation(
                resource="time",
                current=int(elapsed),
                limit=self.limits.max_response_wait_sec,
                action="rejected",
            ))
            raise TimeoutExceeded(
                f"Timeout exceeded: {elapsed:.1f}s > {self.limits.max_response_wait_sec}s"
            )

    # ─────────────────────────────────────────────────────────────────
    # Truncation helpers
    # ─────────────────────────────────────────────────────────────────

    def truncate_to_limit(self, items: list, limit_type: str = "files") -> list:
        """Truncate a list to fit within limits.

        Args:
            items: List of items to potentially truncate
            limit_type: One of "files", "snippets"

        Returns:
            Truncated list
        """
        if limit_type == "files":
            remaining = self.files_remaining
        elif limit_type == "snippets":
            remaining = self.snippets_remaining
        else:
            return items

        if len(items) > remaining:
            self.violations.append(LimitViolation(
                resource=limit_type,
                current=len(items),
                limit=remaining,
                action="truncated",
            ))
            return items[:remaining]
        return items

    def truncate_text(self, text: str, max_chars: int | None = None) -> str:
        """Truncate text to fit within character limits.

        Args:
            text: Text to potentially truncate
            max_chars: Override max chars (default: chars_remaining)

        Returns:
            Truncated text
        """
        limit = max_chars if max_chars is not None else self.chars_remaining

        if len(text) > limit:
            self.violations.append(LimitViolation(
                resource="chars",
                current=len(text),
                limit=limit,
                action="truncated",
            ))
            # Truncate with indicator
            return text[:limit - 20] + "\n... (truncated)"
        return text

    # ─────────────────────────────────────────────────────────────────
    # Reporting
    # ─────────────────────────────────────────────────────────────────

    def update_elapsed(self) -> None:
        """Update elapsed time."""
        self.usage.elapsed_sec = time.time() - self._start_time

    def summary(self) -> dict[str, Any]:
        """Get summary of usage and violations."""
        self.update_elapsed()
        return {
            "usage": self.usage.to_dict(),
            "limits": self.limits.to_dict(),
            "violations": [v.to_dict() for v in self.violations],
            "within_limits": len(self.violations) == 0,
            "budget_remaining": {
                "files": self.files_remaining,
                "snippets": self.snippets_remaining,
                "chars": self.chars_remaining,
                "time_sec": round(self.time_remaining, 1),
            },
        }

    def utilization(self) -> dict[str, float]:
        """Get utilization percentages for each resource."""
        return {
            "files": self.usage.files / self.limits.max_files * 100,
            "snippets": self.usage.snippets / self.limits.max_snippets * 100,
            "chars": self.usage.total_chars / self.limits.max_total_chars * 100,
        }


class LimitExceeded(Exception):
    """Raised when a hard limit is exceeded."""
    pass


class TimeoutExceeded(Exception):
    """Raised when timeout limit is exceeded."""
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────────────

def get_enforcer(
    max_files: int | None = None,
    max_snippets: int | None = None,
    max_chars: int | None = None,
    max_wait_sec: int | None = None,
) -> MetricsEnforcer:
    """Get a metrics enforcer with optional custom limits.

    Args:
        max_files: Override max files (default: 25)
        max_snippets: Override max snippets (default: 80)
        max_chars: Override max chars (default: 120000)
        max_wait_sec: Override max wait (default: 75)

    Returns:
        Configured MetricsEnforcer
    """
    limits = Limits(
        max_files=max_files or MAX_FILES,
        max_snippets=max_snippets or MAX_SNIPPETS,
        max_total_chars=max_chars or MAX_TOTAL_CHARS,
        max_response_wait_sec=max_wait_sec or MAX_RESPONSE_WAIT_SEC,
    )
    return MetricsEnforcer(limits)


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses rough heuristic: 1 token ≈ 4 characters for code.
    """
    return len(text) // CHARS_PER_TOKEN


def check_budget(
    files: int = 0,
    snippets: int = 0,
    chars: int = 0,
) -> dict[str, bool]:
    """Quick check if resources are within default limits.

    Returns dict with pass/fail for each resource type.
    """
    return {
        "files": files <= MAX_FILES,
        "snippets": snippets <= MAX_SNIPPETS,
        "chars": chars <= MAX_TOTAL_CHARS,
        "all_ok": files <= MAX_FILES and snippets <= MAX_SNIPPETS and chars <= MAX_TOTAL_CHARS,
    }
