"""Budget tracking for LLM API calls.

Tracks token usage and costs across LLM calls, with configurable pricing
for different models. Supports budget limits and cost estimation.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Default pricing per 1K tokens (as of early 2025)
PRICING_PRESETS = {
    "claude-sonnet": {"input": 0.003, "output": 0.015},
    "claude-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-opus": {"input": 0.015, "output": 0.075},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
}

DEFAULT_PRICING = PRICING_PRESETS["claude-sonnet"]


@dataclass
class LLMCall:
    """Record of a single LLM API call."""
    timestamp: float
    input_tokens: int
    output_tokens: int
    model: str = ""
    purpose: str = ""  # e.g., "plan_generation", "patch_attempt", "repair"
    success: bool = True
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def cost(self, pricing: dict[str, float] | None = None) -> float:
        """Calculate cost for this call."""
        p = pricing or DEFAULT_PRICING
        return (
            self.input_tokens * p["input"] / 1000 +
            self.output_tokens * p["output"] / 1000
        )

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "purpose": self.purpose,
            "success": self.success,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LLMCall":
        return cls(
            timestamp=data["timestamp"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            model=data.get("model", ""),
            purpose=data.get("purpose", ""),
            success=data.get("success", True),
            error=data.get("error"),
        )


@dataclass
class BudgetTracker:
    """Tracks LLM API costs against a budget limit."""

    budget_limit: float | None = None  # None = unlimited
    pricing: dict[str, float] = field(default_factory=lambda: DEFAULT_PRICING.copy())
    calls: list[LLMCall] = field(default_factory=list)

    # Statistics
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    call_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Persistent storage
    _storage_path: Path | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def create(
        cls,
        budget: float | None = None,
        pricing_preset: str = "claude-sonnet",
        storage_path: Path | None = None,
    ) -> "BudgetTracker":
        """Create a new budget tracker.

        Args:
            budget: Maximum spend in dollars (None = unlimited)
            pricing_preset: One of: claude-sonnet, claude-haiku, claude-opus,
                           gpt-4o, gpt-4o-mini, gpt-4-turbo
            storage_path: Path to persist budget state (optional)
        """
        pricing = PRICING_PRESETS.get(pricing_preset, DEFAULT_PRICING).copy()
        tracker = cls(budget_limit=budget, pricing=pricing)
        tracker._storage_path = storage_path

        # Load existing state if available
        if storage_path and storage_path.exists():
            tracker._load()

        return tracker

    @property
    def remaining_budget(self) -> float | None:
        """Remaining budget in dollars, or None if unlimited."""
        with self._lock:
            if self.budget_limit is None:
                return None
            return max(0.0, self.budget_limit - self.total_cost)

    @property
    def budget_exhausted(self) -> bool:
        """True if budget limit reached."""
        with self._lock:
            if self.budget_limit is None:
                return False
            return self.total_cost >= self.budget_limit

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a hypothetical call."""
        return (
            input_tokens * self.pricing["input"] / 1000 +
            output_tokens * self.pricing["output"] / 1000
        )

    def can_afford(self, estimated_input: int = 4000, estimated_output: int = 2000) -> bool:
        """Check if we can afford another call with estimated tokens."""
        with self._lock:
            if self.budget_limit is None:
                return True
            estimated = self.estimate_cost(estimated_input, estimated_output)
            return (self.total_cost + estimated) <= self.budget_limit

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "",
        purpose: str = "",
        success: bool = True,
        error: str | None = None,
    ) -> LLMCall:
        """Record an LLM API call. Thread-safe.

        Returns the recorded call.
        Raises BudgetExhaustedError if budget would be exceeded.
        """
        call = LLMCall(
            timestamp=time.time(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            purpose=purpose,
            success=success,
            error=error,
        )

        cost = call.cost(self.pricing)

        with self._lock:
            # Update statistics
            self.calls.append(call)
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost += cost
            self.call_count += 1
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1

            # Persist if storage configured
            if self._storage_path:
                self._save()

        return call

    def summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        return {
            "budget_limit": self.budget_limit,
            "total_cost": round(self.total_cost, 4),
            "remaining_budget": round(self.remaining_budget, 4) if self.remaining_budget is not None else None,
            "budget_exhausted": self.budget_exhausted,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_cost_per_call": round(self.total_cost / max(1, self.call_count), 4),
        }

    def format_summary(self) -> str:
        """Format summary as human-readable string."""
        s = self.summary()
        lines = [
            f"Cost: ${s['total_cost']:.2f}",
            f"Tokens: {s['total_tokens']:,} ({s['total_input_tokens']:,} in / {s['total_output_tokens']:,} out)",
            f"Calls: {s['call_count']} ({s['success_count']} success / {s['failure_count']} fail)",
        ]
        if s["budget_limit"]:
            lines.insert(1, f"Budget: ${s['remaining_budget']:.2f} remaining of ${s['budget_limit']:.2f}")
        return "\n".join(lines)

    def _save(self) -> None:
        """Save state to storage path."""
        if not self._storage_path:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "budget_limit": self.budget_limit,
            "pricing": self.pricing,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
            "call_count": self.call_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "calls": [c.to_dict() for c in self.calls[-100:]],  # Keep last 100 calls
        }
        self._storage_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        """Load state from storage path."""
        if not self._storage_path or not self._storage_path.exists():
            return

        try:
            data = json.loads(self._storage_path.read_text())
            self.total_input_tokens = data.get("total_input_tokens", 0)
            self.total_output_tokens = data.get("total_output_tokens", 0)
            self.total_cost = data.get("total_cost", 0.0)
            self.call_count = data.get("call_count", 0)
            self.success_count = data.get("success_count", 0)
            self.failure_count = data.get("failure_count", 0)
            self.calls = [LLMCall.from_dict(c) for c in data.get("calls", [])]
        except (json.JSONDecodeError, KeyError):
            pass  # Start fresh on corrupt data


class BudgetExhaustedError(Exception):
    """Raised when budget limit is reached."""
    def __init__(self, spent: float, limit: float):
        self.spent = spent
        self.limit = limit
        super().__init__(f"Budget exhausted: ${spent:.2f} spent of ${limit:.2f} limit")


def estimate_episode_cost(
    pricing_preset: str = "claude-sonnet",
    avg_input_tokens: int = 3000,
    avg_output_tokens: int = 1500,
) -> float:
    """Estimate cost for a single eval episode.

    Default estimates based on typical plan + patch cycle.
    """
    pricing = PRICING_PRESETS.get(pricing_preset, DEFAULT_PRICING)
    return (
        avg_input_tokens * pricing["input"] / 1000 +
        avg_output_tokens * pricing["output"] / 1000
    )


def estimate_learning_cost(
    rounds: int = 5,
    targets_per_round: int = 16,
    episodes_per_target: int = 1,
    pricing_preset: str = "claude-sonnet",
) -> float:
    """Estimate total cost for a learning run."""
    episode_cost = estimate_episode_cost(pricing_preset)
    total_episodes = rounds * targets_per_round * episodes_per_target
    return total_episodes * episode_cost


__all__ = [
    "BudgetTracker",
    "BudgetExhaustedError",
    "LLMCall",
    "PRICING_PRESETS",
    "estimate_episode_cost",
    "estimate_learning_cost",
]
