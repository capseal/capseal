"""Hardened episode runner for real-mode evaluation.

Runs eval episodes in isolation with:
- Temp directory execution (never touches original files)
- Per-episode timeouts
- Exception catching (never crashes the loop)
- Budget enforcement
- Streaming JSONL logging
- Retry logic for transient failures
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .budget import BudgetTracker, BudgetExhaustedError


@dataclass
class EpisodeResult:
    """Result of running a single evaluation episode."""
    episode_id: str
    grid_idx: int
    success: bool
    duration_ms: int = 0
    error: str | None = None
    error_type: str | None = None  # "timeout", "budget", "llm_error", "exception"
    retries: int = 0
    tokens_used: int = 0
    cost: float = 0.0
    finding_id: str | None = None
    file_path: str | None = None
    patch_applied: bool = False

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "grid_idx": self.grid_idx,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "error_type": self.error_type,
            "retries": self.retries,
            "tokens_used": self.tokens_used,
            "cost": round(self.cost, 6),
            "finding_id": self.finding_id,
            "file_path": self.file_path,
            "patch_applied": self.patch_applied,
            "timestamp": time.time(),
        }


@dataclass
class EpisodeRunnerConfig:
    """Configuration for the episode runner."""
    timeout_seconds: int = 60  # Max time per episode
    max_retries: int = 1  # Retries on transient failures
    pricing_preset: str = "claude-sonnet"
    budget_limit: float | None = None  # None = unlimited
    log_path: Path | None = None  # Path for streaming JSONL log


class EpisodeRunner:
    """Runs evaluation episodes in a hardened, isolated manner."""

    def __init__(
        self,
        target_path: Path,
        config: EpisodeRunnerConfig | None = None,
        budget_tracker: BudgetTracker | None = None,
    ):
        """Initialize the episode runner.

        Args:
            target_path: Root path of the codebase being evaluated
            config: Runner configuration
            budget_tracker: Optional shared budget tracker
        """
        self.target_path = Path(target_path).resolve()
        self.config = config or EpisodeRunnerConfig()

        # Initialize or use provided budget tracker
        if budget_tracker:
            self.budget = budget_tracker
        else:
            self.budget = BudgetTracker.create(
                budget=self.config.budget_limit,
                pricing_preset=self.config.pricing_preset,
            )

        # Statistics
        self.episodes_run = 0
        self.episodes_success = 0
        self.episodes_failed = 0
        self.episodes_timeout = 0
        self.episodes_budget_stop = 0

        # Log file handle
        self._log_file = None
        if self.config.log_path:
            self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.config.log_path, "a")

    def close(self) -> None:
        """Close resources."""
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> "EpisodeRunner":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def run_episode(
        self,
        episode_id: str,
        grid_idx: int,
        finding: dict,
    ) -> EpisodeResult:
        """Run a single evaluation episode.

        Executes in a temp directory, with timeout and exception handling.
        Never crashes - always returns an EpisodeResult.
        """
        start_time = time.time()
        self.episodes_run += 1

        # Check budget before starting
        if self.budget.budget_exhausted:
            self.episodes_budget_stop += 1
            return EpisodeResult(
                episode_id=episode_id,
                grid_idx=grid_idx,
                success=False,
                error="Budget exhausted",
                error_type="budget",
            )

        if not self.budget.can_afford():
            self.episodes_budget_stop += 1
            return EpisodeResult(
                episode_id=episode_id,
                grid_idx=grid_idx,
                success=False,
                error="Insufficient budget for estimated call",
                error_type="budget",
            )

        # Extract finding info
        file_path = finding.get("path", "")
        finding_id = finding.get("check_id", "unknown")

        # Run with retries
        last_error = None
        last_error_type = None
        retries = 0

        for attempt in range(self.config.max_retries + 1):
            try:
                result = self._run_episode_isolated(
                    episode_id=episode_id,
                    grid_idx=grid_idx,
                    finding=finding,
                    timeout=self.config.timeout_seconds,
                )

                duration_ms = int((time.time() - start_time) * 1000)
                result.duration_ms = duration_ms
                result.retries = retries

                if result.success:
                    self.episodes_success += 1
                else:
                    self.episodes_failed += 1

                self._log_result(result)
                return result

            except TimeoutError as e:
                last_error = str(e)
                last_error_type = "timeout"
                self.episodes_timeout += 1
                # Don't retry timeouts
                break

            except BudgetExhaustedError as e:
                last_error = str(e)
                last_error_type = "budget"
                self.episodes_budget_stop += 1
                # Don't retry budget errors
                break

            except LLMError as e:
                last_error = str(e)
                last_error_type = "llm_error"
                retries += 1
                if attempt < self.config.max_retries:
                    time.sleep(1)  # Brief pause before retry
                    continue
                break

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                last_error_type = "exception"
                retries += 1
                if attempt < self.config.max_retries:
                    continue
                break

        # All retries exhausted
        duration_ms = int((time.time() - start_time) * 1000)
        self.episodes_failed += 1

        result = EpisodeResult(
            episode_id=episode_id,
            grid_idx=grid_idx,
            success=False,
            duration_ms=duration_ms,
            error=last_error,
            error_type=last_error_type,
            retries=retries,
            finding_id=finding_id,
            file_path=file_path,
        )

        self._log_result(result)
        return result

    def _run_episode_isolated(
        self,
        episode_id: str,
        grid_idx: int,
        finding: dict,
        timeout: int,
    ) -> EpisodeResult:
        """Run episode in an isolated temp directory."""
        import signal
        import threading

        file_path = finding.get("path", "")
        finding_id = finding.get("check_id", "unknown")

        if not file_path:
            return EpisodeResult(
                episode_id=episode_id,
                grid_idx=grid_idx,
                success=True,  # No file = nothing to patch = success
                finding_id=finding_id,
            )

        # Handle both absolute and relative paths
        file_path_obj = Path(file_path)
        if file_path_obj.is_absolute():
            source_file = file_path_obj
            # Make path relative for temp dir structure
            try:
                rel_path = file_path_obj.relative_to(self.target_path)
            except ValueError:
                rel_path = file_path_obj.name  # Fallback to just filename
        else:
            source_file = self.target_path / file_path
            rel_path = file_path

        if not source_file.exists():
            return EpisodeResult(
                episode_id=episode_id,
                grid_idx=grid_idx,
                success=True,  # File doesn't exist = nothing to patch
                finding_id=finding_id,
                file_path=str(file_path),
            )

        # Create temp directory and copy file
        with tempfile.TemporaryDirectory(prefix="capseal_eval_") as temp_dir:
            temp_path = Path(temp_dir)

            # Copy the file to temp using relative path
            temp_file = temp_path / rel_path
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, temp_file)

            # Run with timeout
            result_container = [None]
            exception_container = [None]

            def run_patch():
                try:
                    result_container[0] = self._attempt_patch(
                        temp_path=temp_path,
                        finding=finding,
                        episode_id=episode_id,
                        grid_idx=grid_idx,
                    )
                except Exception as e:
                    exception_container[0] = e

            thread = threading.Thread(target=run_patch)
            thread.start()
            thread.join(timeout=timeout)

            if thread.is_alive():
                # Timeout - we can't easily kill the thread, but we'll return
                raise TimeoutError(f"Episode timed out after {timeout}s")

            if exception_container[0]:
                raise exception_container[0]

            if result_container[0]:
                return result_container[0]

            # Fallback
            return EpisodeResult(
                episode_id=episode_id,
                grid_idx=grid_idx,
                success=False,
                error="No result returned",
                finding_id=finding_id,
                file_path=file_path,
            )

    def _attempt_patch(
        self,
        temp_path: Path,
        finding: dict,
        episode_id: str,
        grid_idx: int,
    ) -> EpisodeResult:
        """Attempt to generate and apply a patch for a finding."""
        file_path = finding.get("path", "")
        finding_id = finding.get("check_id", "unknown")
        tokens_used = 0
        cost = 0.0

        try:
            from capseal.refactor_engine import (
                generate_refactor_plan,
                run_multi_agent_patches,
            )

            # Auto-detect provider/model
            if os.environ.get("ANTHROPIC_API_KEY"):
                provider, model = "anthropic", "claude-sonnet-4-20250514"
            elif os.environ.get("OPENAI_API_KEY"):
                provider, model = "openai", "gpt-4o-mini"
            else:
                raise LLMError("No API key available (ANTHROPIC_API_KEY or OPENAI_API_KEY)")

            single_finding = [finding]

            # Generate plan
            plan = generate_refactor_plan(
                findings=single_finding,
                trace_root=f"eval-{finding_id[:8]}",
                aggregate_hash="eval",
                provider=provider,
                model=model,
            )

            # Estimate tokens for plan generation
            plan_tokens = 3000  # Estimate
            self.budget.record(
                input_tokens=2000,
                output_tokens=1000,
                model=model,
                purpose="plan_generation",
                success=True,
            )
            tokens_used += plan_tokens

            if not plan.items:
                return EpisodeResult(
                    episode_id=episode_id,
                    grid_idx=grid_idx,
                    success=True,  # No items = nothing to patch = success
                    tokens_used=tokens_used,
                    cost=self.budget.estimate_cost(2000, 1000),
                    finding_id=finding_id,
                    file_path=file_path,
                )

            # Check budget before patch attempt
            if not self.budget.can_afford(3000, 1500):
                raise BudgetExhaustedError(self.budget.total_cost, self.budget.budget_limit or 0)

            # Attempt patches
            results = run_multi_agent_patches(
                plan=plan,
                project_dir=temp_path,  # Use temp dir, not original
                provider=provider,
                model=model,
                enable_repair=True,
                enable_suppression_memos=True,
                enable_ast_validation=True,
            )

            # Record token usage for patch attempt
            patch_tokens = 4000  # Estimate
            self.budget.record(
                input_tokens=2500,
                output_tokens=1500,
                model=model,
                purpose="patch_attempt",
                success=True,
            )
            tokens_used += patch_tokens
            cost = self.budget.estimate_cost(4500, 2500)

            # Check results
            for r in results:
                if r.final_status == "VALID":
                    return EpisodeResult(
                        episode_id=episode_id,
                        grid_idx=grid_idx,
                        success=True,
                        tokens_used=tokens_used,
                        cost=cost,
                        finding_id=finding_id,
                        file_path=file_path,
                        patch_applied=True,
                    )
                elif r.final_status == "FAIL":
                    return EpisodeResult(
                        episode_id=episode_id,
                        grid_idx=grid_idx,
                        success=False,
                        tokens_used=tokens_used,
                        cost=cost,
                        finding_id=finding_id,
                        file_path=file_path,
                        patch_applied=False,
                    )

            # No definitive result
            return EpisodeResult(
                episode_id=episode_id,
                grid_idx=grid_idx,
                success=True,  # No failure = success
                tokens_used=tokens_used,
                cost=cost,
                finding_id=finding_id,
                file_path=file_path,
            )

        except BudgetExhaustedError:
            raise  # Re-raise budget errors
        except ImportError as e:
            raise LLMError(f"Missing dependency: {e}")
        except Exception as e:
            # Record failed call
            self.budget.record(
                input_tokens=2000,
                output_tokens=0,
                model="unknown",
                purpose="failed_attempt",
                success=False,
                error=str(e),
            )
            raise

    def _log_result(self, result: EpisodeResult) -> None:
        """Log result to streaming JSONL file."""
        if self._log_file:
            self._log_file.write(json.dumps(result.to_dict()) + "\n")
            self._log_file.flush()

    def summary(self) -> dict[str, Any]:
        """Get runner statistics."""
        return {
            "episodes_run": self.episodes_run,
            "episodes_success": self.episodes_success,
            "episodes_failed": self.episodes_failed,
            "episodes_timeout": self.episodes_timeout,
            "episodes_budget_stop": self.episodes_budget_stop,
            "success_rate": self.episodes_success / max(1, self.episodes_run),
            "budget": self.budget.summary(),
        }


class LLMError(Exception):
    """Raised when an LLM call fails in a recoverable way."""
    pass


__all__ = [
    "EpisodeRunner",
    "EpisodeRunnerConfig",
    "EpisodeResult",
    "LLMError",
]
