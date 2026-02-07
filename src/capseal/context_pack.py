"""Context Pack: Committed artifacts for ORACLE/agent token governance.

A Context Pack is a deterministic, hash-addressed bundle of context provided to
an ORACLE node (LLM call, Greptile query, external API). It ensures:

1. **Reproducibility**: Same inputs → same context_root hash
2. **Auditability**: What the model "saw" is committed and verifiable
3. **Budget tracking**: Token counts are logged per oracle call

This is the key primitive for making token spend "checkable" rather than "vibes."
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ContextChunk:
    """A single piece of context (file slice, diff, prior receipt, etc.)."""

    source_type: str  # "file", "diff", "receipt", "user_input", "tool_output"
    source_id: str    # file path, receipt hash, etc.
    content: str
    byte_offset: int = 0
    byte_length: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "content_hash": hashlib.sha256(self.content.encode()).hexdigest()[:16],
            "byte_offset": self.byte_offset,
            "byte_length": self.byte_length or len(self.content.encode()),
            "metadata": self.metadata,
        }


@dataclass
class ContextPack:
    """A committed, hash-addressed bundle of context for an ORACLE call.

    The context_root is computed deterministically from the sorted chunks,
    ensuring that the same inputs always produce the same root hash.
    """

    pack_id: str
    chunks: list[ContextChunk] = field(default_factory=list)
    selection_algorithm: str = "manual"  # or "auto_relevance", "diff_focused", etc.
    selection_params: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    _context_root: Optional[str] = field(default=None, repr=False)

    @property
    def context_root(self) -> str:
        """Compute deterministic root hash from chunks."""
        if self._context_root is None:
            # Sort chunks for determinism
            chunk_dicts = sorted(
                [c.to_dict() for c in self.chunks],
                key=lambda x: (x["source_type"], x["source_id"], x["byte_offset"])
            )

            # Include selection metadata for full reproducibility
            manifest = {
                "pack_id": self.pack_id,
                "selection_algorithm": self.selection_algorithm,
                "selection_params": self.selection_params,
                "chunks": chunk_dicts,
            }

            canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            self._context_root = hashlib.sha256(canonical.encode()).hexdigest()

        return self._context_root

    @property
    def total_bytes(self) -> int:
        return sum(len(c.content.encode()) for c in self.chunks)

    @property
    def estimated_tokens(self) -> int:
        """Rough estimate: ~4 chars per token."""
        return self.total_bytes // 4

    def add_file(self, path: str | Path, content: str, offset: int = 0) -> None:
        """Add a file or file slice to the context pack."""
        self.chunks.append(ContextChunk(
            source_type="file",
            source_id=str(path),
            content=content,
            byte_offset=offset,
            byte_length=len(content.encode()),
        ))
        self._context_root = None  # Invalidate cache

    def add_diff(self, diff_id: str, content: str) -> None:
        """Add a diff/patch to the context pack."""
        self.chunks.append(ContextChunk(
            source_type="diff",
            source_id=diff_id,
            content=content,
        ))
        self._context_root = None

    def add_receipt(self, receipt_hash: str, summary: str) -> None:
        """Add a prior receipt reference to the context pack."""
        self.chunks.append(ContextChunk(
            source_type="receipt",
            source_id=receipt_hash,
            content=summary,
        ))
        self._context_root = None

    def add_tool_output(self, tool_name: str, output: str) -> None:
        """Add tool output (grep, glob, etc.) to the context pack."""
        self.chunks.append(ContextChunk(
            source_type="tool_output",
            source_id=tool_name,
            content=output,
        ))
        self._context_root = None

    def to_dict(self) -> dict:
        return {
            "pack_id": self.pack_id,
            "context_root": self.context_root,
            "selection_algorithm": self.selection_algorithm,
            "selection_params": self.selection_params,
            "chunk_count": len(self.chunks),
            "total_bytes": self.total_bytes,
            "estimated_tokens": self.estimated_tokens,
            "created_at": self.created_at,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    def save(self, output_dir: Path) -> Path:
        """Save context pack to disk for audit trail."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save manifest
        manifest_path = output_dir / f"context_pack_{self.pack_id}.json"
        manifest_path.write_text(json.dumps(self.to_dict(), indent=2))

        # Save raw content for full reproducibility
        content_path = output_dir / f"context_pack_{self.pack_id}.txt"
        content_path.write_text("\n---\n".join(c.content for c in self.chunks))

        return manifest_path


@dataclass
class OracleCall:
    """Record of a single ORACLE invocation with budget tracking."""

    oracle_id: str           # "greptile", "claude", "gpt4", etc.
    call_id: str             # Unique identifier for this call
    context_root: str        # Hash of ContextPack used

    # Token accounting
    tokens_in: int = 0       # Input/prompt tokens
    tokens_out: int = 0      # Output/completion tokens
    model: str = ""          # e.g., "claude-3-opus", "gpt-4-turbo"
    cost_usd: float = 0.0    # Estimated cost

    # Timing
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    latency_ms: Optional[float] = None

    # Result
    success: bool = True
    error: Optional[str] = None
    output_hash: Optional[str] = None  # Hash of output for verification

    def complete(self, tokens_out: int, output: str, cost_usd: float = 0.0) -> None:
        """Mark call as complete with output metrics."""
        self.completed_at = time.time()
        self.latency_ms = (self.completed_at - self.started_at) * 1000
        self.tokens_out = tokens_out
        self.cost_usd = cost_usd
        self.output_hash = hashlib.sha256(output.encode()).hexdigest()[:16]

    def fail(self, error: str) -> None:
        """Mark call as failed."""
        self.completed_at = time.time()
        self.latency_ms = (self.completed_at - self.started_at) * 1000
        self.success = False
        self.error = error

    def to_dict(self) -> dict:
        return {
            "oracle_id": self.oracle_id,
            "call_id": self.call_id,
            "context_root": self.context_root,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens_total": self.tokens_in + self.tokens_out,
            "model": self.model,
            "cost_usd": self.cost_usd,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "error": self.error,
            "output_hash": self.output_hash,
        }

    def to_event(self) -> dict:
        """Format as event log entry."""
        return {
            "type": "oracle_call",
            "oracle_id": self.oracle_id,
            "call_id": self.call_id,
            "context_root": self.context_root,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "model": self.model,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "success": self.success,
        }


@dataclass
class BudgetLedger:
    """Tracks cumulative token/cost budget across a run or circuit.

    Enforces budget constraints at GATE nodes.
    """

    run_id: str
    budget_tokens: int = 1_000_000      # Default 1M token budget
    budget_oracle_calls: int = 100      # Default 100 calls
    budget_usd: float = 50.0            # Default $50

    # Cumulative spend
    spent_tokens_in: int = 0
    spent_tokens_out: int = 0
    spent_oracle_calls: int = 0
    spent_usd: float = 0.0

    # History
    calls: list[OracleCall] = field(default_factory=list)

    @property
    def spent_tokens_total(self) -> int:
        return self.spent_tokens_in + self.spent_tokens_out

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.budget_tokens - self.spent_tokens_total)

    @property
    def remaining_calls(self) -> int:
        return max(0, self.budget_oracle_calls - self.spent_oracle_calls)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.budget_usd - self.spent_usd)

    def check_budget(self, estimated_tokens: int = 0) -> tuple[bool, str]:
        """Check if budget allows another oracle call.

        Returns (allowed, reason).
        """
        if self.spent_oracle_calls >= self.budget_oracle_calls:
            return False, f"Oracle call limit reached ({self.budget_oracle_calls})"

        if self.spent_tokens_total + estimated_tokens > self.budget_tokens:
            return False, f"Token budget exceeded ({self.budget_tokens})"

        if self.spent_usd >= self.budget_usd:
            return False, f"USD budget exceeded (${self.budget_usd})"

        return True, "OK"

    def record_call(self, call: OracleCall) -> None:
        """Record an oracle call and update budget."""
        self.calls.append(call)
        self.spent_tokens_in += call.tokens_in
        self.spent_tokens_out += call.tokens_out
        self.spent_oracle_calls += 1
        self.spent_usd += call.cost_usd

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "budget": {
                "tokens": self.budget_tokens,
                "oracle_calls": self.budget_oracle_calls,
                "usd": self.budget_usd,
            },
            "spent": {
                "tokens_in": self.spent_tokens_in,
                "tokens_out": self.spent_tokens_out,
                "tokens_total": self.spent_tokens_total,
                "oracle_calls": self.spent_oracle_calls,
                "usd": self.spent_usd,
            },
            "remaining": {
                "tokens": self.remaining_tokens,
                "oracle_calls": self.remaining_calls,
                "usd": self.remaining_usd,
            },
            "call_count": len(self.calls),
        }

    def to_event(self) -> dict:
        """Format as event log entry for budget checkpoint."""
        return {
            "type": "budget_checkpoint",
            "run_id": self.run_id,
            "spent_tokens": self.spent_tokens_total,
            "spent_calls": self.spent_oracle_calls,
            "spent_usd": self.spent_usd,
            "remaining_tokens": self.remaining_tokens,
            "remaining_calls": self.remaining_calls,
        }


class OracleTracker:
    """Manages ORACLE calls with context packs and budget enforcement.

    Usage:
        tracker = OracleTracker(run_id="my_run", budget_tokens=100000)

        # Build context pack
        pack = tracker.create_context_pack("search_results")
        pack.add_file("src/main.py", content)
        pack.add_tool_output("grep", grep_results)

        # Make oracle call
        with tracker.oracle_call("claude", pack) as call:
            response = call_claude(pack.chunks)
            call.complete(tokens_out=500, output=response)

        # Check budget
        tracker.save_checkpoint(output_dir)
    """

    def __init__(
        self,
        run_id: str,
        budget_tokens: int = 1_000_000,
        budget_oracle_calls: int = 100,
        budget_usd: float = 50.0,
        output_dir: Optional[Path] = None,
    ):
        self.run_id = run_id
        self.output_dir = Path(output_dir) if output_dir else None
        self.ledger = BudgetLedger(
            run_id=run_id,
            budget_tokens=budget_tokens,
            budget_oracle_calls=budget_oracle_calls,
            budget_usd=budget_usd,
        )
        self.context_packs: dict[str, ContextPack] = {}
        self._call_counter = 0

    def create_context_pack(
        self,
        pack_id: str,
        selection_algorithm: str = "manual",
        **selection_params,
    ) -> ContextPack:
        """Create a new context pack for an upcoming oracle call."""
        pack = ContextPack(
            pack_id=pack_id,
            selection_algorithm=selection_algorithm,
            selection_params=selection_params,
        )
        self.context_packs[pack_id] = pack
        return pack

    def oracle_call(
        self,
        oracle_id: str,
        context_pack: ContextPack,
        model: str = "",
        estimated_tokens_out: int = 1000,
    ) -> OracleCall:
        """Create an oracle call record with budget check.

        Raises BudgetExceededError if budget would be exceeded.
        """
        # Check budget
        estimated_total = context_pack.estimated_tokens + estimated_tokens_out
        allowed, reason = self.ledger.check_budget(estimated_total)
        if not allowed:
            raise BudgetExceededError(reason)

        # Save context pack if output dir configured
        if self.output_dir:
            context_pack.save(self.output_dir / "context_packs")

        # Create call record
        self._call_counter += 1
        call = OracleCall(
            oracle_id=oracle_id,
            call_id=f"{self.run_id}_{oracle_id}_{self._call_counter}",
            context_root=context_pack.context_root,
            tokens_in=context_pack.estimated_tokens,
            model=model,
        )

        return call

    def record_call(self, call: OracleCall) -> None:
        """Record a completed oracle call."""
        self.ledger.record_call(call)

    def save_checkpoint(self, output_dir: Optional[Path] = None) -> Path:
        """Save budget ledger and call history."""
        out = Path(output_dir or self.output_dir or ".")
        out.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "ledger": self.ledger.to_dict(),
            "calls": [c.to_dict() for c in self.ledger.calls],
            "context_packs": {
                k: v.to_dict() for k, v in self.context_packs.items()
            },
        }

        path = out / f"oracle_checkpoint_{self.run_id}.json"
        path.write_text(json.dumps(checkpoint, indent=2))
        return path


class BudgetExceededError(Exception):
    """Raised when an oracle call would exceed budget."""
    pass
