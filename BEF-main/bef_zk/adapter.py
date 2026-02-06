"""Abstractions for plugging alternative trace/proof backends into the pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


from .spec import TraceSpecV1
from bef_zk.capsule.eval_trace import EvalTraceRow


class ReplayStatus(Enum):
    """Result of semantic replay verification."""
    IDENTICAL = "identical"           # Exact match
    EQUIVALENT = "equivalent"         # Within tolerance
    DIVERGED = "diverged"             # Mismatch exceeds tolerance
    UNSUPPORTED = "unsupported"       # Adapter doesn't support replay


@dataclass
class RowDivergence:
    """Details about a divergence at a specific row."""
    row_index: int
    column_name: str
    original_value: Any
    replayed_value: Any
    absolute_diff: float
    relative_diff: float


@dataclass
class ReplayResult:
    """Result of replaying a trace for semantic verification."""
    status: ReplayStatus
    divergences: list[RowDivergence] = field(default_factory=list)
    max_absolute_diff: float = 0.0
    max_relative_diff: float = 0.0
    rows_checked: int = 0
    rows_matched: int = 0
    replay_time_sec: float = 0.0
    original_trace_hash: str = ""
    replayed_trace_hash: str = ""
    message: str = ""
    row_indices: list[int] | None = None


@dataclass
class TraceArtifacts:
    """Normalized outputs from TraceAdapter.simulate_trace."""

    trace_id: str
    trace_spec: TraceSpecV1
    trace_spec_hash: str
    bef_trace: dict[str, Any]
    row_width: int
    context: Any
    trace_time_sec: Optional[float] = None
    # For replay: store determinism seed if available
    determinism_seed: Optional[bytes] = None


@dataclass
class TraceCommitment:
    """Outputs from the TraceAdapter.commit_to_trace phase."""

    row_commitment: Any
    row_archive_artifact: dict[str, Any]
    chunk_handles: list[Any]
    chunk_roots_hex: list[str]
    chunk_roots_digest: str
    chunk_roots_paths: dict[str, Path]
    profile_data: dict[str, float]
    extra: dict[str, Any] | None = None
    backend_state: Any | None = None


@dataclass
class ProofArtifacts:
    """Outputs from TraceAdapter.generate_proof used by the pipeline."""

    proof_obj: Any
    proof_json: str
    proof_bytes: bytes
    profile_data: dict[str, float]
    chunk_leaf_stats: dict[str, int]
    extra: dict[str, Any] | None = None


@dataclass
class AuditBundle:
    """Audit trail bundle for a capsule run."""
    run_id: str
    events_path: Path
    events_hash: str
    events_count: int
    first_event_hash: str
    last_event_hash: str
    # Optional signatures covering event ranges
    signatures: list[dict[str, Any]] = field(default_factory=list)


class TraceAdapter(ABC):
    """Interface for pluggable trace/proof backends."""

    name: str = "unknown"

    def __init__(self, args: Any) -> None:
        self.args = args
        self._progress_callback = None

    @classmethod
    def add_arguments(cls, parser: Any) -> None:
        """Hook for adapter-specific CLI arguments."""

    @abstractmethod
    def simulate_trace(self, args: Any) -> TraceArtifacts:
        """Produce a trace along with metadata required by the pipeline."""

    @abstractmethod
    def extract_public_inputs(self, artifacts: TraceArtifacts) -> list[dict[str, Any]]:
        """Return the public inputs derived from the prepared trace."""

    @abstractmethod
    def commit_to_trace(
        self,
        artifacts: TraceArtifacts,
        *,
        row_archive_dir: Path,
    ) -> TraceCommitment:
        """Commit to the execution trace and export STC artifacts."""

    @abstractmethod
    def generate_proof(
        self,
        artifacts: TraceArtifacts,
        commitment: TraceCommitment,
        *,
        statement_hash: bytes,
        binding_hash: bytes | None = None,
        encoding_id: str,
        trace_path: Path,
    ) -> ProofArtifacts:
        """Generate the backend proof using a finalized statement hash."""

    @abstractmethod
    def verify(
        self,
        proof_json: str,
        statement_hash: bytes,
        artifacts: TraceArtifacts,
        *,
        binding_hash: bytes | None = None,
    ) -> tuple[bool, dict[str, Any], float]:
        """Run the backend verifier for sanity checks and benchmarking."""

    def replay_trace(
        self,
        original_artifacts: TraceArtifacts,
        *,
        tolerance: float = 0.0,
        max_divergences: int = 100,
        row_range: tuple[int, int] | None = None,
        sample: int | None = None,
        seed: int | None = None,
        stop_on_first: bool = False,
    ) -> ReplayResult:
        """Re-execute the trace and compare with original for semantic replay.

        This method enables deterministic replay verification. The adapter
        re-executes the computation using the same inputs and compares the
        resulting trace row-by-row with the original.

        Args:
            original_artifacts: The original TraceArtifacts from the capsule
            tolerance: Maximum allowed relative difference (0.0 = exact match)
            max_divergences: Stop after this many divergences found
            row_range: optional (start, end) rows to replay (end exclusive)
            sample: optional number of rows to sample uniformly for replay
            seed: RNG seed for sampling (if sample provided)
            stop_on_first: stop immediately when a divergence is found

        Returns:
            ReplayResult with status and any divergences found

        Note:
            Default implementation returns UNSUPPORTED. Adapters that support
            deterministic replay should override this method.
        """
        return ReplayResult(
            status=ReplayStatus.UNSUPPORTED,
            message=f"Adapter '{self.name}' does not support semantic replay",
        )

    def supports_replay(self) -> bool:
        """Return True if this adapter supports semantic replay verification."""
        return False

    def build_evaluation_rows(self, artifacts: TraceArtifacts) -> list[EvalTraceRow] | None:
        """Return per-sample evaluation rows for binding benchmark metrics.

        Adapters that can expose deterministic evaluation traces should
        override this hook and return EvalTraceRow entries. The pipeline will
        serialize and commit them when sealing capsules.
        """

        return None

    def get_determinism_contract(self) -> dict[str, Any]:
        """Return the determinism contract for this adapter.

        The contract specifies what inputs are required for deterministic
        replay and what tolerance is expected for different column types.

        Returns:
            Dict with:
                - required_inputs: List of input names needed for replay
                - column_tolerances: Dict of column_name -> max_relative_diff
                - random_seed_source: How randomness is seeded (e.g., "capsule_hash")
                - floating_point_mode: Rounding mode used (e.g., "round_to_nearest")
        """
        return {
            "supported": False,
            "required_inputs": [],
            "column_tolerances": {},
            "random_seed_source": None,
            "floating_point_mode": None,
        }

    def set_progress_callback(self, callback: Optional[Callable[[dict[str, Any]], None]]) -> None:
        """Optional hook used by CapsuleBench to observe proving progress."""

        self._progress_callback = callback

    def _emit_progress(self, event_type: str, data: dict[str, Any]) -> None:
        if self._progress_callback is not None:
            self._progress_callback({"type": event_type, "data": data})
