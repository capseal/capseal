"""Data availability provider interfaces."""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from bef_zk.stc.merkle import build_kary_levels, prove_kary


class AvailabilityError(RuntimeError):
    """Raised when the DA provider cannot return requested chunks."""


@dataclass(frozen=True)
class ChunkInclusionProof:
    """Proof that a chunk root lives at a specific index in the chunk tree."""

    siblings: List[List[bytes]]
    arity: int
    tree_size: int


@dataclass(frozen=True)
class ChunkFetchResult:
    """Container for fetched chunk values and its Merkle proof."""

    values: List[int]
    proof: ChunkInclusionProof


class DAProvider(ABC):
    """Abstract data availability provider."""

    @abstractmethod
    def fetch_batch(
        self,
        indices: Sequence[int],
        *,
        timeout_ms: int | None = None,
    ) -> Mapping[int, ChunkFetchResult]:
        """Fetch chunks and inclusion proofs for requested indices."""


class LocalFileSystemProvider(DAProvider):
    """Loads chunks from a local archive directory (testing/reference use)."""

    def __init__(
        self,
        *,
        archive_root: Path,
        chunk_handles: Sequence[str],
        chunk_roots: Sequence[bytes],
        tree_arity: int,
    ) -> None:
        if tree_arity < 2:
            raise ValueError("chunk tree arity must be >= 2")
        if len(chunk_handles) != len(chunk_roots):
            raise ValueError("chunk handle/root length mismatch")
        self.archive_root = archive_root
        self._handles = list(chunk_handles)
        self._chunk_roots = list(chunk_roots)
        self._arity = tree_arity
        self._levels = build_kary_levels(self._chunk_roots, self._arity)

    def _chunk_path(self, handle: str) -> Path:
        path = Path(handle)
        if not path.is_absolute():
            path = self.archive_root / path
        return path

    def _load_values(self, path: Path) -> List[int]:
        try:
            return [int(v) for v in json.loads(path.read_text())]
        except FileNotFoundError:
            raise
        except Exception as exc:  # pragma: no cover - defensive for malformed chunks
            raise ValueError(f"failed to decode chunk file {path}") from exc

    def fetch_batch(
        self,
        indices: Sequence[int],
        *,
        timeout_ms: int | None = None,
    ) -> Dict[int, ChunkFetchResult]:
        resolved: Dict[int, ChunkFetchResult] = {}
        if not indices:
            return resolved
        unique_indices = sorted({int(idx) for idx in indices if idx is not None})
        total = len(self._handles)
        for idx in unique_indices:
            if idx < 0 or idx >= total:
                raise IndexError(f"chunk index {idx} out of range (total {total})")
            handle = self._handles[idx]
            if not handle:
                raise ValueError(f"chunk {idx} missing archive handle")
            path = self._chunk_path(handle)
            values = self._load_values(path)
            siblings = prove_kary(self._levels, idx, self._arity)
            proof = ChunkInclusionProof(
                siblings=siblings,
                arity=self._arity,
                tree_size=total,
            )
            resolved[idx] = ChunkFetchResult(values=values, proof=proof)
        return resolved


class PolicyAwareDAClient(DAProvider):
    """Wraps a provider with retry/timeout semantics from policy."""

    def __init__(
        self,
        provider: DAProvider,
        *,
        retries: int,
        timeout_ms: int,
    ) -> None:
        self._provider = provider
        self._retries = max(0, retries)
        self._timeout_ms = max(0, timeout_ms)

    def fetch_batch(
        self,
        indices: Sequence[int],
        *,
        timeout_ms: int | None = None,
    ) -> Mapping[int, ChunkFetchResult]:
        effective_timeout = self._timeout_ms if timeout_ms is None else timeout_ms
        attempt = 0
        last_exc: Exception | None = None
        while attempt <= self._retries:
            try:
                start = time.monotonic()
                result = self._provider.fetch_batch(
                    indices,
                    timeout_ms=effective_timeout,
                )
                end = time.monotonic()
                if effective_timeout and (end - start) * 1000 > effective_timeout:
                    raise AvailabilityError("DA provider timed out")
                return result
            except (TimeoutError, AvailabilityError) as exc:
                last_exc = exc
                attempt += 1
                if attempt > self._retries:
                    raise AvailabilityError("DA provider unavailable after retries") from exc
        if last_exc is not None:
            raise AvailabilityError("DA provider unavailable") from last_exc
        return {}


__all__ = [
    "ChunkFetchResult",
    "ChunkInclusionProof",
    "DAProvider",
    "LocalFileSystemProvider",
]
