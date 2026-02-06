"""Reusable CPU-side helpers for the streaming trace commitment."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .merkle import build_levels, prove as merkle_prove, root_from_levels, verify as merkle_verify
from .archive import ChunkArchive
from .poseidon2 import poseidon2_hash

MODULUS = (1 << 61) - 1
ROOT_SEED = hashlib.sha256(b"bef-init").digest()


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _int_to_bytes(value: int, length: int = 32) -> bytes:
    return int(value % MODULUS).to_bytes(length, "big", signed=False)


def hash_root_update(root: bytes, offset: int, chunk_root: bytes) -> bytes:
    root_val = int.from_bytes(root, "big") % MODULUS
    chunk_val = int.from_bytes(chunk_root, "big") % MODULUS
    off_val = int(offset) % MODULUS
    new_val = poseidon2_hash([root_val, off_val, chunk_val])
    return int(new_val % MODULUS).to_bytes(32, "big")


def chunk_leaf_hash(offset: int, local_idx: int, value: int) -> bytes:
    return _hash(
        offset.to_bytes(8, "big")
        + local_idx.to_bytes(8, "big")
        + _int_to_bytes(value)
    )


def merkle_from_values(values: Sequence[int], offset: int) -> bytes:
    leaves = [chunk_leaf_hash(offset, idx, val) for idx, val in enumerate(values)]
    if not leaves:
        raise ValueError("chunk must have at least one value")
    return root_from_levels(build_levels(leaves))


def derive_challenges(root: bytes, m: int) -> List[int]:
    out: List[int] = []
    counter = 0
    while len(out) < m:
        h = hashlib.sha256(root + counter.to_bytes(4, "big")).digest()
        candidate = int.from_bytes(h, "big") % MODULUS
        if candidate != 0:
            out.append(candidate)
        counter += 1
    return out


def chunk_sketch(values: Sequence[int], offset: int, challenges: Sequence[int]) -> List[int]:
    sketches: List[int] = []
    for r in challenges:
        acc = 0
        pow_r = pow(r % MODULUS, offset, MODULUS)
        for val in values:
            acc = (acc + (int(val) % MODULUS) * pow_r) % MODULUS
            pow_r = (pow_r * r) % MODULUS
        sketches.append(acc)
    return sketches


@dataclass
class ChunkRecord:
    values: List[int]
    offset: int
    length: int
    root: bytes
    sketch_vec: List[int] = field(default_factory=list)
    archive_handle: Optional[str] = None


class StreamingAccumulatorCPU:
    """In-memory STC accumulator (mirrors the GPU path for testing)."""

    def __init__(
        self,
        *,
        num_challenges: int,
        chunk_len: int,
        challenge_seed: Optional[bytes] = None,
        archive: ChunkArchive | None = None,
    ) -> None:
        self.length = 0
        self.chunk_len = chunk_len
        self.num_challenges = num_challenges
        self.chunks: List[ChunkRecord] = []
        self.root = ROOT_SEED
        seed = challenge_seed if challenge_seed is not None else ROOT_SEED
        self._challenge_seed = seed
        self.challenges: List[int] = []
        self.sketches: List[int] = [0] * num_challenges if num_challenges else []
        self.powers: List[int] = [1] * num_challenges if num_challenges else []
        self.archive = archive
        if num_challenges:
            self.challenges = derive_challenges(seed, num_challenges)

    def _streaming_update(self, reduced: Sequence[int], offset: int) -> List[int]:
        if self.num_challenges == 0 or not self.challenges:
            return []
        chunk_sketch: List[int] = []
        for idx, r in enumerate(self.challenges):
            pow_r = self.powers[idx]
            acc = 0
            for val in reduced:
                acc = (acc + pow_r * val) % MODULUS
                pow_r = (pow_r * r) % MODULUS
            self.powers[idx] = pow_r
            self.sketches[idx] = (self.sketches[idx] + acc) % MODULUS
            chunk_sketch.append(acc)
        return chunk_sketch

    def _ensure_challenges(self) -> None:
        if self.num_challenges == 0 or self.challenges:
            return
        seed = self._challenge_seed or self.root
        self.challenges = derive_challenges(seed, self.num_challenges)
        self.sketches = [0] * self.num_challenges
        self.powers = [1] * self.num_challenges
        for chunk in self.chunks:
            chunk.sketch_vec = self._streaming_update(chunk.values, chunk.offset)
        if self.challenges:
            for idx, r in enumerate(self.challenges):
                self.powers[idx] = pow(r % MODULUS, self.length, MODULUS)

    def add_chunk(self, values: Sequence[int]) -> None:
        if not values:
            raise ValueError("chunk must contain values")
        reduced = [int(v) % MODULUS for v in values]
        chunk_root = merkle_from_values(reduced, self.length)
        archive_handle = None
        stored_values = list(reduced)
        if self.archive is not None:
            archive_handle = self.archive.store_chunk(len(self.chunks), reduced)
            stored_values = []
        sketch_vec = self._streaming_update(reduced, self.length)
        self.chunks.append(
            ChunkRecord(
                values=stored_values,
                offset=self.length,
                length=len(reduced),
                root=chunk_root,
                sketch_vec=sketch_vec,
                archive_handle=archive_handle,
            )
        )
        self.root = hash_root_update(self.root, self.length, chunk_root)
        self.length += len(reduced)

    def export_chunks(self) -> List[ChunkRecord]:
        exported: List[ChunkRecord] = []
        for chunk in self.chunks:
            exported.append(
                ChunkRecord(
                    values=list(chunk.values),
                    offset=chunk.offset,
                    length=chunk.length,
                    root=chunk.root,
                    sketch_vec=list(chunk.sketch_vec),
                    archive_handle=chunk.archive_handle,
                )
            )
        return exported


__all__ = [
    "MODULUS",
    "ROOT_SEED",
    "StreamingAccumulatorCPU",
    "chunk_leaf_hash",
    "chunk_sketch",
    "derive_challenges",
    "hash_root_update",
    "merkle_from_values",
    "merkle_prove",
    "merkle_verify",
    "build_levels",
    "root_from_levels",
]
