"""STC vector commitment backend for PC/FRI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from .aok_cpu import (
    MODULUS,
    ROOT_SEED,
    StreamingAccumulatorCPU,
    chunk_leaf_hash,
    merkle_from_values,
    build_levels,
    merkle_prove,
    root_from_levels,
    merkle_verify,
)
from .merkle import (
    build_kary_levels,
    prove_kary,
    verify_kary,
    multiproof,
    MerkleMultiProof,
)

if TYPE_CHECKING:  # pragma: no cover
    from .vc import VCCommitment


CHUNK_TREE_ARITY = int(os.environ.get("STC_CHUNK_TREE_ARITY", "16"))


@dataclass
class STCCommitment:
    length: int
    chunk_len: int
    num_chunks: int
    global_root: bytes  # Merkle root over chunk roots
    chunk_roots: List[bytes] = field(default_factory=list)
    chain_root: bytes = ROOT_SEED  # streaming hash-chain root
    challenges: List[int] = field(default_factory=list)
    sketches: List[int] = field(default_factory=list)
    powers: List[int] = field(default_factory=list)
    chunk_tree_arity: int = 2

    def to_json(self, include_chunk_roots: bool = False) -> Dict:
        data = {
            "schema": "bef_pc_commit_v1",
            "length": self.length,
            "chunk_len": self.chunk_len,
            "num_chunks": self.num_chunks,
            "global_root": self.global_root.hex(),
            "chunk_tree_arity": self.chunk_tree_arity,
        }
        if include_chunk_roots:
            data["chunk_roots"] = [cr.hex() for cr in self.chunk_roots]
        if self.chain_root:
            data["chain_root"] = self.chain_root.hex()
        if self.challenges:
            data["challenges"] = list(self.challenges)
        if self.sketches:
            data["sketches"] = [int(x) % MODULUS for x in self.sketches]
        if self.powers:
            data["powers"] = [int(x) % MODULUS for x in self.powers]
        return data

    @classmethod
    def from_json(cls, data: Dict) -> STCCommitment:
        if data.get("schema") != "bef_pc_commit_v1":
            raise ValueError("unexpected schema")
        roots = [bytes.fromhex(x) for x in data.get("chunk_roots", [])]
        return cls(
            length=int(data["length"]),
            chunk_len=int(data["chunk_len"]),
            num_chunks=int(data["num_chunks"]),
            global_root=bytes.fromhex(data["global_root"]),
            chunk_roots=roots,
            chain_root=bytes.fromhex(data.get("chain_root", ROOT_SEED.hex())),
            challenges=list(data.get("challenges", [])),
            sketches=[int(x) % MODULUS for x in data.get("sketches", [])],
            powers=[int(x) % MODULUS for x in data.get("powers", [])],
            chunk_tree_arity=int(data.get("chunk_tree_arity", 2)),
        )


@dataclass
class STCIndexProof:
    index: int
    value: int
    chunk_index: int
    chunk_offset: int
    leaf_pos: int
    leaf_path: List[bytes]
    chunk_root: bytes
    chunk_pos: int
    chunk_root_path: List[List[bytes]]

    def to_json(self) -> Dict:
        return {
            "schema": "bef_pc_open_v1",
            "index": self.index,
            "value": int(self.value) % MODULUS,
            "chunk_index": self.chunk_index,
            "chunk_offset": self.chunk_offset,
            "leaf_pos": self.leaf_pos,
            "leaf_path": [hx.hex() for hx in self.leaf_path],
            "chunk_root": self.chunk_root.hex(),
            "chunk_pos": self.chunk_pos,
            "chunk_root_path": [[hx.hex() for hx in level] for level in self.chunk_root_path],
        }

    @classmethod
    def from_json(cls, data: Dict) -> STCIndexProof:
        if data.get("schema") != "bef_pc_open_v1":
            raise ValueError("unexpected schema")
        return cls(
            index=int(data["index"]),
            value=int(data["value"]),
            chunk_index=int(data["chunk_index"]),
            chunk_offset=int(data["chunk_offset"]),
            leaf_pos=int(data["leaf_pos"]),
            leaf_path=[bytes.fromhex(x) for x in data.get("leaf_path", [])],
            chunk_root=bytes.fromhex(data["chunk_root"]),
            chunk_pos=int(data["chunk_pos"]),
            chunk_root_path=[
                [bytes.fromhex(x) for x in level]
                for level in data.get("chunk_root_path", [])
            ],
        )


@dataclass
class STCChunkProof:
    chunk_index: int
    chunk_offset: int
    values: List[int]
    chunk_root: bytes
    chunk_pos: int
    chunk_root_path: List[List[bytes]]

    def to_json(self) -> Dict:
        return {
            "schema": "bef_pc_chunk_open_v1",
            "chunk_index": self.chunk_index,
            "chunk_offset": self.chunk_offset,
            "values": [int(v) % MODULUS for v in self.values],
            "chunk_root": self.chunk_root.hex(),
            "chunk_pos": self.chunk_pos,
            "chunk_root_path": [[hx.hex() for hx in level] for level in self.chunk_root_path],
        }

    @classmethod
    def from_json(cls, data: Dict) -> STCChunkProof:
        if data.get("schema") != "bef_pc_chunk_open_v1":
            raise ValueError("unexpected schema")
        return cls(
            chunk_index=int(data["chunk_index"]),
            chunk_offset=int(data["chunk_offset"]),
            values=[int(v) % MODULUS for v in data.get("values", [])],
            chunk_root=bytes.fromhex(data["chunk_root"]),
            chunk_pos=int(data["chunk_pos"]),
            chunk_root_path=[
                [bytes.fromhex(x) for x in level]
                for level in data.get("chunk_root_path", [])
            ],
        )


def stc_build_pc_commitment(acc: StreamingAccumulatorCPU) -> STCCommitment:
    acc._ensure_challenges()
    chunk_roots = [chunk.root for chunk in acc.chunks]
    if not chunk_roots:
        raise ValueError("no chunks")
    levels = build_kary_levels(chunk_roots, CHUNK_TREE_ARITY)
    commitment = STCCommitment(
        length=acc.length,
        chunk_len=acc.chunk_len,
        num_chunks=len(acc.chunks),
        global_root=root_from_levels(levels),
        chunk_roots=chunk_roots,
        chain_root=acc.root,
        challenges=list(acc.challenges),
        sketches=[int(x) % MODULUS for x in acc.sketches],
        powers=[int(x) % MODULUS for x in acc.powers],
        chunk_tree_arity=CHUNK_TREE_ARITY,
    )
    commitment._chunk_levels = levels  # type: ignore[attr-defined]
    return commitment


def stc_open_index(
    values: Sequence[int] | Any,
    commitment: STCCommitment,
    index: int,
    chunk_root_levels: Optional[List[List[bytes]]] = None,
) -> STCIndexProof:
    if index < 0 or index >= commitment.length:
        raise ValueError("index out of range")
    chunk_len = commitment.chunk_len
    chunk_index = index // chunk_len
    if chunk_index >= commitment.num_chunks:
        raise ValueError("chunk index out of range")
    offset = chunk_index * chunk_len
    local_idx = index - offset
    chunk_values = _chunk_values_from_source(values, chunk_index, offset, min(chunk_len, commitment.length - offset))
    if not chunk_values:
        raise ValueError("missing chunk values")
    leaves = [chunk_leaf_hash(offset, j, val) for j, val in enumerate(chunk_values)]
    chunk_levels = build_levels(leaves)
    leaf_path = merkle_prove(chunk_levels, local_idx)
    chunk_root = root_from_levels(chunk_levels)
    if chunk_root_levels is None:
        if not getattr(commitment, "_chunk_levels", None):
            if not commitment.chunk_roots:
                raise ValueError("chunk roots unavailable")
            commitment._chunk_levels = build_kary_levels(  # type: ignore[attr-defined]
                commitment.chunk_roots,
                commitment.chunk_tree_arity,
            )
        chunk_root_levels = commitment._chunk_levels  # type: ignore[attr-defined]
    chunk_root_path = prove_kary(chunk_root_levels, chunk_index, commitment.chunk_tree_arity)
    return STCIndexProof(
        index=index,
        value=_value_from_source(values, index),
        chunk_index=chunk_index,
        chunk_offset=offset,
        leaf_pos=local_idx,
        leaf_path=leaf_path,
        chunk_root=chunk_root,
        chunk_pos=chunk_index,
        chunk_root_path=chunk_root_path,
    )


def stc_verify_index(commitment: STCCommitment, proof: STCIndexProof) -> bool:
    if proof.index < 0 or proof.index >= commitment.length:
        return False
    if proof.chunk_index != proof.chunk_pos:
        return False
    leaf = chunk_leaf_hash(proof.chunk_offset, proof.leaf_pos, proof.value)
    if not merkle_verify(proof.chunk_root, leaf, proof.leaf_pos, proof.leaf_path):
        return False
    if not verify_kary(
        commitment.global_root,
        proof.chunk_root,
        proof.chunk_pos,
        proof.chunk_root_path,
        commitment.chunk_tree_arity,
        commitment.num_chunks,
    ):
        return False
    return True


def stc_open_chunk(
    values: Sequence[int] | Any,
    commitment: STCCommitment,
    chunk_index: int,
    chunk_root_levels: Optional[List[List[bytes]]] = None,
) -> STCChunkProof:
    if chunk_index < 0 or chunk_index >= commitment.num_chunks:
        raise ValueError("chunk index out of range")
    chunk_len = commitment.chunk_len
    offset = chunk_index * chunk_len
    chunk_values = _chunk_values_from_source(values, chunk_index, offset, min(chunk_len, commitment.length - offset))
    if not chunk_values:
        raise ValueError("missing chunk values")
    reduced = [int(v) % MODULUS for v in chunk_values]
    chunk_root = merkle_from_values(reduced, offset)
    if chunk_root_levels is None:
        if not getattr(commitment, "_chunk_levels", None):
            if not commitment.chunk_roots:
                raise ValueError("chunk roots unavailable")
            commitment._chunk_levels = build_kary_levels(  # type: ignore[attr-defined]
                commitment.chunk_roots,
                commitment.chunk_tree_arity,
            )
        chunk_root_levels = commitment._chunk_levels  # type: ignore[attr-defined]
    chunk_root_path = prove_kary(chunk_root_levels, chunk_index, commitment.chunk_tree_arity)
    return STCChunkProof(
        chunk_index=chunk_index,
        chunk_offset=offset,
        values=reduced,
        chunk_root=chunk_root,
        chunk_pos=chunk_index,
        chunk_root_path=chunk_root_path,
    )


def _get_commit_root(commitment: STCCommitment | 'VCCommitment') -> bytes:
    return getattr(commitment, "global_root", None) or getattr(commitment, "root")


def stc_verify_chunk(
    commitment: STCCommitment | 'VCCommitment',
    proof: STCChunkProof,
) -> bool:
    num_chunks = getattr(commitment, "num_chunks", None)
    chunk_len = getattr(commitment, "chunk_len", None)
    total_len = getattr(commitment, "length", None)
    if num_chunks is None or chunk_len is None or total_len is None:
        return False
    if proof.chunk_index < 0 or proof.chunk_index >= num_chunks:
        return False
    if proof.chunk_pos != proof.chunk_index:
        return False
    if proof.chunk_offset < 0 or proof.chunk_offset >= total_len:
        return False
    # last chunk may be partial; otherwise enforce offset alignment
    if proof.chunk_index < num_chunks - 1 and proof.chunk_offset != proof.chunk_index * chunk_len:
        return False
    expected_root = merkle_from_values(proof.values, proof.chunk_offset)
    if expected_root != proof.chunk_root:
        return False
    root = _get_commit_root(commitment)
    if root is None:
        return False
    chunk_tree_arity = getattr(commitment, "chunk_tree_arity", CHUNK_TREE_ARITY)
    if not verify_kary(
        root,
        proof.chunk_root,
        proof.chunk_pos,
        proof.chunk_root_path,
        chunk_tree_arity,
        num_chunks,
    ):
        return False
    return True
def _chunk_values_from_source(source: Any, chunk_index: int, offset: int, chunk_length: int) -> List[int]:
    getter = getattr(source, "get_chunk_values", None)
    if callable(getter):
        return [int(v) % MODULUS for v in getter(chunk_index)]
    # fallback: assume sequence slicing
    return [int(v) % MODULUS for v in source[offset : offset + chunk_length]]


def _value_from_source(source: Any, index: int) -> int:
    getter = getattr(source, "get_value", None)
    if callable(getter):
        return int(getter(index)) % MODULUS
    return int(source[index]) % MODULUS
