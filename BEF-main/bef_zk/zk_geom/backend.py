"""Backends for committing / opening trace rows."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from bef_zk.stc.aok_cpu import MODULUS, ROOT_SEED, StreamingAccumulatorCPU
from bef_zk.stc.pc_backend import CHUNK_TREE_ARITY
from ..stc.merkle import build_levels, root_from_levels, prove as merkle_prove, verify as merkle_verify
from ..stc.pc_backend import STCChunkProof, stc_verify_chunk
from ..stc.vc import STCVectorCommitment, VCCommitment


FieldRow = List[int]


@dataclass
class RowCommitment:
    backend: str
    row_width: int
    params: Dict[str, Any]
    prover_state: Any | None = None  # cleared before serialization


@dataclass
class RowOpening:
    backend: str
    row_index: int
    row_values: FieldRow
    proof: Dict[str, Any]
    next_index: Optional[int] = None
    next_row_values: Optional[FieldRow] = None
    next_proof: Optional[Dict[str, Any]] = None


class RowBackend(Protocol):
    name: str

    def __init__(self, row_width: int, **kwargs: Any): ...

    def commit_rows(self, rows: List[FieldRow]) -> RowCommitment:
        """Commit to the full row matrix."""

    def open_row(self, commitment: RowCommitment, idx: int) -> tuple[FieldRow, Dict[str, Any]]:
        """Return (row_values, proof) for the specified row index."""

    def verify_leaf(
        self,
        commitment: RowCommitment,
        idx: int,
        row_values: FieldRow,
        proof: Dict[str, Any],
    ) -> bool:
        """Check that row_values belong to the commitment at index idx."""

    def streaming_init(self) -> Dict[str, Any]:  # pragma: no cover - interface default
        return {"rows": []}

    def streaming_append(self, state: Dict[str, Any], row: FieldRow) -> None:  # pragma: no cover - interface default
        state.setdefault("rows", []).append(list(row))

    def streaming_finalize(self, state: Dict[str, Any]) -> RowCommitment:  # pragma: no cover - interface default
        rows = state.get("rows", [])
        return self.commit_rows(rows)

    def streaming_init(self) -> Dict[str, Any]:
        """Return a mutable state used for streaming commits."""

    def streaming_append(self, state: Dict[str, Any], row: FieldRow) -> None:
        """Append a row to the streaming state."""

    def streaming_finalize(self, state: Dict[str, Any]) -> RowCommitment:
        """Finalize the streaming state into a RowCommitment."""


###############################################################################
# Backend registry helpers


_BACKENDS: Dict[str, type[RowBackend]] = {}


def register_backend(cls: type[RowBackend]) -> type[RowBackend]:
    _BACKENDS[cls.name] = cls
    return cls


def get_row_backend(name: str, row_width: int, **kwargs: Any) -> RowBackend:
    try:
        backend_cls = _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"unknown row backend {name}") from exc
    return backend_cls(row_width=row_width, **kwargs)


def available_row_backends() -> List[str]:
    return sorted(_BACKENDS.keys())


###############################################################################
# STC streaming backend


@register_backend
class STCRowBackend(RowBackend):
    name = "geom_stc_fri"

    def __init__(self, row_width: int, archive_dir: str | Path | None = None, chunk_tree_arity: int | None = None, use_gpu: bool = False):
        self.row_width = row_width
        self.archive_dir = Path(archive_dir).resolve() if archive_dir is not None else None
        self.chunk_tree_arity = chunk_tree_arity
        self.use_gpu = use_gpu

    def commit_rows(self, rows: List[FieldRow]) -> RowCommitment:
        flat: List[int] = []
        for row in rows:
            if len(row) != self.row_width:
                raise ValueError("row width mismatch")
            flat.extend(int(v) for v in row)
        vc = STCVectorCommitment(
            chunk_len=self.row_width,
            archive_dir=self.archive_dir,
            use_gpu=self.use_gpu,
        )
        if self.chunk_tree_arity is not None:
            vc.chunk_tree_arity = self.chunk_tree_arity
        vc_commit = vc.commit(flat)
        store = vc._store.get(vc_commit.root, {})
        chunk_records = store.get("chunks", [])
        params = {
            "root": vc_commit.root.hex(),
            "length": vc_commit.length,
            "chunk_len": vc_commit.chunk_len,
            "num_chunks": vc_commit.num_chunks,
            "chain_root": vc_commit.chain_root.hex(),
            "challenges": list(vc_commit.challenges),
            "sketches": [int(x) % MODULUS for x in vc_commit.sketches],
            "powers": [int(x) % MODULUS for x in vc_commit.powers],
            "archive_root": vc.get_archive_root(vc_commit),
            "chunk_handles": vc.get_chunk_handles(vc_commit),
            "chunk_tree_arity": getattr(vc_commit, "chunk_tree_arity", CHUNK_TREE_ARITY),
            "chunk_roots_hex": [getattr(chunk, "root").hex() for chunk in chunk_records],
        }
        return RowCommitment(
            backend=self.name,
            row_width=self.row_width,
            params=params,
            prover_state={"vc": vc, "commitment": vc_commit},
        )

    def open_row(self, commitment: RowCommitment, idx: int) -> tuple[FieldRow, Dict[str, Any]]:
        state = commitment.prover_state
        if state is None:
            raise ValueError("prover state missing for row openings")
        vc: STCVectorCommitment = state["vc"]
        vc_commit: VCCommitment = state["commitment"]
        chunk_proof = vc.open_chunk(vc_commit, idx)
        return list(chunk_proof.values), chunk_proof.to_json()

    def verify_leaf(
        self,
        commitment: RowCommitment,
        idx: int,
        row_values: FieldRow,
        proof: Dict[str, Any],
    ) -> bool:
        params = commitment.params
        vc_commit = VCCommitment(
            root=bytes.fromhex(params["root"]),
            length=int(params["length"]),
            chunk_len=int(params["chunk_len"]),
            num_chunks=int(params["num_chunks"]),
            chain_root=bytes.fromhex(params.get("chain_root", ROOT_SEED.hex())),
            challenges=list(params.get("challenges", [])),
            sketches=[int(x) % MODULUS for x in params.get("sketches", [])],
            powers=[int(x) % MODULUS for x in params.get("powers", [])],
            chunk_tree_arity=int(params.get("chunk_tree_arity", CHUNK_TREE_ARITY)),
        )
        chunk_proof = STCChunkProof.from_json(proof)
        if chunk_proof.chunk_index != idx:
            return False
        if not stc_verify_chunk(vc_commit, chunk_proof):
            return False
        expected = [int(v) for v in row_values]
        return all((chunk_proof.values[i] % MODULUS) == (expected[i] % MODULUS) for i in range(len(expected)))

    def streaming_init(self) -> Dict[str, Any]:
        return {
            "rows": [],
            "acc": StreamingAccumulatorCPU(num_challenges=0, chunk_len=self.row_width),
        }

    def streaming_append(self, state: Dict[str, Any], row: FieldRow) -> None:
        if len(row) != self.row_width:
            raise ValueError("row width mismatch")
        state["rows"].append(list(row))
        state["acc"].add_chunk(row)

    def streaming_finalize(self, state: Dict[str, Any]) -> RowCommitment:
        rows = state.get("rows", [])
        return self.commit_rows(rows)


try:
    from bef_zk import bef_rust
except ImportError:
    bef_rust = None

###############################################################################
# Rust STC streaming backend


@register_backend
class RustSTCRowBackend(RowBackend):
    name = "geom_stc_rust"

    def __init__(self, row_width: int, archive_dir: str | Path | None = None, chunk_tree_arity: int | None = None, **kwargs: Any):
        if bef_rust is None:
            raise RuntimeError("bef_rust extension not available")
        self.row_width = row_width
        self.archive_dir = Path(archive_dir).resolve() if archive_dir is not None else None
        self.chunk_tree_arity = chunk_tree_arity
        # We ignore use_gpu for now in Rust backend commit phase (Rust itself is fast)

    def commit_rows(self, rows: List[FieldRow]) -> RowCommitment:
        from bef_zk.stc.aok_cpu import ROOT_SEED, derive_challenges, hash_root_update
        from bef_zk.stc.vc import STCVectorCommitment, VCCommitment, STCCommitment
        from bef_zk.stc.archive import ChunkArchive
        
        # 1. Setup Parameters
        num_challenges = 2
        challenges = derive_challenges(ROOT_SEED, num_challenges)
        params = bef_rust.PyStcParams(challenges, self.row_width)
        
        # 2. Prepare Data
        flat_values = []
        for row in rows:
            flat_values.extend([int(x) for x in row])
            
        archive = ChunkArchive(root_dir=self.archive_dir)
        archive_dir_str = str(archive.root)
        archive.root.mkdir(parents=True, exist_ok=True)

        # 3. Call Rust Optimized Batch Commit
        result = bef_rust.commit_trace_batch(params, flat_values, self.row_width, archive_dir_str)
        
        # 4. Extract Results
        s_vals = [self._parse_fp_hex(x) for x in result.s_hex]
        pow_vals = [self._parse_fp_hex(x) for x in result.pow_hex]
        chunk_roots_list = [bytes.fromhex(c.root) for c in result.chunks]
        
        # 5. Build Python-compatible Commitment Metadata
        # We standardise on tree_arity=2 because that is what our Rust backend computes.
        tree_arity = 2
        
        # Recompute chain_root in Python (Poseidon) for compatibility
        chain_root = ROOT_SEED
        current_len = 0
        for r in chunk_roots_list:
            chain_root = hash_root_update(chain_root, current_len, r)
            current_len += self.row_width
            
        # The Rust result.global_root is a binary tree root.
        root_hex = result.global_root

        stc_commit = STCCommitment(
            length=len(rows) * self.row_width,
            chunk_len=self.row_width,
            num_chunks=len(rows),
            global_root=bytes.fromhex(root_hex),
            chunk_roots=chunk_roots_list,
            chain_root=chain_root,
            challenges=challenges,
            sketches=s_vals,
            powers=pow_vals,
            chunk_tree_arity=tree_arity
        )
        
        # 6. Populate Prover Store
        from bef_zk.stc.aok_cpu import ChunkRecord
        records_obj = [
            ChunkRecord(
                offset=r.offset,
                length=self.row_width,
                values=[], 
                root=bytes.fromhex(r.root),
                archive_handle=r.archive_handle
            ) for r in result.chunks
        ]
            
        vc = STCVectorCommitment(
            chunk_len=self.row_width,
            num_challenges=len(challenges),
            archive_dir=self.archive_dir
        )
        vc._store[stc_commit.global_root] = {
            "commit": stc_commit,
            "chunks": records_obj,
            "archive": archive,
            "archive_root": str(archive.root),
            "debug_chunk_hist": {}
        }
        
        commit_params = {
            "root": root_hex,
            "length": stc_commit.length,
            "chunk_len": stc_commit.chunk_len,
            "num_chunks": stc_commit.num_chunks,
            "chain_root": chain_root.hex(),
            "challenges": challenges,
            "sketches": s_vals,
            "powers": pow_vals,
            "archive_root": str(archive.root),
            "chunk_handles": [r.archive_handle for r in records_obj],
            "chunk_tree_arity": stc_commit.chunk_tree_arity,
            "chunk_roots_hex": [r.root.hex() for r in records_obj],
        }
        
        # 7. Wrap for Interface
        vc_commitment_wrapper = VCCommitment(
            root=stc_commit.global_root,
            length=stc_commit.length,
            chunk_len=stc_commit.chunk_len,
            num_chunks=stc_commit.num_chunks,
            challenges=stc_commit.challenges,
            sketches=stc_commit.sketches,
            powers=stc_commit.powers,
            chain_root=stc_commit.chain_root,
            chunk_tree_arity=stc_commit.chunk_tree_arity
        )

        return RowCommitment(
            backend=self.name,
            row_width=self.row_width,
            params=commit_params,
            prover_state={"vc": vc, "commitment": vc_commitment_wrapper}, 
        )

    def _parse_fp_hex(self, s: str) -> int:
        # Handles Rust debug format like "0x123..."
        clean = s.strip().lower()
        if "0x" in clean:
            clean = clean.split("0x")[1]
        if not clean:
            return 0
        return int(clean, 16)

    def open_row(self, commitment: RowCommitment, idx: int) -> tuple[FieldRow, Dict[str, Any]]:
        # Delegate to Python impl for opening since the archive format is shared
        delegate = STCRowBackend(self.row_width, self.archive_dir, self.chunk_tree_arity)
        return delegate.open_row(commitment, idx)

    def verify_leaf(
        self,
        commitment: RowCommitment,
        idx: int,
        row_values: FieldRow,
        proof: Dict[str, Any],
    ) -> bool:
        delegate = STCRowBackend(self.row_width, self.archive_dir, self.chunk_tree_arity)
        return delegate.verify_leaf(commitment, idx, row_values, proof)

    def streaming_init(self) -> Dict[str, Any]:
        # TODO: Use Rust state for streaming
        delegate = STCRowBackend(self.row_width, self.archive_dir, self.chunk_tree_arity)
        return delegate.streaming_init()

    def streaming_append(self, state: Dict[str, Any], row: FieldRow) -> None:
        # TODO: Use Rust update
        delegate = STCRowBackend(self.row_width, self.archive_dir, self.chunk_tree_arity)
        delegate.streaming_append(state, row)

    def streaming_finalize(self, state: Dict[str, Any]) -> RowCommitment:
        delegate = STCRowBackend(self.row_width, self.archive_dir, self.chunk_tree_arity)
        commitment = delegate.streaming_finalize(state)
        commitment.backend = self.name
        return commitment


###############################################################################
# Plain Merkle backend


def _hash_row(row_index: int, row: FieldRow) -> bytes:
    h = hashlib.sha256()
    h.update(row_index.to_bytes(8, "big"))
    for val in row:
        h.update(int(val).to_bytes(16, "big", signed=False))
    return h.digest()


@register_backend
class MerkleRowBackend(RowBackend):
    name = "geom_plain_fri"

    def __init__(self, row_width: int, **_: Any):
        self.row_width = row_width

    def commit_rows(self, rows: List[FieldRow]) -> RowCommitment:
        leaves = [_hash_row(idx, row) for idx, row in enumerate(rows)]
        levels = build_levels(leaves)
        root = root_from_levels(levels)
        params = {
            "root": root.hex(),
            "num_rows": len(rows),
        }
        return RowCommitment(
            backend=self.name,
            row_width=self.row_width,
            params=params,
            prover_state={"rows": rows, "levels": levels},
        )

    def open_row(self, commitment: RowCommitment, idx: int) -> tuple[FieldRow, Dict[str, Any]]:
        state = commitment.prover_state
        if state is None:
            raise ValueError("prover state missing for row openings")
        rows = state["rows"]
        levels = state["levels"]
        if idx < 0 or idx >= len(rows):
            raise IndexError("row index out of range")
        row = rows[idx]
        path = [hx.hex() for hx in merkle_prove(levels, idx)]
        return list(row), {"path": path}

    def verify_leaf(
        self,
        commitment: RowCommitment,
        idx: int,
        row_values: FieldRow,
        proof: Dict[str, Any],
    ) -> bool:
        if idx < 0 or idx >= int(commitment.params.get("num_rows", 0)):
            return False
        if len(row_values) != commitment.row_width:
            return False
        root = bytes.fromhex(commitment.params["root"])
        path = [bytes.fromhex(x) for x in proof.get("path", [])]
        leaf = _hash_row(idx, row_values)
        return merkle_verify(root, leaf, idx, path)

    def streaming_init(self) -> Dict[str, Any]:
        return {
            "rows": [],
            "leaves": [],
        }

    def streaming_append(self, state: Dict[str, Any], row: FieldRow) -> None:
        if len(row) != self.row_width:
            raise ValueError("row width mismatch")
        idx = len(state["rows"])
        state["rows"].append(list(row))
        state["leaves"].append(_hash_row(idx, row))

    def streaming_finalize(self, state: Dict[str, Any]) -> RowCommitment:
        rows: List[FieldRow] = state.get("rows", [])
        leaves: List[bytes] = state.get("leaves", [])
        if len(leaves) != len(rows):
            leaves = [_hash_row(idx, row) for idx, row in enumerate(rows)]
        levels = build_levels(leaves)
        root = root_from_levels(levels)
        params = {
            "root": root.hex(),
            "num_rows": len(rows),
        }
        # Store copies so future openings can be derived.
        stored_rows = [list(row) for row in rows]
        commitment = RowCommitment(
            backend=self.name,
            row_width=self.row_width,
            params=params,
            prover_state={"rows": stored_rows, "levels": levels},
        )
        return commitment
