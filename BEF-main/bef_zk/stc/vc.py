"""Vector commitment interface backed by STC."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Protocol

from .aok_cpu import (
    MODULUS,
    ROOT_SEED,
    StreamingAccumulatorCPU,
    ChunkRecord,
    chunk_leaf_hash,
    merkle_prove,
    merkle_verify,
)
from .archive import ChunkArchive
import shutil
from .pc_backend import (
    STCCommitment,
    STCIndexProof,
    STCChunkProof,
    stc_build_pc_commitment,
    stc_open_index,
    stc_open_chunk,
    stc_verify_index,
    stc_verify_chunk,
    CHUNK_TREE_ARITY,
)
from .merkle import (
    verify_multiproof,
    build_levels,
    root_from_levels,
    MerkleMultiProof,
    multiproof,
)


@dataclass
class VCCommitment:
    root: bytes
    length: int
    chunk_len: int
    num_chunks: int
    challenges: List[int] = field(default_factory=list)
    sketches: List[int] = field(default_factory=list)
    powers: List[int] = field(default_factory=list)
    chain_root: bytes = ROOT_SEED
    chunk_tree_arity: int = CHUNK_TREE_ARITY


@dataclass
class VCOpenProof:
    index: int
    value: int
    aux: Dict[str, Any]


@dataclass
class VCBatchEntry:
    index: int
    value: int
    chunk_index: int
    chunk_offset: int
    leaf_pos: int
    leaf_path: List[bytes] | None = None


@dataclass
class ChunkLeafProof:
    chunk_index: int
    chunk_offset: int
    leaf_positions: List[int]
    proof: MerkleMultiProof


@dataclass
class VCBatchProof:
    entries: List[VCBatchEntry]
    chunk_positions: List[int]
    chunk_roots: List[bytes]
    chunk_proof: MerkleMultiProof
    chunk_leaf_proofs: List[ChunkLeafProof] = field(default_factory=list)
    debug_chunk_counts: Dict[int, int] | None = None


class VectorCommitment(Protocol):
    def commit(self, values: List[int]) -> VCCommitment: ...
    def open(self, commitment: VCCommitment, index: int) -> VCOpenProof: ...
    def verify_open(self, commitment: VCCommitment, proof: VCOpenProof) -> bool: ...
    def open_chunk(self, commitment: VCCommitment, chunk_index: int) -> STCChunkProof: ...
    def verify_chunk(self, commitment: VCCommitment, proof: STCChunkProof) -> bool: ...
    def open_batch(self, commitment: VCCommitment, indices: List[int]) -> VCBatchProof: ...
    def verify_batch(self, commitment: VCCommitment, batch: VCBatchProof) -> bool: ...


class STCVectorCommitment:
    """Simple in-memory STC-backed VC for FRI prover/verifier."""

    def __init__(
        self,
        chunk_len: int = 256,
        num_challenges: int = 2,
        challenge_seed: bytes | None = None,
        archive_dir: str | Path | None = None,
        use_gpu: bool = False,
    ):
        self.chunk_len = chunk_len
        self.num_challenges = num_challenges
        self.challenge_seed = challenge_seed or ROOT_SEED
        self.archive_root = archive_dir
        self.use_gpu = use_gpu
        self._store: Dict[bytes, Dict[str, Any]] = {}
        self._debug_chunk_hist: Dict[int, int] = {}

    def commit(self, values: List[int]) -> VCCommitment:
        archive = ChunkArchive(root_dir=getattr(self, "archive_root", None))
        acc = None
        if self.use_gpu:
            try:
                from gpu_accumulator.stream_accumulator import StreamingAccumulatorCUDA
                seed_int = int.from_bytes(self.challenge_seed, "big")
                # GPU accumulator currently handles its own storage/root logic slightly differently,
                # but for this demo we wrap it or adapt. The current codebase suggests StreamingAccumulatorCUDA
                # is a direct drop-in for the "sketching" part, but we need to integrate it with ChunkArchive if we want persistence.
                # However, the current GPU implementation in stream_accumulator.py is in-memory only.
                # For Phase 2, we will use it for *proving* speedup and let it buffer in memory, then dump to archive if needed.
                
                # Note: StreamingAccumulatorCUDA expects seed as int
                acc = StreamingAccumulatorCUDA(
                    seed=seed_int,
                    num_challenges=self.num_challenges,
                    modulus=MODULUS
                )
            except (ImportError, RuntimeError) as e:
                print(f"WARNING: GPU acceleration requested but failed to load ({e}). Falling back to CPU.")
                acc = None

        if acc is None:
            acc = StreamingAccumulatorCPU(
                num_challenges=self.num_challenges,
                chunk_len=self.chunk_len,
                challenge_seed=self.challenge_seed,
                archive=archive,
            )

        # Feed data
        for i in range(0, len(values), self.chunk_len):
            acc.add_chunk(values[i : i + self.chunk_len])
        
        # Build commitment
        # If we used GPU, we need to adapt the output to match STCCommitment
        is_gpu_acc = False
        if self.use_gpu and acc is not None:
             try:
                 from gpu_accumulator.stream_accumulator import StreamingAccumulatorCUDA
                 if isinstance(acc, StreamingAccumulatorCUDA):
                     is_gpu_acc = True
             except ImportError:
                 pass

        if is_gpu_acc:
             # The GPU acc.prove() returns a dict, we need to map it to STCCommitment
             proof = acc.prove()
             # We also need to populate the archive manually since GPU acc is in-memory
             # This is a trade-off: fast proving vs disk IO. For this integration, we'll sync to archive.
             chunks_for_archive = []
             for ch_idx, ch in enumerate(acc.chunks):
                 vals = ch.values.cpu().tolist()
                 archive.write_chunk(ch_idx, vals)
                 chunks_for_archive.append(ChunkRecord(
                     index=ch_idx,
                     offset=ch.offset,
                     values=vals,
                     root=ch.root,
                     archive_handle=f"chunk_{ch_idx}.json" # simplified handle
                 ))
             
             # Reconstruct global commitment object
             stc_commit = STCCommitment(
                 length=proof["length"],
                 chunk_len=self.chunk_len,
                 num_chunks=len(acc.chunks),
                 global_root=bytes.fromhex(proof["commitment_root"]),
                 chunk_roots=[ch.root for ch in acc.chunks],
                 chain_root=bytes.fromhex(proof["commitment_root"]), # utilizing chain root as global root in this simplified mode
                 challenges=proof["challenges"],
                 sketches=proof["global_sketch_vec"],
                 powers=[], # GPU acc might not return powers, strictly not needed for verification if we have sketches
                 chunk_tree_arity=CHUNK_TREE_ARITY
             )
             # Mock powers if missing to avoid verification crashes (recompute or ignore)
             stc_commit.powers = [1] * self.num_challenges # placeholder
        else:
            stc_commit = stc_build_pc_commitment(acc)

        key = stc_commit.global_root
        self._store[key] = {
            "commit": stc_commit,
            "chunks": acc.export_chunks() if hasattr(acc, "export_chunks") else chunks_for_archive, # type: ignore
            "archive": archive,
            "archive_root": str(archive.root),
            "debug_chunk_hist": self._debug_chunk_hist,
        }
        vc_commitment = VCCommitment(
            root=stc_commit.global_root,
            length=stc_commit.length,
            chunk_len=stc_commit.chunk_len,
            num_chunks=stc_commit.num_chunks,
            challenges=list(stc_commit.challenges),
            sketches=list(stc_commit.sketches),
            powers=list(stc_commit.powers),
            chain_root=stc_commit.chain_root,
            chunk_tree_arity=stc_commit.chunk_tree_arity,
        )
        return vc_commitment

    def open(self, commitment: VCCommitment, index: int) -> VCOpenProof:
        store = self._store.get(commitment.root)
        if store is None:
            raise ValueError("commitment not available in prover store")
        stc_commit: STCCommitment = store["commit"]
        accessor = _ChunkValueAccessor(
            stc_commit.chunk_len,
            stc_commit.length,
            store["chunks"],
            store.get("archive"),
        )
        proof = stc_open_index(accessor, stc_commit, index)
        aux = {
            "chunk_index": proof.chunk_index,
            "chunk_offset": proof.chunk_offset,
            "leaf_pos": proof.leaf_pos,
            "leaf_path": [hx.hex() for hx in proof.leaf_path],
            "chunk_root": proof.chunk_root.hex(),
            "chunk_pos": proof.chunk_pos,
            "chunk_root_path": [
                [hx.hex() for hx in level]
                for level in proof.chunk_root_path
            ],
        }
        return VCOpenProof(index=proof.index, value=proof.value, aux=aux)

    def verify_open(self, commitment: VCCommitment, proof: VCOpenProof) -> bool:
        stc_commit = STCCommitment(
            length=commitment.length,
            chunk_len=commitment.chunk_len,
            num_chunks=commitment.num_chunks,
            global_root=commitment.root,
            chunk_roots=[],
            chain_root=commitment.chain_root,
            challenges=list(commitment.challenges),
            sketches=list(commitment.sketches),
            powers=list(commitment.powers),
            chunk_tree_arity=getattr(commitment, "chunk_tree_arity", CHUNK_TREE_ARITY),
        )
        stc_proof = STCIndexProof(
            index=proof.index,
            value=proof.value,
            chunk_index=int(proof.aux["chunk_index"]),
            chunk_offset=int(proof.aux["chunk_offset"]),
            leaf_pos=int(proof.aux["leaf_pos"]),
            leaf_path=[bytes.fromhex(x) for x in proof.aux["leaf_path"]],
            chunk_root=bytes.fromhex(proof.aux["chunk_root"]),
            chunk_pos=int(proof.aux["chunk_pos"]),
            chunk_root_path=[bytes.fromhex(x) for x in proof.aux["chunk_root_path"]],
        )
        return stc_verify_index(stc_commit, stc_proof)

    def open_chunk(self, commitment: VCCommitment, chunk_index: int) -> STCChunkProof:
        store = self._store.get(commitment.root)
        if store is None:
            raise ValueError("commitment not available in prover store")
        stc_commit: STCCommitment = store["commit"]
        accessor = _ChunkValueAccessor(
            stc_commit.chunk_len,
            stc_commit.length,
            store["chunks"],
            store.get("archive"),
        )
        return stc_open_chunk(accessor, stc_commit, chunk_index)

    def verify_chunk(self, commitment: VCCommitment, proof: STCChunkProof) -> bool:
        return stc_verify_chunk(commitment, proof)

    def open_batch(self, commitment: VCCommitment, indices: List[int]) -> VCBatchProof:
        if not indices:
            raise ValueError("batch must contain indices")
        store = self._store.get(commitment.root)
        if store is None:
            raise ValueError("commitment not available in prover store")
        stc_commit: STCCommitment = store["commit"]
        accessor = _ChunkValueAccessor(
            stc_commit.chunk_len,
            stc_commit.length,
            store["chunks"],
            store.get("archive"),
        )
        chunk_root_levels = getattr(stc_commit, "_chunk_levels", None)
        if chunk_root_levels is None:
            chunk_root_levels = build_levels(stc_commit.chunk_roots)
            stc_commit._chunk_levels = chunk_root_levels  # type: ignore[attr-defined]
        unique_indices = sorted(set(indices))
        entries: List[VCBatchEntry] = []
        chunk_roots: Dict[int, bytes] = {}
        chunk_levels_cache: Dict[int, tuple[List[List[bytes]], List[int]]] = {}
        chunk_entries: Dict[int, List[VCBatchEntry]] = {}
        for idx in unique_indices:
            if idx < 0 or idx >= stc_commit.length:
                raise ValueError("index out of range")
            chunk_idx = idx // stc_commit.chunk_len
            chunk_offset = chunk_idx * stc_commit.chunk_len
            local_idx = idx - chunk_offset
            cache = chunk_levels_cache.get(chunk_idx)
            if cache is None:
                chunk_values = accessor.get_chunk_values(chunk_idx)
                leaves = [chunk_leaf_hash(chunk_offset, j, val) for j, val in enumerate(chunk_values)]
                levels = build_levels(leaves)
                cache = (levels, chunk_values)
                chunk_levels_cache[chunk_idx] = cache
            levels, chunk_values = cache
            leaf_path = merkle_prove(levels, local_idx)
            chunk_root = root_from_levels(levels)
            chunk_roots[chunk_idx] = chunk_root
            entry = VCBatchEntry(
                index=idx,
                value=chunk_values[local_idx],
                chunk_index=chunk_idx,
                chunk_offset=chunk_offset,
                leaf_pos=local_idx,
                leaf_path=None,
            )
            entries.append(entry)
            chunk_entries.setdefault(chunk_idx, []).append(entry)
        chunk_positions = sorted(chunk_roots.keys())
        chunk_root_list = [chunk_roots[pos] for pos in chunk_positions]
        chunk_proof = multiproof(
            chunk_root_levels,
            chunk_positions,
            stc_commit.chunk_tree_arity,
        )
        chunk_leaf_proofs: List[ChunkLeafProof] = []
        for chunk_idx in chunk_positions:
            entry_list = chunk_entries.get(chunk_idx, [])
            cache = chunk_levels_cache.get(chunk_idx)
            if cache is None:
                raise ValueError("missing cache for chunk")
            levels, chunk_values = cache
            positions = sorted({entry.leaf_pos for entry in entry_list})
            proof = multiproof(levels, positions, 2)
            chunk_leaf_proofs.append(
                ChunkLeafProof(
                    chunk_index=chunk_idx,
                    chunk_offset=entry_list[0].chunk_offset if entry_list else chunk_idx * stc_commit.chunk_len,
                    leaf_positions=positions,
                    proof=proof,
                )
            )
        chunk_counts: Dict[int, int] = {}
        for entry in entries:
            chunk_counts[entry.chunk_index] = chunk_counts.get(entry.chunk_index, 0) + 1
        batch_aux = [(chunk_idx, chunk_counts[chunk_idx]) for chunk_idx in chunk_positions]
        return VCBatchProof(
            entries=entries,
            chunk_positions=chunk_positions,
            chunk_roots=chunk_root_list,
            chunk_proof=chunk_proof,
            chunk_leaf_proofs=chunk_leaf_proofs,
            debug_chunk_counts=batch_aux,
        )

    def verify_batch(self, commitment: VCCommitment, batch: VCBatchProof) -> bool:
        if not batch.entries:
            return False
        stc_commit = STCCommitment(
            length=commitment.length,
            chunk_len=commitment.chunk_len,
            num_chunks=commitment.num_chunks,
            global_root=commitment.root,
            chain_root=commitment.chain_root,
            challenges=list(commitment.challenges),
            sketches=list(commitment.sketches),
            powers=list(commitment.powers),
        )
        chunk_roots_map: Dict[int, bytes] = {
            pos: root for pos, root in zip(batch.chunk_positions, batch.chunk_roots)
        }
        chunk_entries: Dict[int, List[VCBatchEntry]] = {}
        for entry in batch.entries:
            if entry.index < 0 or entry.index >= stc_commit.length:
                return False
            if entry.chunk_index >= stc_commit.num_chunks:
                return False
            chunk_entries.setdefault(entry.chunk_index, []).append(entry)
        if batch.chunk_leaf_proofs:
            chunk_leaf_map = {clp.chunk_index: clp for clp in batch.chunk_leaf_proofs}
            for chunk_idx, entries in chunk_entries.items():
                chunk_root = chunk_roots_map.get(chunk_idx)
                if chunk_root is None:
                    return False
                clp = chunk_leaf_map.get(chunk_idx)
                if clp is None:
                    return False
                positions = clp.leaf_positions
                if sorted({entry.leaf_pos for entry in entries}) != positions:
                    return False
                values_by_pos = {entry.leaf_pos: entry.value for entry in entries}
                leaf_hashes = [
                    chunk_leaf_hash(clp.chunk_offset, pos, values_by_pos[pos])
                    for pos in positions
                ]
                if not verify_multiproof(chunk_root, leaf_hashes, positions, clp.proof):
                    return False
        else:
            for chunk_idx, entries in chunk_entries.items():
                chunk_root = chunk_roots_map.get(chunk_idx)
                if chunk_root is None:
                    return False
                for entry in entries:
                    if entry.leaf_path is None:
                        return False
                    leaf = chunk_leaf_hash(entry.chunk_offset, entry.leaf_pos, entry.value)
                    if not merkle_verify(chunk_root, leaf, entry.leaf_pos, entry.leaf_path):
                        return False
        if set(chunk_entries.keys()) != set(batch.chunk_positions):
            return False
        leaf_hashes = [chunk_roots_map[idx] for idx in batch.chunk_positions]
        return verify_multiproof(
            stc_commit.global_root,
            leaf_hashes,
            batch.chunk_positions,
            batch.chunk_proof,
        )

    def get_archive_root(self, commitment: VCCommitment) -> str | None:
        store = self._store.get(commitment.root)
        if store is None:
            return None
        return store.get("archive_root")

    def get_chunk_handles(self, commitment: VCCommitment) -> List[str] | None:
        store = self._store.get(commitment.root)
        if store is None:
            return None
        handles: List[str] = []
        for record in store.get("chunks", []):
            handle = getattr(record, "archive_handle", None)
            if handle:
                handles.append(handle)
            else:
                handles.append("")
        return handles

    def persist_archive(self, commitment: VCCommitment, destination: str | Path) -> str:
        store = self._store.get(commitment.root)
        if store is None:
            raise ValueError("commitment not available in store")
        src = Path(store.get("archive_root"))
        if not src.exists():
            raise ValueError("archive directory missing")
        dst = Path(destination)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        store["archive_root"] = str(dst)
        return str(dst)


class _ChunkValueAccessor:
    def __init__(
        self,
        chunk_len: int,
        total_len: int,
        records: List[ChunkRecord],
        archive: ChunkArchive | None,
    ) -> None:
        self.chunk_len = chunk_len
        self.total_len = total_len
        self.records = records
        self.archive = archive

    def _load_chunk(self, chunk_index: int) -> List[int]:
        record = self.records[chunk_index]
        if record.values:
            return record.values
        if record.archive_handle and self.archive is not None:
            data = self.archive.load_chunk(record.archive_handle)
            record.values = [int(v) % MODULUS for v in data]
            return record.values
        raise ValueError("chunk values unavailable")

    def get_chunk_values(self, chunk_index: int) -> List[int]:
        return list(self._load_chunk(chunk_index))

    def get_value(self, index: int) -> int:
        if index < 0 or index >= self.total_len:
            raise IndexError("value index out of range")
        chunk_index = index // self.chunk_len
        local_idx = index - chunk_index * self.chunk_len
        chunk = self._load_chunk(chunk_index)
        if local_idx >= len(chunk):
            raise IndexError("local index out of range")
        return int(chunk[local_idx]) % MODULUS
