#!/usr/bin/env python3
"""CPU-only STC accumulator + AoK-style helper CLI."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import secrets
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bef_zk.stc.merkle import build_levels, prove as merkle_prove, root_from_levels, verify as merkle_verify
from bef_zk.stc.archive import ChunkArchive

MODULUS = (1 << 61) - 1
ROOT_SEED = b"bef-init"
DEFAULT_DA_MODE = "LOCAL_FILE"


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _int_to_bytes(value: int, length: int = 32) -> bytes:
    return value.to_bytes(length, "big", signed=False)


def derive_challenges(root: bytes, m: int) -> List[int]:
    challenges: List[int] = []
    counter = 0
    while len(challenges) < m:
        h = hashlib.sha256(root + counter.to_bytes(4, "big")).digest()
        candidate = int.from_bytes(h, "big") % MODULUS
        if candidate != 0:
            challenges.append(candidate)
        counter += 1
    return challenges


def merkle_from_values(values: Sequence[int], offset: int) -> bytes:
    leaves = [
        _hash(offset.to_bytes(8, "big") + idx.to_bytes(8, "big") + _int_to_bytes(val))
        for idx, val in enumerate(values)
    ]
    if not leaves:
        raise ValueError("chunk must have at least one value")
    level = leaves
    while len(level) > 1:
        nxt: List[bytes] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(_hash(left + right))
        level = nxt
    return level[0]


def hash_root_update(root: bytes, offset: int, chunk_root: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(root)
    h.update(offset.to_bytes(8, "big"))
    h.update(chunk_root)
    return h.digest()


def pow_mul(base: int, exp: int) -> int:
    return pow(base % MODULUS, exp, MODULUS)


def chunk_sketch(values: Sequence[int], offset: int, challenges: Sequence[int]) -> List[int]:
    sketches: List[int] = []
    for r in challenges:
        acc = 0
        pow_r = pow_mul(r, offset)
        for val in values:
            acc = (acc + (val % MODULUS) * pow_r) % MODULUS
            pow_r = (pow_r * r) % MODULUS
        sketches.append(acc)
    return sketches


def chunk_leaf_hash(offset: int, local_idx: int, value: int) -> bytes:
    return _hash(
        offset.to_bytes(8, "big")
        + local_idx.to_bytes(8, "big")
        + _int_to_bytes(int(value) % MODULUS)
    )


@dataclass
class STCCommitment:
    length: int
    chunk_len: int
    num_chunks: int
    global_root: bytes
    chunk_roots: List[bytes] = field(default_factory=list)

    def to_json(self, include_chunk_roots: bool = False) -> Dict:
        data = {
            "schema": "bef_pc_commit_v1",
            "length": self.length,
            "chunk_len": self.chunk_len,
            "num_chunks": self.num_chunks,
            "global_root": self.global_root.hex(),
        }
        if include_chunk_roots:
            data["chunk_roots"] = [cr.hex() for cr in self.chunk_roots]
        return data

    @classmethod
    def from_json(cls, data: Dict) -> STCCommitment:
        if data.get("schema") != "bef_pc_commit_v1":
            raise ValueError("unexpected schema for PC commitment")
        roots = [bytes.fromhex(x) for x in data.get("chunk_roots", [])]
        return cls(
            length=int(data["length"]),
            chunk_len=int(data["chunk_len"]),
            num_chunks=int(data["num_chunks"]),
            global_root=bytes.fromhex(data["global_root"]),
            chunk_roots=roots,
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
    chunk_root_path: List[bytes]

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
            "chunk_root_path": [hx.hex() for hx in self.chunk_root_path],
        }

    @classmethod
    def from_json(cls, data: Dict) -> STCIndexProof:
        if data.get("schema") != "bef_pc_open_v1":
            raise ValueError("unexpected schema for PC opening")
        return cls(
            index=int(data["index"]),
            value=int(data["value"]),
            chunk_index=int(data["chunk_index"]),
            chunk_offset=int(data["chunk_offset"]),
            leaf_pos=int(data["leaf_pos"]),
            leaf_path=[bytes.fromhex(x) for x in data.get("leaf_path", [])],
            chunk_root=bytes.fromhex(data["chunk_root"]),
            chunk_pos=int(data["chunk_pos"]),
            chunk_root_path=[bytes.fromhex(x) for x in data.get("chunk_root_path", [])],
        )


def global_sketch(chunks: Sequence[Dict], challenges: Sequence[int]) -> List[int]:
    totals = [0 for _ in challenges]
    for chunk in chunks:
        for idx, value in enumerate(chunk["sketch_vec"]):
            totals[idx] = (totals[idx] + value) % MODULUS
    return totals


def verify_fast(sketch: Dict) -> bool:
    commitment = sketch.get("trace_commitment")
    if not commitment:
        return False
    chunks = sketch.get("chunks", [])
    if sketch.get("length", 0) == 0:
        return not chunks
    sorted_chunks = sorted(chunks, key=lambda c: c["offset"])
    total = 0
    for idx, chunk in enumerate(sorted_chunks):
        if chunk["chunk_index"] != idx:
            return False
        if chunk["offset"] != total:
            return False
        if chunk["length"] <= 0:
            return False
        total += chunk["length"]
        if len(chunk["sketch_vec"]) != len(commitment["challenges"]):
            return False
    if total != sketch.get("length"):
        return False

    root = ROOT_SEED
    total = 0
    for chunk in sorted_chunks:
        root = hash_root_update(root, chunk["offset"], bytes.fromhex(chunk["root_hex"]))
        total += chunk["length"]
    if root.hex() != sketch.get("commitment_root"):
        return False

    expected = global_sketch(sorted_chunks, commitment["challenges"])
    if expected != commitment["sketches"]:
        return False
    return True


def required_samples(delta: float, epsilon: float) -> int:
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    if not (0 < epsilon < 1):
        raise ValueError("epsilon must be in (0,1)")
    import math as _math
    return max(1, int(_math.ceil(_math.log(epsilon) / _math.log(1 - delta))))


def build_da_profile(mode: str, delta: float, epsilon: float, provider_hint: Optional[Dict[str, object]]) -> Dict[str, object]:
    k_min = required_samples(delta, epsilon)
    profile = {
        "version": 1,
        "mode": mode.upper(),
        "sampling": {
            "delta": delta,
            "epsilon": epsilon,
            "k_min": k_min,
        },
    }
    if provider_hint:
        profile["provider"] = provider_hint
    return profile


@dataclasses.dataclass
class ChunkRecord:
    values: List[int]
    offset: int
    length: int
    root: bytes
    sketch_vec: List[int]
    archive_handle: Optional[str] = None


class StreamingAccumulatorCPU:
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
        chunk_sketches: List[int] = []
        for idx, r in enumerate(self.challenges):
            pow_r = self.powers[idx]
            acc = 0
            for val in reduced:
                acc = (acc + pow_r * val) % MODULUS
                pow_r = (pow_r * r) % MODULUS
            self.powers[idx] = pow_r
            self.sketches[idx] = (self.sketches[idx] + acc) % MODULUS
            chunk_sketches.append(acc)
        return chunk_sketches

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

    def prove(self) -> Dict:
        if not self.chunks:
            raise ValueError("no chunks added")
        self._ensure_challenges()
        trace_commitment = {
            "len": self.length,
            "root_hex": self.root.hex(),
            "challenges": list(self.challenges),
            "sketches": [int(x) % MODULUS for x in self.sketches],
        }
        return {
            "schema": "bef_sketch_v1",
            "length": self.length,
            "commitment_root": self.root.hex(),
            "trace_commitment": trace_commitment,
            "challenges": list(self.challenges),
            "global_sketch_vec": trace_commitment["sketches"],
            "archive_root": str(self.archive.root) if self.archive else None,
            "chunks": [
                {
                    "chunk_index": idx,
                    "offset": chunk.offset,
                    "length": chunk.length,
                    "root_hex": chunk.root.hex(),
                    "sketch_vec": chunk.sketch_vec,
                    "archive_handle": chunk.archive_handle,
                }
                for idx, chunk in enumerate(self.chunks)
            ],
        }


def generate_random_trace(length: int, *, seed: int) -> Dict:
    rng = secrets.SystemRandom(seed)
    values = [rng.randrange(0, MODULUS) for _ in range(length)]
    return {
        "schema": "bef_trace_v1",
        "trace_id": f"random_{length}",
        "field_modulus": MODULUS,
        "num_steps": length,
        "vector_length": length,
        "chunk_length": length,
        "chunks": [
            {"chunk_index": idx, "offset": idx, "values": [value]}
            for idx, value in enumerate(values)
        ],
    }


def load_trace(path: Path) -> Dict:
    data = json.loads(path.read_text())
    if data.get("schema") != "bef_trace_v1":
        raise ValueError("unexpected schema")
    return data


def prove_trace(
    trace: Dict,
    *,
    num_challenges: int,
    chunk_len: int,
    da_profile: Optional[Dict[str, object]] = None,
    archive_dir: Optional[Path] = None,
) -> Dict:
    archive = ChunkArchive(archive_dir)
    acc = StreamingAccumulatorCPU(
        num_challenges=num_challenges,
        chunk_len=chunk_len,
        archive=archive,
    )
    for chunk in trace["chunks"]:
        if len(chunk["values"]) > chunk_len:
            raise ValueError("chunk exceeds chunk_len")
        acc.add_chunk(chunk["values"])
    sketch = acc.prove()
    sketch["trace_id"] = trace.get("trace_id", "trace")
    sketch["seed"] = trace.get("seed", 0)
    if da_profile:
        sketch["da_profile"] = da_profile
    return sketch


def stc_build_pc_commitment(acc: StreamingAccumulatorCPU) -> STCCommitment:
    chunk_roots = [chunk.root for chunk in acc.chunks]
    if not chunk_roots:
        raise ValueError("no chunks to commit")
    levels = build_levels(chunk_roots)
    commitment = STCCommitment(
        length=acc.length,
        chunk_len=acc.chunk_len,
        num_chunks=len(acc.chunks),
        global_root=root_from_levels(levels),
        chunk_roots=chunk_roots,
    )
    commitment._chunk_root_levels = levels  # type: ignore[attr-defined]
    return commitment


def stc_open_index(
    values: Sequence[int],
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
    chunk_offset = chunk_index * chunk_len
    local_idx = index - chunk_offset
    chunk_values = list(values[chunk_offset: chunk_offset + chunk_len])
    if not chunk_values:
        raise ValueError("missing chunk values for opening")
    leaves = [chunk_leaf_hash(chunk_offset, j, val) for j, val in enumerate(chunk_values)]
    chunk_levels = build_levels(leaves)
    leaf_path = merkle_prove(chunk_levels, local_idx)
    chunk_root = root_from_levels(chunk_levels)
    if chunk_root_levels is None:
        if not hasattr(commitment, "_chunk_root_levels") and not commitment.chunk_roots:
            raise ValueError("chunk root data unavailable; provide levels explicitly")
        if not hasattr(commitment, "_chunk_root_levels"):
            commitment._chunk_root_levels = build_levels(commitment.chunk_roots)  # type: ignore[attr-defined]
        chunk_root_levels = commitment._chunk_root_levels  # type: ignore[attr-defined]
    chunk_root_path = merkle_prove(chunk_root_levels, chunk_index)
    return STCIndexProof(
        index=index,
        value=int(values[index]) % MODULUS,
        chunk_index=chunk_index,
        chunk_offset=chunk_offset,
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
    if not merkle_verify(commitment.global_root, proof.chunk_root, proof.chunk_pos, proof.chunk_root_path):
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="STC AoK helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("prove-trace", help="Produce sketch from bef_trace_v1")
    pr.add_argument("trace", type=Path)
    pr.add_argument("output", type=Path)
    pr.add_argument("--num-challenges", type=int, default=4)
    pr.add_argument("--chunk-len", type=int, default=8192)
    pr.add_argument("--archive-dir", type=Path, help="Directory to persist chunk archive")
    pr.add_argument(
        "--da-mode",
        default=DEFAULT_DA_MODE,
        choices=["LOCAL_FILE", "LIGHT_SAMPLING", "EXTERNAL_DA", "COMMITTEE", "NONE"],
        help="Data availability mode to embed",
    )
    pr.add_argument("--da-delta", type=float, default=0.1, help="Minimum withholding fraction detectable")
    pr.add_argument("--da-epsilon", type=float, default=1e-6, help="Failure probability target for sampling")
    pr.add_argument("--da-provider", help="Provider hint (JSON string or path for LOCAL_FILE)")

    vr = sub.add_parser("verify", help="Verify bef_sketch_v1 JSON")
    vr.add_argument("sketch", type=Path)

    args = parser.parse_args()

    if args.cmd == "prove-trace":
        trace = load_trace(args.trace)
        archive_dir = args.archive_dir
        if archive_dir is None:
            archive_dir = args.output.parent / f"{args.output.stem}_chunks"
        archive_dir = archive_dir.resolve()
        if archive_dir.exists():
            shutil.rmtree(archive_dir)
        archive_dir.mkdir(parents=True, exist_ok=True)

        provider_hint: Optional[Dict[str, object]] = None
        mode = args.da_mode.upper()
        if mode != "NONE":
            if args.da_provider:
                provider_arg = args.da_provider.strip()
                try:
                    provider_hint = json.loads(provider_arg)
                except json.JSONDecodeError:
                    provider_hint = {"hint": provider_arg}
            elif mode == "LOCAL_FILE":
                provider_hint = {"path": str(args.trace)}
            elif mode == "LIGHT_SAMPLING":
                provider_hint = {"archive_root": str(archive_dir)}
            else:
                raise SystemExit("da-provider required for mode %s" % mode)
            da_profile = build_da_profile(mode, args.da_delta, args.da_epsilon, provider_hint)
        else:
            da_profile = None
        sketch = prove_trace(
            trace,
            num_challenges=args.num_challenges,
            chunk_len=args.chunk_len,
            da_profile=da_profile,
            archive_dir=archive_dir,
        )
        args.output.write_text(json.dumps(sketch, indent=2))
        print(f"wrote sketch to {args.output}")
    elif args.cmd == "verify":
        sketch = json.loads(args.sketch.read_text())
        if verify_fast(sketch):
            print("verify_fast: OK")
        else:
            raise SystemExit("verify_fast: FAILED")


if __name__ == "__main__":
    main()
