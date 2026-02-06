"""GPU-accelerated streaming accumulator for temporal sketches.

This module refactors the original notebook-based prototype into a reusable
component.  It exposes helpers to build the CUDA extension, construct the
`r^i` table on device, and evaluate chunked sketches Σ v[i]·r^i mod P with a
single pass.
"""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.cpp_extension import load as load_extension

DEFAULT_MODULUS = (1 << 61) - 1
DEFAULT_NUM_CHALLENGES = 4

# Paths to the CUDA/C++ sources we just extracted from the notebook.
_THIS_DIR = Path(__file__).resolve().parent
_CUDA_SRC = [
    _THIS_DIR / "cuda" / "stream_sketch_extension.cpp",
    _THIS_DIR / "cuda" / "stream_sketch_kernel.cu",
]

_EXT_MOD = None


def _ensure_cuda_extension(modulus: int) -> Any:
    """Compile/load the CUDA extension on demand."""
    global _EXT_MOD
    if _EXT_MOD is not None:
        return _EXT_MOD
    if not torch.cuda.is_available():
        raise RuntimeError("StreamingAccumulatorCUDA requires a CUDA-capable GPU")

    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}+PTX")
    _EXT_MOD = load_extension(
        name="stream_sketch_cuda",
        sources=[str(src) for src in _CUDA_SRC],
        extra_cuda_cflags=[
            "-O3",
            f"-arch=sm_{major}{minor}",
            f"-DMODULUS={modulus}ULL",
        ],
        extra_cflags=["-O3", f"-DMODULUS={modulus}ULL"],
        verbose=False,
    )
    return _EXT_MOD


def int_to_bytes(x: int) -> bytes:
    return x.to_bytes(32, "big")


def hash_pair(a: bytes, b: bytes) -> bytes:
    return hashlib.sha256(a + b).digest()


def merkle_build(leaves: Sequence[int], modulus: int) -> bytes:
    if not leaves:
        raise ValueError("cannot build Merkle tree for empty sequence")
    level = [hashlib.sha256(int_to_bytes(int(v) % modulus)).digest() for v in leaves]
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(hash_pair(left, right))
        level = nxt
    return level[0]


def _initial_commitment_root() -> bytes:
    return hashlib.sha256(b"bef-init").digest()


def _commitment_root_update(root: bytes, offset: int, chunk_root: bytes) -> bytes:
    h = hashlib.sha256()
    h.update(root)
    h.update(offset.to_bytes(8, "big"))
    h.update(chunk_root)
    return h.digest()


def build_rpow_gpu(r: int, length: int, modulus: int, tile: int = 1024) -> torch.Tensor:
    """Build [1, r, r^2, ..., r^{length-1}] mod P on the GPU via tiled blocks."""
    if length <= 0:
        raise ValueError("length must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for build_rpow_gpu")

    ext = _ensure_cuda_extension(modulus)
    tile = min(tile, length)
    num_tiles = (length + tile - 1) // tile

    base = [1] * tile
    for i in range(1, tile):
        base[i] = (base[i - 1] * r) % modulus
    base_t = torch.tensor(base, dtype=torch.int64, device="cuda")

    anchors = [0] * num_tiles
    step = pow(r, tile, modulus)
    prod = 1
    for j in range(num_tiles):
        anchors[j] = prod
        prod = (prod * step) % modulus
    anchors_t = torch.tensor(anchors, dtype=torch.int64, device="cuda")

    out = torch.empty(length, dtype=torch.int64, device="cuda")
    ext.fill_rpow_blocks(base_t, anchors_t, out, tile)
    return out


def chunk_dot_cuda(
    values: torch.Tensor,
    r_powers: torch.Tensor,
    modulus: int,
    block_size: int = 512,
    max_blocks: int = 131_072,
) -> int:
    """Compute Σ values[i]·r_powers[i] mod P for a chunk on the GPU."""
    if values.numel() != r_powers.numel():
        raise ValueError("value and r^i slices must have the same length")
    if values.dtype != torch.int64 or r_powers.dtype != torch.int64:
        raise TypeError("chunk_dot_cuda expects int64 tensors")
    if values.device.type != "cuda" or r_powers.device.type != "cuda":
        raise RuntimeError("chunk_dot_cuda requires CUDA tensors")

    ext = _ensure_cuda_extension(modulus)
    out = ext.dot_mod_p(values.contiguous(), r_powers.contiguous(), int(block_size), int(max_blocks))
    return int(out.sum().item() % modulus)


@dataclass
class Chunk:
    offset: int
    values: torch.Tensor
    root: bytes


class StreamingAccumulatorCUDA:
    """One-pass streaming accumulator that emits per-chunk sketches."""

    def __init__(
        self,
        *,
        seed: int = 0xC0FFEE,
        modulus: int = DEFAULT_MODULUS,
        num_challenges: int = DEFAULT_NUM_CHALLENGES,
        tile_size: int = 1024,
        use_fused: bool = True,
    ):
        self.seed = seed
        self.modulus = modulus
        self.num_challenges = num_challenges
        self.tile_size = tile_size
        self.use_fused = use_fused
        self.length = 0
        self.chunks: List[Chunk] = []

    def _challenge(self, transcript: bytes) -> int:
        h = hashlib.sha256(transcript).digest()
        return int.from_bytes(h, "big") % self.modulus or 1

    def _derive_challenges(self, transcript: bytes) -> List[int]:
        challenges: List[int] = []
        counter = 0
        while len(challenges) < self.num_challenges:
            h = hashlib.sha256(transcript + counter.to_bytes(4, "big")).digest()
            r = int.from_bytes(h, "big") % self.modulus
            if r != 0:
                challenges.append(r)
            counter += 1
        return challenges

    def add_chunk(self, values: Sequence[int]) -> Chunk:
        reduced = [int(v) % self.modulus for v in values]
        root = merkle_build(reduced, self.modulus)
        tensor = torch.tensor(reduced, dtype=torch.int64, device="cuda")
        chunk = Chunk(offset=self.length, values=tensor, root=root)
        self.chunks.append(chunk)
        self.length += tensor.numel()
        return chunk

    @torch.inference_mode()
    def prove(self) -> Dict[str, Any]:
        commitment_root = _initial_commitment_root()
        chunk_records: List[Dict[str, Any]] = []
        for chunk in self.chunks:
            chunk_records.append(
                {
                    "root": chunk.root.hex(),
                    "offset": chunk.offset,
                    "length": chunk.values.numel(),
                    "sketch_vec": [0] * self.num_challenges,
                }
            )
            commitment_root = _commitment_root_update(commitment_root, chunk.offset, chunk.root)

        challenges = self._derive_challenges(commitment_root)
        global_vec = [0] * self.num_challenges

        if self.use_fused:
            if not self.chunks:
                raise RuntimeError("cannot produce sketch with zero chunks")
            ext = _ensure_cuda_extension(self.modulus)
            r_tensor = torch.tensor(challenges, dtype=torch.int64, device="cuda")

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if len(self.chunks) == 1:
                vals_all = self.chunks[0].values
            else:
                vals_all = torch.cat([chunk.values for chunk in self.chunks], dim=0)
            fused_global = ext.fused_sketch(vals_all.contiguous(), r_tensor, 512, 131_072)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            fused_global_ms = (t1 - t0) * 1e3

            torch.cuda.synchronize()
            t2 = time.perf_counter()
            for idx, chunk in enumerate(self.chunks):
                chunk_vec = ext.fused_sketch(chunk.values.contiguous(), r_tensor, 512, 131_072)
                adjusted = []
                for j, val in enumerate(chunk_vec):
                    contrib = int(val.item() % self.modulus)
                    shift = pow(challenges[j], chunk.offset, self.modulus)
                    adjusted.append((contrib * shift) % self.modulus)
                chunk_records[idx]["sketch_vec"] = adjusted
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            fused_chunks_ms = (t3 - t2) * 1e3

            global_vec = [int(x.item() % self.modulus) for x in fused_global]

            trace_commitment = {
                "len": self.length,
                "root_hex": commitment_root.hex(),
                "challenges": challenges,
                "sketches": global_vec,
            }

            return {
                "length": self.length,
                "r": challenges[0],
                "global_sketch": global_vec[0],
                "challenges": challenges,
                "global_sketch_vec": global_vec,
                "chunks": chunk_records,
                "cuda_rpow_ms": fused_global_ms,
                "cuda_chunks_ms": fused_chunks_ms,
                "cuda_fused_global_ms": fused_global_ms,
                "cuda_fused_chunks_ms": fused_chunks_ms,
                "commitment_root": commitment_root.hex(),
                "trace_commitment": trace_commitment,
            }

        # Legacy unfused path (build r^i per challenge)
        rpow_build_ms = 0.0
        accumulation_ms = 0.0

        for j, r in enumerate(challenges):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            rpow = build_rpow_gpu(r, self.length, self.modulus, tile=self.tile_size)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            rpow_build_ms += (t1 - t0) * 1e3

            torch.cuda.synchronize()
            t2 = time.perf_counter()
            for idx, chunk in enumerate(self.chunks):
                start = chunk.offset
                end = start + chunk.values.numel()
                sketch = chunk_dot_cuda(
                    chunk.values,
                    rpow[start:end],
                    modulus=self.modulus,
                )
                chunk_records[idx]["sketch_vec"][j] = sketch
                global_vec[j] = (global_vec[j] + sketch) % self.modulus
            torch.cuda.synchronize()
            t3 = time.perf_counter()
            accumulation_ms += (t3 - t2) * 1e3

            trace_commitment = {
                "len": self.length,
                "root_hex": commitment_root.hex(),
                "challenges": challenges,
                "sketches": global_vec,
            }

            return {
                "length": self.length,
                "r": challenges[0],
                "global_sketch": global_vec[0],
                "challenges": challenges,
                "global_sketch_vec": global_vec,
                "chunks": chunk_records,
                "cuda_rpow_ms": rpow_build_ms,
                "cuda_chunks_ms": accumulation_ms,
                "commitment_root": commitment_root.hex(),
                "trace_commitment": trace_commitment,
            }


def demo_cuda(
    num_chunks: int = 8,
    chunk_len: int = 250_000,
    seed: int = 1234,
    num_challenges: int = DEFAULT_NUM_CHALLENGES,
) -> None:
    """Benchmark the CUDA accumulator against a CPU reference path."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime required for demo_cuda")

    random = torch.Generator().manual_seed(seed)
    acc = StreamingAccumulatorCUDA(seed=seed, num_challenges=num_challenges, use_fused=True)
    cpu_values: List[int] = []
    for _ in range(num_chunks):
        vals = torch.randint(low=0, high=DEFAULT_MODULUS, size=(chunk_len,), generator=random, dtype=torch.int64)
        cpu_values.extend(int(v) for v in vals.tolist())
        acc.add_chunk(vals.tolist())

    proof = acc.prove()
    transcript = b"".join(bytes.fromhex(ch["root"]) for ch in proof["chunks"]) + int_to_bytes(proof["length"])
    r = proof["challenges"][0]

    t0 = time.perf_counter()
    s_full = 0
    for i, v in enumerate(cpu_values):
        s_full = (s_full + (v % DEFAULT_MODULUS) * pow(r, i, DEFAULT_MODULUS)) % DEFAULT_MODULUS
    t1 = time.perf_counter()

    s_chunks = 0
    for ch in proof["chunks"]:
        s_chunks = (s_chunks + ch["sketch_vec"][0] % DEFAULT_MODULUS) % DEFAULT_MODULUS

    total = len(cpu_values)
    cpu_ms = (t1 - t0) * 1e3
    fused_global = proof.get("cuda_fused_global_ms", proof.get("cuda_rpow_ms", 0.0))
    fused_chunks = proof.get("cuda_fused_chunks_ms", proof.get("cuda_chunks_ms", 0.0))
    first_gpu_ms = fused_global + fused_chunks
    amort_gpu_ms = fused_chunks

    print("\nStreaming Accumulator (CUDA) — End-to-End Sketch")
    print(f"GPU: {torch.cuda.get_device_name()} (sm_{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]})")
    print(f"workload: total={total:,} elems  chunks={len(proof['chunks'])}  chunk_len={chunk_len:,}")
    print(f"challenges: {proof['challenges']}")
    if "cuda_fused_global_ms" in proof:
        print(f"timing:   fused global           = {proof['cuda_fused_global_ms']:8.2f} ms")
        print(f"          fused per-chunk        = {proof['cuda_fused_chunks_ms']:8.2f} ms")
    else:
        print(f"timing:   r^i build (one-time)   = {proof['cuda_rpow_ms']:8.2f} ms")
        print(f"          chunk sketches         = {proof['cuda_chunks_ms']:8.2f} ms")
    print(f"          CPU full sketch (ref.)= {cpu_ms:8.2f} ms")
    print(f"check:    GPU vs CPU equality → {'OK' if s_full == s_chunks else 'FAIL'}")
    print(f"speedup (first audit):  {cpu_ms / first_gpu_ms if first_gpu_ms else float('inf'):6.1f}×")
    print(f"speedup (amortized):    {cpu_ms / amort_gpu_ms if amort_gpu_ms else float('inf'):6.1f}×")


__all__ = [
    "DEFAULT_MODULUS",
    "DEFAULT_NUM_CHALLENGES",
    "StreamingAccumulatorCUDA",
    "Chunk",
    "build_rpow_gpu",
    "chunk_dot_cuda",
    "demo_cuda",
]

if __name__ == "__main__":
    demo_cuda()
