#!/usr/bin/env python3
"""Run DA sampling checks for an STC sketch."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional

from stc_aok import (
    chunk_sketch,
    load_trace,
    merkle_from_values,
    verify_fast,
)
from bef_zk.codec import canonical_decode, derive_capsule_seed
from bef_zk.stc.archive import safe_join
from scripts.artifact_manifest import load_manifest, encoding_for_path, resolve_manifest_path
from bef_zk.stc.merkle import build_kary_levels, root_from_levels


def _load_artifact(path: Path, encoding_id: str | None = None) -> dict:
    raw = path.read_bytes()
    enc = encoding_id
    if enc is None and path.suffix == ".bin":
        enc = "dag_cbor_canonical_v1"
    if enc == "dag_cbor_canonical_v1":
        data = canonical_decode(raw)
    else:
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            data = canonical_decode(raw)
    if not isinstance(data, dict):
        raise SystemExit("artifact must decode to an object")
    return data


def sample_chunk_indices(total: int, k: int, seed: int) -> List[int]:
    if total <= 0:
        return []
    if k <= 0:
        return []
    if k >= total:
        return list(range(total))
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    return indices[:k]


def check_local_file(sketch: dict, trace_path: Path, samples: List[int]) -> None:
    trace = load_trace(trace_path)
    trace_chunks = trace["chunks"]
    chunks = sketch["chunks"]
    if len(trace_chunks) != len(chunks):
        raise SystemExit("trace and sketch chunk counts differ")
    challenges = sketch["trace_commitment"]["challenges"]
    for idx in samples:
        chunk = chunks[idx]
        trace_chunk = trace_chunks[idx]
        if trace_chunk["offset"] != chunk["offset"] or trace_chunk["chunk_index"] != chunk["chunk_index"]:
            raise SystemExit(f"chunk {idx} offset/index mismatch")
        calc_root = merkle_from_values(trace_chunk["values"], trace_chunk["offset"])
        if calc_root.hex() != chunk["root_hex"]:
            raise SystemExit(f"chunk {idx} root mismatch")
        calc_sketch = chunk_sketch(trace_chunk["values"], trace_chunk["offset"], challenges)
        if calc_sketch != chunk["sketch_vec"]:
            raise SystemExit(f"chunk {idx} sketch mismatch")


def check_light_sampling(sketch: dict, provider: Dict[str, object], samples: List[int]) -> None:
    archive_root = provider.get("archive_root") or sketch.get("archive_root")
    if not archive_root:
        raise SystemExit("LIGHT_SAMPLING requires archive_root in profile or sketch")
    archive_root = Path(str(archive_root)).resolve()
    if not archive_root.exists():
        raise SystemExit(f"archive_root {archive_root} missing")
    challenges = sketch["trace_commitment"]["challenges"]
    withheld = set(int(x) for x in provider.get("withheld", []))
    chunks = sketch.get("chunks", [])
    for idx in samples:
        if idx in withheld:
            raise SystemExit(f"DA sampling detected withheld chunk {idx}")
        chunk = chunks[idx]
        handle = chunk.get("archive_handle")
        if not handle:
            raise SystemExit(f"chunk {idx} missing archive_handle; cannot retrieve")
        try:
            path = safe_join(archive_root, handle)
        except ValueError as e:
            raise SystemExit(f"chunk {idx} has unsafe path: {e}")
        if not path.exists():
            raise SystemExit(f"chunk {idx} file {path} missing")
        values = [int(v) for v in json.loads(path.read_text())]
        if chunk.get("offset") is None:
            raise SystemExit("chunk entry missing offset")
        calc_root = merkle_from_values(values, chunk["offset"])
        if calc_root.hex() != chunk["root_hex"]:
            raise SystemExit(f"chunk {idx} root mismatch")
        calc_sketch = chunk_sketch(values, chunk["offset"], challenges)
        if calc_sketch != chunk["sketch_vec"]:
            raise SystemExit(f"chunk {idx} sketch mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description="STC DA sampling (sketch or capsule)")
    parser.add_argument("artifact", type=Path, help="sketch.json or strategy_capsule.json")
    parser.add_argument("--seed", type=int, help="override deterministic seed")
    parser.add_argument("--extra-samples", type=int, default=0)
    args = parser.parse_args()

    base_dir = args.artifact.resolve().parent
    manifest = load_manifest(base_dir)
    encoding = encoding_for_path(manifest, base_dir, args.artifact.resolve()) if manifest else None
    data = _load_artifact(args.artifact, encoding)
    if data.get("schema") == "bef_capsule_v1":
        seed_value = args.seed if args.seed is not None else derive_seed_from_capsule(data)
        run_capsule_sampling(args.artifact, data, seed_value, args.extra_samples)
        return

    sketch = data
    if not verify_fast(sketch):
        raise SystemExit("verify_fast failed; cannot run DA sampling")
    profile = sketch.get("da_profile")
    if not profile:
        raise SystemExit("sketch missing da_profile")
    k = int(profile.get("sampling", {}).get("k_min", 0)) + max(args.extra_samples, 0)
    total_chunks = len(sketch.get("chunks", []))
    seed_value = args.seed if args.seed is not None else 0
    samples = sample_chunk_indices(total_chunks, k, seed_value)

    mode = profile.get("mode", "NONE")
    provider = profile.get("provider", {})
    if mode == "LOCAL_FILE":
        path = provider.get("path")
        if not path:
            raise SystemExit("LOCAL_FILE profile missing provider.path")
        check_local_file(sketch, Path(path), samples)
    elif mode == "LIGHT_SAMPLING":
        check_light_sampling(sketch, provider, samples)
    else:
        raise SystemExit(f"DA mode {mode} not supported by sampler")
    print(f"DA sampling OK ({len(samples)} chunks, mode {mode})")


def _resolve_path(base: Path, entry: str) -> Path:
    path = Path(entry)
    if not path.is_absolute():
        cand = (base / path)
        if cand.exists():
            return cand.resolve()
        repo_cand = Path.cwd() / path
        if repo_cand.exists():
            return repo_cand.resolve()
        return cand.resolve()
    return path


def _row_archive_info(
    capsule: dict,
    base: Path,
    manifest: dict | None,
) -> tuple[Path, List[str], List[str]]:
    geom = capsule.get("proofs", {}).get("geom", {})
    art = capsule.get("artifacts", {})
    info = capsule.get("row_archive") or geom.get("row_archive") or art.get("row_archive")
    if not info:
        raise SystemExit("capsule missing row_archive metadata")
    root = info.get("abs_path") or info.get("path")
    if not root:
        raise SystemExit("row_archive metadata missing path/abs_path")
    handles: List[str] = []
    for entry in info.get("chunk_handles", []) or []:
        if isinstance(entry, dict):
            uri = entry.get("uri") or entry.get("path") or entry.get("rel_path") or entry.get("abs_path")
            handles.append(str(uri) if uri else "")
        else:
            handles.append(str(entry))
    roots_inline = list(info.get("chunk_roots_hex", []))
    roots: List[str]
    if manifest:
        chunk_entry = manifest.get("chunk_roots", {})
        default_fmt = chunk_entry.get("default")
        fmt_entry = chunk_entry.get("formats", {}).get(default_fmt) if default_fmt else None
        if fmt_entry:
            resolved = resolve_manifest_path(base, fmt_entry["path"])
            enc = fmt_entry.get("encoding_id")
            if enc == "raw32_v1":
                data = Path(resolved).read_bytes()
                if len(data) % 32 != 0:
                    raise SystemExit("chunk_roots bin malformed")
                roots = [data[i : i + 32].hex() for i in range(0, len(data), 32)]
            else:
                roots = json.loads(resolved.read_text())
        elif roots_inline:
            roots = roots_inline
        else:
            roots = []
    elif roots_inline:
        roots = roots_inline
    else:
        roots_bin = info.get("chunk_roots_bin_abs") or info.get("chunk_roots_bin_path")
        if roots_bin:
            resolved = _resolve_path(base, str(roots_bin))
            data = resolved.read_bytes()
            if len(data) % 32 != 0:
                raise SystemExit("chunk_roots.bin malformed")
            roots = [data[i : i + 32].hex() for i in range(0, len(data), 32)]
        else:
            roots_path = info.get("chunk_roots_abs") or info.get("chunk_roots_path")
            if not roots_path:
                roots = []
            else:
                resolved = _resolve_path(base, str(roots_path))
                roots = json.loads(resolved.read_text())
    return _resolve_path(base, root), handles, roots


def run_capsule_sampling(path: Path, capsule: dict, seed: int, extra: int) -> None:
    base = path.parent
    manifest = load_manifest(base)
    da_policy = capsule.get("da_policy")
    da_profile = capsule.get("da_profile") if not da_policy else None
    if da_policy:
        provider = da_policy.get("provider") or {}
        mode = provider.get("mode", "LIGHT_SAMPLING")
        base_samples = int(da_policy.get("k_samples") or 0)
    elif da_profile:
        mode = da_profile.get("mode", "NONE")
        sampling_cfg = da_profile.get("sampling") or {}
        base_samples = int(sampling_cfg.get("k_min") or 0)
    else:
        raise SystemExit("capsule missing DA configuration")
    if mode not in {"LIGHT_SAMPLING", "LOCAL_FILE"}:
        raise SystemExit(f"DA mode {mode} not supported for capsules")

    geom_entry = capsule.get("proofs", {}).get("geom", {})
    geom_path = (
        geom_entry.get("path")
        or geom_entry.get("json_path")
        or geom_entry.get("bin_path")
    )
    if not geom_path:
        raise SystemExit("capsule missing geom proof path")
    geom_resolved = _resolve_path(base, geom_path)
    geom_enc = encoding_for_path(manifest, base, geom_resolved) if manifest else None
    geom_proof = _load_artifact(geom_resolved, geom_enc)
    row_commitment = geom_proof.get("row_commitment", {})
    params = row_commitment.get("params", {})
    chunk_len = int(params.get("chunk_len", 0))
    num_chunks = int(params.get("num_chunks", 0))
    archive_root, chunk_handles, chunk_roots_hex = _row_archive_info(capsule, base, manifest)
    if len(chunk_handles) != num_chunks:
        raise SystemExit("chunk_handles length mismatch")
    if len(chunk_roots_hex) != num_chunks:
        raise SystemExit("chunk_roots_hex length mismatch")
    if len(chunk_roots_hex) != num_chunks:
        raise SystemExit("chunk_roots_hex length mismatch")
    chunk_roots = [bytes.fromhex(h) for h in chunk_roots_hex]
    arity = int(params.get("chunk_tree_arity", 2))
    levels = build_kary_levels(chunk_roots, arity)
    derived_root = root_from_levels(levels)
    expected_root_hex = params.get("root")
    if not expected_root_hex:
        raise SystemExit("row commitment missing root")
    if derived_root != bytes.fromhex(expected_root_hex):
        raise SystemExit("chunk roots do not match row commitment root")

    k_required = base_samples + max(extra, 0)
    samples = sample_chunk_indices(num_chunks, k_required, seed)
    if not samples:
        print("DA sampling OK (no samples requested)")
        return

    for idx in samples:
        handle = chunk_handles[idx]
        try:
            chunk_path = safe_join(archive_root, handle)
        except ValueError as e:
            raise SystemExit(f"chunk {idx} has unsafe path: {e}")
        if not chunk_path.exists():
            raise SystemExit(f"chunk {idx} missing: {chunk_path}")
        values = [int(v) for v in json.loads(chunk_path.read_text())]
        offset = idx * chunk_len
        calc_root = merkle_from_values(values, offset)
        if calc_root != chunk_roots[idx]:
            raise SystemExit(f"chunk {idx} root mismatch")

    print(
        f"DA sampling OK ({len(samples)} chunks, mode {mode}, seed={seed})"
    )


def derive_seed_from_capsule(capsule: dict) -> int:
    profile = capsule.get("da_policy") or capsule.get("da_profile") or {}
    anchor = (capsule.get("anchor") or {}).get("anchor_ref")
    capsule_hash = capsule.get("capsule_hash") or ""
    return derive_capsule_seed(
        capsule_hash,
        anchor_ref=anchor,
        policy_id=profile.get("policy_id"),
        policy_version=profile.get("version") or profile.get("policy_version"),
    )


if __name__ == "__main__":
    main()
