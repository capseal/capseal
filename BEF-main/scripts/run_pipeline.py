#!/usr/bin/env python3
"""Adapter-driven pipeline orchestrator for BEF capsules."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from copy import deepcopy
import resource
from pathlib import Path
from typing import Any
from urllib import error as url_error
from urllib import request as url_request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backends import ADAPTERS
from bef_zk.adapter import TraceAdapter
from bef_zk.capsule.da import build_da_challenge, hash_da_challenge
from bef_zk.capsule.header import (
    build_capsule_header,
    compute_header_commit_hash,
    compute_header_hash,
    hash_air_params,
    hash_chunk_handles,
    hash_chunk_meta,
    hash_da_policy,
    hash_fri_config,
    hash_manifest_descriptor,
    hash_params,
    hash_program_descriptor,
    hash_proof_system,
    hash_row_index_ref,
    hash_capsule_identity,
    hash_instance_binding,
    sanitize_da_policy,
    sanitize_row_index_ref,
)
from bef_zk.capsule.payload import compute_payload_hash
from bef_zk.codec import ENCODING_ID, canonical_encode, compute_capsule_hash
from bef_zk.spec import StatementV1, compute_statement_hash
from capsule_bench.events import EventLogger, ProgressSink

try:
    from coincurve import PrivateKey
except ImportError:  # pragma: no cover
    PrivateKey = None


def _relpath(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def _compute_payload_hash(path: Path) -> str:
    magic = b"\xBE\xF0\xC0\xDE"
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        header = fh.read(6)
        if len(header) < 6:
            hasher.update(header)
        elif header[:4] == magic:
            remainder = header[6:]
            if remainder:
                hasher.update(remainder)
        else:
            hasher.update(header)
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _rss_bytes() -> int:
    """Return current RSS in bytes (best-effort, Linux-friendly)."""
    # Try /proc/self/status first for speed and accuracy
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        # Value in kB
                        return int(parts[1]) * 1024
    except Exception:
        pass
    # Fallback to resource ru_maxrss (kB on Linux)
    try:
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        return 0


def _portable_entry(path: Path, *, rel_path: str | None = None, base: Path | None = None) -> dict[str, str]:
    entry: dict[str, str] = {"path": str(path)}
    if rel_path:
        entry["rel_path"] = rel_path
    elif base is not None:
        entry["rel_path"] = _relpath(path, base)
    return entry


def required_samples(delta: float, epsilon: float) -> int:
    return max(1, math.ceil(math.log(1.0 / epsilon) / delta))


def _posix_rel(path: Path, base: Path) -> str:
    rel = _relpath(path, base)
    return rel.replace(os.sep, "/")


def _build_chunk_manifest(handles: list[Any], row_archive_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for idx, handle in enumerate(handles):
        uri = str(handle)
        handle_path = Path(uri)
        if not handle_path.is_absolute():
            handle_path = (row_archive_dir / uri).resolve()
        rel_uri = _posix_rel(handle_path, row_archive_dir) if handle_path.exists() else uri
        size = handle_path.stat().st_size if handle_path.exists() else 0
        sha = _compute_file_hash(handle_path) if handle_path.exists() else ""
        entries.append(
            {
                "id": idx,
                "uri": rel_uri,
                "sha256": sha,
                "size": size,
                "content_type": "application/octet-stream",
            }
        )
    return entries


def _write_chunk_manifest(entries: list[dict[str, Any]], directory: Path) -> Path:
    manifest_path = directory / "chunk_manifest.json"
    payload = {"schema": "chunk_manifest_v1", "chunks": entries}
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def _relay_request_json(
    base_url: str,
    endpoint: str,
    payload: dict[str, Any],
    token: str | None,
    timeout: float,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = url_request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with url_request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except url_error.HTTPError as exc:  # pragma: no cover - network dependent
        detail = exc.read().decode("utf-8", "ignore") if hasattr(exc, "read") else exc.reason
        raise RuntimeError(
            f"relay request failed ({exc.code}): {detail or exc.reason}"
        ) from exc
    except url_error.URLError as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"relay request failed: {exc.reason}") from exc
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:  # pragma: no cover - relay failure
        raise RuntimeError("relay returned invalid JSON") from exc


def _fetch_da_challenge_from_relay(
    *,
    relay_base: str,
    token: str | None,
    timeout: float,
    capsule_commit_hash: str,
    payload_hash: str,
    chunk_manifest_hash: str,
    chunk_meta: dict[str, Any],
) -> dict[str, Any]:
    commit_payload = {
        "capsule_commit_hash": capsule_commit_hash,
        "payload_hash": payload_hash,
        "chunk_handles_root": chunk_manifest_hash,
        "num_chunks": int(chunk_meta.get("num_chunks") or 0),
    }
    _relay_request_json(relay_base, "/v1/da/commit", commit_payload, token, timeout)
    response = _relay_request_json(
        relay_base,
        "/v1/da/challenge",
        {"capsule_commit_hash": capsule_commit_hash},
        token,
        timeout,
    )
    challenge = response.get("challenge")
    if not isinstance(challenge, dict):
        raise RuntimeError("relay DA challenge missing from response")
    return challenge


def _build_proof_system_meta(
    adapter: TraceAdapter,
    *,
    row_commitment: Any,
    trace_artifacts: TraceArtifacts,
    air_params_hash: str | None,
    fri_params_hash: str | None,
    program_hash: str | None,
) -> dict[str, Any]:
    backend_id = getattr(row_commitment, "backend", adapter.name)
    meta: dict[str, Any] = {
        "scheme_id": adapter.name,
        "backend_id": backend_id,
        "circuit_id": trace_artifacts.trace_spec.trace_format_id,
        "hash_fn_id": "sha256",
    }
    if air_params_hash:
        meta["air_params_hash"] = air_params_hash
    if fri_params_hash:
        meta["fri_params_hash"] = fri_params_hash
    if program_hash:
        meta["program_hash"] = program_hash
    vk_payload = {
        "schema": "capsule_vk_v1",
        "scheme_id": adapter.name,
        "backend_id": backend_id,
        "circuit_id": trace_artifacts.trace_spec.trace_format_id,
        "air_params_hash": air_params_hash,
        "fri_params_hash": fri_params_hash,
        "program_hash": program_hash,
    }
    meta["vk_hash"] = hash_proof_system(vk_payload)
    meta["hash"] = hash_proof_system({k: v for k, v in meta.items() if k != "hash"})
    return meta


def _load_da_challenge(
    *,
    args: argparse.Namespace,
    header_commit_hash: str,
    payload_hash: str,
    chunk_manifest_hash: str,
    chunk_meta: dict[str, Any],
    da_enabled: bool,
    out_dir: Path,
) -> tuple[dict[str, Any] | None, Path | None]:
    if not da_enabled:
        return None, None
    challenge: dict[str, Any] | None = None
    challenge_path: Path | None = None
    if args.da_challenge_file:
        src_path = args.da_challenge_file.expanduser().resolve()
        challenge = json.loads(src_path.read_text())
        challenge_path = out_dir / "da_challenge.json"
        shutil.copy2(src_path, challenge_path)
    elif args.da_relay_url:
        timeout = max(float(getattr(args, "da_relay_timeout", 15.0) or 15.0), 1.0)
        challenge = _fetch_da_challenge_from_relay(
            relay_base=args.da_relay_url,
            token=getattr(args, "da_relay_token", None),
            timeout=timeout,
            capsule_commit_hash=header_commit_hash,
            payload_hash=payload_hash,
            chunk_manifest_hash=chunk_manifest_hash,
            chunk_meta=chunk_meta,
        )
        challenge_path = out_dir / "da_challenge.json"
        challenge_path.write_text(json.dumps(challenge, indent=2))
    elif args.allow_insecure_da_challenge:
        profile = (args.verification_profile or "full").lower()
        if profile == "full":
            raise ValueError("verification_profile=full requires a relay-issued DA challenge")
        challenge = build_da_challenge(capsule_commit_hash=header_commit_hash)
        challenge_path = out_dir / "da_challenge.json"
        challenge_path.write_text(json.dumps(challenge, indent=2))
    if challenge is None:
        return None, None
    commit_ref = challenge.get("capsule_commit_hash") or challenge.get("capsule_hash")
    if commit_ref and commit_ref.lower() != header_commit_hash.lower():
        raise ValueError("DA challenge does not match capsule commit hash")
    return challenge, challenge_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the adapter-driven BEF pipeline", conflict_handler="resolve")
    parser.add_argument("--backend", type=str, default="geom", help="trace adapter id")
    parser.add_argument("--trace-id", type=str, default="demo_trace")
    parser.add_argument("--output-dir", type=Path, default=Path("out/demo"))
    parser.add_argument("--artifact-formats", choices=["json", "bin", "both"], default="both")
    parser.add_argument("--encoding-id", type=str, default=ENCODING_ID)
    parser.add_argument("--row-archive-dir", type=Path, help="override row archive output dir")
    parser.add_argument("--stats-out", type=Path, help="pipeline stats output path")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-id", type=str, required=True)
    parser.add_argument("--policy-version", type=str, default="unspecified")
    parser.add_argument("--private-key", type=Path, help="secp256k1 private key for signing the capsule")
    parser.add_argument("--prev-capsule-hash", type=str)
    parser.add_argument("--anchor-ref", type=str)
    parser.add_argument("--anchor-rule-id", type=str, default="unspecified")
    parser.add_argument("--track-id", type=str, default="unspecified")
    parser.add_argument("--docker-image-digest", type=str)
    parser.add_argument("--events-log", type=Path, help="path to events.jsonl log")
    parser.add_argument("--da-policy-id", type=str, default="light_sampling_local_v1")
    parser.add_argument("--da-k-samples", type=int)
    parser.add_argument("--da-provider-timeout-ms", type=int, default=5000)
    parser.add_argument("--da-provider-retry-count", type=int, default=2)
    parser.add_argument("--da-challenge-file", type=Path, help="path to a relay-issued DA challenge JSON")
    parser.add_argument("--da-relay-url", type=str, help="base URL for the DA relay commit/challenge API")
    parser.add_argument("--da-relay-token", type=str, help="bearer token used to authenticate with the DA relay")
    parser.add_argument(
        "--da-relay-timeout",
        type=float,
        default=15.0,
        help="timeout (seconds) for DA relay HTTP requests",
    )
    parser.add_argument(
        "--allow-insecure-da-challenge",
        action="store_true",
        help="allow locally generated DA challenge when relay-issued challenge is unavailable",
    )
    parser.add_argument(
        "--verification-profile",
        choices=["proof_only", "policy_self_reported", "policy_enforced", "full"],
        default="full",
        help="declare the verification profile embedded in the capsule header",
    )
    return parser


def _get_adapter_cls(name: str):
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        raise SystemExit(f"unknown backend '{name}'") from exc


def _prepare_manifest(output_dir: Path, encoding_id: str) -> tuple[dict[str, Any], Any]:
    manifest: dict[str, Any] = {
        "schema": "bef_artifacts_manifest_v1",
        "version": 1,
        "encoding_id": encoding_id,
        "path_index": {},
        "proofs": {},
    }

    def register_path(path: Path | None, enc_id: str) -> None:
        if not path:
            return
        rel = _relpath(path, output_dir)
        manifest.setdefault("path_index", {})[rel] = {"encoding_id": enc_id}

    return manifest, register_path



def main() -> None:
    parser = _build_parser()
    preliminary, _ = parser.parse_known_args()
    adapter_cls = _get_adapter_cls(preliminary.backend)
    adapter_cls.add_arguments(parser)
    args = parser.parse_args()
    adapter = adapter_cls(args)
    verification_profile = (args.verification_profile or "full").upper()

    formats = args.artifact_formats
    selected_encoding = args.encoding_id
    want_json = formats in {"json", "both"}
    want_bin = formats in {"bin", "both"}

    policy_path = args.policy.expanduser().resolve()
    if not policy_path.exists():
        raise FileNotFoundError(f"policy file missing: {policy_path}")
    policy_digest = _compute_file_hash(policy_path)

    private_key_bytes = None
    if args.private_key:
        if PrivateKey is None:
            raise RuntimeError("coincurve package required for signing")
        pk_path = args.private_key.expanduser().resolve()
        if not pk_path.exists():
            raise FileNotFoundError(f"private key file missing: {pk_path}")
        data = pk_path.read_text().strip()
        if data.startswith("0x"):
            data = data[2:]
        private_key_bytes = bytes.fromhex(data)

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events_log_path = (args.events_log or (out_dir / "events.jsonl")).expanduser().resolve()
    event_logger = EventLogger(
        events_log_path,
        run_id=args.trace_id,
        trace_id=args.trace_id,
        default_source="pipeline",
    )
    adapter.set_progress_callback(ProgressSink(event_logger).callback)
    event_logger.emit(
        "run_started",
        {
            "backend": args.backend,
            "policy_id": args.policy_id,
            "track_id": args.track_id,
            "trace_id": args.trace_id,
        },
    )
    manifest, register_path = _prepare_manifest(out_dir, selected_encoding)

    default_archive = out_dir / "row_archive"
    row_archive_dir = (args.row_archive_dir or default_archive).expanduser().resolve()
    if row_archive_dir.exists():
        shutil.rmtree(row_archive_dir)
    row_archive_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    trace_artifacts = adapter.simulate_trace(args)
    t_trace = trace_artifacts.trace_time_sec if trace_artifacts.trace_time_sec is not None else (time.perf_counter() - t0)
    bef_trace = trace_artifacts.bef_trace
    trace_path = out_dir / "stc_trace.json"
    w_start = time.perf_counter()
    trace_path.write_text(json.dumps(bef_trace, indent=2))
    t_trace_serialize = time.perf_counter() - w_start
    print(f"[1/4] wrote bef_trace to {trace_path}")
    event_logger.emit(
        "stage_metrics",
        {
            "stage": "trace",
            "time_sec": t_trace,
            "serialize_time_sec": t_trace_serialize,
            "rss_bytes": _rss_bytes(),
            "row_width": trace_artifacts.row_width,
            "trace_chunks": len(bef_trace.get("chunks", [])),
        },
    )
    event_logger.emit(
        "spec_locked",
        {
            "trace_spec_hash": trace_artifacts.trace_spec_hash,
            "policy_hash": policy_digest,
            "track_id": args.track_id,
        },
    )

    anchor_meta = {
        "anchor_rule_id": args.anchor_rule_id,
        "anchor_ref": args.anchor_ref,
        "track_id": args.track_id,
    }
    if args.docker_image_digest:
        anchor_meta["docker_image_digest"] = args.docker_image_digest
    t1 = time.perf_counter()
    trace_commitment = adapter.commit_to_trace(
        trace_artifacts,
        row_archive_dir=row_archive_dir,
    )
    t_commit = time.perf_counter() - t1
    row_commitment = trace_commitment.row_commitment
    row_archive_rel = _relpath(row_archive_dir, out_dir)
    row_commitment.params["archive_root_abs"] = str(row_archive_dir)
    row_commitment.params["archive_root"] = row_archive_rel

    anchors_list: list[dict[str, object]] = [dict(anchor_meta)] if anchor_meta else []

    public_inputs = adapter.extract_public_inputs(trace_artifacts)
    statement_event_hash = event_logger.emit(
        "statement_locked",
        {
            "trace_root": row_commitment.params.get("root"),
            "public_input_count": len(public_inputs),
        },
    )
    anchor_meta["event_chain_head"] = statement_event_hash
    anchors_list = [dict(anchor_meta)] if anchor_meta else []
    statement_obj = StatementV1(
        statement_version="1.0",
        trace_spec_hash=trace_artifacts.trace_spec_hash,
        policy_hash=policy_digest,
        trace_root=row_commitment.params.get("root", ""),
        public_inputs=public_inputs,
        anchors=anchors_list,
    )
    statement_hash_hex = compute_statement_hash(statement_obj)

    chunk_handles = trace_commitment.chunk_handles
    chunk_roots_paths = trace_commitment.chunk_roots_paths
    chunk_len = row_commitment.row_width
    num_chunks = len(chunk_handles)
    chunk_meta = {
        "num_chunks": num_chunks,
        "chunk_len": chunk_len,
        "chunk_size_bytes": chunk_len * 8,
        "data_length_bytes": chunk_len * num_chunks * 8,
        "chunking_rule_id": "fixed_range_v1",
    }
    params_obj = {"row_width": trace_artifacts.row_width}
    params_hash = hash_params(params_obj)
    chunk_meta_hash = hash_chunk_meta(chunk_meta)
    context = getattr(trace_artifacts, "context", None)
    air_params_hash = hash_air_params(getattr(context, "params", None))
    fri_params_hash = hash_fri_config(getattr(context, "fri_cfg", None))
    program_descriptor = getattr(adapter, "program_descriptor", None) or getattr(adapter_cls, "PROGRAM", None)
    program_hash = hash_program_descriptor(program_descriptor)
    proof_system_meta = _build_proof_system_meta(
        adapter,
        row_commitment=row_commitment,
        trace_artifacts=trace_artifacts,
        air_params_hash=air_params_hash,
        fri_params_hash=fri_params_hash,
        program_hash=program_hash,
    )
    vk_hash = proof_system_meta.get("vk_hash") or proof_system_meta.get("hash")
    instance_hash = hash_instance_binding(
        statement_hash=statement_hash_hex,
        row_root=row_commitment.params.get("root", ""),
        trace_spec_hash=trace_artifacts.trace_spec_hash,
        vk_hash=vk_hash,
        params_hash=params_hash,
        chunk_meta_hash=chunk_meta_hash,
        row_tree_arity=row_commitment.params.get("chunk_tree_arity"),
        air_params_hash=air_params_hash,
        fri_params_hash=fri_params_hash,
        program_hash=program_hash,
    )
    proof_system_meta["instance_hash"] = instance_hash

    t2 = time.perf_counter()
    proof_artifacts = adapter.generate_proof(
        trace_artifacts,
        trace_commitment,
        statement_hash=bytes.fromhex(statement_hash_hex),
        binding_hash=bytes.fromhex(instance_hash),
        encoding_id=selected_encoding,
        trace_path=trace_path,
    )
    t_prove = time.perf_counter() - t2
    # Emit stage metrics for commit/prove using adapter-provided timings when available
    commit_profile = dict(trace_commitment.profile_data or {})
    prove_profile = dict(proof_artifacts.profile_data or {})
    event_logger.emit(
        "stage_metrics",
        {
            "stage": "commit",
            "time_sec": commit_profile.get("time_row_commit_sec", t_commit),
            "rss_bytes": _rss_bytes(),
            "num_chunks": len(trace_commitment.chunk_handles),
            "row_width": trace_artifacts.row_width,
        },
    )
    event_logger.emit(
        "stage_metrics",
        {
            "stage": "prove",
            "time_total_sec": prove_profile.get("time_total_sec", t_prove),
            "time_pc_commit_sec": prove_profile.get("time_pc_commit_sec"),
            "time_fri_sec": prove_profile.get("time_fri_sec"),
            "rss_bytes": _rss_bytes(),
            "fri_queries": getattr(getattr(trace_artifacts, "context", None), "fri_cfg", None).num_queries if getattr(getattr(trace_artifacts, "context", None), "fri_cfg", None) else None,
        },
    )
    register_path(chunk_roots_paths.get("json"), "hex_json_v1")
    register_path(chunk_roots_paths.get("bin"), "raw32_v1")
    row_archive_artifact = {
        **trace_commitment.row_archive_artifact,
        "mode": "LOCAL_FILE",
        "path": row_archive_rel,
        "rel_path": row_archive_rel,
        "abs_path": str(row_archive_dir),
        "chunk_roots_path": os.path.join(row_archive_rel, Path(chunk_roots_paths["json"]).name),
        "chunk_roots_abs": str(chunk_roots_paths["json"]),
        "chunk_roots_bin_path": os.path.join(row_archive_rel, Path(chunk_roots_paths["bin"]).name),
        "chunk_roots_bin_abs": str(chunk_roots_paths["bin"]),
    }
    row_archive_artifact["chunk_roots_rel_path"] = row_archive_artifact["chunk_roots_path"]
    row_archive_artifact["chunk_roots_bin_rel_path"] = row_archive_artifact["chunk_roots_bin_path"]
    chunk_manifest_entries = _build_chunk_manifest(chunk_handles, row_archive_dir)
    row_archive_artifact["chunk_handles"] = chunk_manifest_entries
    chunk_manifest_path = _write_chunk_manifest(chunk_manifest_entries, row_archive_dir)
    row_archive_artifact["chunk_manifest_path"] = _posix_rel(chunk_manifest_path, out_dir)
    register_path(chunk_manifest_path, "json_hex_v1")

    default_k = required_samples(0.1, 1e-6)
    da_policy = {
        "policy_id": args.da_policy_id,
        "k_samples": int(args.da_k_samples or default_k),
        "provider_timeout_ms": max(0, args.da_provider_timeout_ms),
        "provider_retry_count": max(0, args.da_provider_retry_count),
        "verification_level": "probabilistic_da_sampling",
        "provider": {
            "mode": "LOCAL_FILE",
            "archive_root": row_archive_artifact["abs_path"],
        },
    }

    row_index_ref = {
        "commitment_type": "merkle_root",
        "commitment": row_commitment.params.get("root"),
        "tree_arity": row_commitment.params.get("chunk_tree_arity"),
        "proof_fetch_rule_id": "proof_attached_to_chunk_v1",
        "pointer": {
            "path": row_archive_rel,
            "provider_root": row_archive_artifact["abs_path"],
        },
    }
    hashing_meta = {
        "hash_fn_id": "sha256",
        "encoding_id": selected_encoding,
    }
    register_path(events_log_path, "events_log_v1")

    proof_json_path = out_dir / "adapter_proof.json"
    proof_bin_path = out_dir / "adapter_proof.bin"
    manifest_geom_formats: dict[str, dict[str, object]] = {}
    proof_json = proof_artifacts.proof_json
    if want_json:
        s0 = time.perf_counter()
        proof_json_path.write_text(proof_json)
        t_write_proof_json = time.perf_counter() - s0
        register_path(proof_json_path, "json_hex_v1")
        manifest_geom_formats["json"] = {
            "path": _relpath(proof_json_path, out_dir),
            "encoding_id": "json_hex_v1",
            "sha256_payload_hash": _compute_payload_hash(proof_json_path),
        }
        print(f"[2/4] wrote proof JSON to {proof_json_path}")
    proof_bytes = proof_artifacts.proof_bytes
    if want_bin:
        s1 = time.perf_counter()
        proof_bin_path.write_bytes(proof_bytes)
        t_write_proof_bin = time.perf_counter() - s1
        register_path(proof_bin_path, selected_encoding)
        manifest_geom_formats["bin"] = {
            "path": _relpath(proof_bin_path, out_dir),
            "encoding_id": selected_encoding,
            "sha256_payload_hash": _compute_payload_hash(proof_bin_path),
        }
        print(f"[2/4b] wrote proof BIN to {proof_bin_path}")
    # Emit serialize metrics
    event_logger.emit(
        "stage_metrics",
        {
            "stage": "serialize_proof",
            "time_json_sec": locals().get("t_write_proof_json"),
            "time_bin_sec": locals().get("t_write_proof_bin"),
            "rss_bytes": _rss_bytes(),
            "size_json_bytes": len(proof_json.encode("utf-8")) if want_json else None,
            "size_bin_bytes": len(proof_bytes) if want_bin else None,
        },
    )

    primary_proof_path = proof_json_path if want_json else proof_bin_path
    primary_size = len(proof_json.encode("utf-8")) if want_json else len(proof_bytes)
    primary_proof_rel = f"proofs/primary/{primary_proof_path.name}"

    event_logger.emit(
        "proof_artifact",
        {
            "path": str(primary_proof_path),
            "size_bytes": primary_size,
        },
    )

    v0 = time.perf_counter()
    verify_ok, verify_stats, verify_time = adapter.verify(
        proof_json,
        bytes.fromhex(statement_hash_hex),
        trace_artifacts,
        binding_hash=bytes.fromhex(instance_hash),
    )
    if not verify_ok:
        raise RuntimeError("adapter verification failed")
    event_logger.emit(
        "stage_metrics",
        {
            "stage": "verify",
            "time_sec": verify_time or (time.perf_counter() - v0),
            "rss_bytes": _rss_bytes(),
        },
    )

    # Check for chunk_leaf_multiproof (Geom-specific feature)
    chunk_leaf_enabled = False
    if hasattr(proof_artifacts.proof_obj, "fri_proof"):
        chunk_leaf_enabled = any(
            bool(batch.proof.chunk_leaf_proofs)
            for batch in proof_artifacts.proof_obj.fri_proof.batches
        )
    manifest["chunk_roots"] = {
        "default": "json",
        "formats": {
            "json": {
                "path": _relpath(chunk_roots_paths["json"], out_dir),
                "encoding_id": "hex_json_v1",
            },
            "bin": {
                "path": _relpath(chunk_roots_paths["bin"], out_dir),
                "encoding_id": "raw32_v1",
            },
        },
    }

    capsule_path_json = out_dir / "strategy_capsule.json"
    capsule_path_bin = out_dir / "strategy_capsule.bin"
    capsule_manifest_formats: dict[str, dict[str, str]] = {}
    if want_json:
        register_path(capsule_path_json, "json_hex_v1")
        capsule_manifest_formats["json"] = {
            "path": _relpath(capsule_path_json, out_dir),
            "encoding_id": "json_hex_v1",
        }
    if want_bin:
        register_path(capsule_path_bin, selected_encoding)
        capsule_manifest_formats["bin"] = {
            "path": _relpath(capsule_path_bin, out_dir),
            "encoding_id": selected_encoding,
        }

    # primary_proof_path/primary_size computed above for event logging
    geom_formats: dict[str, dict[str, object]] = {}
    if want_json:
        json_rel = f"proofs/primary/{proof_json_path.name}"
        geom_formats["json"] = {
            "path": str(proof_json_path),
            "rel_path": json_rel,
            "size_bytes": len(proof_json.encode("utf-8")),
            "sha256_payload_hash": manifest_geom_formats["json"]["sha256_payload_hash"],
        }
    if want_bin:
        bin_rel = f"proofs/primary/{proof_bin_path.name}"
        geom_formats["bin"] = {
            "path": str(proof_bin_path),
            "rel_path": bin_rel,
            "size_bytes": len(proof_bytes),
            "sha256_payload_hash": manifest_geom_formats["bin"]["sha256_payload_hash"],
        }
    manifest["proofs"]["primary"] = {
        "default_format": "json" if want_json else "bin",
        "formats": manifest_geom_formats,
        "features": {"chunk_leaf_multiproof": chunk_leaf_enabled},
    }

    manifest_path = out_dir / "artifact_manifest.json"
    register_path(manifest_path, "json_manifest_v1")
    manifest["capsule"] = {
        "default_format": "json" if want_json else "bin",
        "formats": capsule_manifest_formats,
    }
    manifest_hash = hash_manifest_descriptor(manifest)

    extra_proofs = proof_artifacts.extra or {}

    # DA challenge may be populated later; initialize path to avoid UnboundLocal
    da_challenge_path = None

    capsule_artifacts: dict[str, object] = {
        "trace": _portable_entry(trace_path, rel_path=f"artifacts/{trace_path.name}"),
        "proof": _portable_entry(primary_proof_path, rel_path=primary_proof_rel),
        "row_archive": row_archive_artifact,
    }
    if events_log_path:
        capsule_artifacts["events_log"] = _portable_entry(
            events_log_path,
            rel_path=f"events/{events_log_path.name}",
        )
    if da_challenge_path:
        capsule_artifacts["da_challenge"] = _portable_entry(
            da_challenge_path,
            rel_path=f"da/{da_challenge_path.name}",
            base=out_dir,
        )

    events_log_hash = None
    events_log_len = None
    if events_log_path and events_log_path.exists():
        events_log_hash = f"sha256:{_compute_file_hash(events_log_path)}"
        events_log_len = events_log_path.stat().st_size
        anchor_meta["events_log_hash"] = events_log_hash
        anchor_meta["events_log_len"] = events_log_len

    policy_ref = {
        "policy_id": args.policy_id,
        "policy_version": args.policy_version,
        "policy_hash": policy_digest,
        "track_id": args.track_id,
    }
    policy_section = dict(policy_ref)
    policy_section["policy_path"] = str(policy_path)
    sanitized_row_index_ref = sanitize_row_index_ref(row_index_ref)
    row_index_ref_hash = hash_row_index_ref(sanitized_row_index_ref)
    sanitized_da_policy = sanitize_da_policy(da_policy)
    da_policy_hash = hash_da_policy(sanitized_da_policy)
    chunk_manifest_hash = hash_chunk_handles(chunk_manifest_entries)

    # Get row_openings count (Geom-specific, 0 for succinct backends like RISC0)
    row_openings_count = 0
    if hasattr(proof_artifacts.proof_obj, "row_openings"):
        row_openings_count = len(proof_artifacts.proof_obj.row_openings)

    proofs_section = {
        "primary": {
            "path": str(primary_proof_path),
            "rel_path": primary_proof_rel,
            "size_bytes": primary_size,
            "row_openings": row_openings_count,
            "row_backend": row_commitment.backend,
            "row_archive": row_archive_artifact,
            "formats": geom_formats,
        }
    }

    capsule_payload: dict[str, Any] = {
        "schema": "bef_capsule_v1",
        "vm_id": args.backend,
        "trace_id": args.trace_id,
        "prev_capsule_hash": args.prev_capsule_hash,
        "trace_spec": trace_artifacts.trace_spec.to_obj(),
        "trace_spec_hash": trace_artifacts.trace_spec_hash,
        "policy": policy_section,
        "params": params_obj,
        "da_policy": da_policy,
        "chunk_meta": chunk_meta,
        "row_index_ref": row_index_ref,
        "hashing": hashing_meta,
        "proof_system": proof_system_meta,
        "anchor": anchor_meta,
        "proofs": proofs_section,
        "row_archive": row_archive_artifact,
        "artifacts": capsule_artifacts,
    }
    if extra_proofs.get("nova"):
        nova_info = {
            "stats_path": extra_proofs["nova"].get("stats_path"),
            "recursive_proof_bytes": extra_proofs["nova"].get("recursive_proof_bytes"),
            "compressed": extra_proofs["nova"].get("compressed"),
        }
        capsule_payload["proofs"]["nova"] = nova_info
        stats_path = extra_proofs["nova"].get("stats_path")
        if stats_path:
            path_obj = Path(stats_path)
            register_path(path_obj, "json_hex_v1")
            capsule_payload["artifacts"]["nova_stats"] = _portable_entry(
                path_obj,
                rel_path=f"proofs/nova/{path_obj.name}",
            )

    capsule_payload.setdefault("artifacts", {})["manifest"] = _portable_entry(
        manifest_path,
        rel_path="manifests/artifact_manifest.json",
    )

    capsule_payload["statement"] = statement_obj.to_obj()
    capsule_payload["statement_hash"] = statement_hash_hex
    capsule_payload["verification_profile"] = verification_profile

    payload_hash = compute_payload_hash(capsule_payload, encoding_id=selected_encoding)

    capsule_header = build_capsule_header(
        vm_id=args.backend,
        backend_id=row_commitment.backend,
        circuit_id=trace_artifacts.trace_spec.trace_format_id,
        trace_id=args.trace_id,
        prev_capsule_hash=args.prev_capsule_hash,
        trace_spec_hash=trace_artifacts.trace_spec_hash,
        statement_hash=statement_hash_hex,
        params_hash=params_hash,
        row_root=row_commitment.params.get("root", ""),
        row_tree_arity=row_commitment.params.get("chunk_tree_arity"),
        row_index_ref_hash=row_index_ref_hash,
        chunk_meta_hash=chunk_meta_hash,
        chunk_handles_root=chunk_manifest_hash,
        policy_ref=policy_ref,
        da_policy_hash=da_policy_hash,
        anchor=anchor_meta,
        proof_system=proof_system_meta,
        chunk_manifest_hash=chunk_manifest_hash,
        manifest_hash=manifest_hash,
        air_params_hash=air_params_hash,
        fri_params_hash=fri_params_hash,
        program_hash=program_hash,
        payload_hash=payload_hash,
        verification_profile=verification_profile,
    )

    da_enabled = bool(sanitized_da_policy)
    header_commit_hash = compute_header_commit_hash(capsule_header)
    da_challenge, da_challenge_path = _load_da_challenge(
        args=args,
        header_commit_hash=header_commit_hash,
        payload_hash=payload_hash,
        chunk_manifest_hash=chunk_manifest_hash,
        chunk_meta=chunk_meta,
        da_enabled=da_enabled,
        out_dir=out_dir,
    )
    if da_challenge:
        capsule_header["da_ref"]["challenge_hash"] = hash_da_challenge(da_challenge)
    header_hash = compute_header_hash(capsule_header)

    capsule = dict(capsule_payload)
    capsule["payload_hash"] = payload_hash
    capsule["header"] = capsule_header
    capsule["header_hash"] = header_hash
    capsule["header_commit_hash"] = header_commit_hash
    capsule_hash = hash_capsule_identity(header_commit_hash, payload_hash)
    capsule["capsule_hash"] = capsule_hash

    if da_challenge:
        capsule["da_challenge"] = da_challenge

    if private_key_bytes:
        if len(private_key_bytes) != 32:
            raise ValueError("secp256k1 private key must be 32 bytes")
        signer = PrivateKey(private_key_bytes)
        signature = signer.sign_recoverable(bytes.fromhex(capsule_hash), hasher=None)
        pubkey = signer.public_key.format(compressed=False)
        capsule["authorship"] = {
            "signer_pubkey": pubkey.hex(),
            "signature": signature.hex(),
        }

    capsule_json = json.dumps(capsule, indent=2)
    if want_json:
        c0 = time.perf_counter()
        capsule_path_json.write_text(capsule_json)
        t_capsule_json = time.perf_counter() - c0
        print(f"[4/4] wrote capsule descriptor to {capsule_path_json}")
    if want_bin:
        c1 = time.perf_counter()
        capsule_path_bin.write_bytes(canonical_encode(capsule, encoding_id=selected_encoding))
        t_capsule_bin = time.perf_counter() - c1
        print(f"[4/4b] wrote capsule BIN to {capsule_path_bin}")
    event_logger.emit(
        "capsule_sealed",
        {
            "capsule_hash": capsule_hash,
        },
    )
    event_logger.emit(
        "stage_metrics",
        {
            "stage": "serialize_capsule",
            "time_json_sec": locals().get("t_capsule_json"),
            "time_bin_sec": locals().get("t_capsule_bin"),
            "rss_bytes": _rss_bytes(),
        },
    )

    m0 = time.perf_counter()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    event_logger.emit(
        "stage_metrics",
        {
            "stage": "serialize_manifest",
            "time_sec": time.perf_counter() - m0,
            "rss_bytes": _rss_bytes(),
        },
    )

    stats_path = args.stats_out or (out_dir / "pipeline_stats.json")
    combined_profile = dict(trace_commitment.profile_data)
    combined_profile.update(proof_artifacts.profile_data)
    pipeline_stats = {
        "schema": "adapter_pipeline_stats_v1",
        "trace_id": args.trace_id,
        "row_width": trace_artifacts.row_width,
        "trace_chunks": len(bef_trace.get("chunks", [])),
        "trace_time_sec": trace_artifacts.trace_time_sec,
        "proof": {
            "size_bytes": len(proof_json.encode("utf-8")) if want_json else len(proof_bytes),
            "profile": combined_profile,
            "verify_stats": verify_stats,
            "verify_time_sec": verify_time,
            **proof_artifacts.chunk_leaf_stats,
        },
        "row_archive": row_archive_artifact,
        "da_policy": da_policy,
        "chunk_meta": chunk_meta,
        "row_index_ref": row_index_ref,
    }
    if extra_proofs.get("nova"):
        pipeline_stats["nova"] = extra_proofs["nova"]
    stats_path.write_text(json.dumps(pipeline_stats, indent=2))
    print(f"[4b] wrote pipeline stats to {stats_path}")
    event_logger.emit(
        "run_completed",
        {
            "status": "ok",
        },
    )
    event_logger.close()


if __name__ == "__main__":
    main()
