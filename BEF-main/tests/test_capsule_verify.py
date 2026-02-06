from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable

import pytest

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_RELAY_PRIV = "3" * 64
DEFAULT_AUTH_PRIV = "1".zfill(64)
TEST_MANIFEST_PRIV = TEST_RELAY_PRIV


def _test_relay_pubkey() -> str:
    coincurve = pytest.importorskip("coincurve")
    return coincurve.PrivateKey(bytes.fromhex(TEST_RELAY_PRIV)).public_key.format(compressed=False).hex()


def _default_trusted_relays() -> dict[str, str]:
    return {"test_relay": _test_relay_pubkey()}


def _relay_registry_hash(relays: dict[str, str]) -> str:
    ordered = {k: relays[k] for k in sorted(relays)}
    blob = json.dumps(ordered, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

from bef_zk.air.geom_air import GeomAIRParams, GeomInitialState
from bef_zk.fri.config import FRIConfig
from scripts.geom_programs import GEOM_PROGRAM
from scripts.verify_capsule import verify_capsule
from bef_zk.stc.vc import STCVectorCommitment
from bef_zk.stc.aok_cpu import merkle_from_values
from bef_zk.stc.merkle import build_kary_levels, root_from_levels, prove_kary
from bef_zk.zk_geom.columns import column_names
from bef_zk.zk_geom.prover import zk_prove_geom
from bef_zk.zk_geom.serialization import proof_to_json
from bef_zk.capsule.header import (
    build_capsule_header,
    compute_header_commit_hash,
    compute_header_hash,
    hash_chunk_handles,
    hash_chunk_meta,
    hash_da_policy,
    hash_params,
    hash_proof_system,
    hash_row_index_ref,
    hash_capsule_identity,
    hash_air_params,
    hash_fri_config,
    hash_program_descriptor,
    hash_instance_binding,
    sanitize_da_policy,
    sanitize_row_index_ref,
)
from bef_zk.capsule.payload import compute_payload_hash
from bef_zk.capsule.da import (
    build_da_challenge,
    hash_da_challenge,
    challenge_signature_payload,
    build_signed_da_challenge,
    hash_signed_da_challenge,
    sign_da_challenge,
)
from bef_zk.spec import (
    TraceSpecV1,
    StatementV1,
    compute_trace_spec_hash,
    compute_statement_hash,
)
from bef_zk.verifier_errors import E201_EVENT_LOG_MISMATCH


def _default_params(steps: int = 8) -> GeomAIRParams:
    return GeomAIRParams(
        steps=steps,
        num_challenges=2,
        r_challenges=[1234567, 89101112],
        matrix=[[2, 1], [1, 1]],
    )


def _fri_cfg(steps: int) -> FRIConfig:
    domain_size = 1 << (steps - 1).bit_length()
    max_rounds = max(1, domain_size.bit_length() - 1)
    return FRIConfig(
        field_modulus=(1 << 61) - 1,
        domain_size=domain_size,
        max_degree=steps - 1,
        num_rounds=min(4, max_rounds),
        num_queries=4,
    )


def _row_width(params: GeomAIRParams) -> int:
    return len(column_names(params))


def _payload_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _write_event_chain(path: Path, trace_id: str, events: list[tuple[str, dict]]) -> str:
    prev_hash = "0" * 64
    statement_event_hash: str | None = None
    with path.open("w", encoding="utf-8") as fh:
        for seq, (event_type, data) in enumerate(events, start=1):
            base_event = {
                "schema": "bef_capsule_stream_v1",
                "v": 1,
                "trace_id": trace_id,
                "seq": seq,
                "ts_ms": seq,
                "type": event_type,
                "data": data,
            }
            serialized = json.dumps(base_event, sort_keys=True, separators=(",", ":")).encode("utf-8")
            event_hash = hashlib.sha256(bytes.fromhex(prev_hash) + serialized).hexdigest()
            payload = dict(base_event)
            payload["prev_event_hash"] = prev_hash
            payload["event_hash"] = event_hash
            fh.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
            if event_type == "statement_locked":
                statement_event_hash = event_hash
            prev_hash = event_hash
    if not statement_event_hash:
        raise RuntimeError("statement_locked event missing from test log")
    return statement_event_hash


def _write_manifest_signature(
    manifest_root: Path,
    anchor_hash: str,
    *,
    signer_id: str = "test_manifest",
    priv_hex: str | None = None,
) -> None:
    coincurve = pytest.importorskip("coincurve")
    priv_hex = priv_hex or TEST_MANIFEST_PRIV
    priv = coincurve.PrivateKey(bytes.fromhex(priv_hex))
    digest = anchor_hash.split(":", 1)[1] if ":" in anchor_hash else anchor_hash
    message = bytes.fromhex(digest)
    signature = priv.sign_recoverable(message, hasher=None).hex()
    payload = {
        "schema": "capsule_manifest_signature_v1",
        "signer_id": signer_id,
        "signature": signature,
    }
    (manifest_root / "manifest_signature.json").write_text(json.dumps(payload, indent=2))


def _policy_path_for_capsule(capsule_path: Path) -> Path:
    return capsule_path.parent / "policy.json"


def _manifest_root_for_capsule(capsule_path: Path) -> Path:
    return capsule_path.parent / "manifests"


def _verify_with_policy(capsule_path: Path, *, required_level: str = "proof_only", **kwargs):
    kwargs.setdefault("policy_path", _policy_path_for_capsule(capsule_path))
    kwargs.setdefault("manifest_root", _manifest_root_for_capsule(capsule_path))
    if (required_level or "").lower() == "full":
        relays = _default_trusted_relays()
        kwargs.setdefault("trusted_relay_keys", relays)
        kwargs.setdefault("trusted_relay_root", _relay_registry_hash(relays))
    return verify_capsule(capsule_path, required_level=required_level, **kwargs)


def _policy_ref_view(policy: dict[str, Any] | None) -> dict[str, Any]:
    policy = policy or {}
    return {
        "policy_id": policy.get("policy_id"),
        "policy_version": policy.get("policy_version"),
        "policy_hash": policy.get("policy_hash"),
        "track_id": policy.get("track_id"),
    }


def _infer_backend_id(capsule: dict) -> str:
    proofs = capsule.get("proofs") or {}
    geom_entry = proofs.get("geom") or proofs.get("primary") or {}
    return geom_entry.get("row_backend") or capsule.get("vm_id", "")


def _build_header_from_capsule(capsule: dict) -> dict:
    params = capsule.get("params") or {}
    chunk_meta = capsule.get("chunk_meta") or {}
    row_index_ref = capsule.get("row_index_ref") or {}
    row_archive_info = capsule.get("row_archive") or {}
    da_policy = capsule.get("da_policy") or {}
    anchor = capsule.get("anchor") or {}
    policy_ref = _policy_ref_view(capsule.get("policy"))
    proof_system = capsule.get("proof_system") or {}
    if not proof_system:
        proof_system = {
            "scheme_id": capsule.get("vm_id", ""),
            "backend_id": _infer_backend_id(capsule),
            "circuit_id": (capsule.get("trace_spec") or {}).get("trace_format_id", ""),
            "hash_fn_id": "sha256",
        }
    proof_system.setdefault("circuit_id", (capsule.get("trace_spec") or {}).get("trace_format_id", ""))
    proof_system.setdefault("hash_fn_id", "sha256")
    proof_system.setdefault("scheme_id", capsule.get("vm_id", ""))
    proof_system.setdefault("backend_id", _infer_backend_id(capsule))
    params_hash_val = hash_params(params)
    chunk_meta_hash_val = hash_chunk_meta(chunk_meta)
    row_tree_arity = row_index_ref.get("tree_arity")
    row_root = row_index_ref.get("commitment") or ""
    return build_capsule_header(
        vm_id=capsule.get("vm_id", ""),
        backend_id=_infer_backend_id(capsule),
        circuit_id=(capsule.get("trace_spec") or {}).get("trace_format_id", ""),
        trace_id=capsule.get("trace_id", ""),
        prev_capsule_hash=capsule.get("prev_capsule_hash"),
        trace_spec_hash=capsule.get("trace_spec_hash", ""),
        statement_hash=capsule.get("statement_hash", ""),
        params_hash=params_hash_val,
        row_root=row_root,
        row_tree_arity=row_tree_arity,
        row_index_ref_hash=hash_row_index_ref(row_index_ref),
        chunk_meta_hash=chunk_meta_hash_val,
        chunk_handles_root=hash_chunk_handles(row_archive_info.get("chunk_handles", [])),
        policy_ref=policy_ref,
        da_policy_hash=hash_da_policy(da_policy),
        anchor=anchor,
        chunk_manifest_hash=hash_chunk_handles(row_archive_info.get("chunk_handles", [])),
        proof_system=proof_system,
        payload_hash=capsule.get("payload_hash"),
        verification_profile=capsule.get("verification_profile"),
    )


def _refresh_capsule_header(capsule: dict) -> None:
    from bef_zk.capsule.da import DA_CHALLENGE_V2_SCHEMA
    header = _build_header_from_capsule(capsule)
    header_commit_hash = compute_header_commit_hash(header)
    capsule["header_commit_hash"] = header_commit_hash
    da_challenge = capsule.get("da_challenge")
    if da_challenge:
        # Check if this is a v2 challenge - don't modify it
        if da_challenge.get("schema") == DA_CHALLENGE_V2_SCHEMA:
            # V2 challenges are already signed - just use the existing hash
            da_challenge_hash = hash_signed_da_challenge(da_challenge)
        else:
            # V1 challenge - add default fields
            da_challenge["capsule_commit_hash"] = capsule["header_commit_hash"]
            da_challenge.setdefault("relay_pubkey_id", "test_relay")
            da_challenge.setdefault("relay_signature", "de" * 64)
            da_challenge_hash = hash_da_challenge(da_challenge)
        header.setdefault("da_ref", {})["challenge_hash"] = da_challenge_hash
    capsule["header"] = header
    capsule["header_hash"] = compute_header_hash(header)
    payload_hash = compute_payload_hash(capsule)
    capsule["payload_hash"] = payload_hash
    capsule["capsule_hash"] = hash_capsule_identity(header_commit_hash, payload_hash)


def _update_capsule_hash(capsule: dict) -> None:
    _refresh_capsule_header(capsule)
    _refresh_da_challenge(capsule)
    _refresh_capsule_header(capsule)


def _refresh_da_challenge(capsule: dict) -> None:
    from bef_zk.capsule.da import DA_CHALLENGE_V2_SCHEMA
    challenge = capsule.get("da_challenge")
    if not challenge:
        return
    # V2 challenges should not be modified - they are pre-signed
    if challenge.get("schema") == DA_CHALLENGE_V2_SCHEMA:
        capsule.setdefault("header", {}).setdefault("da_ref", {})["challenge_hash"] = hash_signed_da_challenge(challenge)
        return
    # V1 challenge - add default fields
    challenge["capsule_commit_hash"] = capsule.get("header_commit_hash")
    challenge.setdefault("relay_pubkey_id", "test_relay")
    challenge.setdefault("relay_signature", "de" * 64)
    capsule.setdefault("header", {}).setdefault("da_ref", {})["challenge_hash"] = hash_da_challenge(challenge)


def _sync_capsule_after_proof_change(capsule_path: Path, proof_path: Path) -> None:
    capsule = json.loads(capsule_path.read_text())
    geom_entry = (capsule.get("proofs") or {}).get("geom") or {}
    json_fmt = (geom_entry.get("formats") or {}).get("json")
    if json_fmt is not None:
        json_fmt["sha256_payload_hash"] = _payload_hash(proof_path)
    capsule.setdefault("proofs", {})["geom"] = geom_entry
    _update_capsule_hash(capsule)
    capsule_path.write_text(json.dumps(capsule))


def _write_policy_proof(proof_path: Path, policy_hash: str) -> None:
    leaf = bytes.fromhex(policy_hash)
    levels = build_kary_levels([leaf], 2)
    siblings = prove_kary(levels, 0, 2)
    proof = {
        "leaf_hash": policy_hash,
        "leaf_index": 0,
        "total_leaves": 1,
        "arity": 2,
        "siblings_by_level": [[s.hex() for s in level] for level in siblings],
    }
    proof_path.write_text(json.dumps(proof))


def _write_policy_file(path: Path, *, track_id: str = "baseline_no_accel") -> str:
    policy = {
        "schema": "bef_benchmark_policy_v1",
        "policy_id": "baseline_policy_v1",
        "policy_version": "1.0",
        "tracks": [
            {
                "track_id": track_id,
                "rules": {
                    "forbid_gpu": True,
                    "require_deterministic_build": True,
                    "required_public_outputs": ["final_cnt"],
                },
            }
        ],
    }
    path.write_text(json.dumps(policy))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_capsule(
    tmpdir: Path,
    *,
    mutate_proof: Callable[[dict], None] | None = None,
    add_archive: bool = False,
    policy_hash_override: str | None = None,
    policy_id: str = "test_policy",
    track_id: str = "test_track",
    anchor_ref: str | None = None,
    docker_image_digest: str | None = None,
    manifest_gpu_detected: bool = False,
    policy_doc_override: dict | None = None,
) -> tuple[Path, Path, dict, Path]:
    params = _default_params()
    init = GeomInitialState()
    fri_cfg = _fri_cfg(params.steps)
    vc = STCVectorCommitment(chunk_len=256)
    columns = column_names(params)
    schema_doc = {"columns": columns}
    schema_hash = hashlib.sha256(json.dumps(schema_doc, sort_keys=True).encode()).hexdigest()
    trace_spec = TraceSpecV1(
        spec_version="1.0",
        trace_format_id="GEOM_AIR_V1",
        record_schema_ref=f"sha256:{schema_hash}",
        encoding_id="dag_cbor_canonical_v1",
        field_modulus_id="goldilocks_61",
    )
    trace_spec_hash = compute_trace_spec_hash(trace_spec)
    policy_doc = {
        "schema": "bef_benchmark_policy_v1",
        "policy_id": policy_id,
        "policy_version": "v1",
        "tracks": [
            {
                "track_id": track_id,
                "rules": {
                    "forbid_gpu": True,
                    "require_deterministic_build": True,
                    "required_public_outputs": ["final_cnt"],
                },
            }
        ],
    }
    if policy_doc_override is not None:
        policy_doc = copy.deepcopy(policy_doc_override)
    policy_path = tmpdir / "policy.json"
    policy_path.write_text(json.dumps(policy_doc))
    computed_policy_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    policy_info = {
        "policy_id": policy_id,
        "policy_version": "v1",
        "policy_hash": policy_hash_override or computed_policy_hash,
        "track_id": track_id,
        "policy_path": str(policy_path),
    }
    manifest_root, inferred_anchor = _make_manifest_root(tmpdir, gpu_detected=manifest_gpu_detected)
    anchor_meta = {
        "anchor_rule_id": "unspecified",
        "anchor_ref": anchor_ref or inferred_anchor,
        "track_id": track_id,
        "docker_image_digest": docker_image_digest or "sha256:test_fixture",
    }
    events_path = tmpdir / "events.jsonl"
    statement_event_hash = _write_event_chain(
        events_path,
        trace_id="test_capsule",
        events=[
            ("run_started", {"backend": "geom", "track_id": track_id}),
            ("spec_locked", {"trace_id": "test_capsule"}),
            ("statement_locked", {"trace_root": "deadbeef"}),
            ("proof_artifact", {"path": "proofs/primary/geom_proof.json"}),
            ("capsule_sealed", {"status": "ok"}),
            ("run_completed", {"status": "ok"}),
        ],
    )
    anchor_meta["event_chain_head"] = statement_event_hash
    anchor_meta["events_log_hash"] = f"sha256:{_payload_hash(events_path)}"
    anchor_meta["events_log_len"] = events_path.stat().st_size
    if docker_image_digest:
        anchor_meta["docker_image_digest"] = docker_image_digest
    proof_archive_dir = tmpdir / "proof_row_archive"
    if proof_archive_dir.exists():
        shutil.rmtree(proof_archive_dir)
    proof_archive_dir.mkdir(exist_ok=True)
    anchors_list = [anchor_meta] if anchor_meta else []
    statement_holder: dict[str, object] = {}

    def _build_statement_hash(row_commitment, geom_statement):
        public_inputs = [
            {"name": "final_m11", "value": int(geom_statement.final_m11)},
            {"name": "final_m12", "value": int(geom_statement.final_m12)},
            {"name": "final_m22", "value": int(geom_statement.final_m22)},
            {"name": "final_cnt", "value": int(geom_statement.final_cnt)},
        ]
        statement = StatementV1(
            statement_version="1.0",
            trace_spec_hash=trace_spec_hash,
            policy_hash=policy_info["policy_hash"],
            trace_root=row_commitment.params.get("root", ""),
            public_inputs=public_inputs,
            anchors=anchors_list,
        )
        statement_hash_hex = compute_statement_hash(statement)
        statement_holder["statement"] = statement
        statement_holder["statement_hash"] = statement_hash_hex
        chunk_handles = list(row_commitment.params.get("chunk_handles") or [])
        chunk_meta = {
            "num_chunks": len(chunk_handles),
            "chunk_len": row_commitment.row_width,
            "chunk_size_bytes": row_commitment.row_width * 8,
            "data_length_bytes": row_commitment.row_width * len(chunk_handles) * 8,
            "chunking_rule_id": "fixed_range_v1",
        }
        statement_holder["chunk_meta"] = chunk_meta
        chunk_meta_hash = hash_chunk_meta(chunk_meta)
        statement_holder["chunk_meta_hash"] = chunk_meta_hash
        params_hash = hash_params({"row_width": row_commitment.row_width})
        statement_holder["params_hash"] = params_hash
        air_hash = hash_air_params(params)
        fri_hash = hash_fri_config(fri_cfg)
        program_hash = hash_program_descriptor(GEOM_PROGRAM)
        vk_payload = {
            "schema": "capsule_vk_v1",
            "scheme_id": "geom",
            "backend_id": row_commitment.backend,
            "circuit_id": trace_spec.trace_format_id,
            "air_params_hash": air_hash,
            "fri_params_hash": fri_hash,
            "program_hash": program_hash,
        }
        vk_hash = hash_proof_system(vk_payload)
        proof_system_meta = {
            "scheme_id": "geom",
            "backend_id": row_commitment.backend,
            "circuit_id": trace_spec.trace_format_id,
            "hash_fn_id": "sha256",
        }
        if air_hash:
            proof_system_meta["air_params_hash"] = air_hash
        if fri_hash:
            proof_system_meta["fri_params_hash"] = fri_hash
        if program_hash:
            proof_system_meta["program_hash"] = program_hash
        proof_system_meta["vk_hash"] = vk_hash
        proof_system_meta["hash"] = hash_proof_system(
            {k: v for k, v in proof_system_meta.items() if k != "hash"}
        )
        statement_holder["proof_system_meta"] = proof_system_meta
        instance_hash = hash_instance_binding(
            statement_hash=statement_hash_hex,
            row_root=row_commitment.params.get("root", ""),
            trace_spec_hash=trace_spec_hash,
            vk_hash=vk_hash,
            params_hash=params_hash,
            chunk_meta_hash=chunk_meta_hash,
            row_tree_arity=row_commitment.params.get("chunk_tree_arity"),
            air_params_hash=air_hash,
            fri_params_hash=fri_hash,
            program_hash=program_hash,
        )
        statement_holder["instance_hash"] = instance_hash
        statement_holder["binding_inputs"] = {
            "params_hash": params_hash,
            "chunk_meta_hash": chunk_meta_hash,
            "row_tree_arity": row_commitment.params.get("chunk_tree_arity"),
            "air_params_hash": air_hash,
            "fri_params_hash": fri_hash,
            "program_hash": program_hash,
        }
        return bytes.fromhex(instance_hash)

    proof = zk_prove_geom(
        GEOM_PROGRAM,
        params,
        init,
        fri_cfg,
        vc,
        row_backend_params={"archive_dir": proof_archive_dir},
        statement_hash_fn=_build_statement_hash,
    )
    proof_dict = json.loads(proof_to_json(proof))
    if mutate_proof is not None:
        mutate_proof(proof_dict)
    proof_path = tmpdir / "geom_proof.json"
    proof_path.write_text(json.dumps(proof_dict))

    state = {
        "n": "0x0",
        "root": "0x1",
        "s": ["0x2", "0x3"],
        "pow": ["0x4", "0x5"],
    }
    nova_stats = {"nova_state": state}
    nova_stats_path = tmpdir / "nova_stats.json"
    nova_stats_path.write_text(json.dumps(nova_stats))

    row_commitment = proof.row_commitment
    if row_commitment is None:
        raise RuntimeError("expected row commitment")
    row_commitment.params["archive_root_abs"] = str(proof_archive_dir)
    row_commitment.params["archive_root"] = str(proof_archive_dir.relative_to(tmpdir))
    row_params = row_commitment.params
    artifacts = {
        "geom_proof": {"path": str(proof_path), "rel_path": proof_path.name},
        "nova_stats": {"path": str(nova_stats_path), "rel_path": nova_stats_path.name},
        "events_log": {"path": str(events_path), "rel_path": f"events/{events_path.name}"},
    }
    row_archive_artifact = None
    if add_archive:
        chunk_roots_hex = list(row_params.get("chunk_roots_hex", []))
        proof_archive_dir.mkdir(exist_ok=True)
        chunk_roots_path = proof_archive_dir / "chunk_roots.json"
        chunk_roots_path.write_text(json.dumps(chunk_roots_hex))
        row_archive_artifact = {
            "mode": "LOCAL_FILE",
            "path": str(proof_archive_dir.relative_to(tmpdir)),
            "abs_path": str(proof_archive_dir),
            "chunk_roots_path": str(chunk_roots_path),
            "chunk_roots_format": "hex_json_v1",
            "chunk_tree_arity": row_params.get("chunk_tree_arity", 2),
        }
        manifest_handles: list[dict[str, Any]] = []
        for idx, handle in enumerate(row_params.get("chunk_handles", []) or []):
            entry = {
                "id": idx,
                "uri": str(Path(handle).name),
                "sha256": "",
                "size": 0,
                "content_type": "application/octet-stream",
            }
            manifest_handles.append(entry)
        if manifest_handles:
            row_archive_artifact["chunk_handles"] = manifest_handles
        artifacts["row_archive"] = row_archive_artifact

    proof_backend = row_commitment.backend if row_commitment is not None else "geom"
    proof_system_meta = statement_holder.get("proof_system_meta")
    if not proof_system_meta:
        proof_system_meta = {
            "scheme_id": "geom",
            "backend_id": proof_backend,
            "circuit_id": trace_spec.trace_format_id,
            "hash_fn_id": "sha256",
        }
        proof_system_meta["hash"] = hash_proof_system({k: v for k, v in proof_system_meta.items() if k != "hash"})

    capsule = {
        "schema": "bef_capsule_v1",
        "vm_id": "geom_vm_v1",
        "air_id": "geom_vm_v1",
        "trace_id": "test_capsule",
        "verification_profile": "PROOF_ONLY",
        "params": {
            "steps": params.steps,
            "num_challenges": params.num_challenges,
            "num_queries": 4,
            "row_width": _row_width(params),
            "challenge_seed": 0,
            "r_challenges": params.r_challenges,
        },
        "trace_commitment": state,
        "artifacts": artifacts,
        "policy": policy_info,
        "trace_spec": trace_spec.to_obj(),
        "trace_spec_hash": trace_spec_hash,
        "anchor": anchor_meta,
        "proof_system": proof_system_meta,
    }
    statement = statement_holder.get("statement")
    statement_hash_hex = statement_holder.get("statement_hash")
    instance_hash_hex = statement_holder.get("instance_hash")
    if instance_hash_hex:
        proof_system_meta["instance_hash"] = instance_hash_hex
    if statement is not None and statement_hash_hex is not None:
        capsule["statement"] = statement.to_obj()
        capsule["statement_hash"] = statement_hash_hex
    row_root = row_params.get("root") if row_params else None
    if row_root:
        capsule.setdefault("row_index_ref", {
            "commitment_type": "merkle_root",
            "commitment": row_root,
            "tree_arity": row_params.get("chunk_tree_arity", 2),
            "proof_fetch_rule_id": "test",
        })
    capsule.setdefault("chunk_meta", statement_holder.get("chunk_meta") or {
            "num_chunks": row_params.get("num_chunks"),
            "chunk_len": row_params.get("chunk_len"),
        })
    capsule["proofs"] = {
        "geom": {
            "path": str(proof_path),
            "row_openings": len(proof_dict.get("row_openings", [])),
            "row_backend": (proof_dict.get("row_commitment") or {}).get("backend", "geom_stc_fri"),
            "row_archive": row_archive_artifact,
            "formats": {
                "json": {
                    "path": str(proof_path),
                    "encoding_id": "json_hex_v1",
                    "sha256_payload_hash": _payload_hash(proof_path),
                }
            },
        }
    }
    capsule["row_archive"] = row_archive_artifact
    capsule.setdefault("hashing", {"hash_fn_id": "sha256", "encoding_id": "dag_cbor_canonical_v1"})
    _update_capsule_hash(capsule)
    capsule_path = tmpdir / "strategy_capsule.json"
    capsule_path.write_text(json.dumps(capsule))
    artifact_manifest = {
        "schema": "capsule_artifact_manifest_v1",
        "entries": [],
    }
    (tmpdir / "artifact_manifest.json").write_text(json.dumps(artifact_manifest))
    return capsule_path, nova_stats_path, nova_stats, manifest_root


def _attach_policy(
    tmpdir: Path,
    policy_hash: str,
    policy_id: str = "test_policy",
    track_id: str = "test_track",
    *,
    anchor_ref: str | None = None,
    docker_image_digest: str | None = None,
    policy_file_src: Path | None = None,
    manifest_gpu_detected: bool = False,
) -> tuple[Path, Path]:
    run_dir = tmpdir / "capsule"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    policy_override = None
    if policy_file_src:
        policy_override = json.loads(policy_file_src.read_text())
    cap_path, _, _, manifest_root = _write_capsule(
        run_dir,
        policy_hash_override=policy_hash,
        policy_id=policy_id,
        track_id=track_id,
        anchor_ref=anchor_ref,
        docker_image_digest=docker_image_digest,
        manifest_gpu_detected=manifest_gpu_detected,
        policy_doc_override=policy_override,
    )
    if policy_file_src:
        dst = _policy_path_for_capsule(cap_path)
        if policy_file_src.resolve() != dst.resolve():
            shutil.copy2(policy_file_src, dst)
    _attach_authorship(cap_path, DEFAULT_AUTH_PRIV)
    return cap_path, manifest_root


def _make_manifest_root(tmp_path: Path, *, gpu_detected: bool = False) -> tuple[Path, str]:
    root = tmp_path / "manifests"
    root.mkdir(exist_ok=True)
    hardware = {
        "schema": "bef_hardware_manifest_v1",
        "gpu": {
            "detected": gpu_detected,
            "devices": [] if not gpu_detected else [{"model": "Test GPU"}],
        },
    }
    os_manifest = {"schema": "bef_os_fingerprint_v1", "platform": "test-os"}
    toolchain = {"schema": "bef_toolchain_manifest_v1", "python": sys.version}
    manifest_index = {
        "schema": "bef_manifest_index_v1",
        "entries": [],
    }
    files = {
        "hardware_manifest": hardware,
        "os_fingerprint": os_manifest,
        "toolchain_manifest": toolchain,
        "manifest_index": manifest_index,
    }
    hashes: dict[str, str] = {}
    for name, payload in files.items():
        path = root / f"{name}.json"
        path.write_text(json.dumps(payload))
        hashes[name] = f"sha256:{_payload_hash(path)}"
    anchor_payload = json.dumps(
        {
            "schema": "capsule_bench_manifest_anchor_v1",
            "hashes": hashes,
        },
        sort_keys=True,
    ).encode()
    anchor = f"capsulebench_manifest_v1:{hashlib.sha256(anchor_payload).hexdigest()}"
    _write_manifest_signature(root, anchor)
    return root, anchor


def _refresh_statement(capsule: dict) -> None:
    statement_obj = capsule.get("statement")
    trace_spec_hash = capsule.get("trace_spec_hash")
    if not statement_obj or not trace_spec_hash:
        return
    statement = StatementV1.from_obj(statement_obj)
    policy_info = capsule.get("policy") or {}
    policy_hash = policy_info.get("policy_hash")
    if policy_hash:
        statement.policy_hash = policy_hash
    trace_root = (capsule.get("row_index_ref") or {}).get("commitment")
    if trace_root:
        statement.trace_root = trace_root
    anchor_meta = capsule.get("anchor")
    anchors = [anchor_meta] if anchor_meta else []
    statement.anchors = anchors
    capsule["statement"] = statement.to_obj()
    capsule["statement_hash"] = compute_statement_hash(statement)
    _update_capsule_hash(capsule)


def _attach_da_capsule(
    tmpdir: Path,
    sample_k: int = 2,
    relay_priv_hex: str | None = None,
    attach_authorship: bool = True,
) -> tuple[Path, Path]:
    cap_path, _, _, _ = _write_capsule(tmpdir)
    capsule = json.loads(cap_path.read_text())
    capsule["verification_profile"] = "FULL"
    proof_path = tmpdir / "geom_proof.json"
    proof = json.loads(proof_path.read_text())
    rc_params = (proof.get("row_commitment") or {}).get("params") or {}
    archive_src = Path(rc_params.get("archive_root") or "")
    if not archive_src.exists():
        raise FileNotFoundError("row commitment archive missing for test fixture")
    archive_dir = tmpdir / "row_archive_da"
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    shutil.copytree(archive_src, archive_dir)
    raw_handles = [str(h) for h in rc_params.get("chunk_handles", [])]
    chunk_roots_hex = list(rc_params.get("chunk_roots_hex", []))
    chunk_roots = [bytes.fromhex(h) for h in chunk_roots_hex]
    chunk_len = int(rc_params.get("chunk_len") or 0)
    chunk_tree_arity = int(rc_params.get("chunk_tree_arity") or 2)
    root = root_from_levels(build_kary_levels(chunk_roots, chunk_tree_arity))
    rel_path = archive_dir.relative_to(tmpdir)
    chunk_roots_path = archive_dir / "chunk_roots.json"
    chunk_roots_path.write_text(json.dumps(chunk_roots_hex, indent=2))
    chunk_roots_bin_path = archive_dir / "chunk_roots.bin"
    chunk_roots_bin_path.write_bytes(b"".join(bytes.fromhex(h) for h in chunk_roots_hex))
    chunk_roots_digest = hashlib.sha256(chunk_roots_bin_path.read_bytes()).hexdigest()
    manifest_handles = [
        {
            "id": idx,
            "uri": Path(handle).name,
            "sha256": "",
            "size": 0,
            "content_type": "application/octet-stream",
        }
        for idx, handle in enumerate(raw_handles)
    ]
    row_archive_info = {
        "mode": "LOCAL_FILE",
        "path": rel_path.as_posix(),
        "abs_path": str(archive_dir),
        "chunk_handles": manifest_handles,
        "chunk_roots_hex": chunk_roots_hex,
        "chunk_tree_arity": chunk_tree_arity,
        "chunk_roots_digest": chunk_roots_digest,
        "chunk_roots_bin_path": str(chunk_roots_bin_path.relative_to(tmpdir)),
        "chunk_roots_bin_abs": str(chunk_roots_bin_path),
        "chunk_roots_path": str(chunk_roots_path.relative_to(tmpdir)),
        "chunk_roots_abs": str(chunk_roots_path),
    }
    artifacts = capsule.setdefault("artifacts", {})
    artifacts["row_archive"] = {
        "mode": "LOCAL_FILE",
        "path": rel_path.as_posix(),
    }
    capsule["row_archive"] = row_archive_info
    capsule["row_index_ref"] = {
        "commitment_type": "merkle_root",
        "commitment": root.hex(),
        "tree_arity": chunk_tree_arity,
        "proof_fetch_rule_id": "local_test",
        "pointer": {"path": rel_path.as_posix()},
    }
    capsule["chunk_meta"] = {
        "num_chunks": len(raw_handles),
        "chunk_len": chunk_len,
        "chunk_size_bytes": chunk_len * 8,
        "data_length_bytes": chunk_len * len(raw_handles) * 8,
        "chunking_rule_id": "fixed_range_v1",
    }
    effective_k = sample_k if sample_k >= 0 else len(raw_handles)
    capsule["da_policy"] = {
        "policy_id": "da_local_test",
        "k_samples": effective_k,
        "provider_timeout_ms": 1000,
        "provider_retry_count": 1,
        "provider": {
            "mode": "LOCAL_FILE",
            "archive_root": str(archive_dir),
        },
    }
    _refresh_statement(capsule)
    _update_capsule_hash(capsule)
    header_commit_hash = capsule.get("header_commit_hash")
    da_challenge = build_da_challenge(capsule_commit_hash=header_commit_hash, relay_pubkey_id="test_relay")
    if relay_priv_hex:
        coincurve = pytest.importorskip("coincurve")
        priv = coincurve.PrivateKey(bytes.fromhex(relay_priv_hex))
        payload = challenge_signature_payload(da_challenge)
        digest = hashlib.sha256(payload).digest()
        da_challenge["relay_signature"] = priv.sign(digest, hasher=None).hex()
    else:
        da_challenge.setdefault("relay_signature", "ab" * 64)
    capsule["da_challenge"] = da_challenge
    capsule.setdefault("header", {}).setdefault("da_ref", {})["challenge_hash"] = hash_da_challenge(da_challenge)
    _update_capsule_hash(capsule)
    cap_path.write_text(json.dumps(capsule))
    if attach_authorship:
        _attach_authorship(cap_path, DEFAULT_AUTH_PRIV)
    return cap_path, archive_dir


def _attach_authorship(cap_path: Path, private_key_hex: str) -> str:
    coincurve = pytest.importorskip("coincurve")
    PrivateKey = coincurve.PrivateKey
    capsule = json.loads(cap_path.read_text())
    capsule.pop("authorship", None)
    _refresh_capsule_header(capsule)
    capsule_hash = capsule.get("capsule_hash")
    priv = PrivateKey(bytes.fromhex(private_key_hex))
    signature = priv.sign_recoverable(bytes.fromhex(capsule_hash), hasher=None)
    pubkey = priv.public_key.format(compressed=False).hex()
    capsule["authorship"] = {
        "signer_pubkey": pubkey,
        "signature": signature.hex(),
    }
    cap_path.write_text(json.dumps(capsule))
    return pubkey


def test_verify_capsule_succeeds(tmp_path: Path) -> None:
    cap_dir = tmp_path / "valid"
    cap_dir.mkdir()
    cap_path, _, _, manifest_root = _write_capsule(cap_dir)
    _attach_authorship(cap_path, DEFAULT_AUTH_PRIV)
    result = _verify_with_policy(cap_path)
    assert result["trace_commitment"]["root"] == "0x1"
    assert result["status"] == "POLICY_SELF_REPORTED"
    assert result["verification_profile"] == "PROOF_ONLY"


def test_verify_capsule_detects_nova_mismatch(tmp_path: Path) -> None:
    cap_dir = tmp_path / "nova_mismatch"
    cap_dir.mkdir()
    cap_path, stats_path, stats, manifest_root = _write_capsule(cap_dir)
    bad_state = dict(stats["nova_state"])
    bad_state["root"] = "0xdead"
    stats_path.write_text(json.dumps({"nova_state": bad_state}))
    with pytest.raises(RuntimeError):
        _verify_with_policy(cap_path)


def test_verify_capsule_detects_corrupted_geom_proof(tmp_path: Path) -> None:
    cap_dir = tmp_path / "geom_corrupt"
    cap_dir.mkdir()

    def mutate(proof_dict: dict) -> None:
        proof_dict["row_openings"][0]["row_values"][0] += 1

    cap_path, _, _, manifest_root = _write_capsule(cap_dir, mutate_proof=mutate)
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(cap_path, required_level="full")


def test_verification_profile_requires_policy(tmp_path: Path) -> None:
    cap_dir = tmp_path / "profile_policy"
    cap_dir.mkdir()
    cap_path, _, _, _ = _write_capsule(cap_dir)
    capsule = json.loads(cap_path.read_text())
    capsule["verification_profile"] = "POLICY_ENFORCED"
    _update_capsule_hash(capsule)
    cap_path.write_text(json.dumps(capsule))
    manifest_root = _manifest_root_for_capsule(cap_path)
    with pytest.raises(ValueError) as excinfo:
        verify_capsule(cap_path, manifest_root=manifest_root, required_level="policy_enforced")
    assert "E036" in str(excinfo.value)


def test_verification_profile_full_without_da(tmp_path: Path) -> None:
    cap_dir = tmp_path / "profile_full"
    cap_dir.mkdir()
    cap_path, _, _, _ = _write_capsule(cap_dir)
    capsule = json.loads(cap_path.read_text())
    capsule["verification_profile"] = "FULL"
    capsule.pop("da_policy", None)
    capsule.pop("da_challenge", None)
    if "header" in capsule:
        capsule["header"].setdefault("da_ref", {})["challenge_hash"] = None
    _update_capsule_hash(capsule)
    cap_path.write_text(json.dumps(capsule))
    _attach_authorship(cap_path, DEFAULT_AUTH_PRIV)
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(cap_path, required_level="full")
    assert "E071" in str(excinfo.value)


def test_full_requires_signed_da_challenge(tmp_path: Path) -> None:
    cap_dir = tmp_path / "full_signed"
    cap_dir.mkdir()
    cap_path, _ = _attach_da_capsule(cap_dir, relay_priv_hex=TEST_RELAY_PRIV)
    policy_path = _policy_path_for_capsule(cap_path)
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    proof_path = cap_dir / "policy_proof.json"
    _write_policy_proof(proof_path, policy_hash)
    result = _verify_with_policy(
        cap_path,
        required_level="full",
        policy_path=policy_path,
        policy_proof_path=proof_path,
        policy_registry_root=policy_hash,
    )
    assert result["da_audit_verified"]


def test_full_rejects_unsigned_da_challenge(tmp_path: Path) -> None:
    cap_dir = tmp_path / "full_unsigned"
    cap_dir.mkdir()
    cap_path, _ = _attach_da_capsule(cap_dir, attach_authorship=False)
    policy_path = _policy_path_for_capsule(cap_path)
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    proof_path = cap_dir / "policy_proof.json"
    _write_policy_proof(proof_path, policy_hash)
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            cap_path,
            required_level="full",
            policy_path=policy_path,
            policy_proof_path=proof_path,
            policy_registry_root=policy_hash,
        )
    assert "E073" in str(excinfo.value)


def test_verify_capsule_with_row_archive(tmp_path: Path) -> None:
    cap_dir = tmp_path / "with_archive"
    cap_dir.mkdir()
    cap_path, _, _, manifest_root = _write_capsule(cap_dir, add_archive=True)
    result = _verify_with_policy(cap_path)
    assert result["row_archive"] is not None


def test_verify_capsule_missing_row_archive(tmp_path: Path) -> None:
    cap_dir = tmp_path / "missing_archive"
    cap_dir.mkdir()
    cap_path, _, _, manifest_root = _write_capsule(cap_dir, add_archive=True)
    archive_dir = cap_dir / "proof_row_archive"
    shutil.rmtree(archive_dir)
    with pytest.raises(FileNotFoundError):
        _verify_with_policy(cap_path)


def test_verify_capsule_policy_binding(tmp_path: Path, monkeypatch) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    capsule_path, manifest_root = _attach_policy(
        tmp_path,
        policy_hash,
        track_id="baseline_no_accel",
        docker_image_digest="sha256:deadbeef",
        policy_file_src=policy_file,
    )
    result = _verify_with_policy(
        capsule_path,
        required_level="policy_self_reported",
        policy_path=policy_file,
        manifest_root=manifest_root,
    )
    assert result["policy_track"] == "baseline_no_accel"


def test_verify_capsule_policy_mismatch(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    capsule_path, manifest_root = _attach_policy(
        tmp_path,
        policy_hash,
        track_id="baseline_no_accel",
        docker_image_digest="sha256:deadbeef",
        policy_file_src=policy_file,
    )
    policy_file.write_text("tampered")
    with pytest.raises(ValueError):
        _verify_with_policy(
            capsule_path,
            required_level="policy_self_reported",
            policy_path=policy_file,
            manifest_root=manifest_root,
        )


def test_policy_forbid_gpu_violation(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    capsule_path, manifest_root = _attach_policy(
        tmp_path,
        policy_hash,
        track_id="baseline_no_accel",
        docker_image_digest="sha256:deadbeef",
        policy_file_src=policy_file,
        manifest_gpu_detected=True,
    )
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            capsule_path,
            required_level="policy_self_reported",
            policy_path=policy_file,
            manifest_root=manifest_root,
        )


def test_manifest_hash_mismatch_rejected(tmp_path: Path) -> None:
    cap_dir = tmp_path / "manifest_mismatch"
    cap_dir.mkdir()
    cap_path, _, _, _ = _write_capsule(cap_dir)
    capsule = json.loads(cap_path.read_text())
    capsule.setdefault("header", {})["artifact_manifest_hash"] = "deadbeef" * 8
    capsule["header_commit_hash"] = compute_header_commit_hash(capsule["header"])
    capsule["header_hash"] = compute_header_hash(capsule["header"])
    capsule["payload_hash"] = compute_payload_hash(capsule)
    capsule_hash = hash_capsule_identity(capsule.get("header_commit_hash", ""), capsule["payload_hash"])
    capsule["capsule_hash"] = capsule_hash
    cap_path.write_text(json.dumps(capsule))
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(cap_path)
    assert "E013" in str(excinfo.value)


def test_event_log_length_mismatch(tmp_path: Path) -> None:
    """Event log mismatch produces warning at PROOF_ONLY level."""
    cap_dir = tmp_path / "event_len"
    cap_dir.mkdir()
    cap_path, _, _, _ = _write_capsule(cap_dir)
    capsule = json.loads(cap_path.read_text())
    events_entry = capsule["artifacts"]["events_log"]
    events_path = Path(events_entry["path"])
    events_path.write_text("")
    # At PROOF_ONLY level, mismatch is a warning, not a failure
    result = _verify_with_policy(cap_path, required_level="proof_only")
    assert not result["events_verified"]
    assert E201_EVENT_LOG_MISMATCH in result.get("warnings", [])


def test_event_log_mismatch_fails_at_full_level(tmp_path: Path) -> None:
    """Event log mismatch fails verification at FULLY_VERIFIED level."""
    cap_dir = tmp_path / "event_full"
    cap_dir.mkdir()
    cap_path, _, _, _ = _write_capsule(cap_dir)
    capsule = json.loads(cap_path.read_text())
    events_entry = capsule["artifacts"]["events_log"]
    events_path = Path(events_entry["path"])
    # Corrupt the events log to trigger mismatch
    events_path.write_text("")
    # At FULLY_VERIFIED level, mismatch should fail verification
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(cap_path, required_level="full")
    assert "E201" in str(excinfo.value)


def test_event_log_mismatch_warning_at_policy_enforced(tmp_path: Path) -> None:
    """Event log mismatch is warning-only at POLICY_ENFORCED level (below FULLY_VERIFIED)."""
    cap_dir = tmp_path / "event_policy"
    cap_dir.mkdir()
    cap_path, _, _, _ = _write_capsule(cap_dir)
    capsule = json.loads(cap_path.read_text())
    events_entry = capsule["artifacts"]["events_log"]
    events_path = Path(events_entry["path"])
    events_path.write_text("")
    # At POLICY_ENFORCED level (below FULLY_VERIFIED threshold), mismatch is a warning
    result = _verify_with_policy(cap_path, required_level="policy_enforced")
    assert not result["events_verified"]
    assert E201_EVENT_LOG_MISMATCH in result.get("warnings", [])


def test_backend_param_tamper_detected(tmp_path: Path) -> None:
    cap_dir = tmp_path / "fri_tamper"
    cap_dir.mkdir()
    cap_path, _, _, _ = _write_capsule(cap_dir)
    capsule = json.loads(cap_path.read_text())
    header = capsule.setdefault("header", {})
    proof_meta = header.setdefault("proof_system", {})
    proof_meta["fri_params_hash"] = "00" * 32
    descriptor = {k: v for k, v in proof_meta.items() if k != "hash"}
    proof_meta["hash"] = hash_proof_system(descriptor)
    capsule["header_commit_hash"] = compute_header_commit_hash(header)
    capsule["header_hash"] = compute_header_hash(header)
    capsule["payload_hash"] = compute_payload_hash(capsule)
    capsule["capsule_hash"] = hash_capsule_identity(capsule["header_commit_hash"], capsule["payload_hash"])
    cap_path.write_text(json.dumps(capsule))
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(cap_path)
    assert "E301" in str(excinfo.value)


def test_policy_required_output_violation(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    policy_override = json.loads(policy_file.read_text())
    capsule_path, _, _, manifest_root = _write_capsule(
        tmp_path,
        policy_hash_override=policy_hash,
        track_id="baseline_no_accel",
        docker_image_digest="sha256:deadbeef",
        mutate_proof=None,
        policy_doc_override=policy_override,
    )
    capsule = json.loads(capsule_path.read_text())
    statement = capsule["statement"]
    statement["public_inputs"] = [entry for entry in statement["public_inputs"] if entry["name"] != "final_cnt"]
    capsule["statement"] = statement
    capsule["statement_hash"] = compute_statement_hash(StatementV1.from_obj(statement))
    _update_capsule_hash(capsule)
    capsule_path.write_text(json.dumps(capsule))
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            capsule_path,
            required_level="policy_self_reported",
            policy_path=policy_file,
            manifest_root=manifest_root,
        )
    assert "E301" in str(excinfo.value)


def test_policy_manifest_signature_required(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    capsule_path, manifest_root = _attach_policy(
        tmp_path,
        policy_hash=policy_hash,
        policy_id="baseline_policy_v1",
        track_id="baseline_no_accel",
        docker_image_digest="sha256:deadbeef",
        policy_file_src=policy_file,
    )
    sig_path = manifest_root / "manifest_signature.json"
    sig_path.unlink()
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            capsule_path,
            required_level="policy_enforced",
            policy_path=policy_file,
            manifest_root=manifest_root,
        )
    assert "E106" in str(excinfo.value)


def test_policy_manifest_signature_invalid(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    capsule_path, manifest_root = _attach_policy(
        tmp_path,
        policy_hash=policy_hash,
        policy_id="baseline_policy_v1",
        track_id="baseline_no_accel",
        docker_image_digest="sha256:deadbeef",
        policy_file_src=policy_file,
    )
    sig_path = manifest_root / "manifest_signature.json"
    data = json.loads(sig_path.read_text())
    data["signature"] = "00" * 65
    sig_path.write_text(json.dumps(data))
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            capsule_path,
            required_level="policy_enforced",
            policy_path=policy_file,
            manifest_root=manifest_root,
        )
    assert "E107" in str(excinfo.value)


def test_policy_manifest_signer_untrusted(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    capsule_path, manifest_root = _attach_policy(
        tmp_path,
        policy_hash=policy_hash,
        policy_id="baseline_policy_v1",
        track_id="baseline_no_accel",
        docker_image_digest="sha256:deadbeef",
        policy_file_src=policy_file,
    )
    override = {"rogue": _test_relay_pubkey()}
    registry_root = _relay_registry_hash(override)
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            capsule_path,
            required_level="policy_enforced",
            policy_path=policy_file,
            manifest_root=manifest_root,
            trusted_manifest_signers=override,
            trusted_manifest_root=registry_root,
        )
    assert "E108" in str(excinfo.value)


def test_policy_manifest_registry_mismatch(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    capsule_path, manifest_root = _attach_policy(
        tmp_path,
        policy_hash=policy_hash,
        policy_id="baseline_policy_v1",
        track_id="baseline_no_accel",
        docker_image_digest="sha256:deadbeef",
        policy_file_src=policy_file,
    )
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            capsule_path,
            required_level="policy_enforced",
            policy_path=policy_file,
            manifest_root=manifest_root,
            trusted_manifest_root="0" * 64,
        )
    assert "E109" in str(excinfo.value)


def test_verify_capsule_signature_and_acl(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    capsule_path, manifest_root = _attach_policy(
        tmp_path,
        policy_hash,
        policy_id="signed_policy",
        track_id="baseline_no_accel",
        docker_image_digest="sha256:deadbeef",
        policy_file_src=policy_file,
    )
    signer_pub = _attach_authorship(capsule_path, "1".zfill(64))
    acl_path = tmp_path / "acl.json"
    acl = {
        "schema": "bef_acl_v1",
        "authorizations": {
            "signed_policy": [
                {"pubkey": signer_pub, "status": "active", "description": "test"}
            ]
        },
    }
    acl_path.write_text(json.dumps(acl))
    result = _verify_with_policy(
        capsule_path,
        required_level="policy_self_reported",
        policy_path=policy_file,
        manifest_root=manifest_root,
        acl_path=acl_path,
    )
    assert result["authorship_verified"]
    assert result["acl_authorized"]


def test_verify_capsule_signature_invalid(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    capsule_path, manifest_root = _attach_policy(
        tmp_path,
        policy_hash,
        policy_id="signed_policy",
        track_id="baseline_no_accel",
        docker_image_digest="sha256:deadbeef",
        policy_file_src=policy_file,
    )
    _attach_authorship(capsule_path, "1".zfill(64))
    tampered = json.loads(capsule_path.read_text())
    tampered["authorship"]["signature"] = "00" * 65
    _update_capsule_hash(tampered)
    capsule_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError):
        _verify_with_policy(
            capsule_path,
            required_level="policy_self_reported",
            policy_path=policy_file,
            manifest_root=manifest_root,
        )


def test_verify_capsule_acl_rejects_unauthorized(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    capsule_path, manifest_root = _attach_policy(
        tmp_path,
        policy_hash,
        policy_id="signed_policy",
        track_id="baseline_no_accel",
        docker_image_digest="sha256:deadbeef",
        policy_file_src=policy_file,
    )
    signer_pub = _attach_authorship(capsule_path, "1".zfill(64))
    acl_path = tmp_path / "acl.json"
    acl = {
        "schema": "bef_acl_v1",
        "authorizations": {
            "signed_policy": [
                {"pubkey": "deadbeef", "status": "active"}
            ]
        },
    }
    acl_path.write_text(json.dumps(acl))
    with pytest.raises(ValueError):
        _verify_with_policy(
            capsule_path,
            required_level="policy_self_reported",
            policy_path=policy_file,
            manifest_root=manifest_root,
            acl_path=acl_path,
        )


def test_event_chain_mismatch(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.json"
    policy_hash = _write_policy_file(policy_file)
    capsule_path, manifest_root = _attach_policy(
        tmp_path,
        policy_hash,
        track_id="baseline_no_accel",
        docker_image_digest="sha256:deadbeef",
        policy_file_src=policy_file,
    )
    capsule = json.loads(capsule_path.read_text())
    events_entry = capsule["artifacts"]["events_log"]
    events_path = Path(events_entry["path"])
    events_path.write_text("tampered\n")
    _update_capsule_hash(capsule)
    capsule_path.write_text(json.dumps(capsule))
    result = _verify_with_policy(
        capsule_path,
        policy_path=policy_file,
        manifest_root=manifest_root,
    )
    assert not result["events_verified"]
    assert E201_EVENT_LOG_MISMATCH in result.get("warnings", [])


def test_statement_hash_mismatch(tmp_path: Path) -> None:
    cap_dir = tmp_path / "stmt_hash_mismatch"
    cap_dir.mkdir()
    capsule_path, _, _, _ = _write_capsule(cap_dir)
    capsule = json.loads(capsule_path.read_text())
    capsule["statement_hash"] = "00" * 32
    _update_capsule_hash(capsule)
    capsule_path.write_text(json.dumps(capsule))
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(capsule_path)
    assert "E301" in str(excinfo.value)


def test_statement_trace_root_mismatch(tmp_path: Path) -> None:
    cap_dir = tmp_path / "stmt_root_mismatch"
    cap_dir.mkdir()
    capsule_path, _, _, _ = _write_capsule(cap_dir)
    capsule = json.loads(capsule_path.read_text())
    capsule["statement"]["trace_root"] = "deadbeef"
    capsule["statement_hash"] = compute_statement_hash(StatementV1.from_obj(capsule["statement"]))
    _update_capsule_hash(capsule)
    capsule_path.write_text(json.dumps(capsule))
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(capsule_path)
    assert "E301" in str(excinfo.value)


def test_verify_capsule_da_audit_passes(tmp_path: Path) -> None:
    cap_dir = tmp_path / "da_ok"
    cap_dir.mkdir()
    capsule_path, _ = _attach_da_capsule(cap_dir, relay_priv_hex=TEST_RELAY_PRIV)
    result = _verify_with_policy(
        capsule_path,
        required_level="proof_only",
        trusted_relay_keys=_default_trusted_relays(),
    )
    assert result["da_audit_verified"]
    assert result["status"] == "POLICY_SELF_REPORTED"
    assert result.get("warnings", []) == ["W_POLICY_SELF_REPORTED"]


def test_verify_capsule_da_audit_detects_corruption(tmp_path: Path) -> None:
    cap_dir = tmp_path / "da_corrupt"
    cap_dir.mkdir()
    capsule_path, archive_dir = _attach_da_capsule(cap_dir, sample_k=-1, relay_priv_hex=TEST_RELAY_PRIV)
    chunk0 = archive_dir / "chunk_0.json"
    values = json.loads(chunk0.read_text())
    values[0] += 1
    chunk0.write_text(json.dumps(values))
    result = _verify_with_policy(
        capsule_path,
        required_level="proof_only",
        trusted_relay_keys=_default_trusted_relays(),
    )
    assert "E065_CHUNK_ROOT_MISMATCH" in result.get("warnings", [])


def test_verify_capsule_detects_row_commitment_root_mismatch(tmp_path: Path) -> None:
    cap_dir = tmp_path / "row_root_mismatch"
    cap_dir.mkdir()
    capsule_path, _ = _attach_da_capsule(cap_dir, relay_priv_hex=TEST_RELAY_PRIV)
    proof_path = cap_dir / "geom_proof.json"
    proof = json.loads(proof_path.read_text())
    proof["row_commitment"]["params"]["root"] = "00" * 32
    proof_path.write_text(json.dumps(proof))
    _sync_capsule_after_proof_change(capsule_path, proof_path)
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            capsule_path,
            required_level="proof_only",
        )
    assert "E053" in str(excinfo.value)


def test_verify_capsule_detects_chunk_len_mismatch(tmp_path: Path) -> None:
    cap_dir = tmp_path / "chunk_len_mismatch"
    cap_dir.mkdir()
    capsule_path, _ = _attach_da_capsule(cap_dir, relay_priv_hex=TEST_RELAY_PRIV)
    proof_path = cap_dir / "geom_proof.json"
    proof = json.loads(proof_path.read_text())
    params = proof["row_commitment"]["params"]
    params["chunk_len"] = int(params.get("chunk_len", 0)) + 1
    proof_path.write_text(json.dumps(proof))
    _sync_capsule_after_proof_change(capsule_path, proof_path)
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            capsule_path,
            required_level="proof_only",
        )
    assert "E053" in str(excinfo.value)


# =============================================================================
# Signed DA Challenge v2 Tests
# =============================================================================


def _attach_da_capsule_v2(
    tmpdir: Path,
    sample_k: int = 2,
    issuer_priv_hex: str | None = None,
    attach_authorship: bool = True,
    wrong_commit_root: bool = False,
    wrong_payload_hash: bool = False,
) -> tuple[Path, Path]:
    """Attach a v2 signed DA challenge to a capsule."""
    cap_path, _, _, _ = _write_capsule(tmpdir)
    capsule = json.loads(cap_path.read_text())
    capsule["verification_profile"] = "FULL"
    proof_path = tmpdir / "geom_proof.json"
    proof = json.loads(proof_path.read_text())
    rc_params = (proof.get("row_commitment") or {}).get("params") or {}
    archive_src = Path(rc_params.get("archive_root") or "")
    if not archive_src.exists():
        raise FileNotFoundError("row commitment archive missing for test fixture")
    archive_dir = tmpdir / "row_archive_da_v2"
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    shutil.copytree(archive_src, archive_dir)
    raw_handles = [str(h) for h in rc_params.get("chunk_handles", [])]
    chunk_roots_hex = list(rc_params.get("chunk_roots_hex", []))
    chunk_roots = [bytes.fromhex(h) for h in chunk_roots_hex]
    chunk_len = int(rc_params.get("chunk_len") or 0)
    chunk_tree_arity = int(rc_params.get("chunk_tree_arity") or 2)
    root = root_from_levels(build_kary_levels(chunk_roots, chunk_tree_arity))
    rel_path = archive_dir.relative_to(tmpdir)
    chunk_roots_path = archive_dir / "chunk_roots.json"
    chunk_roots_path.write_text(json.dumps(chunk_roots_hex, indent=2))
    chunk_roots_bin_path = archive_dir / "chunk_roots.bin"
    chunk_roots_bin_path.write_bytes(b"".join(bytes.fromhex(h) for h in chunk_roots_hex))
    chunk_roots_digest = hashlib.sha256(chunk_roots_bin_path.read_bytes()).hexdigest()
    manifest_handles = [
        {
            "id": idx,
            "uri": Path(handle).name,
            "sha256": "",
            "size": 0,
            "content_type": "application/octet-stream",
        }
        for idx, handle in enumerate(raw_handles)
    ]
    row_archive_info = {
        "mode": "LOCAL_FILE",
        "path": rel_path.as_posix(),
        "abs_path": str(archive_dir),
        "chunk_handles": manifest_handles,
        "chunk_roots_hex": chunk_roots_hex,
        "chunk_tree_arity": chunk_tree_arity,
        "chunk_roots_digest": chunk_roots_digest,
        "chunk_roots_bin_path": str(chunk_roots_bin_path.relative_to(tmpdir)),
        "chunk_roots_bin_abs": str(chunk_roots_bin_path),
        "chunk_roots_path": str(chunk_roots_path.relative_to(tmpdir)),
        "chunk_roots_abs": str(chunk_roots_path),
    }
    artifacts = capsule.setdefault("artifacts", {})
    artifacts["row_archive"] = {
        "mode": "LOCAL_FILE",
        "path": rel_path.as_posix(),
    }
    capsule["row_archive"] = row_archive_info
    capsule["row_index_ref"] = {
        "commitment_type": "merkle_root",
        "commitment": root.hex(),
        "tree_arity": chunk_tree_arity,
        "proof_fetch_rule_id": "local_test",
        "pointer": {"path": rel_path.as_posix()},
    }
    capsule["chunk_meta"] = {
        "num_chunks": len(raw_handles),
        "chunk_len": chunk_len,
        "chunk_size_bytes": chunk_len * 8,
        "data_length_bytes": chunk_len * len(raw_handles) * 8,
        "chunking_rule_id": "fixed_range_v1",
    }
    effective_k = sample_k if sample_k >= 0 else len(raw_handles)
    capsule["da_policy"] = {
        "policy_id": "da_local_test",
        "k_samples": effective_k,
        "provider_timeout_ms": 1000,
        "provider_retry_count": 1,
        "provider": {
            "mode": "LOCAL_FILE",
            "archive_root": str(archive_dir),
        },
    }
    _refresh_statement(capsule)
    # Update hashes but don't add a DA challenge yet
    _refresh_capsule_header(capsule)

    # Build v2 signed challenge with bindings
    header = capsule.get("header", {})
    row_commitment = header.get("row_commitment", {})
    commit_root = row_commitment.get("root", root.hex())
    payload_hash = header.get("payload_hash", "00" * 32)

    # Optionally use wrong bindings for testing
    if wrong_commit_root:
        commit_root = "ff" * 32
    if wrong_payload_hash:
        payload_hash = "ff" * 32

    da_challenge = build_signed_da_challenge(
        commit_root=commit_root,
        payload_hash=payload_hash,
        k_samples=effective_k,
        chunk_len=chunk_len,
        chunk_tree_arity=chunk_tree_arity,
        issuer_key_id="test_relay",
    )

    # Sign challenge if private key provided
    if issuer_priv_hex:
        da_challenge = sign_da_challenge(da_challenge, issuer_priv_hex)

    capsule["da_challenge"] = da_challenge
    # Set the challenge hash using v2 hash function
    capsule.setdefault("header", {}).setdefault("da_ref", {})["challenge_hash"] = hash_signed_da_challenge(da_challenge)
    # Recalculate header hash and capsule hash (but NOT refresh_da_challenge which uses v1)
    _refresh_capsule_header(capsule)
    cap_path.write_text(json.dumps(capsule))
    if attach_authorship:
        _attach_authorship(cap_path, DEFAULT_AUTH_PRIV)
    return cap_path, archive_dir


def test_full_with_signed_da_challenge_v2(tmp_path: Path) -> None:
    """V2 signed DA challenge with correct bindings passes FULL verification."""
    cap_dir = tmp_path / "full_signed_v2"
    cap_dir.mkdir()
    cap_path, _ = _attach_da_capsule_v2(cap_dir, issuer_priv_hex=TEST_RELAY_PRIV)
    policy_path = _policy_path_for_capsule(cap_path)
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    proof_path = cap_dir / "policy_proof.json"
    _write_policy_proof(proof_path, policy_hash)
    result = _verify_with_policy(
        cap_path,
        required_level="full",
        policy_path=policy_path,
        policy_proof_path=proof_path,
        policy_registry_root=policy_hash,
    )
    assert result["da_audit_verified"]


def test_full_rejects_v2_challenge_wrong_commit_root(tmp_path: Path) -> None:
    """V2 challenge with wrong commit_root binding is rejected."""
    cap_dir = tmp_path / "full_wrong_root"
    cap_dir.mkdir()
    cap_path, _ = _attach_da_capsule_v2(
        cap_dir, issuer_priv_hex=TEST_RELAY_PRIV, wrong_commit_root=True
    )
    policy_path = _policy_path_for_capsule(cap_path)
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    proof_path = cap_dir / "policy_proof.json"
    _write_policy_proof(proof_path, policy_hash)
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            cap_path,
            required_level="full",
            policy_path=policy_path,
            policy_proof_path=proof_path,
            policy_registry_root=policy_hash,
        )
    assert "E077" in str(excinfo.value)


def test_full_rejects_v2_challenge_wrong_payload_hash(tmp_path: Path) -> None:
    """V2 challenge with wrong payload_hash binding is rejected."""
    cap_dir = tmp_path / "full_wrong_payload"
    cap_dir.mkdir()
    cap_path, _ = _attach_da_capsule_v2(
        cap_dir, issuer_priv_hex=TEST_RELAY_PRIV, wrong_payload_hash=True
    )
    policy_path = _policy_path_for_capsule(cap_path)
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    proof_path = cap_dir / "policy_proof.json"
    _write_policy_proof(proof_path, policy_hash)
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            cap_path,
            required_level="full",
            policy_path=policy_path,
            policy_proof_path=proof_path,
            policy_registry_root=policy_hash,
        )
    assert "E077" in str(excinfo.value)


def test_full_rejects_unsigned_v2_challenge(tmp_path: Path) -> None:
    """V2 challenge without signature is rejected for FULL verification."""
    cap_dir = tmp_path / "full_unsigned_v2"
    cap_dir.mkdir()
    # No issuer_priv_hex means the challenge won't be signed
    cap_path, _ = _attach_da_capsule_v2(cap_dir, attach_authorship=False)
    policy_path = _policy_path_for_capsule(cap_path)
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    proof_path = cap_dir / "policy_proof.json"
    _write_policy_proof(proof_path, policy_hash)
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            cap_path,
            required_level="full",
            policy_path=policy_path,
            policy_proof_path=proof_path,
            policy_registry_root=policy_hash,
        )
    assert "E073" in str(excinfo.value)


def test_full_rejects_v1_downgrade(tmp_path: Path) -> None:
    """V1 unsigned challenge is rejected at FULL level (downgrade protection)."""
    cap_dir = tmp_path / "full_v1_downgrade"
    cap_dir.mkdir()
    # Use v1 challenge helper (not v2)
    cap_path, _ = _attach_da_capsule(cap_dir, attach_authorship=False)
    policy_path = _policy_path_for_capsule(cap_path)
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    proof_path = cap_dir / "policy_proof.json"
    _write_policy_proof(proof_path, policy_hash)
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            cap_path,
            required_level="full",
            policy_path=policy_path,
            policy_proof_path=proof_path,
            policy_registry_root=policy_hash,
        )
    # V1 unsigned challenge must be rejected
    assert "E073" in str(excinfo.value)


def test_full_rejects_sig_over_modified_fields(tmp_path: Path) -> None:
    """Signature over modified challenge fields is rejected (proves sig covers params)."""
    cap_dir = tmp_path / "full_sig_modified"
    cap_dir.mkdir()
    cap_path, _ = _attach_da_capsule_v2(cap_dir, issuer_priv_hex=TEST_RELAY_PRIV)

    # Modify the challenge after signing - change 'k' field
    capsule = json.loads(cap_path.read_text())
    challenge = capsule.get("da_challenge", {})
    original_k = challenge.get("k", 2)
    challenge["k"] = original_k + 100  # Modify after signing
    cap_path.write_text(json.dumps(capsule))

    policy_path = _policy_path_for_capsule(cap_path)
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    proof_path = cap_dir / "policy_proof.json"
    _write_policy_proof(proof_path, policy_hash)
    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            cap_path,
            required_level="full",
            policy_path=policy_path,
            policy_proof_path=proof_path,
            policy_registry_root=policy_hash,
        )
    # Signature verification should fail because k was modified
    assert "E073" in str(excinfo.value) or "E072" in str(excinfo.value)


def test_full_rejects_untrusted_issuer_key(tmp_path: Path) -> None:
    """Valid signature from untrusted issuer key_id is rejected."""
    # Generate a different key pair (not in trusted registry)
    coincurve = pytest.importorskip("coincurve")
    UNTRUSTED_PRIV = "4" * 64  # Different from TEST_RELAY_PRIV

    cap_dir = tmp_path / "full_untrusted_issuer"
    cap_dir.mkdir()

    # Use the v2 helper but with a custom issuer key - reuse the helper's structure
    # but manually create to use untrusted key
    cap_path, archive_dir = _attach_da_capsule_v2(cap_dir, issuer_priv_hex=TEST_RELAY_PRIV)
    capsule = json.loads(cap_path.read_text())

    # Now replace the challenge with one signed by an untrusted key
    header = capsule.get("header", {})
    row_commitment = header.get("row_commitment", {})
    commit_root = row_commitment.get("root", "00" * 32)
    payload_hash = header.get("payload_hash", "00" * 32)
    old_challenge = capsule.get("da_challenge", {})

    da_challenge = build_signed_da_challenge(
        commit_root=commit_root,
        payload_hash=payload_hash,
        k_samples=old_challenge.get("k", 2),
        chunk_len=old_challenge.get("chunk_len", 64),
        chunk_tree_arity=old_challenge.get("chunk_tree_arity", 16),
        issuer_key_id="untrusted_issuer",  # Not in trusted list
    )
    da_challenge = sign_da_challenge(da_challenge, UNTRUSTED_PRIV)

    capsule["da_challenge"] = da_challenge
    # Recalculate header and capsule hashes properly
    _refresh_capsule_header(capsule)
    cap_path.write_text(json.dumps(capsule))
    _attach_authorship(cap_path, DEFAULT_AUTH_PRIV)

    policy_path = _policy_path_for_capsule(cap_path)
    policy_hash = hashlib.sha256(Path(policy_path).read_bytes()).hexdigest()
    proof_path = cap_dir / "policy_proof.json"
    _write_policy_proof(proof_path, policy_hash)

    # Provide trusted keys that don't include the untrusted issuer
    trusted_keys = {"test_relay": _test_relay_pubkey()}  # Does NOT include "untrusted_issuer"

    with pytest.raises(ValueError) as excinfo:
        _verify_with_policy(
            cap_path,
            required_level="full",
            policy_path=policy_path,
            policy_proof_path=proof_path,
            policy_registry_root=policy_hash,
            trusted_relay_keys=trusted_keys,
        )
    # Must reject because issuer key_id not in trusted list
    assert "E073" in str(excinfo.value)
