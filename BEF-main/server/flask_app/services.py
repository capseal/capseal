"""Helper utilities for the CapSeal Flask API."""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Tuple, Type, TypeVar

from flask import current_app, has_app_context
from pydantic import BaseModel

from server.event_store import EventStore

from .errors import APIError
from .models import (
    AuditRequest,
    EmitRequest,
    FetchRequest,
    ReplayRequest,
    RowRequest,
    RunRequest,
    SandboxTestRequest,
    VerifyRequest,
    DatasetMapping,
    PolicyRef,
)

ModelT = TypeVar("ModelT", bound=BaseModel)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
INLINE_POLICY_DIR = PROJECT_ROOT / "server_data" / "policies"
DA_PROFILE_DIR = PROJECT_ROOT / "da_profiles"


@dataclass
class CLIResult:
    """Represents the outcome of invoking a CapSeal CLI helper."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    payload: Any | None = None

    def ok(self) -> bool:
        return self.returncode == 0


_event_store: EventStore | None = None


def _ensure_event_store(root: Path) -> EventStore:
    global _event_store
    if _event_store is None or _event_store.root != root:
        _event_store = EventStore(root)
        _event_store.load_existing()
    return _event_store


def _resolve_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return str(path)


def _dataset_flags(entries: Iterable[Any]) -> list[str]:
    flags: list[str] = []
    if not entries:
        return flags
    for entry in entries:
        spec: str | None = None
        if isinstance(entry, str):
            resolved = _resolve_path(entry) or entry
            spec = resolved
        elif isinstance(entry, dict):
            path = _resolve_path(entry.get("path")) or entry.get("path")
            dataset_id = entry.get("id")
            if dataset_id and path:
                spec = f"{dataset_id}={path}"
            elif path:
                spec = path
        if spec:
            flags.extend(["-d", spec])
    return flags


def _policy_path(ref: PolicyRef) -> str:
    if ref.policyPath:
        return _resolve_path(ref.policyPath) or ref.policyPath
    if ref.policy is None:
        raise APIError(400, "policyPath required for this operation")
    INLINE_POLICY_DIR.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(ref.policy, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    fname = f"inline_{(ref.policyId or 'policy').replace(' ', '_')}_{digest[:12]}.json"
    path = INLINE_POLICY_DIR / fname
    if not path.exists():
        path.write_text(json.dumps(ref.policy, indent=2))
    return str(path)


def _load_da_profile(
    profile_id: str | None,
    profile_path: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    path: Path | None = None
    if profile_path:
        path = Path(profile_path).expanduser()
        if not path.is_absolute():
            path = (PROJECT_ROOT / profile_path).resolve()
    elif profile_id:
        path = DA_PROFILE_DIR / f"{profile_id}.json"
    if not path:
        return None, None
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise APIError(404, f"DA profile not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise APIError(400, f"Invalid DA profile JSON: {path}") from exc
    return data, profile_id or path.stem


def _da_env(profile: dict[str, Any], profile_id: str | None) -> dict[str, str]:
    env: dict[str, str] = {}
    relays = profile.get("trusted_relays")
    if isinstance(relays, list) and relays:
        env["CAPSULE_TRUSTED_RELAYS"] = ",".join(str(relay) for relay in relays)
    relay_keys = profile.get("trusted_relay_keys") or profile.get("relay_keys")
    if isinstance(relay_keys, dict) and relay_keys:
        env["CAPSULE_TRUSTED_RELAY_KEYS"] = ",".join(f"{rid}={key}" for rid, key in relay_keys.items())
    relays_root = profile.get("relays_root") or profile.get("trusted_relays_root")
    if isinstance(relays_root, str) and relays_root:
        env["CAPSULE_TRUSTED_RELAYS_ROOT"] = relays_root
    manifest_signers = profile.get("manifest_signers")
    if isinstance(manifest_signers, dict) and manifest_signers:
        env["CAPSULE_TRUSTED_MANIFEST_SIGNERS"] = ",".join(
            f"{sid}={key}" for sid, key in manifest_signers.items()
        )
    if profile_id:
        env["CAPSEAL_DA_PROFILE_ID"] = profile_id
    return env


def _dataset_entries(mappings: Iterable[DatasetMapping]) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for mapping in mappings:
        if not mapping.path:
            raise APIError(400, f"dataset {mapping.id} missing path; use fetch endpoint first")
        entries.append({"id": mapping.id, "path": mapping.path})
    return entries


def _json_or_none(output: str) -> Any | None:
    text = output.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _run(
    cmd: list[str],
    expect_json: bool = False,
    env_overrides: dict[str, str] | None = None,
) -> CLIResult:
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    if env_overrides:
        env.update({k: v for k, v in env_overrides.items() if v is not None})
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    payload = _json_or_none(proc.stdout) if expect_json else None
    return CLIResult(command=cmd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, payload=payload)


def fetch_dataset_task(params: dict[str, Any]) -> CLIResult:
    script = PROJECT_ROOT / "scripts" / "fetch_dataset.py"
    if not script.exists():
        return CLIResult([str(script)], 1, "", f"Fetch script missing: {script}")

    cmd: list[str] = [
        PYTHON,
        str(script),
        "--url",
        params["url"],
        "--policy",
        _resolve_path(params["policy"]) or params["policy"],
        "--policy-id",
        params["policy_id"],
    ]

    dataset_id = params.get("dataset_id")
    if dataset_id:
        cmd.extend(["--dataset-id", dataset_id])

    policy_version = params.get("policy_version")
    if policy_version:
        cmd.extend(["--policy-version", policy_version])

    output_dir = params.get("output")
    if output_dir:
        cmd.extend(["--output-dir", _resolve_path(output_dir) or output_dir])

    tree_arity = params.get("dataset_tree_arity")
    if tree_arity:
        cmd.extend(["--dataset-tree-arity", str(tree_arity)])

    return _run(cmd, expect_json=True, env_overrides=env)


def run_capsule_task(params: dict[str, Any]) -> CLIResult:
    cmd: list[str] = [
        PYTHON,
        "-m",
        "bef_zk.capsule.cli",
        "run",
        "--policy",
        _resolve_path(params["policy"]) or params["policy"],
        "--policy-id",
        params["policy_id"],
        "--trace-id",
        params.get("trace_id", "api_run"),
        "--json",
    ]

    output_dir = params.get("output")
    if output_dir:
        cmd.extend(["--output", _resolve_path(output_dir) or output_dir])

    steps = params.get("steps")
    if steps is not None:
        cmd.extend(["--steps", str(steps)])

    queries = params.get("queries")
    if queries is not None:
        cmd.extend(["--queries", str(queries)])

    challenges = params.get("challenges")
    if challenges is not None:
        cmd.extend(["--challenges", str(challenges)])

    backend = params.get("backend")
    if backend:
        cmd.extend(["--backend", backend])

    if params.get("sandbox"):
        cmd.append("--sandbox")
        memory_mb = params.get("sandbox_memory")
        if memory_mb:
            cmd.extend(["--sandbox-memory", str(memory_mb)])
        timeout = params.get("sandbox_timeout")
        if timeout:
            cmd.extend(["--sandbox-timeout", str(timeout)])
        if params.get("sandbox_allow_network"):
            cmd.append("--sandbox-allow-network")
        else:
            cmd.append("--sandbox-deny-network")

    cmd.extend(_dataset_flags(params.get("datasets", [])))

    return _run(cmd, expect_json=True)


def emit_capsule_task(params: dict[str, Any]) -> CLIResult:
    cmd: list[str] = [
        PYTHON,
        "-m",
        "bef_zk.capsule.cli",
        "emit",
        "--out",
        _resolve_path(params["out"]) or params["out"],
    ]

    capsule = params.get("capsule")
    if capsule:
        cmd.extend(["--capsule", _resolve_path(capsule) or capsule])

    source = params.get("source") or params.get("receipt")
    if source:
        cmd.extend(["--source", _resolve_path(source) or source])

    artifacts = params.get("artifacts")
    if artifacts:
        cmd.extend(["--artifacts", _resolve_path(artifacts) or artifacts])

    archive = params.get("archive")
    if archive:
        cmd.extend(["--archive", _resolve_path(archive) or archive])

    policy = params.get("policy")
    if policy:
        cmd.extend(["--policy", _resolve_path(policy) or policy])

    profile = params.get("profile")
    if profile:
        cmd.extend(["--profile", profile])

    manifests = params.get("manifests")
    if manifests:
        cmd.extend(["--manifests", _resolve_path(manifests) or manifests])

    return _run(cmd, expect_json=False)


def verify_capsule_task(params: dict[str, Any], env: dict[str, str] | None = None) -> CLIResult:
    cmd: list[str] = [
        PYTHON,
        "-m",
        "bef_zk.capsule.cli",
        "verify",
        _resolve_path(params["capsule"]) or params["capsule"],
        "--json",
    ]

    mode = params.get("mode")
    if mode:
        cmd.extend(["--mode", mode])

    policy = params.get("policy")
    if policy:
        cmd.extend(["--policy", _resolve_path(policy) or policy])

    manifests = params.get("manifests")
    if manifests:
        cmd.extend(["--manifests", _resolve_path(manifests) or manifests])

    cmd.extend(_dataset_flags(params.get("datasets", [])))

    return _run(cmd, expect_json=True, env_overrides=env)


def replay_capsule_task(params: dict[str, Any]) -> CLIResult:
    cmd: list[str] = [
        PYTHON,
        "-m",
        "bef_zk.capsule.cli",
        "replay",
        _resolve_path(params["capsule"]) or params["capsule"],
        "--json",
    ]

    tolerance = params.get("tolerance")
    if tolerance is not None:
        cmd.extend(["--tolerance", str(tolerance)])

    sample = params.get("sample")
    if sample:
        cmd.extend(["--sample", str(sample)])

    sample_seed = params.get("sample_seed")
    if sample_seed is not None:
        cmd.extend(["--sample-seed", str(sample_seed)])

    row_range = params.get("row_range")
    if row_range:
        cmd.extend(["--range", row_range])

    max_divergences = params.get("max_divergences")
    if max_divergences is not None:
        cmd.extend(["--max-divergences", str(max_divergences)])

    if params.get("until_diverge"):
        cmd.append("--until-diverge")

    cmd.extend(_dataset_flags(params.get("datasets", [])))

    return _run(cmd, expect_json=True)


def audit_capsule_task(params: dict[str, Any]) -> CLIResult:
    cmd: list[str] = [
        PYTHON,
        "-m",
        "bef_zk.capsule.cli",
        "audit",
        _resolve_path(params["capsule"]) or params["capsule"],
        "-f",
        params.get("format", "json"),
    ]

    if not params.get("verify", True):
        cmd.append("--no-verify")

    filter_type = params.get("filter_type")
    if filter_type:
        cmd.extend(["--filter-type", filter_type])

    from_seq = params.get("from_seq")
    if from_seq:
        cmd.extend(["--from-seq", str(from_seq)])

    to_seq = params.get("to_seq")
    if to_seq is not None:
        cmd.extend(["--to-seq", str(to_seq)])

    return _run(cmd, expect_json=params.get("format", "json") == "json")


def get_run_events(run_id: str, event_root: str) -> list[str]:
    root = Path(event_root).expanduser().resolve()
    if has_app_context():
        store = current_app.extensions.get("event_store")
        if store:
            # Reload to pick up freshly ingested runs
            store.load_existing()
            return store.history(run_id)
    store = _ensure_event_store(root)
    store.load_existing()
    return store.history(run_id)


def _coerce(model: Type[ModelT], value: ModelT | dict[str, Any] | None) -> ModelT:
    if isinstance(value, model):
        return value
    if isinstance(value, dict):
        return model.model_validate(value)
    raise TypeError(f"Expected {model.__name__} payload")


def capseal_fetch(req: FetchRequest | dict[str, Any]) -> CLIResult:
    req = _coerce(FetchRequest, req)
    policy_path = _policy_path(req.policy)
    params = {
        "url": str(req.url),
        "dataset_id": req.datasetId,
        "output": req.outputDir,
        "policy": policy_path,
        "policy_id": req.policy.policyId or req.datasetId,
        "policy_version": req.policy.policyVersion,
        "dataset_tree_arity": req.datasetTreeArity,
    }
    return fetch_dataset_task(params)


def capseal_run(req: RunRequest | dict[str, Any]) -> CLIResult:
    req = _coerce(RunRequest, req)
    policy_path = _policy_path(req.policy)
    dataset_entries = _dataset_entries(req.datasets)
    params = {
        "policy": policy_path,
        "policy_id": req.policyId,
        "trace_id": req.traceId,
        "output": req.outputDir,
        "steps": req.steps,
        "queries": req.queries,
        "challenges": req.challenges,
        "backend": req.backend,
        "datasets": dataset_entries,
        "sandbox": req.sandbox,
        "sandbox_memory": req.sandboxMemory,
        "sandbox_timeout": req.sandboxTimeout,
        "sandbox_allow_network": req.sandboxAllowNetwork,
    }
    return run_capsule_task(params)


def capseal_emit(req: EmitRequest | dict[str, Any]) -> CLIResult:
    req = _coerce(EmitRequest, req)
    params = {
        "out": req.outPath,
        "source": req.source,
        "capsule": req.capsulePath,
        "artifacts": req.artifactsDir,
        "archive": req.archiveDir,
        "policy": req.policyPath,
        "profile": req.profile,
        "manifests": req.manifestsDir,
    }
    return emit_capsule_task(params)


def capseal_verify(req: VerifyRequest | dict[str, Any]) -> CLIResult:
    req = _coerce(VerifyRequest, req)
    params = {
        "capsule": req.capsulePath,
        "mode": req.mode,
        "policy": req.policyPath,
        "datasets": _dataset_entries(req.datasets),
    }
    if req.manifestsDir:
        params["manifests"] = req.manifestsDir
    env = None
    profile_id = None
    if getattr(req, "daProfileId", None) or getattr(req, "daProfilePath", None):
        profile, profile_id = _load_da_profile(req.daProfileId, req.daProfilePath)
        if profile:
            env = _da_env(profile, profile_id)
    result = verify_capsule_task(params, env=env)
    payload = result.payload if isinstance(result.payload, dict) else None
    if payload is not None:
        payload.setdefault("da_verified", payload.get("da_audit_verified"))
        if profile_id:
            payload.setdefault("da_profile_id", profile_id)
    return result


def capseal_replay(req: ReplayRequest | dict[str, Any]) -> CLIResult:
    req = _coerce(ReplayRequest, req)
    params = {
        "capsule": req.capsulePath,
        "datasets": _dataset_entries(req.datasets),
        "tolerance": req.tolerance,
        "sample": req.sample,
        "sample_seed": req.sampleSeed,
        "row_range": req.range,
        "max_divergences": req.maxDivergences,
        "until_diverge": req.untilDiverge,
        "verbose": req.verbose,
    }
    return replay_capsule_task(params)


def capseal_row(req: RowRequest | dict[str, Any]) -> CLIResult:
    req = _coerce(RowRequest, req)
    cmd: list[str] = [
        PYTHON,
        "-m",
        "bef_zk.capsule.cli",
        "row",
        req.capsulePath,
        "--row",
        str(req.row),
        "--json",
    ]
    if req.schemaId:
        cmd.extend(["--schema-id", req.schemaId])
    return _run(cmd, expect_json=True)


def capseal_audit(req: AuditRequest | dict[str, Any]) -> CLIResult:
    req = _coerce(AuditRequest, req)
    params = {
        "capsule": req.capsulePath,
        "format": req.format,
        "verify": req.verifyChain,
        "filter_type": req.filterType,
        "from_seq": req.fromSeq,
        "to_seq": req.toSeq,
    }
    return audit_capsule_task(params)


def capseal_sandbox_status() -> CLIResult:
    cmd = [
        PYTHON,
        "-m",
        "bef_zk.capsule.cli",
        "sandbox",
        "status",
        "--json",
    ]
    return _run(cmd, expect_json=True)


def capseal_sandbox_test(req: SandboxTestRequest) -> CLIResult:
    cmd = [
        PYTHON,
        "-m",
        "bef_zk.capsule.cli",
        "sandbox",
        "test",
    ]
    if req.backend:
        cmd.extend(["--backend", req.backend])
    return _run(cmd, expect_json=False)
