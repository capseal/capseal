"""capsule-bench CLI."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Tuple
from urllib import error as url_error
from urllib import request as url_request

import click

from .artifacts import ArtifactPublisher, records_to_json, save_manifest
from .manifests import ManifestBundle, collect_manifests, write_manifest_signature
from .packing import create_capsulepack

ROOT = Path(__file__).resolve().parents[1]
RUN_META_SCHEMA = "capsule_bench_run_meta_v1"


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")



def _compute_anchor(bundle: ManifestBundle) -> str:
    return bundle.anchor_ref


def _hash_policy(policy_path: Path) -> str:
    import hashlib

    hasher = hashlib.sha256()
    hasher.update(policy_path.read_bytes())
    return hasher.hexdigest()


def _load_manifest_signer_key(spec: str) -> str:
    """Resolve a manifest signer key spec (path or hex string) to raw hex."""

    candidate = Path(spec)
    if candidate.exists():
        data = candidate.read_text().strip()
    else:
        data = spec.strip()
    if data.startswith("0x"):
        data = data[2:]
    if not data:
        raise ValueError("manifest signer key is empty")
    # Validate hex early so we can give a nice error message before signing
    bytes.fromhex(data)
    return data


def _run_pipeline(
    pipeline_args: Tuple[str, ...],
    *,
    backend: str,
    policy_path: Path,
    policy_id: str,
    policy_version: str,
    output_dir: Path,
    anchor_ref: str,
    trace_id: str,
    track_id: str,
    docker_image_digest: str | None,
    events_log: Path | None,
    private_key: Path | None,
    verification_profile: str,
    da_challenge_file: Path | None,
    allow_insecure_da: bool,
    da_relay_url: str | None,
    da_relay_token: str | None,
) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_pipeline.py"),
        "--backend",
        backend,
        "--policy",
        str(policy_path),
        "--policy-id",
        policy_id,
        "--policy-version",
        policy_version,
        "--output-dir",
        str(output_dir),
        "--trace-id",
        trace_id,
        "--anchor-ref",
        anchor_ref,
        "--anchor-rule-id",
        "capsule_bench_manifest_v1",
        "--track-id",
        track_id,
        "--verification-profile",
        verification_profile,
    ]
    if docker_image_digest:
        cmd.extend(["--docker-image-digest", docker_image_digest])
    if events_log:
        cmd.extend(["--events-log", str(events_log)])
    if private_key:
        cmd.extend(["--private-key", str(private_key)])
    if da_challenge_file:
        cmd.extend(["--da-challenge-file", str(da_challenge_file)])
    if da_relay_url:
        cmd.extend(["--da-relay-url", da_relay_url])
    if da_relay_token:
        cmd.extend(["--da-relay-token", da_relay_token])
    if allow_insecure_da:
        cmd.append("--allow-insecure-da-challenge")
    cmd.extend(pipeline_args)
    subprocess.run(cmd, check=True)


def _load_capsule_path(run_dir: Path) -> Path:
    pipeline_dir = run_dir / "pipeline"
    return pipeline_dir / "strategy_capsule.json"


def _register_relay_ingest(relay_base: str, run_id: str, admin_token: str | None) -> dict[str, Any]:
    relay_base = relay_base.rstrip("/")
    url = f"{relay_base}/runs/{run_id}/ingest_token"
    payload = json.dumps({}).encode("utf-8")
    req = url_request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    if admin_token:
        req.add_header("Authorization", f"Bearer {admin_token}")
    try:
        with url_request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
    except url_error.HTTPError as exc:  # pragma: no cover - network dependent
        raise click.ClickException(f"relay registration failed ({exc.code}): {exc.reason}") from exc
    except url_error.URLError as exc:  # pragma: no cover - network dependent
        raise click.ClickException(f"relay registration failed: {exc.reason}") from exc
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:  # pragma: no cover - unexpected response
        raise click.ClickException("relay registration returned invalid JSON") from exc
    required = {"ingest_url", "token", "expires_at"}
    if not required.issubset(data):
        raise click.ClickException("relay registration missing fields")
    return data


def _submit_artifact_manifest(
    relay_base: str | None,
    run_id: str,
    artifacts: list[dict[str, Any]],
    admin_token: str | None,
) -> None:
    if not relay_base:
        return
    relay_base = relay_base.rstrip("/")
    url = f"{relay_base}/runs/{run_id}/artifacts"
    payload = json.dumps({"artifacts": artifacts}).encode("utf-8")
    req = url_request.Request(url, data=payload, method="PUT")
    req.add_header("Content-Type", "application/json")
    if admin_token:
        req.add_header("Authorization", f"Bearer {admin_token}")
    try:
        with url_request.urlopen(req, timeout=10):
            return
    except url_error.HTTPError as exc:  # pragma: no cover - network dependent
        detail = exc.read().decode("utf-8", "ignore") if hasattr(exc, "read") else exc.reason
        raise click.ClickException(
            f"artifact manifest registration failed ({exc.code}): {detail or exc.reason}"
        ) from exc
    except url_error.URLError as exc:  # pragma: no cover - network dependent
        raise click.ClickException(f"artifact manifest registration failed: {exc.reason}") from exc


@click.group()
def cli() -> None:
    """capsule-bench CLI entrypoint."""


@cli.command("run")
@click.option("--backend", default="geom", help="Trace adapter backend id.")
@click.option("--policy", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--policy-id", required=True, help="Policy identifier embedded in the capsule.")
@click.option("--policy-version", default="unspecified")
@click.option("--track-id", required=True, help="Benchmark track identifier (policy track).")
@click.option("--docker-image-digest", type=str, help="Docker image digest used for deterministic builds.")
@click.option("--output-root", type=click.Path(path_type=Path), default=Path("out/capsule_runs"))
@click.option("--run-id", type=str, help="Custom run identifier (default: timestamp slug).")
@click.option("--trace-id", type=str, help="Optional trace id override.")
@click.option(
    "--private-key",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to secp256k1 private key used to sign the capsule.",
)
@click.option(
    "--manifest-signer-id",
    type=str,
    envvar="CAPSULE_MANIFEST_SIGNER_ID",
    help="Identifier for the manifest signing key (must match verifier trust root).",
)
@click.option(
    "--manifest-signer-key",
    type=str,
    envvar="CAPSULE_MANIFEST_SIGNER_KEY",
    help="Path to or literal hex of the manifest signing secp256k1 key.",
)
@click.option("--relay-base", type=str, help="Relay base URL (e.g. https://capsuletech.onrender.com)")
@click.option(
    "--relay-admin-token",
    type=str,
    envvar="CAPSULE_RELAY_ADMIN_TOKEN",
    help="Admin token used to request relay ingest credentials.",
)
@click.option(
    "--verification-profile",
    type=click.Choice(["proof_only", "policy_enforced", "full"], case_sensitive=False),
    default="full",
)
@click.option(
    "--da-challenge-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a relay-issued DA challenge JSON to embed in the capsule.",
)
@click.option(
    "--allow-insecure-da-challenge/--no-allow-insecure-da-challenge",
    default=False,
    help="Allow capsule-bench to mint an insecure local DA challenge if no relay challenge is provided.",
)
@click.option("--da-relay-url", type=str, help="Base URL for the DA relay commit/challenge API.")
@click.option(
    "--da-relay-token",
    type=str,
    envvar="CAPSULE_DA_RELAY_TOKEN",
    help="Bearer token used when contacting the DA relay.",
)
@click.argument("pipeline_args", nargs=-1, type=str)
def run_command(
    backend: str,
    policy: Path,
    policy_id: str,
    policy_version: str,
    track_id: str,
    docker_image_digest: str | None,
    output_root: Path,
    run_id: str | None,
    trace_id: str | None,
    private_key: Path | None,
    manifest_signer_id: str | None,
    manifest_signer_key: str | None,
    relay_base: str | None,
    relay_admin_token: str | None,
    verification_profile: str,
    da_challenge_file: Path | None,
    allow_insecure_da_challenge: bool,
    da_relay_url: str | None,
    da_relay_token: str | None,
    pipeline_args: Tuple[str, ...],
) -> None:
    """Execute the prover pipeline and capture manifests."""

    run_id = run_id or _timestamp_slug()
    trace_id = trace_id or run_id
    run_dir = (output_root / run_id).resolve()
    pipeline_dir = run_dir / "pipeline"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_root = run_dir / "manifests"
    manifest_bundle = collect_manifests(manifest_root)
    manifest_signature_path: Path | None = None
    if manifest_signer_key:
        if not manifest_signer_id:
            raise click.ClickException("--manifest-signer-id is required when signing manifests")
        try:
            key_hex = _load_manifest_signer_key(manifest_signer_key)
            manifest_signature_path = write_manifest_signature(
                manifest_bundle,
                signer_id=manifest_signer_id,
                private_key_hex=key_hex,
            )
            click.echo(f"manifest bundle signed by {manifest_signer_id}")
        except Exception as exc:  # pragma: no cover - depends on coincurve/file IO
            raise click.ClickException(f"failed to sign manifest bundle: {exc}") from exc
    elif (verification_profile or "").lower() != "proof_only":
        click.echo(
            "warning: manifest signer key not provided; policy enforcement will fail closed",
            err=True,
        )
    anchor_ref = _compute_anchor(manifest_bundle)
    policy_copy = run_dir / "policy.json"
    shutil.copy2(policy, policy_copy)
    events_path = run_dir / "events.jsonl"

    relay_registration = None
    if relay_base:
        relay_registration = _register_relay_ingest(relay_base, run_id, relay_admin_token)
        click.echo(f"relay ingest token issued (expires at {relay_registration['expires_at']})")

    try:
        _run_pipeline(
            pipeline_args,
            backend=backend,
            policy_path=policy_copy,
            policy_id=policy_id,
            policy_version=policy_version,
            output_dir=pipeline_dir,
            anchor_ref=anchor_ref,
            trace_id=trace_id,
            track_id=track_id,
            docker_image_digest=docker_image_digest,
            events_log=events_path,
            private_key=private_key,
            verification_profile=verification_profile,
            da_challenge_file=da_challenge_file,
            allow_insecure_da=allow_insecure_da_challenge,
            da_relay_url=da_relay_url,
            da_relay_token=da_relay_token,
        )
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(f"pipeline execution failed: {exc}") from exc

    capsule_path = _load_capsule_path(run_dir)
    da_challenge_path = pipeline_dir / "da_challenge.json"
    run_meta = {
        "schema": RUN_META_SCHEMA,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "backend": backend,
        "policy_id": policy_id,
        "policy_version": policy_version,
        "policy_hash": _hash_policy(policy_copy),
        "policy_copy": str(policy_copy),
        "pipeline_output": str(pipeline_dir),
        "capsule_path": str(capsule_path),
        "manifests_root": str(manifest_bundle.base_dir),
        "manifest_hashes": manifest_bundle.hashes,
        "anchor_ref": anchor_ref,
        "events_path": str(events_path),
        "manifest_root": str(manifest_root),
        "track_id": track_id,
        "docker_image_digest": docker_image_digest,
        "verification_profile": verification_profile,
    }
    if manifest_signature_path:
        run_meta["manifest_signature"] = {
            "path": str(manifest_signature_path),
            "signer_id": manifest_signer_id,
        }
    if da_challenge_path.exists():
        run_meta["da_challenge_path"] = str(da_challenge_path)
    if relay_registration:
        run_meta["relay_ingest_url"] = relay_registration["ingest_url"]
        run_meta["relay_ingest_token"] = relay_registration["token"]
        run_meta["relay_token_expires_at"] = relay_registration["expires_at"]
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    click.echo(f"capsule-bench run completed: {run_dir}")


@cli.command("pack")
@click.option("--run-dir", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--pack-name", type=str, help="Override output tgz name (without extension).")
@click.option(
    "--upload-artifacts/--no-upload-artifacts",
    default=False,
    help="Upload capsulepack artifacts to configured storage (R2/local).",
)
@click.option("--relay-base", type=str, help="Relay base URL for artifact manifest registration.")
@click.option(
    "--relay-admin-token",
    type=str,
    envvar="CAPSULE_RELAY_ADMIN_TOKEN",
    help="Admin token for relay artifact registration.",
)
def pack_command(
    run_dir: Path,
    pack_name: str | None,
    upload_artifacts: bool,
    relay_base: str | None,
    relay_admin_token: str | None,
) -> None:
    """Assemble capsulepack.tgz from a previous run."""

    run_meta_path = run_dir / "run_meta.json"
    if not run_meta_path.exists():
        raise click.ClickException(f"missing run_meta.json in {run_dir}")
    run_meta = json.loads(run_meta_path.read_text())
    try:
        pack_dir, tar_path = create_capsulepack(run_meta_path, pack_name=pack_name)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"capsulepack assembled at {pack_dir}")
    click.echo(f"archive written to {tar_path}")

    if upload_artifacts:
        publisher = ArtifactPublisher()
        run_id = run_meta.get("run_id") or pack_name or "capsule_run"
        records = publisher.publish(run_id, pack_dir, tar_path)
        manifest = records_to_json(records)
        manifest_path = run_dir / "artifact_manifest.json"
        save_manifest(manifest, manifest_path)
        click.echo(f"uploaded {len(manifest)} artifacts (manifest at {manifest_path})")
        _submit_artifact_manifest(relay_base, run_id, manifest, relay_admin_token)
        if relay_base:
            click.echo("artifact manifest registered with relay")


def main() -> None:
    cli(prog_name="capsule-bench")


if __name__ == "__main__":
    main()
