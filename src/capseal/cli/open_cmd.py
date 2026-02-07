"""capsule open command - selective semantic replay for datasets."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import click

from bef_zk.stc.merkle import build_kary_levels, prove_kary
from bef_zk.policy import load_policy_config, PolicyError
from capseal.policy_openings import (
    resolve_policy_path_for_capsule,
    OpeningPolicyEnforcer,
)

from .cap_format import CapExtractionError, extract_cap_file


def _safe_rel_path(rel: str) -> Path:
    rel = rel.replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    pure = Path(*rel.split("/")) if rel else Path(".")
    if pure.is_absolute():
        raise ValueError(f"absolute paths not allowed in capsule: {rel!r}")
    if ".." in pure.parts:
        raise ValueError(f"path traversal detected in capsule: {rel!r}")
    return pure


def _load_capsule_with_artifacts(
    capsule_path: Path,
) -> Tuple[Dict[str, Any], Path, tempfile.TemporaryDirectory | None]:
    if capsule_path.suffix == ".cap":
        tmpdir = tempfile.TemporaryDirectory()
        extract_dir = Path(tmpdir.name)
        try:
            extract_cap_file(capsule_path, extract_dir)
        except CapExtractionError as exc:
            tmpdir.cleanup()
            raise RuntimeError(f"failed to extract capsule: {exc}") from exc
        capsule_json = extract_dir / "capsule.json"
        if not capsule_json.exists():
            tmpdir.cleanup()
            raise RuntimeError("capsule archive missing capsule.json")
        capsule_data = json.loads(capsule_json.read_text())
        return capsule_data, extract_dir, tmpdir
    capsule_data = json.loads(capsule_path.read_text())
    return capsule_data, capsule_path.parent, None


def _parse_dataset_args(values: Tuple[str, ...]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for spec in values:
        if not spec:
            continue
        if "=" in spec:
            ds_id, ds_path = spec.split("=", 1)
            ds_id = (ds_id or "").strip()
            raw_path = (ds_path or "").strip()
        else:
            raw_path = spec.strip()
            ds_id = Path(raw_path).name
        path_obj = Path(raw_path).expanduser().resolve()
        if ds_id:
            mapping[ds_id] = path_obj
    return mapping


def _load_dataset_manifest(base: Path, manifest_rel: str) -> dict:
    manifest_path = _safe_rel_path(manifest_rel)
    path = base / manifest_path
    if not path.exists():
        raise FileNotFoundError(f"dataset manifest missing: {path}")
    return json.loads(path.read_text())


def _compute_merkle_levels(entries: list[dict], arity: int) -> Tuple[list, list[bytes]]:
    if not entries:
        return [], []
    leaves = [bytes.fromhex(entry["sha256"]) for entry in entries]
    levels = build_kary_levels(leaves, arity)
    return levels, leaves


def _chunk_proof(levels: list, chunk_index: int, arity: int) -> list[list[str]]:
    if not levels:
        return []
    proof = prove_kary(levels, chunk_index, arity)
    formatted = []
    for level in proof:
        formatted.append([leaf.hex() for leaf in level])
    return formatted


def _relative_dataset_file(dataset_root: Path, rel_uri: str) -> Path:
    rel_path = Path(rel_uri.replace("\\", "/"))
    candidate = dataset_root / rel_path
    return candidate.resolve()


@click.command("open")
@click.argument("capsule", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--dataset",
    multiple=True,
    help="Dataset mapping id=path (repeatable). If omitted, chunk bytes are not opened.",
)
@click.option("--dataset-id", required=True, help="Dataset identifier to inspect")
@click.option("--chunk", type=int, help="Chunk index to open")
@click.option("--json", "output_json", is_flag=True, help="Output result as JSON")
@click.option("--policy", type=click.Path(path_type=Path), help="Policy JSON file for governed openings")
@click.option("--ticket", type=click.Path(exists=True, path_type=Path), help="Signed opening ticket JSON")
@click.option(
    "--opening-state-dir",
    type=click.Path(path_type=Path, file_okay=False),
    help="Directory for tracking cumulative opening budgets",
)
@click.option(
    "--allow-opening-reset",
    is_flag=True,
    help="Allow resetting opening ledger if previous state is missing",
)
def open_command(
    capsule: Path,
    dataset: Tuple[str, ...],
    dataset_id: str,
    chunk: int | None,
    output_json: bool,
    policy: Path | None,
    ticket: Path | None,
    opening_state_dir: Path | None,
    allow_opening_reset: bool,
) -> None:
    """Open dataset commitments and optional chunk proofs from a capsule."""
    capsule = capsule.expanduser().resolve()
    ds_map = _parse_dataset_args(dataset)

    capsule_data, base_dir, tmp_ctx = _load_capsule_with_artifacts(capsule)
    resolved_policy_path = resolve_policy_path_for_capsule(capsule_data, base_dir, policy)
    policy_config = None
    if resolved_policy_path:
        try:
            policy_config = load_policy_config(resolved_policy_path)
        except PolicyError as exc:
            raise click.ClickException(f"failed to load policy: {exc}")
    try:
        try:
            ds_ref = capsule_data.get("dataset_ref") or {}
            ds_list = ds_ref.get("datasets") or []
        except AttributeError:
            ds_list = []

        entry = next((item for item in ds_list if (item.get("dataset_id") or "") == dataset_id), None)
        if not entry:
            raise click.ClickException(f"dataset_id '{dataset_id}' not found in capsule")

        manifest_rel = entry.get("manifest_rel_path")
        if not manifest_rel:
            raise click.ClickException("dataset manifest not recorded in capsule")

        manifest_data = _load_dataset_manifest(base_dir, manifest_rel)
        entries = manifest_data.get("entries") or []
        arity = int(manifest_data.get("tree_arity") or entry.get("tree_arity") or 2)
        levels, _ = _compute_merkle_levels(entries, arity)

        result: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "recorded_root": entry.get("root"),
            "manifest_path": str(_safe_rel_path(manifest_rel)),
            "num_chunks": len(entries),
            "tree_arity": arity,
        }

        enforcer = None
        if policy_config and resolved_policy_path and policy_config.rules.get("openings_governance"):
            enforcer = OpeningPolicyEnforcer(
                policy_config=policy_config,
                capsule_data=capsule_data,
                policy_path=resolved_policy_path,
                state_dir=opening_state_dir,
                allow_reset=allow_opening_reset,
            )

        if chunk is not None:
            if chunk < 0 or chunk >= len(entries):
                raise click.ClickException(f"chunk {chunk} out of range (0..{len(entries)-1})")
            chunk_entry = entries[chunk]
            proof = _chunk_proof(levels, chunk, arity)
            chunk_result: Dict[str, Any] = {
                "chunk_id": chunk,
                "uri": chunk_entry.get("uri"),
                "sha256": chunk_entry.get("sha256"),
                "size": chunk_entry.get("size"),
                "proof": proof,
            }
            dataset_path = ds_map.get(dataset_id)
            if dataset_path:
                data_path = _relative_dataset_file(dataset_path, chunk_entry.get("uri", ""))
                if not data_path.exists():
                    raise click.ClickException(f"dataset chunk missing locally: {data_path}")
                actual_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
                chunk_result["local_path"] = str(data_path)
                chunk_result["local_sha256"] = actual_hash
                chunk_result["hash_match"] = actual_hash.lower() == str(chunk_entry.get("sha256", "")).lower()
            result["chunk"] = chunk_result

        if enforcer and chunk is not None:
            try:
                enforcer.enforce(
                    ticket_path=ticket,
                    dataset_id=dataset_id,
                    row_indices=[chunk],
                    bytes_requested=int(result.get("chunk", {}).get("size") or 0),
                )
            except PolicyError as exc:
                raise click.ClickException(str(exc))

        if output_json:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Dataset: {dataset_id}")
            click.echo(f"  Root: {result['recorded_root']}")
            click.echo(f"  Manifest: {result['manifest_path']}")
            click.echo(f"  Chunks: {result['num_chunks']} (arity={arity})")
            if chunk is not None and "chunk" in result:
                chunk_info = result["chunk"]
                click.echo(f"  Chunk {chunk}: uri={chunk_info.get('uri')}, sha256={chunk_info.get('sha256')}")
                if chunk_info.get("proof"):
                    click.echo(f"    Proof: {len(chunk_info['proof'])} levels")
                if chunk_info.get("local_path"):
                    click.echo(f"    Local file: {chunk_info['local_path']} (match={chunk_info.get('hash_match')})")
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
