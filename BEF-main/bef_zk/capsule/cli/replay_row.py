"""capsule row replay command - open trace row commitments."""
from __future__ import annotations

import hashlib
import json
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click

from bef_zk.capsule.dataset_spec import AccessLogEntry, compute_access_log_root
from bef_zk.stc.merkle import build_kary_levels, prove_kary, root_from_levels, verify_kary
from bef_zk.policy import load_policy_config, PolicyError
from bef_zk.capsule.policy_openings import (
    resolve_policy_path_for_capsule,
    OpeningPolicyEnforcer,
)
from .security_assumptions import (
    build_security_assumptions,
    dataset_info_from_capsule,
    policy_info_from_capsule,
    print_security_assumptions,
)

from .cap_format import CapExtractionError, extract_cap_file
from .trace_schema import get_schema, TraceSchema, SCHEMAS


HEX_PREFIXES = ("0x", "sha256:")


def _load_capsule_with_artifacts(
    capsule_path: Path,
) -> Tuple[dict, Path, tempfile.TemporaryDirectory | None]:
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


def _resolve_entry(base: Path, entry: Dict) -> Path:
    rel = entry.get("rel_path") or entry.get("path") or entry.get("abs_path")
    if not rel:
        raise RuntimeError("missing path in artifact entry")
    p = Path(rel)
    if not p.is_absolute():
        candidate = base / p
        if not candidate.exists() and str(rel).startswith("row_archive"):
            suffix = str(rel)[len("row_archive") :]
            alt = base / Path("archive" + suffix)
            if alt.exists():
                return alt
        return candidate
    return p


def _load_chunk_roots(info: dict, base: Path) -> Tuple[list[str], int]:
    json_path = info.get("chunk_roots_path") or info.get("chunk_roots_abs")
    if not json_path:
        raise RuntimeError("chunk_roots path missing")
    path = Path(json_path)
    if not path.is_absolute():
        candidate = base / path
        if not candidate.exists() and str(json_path).startswith("row_archive"):
            alt = base / Path("archive" + str(json_path)[len("row_archive") :])
            if alt.exists():
                candidate = alt
        path = candidate
    roots = json.loads(path.read_text())
    arity = int(info.get("chunk_tree_arity") or 2)
    return roots, arity


def _load_row_chunk(row_archive_dir: Path, row_index: int) -> Tuple[list[int], Path]:
    chunk_path = row_archive_dir / f"chunk_{row_index}.json"
    if not chunk_path.exists():
        raise RuntimeError(f"chunk not found: {chunk_path}")
    values = json.loads(chunk_path.read_text())
    return values, chunk_path


def _parse_dataset_args(values: Tuple[str, ...]) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for spec in values:
        if not spec:
            continue
        if "=" in spec:
            ds_id, raw_path = spec.split("=", 1)
            ds_id = (ds_id or "").strip()
            raw_path = (raw_path or "").strip()
        else:
            raw_path = spec.strip()
            ds_id = Path(raw_path).name
        path_obj = Path(raw_path).expanduser().resolve()
        if ds_id:
            mapping[ds_id] = path_obj
    return mapping


def _load_access_log(base_dir: Path, capsule_data: dict) -> List[dict]:
    artifact = (capsule_data.get("artifacts") or {}).get("access_log") or {}
    rel = artifact.get("rel_path") or artifact.get("path") or artifact.get("abs_path")
    if not rel:
        return []
    path = Path(rel)
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        return []
    entries: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _load_dataset_manifest(base: Path, manifest_rel: str) -> dict:
    rel_path = Path(manifest_rel)
    if not rel_path.is_absolute():
        rel_path = base / rel_path
    if not rel_path.exists():
        raise click.ClickException(f"dataset manifest missing: {rel_path}")
    return json.loads(rel_path.read_text())


def _strip_hex_prefix(value: str | None) -> str:
    if not value:
        return ""
    val = value.strip()
    for prefix in HEX_PREFIXES:
        if val.startswith(prefix):
            val = val[len(prefix) :]
    return val


def _hex_to_bytes(value: str | None) -> bytes | None:
    cleaned = _strip_hex_prefix(value)
    if not cleaned:
        return None
    try:
        return bytes.fromhex(cleaned)
    except ValueError:
        return None


def _find_dataset_entry(capsule_data: dict[str, Any], dataset_id: str) -> dict[str, Any] | None:
    ds_ref = capsule_data.get("dataset_ref") or {}
    for entry in ds_ref.get("datasets", []):
        if str(entry.get("dataset_id", "")) == dataset_id:
            return entry
    return None


def _dataset_chunk_context(
    base_dir: Path,
    capsule_data: dict[str, Any],
    dataset_id: str,
    chunk_id: int,
) -> dict[str, Any] | None:
    ds_entry = _find_dataset_entry(capsule_data, dataset_id)
    if not ds_entry:
        return None
    manifest_rel = ds_entry.get("manifest_rel_path")
    if not manifest_rel:
        raise click.ClickException("dataset manifest not recorded in capsule")
    manifest = _load_dataset_manifest(base_dir, manifest_rel)
    entries: list[dict[str, Any]] = manifest.get("entries") or []
    if chunk_id < 0 or chunk_id >= len(entries):
        raise click.ClickException(
            f"chunk_id {chunk_id} out of range for dataset '{dataset_id}' ({len(entries)} chunks)"
        )
    arity = int(manifest.get("tree_arity") or ds_entry.get("tree_arity") or 2)
    leaves: list[bytes] = []
    for entry in entries:
        leaf_hex = _strip_hex_prefix(entry.get("sha256") or entry.get("hash"))
        if not leaf_hex:
            raise click.ClickException("dataset manifest entry missing sha256")
        leaves.append(bytes.fromhex(leaf_hex))
    levels = build_kary_levels(leaves, arity)
    proof = prove_kary(levels, chunk_id, arity)
    proof_hex = [[s.hex() for s in level] for level in proof]
    chunk_entry = entries[chunk_id]
    chunk_hash_hex = _strip_hex_prefix(chunk_entry.get("sha256") or chunk_entry.get("hash"))
    if not chunk_hash_hex:
        raise click.ClickException("dataset chunk missing sha256 hash")
    chunk_leaf = bytes.fromhex(chunk_hash_hex)
    computed_root = root_from_levels(levels).hex()
    dataset_root_hex = _strip_hex_prefix(ds_entry.get("root") or manifest.get("root") or computed_root)
    root_bytes = bytes.fromhex(dataset_root_hex)
    verified = verify_kary(root_bytes, chunk_leaf, chunk_id, proof, arity, len(entries))
    return {
        "dataset_entry": ds_entry,
        "manifest_rel": manifest_rel,
        "chunk_entry": chunk_entry,
        "chunk_id": chunk_id,
        "proof": proof_hex,
        "tree_arity": arity,
        "num_chunks": len(entries),
        "computed_root": computed_root,
        "recorded_root": dataset_root_hex,
        "chunk_hash": chunk_hash_hex,
        "proof_valid": verified,
    }


def _access_log_anchor(capsule_data: dict[str, Any]) -> str | None:
    try:
        anchors = (capsule_data.get("statement") or {}).get("anchors") or []
    except AttributeError:
        return None
    for entry in anchors:
        if entry.get("anchor_rule_id") == "access_log_v1":
            return _strip_hex_prefix(entry.get("anchor_ref")) or None
    return None


def _format_access_entry(entry: dict[str, Any]) -> AccessLogEntry:
    return AccessLogEntry(
        dataset_id=str(entry.get("dataset_id", "")),
        chunk_id=int(entry.get("chunk_id", 0)),
        chunk_hash=str(entry.get("chunk_hash", "")),
        row_indices=list(entry.get("row_indices") or []),
        extra=dict(entry.get("extra", {})) if isinstance(entry.get("extra"), dict) else None,
    )


def _compute_access_log_digest(entries: List[dict], capsule_data: dict[str, Any]) -> dict[str, Any] | None:
    if not entries:
        return None
    try:
        formatted = [_format_access_entry(entry) for entry in entries]
    except Exception:
        formatted = []
    if not formatted:
        return None
    computed = compute_access_log_root(formatted)
    anchor_hex = _access_log_anchor(capsule_data)
    return {
        "computed_root": computed,
        "anchored_root": anchor_hex,
        "match": anchor_hex is None or anchor_hex.lower() == computed.lower(),
    }


def _compute_replay_digest(
    *,
    row_root_hex: str | None,
    dataset_root_hex: str | None,
    row_index: int,
    chunk_id: int,
    chunk_hash_hex: str | None,
    access_root_hex: str | None,
) -> str | None:
    row_root = _hex_to_bytes(row_root_hex)
    dataset_root = _hex_to_bytes(dataset_root_hex)
    chunk_hash = _hex_to_bytes(chunk_hash_hex)
    if not row_root or not dataset_root or not chunk_hash:
        return None
    h = hashlib.sha256()
    h.update(b"REPLAY_SLICE_V1")
    h.update(row_root)
    h.update(dataset_root)
    h.update(struct.pack(">Q", row_index))
    h.update(struct.pack(">Q", chunk_id))
    h.update(chunk_hash)
    if access_root_hex:
        access_bytes = _hex_to_bytes(access_root_hex)
        if access_bytes:
            h.update(access_bytes)
    return h.hexdigest()


@click.command("row")
@click.argument("capsule", type=click.Path(exists=True, path_type=Path))
@click.option("--row", type=int, required=True, help="Row index to open")
@click.option(
    "--dataset",
    multiple=True,
    help="Dataset mapping id=path (optional, for local chunk validation)",
)
@click.option("--json", "output_json", is_flag=True, help="Output JSON")
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
@click.option(
    "--schema",
    "schema_id",
    type=str,
    default=None,
    help="Schema ID for semantic decoding (e.g., momentum_v2_row_v1)",
)
@click.option(
    "--strict-schema",
    is_flag=True,
    help="Fail if schema hash mismatches capsule trace_schema_hash",
)
def replay_row_command(
    capsule: Path,
    row: int,
    dataset: Tuple[str, ...],
    output_json: bool,
    policy: Path | None,
    ticket: Path | None,
    opening_state_dir: Path | None,
    allow_opening_reset: bool,
    schema_id: str | None,
    strict_schema: bool,
) -> None:
    """Open a specific trace row with STC membership proof."""
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
    enforcer = None
    if policy_config and resolved_policy_path and policy_config.rules.get("openings_governance"):
        enforcer = OpeningPolicyEnforcer(
            policy_config=policy_config,
            capsule_data=capsule_data,
            policy_path=resolved_policy_path,
            state_dir=opening_state_dir,
            allow_reset=allow_opening_reset,
        )
    try:
        if row < 0:
            raise click.ClickException("row index must be non-negative")
        row_archive_info = capsule_data.get("row_archive") or {}
        if not row_archive_info:
            raise click.ClickException("capsule missing row_archive metadata")
        row_archive_path = _resolve_entry(base_dir, row_archive_info)
        values, chunk_path = _load_row_chunk(row_archive_path, row)
        chunk_roots_hex, arity = _load_chunk_roots(row_archive_info, base_dir)
        if row >= len(chunk_roots_hex):
            raise click.ClickException(
                f"row index {row} exceeds chunk roots length {len(chunk_roots_hex)}"
            )
        chunk_roots = [bytes.fromhex(_strip_hex_prefix(h)) for h in chunk_roots_hex]
        levels = build_kary_levels(chunk_roots, arity)
        proof = prove_kary(levels, row, arity)
        proof_hex = [[s.hex() for s in level] for level in proof]
        row_index_ref = capsule_data.get("row_index_ref") or {}
        commitment = row_index_ref.get("commitment") or row_archive_info.get("row_commitment", {}).get("root")
        arity_override = int(row_index_ref.get("tree_arity") or arity)
        chunk_leaf = chunk_roots[row]
        proof_valid = False
        if commitment:
            root_bytes = _hex_to_bytes(commitment)
            if root_bytes:
                proof_valid = verify_kary(root_bytes, chunk_leaf, row, proof, arity_override, len(chunk_roots))

        result = {
            "row_index": row,
            "row_values": values,
            "chunk_file": str(chunk_path),
            "commitment": commitment,
            "proof": proof_hex,
            "tree_arity": arity_override,
            "row_leaf": chunk_leaf.hex(),
            "row_proof_valid": proof_valid,
        }

        # Schema decoding
        if schema_id:
            schema = get_schema(schema_id)
            if not schema:
                available = ", ".join(SCHEMAS.keys())
                raise click.ClickException(f"Unknown schema '{schema_id}'. Available: {available}")

            schema_hash = schema.hash()
            capsule_schema_hash = capsule_data.get("trace_schema_hash")

            # Check for hash mismatch
            schema_mismatch = False
            if capsule_schema_hash and capsule_schema_hash != schema_hash:
                schema_mismatch = True
                if strict_schema:
                    raise click.ClickException(
                        f"Schema hash mismatch: provided={schema_hash[:16]}... "
                        f"capsule={capsule_schema_hash[:16]}..."
                    )

            # Decode row values
            decoded = schema.decode_row(values)
            row_fields = []
            for i, field_def in enumerate(schema.fields):
                field_entry = {
                    "name": field_def.name,
                    "type": field_def.field_type,
                    "meaning": field_def.meaning,
                    "value": decoded.get(field_def.name),
                }
                if field_def.unit:
                    field_entry["unit"] = field_def.unit
                if field_def.scale:
                    field_entry["scale"] = field_def.scale
                if i < len(values):
                    field_entry["raw_value"] = values[i]
                row_fields.append(field_entry)

            result["row_raw_values"] = values
            result["row_fields"] = row_fields
            result["schema_id"] = schema_id
            result["schema_hash"] = schema_hash
            if schema_mismatch:
                result["schema_warning"] = "hash mismatch with capsule trace_schema_hash"

        access_entries = _load_access_log(base_dir, capsule_data)
        access_digest = _compute_access_log_digest(access_entries, capsule_data)
        ds_context = None
        for entry in access_entries:
            rows = entry.get("row_indices") or []
            if row in rows:
                ds_context = entry
                break
        if ds_context:
            ds_id = ds_context.get("dataset_id")
            chunk_id = int(ds_context.get("chunk_id", -1))
            dataset_section = dict(ds_context)
            ds_status = None
            if ds_id and chunk_id >= 0:
                ds_status = _dataset_chunk_context(base_dir, capsule_data, ds_id, chunk_id)
                if ds_status:
                    dataset_section.setdefault("chunk_uri", ds_status["chunk_entry"].get("uri"))
                    dataset_section["chunk_sha256"] = ds_status["chunk_hash"]
                    dataset_section["chunk_proof"] = ds_status["proof"]
                    dataset_section["chunk_proof_valid"] = ds_status["proof_valid"]
                    dataset_section["chunk_size"] = ds_status["chunk_entry"].get("size")
                    dataset_section["dataset_root_recorded"] = ds_status["recorded_root"]
                    dataset_section["dataset_root_computed"] = ds_status["computed_root"]
                    dataset_section["dataset_root_match"] = (
                        ds_status["recorded_root"].lower() == ds_status["computed_root"].lower()
                    )
            if ds_id in ds_map and dataset_section.get("chunk_uri"):
                local_file = ds_map[ds_id] / dataset_section["chunk_uri"]
                local_file = local_file.resolve()
                if local_file.exists():
                    data = local_file.read_bytes()
                    local_hash = hashlib.sha256(data).hexdigest()
                    dataset_section["local_path"] = str(local_file)
                    dataset_section["local_sha256"] = local_hash
                    dataset_section["hash_match"] = local_hash.lower() == str(
                        dataset_section.get("chunk_sha256", "")
                    ).lower()
            result["dataset"] = dataset_section

            if ds_status:
                joined_digest = _compute_replay_digest(
                    row_root_hex=commitment,
                    dataset_root_hex=ds_status["recorded_root"],
                    row_index=row,
                    chunk_id=ds_status.get("chunk_id", chunk_id),
                    chunk_hash_hex=ds_status["chunk_hash"],
                    access_root_hex=(access_digest or {}).get("anchored_root"),
                )
                if joined_digest:
                    result["replay_digest"] = joined_digest
            if enforcer and ds_id is not None and chunk_id >= 0 and ds_status:
                try:
                    enforcer.enforce(
                        ticket_path=ticket,
                        dataset_id=ds_id,
                        row_indices=[row],
                        bytes_requested=int(ds_status["chunk_entry"].get("size") or 0),
                    )
                except PolicyError as exc:
                    raise click.ClickException(str(exc))

        if access_digest:
            result["access_log_digest"] = access_digest
        if enforcer and not ds_context:
            raise click.ClickException("policy requires dataset bindings for row openings")

        assumptions = build_security_assumptions(
            operation="row",
            capsule=capsule_data,
            policy_info=policy_info_from_capsule(capsule_data),
            dataset_info=dataset_info_from_capsule(capsule_data),
        )
        result["security_assumptions"] = assumptions

        if output_json:
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(f"Row {row} (arity={arity_override})")
            click.echo(f"  Chunk file: {chunk_path}")
            click.echo(f"  Row proof valid: {proof_valid}")
            if schema_id and "row_fields" in result:
                click.echo(f"  Schema: {schema_id} (hash={result['schema_hash'][:16]}...)")
                if result.get("schema_warning"):
                    click.echo(f"  ⚠ WARNING: {result['schema_warning']}")
                click.echo("  Fields:")
                for f in result["row_fields"]:
                    unit_str = f" {f['unit']}" if f.get('unit') else ""
                    click.echo(f"    {f['name']}: {f['value']}{unit_str} ({f['meaning']})")
            if ds_context:
                click.echo(
                    "  Dataset: id=%s chunk_id=%s" % (ds_context.get("dataset_id"), ds_context.get("chunk_id"))
                )
                if result["dataset"].get("chunk_uri"):
                    click.echo(f"    Chunk URI: {result['dataset']['chunk_uri']}")
                if result["dataset"].get("chunk_proof_valid") is not None:
                    click.echo(
                        "    Chunk proof valid: %s"
                        % ("yes" if result["dataset"].get("chunk_proof_valid") else "no")
                    )
                if result["dataset"].get("local_path"):
                    click.echo(
                        f"    Local file: {result['dataset']['local_path']} (match={result['dataset'].get('hash_match')})"
                    )
            if access_digest:
                click.echo(
                    "  Access log root match: %s" % ("yes" if access_digest.get("match") else "no")
                )
            if result.get("replay_digest"):
                click.echo(f"  Replay digest: {result['replay_digest']}")
            click.echo(f"  Values: {values}")
            click.echo(f"  Proof levels: {len(proof_hex)}")
            print_security_assumptions(assumptions)
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()
