#!/usr/bin/env python3
"""Dataset fetch helper: downloads remote data under policy-governed network rules."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib import parse as url_parse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bef_zk.policy import load_policy_config, PolicyError  # noqa: E402
from bef_zk.policy.network import NetworkGateway, compute_network_log_root  # noqa: E402
from bef_zk.capsule.dataset_spec import DatasetSpecV1, compute_dataset_spec_hash  # noqa: E402
from bef_zk.stc.merkle import build_kary_levels, root_from_levels  # noqa: E402


def _compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _relpath(path: Path, base: Path) -> str:
    return os.path.relpath(path.resolve(), base.resolve())


def _posix_rel(path: Path, base: Path) -> str:
    return _relpath(path, base).replace(os.sep, "/")


def _scan_dataset_dir(dataset_dir: Path) -> list[dict[str, any]]:
    base = dataset_dir.resolve()
    entries: list[dict[str, any]] = []
    idx = 0
    for root, _, files in os.walk(base):
        for name in sorted(files):
            p = Path(root) / name
            if not p.is_file():
                continue
            rel = _posix_rel(p, base)
            size = p.stat().st_size
            digest = _compute_file_hash(p)
            entries.append(
                {
                    "id": idx,
                    "uri": rel,
                    "sha256": digest,
                    "size": size,
                }
            )
            idx += 1
    return entries


def _compute_dataset_root(entries: list[dict[str, any]], arity: int) -> str:
    if not entries:
        return hashlib.sha256(b"").hexdigest()
    leaves = [bytes.fromhex(entry["sha256"]) for entry in entries]
    levels = build_kary_levels(leaves, arity)
    return root_from_levels(levels).hex()


def _default_dataset_id(url: str) -> str:
    parsed = url_parse.urlparse(url)
    candidate = Path(parsed.path).name or "dataset"
    return candidate.replace("/", "_") or "dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch remote dataset under a network policy")
    parser.add_argument("--url", required=True, help="Remote URL to download")
    parser.add_argument("--dataset-id", dest="dataset_id", help="Dataset identifier to assign")
    parser.add_argument("--output-dir", dest="output_dir", type=Path, help="Output directory")
    parser.add_argument("--policy", required=True, type=Path, help="Policy JSON file")
    parser.add_argument("--policy-id", required=True, help="Policy identifier")
    parser.add_argument("--policy-version", dest="policy_version", help="Policy version label")
    parser.add_argument("--dataset-tree-arity", type=int, default=16, help="Dataset Merkle arity")
    args = parser.parse_args()

    policy_path = args.policy.expanduser().resolve()
    if not policy_path.exists():
        raise SystemExit(f"policy file not found: {policy_path}")
    policy_hash = _compute_file_hash(policy_path)
    try:
        policy_config = load_policy_config(policy_path)
    except PolicyError as exc:
        raise SystemExit(f"Policy load error: {exc}") from exc
    network_rules = policy_config.rules.get("network_access") if policy_config else None
    if not network_rules or not network_rules.get("enabled"):
        raise SystemExit("policy does not permit network access (network_access.enabled=false)")

    dataset_id = args.dataset_id or _default_dataset_id(args.url)
    out_dir = (args.output_dir or (ROOT / "out" / f"fetch_{dataset_id}")).expanduser().resolve()
    materialized_dir = out_dir / "datasets" / dataset_id
    data_dir = materialized_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    gateway = NetworkGateway(network_rules, base_dir=out_dir)
    target_name = Path(url_parse.urlparse(args.url).path).name or "payload.bin"
    dest_path = data_dir / target_name
    record = gateway.fetch(args.url, dest_path)

    entries = _scan_dataset_dir(data_dir)
    dataset_root = _compute_dataset_root(entries, arity=int(args.dataset_tree_arity))
    manifest_path = materialized_dir / "manifest.json"
    manifest = {
        "schema": "dataset_manifest_v1",
        "dataset_id": dataset_id,
        "tree_arity": int(args.dataset_tree_arity),
        "num_chunks": len(entries),
        "root": dataset_root,
        "entries": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    spec = DatasetSpecV1(
        dataset_id=dataset_id,
        root=dataset_root,
        chunk_arity=int(args.dataset_tree_arity),
        num_chunks=len(entries),
        manifest_rel_path=_posix_rel(manifest_path, out_dir),
    )
    ds_spec_hash = compute_dataset_spec_hash(spec)

    network_dir = out_dir / "network"
    network_dir.mkdir(parents=True, exist_ok=True)
    network_log_path = network_dir / "network_log.jsonl"
    with network_log_path.open("w", encoding="utf-8") as fh:
        for entry in gateway.iter_entries():
            fh.write(json.dumps(entry.to_obj()) + "\n")
    network_root = compute_network_log_root(gateway.iter_entries())

    receipt = {
        "schema": "dataset_fetch_receipt_v1",
        "dataset_id": dataset_id,
        "source_url": args.url,
        "policy": {
            "policy_id": args.policy_id,
            "policy_version": args.policy_version or policy_config.raw.get("policy_version") or "unspecified",
            "policy_hash": policy_hash,
        },
        "network": {
            "log_path": _posix_rel(network_log_path, out_dir),
            "trace_root": network_root,
            "entries": len(gateway.iter_entries()),
            "bytes_total": gateway.bytes_total,
        },
        "dataset": {
            "root": dataset_root,
            "tree_arity": int(args.dataset_tree_arity),
            "num_chunks": len(entries),
            "materialized_path": _posix_rel(data_dir, out_dir),
            "manifest_path": _posix_rel(manifest_path, out_dir),
            "dataset_spec_hash": ds_spec_hash,
        },
        "output_dir": str(out_dir),
    }
    receipt_path = out_dir / "fetch_receipt.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2))

    summary = {
        "status": "OK",
        "dataset_id": dataset_id,
        "output_dir": str(out_dir),
        "materialized_path": str(data_dir),
        "dataset_root": dataset_root,
        "dataset_spec_hash": ds_spec_hash,
        "fetch_receipt": str(receipt_path),
        "network_log": str(network_log_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
