from __future__ import annotations

import json
from pathlib import Path

import pytest

from capsule_bench.manifests import (
    collect_manifests,
    load_manifest_bundle,
    write_manifest_signature,
)
from capsule_bench.packing import write_pack_meta


def test_collect_manifests_creates_files(tmp_path: Path) -> None:
    bundle = collect_manifests(tmp_path)
    assert bundle.base_dir == tmp_path
    # hardware/os/toolchain manifests should exist
    for key in ["hardware_manifest", "os_fingerprint", "toolchain_manifest"]:
        assert key in bundle.files
        path = bundle.files[key]
        assert path.exists()
        payload = json.loads(path.read_text())
        assert payload["schema"].startswith("bef_")
    assert bundle.anchor_ref.startswith("capsulebench_manifest_v1:")


def test_write_pack_meta(tmp_path: Path) -> None:
    pack_dir = tmp_path / "capsulepack"
    pack_dir.mkdir()
    (pack_dir / "capsule.json").write_text("{}")
    (pack_dir / "policy.json").write_text("policy")
    meta = write_pack_meta(pack_dir)
    pack_meta_path = pack_dir / "pack_meta.json"
    assert pack_meta_path.exists()
    recorded = json.loads(pack_meta_path.read_text())
    assert recorded["schema"] == "capsulepack_meta_v1"
    entries = {entry["path"] for entry in meta["entries"]}
    assert "capsule.json" in entries
    assert "policy.json" in entries


def test_write_manifest_signature(tmp_path: Path) -> None:
    pytest.importorskip("coincurve")
    bundle = collect_manifests(tmp_path)
    signer_id = "test_manifest"
    sig_path = write_manifest_signature(
        bundle,
        signer_id=signer_id,
        private_key_hex="3" * 64,
    )
    payload = json.loads(sig_path.read_text())
    assert payload["signer_id"] == signer_id
    sig_bytes = bytes.fromhex(payload["signature"])
    assert len(sig_bytes) == 65


def test_load_manifest_bundle_reads_existing_files(tmp_path: Path) -> None:
    bundle = collect_manifests(tmp_path)
    loaded = load_manifest_bundle(tmp_path)
    assert loaded.anchor_ref == bundle.anchor_ref
    assert set(loaded.files.keys()) == set(bundle.files.keys())
