"""The .cap file format - portable capsule archive.

A .cap file is a tarball containing:
    manifest.json       - Capsule metadata (id, version, policy_hash, backend, sizes)
    proof.bin.zst       - Compressed binary proof
    commitments.json    - Root commitment, chunks, index metadata
    capsule.json        - Full capsule data (optional, for compatibility)
    archive/            - Binary row archive (optional, for DA audit)
    signatures/         - Detached key-bound signatures (optional)

Archive auto-switch: When chunk count exceeds threshold, switch from JSON to binary.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

# Optional zstd compression
try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False

CAP_MANIFEST_SCHEMA = "cap_manifest_v1"
BINARY_ARCHIVE_THRESHOLD = int(os.environ.get("CAP_BINARY_THRESHOLD", "1000"))
MAX_TAR_MEMBER_BYTES = int(os.environ.get("CAP_MAX_MEMBER_BYTES", str(512 * 1024 * 1024)))


@dataclass
class CapManifest:
    """Manifest embedded in .cap files."""
    schema: str = CAP_MANIFEST_SCHEMA
    capsule_id: str = ""
    trace_id: str = ""
    policy_id: str = ""
    policy_hash: str = ""
    backend: str = ""
    verification_profile: str = "proof_only"
    root_hex: str = ""
    num_chunks: int = 0
    proof_size: int = 0
    archive_format: str = "json"  # "json" or "binary"
    created_at: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "capsule_id": self.capsule_id,
            "trace_id": self.trace_id,
            "policy_id": self.policy_id,
            "policy_hash": self.policy_hash,
            "backend": self.backend,
            "verification_profile": self.verification_profile,
            "root_hex": self.root_hex,
            "num_chunks": self.num_chunks,
            "proof_size": self.proof_size,
            "archive_format": self.archive_format,
            "created_at": self.created_at,
            **self.extras,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapManifest":
        known_keys = {
            "schema", "capsule_id", "trace_id", "policy_id", "policy_hash",
            "backend", "verification_profile", "root_hex", "num_chunks",
            "proof_size", "archive_format", "created_at",
        }
        extras = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            schema=data.get("schema", CAP_MANIFEST_SCHEMA),
            capsule_id=data.get("capsule_id", ""),
            trace_id=data.get("trace_id", ""),
            policy_id=data.get("policy_id", ""),
            policy_hash=data.get("policy_hash", ""),
            backend=data.get("backend", ""),
            verification_profile=data.get("verification_profile", "proof_only"),
            root_hex=data.get("root_hex", ""),
            num_chunks=data.get("num_chunks", 0),
            proof_size=data.get("proof_size", 0),
            archive_format=data.get("archive_format", "json"),
            created_at=data.get("created_at", ""),
            extras=extras,
        )


def _hash_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def _compress_zstd(data: bytes) -> bytes:
    """Compress data with zstd if available, else return as-is."""
    if not _HAS_ZSTD:
        return data
    cctx = zstd.ZstdCompressor(level=3)
    return cctx.compress(data)


def _decompress_zstd(data: bytes) -> bytes:
    """Decompress zstd data if available, else return as-is."""
    if not _HAS_ZSTD:
        return data
    if len(data) < 4:
        return data
    # Check for zstd magic number (0x28 0xB5 0x2F 0xFD)
    if data[:4] == b'\x28\xb5\x2f\xfd':
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(data)
    return data


def should_use_binary_archive(num_chunks: int) -> bool:
    """Determine if binary archive format should be used based on chunk count."""
    return num_chunks > BINARY_ARCHIVE_THRESHOLD


class CapExtractionError(RuntimeError):
    """Raised when a .cap archive fails safety checks during extraction."""


def _normalize_rel_path(rel: str) -> Path | None:
    rel = rel.replace("\\", "/")
    while rel.startswith("./"):
        rel = rel[2:]
    if rel in ("", "."):
        return None
    pure = PurePosixPath(rel)
    if pure.is_absolute():
        raise CapExtractionError(f"Absolute paths not allowed in archive: {rel!r}")
    if any(part in ("", "..") for part in pure.parts):
        raise CapExtractionError(f"Path traversal detected in archive: {rel!r}")
    if not pure.parts:
        return None
    return Path(*pure.parts)


def _safe_extract_tar(tar_path: Path, output_dir: Path) -> None:
    root = output_dir.resolve()
    seen: set[Path] = set()
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            name = member.name
            if "\x00" in name:
                raise CapExtractionError("Null byte in archive path")
            try:
                rel = _normalize_rel_path(name)
            except CapExtractionError:
                raise
            except Exception:
                raise CapExtractionError(f"Invalid archive path: {name!r}")

            if rel is None:
                continue

            dest = (root / rel).resolve()
            try:
                dest.relative_to(root)
            except ValueError:
                raise CapExtractionError(f"Archive entry escapes sandbox: {name!r}")

            if rel in seen:
                raise CapExtractionError(f"Duplicate entry in archive: {name!r}")
            seen.add(rel)

            if member.issym() or member.islnk():
                raise CapExtractionError(f"Links not allowed in archive: {name!r}")
            if not (member.isdir() or member.isreg()):
                raise CapExtractionError(f"Unsupported archive entry type: {name!r}")

            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue

            if member.size > MAX_TAR_MEMBER_BYTES:
                raise CapExtractionError(
                    f"Archive file too large: {name!r} ({member.size} bytes)"
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise CapExtractionError(f"Archive attempted to overwrite file: {name!r}")

            with tar.extractfile(member) as src:
                if src is None:
                    raise CapExtractionError(f"Failed to read archive entry: {name!r}")
                remaining = member.size
                with open(dest, "xb") as dst:
                    while remaining > 0:
                        chunk = src.read(min(1 << 20, remaining))
                        if not chunk:
                            raise CapExtractionError(
                                f"Archive entry truncated: {name!r}"
                            )
                        dst.write(chunk)
                        remaining -= len(chunk)
# removed final blank? ensure newline


def create_cap_file(
    capsule_path: Path,
    output_path: Path,
    *,
    proof_path: Path | None = None,
    archive_path: Path | None = None,
    signatures_path: Path | None = None,
    policy_path: Path | None = None,
    manifests_path: Path | None = None,
) -> CapManifest:
    """Create a .cap archive from capsule components.

    Args:
        capsule_path: Path to the capsule JSON file
        output_path: Output path for the .cap file
        proof_path: Optional path to binary proof file
        archive_path: Optional path to row archive directory
        signatures_path: Optional path to signatures directory
        policy_path: Optional path to policy file
        manifests_path: Optional path to manifests directory (for policy enforcement)

    Returns:
        The manifest written to the archive
    """
    from datetime import datetime, timezone

    capsule = json.loads(capsule_path.read_text())
    header = capsule.get("header", {})

    # Extract manifest fields from capsule
    manifest = CapManifest(
        capsule_id=capsule.get("capsule_hash", "")[:16],
        trace_id=capsule.get("trace_id", ""),
        policy_id=header.get("policy_id", capsule.get("policy_id", "")),
        policy_hash=header.get("policy_hash", capsule.get("policy_hash", "")),
        backend=header.get("backend_id", capsule.get("backend", "")),
        verification_profile=header.get(
            "verification_profile",
            capsule.get("verification_profile", "proof_only")
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    # Get commitment info
    commitments = capsule.get("commitments", {})
    if isinstance(commitments, dict):
        manifest.root_hex = commitments.get("root", "")
        manifest.num_chunks = commitments.get("num_chunks", 0)

    # Determine archive format
    if archive_path and archive_path.exists():
        # Count chunks to decide format
        chunk_count = sum(1 for _ in archive_path.glob("chunk_*.bin"))
        if chunk_count == 0:
            chunk_count = sum(1 for _ in archive_path.glob("chunk_*.json"))
        manifest.num_chunks = max(manifest.num_chunks, chunk_count)
        manifest.archive_format = "binary" if should_use_binary_archive(chunk_count) else "json"

    # Create temp directory for packing
    with tempfile.TemporaryDirectory() as tmpdir:
        pack_dir = Path(tmpdir) / "cap"
        pack_dir.mkdir()

        # Write manifest
        (pack_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2)
        )

        # Write capsule.json (full data for compatibility)
        (pack_dir / "capsule.json").write_text(
            json.dumps(capsule, indent=2)
        )

        # Copy artifact_manifest.json if present
        artifact_manifest = (capsule.get("artifacts") or {}).get("manifest")
        manifest_src: Path | None = None
        if isinstance(artifact_manifest, dict):
            manifest_path = artifact_manifest.get("path") or artifact_manifest.get("abs_path")
            if manifest_path:
                candidate = Path(manifest_path)
                if not candidate.is_absolute():
                    candidate = capsule_path.parent / manifest_path
                if candidate.exists():
                    manifest_src = candidate
        if manifest_src is None:
            candidate = capsule_path.parent / "artifact_manifest.json"
            if candidate.exists():
                manifest_src = candidate
        if manifest_src and manifest_src.exists():
            shutil.copy2(manifest_src, pack_dir / "artifact_manifest.json")

        # Copy events log if present
        events_entry = (capsule.get("artifacts") or {}).get("events_log")
        events_src: Path | None = None
        events_rel: str | None = None
        if isinstance(events_entry, dict):
            events_rel = events_entry.get("rel_path")
            events_path = events_entry.get("path") or events_entry.get("abs_path")
            if events_path:
                candidate = Path(events_path)
                if not candidate.is_absolute():
                    candidate = capsule_path.parent / events_path
                if candidate.exists():
                    events_src = candidate
        if events_src and events_src.exists():
            rel_path = _normalize_rel_path(events_rel or "events/events.jsonl")
            dest = pack_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(events_src, dest)

        # Extract and write commitments.json
        commitments_data = {
            "root": manifest.root_hex,
            "num_chunks": manifest.num_chunks,
            "chunk_len": commitments.get("chunk_len", 64),
            "chunk_tree_arity": commitments.get("chunk_tree_arity", 16),
        }
        if "chunk_roots" in commitments:
            commitments_data["chunk_roots"] = commitments["chunk_roots"]
        (pack_dir / "commitments.json").write_text(
            json.dumps(commitments_data, indent=2)
        )

        # Handle proof
        if proof_path and proof_path.exists():
            proof_data = proof_path.read_bytes()
            compressed = _compress_zstd(proof_data)
            (pack_dir / "proof.bin.zst").write_bytes(compressed)
            manifest.proof_size = len(proof_data)
        else:
            # Try to extract proof from capsule
            proofs = capsule.get("proofs", {})
            for proof_name, proof_info in proofs.items():
                formats = proof_info.get("formats", {})
                for fmt_name, fmt_info in formats.items():
                    if "path" in fmt_info:
                        src = capsule_path.parent / fmt_info["path"]
                        if src.exists():
                            proof_data = src.read_bytes()
                            compressed = _compress_zstd(proof_data)
                            (pack_dir / "proof.bin.zst").write_bytes(compressed)
                            manifest.proof_size = len(proof_data)
                            break

        # Copy archive if present
        if archive_path and archive_path.exists():
            dest = pack_dir / "archive"
            shutil.copytree(archive_path, dest)

        # Copy signatures if present
        if signatures_path and signatures_path.exists():
            dest = pack_dir / "signatures"
            shutil.copytree(signatures_path, dest)

        # Copy policy if present
        if policy_path and policy_path.exists():
            shutil.copy2(policy_path, pack_dir / "policy.json")

        # Copy manifests directory if present (for policy enforcement)
        if manifests_path and manifests_path.is_dir():
            dest = pack_dir / "manifests"
            shutil.copytree(manifests_path, dest)

        # Update manifest with final values
        (pack_dir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2)
        )

        # Create tarball
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(pack_dir, arcname=".")

    return manifest


def extract_cap_file(cap_path: Path, output_dir: Path) -> CapManifest:
    """Extract a .cap archive to a directory.

    Args:
        cap_path: Path to the .cap file
        output_dir: Directory to extract to

    Returns:
        The manifest from the archive
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        _safe_extract_tar(cap_path, output_dir)
    except CapExtractionError as exc:
        raise CapExtractionError(f"Failed to extract {cap_path}: {exc}") from exc

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        return CapManifest.from_dict(json.loads(manifest_path.read_text()))

    return CapManifest()


def read_cap_manifest(cap_path: Path) -> CapManifest:
    """Read just the manifest from a .cap file without full extraction."""
    with tarfile.open(cap_path, "r:*") as tar:
        for member in tar.getmembers():
            # Match exactly "manifest.json" or "./manifest.json", not "artifact_manifest.json"
            basename = member.name.split("/")[-1]
            if basename == "manifest.json":
                f = tar.extractfile(member)
                if f:
                    data = json.loads(f.read().decode("utf-8"))
                    return CapManifest.from_dict(data)
    return CapManifest()


def read_cap_capsule(cap_path: Path) -> dict[str, Any]:
    """Read the full capsule.json from a .cap file."""
    with tarfile.open(cap_path, "r:*") as tar:
        for member in tar.getmembers():
            if member.name.endswith("capsule.json"):
                f = tar.extractfile(member)
                if f:
                    return json.loads(f.read().decode("utf-8"))
    return {}


def create_run_cap_file(
    run_dir: Path,
    output_path: Path,
    *,
    run_type: str = "learn",
    extras: dict | None = None,
) -> CapManifest:
    """Create a .cap archive from a run directory.

    This is a simplified version for packaging learn/review runs that don't
    have full FRI proofs but still have receipt chains.

    Args:
        run_dir: Path to the run directory (e.g., .capseal/runs/20240101T120000-learn)
        output_path: Output path for the .cap file
        run_type: Type of run ("learn" or "review")
        extras: Additional metadata to include in manifest

    Returns:
        The manifest written to the archive
    """
    from datetime import datetime, timezone

    # Load run metadata
    metadata_path = run_dir / "run_metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    # Load receipt for chain hash
    receipt_path = run_dir / "run_receipt.json"
    receipt = {}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())

    # Load gate result for review runs
    gate_result_path = run_dir / "gate_result.json"
    gate_result = {}
    if gate_result_path.exists():
        gate_result = json.loads(gate_result_path.read_text())

    # Build manifest
    chain_hash = receipt.get("chain_hash", "")
    run_id = run_dir.name

    manifest = CapManifest(
        schema="run_cap_v1",
        capsule_id=chain_hash[:16] if chain_hash else run_id[:16],
        trace_id=run_id,
        policy_id="",
        policy_hash="",
        backend="capseal",
        verification_profile="receipt_chain",
        root_hex=chain_hash,
        num_chunks=0,
        proof_size=0,
        archive_format="json",
        created_at=datetime.now(timezone.utc).isoformat(),
        extras={
            "run_type": run_type,
            "run_id": run_id,
            "chain_hash": chain_hash,
            "total_rounds": receipt.get("total_rounds", 0),
            **(extras or {}),
        },
    )

    # Add run-type specific extras
    if run_type == "learn":
        # Count episodes from episodes.jsonl
        episodes_path = run_dir / "episodes.jsonl"
        if episodes_path.exists():
            lines = episodes_path.read_text().strip().split("\n")
            manifest.extras["episodes"] = len([l for l in lines if l.strip()])
    elif run_type == "review":
        summary = gate_result.get("summary", {})
        manifest.extras["approved"] = summary.get("approved", 0)
        manifest.extras["gated"] = summary.get("gated", 0)
        manifest.extras["flagged"] = summary.get("flagged", 0)
        manifest.extras["total_findings"] = summary.get("total", 0)

    # Create tarball with all run directory contents
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_path, "w:gz") as tar:
        # Add manifest first
        manifest_json = json.dumps(manifest.to_dict(), indent=2).encode("utf-8")
        manifest_info = tarfile.TarInfo(name="manifest.json")
        manifest_info.size = len(manifest_json)
        import io
        tar.addfile(manifest_info, io.BytesIO(manifest_json))

        # Add all files from run directory
        for file_path in run_dir.rglob("*"):
            if file_path.is_file() and "__pycache__" not in str(file_path):
                arcname = str(file_path.relative_to(run_dir))
                tar.add(file_path, arcname=arcname)

    return manifest
