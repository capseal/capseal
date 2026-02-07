"""capsule emit command - generate a portable verification artifact.

Usage:
    capsule emit \
        --capsule <capsule.json> \
        --out <receipt.cap> \
        [--artifacts <proof_artifacts/>] \
        [--archive <row_archive/>] \
        [--policy <policy.json>] \
        [--profile proof-only|da|replay]

The emit command packages existing proof artifacts into a portable .cap file
that can be verified independently. For full proof generation, use run_pipeline.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from .cap_format import (
    create_cap_file,
    should_use_binary_archive,
    BINARY_ARCHIVE_THRESHOLD,
)


def _find_proof_file(artifacts_dir: Path) -> Path | None:
    """Find the proof binary in an artifacts directory."""
    candidates = [
        "proof.bin",
        "proof.bin.zst",
        "geom_proof.bin",
        "geom_proof.bin.zst",
    ]
    for name in candidates:
        path = artifacts_dir / name
        if path.exists():
            return path

    # Check in proofs/ subdirectory
    proofs_dir = artifacts_dir / "proofs"
    if proofs_dir.exists():
        for name in candidates:
            path = proofs_dir / name
            if path.exists():
                return path

        # Look for any .bin file
        for bin_file in proofs_dir.glob("*.bin"):
            return bin_file

    return None


def _find_signatures(artifacts_dir: Path) -> Path | None:
    """Find signatures directory in artifacts."""
    candidates = [
        artifacts_dir / "signatures",
        artifacts_dir / "sigs",
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


def _find_archive(artifacts_dir: Path) -> Path | None:
    """Find row archive directory in artifacts."""
    candidates = [
        artifacts_dir / "row_archive",
        artifacts_dir / "archive",
        artifacts_dir / "chunks",
    ]
    for path in candidates:
        if path.exists() and path.is_dir():
            return path
    return None


@click.command("emit")
@click.option(
    "--capsule",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=False,
    help="Path to capsule.json file",
)
@click.option(
    "--source",
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Source path to infer from (run directory or capsule folder). "
        "If a directory, the command will look for one of: "
        "capsule.json, capsule/capsule.json, pipeline/strategy_capsule.json. "
        "If a file, it must be a capsule JSON."
    ),
)
@click.option(
    "--receipt",
    type=click.Path(exists=True, path_type=Path),
    help=(
        "Alias for --source (for backward compatibility). Provide a directory "
        "containing a capsule (e.g., fixtures/golden_run_latest/capsule) or a "
        "run pipeline directory (…/pipeline)."
    ),
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for .cap file",
)
@click.option(
    "--artifacts",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing proof artifacts",
)
@click.option(
    "--archive",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Row archive directory (for DA audit support)",
)
@click.option(
    "--policy",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Policy file to include",
)
@click.option(
    "--profile",
    type=click.Choice(["proof-only", "da", "replay"], case_sensitive=False),
    default="proof-only",
    help="Verification profile",
)
@click.option(
    "--binary-threshold",
    type=int,
    default=BINARY_ARCHIVE_THRESHOLD,
    help=f"Chunk count threshold for binary archive format (default: {BINARY_ARCHIVE_THRESHOLD})",
)
@click.option(
    "--manifests",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Manifests directory to include (for policy_enforced/DA verification)",
)
def emit_command(
    capsule: Path | None,
    out: Path,
    artifacts: Path | None,
    archive: Path | None,
    policy: Path | None,
    profile: str,
    binary_threshold: int,
    manifests: Path | None,
    source: Path | None,
    receipt: Path | None,
) -> None:
    """Generate a portable verification artifact (.cap file).

    Packages a capsule and its associated proof artifacts into a single
    .cap archive that can be verified independently.

    \b
    The resulting .cap file contains:
    - manifest.json: Capsule metadata
    - capsule.json: Full capsule data
    - proof.bin.zst: Compressed proof (if available)
    - commitments.json: Root commitment data
    - archive/: Row archive (if provided, for DA mode)
    - signatures/: Detached signatures (if present)
    - policy.json: Policy file (if provided)
    - manifests/: Hardware/toolchain manifests (if provided, for policy enforcement)

    Archive format auto-switches to binary when chunk count exceeds threshold.
    """
    # Infer capsule/artifacts from --source/--receipt if provided
    src = source or receipt
    if capsule is None and src is not None:
        def _candidate_capsule(p: Path) -> Path | None:
            # Accept exact file
            if p.is_file():
                name = p.name.lower()
                if name.endswith("capsule.json") or name == "strategy_capsule.json":
                    return p
                return None
            # Directory candidates
            for rel in (
                "capsule.json",
                "capsule/capsule.json",
                "pipeline/strategy_capsule.json",
                "strategy_capsule.json",
            ):
                c = p / rel
                if c.exists() and c.is_file():
                    return c
            return None

        cap_guess = _candidate_capsule(src)
        if not cap_guess:
            click.echo(
                "Error: Could not locate capsule JSON under --source/--receipt. "
                "Looked for: capsule.json, capsule/capsule.json, "
                "pipeline/strategy_capsule.json, strategy_capsule.json",
                err=True,
            )
            sys.exit(2)
        capsule = cap_guess
        # If artifacts not provided, default to capsule parent
        if artifacts is None:
            artifacts = capsule.parent

    if capsule is None:
        click.echo("Error: please provide --capsule or --source/--receipt", err=True)
        sys.exit(2)

    # Find proof file
    proof_path = None
    signatures_path = None

    if artifacts:
        proof_path = _find_proof_file(artifacts)
        signatures_path = _find_signatures(artifacts)

        # Use archive from artifacts if not explicitly provided
        if not archive:
            archive = _find_archive(artifacts)

    # Ensure output has .cap extension
    if not out.suffix == ".cap":
        out = out.with_suffix(out.suffix + ".cap")

    # Validate capsule exists and is valid JSON
    try:
        capsule_data = json.loads(capsule.read_text())
    except json.JSONDecodeError as e:
        click.echo(f"Error: Invalid capsule.json: {e}", err=True)
        sys.exit(1)

    # Get chunk count for archive format decision
    commitments = capsule_data.get("commitments", {})
    num_chunks = commitments.get("num_chunks", 0)

    # Check if we should use binary archive
    if archive:
        chunk_count = sum(1 for _ in archive.glob("chunk_*.bin"))
        if chunk_count == 0:
            chunk_count = sum(1 for _ in archive.glob("chunk_*.json"))
        num_chunks = max(num_chunks, chunk_count)

    archive_format = "binary" if num_chunks > binary_threshold else "json"

    # Create the .cap file
    try:
        manifest = create_cap_file(
            capsule,
            out,
            proof_path=proof_path,
            archive_path=archive,
            signatures_path=signatures_path,
            policy_path=policy,
            manifests_path=manifests,
        )
    except Exception as e:
        click.echo(f"Error creating .cap file: {e}", err=True)
        sys.exit(1)

    # Output summary
    click.echo(f"Created: {out}")
    click.echo(f"  Capsule ID: {manifest.capsule_id}")
    click.echo(f"  Policy ID:  {manifest.policy_id or 'N/A'}")
    click.echo(f"  Backend:    {manifest.backend}")
    click.echo(f"  Profile:    {profile}")
    click.echo(f"  Chunks:     {manifest.num_chunks}")
    click.echo(f"  Archive:    {archive_format}")
    if manifest.proof_size:
        size_kb = manifest.proof_size / 1024
        click.echo(f"  Proof size: {size_kb:.1f} KB")

    # Size of output file
    out_size = out.stat().st_size
    if out_size < 1024:
        click.echo(f"  Total size: {out_size} B")
    elif out_size < 1024 * 1024:
        click.echo(f"  Total size: {out_size / 1024:.1f} KB")
    else:
        click.echo(f"  Total size: {out_size / (1024 * 1024):.1f} MB")
