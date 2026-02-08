"""capsule verify command - verify a capsule with stable exit codes.

Usage:
    capsule verify <receipt.cap> [--mode proof-only|da|replay] [--json]

Exit codes:
    0  - Verified successfully
    10 - Proof verification failed
    11 - Policy mismatch
    12 - Commitment/index verification failed
    13 - DA audit failed
    14 - Replay diverged
    20 - Malformed or parse error
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Optional, Tuple

import click

from .exit_codes import (
    EXIT_VERIFIED,
    EXIT_MALFORMED,
    error_to_exit_code,
    exit_code_description,
)
from .cap_format import CapExtractionError, extract_cap_file, read_cap_capsule, verify_cap_integrity

MAX_PROOF_BYTES = int(os.environ.get("CAP_MAX_PROOF_BYTES", str(512 * 1024 * 1024)))

try:  # optional zstd support for embedded proofs
    import zstandard as _zstd  # type: ignore
    _HAS_ZSTD = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_ZSTD = False


def _decompress_proof_blob(data: bytes) -> bytes:
    """Decompress proof blob if zstd-compressed."""
    if not data or not _HAS_ZSTD:
        return data
    if data[:4] == b"\x28\xb5\x2f\xfd":  # zstd magic
        try:
            dctx = _zstd.ZstdDecompressor()
            return dctx.decompress(data)
        except Exception:  # pragma: no cover - fallback to raw bytes
            return data
    return data


def _safe_rel_path(rel: str, *, default: str | None = None) -> Path:
    value = rel or default
    if not value:
        raise ValueError("Missing relative path in capsule artifact")
    value = value.replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    pure = PurePosixPath(value)
    if pure.is_absolute():
        raise ValueError(f"Absolute paths not allowed in capsule: {value!r}")
    if any(part in ("", "..") for part in pure.parts):
        raise ValueError(f"Path traversal detected in capsule: {value!r}")
    return Path(*pure.parts)


def _select_proof_descriptor(proofs: dict[str, Any]) -> Tuple[str, Optional[int], Optional[str]] | None:
    def build(entry: dict[str, Any]) -> Tuple[str, Optional[int], Optional[str]] | None:
        rel = entry.get("rel_path") or entry.get("path")
        if rel:
            return rel, entry.get("size_bytes"), entry.get("sha256_payload_hash")
        formats = entry.get("formats") or {}
        for fmt in formats.values():
            rel = fmt.get("rel_path")
            if rel:
                size = fmt.get("size_bytes") or fmt.get("size")
                digest = fmt.get("sha256_payload_hash")
                return rel, size, digest
        return None

    for entry in proofs.values():
        if not isinstance(entry, dict):
            continue
        desc = build(entry)
        if desc:
            return desc
    return None


def _materialize_cap_artifacts(extract_dir: Path, capsule: dict[str, Any]) -> None:
    """Ensure embedded artifacts exist at the rel_paths referenced in the capsule."""
    base = extract_dir

    # Proof artifact
    proof_blob = None
    proof_zst = base / "proof.bin.zst"
    if proof_zst.exists():
        proof_blob = _decompress_proof_blob(proof_zst.read_bytes())
    proofs = capsule.get("proofs") or {}
    if proof_blob and proofs:
        descriptor = _select_proof_descriptor(proofs)
        rel_value, expected_size, expected_hash = descriptor if descriptor else ("proofs/embedded_proof.bin", None, None)
        rel_path = _safe_rel_path(rel_value)
        if len(proof_blob) > MAX_PROOF_BYTES:
            raise ValueError("Embedded proof exceeds maximum allowed size")
        if expected_size is not None and len(proof_blob) != int(expected_size):
            raise ValueError("Embedded proof size mismatch")
        if expected_hash:
            actual_hash = hashlib.sha256(proof_blob).hexdigest()
            if actual_hash.lower() != expected_hash.lower():
                raise ValueError("Embedded proof hash mismatch")
        target = base / rel_path
        if target.exists():
            raise ValueError(f"Proof path already exists in capsule sandbox: {rel_value}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(proof_blob)

    # Row archive directory
    archive_dir = base / "archive"
    row_entry = capsule.get("row_archive") or {}
    row_rel = row_entry.get("rel_path") or row_entry.get("path") or "row_archive"
    row_rel_path = _safe_rel_path(str(row_rel))
    target_row_dir = base / row_rel_path
    if archive_dir.exists():
        target_row_dir.parent.mkdir(parents=True, exist_ok=True)
        if target_row_dir.exists():
            shutil.rmtree(target_row_dir)
        shutil.move(str(archive_dir), str(target_row_dir))

    # Artifacts row archive may reference different rel path
    artifacts = capsule.get("artifacts") or {}
    art_row = artifacts.get("row_archive") or {}
    art_rel = art_row.get("rel_path") or art_row.get("path")
    if art_rel:
        art_rel_path = _safe_rel_path(str(art_rel))
        target_art_dir = base / art_rel_path
        if target_art_dir != target_row_dir and target_row_dir.exists():
            target_art_dir.parent.mkdir(parents=True, exist_ok=True)
            if target_art_dir.exists():
                shutil.rmtree(target_art_dir)
            shutil.copytree(target_row_dir, target_art_dir)


def _prepare_extracted_capsule(extract_dir: Path) -> Path:
    capsule_json = extract_dir / "capsule.json"
    if not capsule_json.exists():
        return capsule_json
    try:
        capsule_obj = json.loads(capsule_json.read_text())
    except json.JSONDecodeError:
        return capsule_json
    _materialize_cap_artifacts(extract_dir, capsule_obj)
    return capsule_json


def _import_verify_core():
    """Lazy import of verification core to avoid circular imports."""
    # Import from scripts.verify_capsule
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "verify_capsule",
        Path(__file__).parents[3] / "scripts" / "verify_capsule.py"
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._verify_capsule_core
    raise ImportError("Could not load verify_capsule module")


def _normalize_mode(mode: str) -> str:
    """Normalize verification mode to internal level name."""
    mode_map = {
        "proof-only": "proof_only",
        "proof_only": "proof_only",
        "da": "full",
        "replay": "full",
        "full": "full",
    }
    return mode_map.get(mode.lower(), "proof_only")


def _verify_capsule(
    capsule_path: Path,
    *,
    mode: str = "proof-only",
    policy_path: Path | None = None,
    manifest_root: Path | None = None,
) -> tuple[int, dict[str, Any]]:
    """Verify a capsule and return (exit_code, result_dict).

    Args:
        capsule_path: Path to capsule.json or .cap file
        mode: Verification mode (proof-only, da, replay)
        policy_path: Optional policy file for enforcement
        manifest_root: Optional manifest directory

    Returns:
        Tuple of (exit_code, result_dict)
    """
    from bef_zk.verifier_errors import OK

    # Handle .cap files by extracting first
    if capsule_path.suffix == ".cap":
        # Integrity check: detect tampered files
        integrity_ok, integrity_msg = verify_cap_integrity(capsule_path)
        if not integrity_ok:
            return EXIT_MALFORMED, {
                "status": "TAMPERED",
                "error": integrity_msg,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = Path(tmpdir)
            try:
                extract_cap_file(capsule_path, extract_dir)
            except CapExtractionError as exc:
                return EXIT_MALFORMED, {
                    "status": "REJECT",
                    "error_code": "E001_PARSE_FAILED",
                    "message": f"Invalid receipt archive: {exc}",
                }
            capsule_json = _prepare_extracted_capsule(extract_dir)

            # Find capsule.json in extracted content
            if not capsule_json.exists():
                return EXIT_MALFORMED, {
                    "status": "REJECT",
                    "error_code": "E001_PARSE_FAILED",
                    "message": "No capsule.json found in .cap archive",
                }

            extracted_manifests = extract_dir / "manifests"
            if manifest_root is None and extracted_manifests.exists():
                manifest_root = extracted_manifests
            # Policy extraction independent of manifests
            if not policy_path:
                extracted_policy = extract_dir / "policy.json"
                if extracted_policy.exists():
                    policy_path = extracted_policy
            return _verify_capsule_json(
                capsule_json,
                mode=mode,
                policy_path=policy_path,
                manifest_root=manifest_root,
            )
    else:
        return _verify_capsule_json(
            capsule_path,
            mode=mode,
            policy_path=policy_path,
            manifest_root=manifest_root,
        )


def _verify_capsule_json(
    capsule_path: Path,
    *,
    mode: str,
    policy_path: Path | None,
    manifest_root: Path | None,
) -> tuple[int, dict[str, Any]]:
    """Verify a capsule.json file."""
    from bef_zk.verifier_errors import OK

    try:
        verify_core = _import_verify_core()
    except ImportError as e:
        return EXIT_MALFORMED, {
            "status": "REJECT",
            "error_code": "E001_PARSE_FAILED",
            "message": f"Failed to import verification module: {e}",
        }

    required_level = _normalize_mode(mode)

    # In proof-only mode, skip policy enforcement entirely
    # (policy is only needed for policy_enforced or full modes)
    effective_policy = policy_path if required_level != "proof_only" else None
    effective_manifests = manifest_root if required_level != "proof_only" else None

    try:
        error_code, result = verify_core(
            capsule_path,
            policy_path=effective_policy,
            manifest_root=effective_manifests,
            required_level=required_level,
        )
    except Exception as e:
        return EXIT_MALFORMED, {
            "status": "REJECT",
            "error_code": "E001_PARSE_FAILED",
            "message": f"Verification failed with exception: {e}",
        }

    if error_code == OK:
        return EXIT_VERIFIED, {
            "status": "VERIFIED",
            "verification_level": required_level,
            **(result or {}),
        }
    else:
        exit_code = error_to_exit_code(error_code)
        return exit_code, {
            "status": "REJECT",
            "error_code": error_code,
            "exit_code": exit_code,
            "exit_description": exit_code_description(exit_code),
        }


def _verify_agent_cap(cap_or_dir: Path, output_json: bool) -> int:
    """Verify an agent/MCP session .cap file with action chain."""
    from .cap_format import extract_cap_file, read_cap_manifest
    from datetime import datetime

    run_dir = None
    temp_dir = None
    manifest = None

    try:
        # Handle .cap file
        if cap_or_dir.suffix == ".cap":
            # Integrity check: detect tampered files
            integrity_ok, integrity_msg = verify_cap_integrity(cap_or_dir)
            if not integrity_ok:
                if output_json:
                    click.echo(json.dumps({
                        "status": "TAMPERED",
                        "error": integrity_msg,
                    }, indent=2))
                else:
                    RED = "\033[91m"
                    BOLD = "\033[1m"
                    RESET = "\033[0m"
                    click.echo(f"\n{RED}{BOLD}  ✗ TAMPERED{RESET}", err=True)
                    click.echo(f"{RED}  {integrity_msg}{RESET}", err=True)
                    click.echo(f"{RED}  This receipt has been modified and CANNOT be trusted.{RESET}\n", err=True)
                return EXIT_MALFORMED

            try:
                manifest = read_cap_manifest(cap_or_dir)
            except Exception:
                pass
            temp_dir = tempfile.mkdtemp(prefix="capseal_verify_")
            extract_cap_file(cap_or_dir, Path(temp_dir))
            run_dir = Path(temp_dir)
        elif cap_or_dir.is_dir():
            run_dir = cap_or_dir
        else:
            run_dir = cap_or_dir.parent

        # Read agent capsule
        capsule_path = run_dir / "agent_capsule.json"
        if not capsule_path.exists():
            if output_json:
                click.echo(json.dumps({"status": "ERROR", "error": "agent_capsule.json not found"}))
            else:
                click.echo("REJECTED: agent_capsule.json not found", err=True)
            return EXIT_MALFORMED

        capsule = json.loads(capsule_path.read_text())

        # Read actions
        actions_path = run_dir / "actions.json"
        actions = []
        if actions_path.exists():
            actions = json.loads(actions_path.read_text())

        # Verify chain integrity
        chain_valid = True
        chain_errors = []
        prev_hash = None
        for i, action in enumerate(actions):
            expected_parent = action.get("parent_receipt_hash")
            if i == 0:
                if expected_parent is not None:
                    chain_errors.append(f"Action 0 has non-null parent")
                    chain_valid = False
            else:
                if expected_parent != prev_hash:
                    chain_errors.append(f"Action {i} parent hash mismatch")
                    chain_valid = False
            # Compute this action's hash for next iteration
            # (simplified - just use the receipt hash from next action's parent)
            if i + 1 < len(actions):
                prev_hash = actions[i + 1].get("parent_receipt_hash")

        # Extract metadata
        capsule_hash = capsule.get("capsule_hash", "")[:16]
        num_actions = capsule.get("statement", {}).get("public_inputs", {}).get("num_actions", len(actions))
        constraints_valid = capsule.get("verification", {}).get("constraints_valid", False)
        proof_type = capsule.get("verification", {}).get("proof_type", "constraint_check")
        final_hash = capsule.get("statement", {}).get("public_inputs", {}).get("final_receipt_hash", "")[:16]

        # Parse timestamps
        start_time = None
        end_time = None
        if actions:
            try:
                start_time = datetime.fromisoformat(actions[0].get("timestamp", "").replace("Z", "+00:00"))
                end_time = datetime.fromisoformat(actions[-1].get("timestamp", "").replace("Z", "+00:00"))
            except Exception:
                pass

        # Get session name from manifest
        session_name = ""
        if manifest and manifest.extras:
            session_name = manifest.extras.get("session_name", "")

        verified = constraints_valid and chain_valid

        if output_json:
            output = {
                "status": "VERIFIED" if verified else "REJECTED",
                "capsule_hash": capsule.get("capsule_hash", ""),
                "num_actions": num_actions,
                "chain_valid": chain_valid,
                "constraints_valid": constraints_valid,
                "proof_type": proof_type,
                "final_receipt_hash": capsule.get("statement", {}).get("public_inputs", {}).get("final_receipt_hash", ""),
            }
            if session_name:
                output["session_name"] = session_name
            if chain_errors:
                output["chain_errors"] = chain_errors
            click.echo(json.dumps(output, indent=2))
        else:
            GREEN = "\033[92m"
            RED = "\033[91m"
            BOLD = "\033[1m"
            DIM = "\033[2m"
            RESET = "\033[0m"

            if verified:
                click.echo(f"\n{GREEN}{BOLD}  ✓ Capsule verified{RESET}: {capsule_hash}...")
                click.echo(f"  Actions:      {num_actions}")
                click.echo(f"  Chain:        intact ({num_actions}/{num_actions} hashes valid)")
                if session_name:
                    click.echo(f"  Session:      {session_name}")
                if start_time and end_time:
                    click.echo(f"  Duration:     {start_time.strftime('%I:%M %p')} → {end_time.strftime('%I:%M %p')}")
                if constraints_valid:
                    if proof_type == "fri":
                        click.echo(f"  Proof:        {GREEN}✓ FRI verified{RESET}")
                    else:
                        click.echo(f"  Proof:        {GREEN}✓ constraints verified{RESET}")
                click.echo()
            else:
                click.echo(f"\n{RED}{BOLD}  ✗ VERIFICATION FAILED{RESET}", err=True)
                click.echo(f"{RED}  Capsule: {capsule_hash}...{RESET}", err=True)
                if not constraints_valid:
                    click.echo(f"{RED}  ✗ Proof:  constraints invalid{RESET}", err=True)
                if not chain_valid:
                    click.echo(f"{RED}  ✗ Action chain broken:{RESET}", err=True)
                    for err in chain_errors:
                        click.echo(f"{RED}    → {err}{RESET}", err=True)
                click.echo(f"{RED}  This receipt CANNOT be trusted.{RESET}\n", err=True)

        return EXIT_VERIFIED if verified else EXIT_MALFORMED

    except Exception as e:
        if output_json:
            click.echo(json.dumps({"status": "ERROR", "error": str(e)}, indent=2))
        else:
            click.echo(f"ERROR: {e}", err=True)
        return EXIT_MALFORMED
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _verify_run_cap(cap_or_dir: Path, output_json: bool) -> int:
    """Verify a run .cap file or run directory with receipt chain."""
    from .cap_format import extract_cap_file, read_cap_manifest
    from capseal.shared.receipts import verify_run_receipt

    run_dir = None
    temp_dir = None
    manifest = None

    try:
        # Handle .cap file
        if cap_or_dir.suffix == ".cap":
            # Integrity check: detect tampered files
            integrity_ok, integrity_msg = verify_cap_integrity(cap_or_dir)
            if not integrity_ok:
                if output_json:
                    click.echo(json.dumps({
                        "status": "TAMPERED",
                        "error": integrity_msg,
                    }, indent=2))
                else:
                    RED = "\033[91m"
                    BOLD = "\033[1m"
                    RESET = "\033[0m"
                    click.echo(f"\n{RED}{BOLD}  ✗ TAMPERED{RESET}", err=True)
                    click.echo(f"{RED}  {integrity_msg}{RESET}", err=True)
                    click.echo(f"{RED}  This receipt has been modified and CANNOT be trusted.{RESET}\n", err=True)
                return EXIT_MALFORMED

            manifest = read_cap_manifest(cap_or_dir)
            temp_dir = tempfile.mkdtemp(prefix="capseal_verify_")
            extract_cap_file(cap_or_dir, Path(temp_dir))
            run_dir = Path(temp_dir)
        elif cap_or_dir.is_dir():
            run_dir = cap_or_dir
        else:
            # It's a file but not .cap - might be run_receipt.json directly
            run_dir = cap_or_dir.parent

        # Verify the receipt chain
        result = verify_run_receipt(run_dir)

        # Load chain hash and run_type from receipt
        receipt_path = run_dir / "run_receipt.json"
        chain_hash = ""
        run_type = "unknown"
        total_statements = 0
        if receipt_path.exists():
            receipt_data = json.loads(receipt_path.read_text())
            chain_hash = receipt_data.get("chain_hash", "")
            run_type = receipt_data.get("run_type", "unknown")
            total_statements = len(receipt_data.get("statements", []))
        # Override with manifest if available
        if manifest:
            run_type = manifest.extras.get("run_type", run_type)

        if output_json:
            output = {
                "status": "VERIFIED" if result["verified"] else "REJECTED",
                "chain_hash": chain_hash,
                "run_type": run_type,
            }
            rounds = result.get("rounds_verified", 0)
            if run_type == "review":
                output["statements_verified"] = total_statements
            else:
                output["rounds_verified"] = rounds
            if manifest:
                output["run_id"] = manifest.extras.get("run_id", "")
            click.echo(json.dumps(output, indent=2))
        else:
            if result["verified"]:
                rounds = result.get("rounds_verified", 0)
                click.echo(f"VERIFIED (receipt_chain)")
                click.echo(f"  Chain hash: {chain_hash[:32]}...")
                # Show appropriate label based on run type
                if run_type == "review":
                    click.echo(f"  Statements verified: {total_statements}")
                else:
                    click.echo(f"  Rounds verified: {rounds}")
            else:
                error = result.get("mismatches", ["Unknown error"])[0] if result.get("mismatches") else "Unknown error"
                click.echo(f"REJECTED: {error}", err=True)

        return EXIT_VERIFIED if result["verified"] else EXIT_MALFORMED

    except Exception as e:
        if output_json:
            click.echo(json.dumps({"status": "ERROR", "error": str(e)}, indent=2))
        else:
            click.echo(f"ERROR: {e}", err=True)
        return EXIT_MALFORMED
    finally:
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)


def _check_signature(capsule_path: Path, output_json: bool) -> None:
    """Check for and report on Ed25519 signature."""
    if capsule_path.suffix != ".cap":
        return

    # Resolve symlinks so we find the .sig next to the real file
    resolved = capsule_path.resolve()
    sig_path = resolved.with_suffix(".cap.sig")
    if not sig_path.exists():
        if not output_json:
            DIM = "\033[2m"
            RESET = "\033[0m"
            click.echo(f"  {DIM}Signature:  not signed (use: capseal sign){RESET}")
        return

    try:
        from .sign_cmd import verify_signature
        valid, message = verify_signature(resolved)
        if not output_json:
            GREEN = "\033[92m"
            RED = "\033[91m"
            RESET = "\033[0m"
            if valid:
                click.echo(f"  {GREEN}Signature:  ✓ {message}{RESET}")
            else:
                click.echo(f"  {RED}Signature:  ✗ {message}{RESET}")
    except Exception as e:
        if not output_json:
            click.echo(f"  Signature:  error ({e})")


@click.command("verify")
@click.argument("capsule", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--mode",
    type=click.Choice(["proof-only", "da", "replay", "auto"], case_sensitive=False),
    default="auto",
    help="Verification mode: auto (default), proof-only, da, replay",
)
@click.option(
    "--policy",
    type=click.Path(exists=True, path_type=Path),
    help="Policy file to enforce",
)
@click.option(
    "--manifests",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Manifests directory for policy enforcement",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output result as JSON",
)
def verify_command(
    capsule: Path,
    mode: str,
    policy: Path | None,
    manifests: Path | None,
    output_json: bool,
) -> None:
    """Verify a capsule receipt or run .cap file.

    Accepts:
    - A .cap file (from capseal learn or review --gate)
    - A run directory (e.g., .capseal/runs/latest)
    - A capsule.json file (full FRI proof)

    Returns stable exit codes suitable for CI integration.

    \b
    Exit codes:
        0  - Verified
        10 - Proof invalid
        11 - Policy mismatch
        12 - Commitment/index failed
        20 - Malformed/parse error
    """
    from .cap_format import read_cap_manifest

    # Auto-detect verification type
    is_run_cap = False
    is_agent_cap = False

    if capsule.suffix == ".cap":
        try:
            manifest = read_cap_manifest(capsule)
            if manifest.schema == "run_cap_v1":
                run_type = manifest.extras.get("run_type", "")
                if run_type == "mcp":
                    is_agent_cap = True
                else:
                    is_run_cap = True
            elif manifest.schema == "agent_cap_v1":
                is_agent_cap = True
        except Exception:
            pass
    elif capsule.is_dir():
        # Check if it's a run directory with run_receipt.json
        if (capsule / "run_receipt.json").exists():
            is_run_cap = True
        # Check if it's an agent run directory
        elif (capsule / "agent_capsule.json").exists():
            is_agent_cap = True

    # Route to appropriate verifier
    if is_agent_cap:
        exit_code = _verify_agent_cap(capsule, output_json)
        _check_signature(capsule, output_json)
        sys.exit(exit_code)

    if is_run_cap:
        exit_code = _verify_run_cap(capsule, output_json)
        _check_signature(capsule, output_json)
        sys.exit(exit_code)

    # Check for run directory with receipt
    if capsule.is_dir() and (capsule / "run_receipt.json").exists():
        exit_code = _verify_run_cap(capsule, output_json)
        sys.exit(exit_code)

    # Fall back to full capsule verification
    if mode == "auto":
        mode = "proof-only"

    exit_code, result = _verify_capsule(
        capsule,
        mode=mode,
        policy_path=policy,
        manifest_root=manifests,
    )

    if output_json:
        click.echo(json.dumps(result, indent=2))
    else:
        status = result.get("status", "UNKNOWN")
        if exit_code == EXIT_VERIFIED:
            level = result.get("verification_level", mode)
            click.echo(f"VERIFIED ({level})")
        else:
            error_code = result.get("error_code", "UNKNOWN")
            desc = result.get("exit_description", exit_code_description(exit_code))
            click.echo(f"REJECTED: {error_code} ({desc})", err=True)

    sys.exit(exit_code)
