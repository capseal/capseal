"""capseal export-receipt — Standalone, machine-checkable receipt export.

Produces a self-describing JSON file that any third party can verify
independently using only sha256sum and jq — no capseal installation needed.

Usage:
    capseal export-receipt .capseal/runs/latest.cap
    capseal export-receipt .capseal/runs/latest.cap -o receipt.json
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime
from pathlib import Path

import click


@click.command("export-receipt")
@click.argument("cap_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Output path (default: <cap_name>-receipt.json)")
@click.option("--print", "print_output", is_flag=True,
              help="Print to stdout instead of writing a file")
@click.option("--with-script", is_flag=True,
              help="Generate standalone verify.sh alongside the receipt")
def export_receipt_command(cap_file: str, output: str | None, print_output: bool, with_script: bool) -> None:
    """Export a standalone, machine-checkable receipt from a .cap file.

    The exported JSON contains everything needed for independent verification:
    the full action chain with hashes, chain integrity status, and canonical
    fields for recomputing every hash with sha256sum.

    \b
    Examples:
        capseal export-receipt .capseal/runs/latest.cap
        capseal export-receipt .capseal/runs/latest.cap -o receipt.json
        capseal export-receipt .capseal/runs/latest.cap --with-script
        capseal export-receipt .capseal/runs/latest.cap --print | jq .
    """
    cap_path = Path(cap_file).resolve()
    receipt = _build_receipt(cap_path)

    content = json.dumps(receipt, indent=2) + "\n"

    if print_output:
        click.echo(content)
    else:
        if output:
            out_path = Path(output)
        else:
            out_path = cap_path.parent / f"{cap_path.stem}-receipt.json"
        out_path.write_text(content)
        click.echo(f"Receipt exported to: {out_path}")
        click.echo(f"  Actions:  {len(receipt.get('actions', []))}")
        click.echo(f"  Chain:    {'valid' if receipt['integrity']['chain_valid'] else 'BROKEN'}")
        click.echo(f"  Hash:     {receipt.get('chain_hash', 'none')[:16]}...")

        if with_script:
            script_path = _generate_verify_script(
                receipt["actions"], receipt["chain_hash"], out_path.parent,
            )
            click.echo(f"  Script:   {script_path}")
            recomputable = receipt["integrity"].get("fully_recomputable", False)
            if recomputable:
                click.echo(f"  Run: bash {script_path}")
            else:
                click.echo(f"  Note: some actions lack canonical_fields (recorded before v0.3.0)")


def _build_receipt(cap_path: Path) -> dict:
    """Build a standalone receipt dict from a .cap file."""
    from capseal.agent_protocol import AgentAction

    # Load manifest
    manifest = _load_manifest(cap_path)

    # Load actions
    raw_actions = _load_actions(cap_path)

    # Parse session info from filename
    name = cap_path.stem
    try:
        ts_part = name[:15]
        dt = datetime.strptime(ts_part, "%Y%m%dT%H%M%S")
        timestamp = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, IndexError):
        timestamp = manifest.get("created_at", "")

    run_type = name.split("-", 1)[1] if "-" in name else "unknown"
    session_name = manifest.get("session_name", run_type)
    agent = ""
    for a in raw_actions:
        meta = a.get("metadata") or {}
        if meta.get("agent"):
            agent = meta["agent"]
            break

    # Build action entries with verification
    actions = []
    chain_valid = True
    prev_hash: str | None = None
    final_hash = ""

    for i, raw in enumerate(raw_actions):
        # Compute receipt hash
        try:
            aa = AgentAction.from_dict(raw)
            receipt_hash = aa.compute_receipt_hash()
            canonical = aa.canonical_json()
        except Exception:
            receipt_hash = "error"
            canonical = ""
            chain_valid = False

        # Verify chain link
        expected_parent = raw.get("parent_receipt_hash")
        link_valid = True
        if i == 0:
            if expected_parent is not None:
                link_valid = False
        else:
            if expected_parent != prev_hash:
                link_valid = False

        if not link_valid:
            chain_valid = False

        # Extract display fields from action
        metadata = raw.get("metadata") or {}
        action_entry = {
            "index": i,
            "type": raw.get("action_type", "unknown"),
            "description": metadata.get("description", ""),
            "files": metadata.get("files_affected", []),
            "gate_decision": _normalize_decision(raw.get("gate_decision")),
            "gate_p_fail": raw.get("gate_score"),
            "success": raw.get("success", True),
            "receipt_hash": receipt_hash,
            "parent_hash": expected_parent,
            "timestamp": raw.get("timestamp", ""),
            "canonical_fields": raw.get("canonical_fields"),
        }
        actions.append(action_entry)
        prev_hash = receipt_hash
        final_hash = receipt_hash

    # Compute chain hash (hash of all receipt hashes concatenated)
    if actions:
        all_hashes = "".join(a["receipt_hash"] for a in actions if a["receipt_hash"] != "error")
        chain_hash = hashlib.sha256(all_hashes.encode()).hexdigest()
    else:
        chain_hash = ""

    # Build verification instructions
    verify_instructions = _build_verify_instructions(actions)

    fully_recomputable = all(a.get("canonical_fields") is not None for a in actions)

    return {
        "schema": "capseal_receipt_v1",
        "session": session_name,
        "timestamp": timestamp,
        "agent": agent,
        "source_file": cap_path.name,
        "actions": actions,
        "chain_hash": chain_hash,
        "integrity": {
            "algorithm": "sha256",
            "canonical_encoding": "json.dumps(canonical_fields, sort_keys=True, separators=(',',':'))",
            "chain_construction": "sha256(receipt_hash_0 + receipt_hash_1 + ... + receipt_hash_N)",
            "chain_hash": chain_hash,
            "chain_valid": chain_valid,
            "fully_recomputable": fully_recomputable,
            "total_actions": len(actions),
            "hash_fields": [
                "action_id", "action_type", "instruction_hash", "input_hash",
                "output_hash", "parent_action_id", "parent_receipt_hash",
                "gate_score", "gate_decision", "policy_verdict",
                "success", "duration_ms", "timestamp",
            ],
            "excluded_fields": ["metadata"],
            "verify_action": "echo -n '<canonical_json>' | sha256sum",
            "verify_chain": "echo -n '<all_receipt_hashes_concatenated>' | sha256sum",
            "verification_steps": verify_instructions,
        },
    }


def _normalize_decision(decision: str | None) -> str | None:
    """Map internal decision names to human-readable ones."""
    if decision is None:
        return None
    return {
        "pass": "approve",
        "skip": "deny",
        "human_review": "flag",
        "human_approved": "approved (after review)",
    }.get(decision, decision)


def _build_verify_instructions(actions: list[dict]) -> list[str]:
    """Generate step-by-step verification instructions."""
    steps = [
        "To verify this receipt independently (no capseal installation needed):",
        "",
        "1. For each action, reconstruct the canonical JSON from the hash_fields",
        "   (see integrity.hash_fields for the exact field list).",
        "   Use sorted keys and compact separators: json.dumps(obj, sort_keys=True, separators=(',',':'))",
        "",
        "2. Compute SHA256 of the canonical JSON:",
        "   echo -n '<canonical_json>' | sha256sum",
        "",
        "3. Verify each action's receipt_hash matches the computed hash.",
        "",
        "4. Verify the chain: each action's parent_hash must equal the",
        "   previous action's receipt_hash. The first action has parent_hash=null.",
        "",
        "5. Verify chain_hash: concatenate all receipt_hashes and SHA256 the result:",
        "   echo -n '<hash_0><hash_1>...<hash_n>' | sha256sum",
    ]

    # Add a concrete example if we have actions
    if actions and actions[0].get("receipt_hash") != "error":
        steps.extend([
            "",
            f"Example — verify action 0:",
            f"  Expected receipt_hash: {actions[0]['receipt_hash']}",
            f"  Compute: echo -n '{{canonical_json_of_action_0}}' | sha256sum",
            f"  The output should match the receipt_hash above.",
        ])

    return steps


def _generate_verify_script(
    actions: list[dict], chain_hash: str, output_dir: Path,
) -> Path:
    """Generate a standalone verify.sh that recomputes all hashes."""
    lines = [
        '#!/bin/bash',
        '# Auto-generated by capseal export-receipt --with-script',
        '# Verifies all action hashes and chain integrity',
        '# Requires: sha256sum, bash',
        '',
        'PASS=0; FAIL=0',
        '',
    ]

    hash_vars = []
    for i, action in enumerate(actions):
        cf = action.get("canonical_fields")
        desc = (action.get("description", "") or "")[:60]
        lines.append(f'# Action {i}: {desc}')

        if cf is None:
            lines.append(f'echo "⚠ Action {i}: no canonical_fields (recorded before v0.3.0)"')
            lines.append(f"RECEIPT_{i}='{action['receipt_hash']}'")
            hash_vars.append(f'${{RECEIPT_{i}}}')
            lines.append('')
            continue

        # Reconstruct the exact canonical JSON
        canonical = json.dumps(cf, sort_keys=True, separators=(',', ':'))
        # Escape single quotes for bash
        escaped = canonical.replace("'", "'\\''")

        lines.append(f"EXPECTED_{i}='{action['receipt_hash']}'")
        lines.append(f"ACTUAL_{i}=$(printf '%s' '{escaped}' | sha256sum | cut -d' ' -f1)")
        lines.append(f'if [ "$EXPECTED_{i}" = "$ACTUAL_{i}" ]; then')
        lines.append(f'  echo "✓ Action {i}: verified"; PASS=$((PASS+1))')
        lines.append('else')
        lines.append(f'  echo "✗ Action {i}: MISMATCH"; FAIL=$((FAIL+1))')
        lines.append('fi')
        lines.append('')
        hash_vars.append(f'${{ACTUAL_{i}}}')

    # Chain verification
    concat_expr = ''.join(hash_vars)
    lines.append('# Chain integrity')
    lines.append(f"EXPECTED_CHAIN='{chain_hash}'")
    lines.append(f'ACTUAL_CHAIN=$(printf \'%s\' "{concat_expr}" | sha256sum | cut -d\' \' -f1)')
    lines.append('if [ "$EXPECTED_CHAIN" = "$ACTUAL_CHAIN" ]; then')
    lines.append('  echo "✓ Chain: intact"; PASS=$((PASS+1))')
    lines.append('else')
    lines.append('  echo "✗ Chain: BROKEN"; FAIL=$((FAIL+1))')
    lines.append('fi')
    lines.append('')
    lines.append('echo ""')
    lines.append('echo "Results: $PASS passed, $FAIL failed"')
    lines.append('[ $FAIL -eq 0 ] && exit 0 || exit 1')

    script_path = output_dir / 'verify.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    return script_path


def _load_manifest(cap_path: Path) -> dict:
    """Load manifest.json from a .cap tarball."""
    try:
        with tarfile.open(cap_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith("manifest.json"):
                    f = tar.extractfile(member)
                    if f:
                        return json.loads(f.read().decode("utf-8"))
    except Exception:
        pass
    return {}


def _load_actions(cap_path: Path) -> list[dict]:
    """Load actions.jsonl from a .cap tarball or its run directory."""
    actions: list[dict] = []

    # Try tarball first
    try:
        with tarfile.open(cap_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith("actions.jsonl"):
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8").strip()
                        for line in content.split("\n"):
                            if line.strip():
                                try:
                                    actions.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                        return actions
    except Exception:
        pass

    # Fallback: run directory
    run_dir = cap_path.parent / cap_path.stem
    actions_file = run_dir / "actions.jsonl"
    if actions_file.exists():
        for line in actions_file.read_text().strip().split("\n"):
            if line.strip():
                try:
                    actions.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    return actions


__all__ = ["export_receipt_command"]
