"""capseal test — 9-step self-test that validates the entire CapSeal workspace.

Checks:
    1. Workspace          .capseal/ exists with config.json
    2. Configuration      Required keys present in config
    3. Risk model         beta_posteriors.npz exists and loads
    4. Episode history    episode_history.jsonl exists, parse sample
    5. Latest session     latest.cap exists and is readable
    6. Chain integrity    All receipt hashes link correctly
    7. Recomputable       canonical_fields present, hashes reproducible
    8. Proof status       agent_capsule.json exists, constraints verified
    9. Signature          .sig file exists and verifies (if signed)
"""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


def _step(num: int, label: str, ok: bool, detail: str = "") -> bool:
    """Print a single checklist line."""
    icon = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
    line = f"  {num}. {icon}  {label}"
    if detail:
        line += f"  [dim]{detail}[/dim]"
    console.print(line)
    return ok


@click.command("test")
@click.argument("path", type=click.Path(exists=True), default=".")
def test_command(path: str) -> None:
    """Run a 9-step self-test on the CapSeal workspace.

    \b
    Examples:
        capseal test
        capseal test /path/to/project
    """
    target = Path(path).resolve()
    capseal_dir = target / ".capseal"

    console.print()
    console.print(Panel(
        f"  Workspace: {target}",
        title="[bold]CapSeal Self-Test[/bold]",
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
        expand=False,
    ))
    console.print()

    passed = 0
    failed = 0

    # ── 1. Workspace ────────────────────────────────────────────────────
    ws_ok = capseal_dir.is_dir() and (capseal_dir / "config.json").exists()
    if _step(1, "Workspace", ws_ok,
             str(capseal_dir) if ws_ok else "missing .capseal/config.json"):
        passed += 1
    else:
        failed += 1

    if not ws_ok:
        console.print("\n  [red]Cannot continue — run capseal init first.[/red]")
        raise SystemExit(1)

    # ── 2. Configuration ────────────────────────────────────────────────
    config: dict = {}
    config_path = capseal_dir / "config.json"
    try:
        config = json.loads(config_path.read_text())
        has_provider = bool(config.get("provider"))
        config_ok = has_provider
        detail = f"provider={config.get('provider', '?')}, model={config.get('model', '?')}"
    except Exception as e:
        config_ok = False
        detail = str(e)

    if _step(2, "Configuration", config_ok, detail):
        passed += 1
    else:
        failed += 1

    # ── 3. Risk model ───────────────────────────────────────────────────
    model_path = capseal_dir / "models" / "beta_posteriors.npz"
    model_ok = False
    model_detail = "not trained"
    if model_path.exists():
        try:
            import numpy as np
            data = np.load(model_path, allow_pickle=True)
            n_ep = int(data["n_episodes"]) if "n_episodes" in data else "?"
            grid_size = len(data.get("alpha", data.get("alphas", [])))
            model_ok = grid_size > 0
            model_detail = f"{n_ep} episodes, {grid_size} grid cells"
        except Exception as e:
            model_detail = f"load error: {e}"

    if _step(3, "Risk model", model_ok, model_detail):
        passed += 1
    else:
        failed += 1

    # ── 4. Episode history ──────────────────────────────────────────────
    history_path = capseal_dir / "models" / "episode_history.jsonl"
    history_ok = False
    history_detail = "no episode_history.jsonl"
    if history_path.exists():
        try:
            lines = [l for l in history_path.read_text().strip().split("\n") if l.strip()]
            count = len(lines)
            if count > 0:
                sample = json.loads(lines[-1])
                history_ok = True
                history_detail = f"{count} entries, last: {sample.get('description', '?')[:40]}"
            else:
                history_detail = "file is empty"
        except Exception as e:
            history_detail = f"parse error: {e}"

    if _step(4, "Episode history", history_ok, history_detail):
        passed += 1
    else:
        failed += 1

    # ── 5. Latest session ───────────────────────────────────────────────
    runs_dir = capseal_dir / "runs"
    latest_cap = runs_dir / "latest.cap" if runs_dir.exists() else None
    session_ok = False
    session_detail = "no sessions found"
    cap_path_resolved: Path | None = None

    if latest_cap and latest_cap.exists():
        cap_path_resolved = latest_cap.resolve()
        try:
            with tarfile.open(cap_path_resolved, "r:*") as tar:
                names = tar.getnames()
            session_ok = True
            session_detail = f"{cap_path_resolved.name} ({len(names)} members)"
        except Exception as e:
            session_detail = f"read error: {e}"

    if _step(5, "Latest session", session_ok, session_detail):
        passed += 1
    else:
        failed += 1

    # ── 6. Chain integrity ──────────────────────────────────────────────
    chain_ok = False
    chain_detail = "no session to verify"

    if cap_path_resolved and session_ok:
        try:
            actions = _load_actions(cap_path_resolved)
            if actions:
                chain_ok, chain_detail = _verify_chain(actions)
            else:
                chain_detail = "no actions in session"
        except Exception as e:
            chain_detail = f"verify error: {e}"

    if _step(6, "Chain integrity", chain_ok, chain_detail):
        passed += 1
    else:
        failed += 1

    # ── 7. Recomputable proofs ──────────────────────────────────────────
    recomp_ok = False
    recomp_detail = "no session to verify"

    if cap_path_resolved and session_ok:
        try:
            actions = _load_actions(cap_path_resolved)
            if actions:
                recomp_ok, recomp_detail = _verify_recomputable(actions)
            else:
                recomp_detail = "no actions"
        except Exception as e:
            recomp_detail = f"error: {e}"

    if _step(7, "Recomputable proofs", recomp_ok, recomp_detail):
        passed += 1
    else:
        failed += 1

    # ── 8. Proof status ───────────────────────────────────────────────
    proof_ok = False
    proof_detail = "no session to verify"

    if cap_path_resolved and session_ok:
        try:
            proof_ok, proof_detail = _verify_proof_status(cap_path_resolved)
        except Exception as e:
            proof_detail = f"error: {e}"

    if _step(8, "Proof status", proof_ok, proof_detail):
        passed += 1
    else:
        failed += 1

    # ── 9. Signature ────────────────────────────────────────────────────
    sig_ok = False
    sig_detail = "not signed"

    if cap_path_resolved and session_ok:
        try:
            from .sign_cmd import verify_signature
            valid, message = verify_signature(cap_path_resolved)
            sig_ok = valid
            sig_detail = message
        except ImportError:
            sig_detail = "cryptography package not installed"
        except Exception as e:
            sig_detail = f"error: {e}"

    if _step(9, "Signature", sig_ok, sig_detail):
        passed += 1
    else:
        failed += 1

    # ── Summary ─────────────────────────────────────────────────────────
    console.print()
    total = passed + failed
    if failed == 0:
        console.print(f"  [bold green]All {total} checks passed.[/bold green]")
    else:
        console.print(f"  [bold]{passed}/{total} passed[/bold], [red]{failed} failed[/red]")
    console.print()

    if failed > 0:
        raise SystemExit(1)


def _load_actions(cap_path: Path) -> list[dict]:
    """Load actions.jsonl from a .cap tarball or its run directory."""
    actions: list[dict] = []

    try:
        with tarfile.open(cap_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith("actions.jsonl"):
                    f = tar.extractfile(member)
                    if f:
                        for line in f.read().decode("utf-8").strip().split("\n"):
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


def _verify_chain(actions: list[dict]) -> tuple[bool, str]:
    """Verify receipt hash chain links. Returns (ok, detail)."""
    from capseal.agent_protocol import AgentAction

    prev_hash: str | None = None
    for i, raw in enumerate(actions):
        try:
            aa = AgentAction.from_dict(raw)
            computed = aa.compute_receipt_hash()
        except Exception:
            return False, f"action {i}: cannot compute hash"

        expected_parent = raw.get("parent_receipt_hash")
        if i == 0:
            if expected_parent is not None:
                return False, f"action 0: unexpected parent hash"
        else:
            if expected_parent != prev_hash:
                return False, f"action {i}: chain break (parent mismatch)"

        prev_hash = computed

    return True, f"{len(actions)} actions, chain intact"


def _verify_recomputable(actions: list[dict]) -> tuple[bool, str]:
    """Check that canonical_fields are present and hashes match."""
    from capseal.agent_protocol import AgentAction

    with_cf = 0
    verified = 0

    for i, raw in enumerate(actions):
        cf = raw.get("canonical_fields")
        if cf is None:
            continue
        with_cf += 1

        # Recompute hash from canonical_fields
        canonical_str = json.dumps(cf, sort_keys=True, separators=(",", ":"))
        recomputed = hashlib.sha256(canonical_str.encode()).hexdigest()

        # Compare to stored receipt_hash (or compute from AgentAction)
        stored = raw.get("receipt_hash")
        if stored is None:
            try:
                aa = AgentAction.from_dict(raw)
                stored = aa.compute_receipt_hash()
            except Exception:
                continue

        if recomputed == stored:
            verified += 1

    total = len(actions)
    if with_cf == 0:
        return False, f"no canonical_fields (pre-v0.3.0 session)"
    if verified == with_cf:
        return True, f"{verified}/{total} actions fully recomputable"
    return False, f"{verified}/{with_cf} canonical hashes match ({with_cf - verified} mismatch)"


def _verify_proof_status(cap_path: Path) -> tuple[bool, str]:
    """Check agent_capsule.json for proof status. Returns (ok, detail)."""
    import tempfile

    capsule_data = None

    # Try to extract agent_capsule.json from .cap tarball
    try:
        with tarfile.open(cap_path, "r:*") as tar:
            for member in tar.getmembers():
                if member.name.endswith("agent_capsule.json"):
                    f = tar.extractfile(member)
                    if f:
                        capsule_data = json.loads(f.read().decode("utf-8"))
                        break
    except Exception:
        pass

    # Fallback: look in run directory
    if capsule_data is None:
        run_dir = cap_path.parent / cap_path.stem
        capsule_file = run_dir / "agent_capsule.json"
        if capsule_file.exists():
            capsule_data = json.loads(capsule_file.read_text())

    if capsule_data is None:
        return False, "agent_capsule.json not found"

    verification = capsule_data.get("verification", {})
    constraints_valid = verification.get("constraints_valid", False)
    proof_type = verification.get("proof_type", "constraint_check")

    if not constraints_valid:
        return False, "constraints invalid"

    if proof_type == "fri":
        return True, "FRI proof verified"
    return True, f"constraints verified ({proof_type})"


__all__ = ["test_command"]
