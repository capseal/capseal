"""One-command CapSeal onboarding flow."""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from capseal.agent_runtime import AgentRuntime
from capseal.cli.cap_format import create_run_cap_file, verify_cap_integrity
from capseal.demo_diffs import DEMO_DIFFS, DemoDiff
from capseal.risk_engine import THRESHOLD_DENY, evaluate_risk, to_internal_decision

# Keep quickstart bounded so first-run never looks hung.
QUICKSTART_GIT_MAX_COMMITS = 8
QUICKSTART_GIT_MAX_SECONDS = 25
QUICKSTART_GIT_SEMGREP_TIMEOUT_SECONDS = 6


@dataclass
class QuickstartOutcome:
    cap_path: Path
    verified: bool
    verify_message: str


class QuickstartPrinter:
    def __init__(self, color: bool = True):
        self.color = color

    def _c(self, text: str, code: str) -> str:
        if not self.color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def header(self, text: str) -> None:
        print(self._c(text, "1;36"))
        print()

    def step(self, text: str) -> None:
        print(f"- {text}")

    def done(self, text: str) -> None:
        print(f"  {self._c('ok', '32')} {text}")

    def note(self, text: str) -> None:
        print(f"  {self._c('note', '33')} {text}")

    def gate_result(self, result: Any) -> None:
        border = "+" + "-" * 62 + "+"
        print()
        print(border)
        print(f"| Decision: {result.decision.upper():<53}|")
        print(f"| Label:    {result.label[:53]:<53}|")
        print(f"| p_fail:   {result.p_fail:0.2f}  confidence: {result.confidence:0.2f}{'':<21}|")
        print(f"| Grid:     {result.grid_cell:<53}|")
        print(border)
        print()

    def next_steps(self) -> None:
        print("Next steps:")
        print("  capseal learn . --rounds 5")
        print("  capseal mcp-serve")
        print("  capseal doctor")


def _ensure_workspace_initialized(workspace: Path) -> None:
    cap = workspace / ".capseal"
    (cap / "runs").mkdir(parents=True, exist_ok=True)
    (cap / "models").mkdir(parents=True, exist_ok=True)
    (cap / "policies").mkdir(parents=True, exist_ok=True)
    (cap / "events.jsonl").touch(exist_ok=True)

    config_path = cap / "config.json"
    if not config_path.exists():
        # Reuse the same non-interactive init path as autopilot/hub.
        from capseal.cli.autopilot_cmd import _auto_init, _detect_provider

        detected = _detect_provider()
        if detected:
            provider, env_var, model = detected
        else:
            provider, env_var, model = ("anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-20250514")
        _auto_init(workspace, provider, env_var, model)

    # Ensure quickstart-required defaults are present without clobbering user config.
    try:
        config = json.loads(config_path.read_text())
    except Exception:
        config = {}
    config.setdefault("workspace", str(workspace))
    gate_cfg = config.setdefault("gate", {})
    gate_cfg.setdefault("threshold", THRESHOLD_DENY)
    gate_cfg.setdefault("uncertainty_threshold", 0.15)
    learning_cfg = config.setdefault("learning", {})
    learning_cfg.setdefault("episode_timeout_seconds", 180)
    config_path.write_text(json.dumps(config, indent=2))


def _quick_learn_from_git(
    workspace: Path,
    max_commits: int = QUICKSTART_GIT_MAX_COMMITS,
    max_duration_seconds: float = QUICKSTART_GIT_MAX_SECONDS,
    semgrep_timeout_seconds: int = QUICKSTART_GIT_SEMGREP_TIMEOUT_SECONDS,
) -> int:
    """Run a fast git-history learn pass and persist posteriors."""
    try:
        from capseal.cli.learn_cmd import _run_git_learn
    except Exception:
        return 0

    _run_git_learn(
        target_path=workspace,
        quiet=True,
        CYAN="",
        GREEN="",
        YELLOW="",
        RED="",
        DIM="",
        BOLD="",
        RESET="",
        max_commits=max_commits,
        max_duration_seconds=max_duration_seconds,
        semgrep_timeout_seconds=semgrep_timeout_seconds,
    )

    model_path = workspace / ".capseal" / "models" / "beta_posteriors.npz"
    if not model_path.exists():
        return 0
    data = np.load(model_path, allow_pickle=True)
    alpha = data["alpha"] if "alpha" in data.files else (data["alphas"] if "alphas" in data.files else None)
    beta = data["beta"] if "beta" in data.files else (data["betas"] if "betas" in data.files else None)
    if alpha is None or beta is None:
        return 0
    return int(np.sum((alpha > 1) | (beta > 1)))


def _bootstrap_demo_model(workspace: Path, demo_diff: DemoDiff) -> Path:
    """Boost the demo diff cell inside the canonical workspace model.

    This keeps quickstart and regular gate paths consistent because both
    read the same posteriors file afterward.
    """
    source_model_path = workspace / ".capseal" / "models" / "beta_posteriors.npz"
    existing: dict[str, Any] = {}
    if source_model_path.exists():
        data = np.load(source_model_path, allow_pickle=True)
        alpha = data["alpha"] if "alpha" in data.files else (data["alphas"] if "alphas" in data.files else None)
        beta = data["beta"] if "beta" in data.files else (data["betas"] if "betas" in data.files else None)
        if alpha is None or beta is None:
            alpha = np.ones(1024, dtype=np.int64)
            beta = np.ones(1024, dtype=np.int64)
        alpha = alpha.copy()
        beta = beta.copy()
        for key in data.files:
            if key not in ("alpha", "beta", "alphas", "betas"):
                existing[key] = data[key]
    else:
        alpha = np.ones(1024, dtype=np.int64)
        beta = np.ones(1024, dtype=np.int64)

    seed = evaluate_risk(
        demo_diff.content,
        workspace=workspace,
        model_path=source_model_path if source_model_path.exists() else None,
    )
    idx = seed.grid_cell
    alpha[idx] += 12  # strong demo-only prior for the chosen diff profile
    if "n_episodes" in existing:
        try:
            existing["n_episodes"] = int(existing["n_episodes"]) + 12
        except Exception:
            existing["n_episodes"] = 12
    else:
        existing["n_episodes"] = 12
    existing["run_uuid"] = "quickstart-bootstrap"
    source_model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(source_model_path, alpha=alpha, beta=beta, **existing)
    return source_model_path


def _seal_demo_session(workspace: Path, result: Any, demo_diff: DemoDiff) -> QuickstartOutcome:
    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = workspace / ".capseal" / "runs" / f"{ts}-quickstart"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = workspace / ".capseal" / "models" / "beta_posteriors.npz"
    runtime = AgentRuntime(
        output_dir=run_dir,
        gate_posteriors=model_path if model_path.exists() else None,
    )
    runtime.record_simple(
        action_type="code_edit",
        instruction=demo_diff.description,
        inputs={"diff": demo_diff.content},
        outputs={
            "decision": result.decision,
            "p_fail": result.p_fail,
            "label": result.label,
        },
        success=result.decision != "deny",
        gate_score=result.p_fail,
        gate_decision=to_internal_decision(result.decision),
        metadata={
            "description": demo_diff.description,
            "files_affected": ["quickstart/demo.diff"],
            "label": result.label,
        },
    )
    runtime.finalize(prove=True)

    cap_path = workspace / ".capseal" / "runs" / f"{run_dir.name}.cap"
    create_run_cap_file(
        run_dir=run_dir,
        output_path=cap_path,
        run_type="mcp",
        extras={
            "session_name": run_dir.name,
            "actions_count": len(runtime.actions),
            "agent": "quickstart",
        },
    )

    runs_dir = workspace / ".capseal" / "runs"
    latest_cap = runs_dir / "latest.cap"
    if latest_cap.exists() or latest_cap.is_symlink():
        latest_cap.unlink()
    latest_cap.symlink_to(cap_path.name)

    latest = runs_dir / "latest"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_dir.name)

    verified, verify_message = verify_cap_integrity(cap_path)
    return QuickstartOutcome(cap_path=cap_path, verified=verified, verify_message=verify_message)


def run_quickstart(workspace: str = ".", color: bool = True) -> int:
    ws = Path(workspace).expanduser().resolve()
    printer = QuickstartPrinter(color=color)
    printer.header("CapSeal Quick Start")

    if not (ws / ".git").exists():
        print("Not a git repository. Run from a project root.")
        return 1

    _ensure_workspace_initialized(ws)
    printer.done("Workspace initialized")

    printer.step(
        f"Scanning git history for risk patterns "
        f"(up to {QUICKSTART_GIT_MAX_COMMITS} commits, {QUICKSTART_GIT_MAX_SECONDS:.0f}s budget)"
    )
    learned_cells = _quick_learn_from_git(
        ws,
        max_commits=QUICKSTART_GIT_MAX_COMMITS,
        max_duration_seconds=QUICKSTART_GIT_MAX_SECONDS,
        semgrep_timeout_seconds=QUICKSTART_GIT_SEMGREP_TIMEOUT_SECONDS,
    )
    if learned_cells > 0:
        printer.done(f"Learned {learned_cells} distinct risk profile cells")
    else:
        printer.note("No strong history signal found; bootstrapping demo profile")
        _bootstrap_demo_model(ws, DEMO_DIFFS[0])

    demo = DEMO_DIFFS[0]
    printer.step(f"Running gate check on demo change: {demo.description}")
    result = evaluate_risk(demo.content, workspace=ws)
    if result.decision == "approve":
        _bootstrap_demo_model(ws, demo)
        result = evaluate_risk(demo.content, workspace=ws)
    printer.gate_result(result)

    printer.step("Sealing and verifying receipt")
    outcome = _seal_demo_session(ws, result, demo)
    printer.done(f"Receipt generated: {outcome.cap_path.relative_to(ws)}")
    if outcome.verified:
        printer.done("Verification passed: chain intact")
    else:
        printer.note(f"Verification warning: {outcome.verify_message}")

    print()
    printer.next_steps()
    return 0


__all__ = ["run_quickstart"]
