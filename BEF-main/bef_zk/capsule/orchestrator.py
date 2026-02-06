"""Multi-agent orchestrator for diff review workflows.

Orchestrates tool calls with hash-chained receipts:
1. Generate diff bundle (MCP tool)
2. Review changes (LLM analysis)
3. Propose solution (LLM recommendation)

All tool invocations logged with hash-chained receipts.

Usage:
    python -m bef_zk.capsule.orchestrator --repo-b ~/BEF-main
    python -m bef_zk.capsule.orchestrator --mode direct  # Direct tool calls (fast)
    python -m bef_zk.capsule.orchestrator --mode cline   # Via Cline (slow)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from bef_zk.capsule.cli.utils import get_workspace_root

# ============================================================
# Config
# ============================================================
WORKSPACE = str(get_workspace_root())
CLINE_BIN = os.environ.get("CLINE_BIN", os.path.expanduser("~/.local/node_modules/.bin/cline"))
SESSION_ROOT = Path(WORKSPACE) / ".capseal" / "sessions"
ORCHESTRATOR_LOG = Path(WORKSPACE) / ".capseal" / "orchestrator_events.jsonl"

# Import MCP tools for direct mode
try:
    from bef_zk.capsule.mcp_server import (
        tool_diff_bundle,
        tool_verify,
        tool_doctor,
        tool_audit,
    )
    MCP_TOOLS_AVAILABLE = True
except ImportError:
    MCP_TOOLS_AVAILABLE = False


@dataclass
class AgentStep:
    """A step in the orchestration pipeline."""
    name: str
    prompt: str
    depends_on: list[str] = field(default_factory=list)
    output_file: str | None = None


@dataclass
class StepResult:
    """Result of an agent step."""
    name: str
    status: str  # "success", "failed", "skipped"
    output: str
    duration_ms: int
    event_hash: str


def _now_ms() -> int:
    return int(time.time() * 1000)


def _hash_event(event: dict) -> str:
    """Hash an event for the chain."""
    event_bytes = json.dumps(event, sort_keys=True).encode()
    return hashlib.sha256(event_bytes).hexdigest()[:32]


class Orchestrator:
    """Multi-agent orchestrator with receipt logging."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.session_dir = SESSION_ROOT / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, StepResult] = {}
        self._last_hash = ""

    def _log_event(self, event_type: str, data: dict[str, Any]) -> str:
        """Append event to hash-chained log."""
        ORCHESTRATOR_LOG.parent.mkdir(parents=True, exist_ok=True)

        event = {
            "ts_ms": _now_ms(),
            "session_id": self.session_id,
            "event_type": event_type,
            "data": data,
            "prev_hash": self._last_hash,
        }
        event_hash = _hash_event(event)
        event["event_hash"] = event_hash

        with open(ORCHESTRATOR_LOG, "a") as f:
            f.write(json.dumps(event) + "\n")

        self._last_hash = event_hash
        return event_hash

    def _run_cline(self, prompt: str, timeout: int = 300) -> tuple[bool, str]:
        """Run Cline CLI with a prompt."""
        try:
            result = subprocess.run(
                [CLINE_BIN, "--oneshot", prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=WORKSPACE,
                env={
                    **os.environ,
                    "PYTHONPATH": WORKSPACE,
                    "CAPSEAL_WORKSPACE_ROOT": WORKSPACE,
                },
            )
            output = result.stdout + result.stderr
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, "Agent timed out"
        except FileNotFoundError:
            return False, f"Cline not found at {CLINE_BIN}"
        except Exception as e:
            return False, str(e)

    def run_step(self, step: AgentStep) -> StepResult:
        """Run a single agent step."""
        # Check dependencies
        for dep in step.depends_on:
            if dep not in self.results:
                return StepResult(
                    name=step.name,
                    status="skipped",
                    output=f"Dependency '{dep}' not completed",
                    duration_ms=0,
                    event_hash="",
                )
            if self.results[dep].status != "success":
                return StepResult(
                    name=step.name,
                    status="skipped",
                    output=f"Dependency '{dep}' failed",
                    duration_ms=0,
                    event_hash="",
                )

        # Log start
        self._log_event("step_start", {"step": step.name, "prompt": step.prompt[:200]})

        # Run agent
        start = _now_ms()
        success, output = self._run_cline(step.prompt)
        duration = _now_ms() - start

        # Save output
        output_path = self.session_dir / f"{step.name}.txt"
        output_path.write_text(output)

        # Log completion
        event_hash = self._log_event("step_complete", {
            "step": step.name,
            "status": "success" if success else "failed",
            "duration_ms": duration,
            "output_path": str(output_path),
        })

        result = StepResult(
            name=step.name,
            status="success" if success else "failed",
            output=output,
            duration_ms=duration,
            event_hash=event_hash,
        )
        self.results[step.name] = result
        return result

    def run_pipeline(self, steps: list[AgentStep]) -> dict[str, StepResult]:
        """Run a full pipeline of agent steps."""
        self._log_event("pipeline_start", {
            "steps": [s.name for s in steps],
        })

        for step in steps:
            print(f"[{step.name}] Starting...", file=sys.stderr)
            result = self.run_step(step)
            print(f"[{step.name}] {result.status} ({result.duration_ms}ms)", file=sys.stderr)

            if result.status == "failed":
                print(f"[{step.name}] Failed, stopping pipeline", file=sys.stderr)
                break

        self._log_event("pipeline_complete", {
            "results": {k: v.status for k, v in self.results.items()},
        })

        # Write session summary
        summary = {
            "session_id": self.session_id,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": {
                k: {
                    "status": v.status,
                    "duration_ms": v.duration_ms,
                    "event_hash": v.event_hash,
                }
                for k, v in self.results.items()
            },
        }
        (self.session_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        return self.results


def create_diff_review_pipeline(repo_a: str, repo_b: str, base_ref: str = "HEAD") -> list[AgentStep]:
    """Create a diff review pipeline."""
    return [
        AgentStep(
            name="diff_bundle",
            prompt=f"""Use the diff_bundle MCP tool to generate a diff between repos.
Call diff_bundle with:
- repo_path: "{repo_b}"
- base_ref: "{base_ref}~10"
- head_ref: "{base_ref}"

Save the result and summarize what files changed.""",
        ),
        AgentStep(
            name="diff_review",
            prompt=f"""Review the diff output from the previous step.

For each changed file:
1. Identify the nature of the change (new feature, bug fix, refactor, etc.)
2. Note any potential issues or concerns
3. Rate the change quality (good/needs-work/problematic)

Be specific and cite line numbers where relevant.
Output a structured review in markdown format.""",
            depends_on=["diff_bundle"],
        ),
        AgentStep(
            name="solution_proposal",
            prompt="""Based on the diff review, propose a solution:

1. If changes look good: Recommend merging with any minor suggestions
2. If changes need work: List specific fixes required
3. If changes are problematic: Explain why and suggest alternatives

Use the capseal doctor and verify tools to validate any capsules mentioned.

Output a final recommendation with clear next steps.""",
            depends_on=["diff_review"],
        ),
    ]


class DirectToolRunner:
    """Run MCP tools directly with receipt generation."""

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.context: dict[str, Any] = {}

    def run_diff_bundle(self, repo_path: str, base_ref: str, head_ref: str) -> dict:
        """Run diff_bundle and log receipt."""
        if not MCP_TOOLS_AVAILABLE:
            raise ImportError("MCP tools not available")

        result = tool_diff_bundle(repo_path, base_ref, head_ref)
        self.context["diff_bundle"] = result
        return result

    def run_verify(self, capsule_path: str) -> dict:
        """Run verify and log receipt."""
        if not MCP_TOOLS_AVAILABLE:
            raise ImportError("MCP tools not available")

        result = tool_verify(capsule_path)
        self.context["verify"] = result
        return result

    def run_doctor(self, capsule_path: str, sample_rows: int = 1) -> dict:
        """Run doctor and log receipt."""
        if not MCP_TOOLS_AVAILABLE:
            raise ImportError("MCP tools not available")

        result = tool_doctor(capsule_path, sample_rows=sample_rows)
        self.context["doctor"] = result
        return result

    def run_audit(self, capsule_path: str) -> dict:
        """Run audit and log receipt."""
        if not MCP_TOOLS_AVAILABLE:
            raise ImportError("MCP tools not available")

        result = tool_audit(capsule_path)
        self.context["audit"] = result
        return result


def run_direct_demo(repo_path: str, base_ref: str = "HEAD", capsule_path: str | None = None):
    """Run a direct demo showing receipts being generated."""
    from bef_zk.capsule.mcp_server import EVENT_LOG_PATH
    import bef_zk.capsule.mcp_server as mcp_mod

    print("=" * 70)
    print("  Direct Tool Demo - Hash-Chained Receipt Generation")
    print("=" * 70)
    print()

    # Clear event log for clean demo
    if os.path.exists(EVENT_LOG_PATH):
        os.remove(EVENT_LOG_PATH)
    mcp_mod._last_hash = None
    print(f"[Setup] Cleared receipt log: {EVENT_LOG_PATH}")
    print()

    orchestrator = Orchestrator()
    runner = DirectToolRunner(orchestrator)

    # Step 1: diff_bundle
    print("─" * 70)
    print("[Step 1] Calling diff_bundle...")
    print(f"  repo_path: {repo_path}")
    print(f"  base_ref:  {base_ref}~5")
    print(f"  head_ref:  {base_ref}")
    result = runner.run_diff_bundle(repo_path, f"{base_ref}~5", base_ref)
    print(f"  → {result.get('file_count', '?')} files changed")
    _show_last_receipt(EVENT_LOG_PATH)
    print()

    # Step 2: verify (if capsule provided)
    if capsule_path and os.path.exists(capsule_path):
        print("─" * 70)
        print("[Step 2] Calling verify...")
        print(f"  capsule_path: {capsule_path}")
        result = runner.run_verify(capsule_path)
        status = "✓ OK" if result.get("ok") else "✗ FAIL"
        print(f"  → {status}")
        _show_last_receipt(EVENT_LOG_PATH)
        print()

        # Step 3: doctor
        print("─" * 70)
        print("[Step 3] Calling doctor...")
        result = runner.run_doctor(capsule_path, sample_rows=0)
        status = "✓ OK" if result.get("ok") else "✗ FAIL"
        print(f"  → {status}")
        _show_last_receipt(EVENT_LOG_PATH)
        print()

        # Step 4: audit
        print("─" * 70)
        print("[Step 4] Calling audit...")
        result = runner.run_audit(capsule_path)
        status = "✓ OK" if result.get("ok") else "✗ FAIL"
        print(f"  → {status}")
        _show_last_receipt(EVENT_LOG_PATH)
        print()

    # Verify chain
    print("=" * 70)
    print("[Chain Verification]")
    events = _load_events(EVENT_LOG_PATH)
    valid, msg = _verify_chain(events)
    print(f"  {len(events)} receipts generated")
    print(f"  Chain status: {'✓ VALID' if valid else '✗ BROKEN'} - {msg}")
    print("=" * 70)
    print()
    print(f"Receipt log: {EVENT_LOG_PATH}")
    print(f"View with: scripts/logs mcp -j")


def _show_last_receipt(log_path: str):
    """Show the last receipt in abbreviated form."""
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
            if lines:
                event = json.loads(lines[-1])
                prev = event.get("prev_hash", "")[:12] or "(genesis)"
                curr = event.get("event_hash", "")[:12]
                print(f"  [Receipt] prev={prev}... → hash={curr}...")
    except (FileNotFoundError, json.JSONDecodeError):
        pass


def _load_events(log_path: str) -> list[dict]:
    """Load all events from log."""
    events = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    except FileNotFoundError:
        pass
    return events


def _verify_chain(events: list[dict]) -> tuple[bool, str]:
    """Verify hash chain integrity."""
    prev_hash = ""
    for i, event in enumerate(events):
        if event.get("prev_hash", "") != prev_hash:
            return False, f"prev_hash mismatch at #{i}"
        event_copy = {k: v for k, v in event.items() if k != "event_hash"}
        computed = hashlib.sha256(
            json.dumps(event_copy, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:32]
        if computed != event.get("event_hash"):
            return False, f"hash mismatch at #{i}"
        prev_hash = computed
    return True, "intact"


def main():
    import argparse
    from bef_zk.capsule.cli.utils import get_workspace_root

    workspace = str(get_workspace_root())
    parser = argparse.ArgumentParser(description="Multi-agent diff review orchestrator")
    parser.add_argument("--repo-a", default=None, help="First repo (required for cross-repo diff)")
    parser.add_argument("--repo-b", default=workspace, help="Second repo (default: workspace root)")
    parser.add_argument("--base-ref", default="HEAD", help="Base git ref")
    parser.add_argument("--session-id", help="Session ID (auto-generated if not provided)")
    parser.add_argument("--dry-run", action="store_true", help="Show pipeline without running")
    parser.add_argument("--mode", choices=["cline", "direct"], default="direct",
                        help="Execution mode: 'direct' for fast tool calls, 'cline' for agent")
    parser.add_argument("--capsule", help="Capsule path for verify/doctor/audit steps")
    args = parser.parse_args()

    # Direct mode - fast demo with receipt generation
    if args.mode == "direct":
        capsule = args.capsule
        if not capsule:
            # Try to find a capsule
            candidates = [
                f"{args.repo_b}/fixtures/golden_run_latest/capsule/strategy_capsule.json",
                f"{args.repo_b}/fixtures/golden_run/capsule/capsule.json",
            ]
            for c in candidates:
                if os.path.exists(c):
                    capsule = c
                    break
        run_direct_demo(args.repo_b, args.base_ref, capsule)
        return

    # Cline mode - original behavior
    pipeline = create_diff_review_pipeline(args.repo_a, args.repo_b, args.base_ref)

    if args.dry_run:
        print("Pipeline steps:")
        for i, step in enumerate(pipeline, 1):
            deps = f" (depends: {', '.join(step.depends_on)})" if step.depends_on else ""
            print(f"  {i}. {step.name}{deps}")
        return

    print(f"Starting diff review pipeline...", file=sys.stderr)
    print(f"  Repo A: {args.repo_a}", file=sys.stderr)
    print(f"  Repo B: {args.repo_b}", file=sys.stderr)
    print(f"  Base ref: {args.base_ref}", file=sys.stderr)
    print("", file=sys.stderr)

    orchestrator = Orchestrator(session_id=args.session_id)
    results = orchestrator.run_pipeline(pipeline)

    print("", file=sys.stderr)
    print(f"Session: {orchestrator.session_id}", file=sys.stderr)
    print(f"Results: {orchestrator.session_dir}", file=sys.stderr)

    # Output final summary
    print(json.dumps({
        "session_id": orchestrator.session_id,
        "session_dir": str(orchestrator.session_dir),
        "results": {k: v.status for k, v in results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
