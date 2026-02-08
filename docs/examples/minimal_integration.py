"""Minimal CapSeal integration example.

Shows how to use AgentRuntime directly to gate, record, and seal
agent actions with cryptographic proofs.

Usage:
    python docs/examples/minimal_integration.py

Prerequisites:
    pip install capseal
    capseal init --path /tmp/capseal-example --no-tui
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from capseal.agent_runtime import AgentRuntime


def main() -> None:
    # Create a temporary workspace
    workspace = Path(tempfile.mkdtemp(prefix="capseal-example-"))
    capseal_dir = workspace / ".capseal"
    capseal_dir.mkdir(parents=True)
    runs_dir = capseal_dir / "runs"
    run_dir = runs_dir / "example-run"
    run_dir.mkdir(parents=True)

    # Initialize the runtime
    runtime = AgentRuntime(output_dir=run_dir)

    # Step 1: Gate a proposed action
    gate_result = runtime.gate(
        diff_text="--- a/app.py\n+++ b/app.py\n-os.system(cmd)\n+subprocess.run(cmd.split())",
        findings=[{"path": "app.py", "extra": {"severity": "warning"}}],
    )
    print(f"Gate decision: {gate_result['decision']} (p_fail={gate_result['q']:.2f})")

    # Step 2: Record the action (if approved)
    if gate_result["decision"] != "skip":
        receipt_hash = runtime.record_simple(
            action_type="code_edit",
            instruction=hashlib.sha256(b"Fix command injection in app.py").hexdigest(),
            inputs={"tool_name": "edit", "files_affected": ["app.py"]},
            outputs={"output": "Replaced os.system with subprocess.run"},
            success=True,
            duration_ms=150,
            gate_score=gate_result.get("q"),
            gate_decision=gate_result.get("decision"),
            metadata={
                "description": "Fix command injection vulnerability",
                "files_affected": ["app.py"],
            },
        )
        print(f"Recorded action, receipt: {receipt_hash[:16]}...")

    # Step 3: Seal the session (generates proof)
    capsule = runtime.finalize(prove=True)
    print(f"Session sealed: {capsule.get('status', 'unknown')}")
    print(f"Proof type: {capsule.get('proof_type', 'unknown')}")
    print(f"Actions: {len(runtime.actions)}")

    # The run directory now contains:
    # - actions.jsonl (action log with receipt hashes)
    # - agent_capsule.json (proof capsule)
    capsule_file = run_dir / "agent_capsule.json"
    if capsule_file.exists():
        print(f"Capsule written to: {capsule_file}")

    print(f"\nWorkspace: {workspace}")
    print("Verify with: capseal verify <path-to-cap-file>")


if __name__ == "__main__":
    main()
