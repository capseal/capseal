"""AgentRuntime -- wraps any agent execution with proof-carrying verification.

This is what external agent frameworks interact with. It provides a simple
interface for recording actions and generating proofs.

Usage:
    runtime = AgentRuntime(output_dir=Path(".capseal/runs/my-agent-run"))
    runtime.record(AgentAction(...))
    runtime.record(AgentAction(...))
    capsule = runtime.finalize(prove=True)

Can also be used as a context manager:
    with AgentRuntime(output_dir=...) as runtime:
        runtime.record(action1)
    # finalize() called automatically
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from capseal.agent_protocol import AgentAction, create_action
from capseal.agent_adapter import AgentAdapter, verify_agent_capsule


@dataclass
class GateResult:
    """Result from evaluating the committor gate."""
    q: float
    uncertainty: float
    decision: str  # "pass", "skip", "human_review"
    grid_idx: int = 0
    reason: str = ""


class AgentRuntime:
    """Wraps any agent execution with proof-carrying verification.

    This runtime:
    1. Records actions as they happen (append-only log)
    2. Automatically chains actions via parent_receipt_hash
    3. Optionally evaluates the committor gate
    4. Generates FRI proof on finalize()

    Usage:
        runtime = AgentRuntime(output_dir=Path(".capseal/runs/my-agent-run"))
        runtime.record(AgentAction(...))
        runtime.record(AgentAction(...))
        capsule = runtime.finalize(prove=True)

    Can also be used as a context manager:
        with AgentRuntime(output_dir=...) as runtime:
            runtime.record(action1)
        # finalize() called automatically
    """

    def __init__(
        self,
        output_dir: Path,
        gate_posteriors: Path | None = None,
        auto_chain: bool = True,
    ):
        """Initialize the agent runtime.

        Args:
            output_dir: Directory to store actions, proofs, and capsule
            gate_posteriors: Path to beta_posteriors.npz for committor gate (optional)
            auto_chain: If True, automatically set parent chain on recorded actions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gate_posteriors = Path(gate_posteriors) if gate_posteriors else None
        self.auto_chain = auto_chain

        self._actions: list[AgentAction] = []
        self._action_log_path = self.output_dir / "actions.jsonl"
        self._start_time = time.time()
        self._run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self._finalized = False

        # Initialize action log file
        self._action_log_path.touch()

        # Load existing actions if resuming
        self._load_existing_actions()

    def _load_existing_actions(self) -> None:
        """Load existing actions from the log file if resuming."""
        if not self._action_log_path.exists():
            return

        try:
            content = self._action_log_path.read_text().strip()
            if not content:
                return

            for line in content.split("\n"):
                if line.strip():
                    data = json.loads(line)
                    action = AgentAction.from_dict(data)
                    self._actions.append(action)
        except (json.JSONDecodeError, KeyError):
            # Log file is corrupt, start fresh
            self._actions = []

    @property
    def run_id(self) -> str:
        """Get the run ID for this runtime instance."""
        return self._run_id

    @property
    def action_count(self) -> int:
        """Get the number of recorded actions."""
        return len(self._actions)

    @property
    def last_action(self) -> AgentAction | None:
        """Get the last recorded action, or None if no actions recorded."""
        return self._actions[-1] if self._actions else None

    @property
    def last_receipt_hash(self) -> str | None:
        """Get the receipt hash of the last action, or None if no actions."""
        if not self._actions:
            return None
        return self._actions[-1].compute_receipt_hash()

    @property
    def actions(self) -> list[AgentAction]:
        """Get all recorded actions."""
        return self._actions.copy()

    @property
    def last_capsule(self) -> dict | None:
        """Get the last generated capsule, or None if not finalized."""
        capsule_path = self.output_dir / "agent_capsule.json"
        if capsule_path.exists():
            return json.loads(capsule_path.read_text())
        return None

    def record(self, action: AgentAction) -> str:
        """Record an action. Returns the action's receipt_hash.

        If auto_chain is True and the action doesn't have parent chain set,
        this will automatically set parent_action_id and parent_receipt_hash
        based on the previous action.

        Args:
            action: The AgentAction to record

        Returns:
            The action's receipt hash

        Raises:
            RuntimeError: If the runtime has already been finalized
        """
        if self._finalized:
            raise RuntimeError("Cannot record actions after finalize() has been called")

        # Auto-chain if enabled and parent not set
        if self.auto_chain and action.parent_action_id is None and self._actions:
            last = self._actions[-1]
            # Create a new action with parent chain set
            action = AgentAction(
                action_id=action.action_id,
                action_type=action.action_type,
                instruction_hash=action.instruction_hash,
                input_hash=action.input_hash,
                output_hash=action.output_hash,
                parent_action_id=last.action_id,
                parent_receipt_hash=last.compute_receipt_hash(),
                gate_score=action.gate_score,
                gate_decision=action.gate_decision,
                policy_verdict=action.policy_verdict,
                success=action.success,
                duration_ms=action.duration_ms,
                timestamp=action.timestamp,
                metadata=action.metadata,
            )

        # Append to in-memory list
        self._actions.append(action)

        # Append to log file (append-only for durability)
        with open(self._action_log_path, "a") as f:
            f.write(json.dumps(action.to_dict()) + "\n")

        return action.compute_receipt_hash()

    def record_simple(
        self,
        action_type: str,
        instruction: str,
        inputs: Any,
        outputs: Any,
        *,
        success: bool = True,
        duration_ms: int = 0,
        gate_score: float | None = None,
        gate_decision: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Convenience method to record an action with automatic hashing.

        Args:
            action_type: Category of action
            instruction: The instruction/prompt that triggered this action
            inputs: The action's inputs (will be JSON-serialized and hashed)
            outputs: The action's outputs (will be JSON-serialized and hashed)
            success: Whether the action succeeded
            duration_ms: Execution time in milliseconds
            gate_score: Committor q(x) score if gate was evaluated
            gate_decision: Gate decision ("pass", "skip", "human_review", "human_approved")
            metadata: Additional agent-specific metadata

        Returns:
            The action's receipt hash
        """
        action_id = f"act_{len(self._actions):04d}_{uuid.uuid4().hex[:8]}"

        parent = self._actions[-1] if self._actions else None

        action = create_action(
            action_id=action_id,
            action_type=action_type,
            instruction=instruction,
            inputs=inputs,
            outputs=outputs,
            parent=parent,
            success=success,
            duration_ms=duration_ms,
            gate_score=gate_score,
            gate_decision=gate_decision,
            metadata=metadata,
        )

        return self.record(action)

    def gate(self, diff_text: str = "", findings: list | None = None) -> dict[str, Any]:
        """Evaluate the committor gate. Returns {q, uncertainty, decision}.

        This uses the learned failure probabilities from the agent bench
        to assess the risk of the current action sequence.

        Args:
            diff_text: Optional diff text to extract features from
            findings: Optional list of static analysis findings

        Returns:
            Dict with q, uncertainty, decision, grid_idx, reason
        """
        if self.gate_posteriors is None or not self.gate_posteriors.exists():
            return {
                "q": 0.0,
                "uncertainty": 1.0,
                "decision": "pass",
                "grid_idx": 0,
                "reason": "no posteriors available",
            }

        try:
            from capseal.shared.features import (
                extract_patch_features,
                discretize_features,
                features_to_grid_idx,
                SKIP_THRESHOLD,
                HUMAN_REVIEW_UNCERTAINTY,
            )
            from capseal.shared.scoring import lookup_posterior_at_idx
            import numpy as np

            # Load posteriors
            data = np.load(self.gate_posteriors, allow_pickle=True)
            alpha = data["alpha"]
            beta = data["beta"]

            # Extract features from diff
            if diff_text:
                features = extract_patch_features(diff_text, findings or [])
                discrete = discretize_features(features)
                grid_idx = features_to_grid_idx(discrete)
            else:
                grid_idx = 0

            # Look up posterior
            posterior = lookup_posterior_at_idx(alpha, beta, grid_idx)
            q = posterior["q"]
            uncertainty = posterior["uncertainty"]

            # Make decision
            if q >= SKIP_THRESHOLD:
                decision = "skip"
                reason = f"q={q:.3f} >= skip_threshold={SKIP_THRESHOLD}"
            elif uncertainty > HUMAN_REVIEW_UNCERTAINTY:
                decision = "human_review"
                reason = f"uncertainty={uncertainty:.3f} > review_threshold={HUMAN_REVIEW_UNCERTAINTY}"
            else:
                decision = "pass"
                reason = f"q={q:.3f}, uncertainty={uncertainty:.3f}"

            return {
                "q": float(q),
                "uncertainty": float(uncertainty),
                "decision": decision,
                "grid_idx": int(grid_idx),
                "reason": reason,
            }

        except Exception as e:
            return {
                "q": 0.0,
                "uncertainty": 1.0,
                "decision": "pass",
                "grid_idx": 0,
                "reason": f"gate evaluation failed: {e}",
            }

    def finalize(self, prove: bool = True) -> dict[str, Any]:
        """Generate receipt chain and optionally FRI proof.

        Args:
            prove: If True, generate FRI proof. If False, just save actions.

        Returns:
            Dict with capsule data if prove=True, otherwise just action summary
        """
        if self._finalized:
            raise RuntimeError("finalize() has already been called")

        self._finalized = True

        if not self._actions:
            return {
                "status": "empty",
                "message": "No actions recorded",
                "run_id": self._run_id,
            }

        # Save run metadata
        run_meta = {
            "run_id": self._run_id,
            "start_time": self._start_time,
            "end_time": time.time(),
            "num_actions": len(self._actions),
            "final_receipt_hash": self._actions[-1].compute_receipt_hash(),
            "proved": prove,
        }
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(run_meta, indent=2)
        )

        if not prove:
            return {
                "status": "recorded",
                "run_id": self._run_id,
                "num_actions": len(self._actions),
                "final_receipt_hash": self._actions[-1].compute_receipt_hash(),
                "output_dir": str(self.output_dir),
            }

        # Generate proof using AgentAdapter
        adapter = AgentAdapter()
        result = adapter.prove_actions(self._actions, self.output_dir)

        return {
            "status": "proved",
            "run_id": self._run_id,
            **result,
        }

    def verify(self) -> tuple[bool, dict[str, Any]]:
        """Verify the capsule generated by this runtime.

        Returns:
            Tuple of (valid, details)
        """
        capsule_path = self.output_dir / "agent_capsule.json"
        if not capsule_path.exists():
            return False, {"error": "Capsule not found. Call finalize(prove=True) first."}

        return verify_agent_capsule(capsule_path)

    def get_action_chain(self) -> list[dict[str, Any]]:
        """Get the full action chain with receipt hashes.

        Returns:
            List of action summaries with receipt hashes
        """
        chain = []
        for i, action in enumerate(self._actions):
            chain.append({
                "index": i,
                "action_id": action.action_id,
                "action_type": action.action_type,
                "success": action.success,
                "receipt_hash": action.compute_receipt_hash()[:16] + "...",
                "parent_receipt": action.parent_receipt_hash[:16] + "..." if action.parent_receipt_hash else None,
                "gate_decision": action.gate_decision,
            })
        return chain

    def __enter__(self) -> "AgentRuntime":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, calling finalize()."""
        if not self._finalized:
            try:
                self.finalize(prove=True)
            except Exception:
                # Don't raise during cleanup if there was already an exception
                if exc_type is None:
                    raise


def inspect_agent_run(run_dir: Path) -> dict[str, Any]:
    """Inspect an agent run directory.

    Args:
        run_dir: Directory containing the agent run

    Returns:
        Dict with run details, action chain, gate scores, receipt hashes
    """
    run_dir = Path(run_dir)

    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
    }

    if not run_dir.exists():
        return result

    # Load run metadata
    meta_path = run_dir / "run_metadata.json"
    if meta_path.exists():
        result["metadata"] = json.loads(meta_path.read_text())

    # Load actions
    actions_path = run_dir / "actions.jsonl"
    if actions_path.exists():
        actions = []
        content = actions_path.read_text().strip()
        for line in content.split("\n"):
            if line.strip():
                data = json.loads(line)
                actions.append(AgentAction.from_dict(data))

        result["num_actions"] = len(actions)
        result["action_chain"] = []

        for i, action in enumerate(actions):
            result["action_chain"].append({
                "index": i,
                "action_id": action.action_id,
                "action_type": action.action_type,
                "success": action.success,
                "receipt_hash": action.compute_receipt_hash()[:16] + "...",
                "gate_score": action.gate_score,
                "gate_decision": action.gate_decision,
                "policy_verdict": action.policy_verdict,
            })

    # Load capsule
    capsule_path = run_dir / "agent_capsule.json"
    if capsule_path.exists():
        capsule = json.loads(capsule_path.read_text())
        result["capsule"] = {
            "schema": capsule.get("schema"),
            "verified": capsule.get("verification", {}).get("constraints_valid"),
            "final_receipt": capsule.get("statement", {}).get("public_inputs", {}).get("final_receipt_hash", "")[:32] + "...",
        }

        # Verify capsule
        valid, details = verify_agent_capsule(capsule_path)
        result["capsule"]["verification_valid"] = valid
        if not valid:
            result["capsule"]["verification_error"] = details.get("error")

    return result
