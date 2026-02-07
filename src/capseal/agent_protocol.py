"""AgentAction schema -- universal format for proof-carrying agent execution.

This is the standard record format that any external agent framework emits.
The CapSeal runtime handles the rest: receipt computation, trace encoding,
constraint verification, and FRI proof generation.

Action Types:
- "tool_call": Agent invokes an external tool (file read, API call, etc.)
- "code_gen": Agent generates code or text content
- "api_request": Agent makes an HTTP/API request
- "decision": Agent makes a routing or control flow decision
- "observation": Agent records an observation from the environment

Usage:
    from capseal.agent_protocol import AgentAction

    action = AgentAction(
        action_id="act_001",
        action_type="tool_call",
        instruction_hash=sha256("Read file foo.py"),
        input_hash=sha256("foo.py"),
        output_hash=sha256(file_content),
        parent_action_id=None,
        parent_receipt_hash=None,
        gate_score=0.12,
        gate_decision="pass",
        policy_verdict="FULLY_VERIFIED",
        success=True,
        duration_ms=45,
        timestamp="2024-01-15T10:30:00Z",
    )
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# Valid action types - accept common agent action categories
# Keep this permissive so agents don't need to memorize our vocabulary
ACTION_TYPES = frozenset({
    # Core types
    "tool_call",
    "code_gen",
    "api_request",
    "decision",
    "observation",
    # File operations
    "file_read",
    "file_write",
    "file_delete",
    "code_edit",
    # Shell/system
    "shell_exec",
    "command",
    # Browser/web
    "browser_action",
    "web_request",
    # Generic fallbacks
    "action",
    "unknown",
})

# Valid gate decisions
GATE_DECISIONS = frozenset({
    "pass",
    "skip",
    "human_review",
    "human_approved",  # Human reviewed and approved
    "skipped",  # Skipped after max retries
})

# Valid policy verdicts
POLICY_VERDICTS = frozenset({
    "PROOF_ONLY",
    "POLICY_ENFORCED",
    "FULLY_VERIFIED",
})


@dataclass
class AgentAction:
    """One discrete action taken by an agent.

    This is what external agent frameworks emit -- the CapSeal runtime handles
    the rest: receipt computation, trace encoding, constraint verification,
    and FRI proof generation.

    The action forms a chain via parent_action_id and parent_receipt_hash,
    creating a verifiable execution trace.
    """

    # Unique ID for this action
    action_id: str

    # Category: "tool_call", "code_gen", "api_request", "decision", "observation"
    action_type: str

    # What triggered this action -- SHA256 of the prompt/instruction
    instruction_hash: str

    # What the action did -- SHA256 of inputs (tool args, code context, API params)
    input_hash: str

    # What the action produced -- SHA256 of outputs (tool result, generated code, API response)
    output_hash: str

    # Provenance chain
    parent_action_id: str | None  # ID of the action that preceded this one (None for first action)
    parent_receipt_hash: str | None  # Receipt hash of parent action (None for first)

    # Safety metadata
    gate_score: float | None  # Committor q(x) if gate was evaluated, None if no gate
    gate_decision: str | None  # "pass", "skip", "human_review", None
    policy_verdict: str | None  # "PROOF_ONLY", "POLICY_ENFORCED", "FULLY_VERIFIED", None

    # Outcome
    success: bool  # Did the action achieve its intended result
    duration_ms: int  # Execution time in milliseconds
    timestamp: str  # ISO 8601 timestamp

    # Extensible metadata (agent-specific data: model name, temperature, token count, etc.)
    metadata: dict | None = None

    def __post_init__(self) -> None:
        """Validate action fields."""
        if self.action_type not in ACTION_TYPES:
            raise ValueError(
                f"Invalid action_type '{self.action_type}'. "
                f"Must be one of: {', '.join(sorted(ACTION_TYPES))}"
            )

        if self.gate_decision is not None and self.gate_decision not in GATE_DECISIONS:
            raise ValueError(
                f"Invalid gate_decision '{self.gate_decision}'. "
                f"Must be one of: {', '.join(sorted(GATE_DECISIONS))}"
            )

        if self.policy_verdict is not None and self.policy_verdict not in POLICY_VERDICTS:
            raise ValueError(
                f"Invalid policy_verdict '{self.policy_verdict}'. "
                f"Must be one of: {', '.join(sorted(POLICY_VERDICTS))}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert action to dictionary for serialization."""
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "instruction_hash": self.instruction_hash,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "parent_action_id": self.parent_action_id,
            "parent_receipt_hash": self.parent_receipt_hash,
            "gate_score": self.gate_score,
            "gate_decision": self.gate_decision,
            "policy_verdict": self.policy_verdict,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentAction":
        """Create action from dictionary."""
        return cls(
            action_id=data["action_id"],
            action_type=data["action_type"],
            instruction_hash=data["instruction_hash"],
            input_hash=data["input_hash"],
            output_hash=data["output_hash"],
            parent_action_id=data.get("parent_action_id"),
            parent_receipt_hash=data.get("parent_receipt_hash"),
            gate_score=data.get("gate_score"),
            gate_decision=data.get("gate_decision"),
            policy_verdict=data.get("policy_verdict"),
            success=data["success"],
            duration_ms=data["duration_ms"],
            timestamp=data["timestamp"],
            metadata=data.get("metadata"),
        )

    def canonical_json(self) -> str:
        """Return canonical JSON representation for hashing.

        This includes only the fields that contribute to the receipt hash.
        Metadata is excluded as it's extensible and agent-specific.
        """
        obj = {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "instruction_hash": self.instruction_hash,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "parent_action_id": self.parent_action_id,
            "parent_receipt_hash": self.parent_receipt_hash,
            "gate_score": self.gate_score,
            "gate_decision": self.gate_decision,
            "policy_verdict": self.policy_verdict,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
        }
        return json.dumps(obj, sort_keys=True, separators=(",", ":"))

    def compute_receipt_hash(self) -> str:
        """Compute SHA256 receipt hash for this action.

        The receipt hash is computed over the canonical fields, creating
        a verifiable fingerprint of the action.
        """
        canonical = self.canonical_json()
        return hashlib.sha256(canonical.encode()).hexdigest()


def create_action(
    action_id: str,
    action_type: str,
    instruction: str,
    inputs: Any,
    outputs: Any,
    *,
    parent: AgentAction | None = None,
    gate_score: float | None = None,
    gate_decision: str | None = None,
    policy_verdict: str | None = None,
    success: bool = True,
    duration_ms: int = 0,
    timestamp: str | None = None,
    metadata: dict | None = None,
) -> AgentAction:
    """Convenience function to create an action with automatic hashing.

    Args:
        action_id: Unique identifier for this action
        action_type: Category of action
        instruction: The instruction/prompt that triggered this action (will be hashed)
        inputs: The action's inputs (will be JSON-serialized and hashed)
        outputs: The action's outputs (will be JSON-serialized and hashed)
        parent: Previous action in the chain (optional)
        gate_score: Committor q(x) score if gate was evaluated
        gate_decision: Gate decision ("pass", "skip", "human_review")
        policy_verdict: Policy verification result
        success: Whether the action succeeded
        duration_ms: Execution time in milliseconds
        timestamp: ISO 8601 timestamp (defaults to now)
        metadata: Additional agent-specific metadata

    Returns:
        AgentAction with computed hashes
    """
    # Compute hashes
    instruction_hash = hashlib.sha256(instruction.encode()).hexdigest()

    if isinstance(inputs, str):
        input_hash = hashlib.sha256(inputs.encode()).hexdigest()
    else:
        input_hash = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    if isinstance(outputs, str):
        output_hash = hashlib.sha256(outputs.encode()).hexdigest()
    else:
        output_hash = hashlib.sha256(
            json.dumps(outputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    # Get parent chain info
    parent_action_id = parent.action_id if parent else None
    parent_receipt_hash = parent.compute_receipt_hash() if parent else None

    # Default timestamp
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    return AgentAction(
        action_id=action_id,
        action_type=action_type,
        instruction_hash=instruction_hash,
        input_hash=input_hash,
        output_hash=output_hash,
        parent_action_id=parent_action_id,
        parent_receipt_hash=parent_receipt_hash,
        gate_score=gate_score,
        gate_decision=gate_decision,
        policy_verdict=policy_verdict,
        success=success,
        duration_ms=duration_ms,
        timestamp=timestamp,
        metadata=metadata,
    )


def hash_str(s: str) -> str:
    """Compute SHA256 hash of a string and return hex digest."""
    return hashlib.sha256(s.encode()).hexdigest()


def hash_json(obj: Any) -> str:
    """Compute SHA256 hash of a JSON-serializable object."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
