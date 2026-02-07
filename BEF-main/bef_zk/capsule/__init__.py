"""BEF Capsule - cryptographic receipt primitives.

Submodules:
    header   - Capsule header construction and hashing
    payload  - Payload hash computation
    da       - Data availability challenge construction
    cli      - Command-line interface for emit/verify/inspect

Public API:
    # Low-level (record actions, generate proofs):
    from capseal import AgentRuntime, AgentAction, wrap_function

    # High-level (tandem agent loop):
    from capseal import AgentLoop, CapSealAgent, SimpleAgent
"""
from __future__ import annotations

# Low-level API: record actions and generate proofs
from bef_zk.capsule.agent_runtime import AgentRuntime
from bef_zk.capsule.agent_protocol import AgentAction, create_action
from bef_zk.capsule.adapters import wrap_function

# High-level API: tandem agent loop with risk gating
from bef_zk.capsule.agent_loop import (
    AgentLoop,
    CapSealAgent,
    ProposedAction,
    RiskFeedback,
    ActionResult,
    LoopResult,
)
from bef_zk.capsule.agents import SimpleAgent


__all__ = [
    # Low-level
    "AgentRuntime",
    "AgentAction",
    "create_action",
    "wrap_function",
    # High-level
    "AgentLoop",
    "CapSealAgent",
    "ProposedAction",
    "RiskFeedback",
    "ActionResult",
    "LoopResult",
    "SimpleAgent",
]
