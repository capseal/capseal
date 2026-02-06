"""CapSeal - Proof-carrying execution for AI agents.

CapSeal makes any AI agent's execution cryptographically provable. An agent
does things — calls tools, writes code, makes decisions. CapSeal records each
action, chains them together, encodes the chain as an algebraic trace over a
Goldilocks field, and generates a FRI proof that the entire execution sequence
is internally consistent and untampered.

Quick Start:
    from capseal import AgentRuntime, AgentAction

    with AgentRuntime(output_dir=Path("my_run")) as runtime:
        runtime.record_simple(action_type="tool_call", instruction="...",
                              inputs={...}, outputs={...}, success=True)
    # Proof-carrying capsule generated automatically
"""

__version__ = "0.1.0"

# Re-export key components for convenience
try:
    # === PHASE 3: Agent Protocol (Main Public API) ===
    from bef_zk.capsule.agent_protocol import AgentAction, create_action
    from bef_zk.capsule.agent_runtime import AgentRuntime
    from bef_zk.capsule.agent_adapter import AgentAdapter, verify_agent_capsule
    from bef_zk.capsule.adapters import wrap_function

    # === PHASE 1 & 2: Workflow and Eval Adapters ===
    from bef_zk.capsule.workflow_adapter import WorkflowAdapter
    from bef_zk.capsule.eval_adapter import EvalAdapter

    # === Scoring and Gating ===
    from bef_zk.shared.scoring import (
        compute_acquisition_score,
        select_targets,
        compute_tube_metrics,
    )
    from bef_zk.shared.features import (
        extract_patch_features,
        discretize_features,
        features_to_grid_idx,
        score_patch,
    )
    from bef_zk.shared.receipts import (
        build_round_receipt,
        build_run_receipt,
        verify_round_receipt,
        verify_run_receipt,
    )
except ImportError:
    # bef_zk not yet on path - will be set up by main.py
    pass

__all__ = [
    # Version
    "__version__",
    # Phase 3: Agent Protocol (Main Public API)
    "AgentAction",
    "AgentRuntime",
    "AgentAdapter",
    "create_action",
    "verify_agent_capsule",
    "wrap_function",
    # Phase 1 & 2: Workflow and Eval
    "WorkflowAdapter",
    "EvalAdapter",
    # Scoring and Gating
    "compute_acquisition_score",
    "select_targets",
    "compute_tube_metrics",
    "extract_patch_features",
    "discretize_features",
    "features_to_grid_idx",
    "score_patch",
    # Receipts
    "build_round_receipt",
    "build_run_receipt",
    "verify_round_receipt",
    "verify_run_receipt",
]
