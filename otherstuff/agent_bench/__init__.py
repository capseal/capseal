"""AgentEvalBench v1 - Agent Evaluation as Active Learning.

This module implements a framework for treating agent evaluation as a structured
stochastic simulation space, using Beta posteriors and acquisition-based sampling.

Key components:
- grid.py: Parameter grid generation (1024 points)
- env_toy_v1.py: ToyToolEnv with deterministic seeding
- agent_toy_v1.py: ToyAgent with explicit averaging
- runner.py: Round orchestration, acquisition, posteriors
- metrics.py: Agent-mode tube metrics

Usage:
    python -m loop_runner --mode agent_eval --agent-bench toy_v1 --run-dir artifacts/agent_test
"""

from agent_bench.grid import generate_grid
from agent_bench.env_toy_v1 import ToyToolEnv, DISTRACTOR_TOKENS
from agent_bench.agent_toy_v1 import ToyAgent
from agent_bench.runner import run_agent_eval_loop
from agent_bench.metrics import compute_agent_tube_metrics

__all__ = [
    "generate_grid",
    "ToyToolEnv",
    "ToyAgent",
    "run_agent_eval_loop",
    "compute_agent_tube_metrics",
    "DISTRACTOR_TOKENS",
]
