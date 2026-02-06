"""Framework adapters for proof-carrying agent execution.

This module provides adapters that wrap existing agent frameworks (LangChain,
OpenAI, etc.) to emit AgentActions that the CapSeal runtime can prove.

Available adapters:
- wrap_function: Decorator to wrap any function call as a proof-carrying agent action
- langchain_adapter: Adapter for LangChain agents (stub)
- openai_adapter: Adapter for OpenAI function calling (stub)

Usage:
    from bef_zk.capsule.adapters import wrap_function
    from bef_zk.capsule.agent_runtime import AgentRuntime

    runtime = AgentRuntime(output_dir=Path(".capseal/runs/my-run"))

    @wrap_function(runtime, action_type="tool_call")
    def read_file(path: str) -> str:
        return Path(path).read_text()

    content = read_file("foo.py")  # Action is recorded automatically
"""
from __future__ import annotations

from bef_zk.capsule.adapters.generic_adapter import wrap_function

# Lazy imports to avoid loading heavy dependencies
def get_langchain_adapter():
    """Get the LangChain adapter (lazy import)."""
    from bef_zk.capsule.adapters.langchain_adapter import LangChainAdapter
    return LangChainAdapter

def get_openai_adapter():
    """Get the OpenAI adapter (lazy import)."""
    from bef_zk.capsule.adapters.openai_adapter import OpenAIAdapter
    return OpenAIAdapter


__all__ = [
    "wrap_function",
    "get_langchain_adapter",
    "get_openai_adapter",
]
