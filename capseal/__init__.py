"""CapSeal - Proof-carrying execution for AI agents.

This module re-exports the public API from capseal_cli for cleaner imports.

Usage:
    from capseal import AgentRuntime, AgentAction

    with AgentRuntime(output_dir=Path("my_run")) as runtime:
        runtime.record_simple(action_type="tool_call", instruction="...",
                              inputs={...}, outputs={...}, success=True)
    # Proof-carrying capsule generated automatically
"""

# Re-export everything from capseal_cli
from capseal_cli import *
from capseal_cli import __version__, __all__
