"""LangChain adapter for proof-carrying agent execution.

This adapter wraps LangChain agents and tools to emit AgentActions that
the CapSeal runtime can prove.

NOTE: This is a stub implementation showing the pattern. Full LangChain
integration requires the langchain package to be installed.

Usage:
    from capseal.adapters.langchain_adapter import LangChainAdapter
    from capseal.agent_runtime import AgentRuntime

    runtime = AgentRuntime(output_dir=Path(".capseal/runs/my-run"))
    adapter = LangChainAdapter(runtime)

    # Wrap a LangChain tool
    wrapped_tool = adapter.wrap_tool(my_langchain_tool)

    # Or wrap an entire agent
    wrapped_agent = adapter.wrap_agent(my_langchain_agent)
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from capseal.agent_runtime import AgentRuntime

from capseal.agent_protocol import AgentAction


class LangChainAdapter:
    """Adapter for wrapping LangChain agents and tools.

    This adapter provides methods to wrap LangChain components so that
    their execution is recorded as AgentActions in the CapSeal runtime.
    """

    def __init__(self, runtime: "AgentRuntime"):
        """Initialize the LangChain adapter.

        Args:
            runtime: The AgentRuntime to record actions to
        """
        self.runtime = runtime

    def wrap_tool(self, tool: Any) -> Any:
        """Wrap a LangChain tool to record actions.

        Args:
            tool: A LangChain tool (BaseTool or similar)

        Returns:
            Wrapped tool that records actions

        Example:
            from langchain.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            wrapped_search = adapter.wrap_tool(search)
        """
        original_run = tool.run if hasattr(tool, 'run') else tool

        def wrapped_run(*args, **kwargs) -> Any:
            start_time = time.time()

            # Get tool info
            tool_name = getattr(tool, 'name', str(tool))
            tool_description = getattr(tool, 'description', f"Run {tool_name}")

            # Capture inputs
            inputs = {"args": args, "kwargs": kwargs}

            # Execute tool
            success = True
            output = None
            error = None
            try:
                if hasattr(tool, 'run'):
                    output = tool.run(*args, **kwargs)
                else:
                    output = tool(*args, **kwargs)
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration_ms = int((time.time() - start_time) * 1000)
                self._record_action(
                    action_type="tool_call",
                    instruction=tool_description,
                    inputs=inputs,
                    outputs=output if success else {"error": error},
                    success=success,
                    duration_ms=duration_ms,
                    metadata={"tool_name": tool_name},
                )

            return output

        # Preserve tool attributes
        if hasattr(tool, 'run'):
            tool.run = wrapped_run
            return tool
        return wrapped_run

    def wrap_agent(self, agent: Any) -> Any:
        """Wrap a LangChain agent to record all actions.

        This wraps the agent's invoke/run method to record the overall
        agent execution as an action.

        Args:
            agent: A LangChain agent (AgentExecutor or similar)

        Returns:
            Wrapped agent that records actions

        Note: Individual tool calls within the agent are NOT recorded
        unless those tools are also wrapped with wrap_tool().
        """
        original_invoke = getattr(agent, 'invoke', None) or getattr(agent, 'run', None)

        if original_invoke is None:
            raise ValueError("Agent must have 'invoke' or 'run' method")

        def wrapped_invoke(input_data: Any, **kwargs) -> Any:
            start_time = time.time()

            # Execute agent
            success = True
            output = None
            error = None
            try:
                output = original_invoke(input_data, **kwargs)
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration_ms = int((time.time() - start_time) * 1000)
                self._record_action(
                    action_type="decision",
                    instruction="Execute LangChain agent",
                    inputs={"input": input_data, "kwargs": kwargs},
                    outputs=output if success else {"error": error},
                    success=success,
                    duration_ms=duration_ms,
                    metadata={"agent_type": type(agent).__name__},
                )

            return output

        if hasattr(agent, 'invoke'):
            agent.invoke = wrapped_invoke
        else:
            agent.run = wrapped_invoke

        return agent

    def _record_action(
        self,
        action_type: str,
        instruction: str,
        inputs: Any,
        outputs: Any,
        success: bool,
        duration_ms: int,
        metadata: dict | None = None,
    ) -> str:
        """Record an action to the runtime.

        Returns the action's receipt hash.
        """
        # Compute hashes
        instruction_hash = hashlib.sha256(instruction.encode()).hexdigest()

        try:
            input_json = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            input_json = json.dumps({"raw": str(inputs)})
        input_hash = hashlib.sha256(input_json.encode()).hexdigest()

        try:
            output_json = json.dumps(outputs, sort_keys=True, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            output_json = json.dumps({"raw": str(outputs)})
        output_hash = hashlib.sha256(output_json.encode()).hexdigest()

        # Get parent info
        parent = self.runtime.last_action
        parent_action_id = parent.action_id if parent else None
        parent_receipt_hash = parent.compute_receipt_hash() if parent else None

        # Create action
        action_id = f"lc_{action_type}_{len(self.runtime._actions):04d}_{uuid.uuid4().hex[:8]}"
        action = AgentAction(
            action_id=action_id,
            action_type=action_type,
            instruction_hash=instruction_hash,
            input_hash=input_hash,
            output_hash=output_hash,
            parent_action_id=parent_action_id,
            parent_receipt_hash=parent_receipt_hash,
            gate_score=None,
            gate_decision=None,
            policy_verdict=None,
            success=success,
            duration_ms=duration_ms,
            timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            metadata=metadata,
        )

        # Record action
        old_auto_chain = self.runtime.auto_chain
        self.runtime.auto_chain = False
        try:
            return self.runtime.record(action)
        finally:
            self.runtime.auto_chain = old_auto_chain
