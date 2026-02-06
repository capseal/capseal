"""OpenAI adapter for proof-carrying agent execution.

This adapter wraps OpenAI API calls to emit AgentActions that the CapSeal
runtime can prove.

NOTE: This is a stub implementation showing the pattern. Full OpenAI
integration requires the openai package to be installed.

Usage:
    from bef_zk.capsule.adapters.openai_adapter import OpenAIAdapter
    from bef_zk.capsule.agent_runtime import AgentRuntime

    runtime = AgentRuntime(output_dir=Path(".capseal/runs/my-run"))
    adapter = OpenAIAdapter(runtime)

    # Wrap OpenAI client
    wrapped_client = adapter.wrap_client(openai_client)

    # Or wrap function calling
    result = adapter.call_with_functions(
        messages=[...],
        functions=[...],
    )
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from bef_zk.capsule.agent_runtime import AgentRuntime

from bef_zk.capsule.agent_protocol import AgentAction


class OpenAIAdapter:
    """Adapter for wrapping OpenAI API calls.

    This adapter provides methods to wrap OpenAI client methods so that
    their execution is recorded as AgentActions in the CapSeal runtime.
    """

    def __init__(self, runtime: "AgentRuntime"):
        """Initialize the OpenAI adapter.

        Args:
            runtime: The AgentRuntime to record actions to
        """
        self.runtime = runtime

    def wrap_client(self, client: Any) -> Any:
        """Wrap an OpenAI client to record all API calls.

        Args:
            client: An OpenAI client instance

        Returns:
            Wrapped client that records actions

        Example:
            from openai import OpenAI
            client = OpenAI()
            wrapped_client = adapter.wrap_client(client)
            response = wrapped_client.chat.completions.create(...)
        """
        # This is a stub - full implementation would wrap specific methods
        # like client.chat.completions.create()
        return client

    def wrap_completion(
        self,
        create_fn: Any,
    ) -> Any:
        """Wrap a chat completion create function.

        Args:
            create_fn: The create function (e.g., client.chat.completions.create)

        Returns:
            Wrapped function that records actions
        """
        def wrapped_create(*args, **kwargs) -> Any:
            start_time = time.time()

            # Extract input info
            messages = kwargs.get('messages', [])
            model = kwargs.get('model', 'unknown')
            functions = kwargs.get('functions', [])

            # Build instruction from messages
            instruction = "OpenAI chat completion"
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, dict):
                    instruction = last_msg.get('content', instruction)[:200]

            # Execute API call
            success = True
            output = None
            error = None
            try:
                output = create_fn(*args, **kwargs)
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration_ms = int((time.time() - start_time) * 1000)

                # Extract output info
                output_data = {}
                if success and output:
                    if hasattr(output, 'choices') and output.choices:
                        choice = output.choices[0]
                        if hasattr(choice, 'message'):
                            output_data['content'] = getattr(choice.message, 'content', None)
                            output_data['function_call'] = getattr(choice.message, 'function_call', None)
                    if hasattr(output, 'usage'):
                        output_data['usage'] = {
                            'prompt_tokens': getattr(output.usage, 'prompt_tokens', 0),
                            'completion_tokens': getattr(output.usage, 'completion_tokens', 0),
                        }
                else:
                    output_data = {"error": error}

                self._record_action(
                    action_type="api_request",
                    instruction=instruction,
                    inputs={
                        "model": model,
                        "num_messages": len(messages),
                        "has_functions": len(functions) > 0,
                    },
                    outputs=output_data,
                    success=success,
                    duration_ms=duration_ms,
                    metadata={
                        "model": model,
                        "api": "chat.completions",
                    },
                )

            return output

        return wrapped_create

    def record_function_call(
        self,
        function_name: str,
        arguments: dict,
        result: Any,
        success: bool = True,
        duration_ms: int = 0,
    ) -> str:
        """Record a function call result.

        Use this when processing function_call responses from OpenAI.

        Args:
            function_name: Name of the function that was called
            arguments: Arguments passed to the function
            result: Result of the function call
            success: Whether the function succeeded
            duration_ms: Execution time in milliseconds

        Returns:
            The action's receipt hash
        """
        return self._record_action(
            action_type="tool_call",
            instruction=f"Execute function: {function_name}",
            inputs={"function": function_name, "arguments": arguments},
            outputs=result if success else {"error": str(result)},
            success=success,
            duration_ms=duration_ms,
            metadata={"function_name": function_name},
        )

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
        action_id = f"oai_{action_type}_{len(self.runtime._actions):04d}_{uuid.uuid4().hex[:8]}"
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
