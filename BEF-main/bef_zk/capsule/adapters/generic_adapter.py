"""Generic adapter for wrapping arbitrary functions as proof-carrying agent actions.

This is the most flexible adapter -- it works with any Python function and
doesn't require any specific agent framework.

Usage:
    from bef_zk.capsule.adapters import wrap_function
    from bef_zk.capsule.agent_runtime import AgentRuntime

    runtime = AgentRuntime(output_dir=Path(".capseal/runs/my-run"))

    @wrap_function(runtime, action_type="tool_call")
    def read_file(path: str) -> str:
        return Path(path).read_text()

    content = read_file("foo.py")  # Action is recorded automatically

Can also be used without decorator syntax:

    def my_function(x, y):
        return x + y

    wrapped = wrap_function(runtime, action_type="tool_call")(my_function)
    result = wrapped(1, 2)
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from bef_zk.capsule.agent_runtime import AgentRuntime

from bef_zk.capsule.agent_protocol import AgentAction


def wrap_function(
    runtime: "AgentRuntime",
    action_type: str = "function_call",
    instruction: str | None = None,
) -> Callable:
    """Decorator that wraps any function call as a proof-carrying agent action.

    This decorator:
    1. Captures the function name and arguments as input
    2. Executes the function
    3. Captures the output
    4. Records an AgentAction with the runtime

    Args:
        runtime: The AgentRuntime to record actions to
        action_type: The type of action (default: "function_call")
        instruction: Optional instruction text (defaults to function docstring or signature)

    Returns:
        Decorator function

    Example:
        @wrap_function(runtime, action_type="tool_call")
        def read_file(path: str) -> str:
            '''Read a file from disk.'''
            return Path(path).read_text()

        content = read_file("foo.py")  # Action recorded automatically
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()

            # Build instruction from function info
            fn_instruction = instruction
            if fn_instruction is None:
                fn_instruction = fn.__doc__ or f"Call {fn.__name__}"

            # Capture inputs
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            inputs = dict(bound.arguments)

            # Execute function
            success = True
            output = None
            error = None
            try:
                output = fn(*args, **kwargs)
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration_ms = int((time.time() - start_time) * 1000)

                # Compute hashes
                instruction_hash = hashlib.sha256(fn_instruction.encode()).hexdigest()

                # Handle non-serializable inputs
                try:
                    input_json = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
                except (TypeError, ValueError):
                    input_json = json.dumps({"args": str(args), "kwargs": str(kwargs)})
                input_hash = hashlib.sha256(input_json.encode()).hexdigest()

                # Handle non-serializable outputs
                if success:
                    try:
                        output_json = json.dumps(output, sort_keys=True, separators=(",", ":"), default=str)
                    except (TypeError, ValueError):
                        output_json = json.dumps({"result": str(output)})
                else:
                    output_json = json.dumps({"error": error})
                output_hash = hashlib.sha256(output_json.encode()).hexdigest()

                # Get parent info
                parent = runtime.last_action
                parent_action_id = parent.action_id if parent else None
                parent_receipt_hash = parent.compute_receipt_hash() if parent else None

                # Create action
                action_id = f"fn_{fn.__name__}_{len(runtime._actions):04d}_{uuid.uuid4().hex[:8]}"
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
                    metadata={
                        "function_name": fn.__name__,
                        "module": fn.__module__,
                    },
                )

                # Record action (disable auto-chain since we set parent manually)
                old_auto_chain = runtime.auto_chain
                runtime.auto_chain = False
                try:
                    runtime.record(action)
                finally:
                    runtime.auto_chain = old_auto_chain

            return output

        return wrapper

    return decorator


def wrap_class_method(
    runtime: "AgentRuntime",
    action_type: str = "method_call",
) -> Callable:
    """Decorator that wraps a class method as a proof-carrying agent action.

    Similar to wrap_function but handles 'self' correctly.

    Args:
        runtime: The AgentRuntime to record actions to
        action_type: The type of action (default: "method_call")

    Returns:
        Decorator function

    Example:
        class MyAgent:
            @wrap_class_method(runtime, action_type="tool_call")
            def process(self, data: str) -> str:
                return data.upper()
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs) -> Any:
            start_time = time.time()

            # Build instruction from method info
            fn_instruction = fn.__doc__ or f"Call {fn.__name__}"

            # Capture inputs (exclude self)
            sig = inspect.signature(fn)
            params = list(sig.parameters.keys())
            # Remove 'self' from params
            if params and params[0] == 'self':
                params = params[1:]

            # Build inputs dict
            inputs = {}
            for i, param in enumerate(params):
                if i < len(args):
                    inputs[param] = args[i]
            inputs.update(kwargs)

            # Execute method
            success = True
            output = None
            error = None
            try:
                output = fn(self, *args, **kwargs)
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration_ms = int((time.time() - start_time) * 1000)

                # Compute hashes
                instruction_hash = hashlib.sha256(fn_instruction.encode()).hexdigest()

                try:
                    input_json = json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str)
                except (TypeError, ValueError):
                    input_json = json.dumps({"args": str(args), "kwargs": str(kwargs)})
                input_hash = hashlib.sha256(input_json.encode()).hexdigest()

                if success:
                    try:
                        output_json = json.dumps(output, sort_keys=True, separators=(",", ":"), default=str)
                    except (TypeError, ValueError):
                        output_json = json.dumps({"result": str(output)})
                else:
                    output_json = json.dumps({"error": error})
                output_hash = hashlib.sha256(output_json.encode()).hexdigest()

                # Get parent info
                parent = runtime.last_action
                parent_action_id = parent.action_id if parent else None
                parent_receipt_hash = parent.compute_receipt_hash() if parent else None

                # Create action
                class_name = self.__class__.__name__
                action_id = f"method_{class_name}_{fn.__name__}_{len(runtime._actions):04d}_{uuid.uuid4().hex[:8]}"
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
                    metadata={
                        "class_name": class_name,
                        "method_name": fn.__name__,
                        "module": fn.__module__,
                    },
                )

                # Record action
                old_auto_chain = runtime.auto_chain
                runtime.auto_chain = False
                try:
                    runtime.record(action)
                finally:
                    runtime.auto_chain = old_auto_chain

            return output

        return wrapper

    return decorator
