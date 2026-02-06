"""Tests for the generic function wrapper adapter.

Tests:
- wrap_function decorator
- Action recording
- Capsule verification
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestWrapFunction:
    """Tests for wrap_function decorator."""

    def test_basic_wrap(self, tmp_path):
        """Wrap a simple function and verify action is recorded."""
        from bef_zk.capsule.agent_runtime import AgentRuntime
        from bef_zk.capsule.adapters import wrap_function

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        @wrap_function(runtime, action_type="tool_call")
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        result = add(1, 2)

        assert result == 3
        assert runtime.action_count == 1

        action = runtime.last_action
        assert action.action_type == "tool_call"
        assert action.success is True

    def test_wrap_with_exception(self, tmp_path):
        """Wrapped function that raises exception still records action."""
        from bef_zk.capsule.agent_runtime import AgentRuntime
        from bef_zk.capsule.adapters import wrap_function

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        @wrap_function(runtime, action_type="tool_call")
        def fail():
            """Always fails."""
            raise ValueError("intentional failure")

        with pytest.raises(ValueError, match="intentional failure"):
            fail()

        assert runtime.action_count == 1

        action = runtime.last_action
        assert action.success is False

    def test_wrap_multiple_calls(self, tmp_path):
        """Multiple calls to wrapped function record multiple actions."""
        from bef_zk.capsule.agent_runtime import AgentRuntime
        from bef_zk.capsule.adapters import wrap_function

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        @wrap_function(runtime, action_type="tool_call")
        def increment(x: int) -> int:
            return x + 1

        increment(1)
        increment(2)
        increment(3)

        assert runtime.action_count == 3

    def test_wrap_and_prove(self, tmp_path):
        """Wrap functions, call them, and generate proof."""
        from bef_zk.capsule.agent_runtime import AgentRuntime
        from bef_zk.capsule.adapters import wrap_function

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        @wrap_function(runtime, action_type="tool_call")
        def read_file(path: str) -> str:
            """Read a file."""
            return f"content of {path}"

        @wrap_function(runtime, action_type="code_gen")
        def generate_code(prompt: str) -> str:
            """Generate code."""
            return f"def hello(): print('{prompt}')"

        @wrap_function(runtime, action_type="decision")
        def decide(options: list) -> str:
            """Make a decision."""
            return options[0] if options else "default"

        # Call the wrapped functions
        content = read_file("foo.py")
        code = generate_code("hello world")
        choice = decide(["option_a", "option_b"])

        assert runtime.action_count == 3

        # Finalize and verify
        result = runtime.finalize(prove=True)

        assert result["verified"] is True
        assert result["num_actions"] == 3

        # Verify capsule
        valid, details = runtime.verify()
        assert valid is True

    def test_wrap_with_custom_instruction(self, tmp_path):
        """Wrap with custom instruction text."""
        from bef_zk.capsule.agent_runtime import AgentRuntime
        from bef_zk.capsule.adapters import wrap_function

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        @wrap_function(runtime, action_type="api_request", instruction="Make API call to service")
        def api_call(endpoint: str) -> dict:
            return {"status": "ok", "endpoint": endpoint}

        result = api_call("/users")

        assert runtime.action_count == 1
        # The instruction is hashed, so we can't directly verify it,
        # but we can verify the action was recorded correctly
        action = runtime.last_action
        assert action.action_type == "api_request"

    def test_wrap_chaining(self, tmp_path):
        """Verify that wrapped function calls are properly chained."""
        from bef_zk.capsule.agent_runtime import AgentRuntime
        from bef_zk.capsule.adapters import wrap_function

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        @wrap_function(runtime, action_type="tool_call")
        def step_one() -> str:
            return "one"

        @wrap_function(runtime, action_type="tool_call")
        def step_two() -> str:
            return "two"

        @wrap_function(runtime, action_type="tool_call")
        def step_three() -> str:
            return "three"

        step_one()
        step_two()
        step_three()

        chain = runtime.get_action_chain()

        assert len(chain) == 3
        assert chain[0]["parent_receipt"] is None
        assert chain[1]["parent_receipt"] is not None
        assert chain[2]["parent_receipt"] is not None

        # Verify chain integrity
        result = runtime.finalize(prove=True)
        assert result["verified"] is True


class TestWrapClassMethod:
    """Tests for wrap_class_method decorator."""

    def test_wrap_method(self, tmp_path):
        """Wrap a class method."""
        from bef_zk.capsule.agent_runtime import AgentRuntime
        from bef_zk.capsule.adapters.generic_adapter import wrap_class_method

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        class MyAgent:
            @wrap_class_method(runtime, action_type="tool_call")
            def process(self, data: str) -> str:
                """Process data."""
                return data.upper()

        agent = MyAgent()
        result = agent.process("hello")

        assert result == "HELLO"
        assert runtime.action_count == 1

        action = runtime.last_action
        assert action.action_type == "tool_call"
        assert "MyAgent" in action.metadata.get("class_name", "")

    def test_wrap_multiple_methods(self, tmp_path):
        """Wrap multiple methods of a class."""
        from bef_zk.capsule.agent_runtime import AgentRuntime
        from bef_zk.capsule.adapters.generic_adapter import wrap_class_method

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        class Calculator:
            @wrap_class_method(runtime, action_type="tool_call")
            def add(self, a: int, b: int) -> int:
                return a + b

            @wrap_class_method(runtime, action_type="tool_call")
            def multiply(self, a: int, b: int) -> int:
                return a * b

        calc = Calculator()
        calc.add(1, 2)
        calc.multiply(3, 4)

        assert runtime.action_count == 2

        # Verify and prove
        result = runtime.finalize(prove=True)
        assert result["verified"] is True


class TestComplexWorkflow:
    """Tests for complex workflows with multiple adapters."""

    def test_mixed_calls(self, tmp_path):
        """Mix wrapped functions, direct record, and class methods."""
        from bef_zk.capsule.agent_runtime import AgentRuntime
        from bef_zk.capsule.adapters import wrap_function
        from bef_zk.capsule.adapters.generic_adapter import wrap_class_method

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        @wrap_function(runtime, action_type="observation")
        def observe_env() -> dict:
            return {"state": "ready"}

        class Planner:
            @wrap_class_method(runtime, action_type="decision")
            def plan(self, state: dict) -> str:
                return "execute_task"

        @wrap_function(runtime, action_type="tool_call")
        def execute_task() -> bool:
            return True

        # Run workflow
        state = observe_env()

        planner = Planner()
        plan = planner.plan(state)

        result = execute_task()

        # Also record a direct action
        runtime.record_simple(
            action_type="observation",
            instruction="Record completion",
            inputs={"plan": plan, "result": result},
            outputs={"status": "complete"},
        )

        assert runtime.action_count == 4

        # Verify
        result = runtime.finalize(prove=True)
        assert result["verified"] is True
        assert result["num_actions"] == 4
