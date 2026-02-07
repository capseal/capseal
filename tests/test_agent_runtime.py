"""Tests for AgentRuntime.

Tests:
- Create AgentRuntime, record actions, finalize with proof
- Context manager interface
- Action chaining
- Proof verification
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestAgentRuntime:
    """Tests for AgentRuntime class."""

    def test_create_runtime(self, tmp_path):
        """Create a runtime and check initial state."""
        from capseal.agent_runtime import AgentRuntime

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        assert runtime.action_count == 0
        assert runtime.last_action is None
        assert runtime.last_receipt_hash is None
        assert output_dir.exists()

    def test_record_action(self, tmp_path):
        """Record an action and verify it's tracked."""
        from capseal.agent_runtime import AgentRuntime
        from capseal.agent_protocol import create_action

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        action = create_action(
            action_id="act_001",
            action_type="tool_call",
            instruction="Read file",
            inputs={"path": "foo.py"},
            outputs={"content": "hello"},
        )

        receipt = runtime.record(action)

        assert runtime.action_count == 1
        assert runtime.last_action is not None
        assert runtime.last_action.action_id == "act_001"
        assert len(receipt) == 64  # SHA256 hex

    def test_record_simple(self, tmp_path):
        """Test record_simple convenience method."""
        from capseal.agent_runtime import AgentRuntime

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        receipt = runtime.record_simple(
            action_type="tool_call",
            instruction="Read file",
            inputs={"path": "foo.py"},
            outputs={"content": "hello"},
            success=True,
            duration_ms=100,
        )

        assert runtime.action_count == 1
        assert len(receipt) == 64

    def test_auto_chaining(self, tmp_path):
        """Test automatic action chaining."""
        from capseal.agent_runtime import AgentRuntime

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir, auto_chain=True)

        # Record first action
        runtime.record_simple(
            action_type="observation",
            instruction="Observe",
            inputs={},
            outputs={"observed": True},
        )

        first_receipt = runtime.last_receipt_hash

        # Record second action (should auto-chain)
        runtime.record_simple(
            action_type="decision",
            instruction="Decide",
            inputs={"context": "observed"},
            outputs={"plan": "proceed"},
        )

        # Second action should have first action as parent
        second_action = runtime.last_action
        assert second_action.parent_receipt_hash == first_receipt

    def test_finalize_with_proof(self, tmp_path):
        """Finalize runtime and generate proof."""
        from capseal.agent_runtime import AgentRuntime

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        # Record 3 actions
        for i in range(3):
            runtime.record_simple(
                action_type="tool_call",
                instruction=f"Step {i}",
                inputs={"step": i},
                outputs={"result": f"done_{i}"},
            )

        # Finalize with proof
        result = runtime.finalize(prove=True)

        assert result["status"] == "proved"
        assert result["num_actions"] == 3
        assert result["verified"] is True
        assert (output_dir / "agent_capsule.json").exists()

    def test_finalize_without_proof(self, tmp_path):
        """Finalize runtime without generating proof."""
        from capseal.agent_runtime import AgentRuntime

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        runtime.record_simple(
            action_type="observation",
            instruction="Observe",
            inputs={},
            outputs={},
        )

        result = runtime.finalize(prove=False)

        assert result["status"] == "recorded"
        assert result["num_actions"] == 1
        assert not (output_dir / "agent_capsule.json").exists()

    def test_context_manager(self, tmp_path):
        """Test context manager interface."""
        from capseal.agent_runtime import AgentRuntime

        output_dir = tmp_path / "test_run"

        with AgentRuntime(output_dir=output_dir) as runtime:
            runtime.record_simple(
                action_type="tool_call",
                instruction="Do something",
                inputs={"x": 1},
                outputs={"y": 2},
            )

        # After exit, finalize should have been called
        assert (output_dir / "agent_capsule.json").exists()

    def test_verify_after_finalize(self, tmp_path):
        """Verify the generated capsule."""
        from capseal.agent_runtime import AgentRuntime

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        for i in range(3):
            runtime.record_simple(
                action_type="tool_call",
                instruction=f"Step {i}",
                inputs={"step": i},
                outputs={"result": f"done_{i}"},
            )

        runtime.finalize(prove=True)

        # Verify
        valid, details = runtime.verify()

        assert valid is True
        assert "capsule_hash_valid" in details["checks"]

    def test_get_action_chain(self, tmp_path):
        """Get the full action chain."""
        from capseal.agent_runtime import AgentRuntime

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        for i in range(3):
            runtime.record_simple(
                action_type="tool_call" if i != 1 else "decision",
                instruction=f"Step {i}",
                inputs={"step": i},
                outputs={"result": f"done_{i}"},
            )

        chain = runtime.get_action_chain()

        assert len(chain) == 3
        assert chain[0]["index"] == 0
        assert chain[1]["index"] == 1
        assert chain[0]["parent_receipt"] is None
        assert chain[1]["parent_receipt"] is not None

    def test_cannot_record_after_finalize(self, tmp_path):
        """Cannot record actions after finalize."""
        from capseal.agent_runtime import AgentRuntime

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        runtime.record_simple(
            action_type="observation",
            instruction="Observe",
            inputs={},
            outputs={},
        )

        runtime.finalize(prove=True)

        with pytest.raises(RuntimeError, match="finalize"):
            runtime.record_simple(
                action_type="observation",
                instruction="Too late",
                inputs={},
                outputs={},
            )

    def test_empty_finalize(self, tmp_path):
        """Finalize with no actions."""
        from capseal.agent_runtime import AgentRuntime

        output_dir = tmp_path / "test_run"
        runtime = AgentRuntime(output_dir=output_dir)

        result = runtime.finalize(prove=True)

        assert result["status"] == "empty"


class TestInspectAgentRun:
    """Tests for inspect_agent_run function."""

    def test_inspect_completed_run(self, tmp_path):
        """Inspect a completed agent run."""
        from capseal.agent_runtime import AgentRuntime, inspect_agent_run

        output_dir = tmp_path / "test_run"

        with AgentRuntime(output_dir=output_dir) as runtime:
            for i in range(3):
                runtime.record_simple(
                    action_type="tool_call",
                    instruction=f"Step {i}",
                    inputs={"step": i},
                    outputs={"result": f"done_{i}"},
                )

        # Inspect the run
        result = inspect_agent_run(output_dir)

        assert result["exists"] is True
        assert result["num_actions"] == 3
        assert "action_chain" in result
        assert len(result["action_chain"]) == 3
        assert "capsule" in result
        assert result["capsule"]["verification_valid"] is True

    def test_inspect_nonexistent_run(self, tmp_path):
        """Inspect a nonexistent directory."""
        from capseal.agent_runtime import inspect_agent_run

        result = inspect_agent_run(tmp_path / "nonexistent")

        assert result["exists"] is False
