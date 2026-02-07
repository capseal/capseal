"""Tests for AgentLoop and related components."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bef_zk.capsule.agent_loop import (
    AgentLoop,
    CapSealAgent,
    ProposedAction,
    RiskFeedback,
    ActionResult,
    LoopResult,
)
from bef_zk.capsule.agents import SimpleAgent


# ============================================================================
# Mock Agents for Testing
# ============================================================================


class CountingAgent(CapSealAgent):
    """Agent that counts to a target then stops."""

    def __init__(self, target: int = 3):
        self.target = target
        self.count = 0

    def plan(self, task: str, context: dict) -> ProposedAction | None:
        if self.count >= self.target:
            return None  # Done
        return ProposedAction(
            action_type="observation",  # Use valid action type
            description=f"Count {self.count + 1}",
            instruction=f"Increment counter to {self.count + 1}",
            inputs={"current": self.count, "target": self.target},
        )

    def adapt(
        self, proposed: ProposedAction, feedback: RiskFeedback, context: dict
    ) -> ProposedAction | None:
        # Always give up on adapt for this simple agent
        return None

    def execute(self, action: ProposedAction) -> ActionResult:
        self.count += 1
        return ActionResult(
            success=True,
            outputs={"count": self.count},
            duration_ms=1,
        )


class AdaptingAgent(CapSealAgent):
    """Agent that adapts when told to."""

    def __init__(self):
        self.attempts = 0
        self.adapted = False

    def plan(self, task: str, context: dict) -> ProposedAction | None:
        if self.adapted:
            return None  # Done after adaptation
        return ProposedAction(
            action_type="code_gen",  # Use valid action type
            description="Do something risky",
            instruction="Execute risky operation",
            inputs={"risk_level": "high"},
            diff_text="- old code\n+ risky new code",
        )

    def adapt(
        self, proposed: ProposedAction, feedback: RiskFeedback, context: dict
    ) -> ProposedAction | None:
        self.attempts += 1
        if self.attempts >= 2:
            # Give up after 2 adaptation attempts
            return None
        # Propose safer alternative
        return ProposedAction(
            action_type="observation",  # Use valid action type
            description="Do something safe instead",
            instruction="Execute safe operation",
            inputs={"risk_level": "low"},
        )

    def execute(self, action: ProposedAction) -> ActionResult:
        self.adapted = action.action_type == "observation"
        return ActionResult(
            success=True,
            outputs={"adapted": self.adapted},
            duration_ms=1,
        )


class FailingAgent(CapSealAgent):
    """Agent whose actions fail."""

    def __init__(self):
        self.action_taken = False

    def plan(self, task: str, context: dict) -> ProposedAction | None:
        if self.action_taken:
            return None
        return ProposedAction(
            action_type="tool_call",  # Use valid action type
            description="This will fail",
            instruction="Do something that fails",
            inputs={},
        )

    def adapt(
        self, proposed: ProposedAction, feedback: RiskFeedback, context: dict
    ) -> ProposedAction | None:
        return None

    def execute(self, action: ProposedAction) -> ActionResult:
        self.action_taken = True
        return ActionResult(
            success=False,
            outputs={},
            error="Intentional failure",
            duration_ms=1,
        )


# ============================================================================
# Test Cases
# ============================================================================


class TestProposedAction:
    """Tests for ProposedAction dataclass."""

    def test_basic_creation(self):
        action = ProposedAction(
            action_type="tool_call",
            description="Test action",
            instruction="Do something",
            inputs={"arg": "value"},
        )
        assert action.action_type == "tool_call"
        assert action.description == "Test action"
        assert action.inputs == {"arg": "value"}
        assert action.diff_text == ""
        assert action.findings == []
        assert action.metadata == {}

    def test_with_diff_and_findings(self):
        action = ProposedAction(
            action_type="code_gen",
            description="Generate code",
            instruction="Write code",
            inputs={"file": "test.py"},
            diff_text="- old\n+ new",
            findings=[{"rule": "test-rule", "severity": "high"}],
        )
        assert action.diff_text == "- old\n+ new"
        assert len(action.findings) == 1


class TestRiskFeedback:
    """Tests for RiskFeedback dataclass."""

    def test_pass_decision(self):
        feedback = RiskFeedback(
            decision="pass",
            risk_score=0.1,
            confidence=0.9,
            features={"complexity": 0},
            grid_idx=42,
            suggestion="Proceed",
            attempt_number=1,
        )
        assert feedback.decision == "pass"
        assert feedback.risk_score == 0.1
        assert feedback.confidence == 0.9

    def test_skip_decision(self):
        feedback = RiskFeedback(
            decision="skip",
            risk_score=0.9,
            confidence=0.8,
            features={"complexity": 3},
            grid_idx=100,
            suggestion="Too risky",
            attempt_number=2,
            prior_attempts=[{"decision": "skip"}],
        )
        assert feedback.decision == "skip"
        assert len(feedback.prior_attempts) == 1


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_success(self):
        result = ActionResult(
            success=True,
            outputs={"result": "done"},
            duration_ms=100,
        )
        assert result.success
        assert result.outputs == {"result": "done"}
        assert result.error is None

    def test_failure(self):
        result = ActionResult(
            success=False,
            outputs={},
            error="Something went wrong",
            duration_ms=50,
        )
        assert not result.success
        assert result.error == "Something went wrong"


class TestLoopResult:
    """Tests for LoopResult dataclass."""

    def test_empty_result(self):
        result = LoopResult(
            capsule=None,
            actions=[],
            risk_log=[],
            context={},
        )
        assert result.success_rate == 0.0
        assert result.total_actions == 0
        assert result.total_adaptations == 0
        assert result.capsule_hash is None

    def test_with_actions(self):
        # Mock actions with success attribute
        class MockAction:
            def __init__(self, success):
                self.success = success

        result = LoopResult(
            capsule={"capsule_hash": "abc123"},
            actions=[MockAction(True), MockAction(True), MockAction(False)],
            risk_log=[],
            context={"adapted_count": 2},
        )
        assert result.success_rate == pytest.approx(2 / 3)
        assert result.total_actions == 3
        assert result.total_adaptations == 2
        assert result.capsule_hash == "abc123"


class TestAgentLoop:
    """Tests for AgentLoop execution."""

    def test_basic_loop_execution(self):
        """Test that a simple agent runs to completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = CountingAgent(target=3)
            loop = AgentLoop(
                agent=agent,
                output_dir=Path(tmpdir),
                prove=False,  # Don't generate proof for speed
            )

            result = loop.run("count to 3")

            assert result.total_actions == 3
            assert agent.count == 3

    def test_empty_task_completion(self):
        """Test agent that immediately returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = CountingAgent(target=0)
            loop = AgentLoop(
                agent=agent,
                output_dir=Path(tmpdir),
                prove=False,
            )

            result = loop.run("do nothing")

            assert result.total_actions == 0
            assert agent.count == 0

    def test_failing_action_recorded(self):
        """Test that failed actions are still recorded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = FailingAgent()
            loop = AgentLoop(
                agent=agent,
                output_dir=Path(tmpdir),
                prove=False,
            )

            result = loop.run("fail")

            assert result.total_actions == 1
            assert result.success_rate == 0.0

    def test_risk_log_populated(self):
        """Test that risk_log is populated during execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = CountingAgent(target=2)
            loop = AgentLoop(
                agent=agent,
                output_dir=Path(tmpdir),
                prove=False,
            )

            result = loop.run("count")

            assert len(result.risk_log) == 2
            for entry in result.risk_log:
                assert "decision" in entry
                assert "action_type" in entry

    def test_context_updated(self):
        """Test that context is updated during execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = CountingAgent(target=2)
            loop = AgentLoop(
                agent=agent,
                output_dir=Path(tmpdir),
                prove=False,
            )

            result = loop.run("count task")

            assert result.context["task"] == "count task"
            assert len(result.context["actions_taken"]) == 2


class TestSimpleAgent:
    """Tests for SimpleAgent implementation."""

    def test_parse_done_response(self):
        """Test that DONE response returns None."""

        def mock_llm(prompt: str) -> str:
            return "DONE"

        agent = SimpleAgent(llm_fn=mock_llm)
        result = agent.plan("test task", {})
        assert result is None

    def test_parse_structured_response(self):
        """Test parsing of structured action response."""

        def mock_llm(prompt: str) -> str:
            return """ACTION_TYPE: tool_call
DESCRIPTION: List files in directory
INSTRUCTION: List all files
INPUTS: {"command": "ls -la", "tool_name": "bash"}"""

        agent = SimpleAgent(llm_fn=mock_llm)
        result = agent.plan("list files", {})

        assert result is not None
        assert result.action_type == "tool_call"
        assert result.description == "List files in directory"
        assert result.inputs["command"] == "ls -la"

    def test_parse_code_gen_response(self):
        """Test parsing of code generation response."""

        def mock_llm(prompt: str) -> str:
            return """ACTION_TYPE: code_gen
DESCRIPTION: Fix bug
INSTRUCTION: Update the code
INPUTS: {"file": "test.py"}
DIFF:
- old_line
+ new_line"""

        agent = SimpleAgent(llm_fn=mock_llm)
        result = agent.plan("fix bug", {})

        assert result is not None
        assert result.action_type == "code_gen"
        assert "- old_line" in result.diff_text

    def test_execute_tool_call(self):
        """Test execution of tool call."""
        executed = []

        def mock_bash(command: str) -> str:
            executed.append(command)
            return "output"

        agent = SimpleAgent(
            llm_fn=lambda p: "DONE",
            tools={"bash": mock_bash},
        )

        action = ProposedAction(
            action_type="tool_call",
            description="Run bash",
            instruction="Run ls",
            inputs={"command": "ls", "tool_name": "bash"},
        )

        result = agent.execute(action)

        assert result.success
        assert executed == ["ls"]
        assert result.outputs["result"] == "output"

    def test_adapt_builds_prompt(self):
        """Test that adapt builds proper adaptation prompt."""
        prompts = []

        def mock_llm(prompt: str) -> str:
            prompts.append(prompt)
            return "DONE"

        agent = SimpleAgent(llm_fn=mock_llm)
        agent.conversation = [{"role": "user", "content": "initial"}]

        feedback = RiskFeedback(
            decision="skip",
            risk_score=0.8,
            confidence=0.9,
            features={},
            grid_idx=0,
            suggestion="Try safer approach",
            attempt_number=1,
        )

        proposed = ProposedAction(
            action_type="code_gen",  # Use valid action type
            description="Risky action",
            instruction="Do risky thing",
            inputs={},
        )

        agent.adapt(proposed, feedback, {})

        # Check that adaptation prompt was added to conversation
        assert any("flagged as risky" in msg["content"] for msg in agent.conversation)


class TestAgentLoopIntegration:
    """Integration tests for AgentLoop with SimpleAgent."""

    def test_simple_agent_with_loop(self):
        """Test SimpleAgent running through AgentLoop."""
        action_count = 0

        def mock_llm(prompt: str) -> str:
            nonlocal action_count
            action_count += 1
            if action_count >= 3:
                return "DONE"
            return f"""ACTION_TYPE: observation
DESCRIPTION: Step {action_count}
INSTRUCTION: Do step {action_count}
INPUTS: {{"step": {action_count}}}"""

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SimpleAgent(llm_fn=mock_llm)
            loop = AgentLoop(
                agent=agent,
                output_dir=Path(tmpdir),
                prove=False,
            )

            result = loop.run("complete task")

            # Should have 2 actions (action_count 1 and 2, then DONE at 3)
            assert result.total_actions == 2

    def test_agent_records_to_runtime(self):
        """Test that actions are properly recorded to runtime."""

        def mock_llm(prompt: str) -> str:
            if "Step 1" in str(prompt) or "PREVIOUS ACTIONS" in prompt:
                return "DONE"
            return """ACTION_TYPE: tool_call
DESCRIPTION: Step 1
INSTRUCTION: Do step 1
INPUTS: {"tool_name": "echo", "message": "hello"}"""

        def mock_echo(message: str) -> str:
            return f"echoed: {message}"

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = SimpleAgent(
                llm_fn=mock_llm,
                tools={"echo": mock_echo},
            )
            loop = AgentLoop(
                agent=agent,
                output_dir=Path(tmpdir),
                prove=False,
            )

            result = loop.run("echo test")

            # Check actions were recorded
            assert result.total_actions >= 1

            # Check actions.jsonl was written
            actions_file = Path(tmpdir) / "actions.jsonl"
            assert actions_file.exists()
            content = actions_file.read_text().strip()
            assert len(content) > 0


class TestHumanReviewCallback:
    """Tests for human review callback functionality."""

    def test_human_approved(self):
        """Test human approval callback."""
        human_calls = []

        def human_callback(action: ProposedAction, feedback: RiskFeedback) -> bool:
            human_calls.append((action, feedback))
            return True  # Approve

        # Create agent that triggers human review
        class HumanReviewAgent(CapSealAgent):
            def __init__(self):
                self.done = False

            def plan(self, task: str, context: dict) -> ProposedAction | None:
                if self.done:
                    return None
                return ProposedAction(
                    action_type="decision",  # Use valid action type
                    description="Action needing review",
                    instruction="Do reviewed action",
                    inputs={},
                )

            def adapt(self, proposed, feedback, context):
                return None

            def execute(self, action):
                self.done = True
                return ActionResult(success=True, outputs={})

        # We can't easily trigger human_review without posteriors,
        # but we can at least verify the callback mechanism exists
        with tempfile.TemporaryDirectory() as tmpdir:
            agent = HumanReviewAgent()
            loop = AgentLoop(
                agent=agent,
                output_dir=Path(tmpdir),
                prove=False,
                human_review_callback=human_callback,
            )

            # Without posteriors, gate returns "pass", so callback won't be called
            result = loop.run("test")
            assert result.total_actions == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
