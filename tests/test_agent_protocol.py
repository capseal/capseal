"""Tests for the proof-carrying agent protocol.

Tests:
- AgentAction creation and validation
- Row encoding/decoding
- Trace simulation and proof generation
- Tampering detection
- Chain constraint violation detection
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


class TestAgentAction:
    """Tests for AgentAction dataclass."""

    def test_create_valid_action(self):
        """Create a valid action with all fields."""
        from bef_zk.capsule.agent_protocol import AgentAction

        action = AgentAction(
            action_id="act_001",
            action_type="tool_call",
            instruction_hash=hashlib.sha256(b"read file").hexdigest(),
            input_hash=hashlib.sha256(b"foo.py").hexdigest(),
            output_hash=hashlib.sha256(b"file content").hexdigest(),
            parent_action_id=None,
            parent_receipt_hash=None,
            gate_score=0.12,
            gate_decision="pass",
            policy_verdict="FULLY_VERIFIED",
            success=True,
            duration_ms=45,
            timestamp="2024-01-15T10:30:00Z",
            metadata={"model": "gpt-4"},
        )

        assert action.action_id == "act_001"
        assert action.action_type == "tool_call"
        assert action.success is True
        assert action.gate_decision == "pass"

    def test_invalid_action_type_raises(self):
        """Invalid action_type should raise ValueError."""
        from bef_zk.capsule.agent_protocol import AgentAction

        with pytest.raises(ValueError, match="Invalid action_type"):
            AgentAction(
                action_id="act_001",
                action_type="invalid_type",  # Not in ACTION_TYPES
                instruction_hash="abc123",
                input_hash="def456",
                output_hash="ghi789",
                parent_action_id=None,
                parent_receipt_hash=None,
                gate_score=None,
                gate_decision=None,
                policy_verdict=None,
                success=True,
                duration_ms=0,
                timestamp="2024-01-15T10:30:00Z",
            )

    def test_invalid_gate_decision_raises(self):
        """Invalid gate_decision should raise ValueError."""
        from bef_zk.capsule.agent_protocol import AgentAction

        with pytest.raises(ValueError, match="Invalid gate_decision"):
            AgentAction(
                action_id="act_001",
                action_type="tool_call",
                instruction_hash="abc123",
                input_hash="def456",
                output_hash="ghi789",
                parent_action_id=None,
                parent_receipt_hash=None,
                gate_score=0.5,
                gate_decision="invalid_decision",  # Not in GATE_DECISIONS
                policy_verdict=None,
                success=True,
                duration_ms=0,
                timestamp="2024-01-15T10:30:00Z",
            )

    def test_receipt_hash_deterministic(self):
        """Receipt hash should be deterministic."""
        from bef_zk.capsule.agent_protocol import AgentAction

        action = AgentAction(
            action_id="act_001",
            action_type="tool_call",
            instruction_hash="abc123",
            input_hash="def456",
            output_hash="ghi789",
            parent_action_id=None,
            parent_receipt_hash=None,
            gate_score=None,
            gate_decision=None,
            policy_verdict=None,
            success=True,
            duration_ms=100,
            timestamp="2024-01-15T10:30:00Z",
        )

        hash1 = action.compute_receipt_hash()
        hash2 = action.compute_receipt_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex

    def test_create_action_helper(self):
        """Test create_action convenience function."""
        from bef_zk.capsule.agent_protocol import create_action

        action = create_action(
            action_id="act_001",
            action_type="code_gen",
            instruction="Generate a function",
            inputs={"prompt": "write hello world"},
            outputs={"code": "print('hello')"},
            success=True,
            duration_ms=500,
        )

        assert action.action_id == "act_001"
        assert action.action_type == "code_gen"
        assert len(action.instruction_hash) == 64
        assert len(action.input_hash) == 64
        assert len(action.output_hash) == 64

    def test_action_chain(self):
        """Test parent action chaining."""
        from bef_zk.capsule.agent_protocol import create_action

        action1 = create_action(
            action_id="act_001",
            action_type="observation",
            instruction="Observe environment",
            inputs={},
            outputs={"observed": True},
        )

        action2 = create_action(
            action_id="act_002",
            action_type="decision",
            instruction="Decide next step",
            inputs={"context": "observed"},
            outputs={"plan": "proceed"},
            parent=action1,
        )

        assert action2.parent_action_id == "act_001"
        assert action2.parent_receipt_hash == action1.compute_receipt_hash()


class TestAgentAIR:
    """Tests for AgentAIR row encoding."""

    def test_encode_decode_row(self):
        """Encode and decode should preserve data."""
        from bef_zk.capsule.agent_protocol import create_action
        from bef_zk.capsule.agent_air import encode_agent_action_row, decode_agent_action_row

        action = create_action(
            action_id="act_001",
            action_type="tool_call",
            instruction="Read file",
            inputs={"path": "foo.py"},
            outputs={"content": "hello"},
        )

        row = encode_agent_action_row(action, action_index=0)

        assert len(row) == 14
        assert all(isinstance(v, int) for v in row)
        assert row[0] == 0  # action_index

        decoded = decode_agent_action_row(row)
        assert decoded.action_index == 0
        assert decoded.status_flags["success"] is True
        assert decoded.status_flags["is_tool_call"] is True

    def test_row_field_bounds(self):
        """All row elements should be within Goldilocks field bounds."""
        from bef_zk.capsule.agent_protocol import create_action
        from bef_zk.capsule.agent_air import encode_agent_action_row, GOLDILOCKS_P, verify_row_field_bounds

        action = create_action(
            action_id="act_001",
            action_type="code_gen",
            instruction="Generate code",
            inputs={"prompt": "hello" * 1000},  # Large input
            outputs={"code": "x" * 10000},  # Large output
        )

        row = encode_agent_action_row(action, action_index=99999)

        assert verify_row_field_bounds(row)
        assert all(0 <= v < GOLDILOCKS_P for v in row)

    def test_build_row_matrix(self):
        """Test building row matrix from multiple actions."""
        from bef_zk.capsule.agent_protocol import create_action
        from bef_zk.capsule.agent_air import build_agent_row_matrix

        actions = []
        parent = None
        for i in range(5):
            action = create_action(
                action_id=f"act_{i:03d}",
                action_type="tool_call" if i % 2 == 0 else "decision",
                instruction=f"Step {i}",
                inputs={"step": i},
                outputs={"result": f"done_{i}"},
                parent=parent,
            )
            actions.append(action)
            parent = action

        rows = build_agent_row_matrix(actions)

        assert len(rows) == 5
        for i, row in enumerate(rows):
            assert len(row) == 14
            assert row[0] == i  # action_index


class TestAgentAdapter:
    """Tests for AgentAdapter proof generation."""

    def test_simulate_trace(self):
        """Test trace simulation from actions."""
        from bef_zk.capsule.agent_protocol import create_action
        from bef_zk.capsule.agent_adapter import AgentAdapter

        adapter = AgentAdapter()

        actions = []
        parent = None
        for i in range(3):
            action = create_action(
                action_id=f"act_{i:03d}",
                action_type="tool_call",
                instruction=f"Step {i}",
                inputs={"step": i},
                outputs={"result": f"done_{i}"},
                parent=parent,
            )
            actions.append(action)
            parent = action

        artifacts = adapter.simulate_trace(actions)

        assert artifacts.statement.num_actions == 3
        assert len(artifacts.rows) == 3
        assert artifacts.statement.final_receipt_hash == actions[-1].compute_receipt_hash()

    def test_prove_actions_5_actions(self, tmp_path):
        """Create 5 AgentActions, prove with AgentAdapter, verify capsule."""
        from bef_zk.capsule.agent_protocol import create_action
        from bef_zk.capsule.agent_adapter import AgentAdapter, verify_agent_capsule

        adapter = AgentAdapter()
        output_dir = tmp_path / "agent_run"

        # Create 5 actions
        actions = []
        parent = None
        for i in range(5):
            action = create_action(
                action_id=f"act_{i:03d}",
                action_type=["tool_call", "code_gen", "decision", "observation", "api_request"][i],
                instruction=f"Action {i}",
                inputs={"step": i, "data": f"input_{i}"},
                outputs={"result": f"output_{i}"},
                parent=parent,
                success=True,
                duration_ms=100 * (i + 1),
            )
            actions.append(action)
            parent = action

        # Prove
        result = adapter.prove_actions(actions, output_dir)

        assert result["verified"] is True
        assert result["num_actions"] == 5
        assert Path(result["capsule_path"]).exists()

        # Verify capsule
        valid, details = verify_agent_capsule(Path(result["capsule_path"]))
        assert valid is True
        assert "capsule_hash_valid" in details["checks"]

    def test_tampering_detection(self, tmp_path):
        """Test that tampering with capsule is detected."""
        from bef_zk.capsule.agent_protocol import create_action
        from bef_zk.capsule.agent_adapter import AgentAdapter, verify_agent_capsule

        adapter = AgentAdapter()
        output_dir = tmp_path / "agent_run"

        actions = []
        parent = None
        for i in range(3):
            action = create_action(
                action_id=f"act_{i:03d}",
                action_type="tool_call",
                instruction=f"Action {i}",
                inputs={"step": i},
                outputs={"result": f"output_{i}"},
                parent=parent,
            )
            actions.append(action)
            parent = action

        result = adapter.prove_actions(actions, output_dir)
        capsule_path = Path(result["capsule_path"])

        # Tamper with capsule
        capsule = json.loads(capsule_path.read_text())
        capsule["statement"]["public_inputs"]["num_actions"] = 999  # Tampered!
        capsule_path.write_text(json.dumps(capsule))

        # Verify should fail
        valid, details = verify_agent_capsule(capsule_path)
        assert valid is False
        assert "Capsule hash mismatch" in details.get("error", "")

    def test_chain_constraint_violation(self, tmp_path):
        """Test that chain constraint violations are detected."""
        from bef_zk.capsule.agent_protocol import AgentAction, create_action
        from bef_zk.capsule.agent_adapter import AgentAdapter

        adapter = AgentAdapter()
        output_dir = tmp_path / "agent_run"

        # Create actions with broken chain (action 2 doesn't link to action 1)
        action1 = create_action(
            action_id="act_001",
            action_type="tool_call",
            instruction="Step 1",
            inputs={"step": 1},
            outputs={"result": "done_1"},
        )

        # This action has WRONG parent (None instead of action1)
        action2 = AgentAction(
            action_id="act_002",
            action_type="tool_call",
            instruction_hash=hashlib.sha256(b"Step 2").hexdigest(),
            input_hash=hashlib.sha256(b'{"step": 2}').hexdigest(),
            output_hash=hashlib.sha256(b"done_2").hexdigest(),
            parent_action_id=None,  # Should be action1.action_id
            parent_receipt_hash=None,  # Should be action1's receipt
            gate_score=None,
            gate_decision=None,
            policy_verdict=None,
            success=True,
            duration_ms=0,
            timestamp="2024-01-15T10:30:00Z",
        )

        # Add a third action that correctly links to action2
        action3 = create_action(
            action_id="act_003",
            action_type="tool_call",
            instruction="Step 3",
            inputs={"step": 3},
            outputs={"result": "done_3"},
            parent=action2,
        )

        actions = [action1, action2, action3]

        # Prove - should detect chain violation
        result = adapter.prove_actions(actions, output_dir)

        # The proof should NOT verify because chain is broken
        assert result["verified"] is False


class TestAgentConstraints:
    """Tests for AgentAIR constraints."""

    def test_verify_valid_trace(self):
        """Verify a valid trace passes all constraints."""
        from bef_zk.capsule.agent_protocol import create_action
        from bef_zk.capsule.agent_air import build_agent_row_matrix
        from bef_zk.capsule.agent_constraints import verify_agent_trace

        actions = []
        parent = None
        for i in range(5):
            action = create_action(
                action_id=f"act_{i:03d}",
                action_type="tool_call",
                instruction=f"Step {i}",
                inputs={"step": i},
                outputs={"result": f"done_{i}"},
                parent=parent,
            )
            actions.append(action)
            parent = action

        rows = build_agent_row_matrix(actions)
        final_receipt = actions[-1].compute_receipt_hash()

        valid, results = verify_agent_trace(rows, final_receipt)

        assert valid is True
        assert all(r.satisfied for r in results)

    def test_first_row_boundary_constraint(self):
        """First row must have prev_receipt_hash == 0."""
        from bef_zk.capsule.agent_constraints import AgentConstraints

        # Valid first row (prev_receipt is 0)
        row_valid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 123, 456, 1]
        assert AgentConstraints.eval_boundary_first_prev_lo(row_valid) == 0
        assert AgentConstraints.eval_boundary_first_prev_hi(row_valid) == 0

        # Invalid first row (prev_receipt is not 0)
        row_invalid = [0, 0, 0, 0, 0, 0, 0, 0, 0, 999, 888, 123, 456, 1]
        assert AgentConstraints.eval_boundary_first_prev_lo(row_invalid) != 0

    def test_transition_chain_constraint(self):
        """Row[i].receipt == Row[i+1].prev_receipt."""
        from bef_zk.capsule.agent_constraints import AgentConstraints

        # Valid chain
        row_curr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 111, 222, 1]  # receipt = (111, 222)
        row_next = [1, 0, 0, 0, 0, 0, 0, 0, 0, 111, 222, 333, 444, 1]  # prev = (111, 222)

        assert AgentConstraints.eval_transition_chain_lo(row_curr, row_next) == 0
        assert AgentConstraints.eval_transition_chain_hi(row_curr, row_next) == 0

        # Broken chain
        row_bad_next = [1, 0, 0, 0, 0, 0, 0, 0, 0, 999, 888, 333, 444, 1]  # prev != curr.receipt

        assert AgentConstraints.eval_transition_chain_lo(row_curr, row_bad_next) != 0

    def test_ordering_constraint(self):
        """Row[i+1].index == Row[i].index + 1."""
        from bef_zk.capsule.agent_constraints import AgentConstraints

        # Valid ordering
        row_0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        row_1 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert AgentConstraints.eval_transition_ordering(row_0, row_1) == 0

        # Invalid ordering (skip from 0 to 5)
        row_5 = [5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        assert AgentConstraints.eval_transition_ordering(row_0, row_5) != 0
