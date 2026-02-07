"""Tests for receipt generation and verification."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestRoundReceipts:
    """Tests for build_round_receipt and verify_round_receipt."""

    def test_build_round_receipt(self, tmp_project):
        """Build a receipt for a round directory."""
        from capseal.shared.receipts import build_round_receipt

        # Create a mock round directory
        round_dir = tmp_project / ".capseal" / "runs" / "R0001_test"
        round_dir.mkdir(parents=True)

        # Create some artifacts
        (round_dir / "metrics.json").write_text(json.dumps({
            "tube": {"tube_var_sum": 0.1, "tube_coverage": 0.5},
            "status": "FIRST_ROUND",
        }))
        (round_dir / "active_sampling_plan.json").write_text(json.dumps({
            "selected": [1, 2, 3],
        }))

        round_config = {
            "grid_version": "test",
            "targets_per_round": 64,
            "episodes_per_budget_unit": 1,
            "seed": 42,
            "use_synthetic": True,
        }

        receipt = build_round_receipt(round_dir, round_config)

        assert receipt["schema"] == "round_receipt_v1"
        assert receipt["round_id"] == "R0001_test"
        assert "trace_spec_hash" in receipt
        assert "statement_hash" in receipt
        assert "artifact_hashes" in receipt
        assert len(receipt["artifact_hashes"]) == 2  # metrics.json + plan

    def test_verify_round_receipt_passes(self, tmp_project):
        """Verify a valid round receipt."""
        from capseal.shared.receipts import build_round_receipt, verify_round_receipt

        round_dir = tmp_project / ".capseal" / "runs" / "R0001_test"
        round_dir.mkdir(parents=True)

        (round_dir / "metrics.json").write_text(json.dumps({
            "tube": {"tube_var_sum": 0.1, "tube_coverage": 0.5},
            "status": "FIRST_ROUND",
        }))

        round_config = {"grid_version": "test", "seed": 42}
        receipt = build_round_receipt(round_dir, round_config)
        (round_dir / "round_receipt.json").write_text(json.dumps(receipt))

        result = verify_round_receipt(round_dir)

        assert result["verified"] is True
        assert len(result["mismatches"]) == 0

    def test_verify_detects_tampering(self, tmp_project):
        """Verify detects when artifacts are modified."""
        from capseal.shared.receipts import build_round_receipt, verify_round_receipt

        round_dir = tmp_project / ".capseal" / "runs" / "R0001_test"
        round_dir.mkdir(parents=True)

        (round_dir / "metrics.json").write_text(json.dumps({
            "tube": {"tube_var_sum": 0.1, "tube_coverage": 0.5},
            "status": "FIRST_ROUND",
        }))

        round_config = {"grid_version": "test", "seed": 42}
        receipt = build_round_receipt(round_dir, round_config)
        (round_dir / "round_receipt.json").write_text(json.dumps(receipt))

        # Now tamper with the artifact
        (round_dir / "metrics.json").write_text(json.dumps({
            "tube": {"tube_var_sum": 999.0, "tube_coverage": 0.0},  # Changed!
            "status": "TAMPERED",
        }))

        result = verify_round_receipt(round_dir)

        assert result["verified"] is False
        assert len(result["mismatches"]) > 0


class TestRunReceipts:
    """Tests for build_run_receipt and verify_run_receipt."""

    def test_build_run_receipt_chains_rounds(self, tmp_project):
        """Run receipt chains multiple round receipts."""
        from capseal.shared.receipts import build_round_receipt, build_run_receipt

        run_dir = tmp_project / ".capseal" / "runs" / "test_run"
        run_dir.mkdir(parents=True)
        (run_dir / "rounds").mkdir()

        round_receipts = []
        for i in range(3):
            round_dir = run_dir / "rounds" / f"R{i:04d}_test"
            round_dir.mkdir()
            (round_dir / "metrics.json").write_text(json.dumps({
                "tube": {"tube_var_sum": 0.1 * i},
            }))

            receipt = build_round_receipt(round_dir, {"seed": i})
            (round_dir / "round_receipt.json").write_text(json.dumps(receipt))
            round_receipts.append(receipt)

        # Create run metadata
        (run_dir / "run_metadata.json").write_text(json.dumps({
            "run_uuid": "test-run-123",
        }))

        run_receipt = build_run_receipt(run_dir, round_receipts)

        assert run_receipt["schema"] == "run_receipt_v1"
        assert run_receipt["run_id"] == "test-run-123"
        assert run_receipt["total_rounds"] == 3
        assert "chain_hash" in run_receipt
        assert len(run_receipt["rounds"]) == 3

    def test_chain_hash_is_deterministic(self, tmp_project):
        """Chain hash is deterministic given same inputs."""
        from capseal.shared.receipts import build_round_receipt, build_run_receipt

        run_dir = tmp_project / ".capseal" / "runs" / "test_run"
        run_dir.mkdir(parents=True)
        (run_dir / "rounds").mkdir()
        (run_dir / "run_metadata.json").write_text(json.dumps({"run_uuid": "x"}))

        round_receipts = []
        for i in range(2):
            round_dir = run_dir / "rounds" / f"R{i:04d}"
            round_dir.mkdir()
            (round_dir / "metrics.json").write_text(json.dumps({"v": i}))
            receipt = build_round_receipt(round_dir, {"seed": i})
            round_receipts.append(receipt)

        run_receipt1 = build_run_receipt(run_dir, round_receipts)
        run_receipt2 = build_run_receipt(run_dir, round_receipts)

        assert run_receipt1["chain_hash"] == run_receipt2["chain_hash"]


class TestHashFunctions:
    """Tests for low-level hash functions."""

    def test_hash_str_deterministic(self):
        """hash_str is deterministic."""
        from capseal.shared.receipts import hash_str

        h1 = hash_str("hello world")
        h2 = hash_str("hello world")

        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_canonical_json_sorted(self):
        """canonical_json produces sorted, compact output."""
        from capseal.shared.receipts import canonical_json

        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(obj)

        assert result == '{"a":2,"m":3,"z":1}'
        assert " " not in result  # No whitespace
