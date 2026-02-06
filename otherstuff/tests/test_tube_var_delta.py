#!/usr/bin/env python3
"""Tests for tube_var_delta computation and summary.csv consistency.

These tests verify that:
1. Delta is computed correctly when previous round exists
2. Delta is None (not 0.0) when no previous round exists
3. summary.csv matches metrics.json
4. Previous round detection skips incomplete directories
5. LOOP COMPLETE output matches metrics.json and summary.csv (consistency)
6. Creating new round dir doesn't cause delta to become 0 (regression test)
"""

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None  # Allow running without pytest


# ---------------------------------------------------------------------------
# Unit tests for delta computation
# ---------------------------------------------------------------------------


class TestTubeMetricsDelta:
    """Test tube_var_delta computation in isolation."""

    def test_delta_computed_when_prev_exists(self):
        """Delta should be prev - current when previous round exists."""
        from loop_metrics import compute_tube_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create mock beta_posteriors.npz and enn.npz
            n_points = 100
            alpha = np.ones(n_points) * 2.0
            beta = np.ones(n_points) * 2.0
            np.savez(run_dir / "beta_posteriors.npz", alpha=alpha, beta=beta)

            q_enn = np.linspace(0, 1, n_points)  # Some points in tube [0.4, 0.6]
            np.savez(run_dir / "enn.npz", q_enn=q_enn)

            # Compute with a known previous tube_var
            prev_tube_var = 50.0
            metrics = compute_tube_metrics(run_dir, prev_tube_var=prev_tube_var)

            # Delta should be computed
            assert metrics.tube_var_delta is not None
            expected_delta = prev_tube_var - metrics.tube_var_sum
            assert abs(metrics.tube_var_delta - expected_delta) < 1e-9

    def test_delta_none_when_no_prev(self):
        """Delta should be None when no previous round (first round)."""
        from loop_metrics import compute_tube_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            # Create mock artifacts
            n_points = 100
            alpha = np.ones(n_points) * 2.0
            beta = np.ones(n_points) * 2.0
            np.savez(run_dir / "beta_posteriors.npz", alpha=alpha, beta=beta)

            q_enn = np.linspace(0, 1, n_points)
            np.savez(run_dir / "enn.npz", q_enn=q_enn)

            # Compute WITHOUT previous tube_var
            metrics = compute_tube_metrics(run_dir, prev_tube_var=None)

            # Delta should be None, NOT 0.0
            assert metrics.tube_var_delta is None

    def test_delta_is_positive_when_improving(self):
        """Delta should be positive when tube_var decreases (improvement)."""
        from loop_metrics import compute_tube_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            n_points = 100
            alpha = np.ones(n_points) * 2.0
            beta = np.ones(n_points) * 2.0
            np.savez(run_dir / "beta_posteriors.npz", alpha=alpha, beta=beta)

            q_enn = np.linspace(0, 1, n_points)
            np.savez(run_dir / "enn.npz", q_enn=q_enn)

            metrics = compute_tube_metrics(run_dir)
            current_var = metrics.tube_var_sum

            # Previous was higher (worse), so delta should be positive
            prev_tube_var = current_var + 10.0
            metrics_with_delta = compute_tube_metrics(run_dir, prev_tube_var=prev_tube_var)

            assert metrics_with_delta.tube_var_delta is not None
            assert metrics_with_delta.tube_var_delta > 0  # Positive = improved

    def test_delta_is_negative_when_regressing(self):
        """Delta should be negative when tube_var increases (regression)."""
        from loop_metrics import compute_tube_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)

            n_points = 100
            alpha = np.ones(n_points) * 2.0
            beta = np.ones(n_points) * 2.0
            np.savez(run_dir / "beta_posteriors.npz", alpha=alpha, beta=beta)

            q_enn = np.linspace(0, 1, n_points)
            np.savez(run_dir / "enn.npz", q_enn=q_enn)

            metrics = compute_tube_metrics(run_dir)
            current_var = metrics.tube_var_sum

            # Previous was lower (better), so delta should be negative
            prev_tube_var = current_var - 5.0
            metrics_with_delta = compute_tube_metrics(run_dir, prev_tube_var=prev_tube_var)

            assert metrics_with_delta.tube_var_delta is not None
            assert metrics_with_delta.tube_var_delta < 0  # Negative = regressed


class TestPreviousRoundDetection:
    """Test that previous round is correctly identified."""

    def test_skips_directory_without_metrics(self):
        """Should skip round directories without metrics.json."""
        from loop_io import get_previous_round_info

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            rounds_dir = base_dir / "rounds"
            rounds_dir.mkdir()

            # Create round 1 WITH metrics.json
            round1 = rounds_dir / "R0001_20260101_100000"
            round1.mkdir()
            metrics1 = {
                "tube": {"tube_var_sum": 100.0, "tube_coverage": 0.5},
                "counts": {"sampled_points_total": 50},
            }
            (round1 / "metrics.json").write_text(json.dumps(metrics1))

            # Create round 2 WITHOUT metrics.json (incomplete)
            round2 = rounds_dir / "R0002_20260101_100100"
            round2.mkdir()
            # No metrics.json!

            # Should return round 1, not round 2
            prev_info = get_previous_round_info(base_dir)

            assert prev_info is not None
            assert prev_info["round_id"] == "R0001_20260101_100000"
            assert prev_info["tube_var"] == 100.0

    def test_exclude_current_round(self):
        """Should skip the excluded round ID."""
        from loop_io import get_previous_round_info

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            rounds_dir = base_dir / "rounds"
            rounds_dir.mkdir()

            # Create round 1
            round1 = rounds_dir / "R0001_20260101_100000"
            round1.mkdir()
            metrics1 = {
                "tube": {"tube_var_sum": 100.0, "tube_coverage": 0.5},
                "counts": {"sampled_points_total": 50},
            }
            (round1 / "metrics.json").write_text(json.dumps(metrics1))

            # Create round 2 WITH metrics.json
            round2 = rounds_dir / "R0002_20260101_100100"
            round2.mkdir()
            metrics2 = {
                "tube": {"tube_var_sum": 90.0, "tube_coverage": 0.6},
                "counts": {"sampled_points_total": 60},
            }
            (round2 / "metrics.json").write_text(json.dumps(metrics2))

            # Exclude round 2 - should return round 1
            prev_info = get_previous_round_info(base_dir, exclude_round_id="R0002_20260101_100100")

            assert prev_info is not None
            assert prev_info["round_id"] == "R0001_20260101_100000"
            assert prev_info["tube_var"] == 100.0

    def test_returns_none_when_no_valid_rounds(self):
        """Should return None when no rounds have metrics.json."""
        from loop_io import get_previous_round_info

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            rounds_dir = base_dir / "rounds"
            rounds_dir.mkdir()

            # Create round without metrics
            round1 = rounds_dir / "R0001_20260101_100000"
            round1.mkdir()
            # No metrics.json!

            prev_info = get_previous_round_info(base_dir)
            assert prev_info is None

    def test_cross_series_isolation(self):
        """Round 1 of new series should NOT find previous series' rounds."""
        from loop_io import get_previous_round_info

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            rounds_dir = base_dir / "rounds"
            rounds_dir.mkdir()

            # Create rounds from PREVIOUS series (older timestamp)
            for i in range(1, 4):
                round_dir = rounds_dir / f"R000{i}_20260101_100000"
                round_dir.mkdir()
                metrics = {
                    "tube": {"tube_var_sum": 100.0 - i*10, "tube_coverage": 0.5},
                    "counts": {"sampled_points_total": 50 + i*10},
                }
                (round_dir / "metrics.json").write_text(json.dumps(metrics))

            # Now looking for previous round for R0001 of NEW series
            # With current_round_num=1, it should find NO previous round
            # (because all existing rounds have round_num >= 1)
            prev_info = get_previous_round_info(base_dir, current_round_num=1)
            assert prev_info is None, "Round 1 should have no previous round"

            # R0002 should find R0001 (from the same or previous series)
            prev_info = get_previous_round_info(base_dir, current_round_num=2)
            assert prev_info is not None
            assert prev_info["round_num"] == 1

    def test_timestamp_ordering(self):
        """Should prefer newer timestamps when multiple rounds exist."""
        from loop_io import get_previous_round_info

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            rounds_dir = base_dir / "rounds"
            rounds_dir.mkdir()

            # Create two R0001 directories with different timestamps
            round1_old = rounds_dir / "R0001_20260101_100000"
            round1_old.mkdir()
            metrics_old = {
                "tube": {"tube_var_sum": 100.0, "tube_coverage": 0.5},
                "counts": {"sampled_points_total": 50},
            }
            (round1_old / "metrics.json").write_text(json.dumps(metrics_old))

            round1_new = rounds_dir / "R0001_20260101_110000"  # Newer timestamp
            round1_new.mkdir()
            metrics_new = {
                "tube": {"tube_var_sum": 80.0, "tube_coverage": 0.6},
                "counts": {"sampled_points_total": 60},
            }
            (round1_new / "metrics.json").write_text(json.dumps(metrics_new))

            # Should return the newer R0001 when looking for previous for R0002
            prev_info = get_previous_round_info(base_dir, current_round_num=2)
            assert prev_info is not None
            assert prev_info["round_id"] == "R0001_20260101_110000"
            assert prev_info["tube_var"] == 80.0


class TestDetermineStatus:
    """Test status determination logic."""

    def test_first_round_status(self):
        """First round (None delta) should return FIRST_ROUND."""
        from loop_metrics import determine_status

        status = determine_status(
            tube_var_delta=None,
            tube_coverage=0.5,
            prev_coverage=0.0,
        )
        assert status == "FIRST_ROUND"

    def test_improved_status(self):
        """Positive delta should return IMPROVED."""
        from loop_metrics import determine_status

        status = determine_status(
            tube_var_delta=0.01,  # Positive = improved
            tube_coverage=0.5,
            prev_coverage=0.5,
        )
        assert status == "IMPROVED"

    def test_regressed_status(self):
        """Negative delta should return REGRESSED."""
        from loop_metrics import determine_status

        status = determine_status(
            tube_var_delta=-0.02,  # Negative = regressed
            tube_coverage=0.5,
            prev_coverage=0.5,
        )
        assert status == "REGRESSED"

    def test_no_change_status(self):
        """Small delta should return NO_CHANGE."""
        from loop_metrics import determine_status

        status = determine_status(
            tube_var_delta=0.0001,  # Small, below threshold
            tube_coverage=0.5,
            prev_coverage=0.5,
        )
        assert status == "NO_CHANGE"


class TestSummaryCsvConsistency:
    """Test that summary.csv matches metrics.json."""

    def test_summary_row_matches_metrics(self):
        """summary_row() should produce values matching the source metrics."""
        from loop_metrics import RoundMetrics, TubeMetrics, PlanSummary, ModelChange, FusionConsistency

        tube = TubeMetrics(
            tube_points_total=100,
            tube_sampled=50,
            tube_coverage=0.5,
            tube_var_sum=25.5,
            tube_var_mean=0.255,
            tube_var_delta=5.5,
        )

        metrics = RoundMetrics(
            round_id="R0001",
            timestamp="2026-01-31T12:00:00",
            seed=42,
        )
        metrics.tube = tube
        metrics.plan = PlanSummary(targets_selected=30, total_budget=1000)
        metrics.status = "IMPROVED"

        row = metrics.summary_row()

        # Check that summary.csv columns match metrics.json values
        assert row["tube_var"] == tube.tube_var_sum  # NOT tube_var_mean!
        assert row["tube_var_delta"] == tube.tube_var_delta
        assert row["tube_coverage"] == tube.tube_coverage
        assert row["targets_selected"] == 30
        assert row["status"] == "IMPROVED"

    def test_summary_row_none_delta_becomes_empty_string(self):
        """None delta should become empty string in CSV."""
        from loop_metrics import RoundMetrics, TubeMetrics

        tube = TubeMetrics(
            tube_var_delta=None,  # First round
        )

        metrics = RoundMetrics(round_id="R0001", timestamp="", seed=42)
        metrics.tube = tube

        row = metrics.summary_row()

        # Empty string, not 0.0 or "None"
        assert row["tube_var_delta"] == ""


# ---------------------------------------------------------------------------
# Integration test: Two-round simulation
# ---------------------------------------------------------------------------


class TestTwoRoundIntegration:
    """Integration test that simulates two rounds and verifies delta."""

    def test_delta_nonzero_when_tube_var_changes(self):
        """After two rounds with different tube_var, delta should be non-zero."""
        from loop_metrics import compute_tube_metrics, compute_round_metrics
        from loop_io import get_previous_round_info, save_json

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            rounds_dir = base_dir / "rounds"
            rounds_dir.mkdir()

            # --- Round 1 ---
            round1_dir = rounds_dir / "R0001_20260101_100000"
            round1_dir.mkdir()

            # Create artifacts with HIGH variance (lots of uncertainty)
            n_points = 100
            alpha1 = np.ones(n_points) * 1.5  # Low counts = high variance
            beta1 = np.ones(n_points) * 1.5
            np.savez(round1_dir / "beta_posteriors.npz", alpha=alpha1, beta=beta1)

            q_enn = np.linspace(0, 1, n_points)
            np.savez(round1_dir / "enn.npz", q_enn=q_enn)

            # Compute and save round 1 metrics
            metrics1 = compute_round_metrics(
                round_dir=round1_dir,
                round_id="R0001_20260101_100000",
                seed=42,
                prev_tube_var=None,  # First round
            )
            save_json(metrics1.to_dict(), round1_dir / "metrics.json")

            # Verify round 1 has None delta
            assert metrics1.tube.tube_var_delta is None
            assert metrics1.status == "FIRST_ROUND"

            # --- Round 2 ---
            round2_dir = rounds_dir / "R0002_20260101_100100"
            round2_dir.mkdir()

            # Create artifacts with LOWER variance (more certainty)
            alpha2 = np.ones(n_points) * 10.0  # Higher counts = lower variance
            beta2 = np.ones(n_points) * 10.0
            np.savez(round2_dir / "beta_posteriors.npz", alpha=alpha2, beta=beta2)
            np.savez(round2_dir / "enn.npz", q_enn=q_enn)

            # Get previous round info
            prev_info = get_previous_round_info(base_dir, exclude_round_id="R0002_20260101_100100")
            assert prev_info is not None
            assert prev_info["round_id"] == "R0001_20260101_100000"

            # Compute round 2 metrics
            metrics2 = compute_round_metrics(
                round_dir=round2_dir,
                round_id="R0002_20260101_100100",
                seed=43,
                prev_round_dir=prev_info["round_dir"],
                prev_tube_var=prev_info["tube_var"],
                prev_coverage=prev_info["coverage"],
            )

            # THE KEY ASSERTION: Delta should be non-zero!
            assert metrics2.tube.tube_var_delta is not None
            assert metrics2.tube.tube_var_delta != 0.0

            # Delta should be positive (improvement: variance decreased)
            assert metrics2.tube.tube_var_delta > 0

            # tube_var should have decreased
            assert metrics2.tube.tube_var_sum < metrics1.tube.tube_var_sum

            # Status should be IMPROVED
            assert metrics2.status == "IMPROVED"


# ---------------------------------------------------------------------------
# Consistency tests (Task 5)
# ---------------------------------------------------------------------------


class TestConsistency:
    """Tests for consistency between metrics.json, summary.csv, and stdout."""

    def test_summary_csv_delta_equals_prev_minus_current(self):
        """Parse summary.csv and confirm delta_prev equals prev - current for all rounds after first."""
        from loop_metrics import compute_round_metrics
        from loop_io import save_json, append_to_summary_csv, load_summary_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            rounds_dir = base_dir / "rounds"
            rounds_dir.mkdir()

            # Create 3 rounds with known tube_var values
            tube_vars = [100.0, 80.0, 90.0]  # R1=100, R2=80 (improved), R3=90 (regressed)
            n_points = 100

            prev_tube_var = None
            baseline_tube_var = None

            for i, tv in enumerate(tube_vars):
                round_num = i + 1
                round_id = f"R000{round_num}_20260101_10000{round_num}"
                round_dir = rounds_dir / round_id
                round_dir.mkdir()

                # Create artifacts with specific variance
                # Beta variance = alpha*beta / ((a+b)^2 * (a+b+1))
                # For a=b=1, var = 1/(4*3) = 0.0833 per point
                # We want total tube_var ~ tv, so scale appropriately
                alpha = np.ones(n_points) * 1.0
                beta = np.ones(n_points) * 1.0
                np.savez(round_dir / "beta_posteriors.npz", alpha=alpha, beta=beta)

                # Set q_enn so ~20 points are in tube [0.4, 0.6]
                q_enn = np.linspace(0, 1, n_points)
                np.savez(round_dir / "enn.npz", q_enn=q_enn)

                metrics = compute_round_metrics(
                    round_dir=round_dir,
                    round_id=round_id,
                    seed=42 + i,
                    prev_tube_var=prev_tube_var,
                    baseline_tube_var=baseline_tube_var,
                )
                
                # Override tube_var_sum for testing
                metrics.tube.tube_var_sum = tv
                if prev_tube_var is not None:
                    metrics.tube.tube_var_delta = prev_tube_var - tv
                    metrics.tube.tube_var_delta_prev = prev_tube_var - tv
                else:
                    metrics.tube.tube_var_delta = None
                    metrics.tube.tube_var_delta_prev = None
                    
                if baseline_tube_var is not None:
                    metrics.tube.tube_var_delta_baseline = baseline_tube_var - tv
                else:
                    baseline_tube_var = tv
                    metrics.tube.tube_var_baseline = tv
                    metrics.tube.tube_var_delta_baseline = 0.0

                save_json(metrics.to_dict(), round_dir / "metrics.json")
                append_to_summary_csv(base_dir, metrics)

                prev_tube_var = tv

            # Now verify summary.csv
            rows = load_summary_csv(base_dir)
            assert len(rows) == 3

            # R1: delta should be empty (first round)
            assert rows[0]['tube_var_delta_prev'] == ''

            # R2: delta = 100 - 80 = 20
            delta_r2 = float(rows[1]['tube_var_delta_prev'])
            expected_r2 = 100.0 - 80.0
            assert abs(delta_r2 - expected_r2) < 0.001, f"R2 delta: {delta_r2} != {expected_r2}"

            # R3: delta = 80 - 90 = -10
            delta_r3 = float(rows[2]['tube_var_delta_prev'])
            expected_r3 = 80.0 - 90.0
            assert abs(delta_r3 - expected_r3) < 0.001, f"R3 delta: {delta_r3} != {expected_r3}"

    def test_metrics_json_matches_summary_csv(self):
        """Confirm metrics.json fields match corresponding summary.csv row."""
        from loop_metrics import compute_round_metrics
        from loop_io import save_json, append_to_summary_csv, load_summary_csv, load_json

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            rounds_dir = base_dir / "rounds"
            rounds_dir.mkdir()

            n_points = 100
            round_id = "R0001_20260101_100000"
            round_dir = rounds_dir / round_id
            round_dir.mkdir()

            alpha = np.ones(n_points) * 2.0
            beta = np.ones(n_points) * 2.0
            np.savez(round_dir / "beta_posteriors.npz", alpha=alpha, beta=beta)
            q_enn = np.linspace(0, 1, n_points)
            np.savez(round_dir / "enn.npz", q_enn=q_enn)

            metrics = compute_round_metrics(
                round_dir=round_dir,
                round_id=round_id,
                seed=42,
                prev_tube_var=None,
            )

            save_json(metrics.to_dict(), round_dir / "metrics.json")
            append_to_summary_csv(base_dir, metrics)

            # Load both and compare
            metrics_json = load_json(round_dir / "metrics.json")
            csv_rows = load_summary_csv(base_dir)
            csv_row = csv_rows[0]

            # Compare key fields
            assert csv_row['round_id'] == metrics_json['round_id']
            assert int(csv_row['seed']) == metrics_json['seed']
            assert abs(float(csv_row['tube_var']) - metrics_json['tube']['tube_var_sum']) < 1e-6
            assert abs(float(csv_row['tube_coverage']) - metrics_json['tube']['tube_coverage']) < 1e-6
            assert csv_row['status'] == metrics_json['status']

            # For first round, delta should be empty in CSV and null in JSON
            assert csv_row['tube_var_delta_prev'] == ''
            assert metrics_json['tube']['tube_var_delta_prev'] is None

    def test_new_round_dir_does_not_cause_zero_delta(self):
        """Regression test: creating new empty round dir must not cause delta to become 0."""
        from loop_io import get_previous_round_info

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            rounds_dir = base_dir / "rounds"
            rounds_dir.mkdir()

            # Create R0001 with valid metrics
            round1 = rounds_dir / "R0001_20260101_100000"
            round1.mkdir()
            metrics1 = {
                "tube": {"tube_var_sum": 50.0, "tube_coverage": 0.5},
                "counts": {"sampled_points_total": 100},
            }
            (round1 / "metrics.json").write_text(json.dumps(metrics1))

            # Create R0002 directory but NO metrics.json (simulating in-progress round)
            round2 = rounds_dir / "R0002_20260101_100100"
            round2.mkdir()
            # Intentionally no metrics.json!

            # When looking for previous round for R0002, should find R0001 (not R0002)
            prev_info = get_previous_round_info(
                base_dir, 
                exclude_round_id="R0002_20260101_100100",
                current_round_num=2
            )

            assert prev_info is not None, "Should find R0001 as previous"
            assert prev_info["round_id"] == "R0001_20260101_100000"
            assert prev_info["tube_var"] == 50.0

            # Also test without exclude_round_id but with current_round_num
            prev_info2 = get_previous_round_info(base_dir, current_round_num=2)
            assert prev_info2 is not None
            assert prev_info2["round_id"] == "R0001_20260101_100000"

    def test_status_is_string_enum_not_boolean(self):
        """Status must be a string enum, never a boolean."""
        from loop_metrics import determine_status

        # Test all status values
        status_first = determine_status(None, 0.5, 0.0)
        status_improved = determine_status(0.01, 0.5, 0.5)
        status_regressed = determine_status(-0.02, 0.5, 0.5)
        status_no_change = determine_status(0.0001, 0.5, 0.5)

        # All must be strings
        assert isinstance(status_first, str)
        assert isinstance(status_improved, str)
        assert isinstance(status_regressed, str)
        assert isinstance(status_no_change, str)

        # All must be valid enum values
        valid_statuses = {"FIRST_ROUND", "IMPROVED", "REGRESSED", "NO_CHANGE"}
        assert status_first in valid_statuses
        assert status_improved in valid_statuses
        assert status_regressed in valid_statuses
        assert status_no_change in valid_statuses

        # None of them should be boolean-like strings
        assert status_first not in {"True", "False", "true", "false"}
        assert status_improved not in {"True", "False", "true", "false"}


class TestBaselinePersistence:
    """Tests for baseline persistence across rounds."""

    def test_baseline_propagates_through_rounds(self):
        """Baseline from R0001 should propagate to all subsequent rounds."""
        from loop_metrics import compute_tube_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            n_points = 100
            alpha = np.ones(n_points) * 2.0
            beta = np.ones(n_points) * 2.0
            np.savez(run_dir / "beta_posteriors.npz", alpha=alpha, beta=beta)
            q_enn = np.linspace(0, 1, n_points)
            np.savez(run_dir / "enn.npz", q_enn=q_enn)

            # R1: First round sets baseline (but delta_baseline is None per spec)
            metrics1 = compute_tube_metrics(run_dir, prev_tube_var=None, baseline_tube_var=None)
            assert metrics1.tube_var_baseline == metrics1.tube_var_sum
            assert metrics1.tube_var_delta_baseline is None  # FIRST_ROUND has no delta
            baseline = metrics1.tube_var_baseline

            # R2: Should use R1's baseline, and have a computed delta_baseline
            metrics2 = compute_tube_metrics(run_dir, prev_tube_var=metrics1.tube_var_sum, baseline_tube_var=baseline)
            assert metrics2.tube_var_baseline == baseline
            assert metrics2.tube_var_delta_baseline is not None  # Second round has delta
            
            # R3: Should still use R1's baseline
            metrics3 = compute_tube_metrics(run_dir, prev_tube_var=metrics2.tube_var_sum, baseline_tube_var=baseline)
            assert metrics3.tube_var_baseline == baseline
            assert metrics3.tube_var_delta_baseline is not None


class TestAtomicWrites:
    """Tests for atomic JSON write behavior."""

    def test_atomic_write_no_partial_file(self):
        """save_json atomic mode should never leave partial file at target."""
        from loop_io import save_json
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            data = {"key": "value", "number": 42}
            
            # Normal write should succeed
            save_json(data, path, atomic=True)
            assert path.exists()
            
            # Read it back
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == data

    def test_atomic_write_failure_leaves_old_data(self):
        """If atomic write fails after tmp write but before rename, old data preserved."""
        from loop_io import save_json
        import unittest.mock as mock
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            
            # Write initial data
            old_data = {"version": 1}
            save_json(old_data, path, atomic=True)
            
            # Try to write new data, but mock os.replace to fail
            new_data = {"version": 2}
            
            original_replace = os.replace
            def failing_replace(src, dst):
                raise OSError("Simulated disk failure")
            
            with mock.patch('os.replace', failing_replace):
                try:
                    save_json(new_data, path, atomic=True)
                except OSError:
                    pass
            
            # Original file should still have old data (not corrupted)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded == old_data, "Old data should be preserved on failure"

    def test_no_tmp_files_left_after_success(self):
        """Successful atomic write should not leave .tmp files."""
        from loop_io import save_json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_json({"key": "value"}, path, atomic=True)
            
            # Check no .tmp files left
            tmp_files = list(Path(tmpdir).glob("*.tmp*"))
            assert len(tmp_files) == 0, f"Leftover tmp files: {tmp_files}"


class TestFirstRoundDeltas:
    """Tests for FIRST_ROUND delta semantics."""

    def test_first_round_deltas_are_none(self):
        """FIRST_ROUND should have delta_prev=None AND delta_base=None."""
        from loop_metrics import compute_tube_metrics
        
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            n_points = 100
            
            alpha = np.ones(n_points) * 2.0
            beta = np.ones(n_points) * 2.0
            np.savez(run_dir / "beta_posteriors.npz", alpha=alpha, beta=beta)
            q_enn = np.linspace(0, 1, n_points)
            np.savez(run_dir / "enn.npz", q_enn=q_enn)
            
            # First round - no prev, no baseline
            metrics = compute_tube_metrics(run_dir, prev_tube_var=None, baseline_tube_var=None)
            
            # Both deltas should be None
            assert metrics.tube_var_delta is None, "FIRST_ROUND delta should be None"
            assert metrics.tube_var_delta_prev is None, "FIRST_ROUND delta_prev should be None"
            assert metrics.tube_var_delta_baseline is None, "FIRST_ROUND delta_baseline should be None"
            
            # But baseline value should be stored for future rounds
            assert metrics.tube_var_baseline is not None
            assert metrics.tube_var_baseline == metrics.tube_var_sum

    def test_second_round_uses_baseline_correctly(self):
        """Second round should have delta_prev and delta_base computed correctly."""
        from loop_metrics import compute_tube_metrics
        
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            n_points = 100
            
            alpha = np.ones(n_points) * 2.0
            beta = np.ones(n_points) * 2.0
            np.savez(run_dir / "beta_posteriors.npz", alpha=alpha, beta=beta)
            q_enn = np.linspace(0, 1, n_points)
            np.savez(run_dir / "enn.npz", q_enn=q_enn)
            
            # First round
            m1 = compute_tube_metrics(run_dir, prev_tube_var=None, baseline_tube_var=None)
            baseline = m1.tube_var_baseline
            
            # Simulate different variance for round 2
            alpha2 = np.ones(n_points) * 5.0  # More samples = lower variance
            beta2 = np.ones(n_points) * 5.0
            np.savez(run_dir / "beta_posteriors.npz", alpha=alpha2, beta=beta2)
            
            # Second round - with prev and baseline
            m2 = compute_tube_metrics(
                run_dir, 
                prev_tube_var=m1.tube_var_sum, 
                baseline_tube_var=baseline
            )
            
            # Both deltas should be computed
            assert m2.tube_var_delta_prev is not None
            assert m2.tube_var_delta_baseline is not None
            
            # Delta = prev - current (positive means improvement)
            expected_delta = m1.tube_var_sum - m2.tube_var_sum
            assert abs(m2.tube_var_delta_prev - expected_delta) < 1e-6


class TestRunInitialization:
    """Tests for run directory initialization."""

    def test_init_creates_beta_posteriors(self):
        """init_run_dir creates beta_posteriors.npz if missing."""
        from loop_io import init_run_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "new_run"
            
            # Should not exist yet
            assert not run_dir.exists()
            
            # Initialize
            info = init_run_dir(run_dir, base_seed=42, seed_mode="increment")
            
            # Should now exist
            assert run_dir.exists()
            assert (run_dir / "rounds").exists()
            assert (run_dir / "beta_posteriors.npz").exists()
            assert (run_dir / "run_metadata.json").exists()
            
            # Check metadata
            with open(run_dir / "run_metadata.json") as f:
                metadata = json.load(f)
            assert "run_uuid" in metadata
            assert metadata["base_seed"] == 42
            assert metadata["seed_mode"] == "increment"
            
            # Check beta_posteriors has correct shape
            data = np.load(run_dir / "beta_posteriors.npz")
            assert "alpha" in data.files
            assert "beta" in data.files
            # Default is 4000 points, with Beta(1,1) priors
            assert np.all(data["alpha"] == 1.0)
            assert np.all(data["beta"] == 1.0)

    def test_init_preserves_existing_run_uuid(self):
        """init_run_dir preserves run_uuid on resume."""
        from loop_io import init_run_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            # First init
            info1 = init_run_dir(run_dir, base_seed=42, seed_mode="increment")
            uuid1 = info1["run_uuid"]
            
            # Second init (simulating resume)
            info2 = init_run_dir(run_dir, base_seed=42, seed_mode="increment")
            uuid2 = info2["run_uuid"]
            
            # UUID should be preserved
            assert uuid1 == uuid2


class TestResumeLogic:
    """Tests for crash/resume behavior."""

    def test_get_next_round_num_empty(self):
        """get_next_round_num returns 1 for empty run dir."""
        from loop_io import get_next_round_num

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "rounds").mkdir()
            
            assert get_next_round_num(run_dir) == 1

    def test_get_next_round_num_with_complete_rounds(self):
        """get_next_round_num returns max + 1 for complete rounds."""
        from loop_io import get_next_round_num

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            rounds_dir = run_dir / "rounds"
            rounds_dir.mkdir()
            
            # Create R0001 with metrics.json (complete)
            r1 = rounds_dir / "R0001_20260101_100000"
            r1.mkdir()
            (r1 / "metrics.json").write_text('{"tube": {}}')
            
            # Create R0002 with metrics.json (complete)
            r2 = rounds_dir / "R0002_20260101_100100"
            r2.mkdir()
            (r2 / "metrics.json").write_text('{"tube": {}}')
            
            # Create R0003 WITHOUT metrics.json (incomplete)
            r3 = rounds_dir / "R0003_20260101_100200"
            r3.mkdir()
            # No metrics.json!
            
            # Should return 3 (max complete is 2)
            assert get_next_round_num(run_dir) == 3

    def test_duplicate_round_id_raises_error(self):
        """append_to_summary_csv should raise error on duplicate round_ids."""
        from loop_metrics import compute_round_metrics
        from loop_io import append_to_summary_csv, load_summary_csv, init_run_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            init_run_dir(run_dir, base_seed=42, seed_mode="increment")
            rounds_dir = run_dir / "rounds"
            
            n_points = 100
            round_id = "R0001_20260101_100000"
            round_dir = rounds_dir / round_id
            round_dir.mkdir()
            
            alpha = np.ones(n_points) * 2.0
            beta = np.ones(n_points) * 2.0
            np.savez(round_dir / "beta_posteriors.npz", alpha=alpha, beta=beta)
            q_enn = np.linspace(0, 1, n_points)
            np.savez(round_dir / "enn.npz", q_enn=q_enn)
            
            metrics = compute_round_metrics(
                round_dir=round_dir,
                round_id=round_id,
                seed=42,
            )
            
            # Append once - should succeed
            append_to_summary_csv(run_dir, metrics, run_uuid="test-uuid")
            rows1 = load_summary_csv(run_dir)
            assert len(rows1) == 1
            
            # Append again with same round_id - should raise error
            try:
                append_to_summary_csv(run_dir, metrics, run_uuid="test-uuid")
                assert False, "Should have raised ValueError for duplicate round_id"
            except ValueError as e:
                assert "duplicate round_id" in str(e).lower()
            
            # Verify CSV unchanged
            rows2 = load_summary_csv(run_dir)
            assert len(rows2) == 1  # Still 1
            
            # Verify the row has run_uuid
            assert rows2[0].get("run_uuid") == "test-uuid"


if __name__ == "__main__":
    if pytest:
        pytest.main([__file__, "-v"])
    else:
        # Run tests manually without pytest
        print("Running tests without pytest...")
        
        # Run consistency tests
        t = TestConsistency()
        print("test_summary_csv_delta_equals_prev_minus_current...", end=" ")
        try:
            t.test_summary_csv_delta_equals_prev_minus_current()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        print("test_metrics_json_matches_summary_csv...", end=" ")
        try:
            t.test_metrics_json_matches_summary_csv()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        print("test_new_round_dir_does_not_cause_zero_delta...", end=" ")
        try:
            t.test_new_round_dir_does_not_cause_zero_delta()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        print("test_status_is_string_enum_not_boolean...", end=" ")
        try:
            t.test_status_is_string_enum_not_boolean()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        # Run baseline tests
        t2 = TestBaselinePersistence()
        print("test_baseline_propagates_through_rounds...", end=" ")
        try:
            t2.test_baseline_propagates_through_rounds()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        # Run initialization tests
        t3 = TestRunInitialization()
        print("test_init_creates_beta_posteriors...", end=" ")
        try:
            t3.test_init_creates_beta_posteriors()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        print("test_init_preserves_existing_run_uuid...", end=" ")
        try:
            t3.test_init_preserves_existing_run_uuid()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        # Run resume tests
        t4 = TestResumeLogic()
        print("test_get_next_round_num_empty...", end=" ")
        try:
            t4.test_get_next_round_num_empty()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        print("test_get_next_round_num_with_complete_rounds...", end=" ")
        try:
            t4.test_get_next_round_num_with_complete_rounds()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        print("test_duplicate_round_id_raises_error...", end=" ")
        try:
            t4.test_duplicate_round_id_raises_error()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        # Run atomic write tests
        t5 = TestAtomicWrites()
        print("test_atomic_write_no_partial_file...", end=" ")
        try:
            t5.test_atomic_write_no_partial_file()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        print("test_atomic_write_failure_leaves_old_data...", end=" ")
        try:
            t5.test_atomic_write_failure_leaves_old_data()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        print("test_no_tmp_files_left_after_success...", end=" ")
        try:
            t5.test_no_tmp_files_left_after_success()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        # Run first round delta tests
        t6 = TestFirstRoundDeltas()
        print("test_first_round_deltas_are_none...", end=" ")
        try:
            t6.test_first_round_deltas_are_none()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
            
        print("test_second_round_uses_baseline_correctly...", end=" ")
        try:
            t6.test_second_round_uses_baseline_correctly()
            print("PASSED")
        except Exception as e:
            print(f"FAILED: {e}")
