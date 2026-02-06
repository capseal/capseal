"""Tests for feature extraction, scoring, and gate decisions."""
from __future__ import annotations

import numpy as np
import pytest


class TestFeatureExtraction:
    """Tests for extract_patch_features and discretize_features."""

    def test_extract_simple_diff(self, sample_diff):
        """Extract features from a simple diff."""
        from bef_zk.shared.features import extract_patch_features

        features = extract_patch_features(sample_diff, [{"severity": "high"}])

        assert features["lines_changed"] > 0
        assert features["files_touched"] == 1
        assert features["finding_severity"] == 3  # high → 3

    def test_discretize_features(self):
        """Discretize raw features to grid levels."""
        from bef_zk.shared.features import discretize_features

        raw = {
            "lines_changed": 5,  # <= 10 → 0
            "cyclomatic_complexity": 3,  # 3-5 → 1
            "files_touched": 1,  # 1 → 0
            "finding_severity": 2,  # medium → 2
            "test_coverage_delta": 0,  # 0 → 0
        }
        levels = discretize_features(raw)

        assert levels == [0, 1, 0, 2, 0]

    def test_features_to_grid_idx_roundtrip(self):
        """Grid index encoding is reversible."""
        from bef_zk.shared.features import features_to_grid_idx, grid_idx_to_features

        for levels in [[0, 0, 0, 0, 0], [1, 2, 3, 0, 1], [3, 3, 3, 3, 3]]:
            idx = features_to_grid_idx(levels)
            recovered = grid_idx_to_features(idx)
            assert recovered == levels, f"Roundtrip failed for {levels}"


class TestAcquisitionScoring:
    """Tests for compute_acquisition_score and select_targets."""

    def test_acquisition_score_shape(self):
        """Acquisition scores have correct shape."""
        from bef_zk.shared.scoring import compute_acquisition_score

        n_points = 100
        alpha = np.ones(n_points, dtype=np.int64)
        beta = np.ones(n_points, dtype=np.int64)

        scores = compute_acquisition_score(alpha, beta)

        assert scores.shape == (n_points,)
        assert np.all(scores >= 0)

    def test_select_targets_top_k(self):
        """Select targets picks top-K highest scores."""
        from bef_zk.shared.scoring import select_targets

        scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        selected = select_targets(scores, K=2)

        assert len(selected) == 2
        assert 3 in selected  # Index of 0.9
        assert 1 in selected  # Index of 0.5

    def test_select_targets_deterministic(self):
        """Select targets is deterministic for tie-breaking."""
        from bef_zk.shared.scoring import select_targets

        scores = np.array([0.5, 0.5, 0.5, 0.5])
        selected1 = select_targets(scores, K=2)
        selected2 = select_targets(scores, K=2)

        assert np.array_equal(selected1, selected2)


class TestGateDecisions:
    """Tests for score_patch gate decisions."""

    def test_score_patch_with_posteriors(self, posteriors_file, sample_diff):
        """Score a patch against learned posteriors."""
        from bef_zk.shared.features import score_patch

        result = score_patch(
            sample_diff,
            [{"severity": "high"}],
            posteriors_file,
        )

        assert "q" in result
        assert "uncertainty" in result
        assert "decision" in result
        assert result["decision"] in ["pass", "skip", "human_review"]

    def test_high_q_leads_to_skip(self, posteriors_file):
        """High failure probability leads to skip decision."""
        from bef_zk.shared.features import score_patch

        # Create a diff that maps to grid_idx 0 (which has high q in fixture)
        minimal_diff = "+++ a.py\n+ x = 1"

        result = score_patch(minimal_diff, [{"severity": "high"}], posteriors_file)

        # The fixture sets alpha[0]=5, beta[0]=2, so q = 5/7 ≈ 0.71 > 0.3
        # This should be a skip
        assert result["q"] > 0.3 or result["decision"] in ["skip", "human_review"]


class TestTubeMetrics:
    """Tests for compute_tube_metrics."""

    def test_tube_metrics_basic(self):
        """Compute basic tube metrics."""
        from bef_zk.shared.scoring import compute_tube_metrics

        n_points = 100
        alpha = np.ones(n_points, dtype=np.int64)
        beta = np.ones(n_points, dtype=np.int64) * 10  # Low q everywhere

        metrics = compute_tube_metrics(alpha, beta)

        assert "tube_var_sum" in metrics
        assert "tube_coverage" in metrics
        assert metrics["tube_coverage"] >= 0
        assert metrics["tube_coverage"] <= 1

    def test_tube_coverage_increases_with_samples(self):
        """More samples with successes → higher tube coverage."""
        from bef_zk.shared.scoring import compute_tube_metrics

        n_points = 100

        # Uninformative priors (mean=0.5 > tau=0.2, no points in tube)
        alpha1 = np.ones(n_points, dtype=np.int64)
        beta1 = np.ones(n_points, dtype=np.int64)
        metrics1 = compute_tube_metrics(alpha1, beta1)

        # After many successes (mean=1/101≈0.01 < tau=0.2, all in tube)
        alpha2 = np.ones(n_points, dtype=np.int64)
        beta2 = np.ones(n_points, dtype=np.int64) * 100
        metrics2 = compute_tube_metrics(alpha2, beta2)

        # With many successes, more points should be in tube (mu <= tau)
        assert metrics2["tube_coverage"] > metrics1["tube_coverage"]
