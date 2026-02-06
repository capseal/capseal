"""End-to-end tests for capseal eval command in synthetic mode."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


class TestEvalSyntheticEndToEnd:
    """End-to-end tests for the eval command with synthetic episodes."""

    @pytest.fixture
    def capseal_cmd(self):
        """Return the capseal command path."""
        # Use the installed capseal or fall back to direct Python invocation
        return [sys.executable, "-m", "capseal_cli.main"]

    def test_eval_help(self, capseal_cmd):
        """Eval command shows help."""
        result = subprocess.run(
            capseal_cmd + ["eval", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Learn which patches fail" in result.stdout

    def test_eval_synthetic_creates_artifacts(self, tmp_project):
        """Synthetic eval creates expected artifacts."""
        from bef_zk.capsule.cli.eval_cmd import eval_command
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(eval_command, [
            str(tmp_project),
            "--rounds", "2",
            "--synthetic",
            "--targets-per-round", "8",
            "--seed", "42",
        ])

        assert result.exit_code == 0, f"Failed: {result.output}"

        # Check artifacts exist
        capseal_dir = tmp_project / ".capseal"
        assert capseal_dir.exists()

        models_dir = capseal_dir / "models"
        assert models_dir.exists()
        assert (models_dir / "beta_posteriors.npz").exists()

        # Find the run directory
        runs_dir = capseal_dir / "runs"
        run_dirs = list(runs_dir.glob("*-eval"))
        assert len(run_dirs) == 1

        run_dir = run_dirs[0]
        assert (run_dir / "run_receipt.json").exists()
        assert (run_dir / "summary.csv").exists()
        assert (run_dir / "grid.npz").exists()

        # Check rounds
        rounds_dir = run_dir / "rounds"
        round_dirs = list(rounds_dir.iterdir())
        assert len(round_dirs) == 2

    def test_eval_posteriors_are_updated(self, tmp_project):
        """Posteriors are updated after each round."""
        from bef_zk.capsule.cli.eval_cmd import eval_command
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(eval_command, [
            str(tmp_project),
            "--rounds", "3",
            "--synthetic",
            "--targets-per-round", "16",
            "--seed", "123",
        ])

        assert result.exit_code == 0

        # Load posteriors
        posteriors_path = tmp_project / ".capseal" / "models" / "beta_posteriors.npz"
        data = np.load(posteriors_path)
        alpha = data["alpha"]
        beta = data["beta"]

        # After 3 rounds with 16 targets each, we should have updates
        # Initial: sum = 1024 each. After episodes: sum > 1024
        assert alpha.sum() > 1024 or beta.sum() > 1024

    def test_eval_receipts_verify(self, tmp_project):
        """Generated receipts pass verification."""
        from bef_zk.capsule.cli.eval_cmd import eval_command
        from bef_zk.shared.receipts import verify_run_receipt
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(eval_command, [
            str(tmp_project),
            "--rounds", "2",
            "--synthetic",
            "--seed", "999",
        ])

        assert result.exit_code == 0

        # Find run directory
        runs_dir = tmp_project / ".capseal" / "runs"
        run_dirs = list(runs_dir.glob("*-eval"))
        run_dir = run_dirs[0]

        # Verify
        verification = verify_run_receipt(run_dir)
        assert verification["verified"] is True

    def test_eval_deterministic_with_seed(self, tmp_project):
        """Same seed produces consistent structure and similar results."""
        from bef_zk.capsule.cli.eval_cmd import eval_command
        from click.testing import CliRunner
        import shutil

        runner = CliRunner()

        # First run
        result1 = runner.invoke(eval_command, [
            str(tmp_project),
            "--rounds", "1",
            "--synthetic",
            "--targets-per-round", "4",
            "--seed", "42",
        ])
        assert result1.exit_code == 0

        # Save posteriors
        p1 = np.load(tmp_project / ".capseal" / "models" / "beta_posteriors.npz")
        alpha1 = p1["alpha"].copy()
        beta1 = p1["beta"].copy()

        # Clean up for second run
        shutil.rmtree(tmp_project / ".capseal")

        # Second run with same seed
        result2 = runner.invoke(eval_command, [
            str(tmp_project),
            "--rounds", "1",
            "--synthetic",
            "--targets-per-round", "4",
            "--seed", "42",
        ])
        assert result2.exit_code == 0

        p2 = np.load(tmp_project / ".capseal" / "models" / "beta_posteriors.npz")
        alpha2 = p2["alpha"]
        beta2 = p2["beta"]

        # Should have same structure (same shape, same sparsity pattern)
        assert alpha1.shape == alpha2.shape
        assert beta1.shape == beta2.shape
        # Total updates should be similar (within tolerance for any iteration order variance)
        assert abs(alpha1.sum() - alpha2.sum()) <= 2
        assert abs(beta1.sum() - beta2.sum()) <= 2
