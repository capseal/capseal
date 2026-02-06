#!/usr/bin/env python3
"""Tests for AgentEvalBench v1.

Test cases:
1. test_fresh_run_creates_required_files - Verify all artifacts created
2. test_delta_prev_matches_prev_minus_current - Delta math invariant
3. test_incomplete_round_is_skipped - Resume safety
4. test_resume_continues_round_numbers - No duplicate R0001
5. test_reproducible_summary_given_seed - Determinism check
6. test_replay_episode_seeds_reproduces_results - Replay audit
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_bench.grid import generate_grid, save_grid, get_params_for_idx
from agent_bench.env_toy_v1 import ToyToolEnv
from agent_bench.agent_toy_v1 import ToyAgent
from agent_bench.runner import (
    derive_episode_seed,
    derive_agent_seed,
    run_episode,
    compute_acquisition_score,
    select_targets,
    synthetic_p_fail,
)
from agent_bench.metrics import compute_agent_tube_metrics, determine_status


class TestFreshRunCreatesRequiredFiles:
    """Test that a fresh run creates all required artifacts."""
    
    def test_fresh_run_creates_required_files(self):
        """Verify all artifacts created after one round."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            # Create grid
            save_grid(run_dir / "grid.npz")
            
            # Run one round (synthetic for speed)
            from agent_bench.runner import run_agent_eval_loop
            run_agent_eval_loop(
                run_dir=run_dir,
                n_rounds=1,
                base_seed=12345,
                seed_mode="increment",
                agent_bench="toy_v1",
                episodes_per_budget_unit=1,
                targets_per_round=4,  # Small for test
                use_synthetic=True,
                verbose=False,
            )
            
            # Check required files exist
            assert (run_dir / "run_metadata.json").exists(), "Missing run_metadata.json"
            assert (run_dir / "beta_posteriors.npz").exists(), "Missing beta_posteriors.npz"
            assert (run_dir / "summary.csv").exists(), "Missing summary.csv"
            assert (run_dir / "rounds").is_dir(), "Missing rounds directory"
            
            # Check round directory
            round_dirs = list((run_dir / "rounds").iterdir())
            assert len(round_dirs) == 1, f"Expected 1 round dir, got {len(round_dirs)}"
            
            round_dir = round_dirs[0]
            assert (round_dir / "metrics.json").exists(), "Missing metrics.json"
            assert (round_dir / "agent_results.csv").exists(), "Missing agent_results.csv"
            assert (round_dir / "active_sampling_plan.json").exists(), "Missing plan"
            assert (round_dir / "round_pre.json").exists(), "Missing round_pre.json"
            assert (round_dir / "round_post.json").exists(), "Missing round_post.json"


class TestDeltaPrevMatchesPrevMinusCurrent:
    """Test delta math invariant: delta = prev - current."""
    
    def test_delta_prev_matches_prev_minus_current(self):
        """Verify delta is computed as prev_tube_var - current_tube_var."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            # Create grid
            save_grid(run_dir / "grid.npz")
            
            # Run two rounds
            from agent_bench.runner import run_agent_eval_loop
            run_agent_eval_loop(
                run_dir=run_dir,
                n_rounds=2,
                base_seed=12345,
                seed_mode="increment",
                agent_bench="toy_v1",
                episodes_per_budget_unit=1,
                targets_per_round=4,
                use_synthetic=True,
                verbose=False,
            )
            
            # Load metrics from both rounds
            round_dirs = sorted((run_dir / "rounds").iterdir())
            assert len(round_dirs) == 2, f"Expected 2 rounds, got {len(round_dirs)}"
            
            with open(round_dirs[0] / "metrics.json") as f:
                m1 = json.load(f)
            with open(round_dirs[1] / "metrics.json") as f:
                m2 = json.load(f)
            
            # Verify delta math
            tube_var_1 = m1["tube"]["tube_var_sum"]
            tube_var_2 = m2["tube"]["tube_var_sum"]
            expected_delta = tube_var_1 - tube_var_2
            actual_delta = m2["tube"]["tube_var_delta_prev"]
            
            assert abs(expected_delta - actual_delta) < 1e-9, \
                f"Delta mismatch: expected {expected_delta}, got {actual_delta}"


class TestIncompleteRoundIsSkipped:
    """Test that incomplete rounds are skipped on resume."""
    
    def test_incomplete_round_is_skipped(self):
        """Verify resume skips rounds without metrics.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            # Create grid
            save_grid(run_dir / "grid.npz")
            
            # Run one round
            from agent_bench.runner import run_agent_eval_loop
            run_agent_eval_loop(
                run_dir=run_dir,
                n_rounds=1,
                base_seed=12345,
                seed_mode="increment",
                targets_per_round=4,
                use_synthetic=True,
            )
            
            # Create an incomplete round directory
            incomplete_dir = run_dir / "rounds" / "R0002_incomplete"
            incomplete_dir.mkdir(parents=True)
            (incomplete_dir / "round_pre.json").write_text("{}")
            # No metrics.json - incomplete!
            
            # Get next round number - should skip incomplete
            from loop_io import get_next_round_num
            next_num = get_next_round_num(run_dir)
            assert next_num == 2, f"Expected next round 2, got {next_num}"


class TestResumeContinuesRoundNumbers:
    """Test that resume continues from correct round number."""
    
    def test_resume_continues_round_numbers(self):
        """Verify no duplicate R0001 after resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            
            # Create grid
            save_grid(run_dir / "grid.npz")
            
            # Run one round
            from agent_bench.runner import run_agent_eval_loop
            run_agent_eval_loop(
                run_dir=run_dir,
                n_rounds=1,
                base_seed=12345,
                seed_mode="increment",
                targets_per_round=4,
                use_synthetic=True,
            )
            
            # Run another round (resume)
            run_agent_eval_loop(
                run_dir=run_dir,
                n_rounds=1,
                base_seed=12345,
                seed_mode="increment",
                targets_per_round=4,
                use_synthetic=True,
            )
            
            # Check that we have R0001 and R0002, not two R0001s
            round_dirs = list((run_dir / "rounds").iterdir())
            round_nums = set()
            for d in round_dirs:
                if d.name.startswith("R"):
                    num = int(d.name.split("_")[0][1:])
                    round_nums.add(num)
            
            assert round_nums == {1, 2}, f"Expected rounds {{1, 2}}, got {round_nums}"


class TestReproducibleSummaryGivenSeed:
    """Test determinism: same run_uuid + seed produces same results."""
    
    def test_reproducible_summary_given_seed(self):
        """Verify same run_uuid produces identical results when replayed.
        
        Note: Different runs have different run_uuids (by design), so we test
        determinism by ensuring the same episode_seed produces the same outcome.
        This is what the replay audit test validates.
        
        Here we test that acquisition selection is deterministic.
        """
        # Test that acquisition selection is deterministic
        alpha = np.ones(1024)
        beta = np.ones(1024)
        
        scores1 = compute_acquisition_score(alpha, beta)
        scores2 = compute_acquisition_score(alpha, beta)
        
        assert np.allclose(scores1, scores2), "Acquisition scores not deterministic"
        
        selected1 = select_targets(scores1, 10)
        selected2 = select_targets(scores2, 10)
        
        assert list(selected1) == list(selected2), \
            f"Target selection not deterministic: {list(selected1)} != {list(selected2)}"
    
    def test_episode_determinism_with_fixed_seed(self):
        """Verify same episode_seed always produces same result."""
        grid = generate_grid()
        
        # Run same episode twice with same seed
        results = []
        for _ in range(2):
            episode_seed = 12345678901234
            success = run_episode(grid, grid_idx=0, episode_seed=episode_seed, use_synthetic=False)
            results.append(success)
        
        assert results[0] == results[1], \
            f"Episode not deterministic with same seed"


class TestReplayEpisodeSeedsReproducesResults:
    """Test replay audit: stored seeds can reproduce results."""
    
    def test_replay_episode_seeds_reproduces_results(self):
        """Verify episodes can be replayed from stored seeds."""
        import csv
        
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            save_grid(run_dir / "grid.npz")
            grid = generate_grid()
            
            # Run one round (real, not synthetic)
            from agent_bench.runner import run_agent_eval_loop
            run_agent_eval_loop(
                run_dir=run_dir,
                n_rounds=1,
                base_seed=12345,
                seed_mode="increment",
                targets_per_round=4,
                use_synthetic=False,  # Real simulation
                episodes_per_budget_unit=1,
            )
            
            # Read first 10 rows of agent_results.csv
            round_dirs = list((run_dir / "rounds").iterdir())
            results_path = round_dirs[0] / "agent_results.csv"
            
            with open(results_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)[:10]
            
            # Replay each episode and verify result matches
            for row in rows:
                grid_idx = int(row["grid_idx"])
                episode_seed = int(row["episode_seed"])
                stored_success = int(row["success"])
                
                # Reconstruct env and run agent
                params = get_params_for_idx(grid, grid_idx)
                env_rng = np.random.default_rng(episode_seed)
                agent_rng = np.random.default_rng(derive_agent_seed(episode_seed))
                
                env = ToyToolEnv(
                    tool_noise=params["tool_noise"],
                    verify_flip=params["verify_flip"],
                    hint_ambiguity=params["hint_ambiguity"],
                    distractor_count=params["distractor_count"],
                    memory_tokens=params["memory_tokens"],
                    rng=env_rng,
                )
                
                agent = ToyAgent()
                guess = agent.act(env, agent_rng)
                replayed_success = 1 if env.check_answer(guess) else 0
                
                assert replayed_success == stored_success, \
                    f"Replay mismatch at grid_idx={grid_idx}, episode_seed={episode_seed}: " \
                    f"stored={stored_success}, replayed={replayed_success}"


class TestUnitFunctions:
    """Unit tests for individual functions."""
    
    def test_derive_episode_seed_deterministic(self):
        """Verify episode seed derivation is deterministic."""
        seed1 = derive_episode_seed("abc123", 1, 0, 0)
        seed2 = derive_episode_seed("abc123", 1, 0, 0)
        assert seed1 == seed2, "Episode seed not deterministic"
        
        # Different inputs produce different seeds
        seed3 = derive_episode_seed("abc123", 1, 0, 1)
        assert seed1 != seed3, "Different episodes have same seed"
    
    def test_derive_agent_seed_different_from_episode(self):
        """Verify agent seed is different from episode seed."""
        episode_seed = derive_episode_seed("abc123", 1, 0, 0)
        agent_seed = derive_agent_seed(episode_seed)
        assert episode_seed != agent_seed, "Agent seed same as episode seed"
    
    def test_select_targets_deterministic_tiebreak(self):
        """Verify target selection has deterministic tie-breaking."""
        # All equal scores - should select by index order
        scores = np.ones(10)
        selected = select_targets(scores, 5)
        assert list(selected) == [0, 1, 2, 3, 4], \
            f"Tie-break not deterministic: {list(selected)}"
    
    def test_synthetic_p_fail_range(self):
        """Verify synthetic p_fail is in [0, 1]."""
        for _ in range(100):
            p = synthetic_p_fail(
                tool_noise=np.random.randint(0, 4),
                verify_flip=np.random.choice([0.0, 0.05, 0.1, 0.2]),
                hint_ambiguity=np.random.randint(0, 4),
                distractor_count=np.random.choice([0, 2, 4, 6]),
                memory_tokens=np.random.choice([16, 32, 64, 128]),
            )
            assert 0.0 <= p <= 1.0, f"p_fail out of range: {p}"
    
    def test_determine_status(self):
        """Verify status determination logic."""
        assert determine_status(None, 1) == "FIRST_ROUND"
        assert determine_status(0.1, 2) == "IMPROVED"  # positive delta = improvement
        assert determine_status(-0.1, 2) == "REGRESSED"  # negative delta = regression
        assert determine_status(0.0, 2) == "NO_CHANGE"
    
    def test_grid_has_1024_points(self):
        """Verify grid has exactly 1024 points."""
        grid = generate_grid()
        assert int(grid["n_points"]) == 1024, f"Expected 1024 points, got {grid['n_points']}"
    
    def test_acquisition_score_shape(self):
        """Verify acquisition score has correct shape."""
        alpha = np.ones(1024)
        beta = np.ones(1024)
        scores = compute_acquisition_score(alpha, beta)
        assert scores.shape == (1024,), f"Wrong shape: {scores.shape}"


def run_all_tests():
    """Run all tests and report results."""
    test_classes = [
        TestFreshRunCreatesRequiredFiles,
        TestDeltaPrevMatchesPrevMinusCurrent,
        TestIncompleteRoundIsSkipped,
        TestResumeContinuesRoundNumbers,
        TestReproducibleSummaryGivenSeed,
        TestReplayEpisodeSeedsReproducesResults,
        TestUnitFunctions,
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for cls in test_classes:
        instance = cls()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    print(f"Running {cls.__name__}.{method_name}...", end=" ")
                    getattr(instance, method_name)()
                    print("PASSED")
                    passed += 1
                except AssertionError as e:
                    print(f"FAILED: {e}")
                    failed += 1
                    errors.append((f"{cls.__name__}.{method_name}", str(e)))
                except Exception as e:
                    print(f"ERROR: {e}")
                    failed += 1
                    errors.append((f"{cls.__name__}.{method_name}", str(e)))
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    
    if errors:
        print("\nFailures:")
        for name, error in errors:
            print(f"  {name}: {error}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
