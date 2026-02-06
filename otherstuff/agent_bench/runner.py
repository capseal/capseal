#!/usr/bin/env python3
"""AgentEvalBench Runner - Round orchestration, acquisition, and posteriors."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent_bench.env_toy_v1 import ToyToolEnv
from agent_bench.agent_toy_v1 import ToyAgent
from agent_bench.grid import load_grid, get_params_for_idx, GRID_VERSION
from agent_bench.metrics import compute_agent_tube_metrics

BUDGET_PER_TARGET = 256


def derive_episode_seed(run_uuid: str, round_num: int, grid_idx: int, episode_idx: int) -> int:
    """Deterministic seed derivation - stable across resume/machines/Python versions."""
    data = f"{run_uuid}:{round_num}:{grid_idx}:{episode_idx}".encode()
    h = hashlib.blake2b(data, digest_size=8)
    return int.from_bytes(h.digest(), 'little')


def derive_agent_seed(episode_seed: int) -> int:
    """Derive separate RNG stream for agent using blake2b with different personalization."""
    data = f"agent:{episode_seed}".encode()
    h = hashlib.blake2b(data, digest_size=8, person=b'AGENTSEED\x00\x00\x00\x00\x00\x00\x00')
    return int.from_bytes(h.digest(), 'little')


def compute_acquisition_score(
    alpha: np.ndarray,
    beta: np.ndarray,
    tau: float = 0.2,
    sigma: float = 0.05,
    w1: float = 1.0,
    w2: float = 0.5,
) -> np.ndarray:
    """Compute acquisition scores for all grid points."""
    alpha_f = alpha.astype(np.float64)
    beta_f = beta.astype(np.float64)
    total = alpha_f + beta_f
    mu = alpha_f / total
    var = (alpha_f * beta_f) / (total ** 2 * (total + 1))
    boundary_bonus = np.exp(-np.abs(mu - tau) / sigma)
    return var * w1 + boundary_bonus * w2


def select_targets(scores: np.ndarray, K: int) -> np.ndarray:
    """Select top-K targets with deterministic tie-breaking."""
    n = len(scores)
    order = np.lexsort((np.arange(n), -scores))
    return order[:min(K, n)]


def synthetic_p_fail(
    tool_noise: int,
    verify_flip: float,
    hint_ambiguity: int,
    distractor_count: int,
    memory_tokens: int,
) -> float:
    """Closed-form failure probability surrogate for synthetic mode."""
    a, b, c, d, e = 0.9, 0.3, 0.3, 0.2, 0.2
    return float(np.clip(
        a * verify_flip +
        b * (tool_noise / 3) +
        c * (hint_ambiguity / 3) +
        d * (distractor_count / 6) +
        e * (16 / memory_tokens),
        0.0, 1.0
    ))


def run_episode(
    grid: Dict[str, np.ndarray],
    grid_idx: int,
    episode_seed: int,
    use_synthetic: bool = False,
) -> bool:
    """Run a single episode and return success/failure."""
    params = get_params_for_idx(grid, grid_idx)
    
    if use_synthetic:
        p_fail = synthetic_p_fail(**params)
        rng = np.random.default_rng(episode_seed)
        return rng.random() >= p_fail
    else:
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
        return env.check_answer(guess)


def load_posteriors(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load beta posteriors from run directory."""
    beta_path = run_dir / "beta_posteriors.npz"
    data = np.load(beta_path, allow_pickle=True)
    alpha = data["alpha"]
    beta = data["beta"]
    metadata = {}
    for key in ["grid_version", "run_uuid", "posterior_semantics"]:
        if key in data.files:
            metadata[key] = str(data[key])
    return alpha, beta, metadata


def save_posteriors(run_dir: Path, alpha: np.ndarray, beta: np.ndarray, run_uuid: str) -> None:
    """Save beta posteriors atomically."""
    from loop_io import save_npz_atomic
    save_npz_atomic(
        run_dir / "beta_posteriors.npz",
        alpha=alpha.astype(np.int64),
        beta=beta.astype(np.int64),
        grid_version=np.array(GRID_VERSION),
        run_uuid=np.array(run_uuid),
        posterior_semantics=np.array("Beta over p_fail; alpha=fails+1, beta=successes+1"),
    )


def update_posteriors(
    alpha: np.ndarray,
    beta: np.ndarray,
    results: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Update posteriors based on episode results."""
    alpha = alpha.copy()
    beta = beta.copy()
    for r in results:
        idx = r["grid_idx"]
        success = r["success"]
        fail = 1 - success
        alpha[idx] += fail
        beta[idx] += success
    return alpha, beta


def assert_round_artifacts_exist(round_dir: Path, expected_episodes: int) -> None:
    """Runtime assertion before committing round."""
    results_path = round_dir / "agent_results.csv"
    assert results_path.exists(), f"Missing agent_results.csv in {round_dir}"
    with open(results_path) as f:
        rows = sum(1 for _ in f) - 1
    assert rows == expected_episodes, f"Expected {expected_episodes} episodes, got {rows}"
    beta_path = round_dir.parent.parent / "beta_posteriors.npz"
    assert beta_path.exists(), f"Missing beta_posteriors.npz"


def _metrics_dict_to_round_metrics(metrics: Dict[str, Any]) -> Any:
    """Convert metrics dict to RoundMetrics-compatible object for summary CSV."""
    from loop_metrics import (
        TubeMetrics, PlanSummary, ModelChange, 
        Counts, FusionConsistency, RoundMetrics, ArrayStats
    )
    
    tube = metrics["tube"]
    counts = metrics["counts"]
    plan = metrics.get("plan", {})
    
    # Create ArrayStats for epistemic/aleatoric/beta_std
    epistemic_dict = tube.get("epistemic", {})
    aleatoric_dict = tube.get("aleatoric", {})
    
    tube_metrics = TubeMetrics(
        tube_var_sum=tube["tube_var_sum"],
        tube_var_delta=tube.get("tube_var_delta_prev"),
        tube_var_delta_prev=tube.get("tube_var_delta_prev"),
        tube_var_delta_baseline=tube.get("tube_var_delta_baseline"),
        tube_var_baseline=tube.get("tube_var_baseline"),
        tube_coverage=tube["tube_coverage"],
        tube_points_total=tube["tube_points_total"],
        beta_std=ArrayStats(mean=epistemic_dict.get("mean", 0.0)),
        epistemic=ArrayStats(
            mean=epistemic_dict.get("mean", 0.0),
            max=epistemic_dict.get("max", 0.0),
            median=epistemic_dict.get("median", 0.0),
        ),
        aleatoric=ArrayStats(
            mean=aleatoric_dict.get("mean", 0.0),
            max=aleatoric_dict.get("max", 0.0),
            median=aleatoric_dict.get("median", 0.0),
        ),
    )
    
    plan_summary = PlanSummary(
        targets_selected=plan.get("targets_selected", 0),
        total_budget=plan.get("total_budget", 0),
        top_targets=[],
    )
    
    model_change = ModelChange(enn_changed=False, fusion_changed=False)
    count_metrics = Counts(
        sampled_points_new=counts["sampled_points_new"],
        sampled_points_total=counts["sampled_points_total"],
    )
    fusion_metrics = FusionConsistency(status="OK")
    
    return RoundMetrics(
        round_id=metrics["round_id"],
        timestamp=datetime.now().isoformat(),
        seed=0,
        status=metrics["status"],
        tube=tube_metrics,
        plan=plan_summary,
        model_change=model_change,
        counts=count_metrics,
        fusion=fusion_metrics,
    )


def run_one_agent_round(
    run_dir: Path,
    round_num: int,
    run_uuid: str,
    grid: Dict[str, np.ndarray],
    targets_per_round: int,
    episodes_per_budget_unit: int,
    use_synthetic: bool,
    verbose: bool,
    prev_tube_var: Optional[float],
    baseline_tube_var: Optional[float],
    emit_receipts: bool = False,
    current_seed: Optional[int] = None,
) -> Tuple[Dict[str, Any], bool]:
    """Run one complete round of agent evaluation."""
    from loop_io import save_json, save_csv_atomic, append_to_summary_csv
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    round_id = f"R{round_num:04d}_{timestamp}"
    
    print(f"\n{'='*70}")
    print(f" AGENT ROUND {round_num} ({round_id}) ".center(70))
    print(f"{'='*70}\n")
    
    round_dir = run_dir / "rounds" / round_id
    round_dir.mkdir(parents=True, exist_ok=True)
    
    # Save round_pre.json
    round_pre = {
        "round_id": round_id,
        "round_num": round_num,
        "run_uuid": run_uuid,
        "timestamp": timestamp,
        "mode": "agent_eval",
        "use_synthetic": use_synthetic,
    }
    save_json(round_pre, round_dir / "round_pre.json")
    
    # Load posteriors and compute acquisition
    alpha, beta, _ = load_posteriors(run_dir)
    scores = compute_acquisition_score(alpha, beta)
    
    K = min(targets_per_round, len(scores))
    selected = select_targets(scores, K)
    episodes_per_target = BUDGET_PER_TARGET * episodes_per_budget_unit
    total_episodes = len(selected) * episodes_per_target
    
    print(f"[Round] Selected {len(selected)} targets, {episodes_per_target} episodes each")
    print(f"[Round] Total episodes: {total_episodes}")
    print(f"[Round] Mode: {'Synthetic' if use_synthetic else 'Real simulation'}")
    
    # Save plan
    plan = {
        "round_id": round_id,
        "selected": selected.tolist(),
        "budget_per_target": BUDGET_PER_TARGET,
        "episodes_per_budget_unit": episodes_per_budget_unit,
        "total_episodes": episodes_per_target,
        "K": K,
        "tau": 0.2,
        "sigma": 0.05,
        "w1": 1.0,
        "w2": 0.5,
        "seed": None,
    }
    save_json(plan, round_dir / "active_sampling_plan.json")
    
    # Run episodes
    results: List[Dict[str, Any]] = []
    
    for target_idx, grid_idx in enumerate(selected):
        grid_idx = int(grid_idx)
        params = get_params_for_idx(grid, grid_idx)
        
        for episode_idx in range(episodes_per_target):
            episode_seed = derive_episode_seed(run_uuid, round_num, grid_idx, episode_idx)
            success = run_episode(grid, grid_idx, episode_seed, use_synthetic)
            
            results.append({
                "round_id": round_id,
                "grid_idx": grid_idx,
                "episode_idx": episode_idx,
                "episode_seed": episode_seed,
                "success": 1 if success else 0,
                **params,
            })
        
        if verbose and (target_idx + 1) % 10 == 0:
            print(f"  Completed {target_idx + 1}/{len(selected)} targets")
    
    # Save agent_results.csv (atomic)
    columns = [
        "round_id", "grid_idx", "episode_idx", "episode_seed", "success",
        "tool_noise", "verify_flip", "hint_ambiguity", "distractor_count", "memory_tokens",
    ]
    save_csv_atomic(round_dir / "agent_results.csv", results, columns)
    
    # Update posteriors
    alpha_new, beta_new = update_posteriors(alpha, beta, results)
    save_posteriors(run_dir, alpha_new, beta_new, run_uuid)
    
    # Compute metrics
    metrics = compute_agent_tube_metrics(
        alpha=alpha_new,
        beta=beta_new,
        round_id=round_id,
        round_num=round_num,
        prev_tube_var=prev_tube_var,
        baseline_tube_var=baseline_tube_var,
        selected=selected,
        episodes_per_target=episodes_per_target,
    )
    
    # Assertion before commit
    assert_round_artifacts_exist(round_dir, total_episodes)
    
    # Save metrics.json (COMMIT MARKER)
    save_json(metrics, round_dir / "metrics.json")
    
    # Append to summary.csv
    metrics_obj = _metrics_dict_to_round_metrics(metrics)
    append_to_summary_csv(run_dir, metrics_obj, run_uuid=run_uuid)
    
    # Save round_post.json
    round_post = {
        "round_id": round_id,
        "completed_at": datetime.now().isoformat(),
        "status": metrics["status"],
        "tube_var_sum": metrics["tube"]["tube_var_sum"],
    }
    save_json(round_post, round_dir / "round_post.json")

    # Emit receipt if requested
    if emit_receipts:
        try:
            from bef_zk.shared.receipts import build_round_receipt
            round_config = {
                "grid_version": grid.get("version", GRID_VERSION),
                "targets_per_round": targets_per_round,
                "episodes_per_budget_unit": episodes_per_budget_unit,
                "seed": current_seed,
                "use_synthetic": use_synthetic,
                "tau": 0.2,
                "sigma": 0.05,
                "w1": 1.0,
                "w2": 0.5,
            }
            receipt = build_round_receipt(round_dir, round_config)
            save_json(receipt, round_dir / "round_receipt.json")
            if verbose:
                print(f"  [Receipt] statement_hash: {receipt['statement_hash'][:16]}...")
        except ImportError as e:
            print(f"  [Receipt] Warning: Could not import receipts module: {e}")

    print(f"\n[Round {round_num}] Complete")
    print(f"  tube_var_sum: {metrics['tube']['tube_var_sum']:.6f}")
    print(f"  tube_coverage: {metrics['tube']['tube_coverage']:.3f}")
    print(f"  status: {metrics['status']}")

    return metrics, True


def run_agent_eval_loop(
    run_dir: Path,
    n_rounds: int,
    base_seed: int,
    seed_mode: str,
    agent_bench: str = "toy_v1",
    episodes_per_budget_unit: int = 1,
    targets_per_round: int = 64,
    use_synthetic: bool = False,
    verbose: bool = False,
    emit_receipts: bool = False,
) -> None:
    """Run the full agent evaluation loop."""
    run_dir = Path(run_dir)
    
    if agent_bench != "toy_v1":
        raise ValueError(f"Unknown agent benchmark: {agent_bench}. Only 'toy_v1' is supported.")
    
    # Load or create run metadata
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        run_uuid = metadata.get("run_uuid", str(uuid.uuid4())[:8])
        print(f"[Init] Resuming run: {run_uuid}")
    else:
        run_uuid = str(uuid.uuid4())[:8]
        metadata = {
            "run_uuid": run_uuid,
            "base_seed": base_seed,
            "seed_mode": seed_mode,
            "mode": "agent_eval",
            "agent_bench": agent_bench,
            "created_at": datetime.now().isoformat(),
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[Init] New run created: {run_uuid}")
    
    # Load grid
    grid_path = run_dir / "grid.npz"
    if not grid_path.exists():
        raise FileNotFoundError(
            f"Grid not found at {grid_path}. "
            f"Generate one with: python -m agent_bench.grid --out {grid_path}"
        )
    grid = load_grid(grid_path)
    n_points = int(grid["n_points"])
    print(f"[Init] Loaded grid with {n_points} points")
    
    # Initialize beta_posteriors if missing
    beta_path = run_dir / "beta_posteriors.npz"
    if not beta_path.exists():
        alpha = np.ones(n_points, dtype=np.int64)
        beta = np.ones(n_points, dtype=np.int64)
        save_posteriors(run_dir, alpha, beta, run_uuid)
        print(f"[Init] Created beta_posteriors.npz with {n_points} points (Beta(1,1) priors)")
    
    (run_dir / "rounds").mkdir(exist_ok=True)
    
    # Determine starting round
    from loop_io import get_next_round_num
    start_round = get_next_round_num(run_dir)
    end_round = start_round + n_rounds - 1
    
    # Load baseline from previous rounds if resuming
    baseline_tube_var: Optional[float] = None
    prev_tube_var: Optional[float] = None
    
    if start_round > 1:
        from loop_io import get_previous_round_info
        prev_info = get_previous_round_info(run_dir, current_round_num=start_round)
        if prev_info:
            baseline_tube_var = prev_info.get("baseline_tube_var")
            prev_tube_var = prev_info.get("tube_var")
            print(f"[Resume] Continuing from round {start_round}")
            if baseline_tube_var:
                print(f"[Resume] Loaded baseline tube_var: {baseline_tube_var:.6f}")
    
    print(f"""
{'='*70}
{'AGENT EVALUATION LOOP'.center(70)}
{'='*70}
  Run UUID:         {run_uuid}
  Rounds to run:    {n_rounds} (R{start_round:04d} -> R{end_round:04d})
  Grid points:      {n_points}
  Targets/round:    {targets_per_round}
  Episodes/target:  {BUDGET_PER_TARGET * episodes_per_budget_unit}
  Mode:             {'Synthetic' if use_synthetic else 'Real simulation'}
{'='*70}
""")
    
    # Import seed function for receipt binding
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from loop_runner import get_seed

    for round_num in range(start_round, end_round + 1):
        # Compute seed for this round (for receipt binding)
        current_seed = get_seed(base_seed, round_num, seed_mode, [])

        metrics, success = run_one_agent_round(
            run_dir=run_dir,
            round_num=round_num,
            run_uuid=run_uuid,
            grid=grid,
            targets_per_round=targets_per_round,
            episodes_per_budget_unit=episodes_per_budget_unit,
            use_synthetic=use_synthetic,
            verbose=verbose,
            prev_tube_var=prev_tube_var,
            baseline_tube_var=baseline_tube_var,
            emit_receipts=emit_receipts,
            current_seed=current_seed,
        )

        if not success:
            print(f"\n[STOPPED] Round {round_num} failed!")
            break

        prev_tube_var = metrics["tube"]["tube_var_sum"]
        if baseline_tube_var is None:
            baseline_tube_var = prev_tube_var

    # Emit run receipt if requested
    if emit_receipts:
        try:
            from bef_zk.shared.receipts import build_run_receipt, collect_round_dirs
            from loop_io import load_json

            round_dirs = collect_round_dirs(run_dir)
            round_receipts = []
            for rd in round_dirs:
                receipt_path = rd / "round_receipt.json"
                if receipt_path.exists():
                    round_receipts.append(load_json(receipt_path))

            if round_receipts:
                run_receipt = build_run_receipt(run_dir, round_receipts)
                from loop_io import save_json as save_json_io
                save_json_io(run_receipt, run_dir / "run_receipt.json")
                print(f"\n[Receipt] Run receipt: chain_hash={run_receipt['chain_hash'][:16]}...")
        except ImportError as e:
            print(f"\n[Receipt] Warning: Could not build run receipt: {e}")

    print(f"""
{'='*70}
{'LOOP COMPLETE'.center(70)}
{'='*70}
  Run UUID: {run_uuid}
  Artifacts: {run_dir}
  Summary CSV: {run_dir / 'summary.csv'}
""")
