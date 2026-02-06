#!/usr/bin/env python3
"""Orchestrate a single adaptive round of the BICEP → ENN → Fusion pipeline.

Steps executed:
1. (Optional) Merge fresh BICEP results into the Beta posterior tracker.
2. Train/refresh ENN (default: `python train_simple_enn.py`).
3. Fuse the field with FusionAlpha (default: `python fuse_field.py`).
4. Compute a new adaptive sampling plan via `adaptive_sampling`.

This script does not launch BICEP simulations; it automates the wiring from
BICEP outputs into ENN/Fusion and updates the next plan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from adaptive_sampling import AdaptiveSampler
from rebuild_training_data import rebuild_dataset


# ---------------------------------------------------------------------------
# Checkpoint fingerprinting & run metadata
# ---------------------------------------------------------------------------


def file_fingerprint(path: Path) -> Optional[Dict[str, Any]]:
    """Compute fingerprint of a file (mtime, size, content hash for small files)."""
    if not path.exists():
        return None
    stat = path.stat()
    info: Dict[str, Any] = {
        "path": str(path),
        "mtime": stat.st_mtime,
        "size": stat.st_size,
    }
    # For small files, also compute content hash
    if stat.st_size < 50_000_000:  # < 50MB
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        info["sha256_prefix"] = h.hexdigest()[:16]
    return info


def npz_array_fingerprint(path: Path) -> Optional[str]:
    """Quick fingerprint of npz by hashing first array's bytes."""
    if not path.exists():
        return None
    try:
        data = np.load(path)
        keys = sorted(data.files)
        if not keys:
            return "empty"
        h = hashlib.sha256()
        for k in keys[:3]:  # First 3 arrays
            arr = data[k]
            h.update(k.encode())
            h.update(arr.tobytes()[:10000])  # First 10KB
        return h.hexdigest()[:16]
    except Exception as e:
        return f"error:{e}"


def collect_run_metadata(run_dir: Path, round_id: Optional[str] = None) -> Dict[str, Any]:
    """Collect comprehensive metadata for debugging checkpoint issues."""
    if round_id is None:
        round_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    meta: Dict[str, Any] = {
        "run_id": round_id,
        "timestamp": datetime.now().isoformat(),
        "run_dir": str(run_dir),
        "artifacts": {},
    }

    # Key artifact files to track
    artifact_names = [
        "enn.npz",
        "fusion.npz",
        "grid.npz",
        "beta_posteriors.npz",
        "enn_weights.pt",
        "enn_weights.json",
        "training_data_merged.csv",
    ]

    for name in artifact_names:
        path = run_dir / name
        if path.suffix == ".npz":
            meta["artifacts"][name] = {
                "fingerprint": npz_array_fingerprint(path),
                **(file_fingerprint(path) or {}),
            }
        else:
            fp = file_fingerprint(path)
            if fp:
                meta["artifacts"][name] = fp

    return meta


def save_run_metadata(meta: Dict[str, Any], run_dir: Path) -> Path:
    """Save run metadata to JSON file."""
    out_path = run_dir / f"run_metadata_{meta['run_id']}.json"
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return out_path


def compare_fingerprints(meta1: Dict, meta2: Dict) -> Dict[str, str]:
    """Compare two metadata dicts, report changes."""
    changes = {}
    for name in set(meta1.get("artifacts", {}).keys()) | set(meta2.get("artifacts", {}).keys()):
        fp1 = meta1.get("artifacts", {}).get(name, {}).get("fingerprint") or meta1.get("artifacts", {}).get(name, {}).get("sha256_prefix")
        fp2 = meta2.get("artifacts", {}).get(name, {}).get("fingerprint") or meta2.get("artifacts", {}).get(name, {}).get("sha256_prefix")
        if fp1 != fp2:
            changes[name] = f"{fp1} → {fp2}"
        elif fp1 is None and fp2 is None:
            pass  # Both missing
        else:
            changes[name] = "unchanged"
    return changes


# ---------------------------------------------------------------------------
# Round invariants & progress tracking
# ---------------------------------------------------------------------------


@dataclass
class RoundInvariants:
    """Invariants that must hold for a valid round."""
    ingest_ok: bool = True  # Beta point count increased if new tallies
    retrain_ok: bool = True  # ENN changed if training ran
    fusion_ok: bool = True  # Fusion changed if ENN changed and fusion ran
    progress_ok: bool = True  # Tube variance decreased or coverage increased
    dataset_ok: bool = True  # Training dataset updated if ingestion happened

    ingest_msg: str = ""
    retrain_msg: str = ""
    fusion_msg: str = ""
    progress_msg: str = ""
    dataset_msg: str = ""

    def all_passed(self) -> bool:
        return self.ingest_ok and self.retrain_ok and self.fusion_ok and self.progress_ok and self.dataset_ok

    def summary(self) -> str:
        lines = []
        for name, ok, msg in [
            ("INGEST", self.ingest_ok, self.ingest_msg),
            ("DATASET", self.dataset_ok, self.dataset_msg),
            ("RETRAIN", self.retrain_ok, self.retrain_msg),
            ("FUSION", self.fusion_ok, self.fusion_msg),
            ("PROGRESS", self.progress_ok, self.progress_msg),
        ]:
            status = "PASS" if ok else "FAIL"
            lines.append(f"  [{status}] {name}: {msg}")
        return "\n".join(lines)


def compute_tube_variance(run_dir: Path, tube_low: float = 0.4, tube_high: float = 0.6) -> Dict[str, float]:
    """Compute integrated posterior variance inside the tube band."""
    beta_path = run_dir / "beta_posteriors.npz"
    enn_path = run_dir / "enn.npz"

    if not beta_path.exists() or not enn_path.exists():
        return {"tube_var_sum": float("nan"), "tube_points": 0, "total_sampled": 0}

    beta = np.load(beta_path)
    enn = np.load(enn_path)

    alpha, beta_arr = beta["alpha"], beta["beta"]
    q_enn = enn["q_enn"]

    # Beta variance
    var = alpha * beta_arr / ((alpha + beta_arr) ** 2 * (alpha + beta_arr + 1.0))

    # Points in tube (based on ENN prediction)
    in_tube = (q_enn >= tube_low) & (q_enn <= tube_high)

    # Points with actual samples (alpha + beta > 2 means we have data beyond prior)
    sampled = (alpha + beta_arr) > 2.0

    tube_var_sum = float(np.sum(var[in_tube]))
    tube_points = int(np.sum(in_tube))
    tube_sampled = int(np.sum(in_tube & sampled))
    total_sampled = int(np.sum(sampled))

    return {
        "tube_var_sum": tube_var_sum,
        "tube_var_mean": tube_var_sum / max(tube_points, 1),
        "tube_points": tube_points,
        "tube_sampled": tube_sampled,
        "total_sampled": total_sampled,
    }


def load_progress_history(run_dir: Path) -> List[Dict]:
    """Load progress history from previous rounds."""
    history_path = run_dir / "progress_history.json"
    if history_path.exists():
        with open(history_path) as f:
            return json.load(f)
    return []


def save_progress_history(history: List[Dict], run_dir: Path) -> None:
    """Save progress history."""
    history_path = run_dir / "progress_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


def check_invariants(
    run_dir: Path,
    meta_before: Dict,
    meta_after: Dict,
    changes: Dict[str, str],
    had_new_data: bool,
    did_train: bool,
    did_fuse: bool,
) -> RoundInvariants:
    """Check all round invariants and return structured result."""
    inv = RoundInvariants()

    # 1. Ingest invariant: if we had new data, beta should change
    if had_new_data:
        beta_change = changes.get("beta_posteriors.npz", "")
        if "→" in beta_change:
            inv.ingest_ok = True
            inv.ingest_msg = "Beta posteriors updated with new data"
        else:
            inv.ingest_ok = False
            inv.ingest_msg = "WARNING: New data provided but Beta posteriors unchanged!"
    else:
        inv.ingest_msg = "No new data this round (skipped)"

    # 1b. Dataset invariant: if ingest happened, dataset must change
    if had_new_data:
        # Check if training_data_merged.csv changed
        ds_change = changes.get("training_data_merged.csv", "")
        if "→" in ds_change or ds_change == "unchanged": 
            # Note: "unchanged" might happen if BICEP produced 0 successes/failures (unlikely) 
            # or if fingerprint didn't catch it. Ideally we see a change.
            # But if it wasn't tracked before (first round using it), it might be "new" or behave differently.
            # compare_fingerprints returns "fp1 -> fp2" if changed, or "unchanged" if same.
            # If it didn't exist before, it might not be in changes? 
            # compare_fingerprints iterates keys in meta1 OR meta2.
            pass
        
        if "→" in ds_change:
            inv.dataset_ok = True
            inv.dataset_msg = "Training dataset updated reflecting new posteriors"
        elif inv.ingest_ok:
            # Posteriors changed but dataset didn't?
            inv.dataset_ok = False
            inv.dataset_msg = "CRITICAL: Posteriors updated but training dataset fingerprint is identical!"
        else:
            inv.dataset_ok = True
            inv.dataset_msg = "Posteriors didn't change, so dataset didn't change"
    else:
        inv.dataset_msg = "No new data, dataset update not required"

    # 2. Retrain invariant: if we trained, ENN should change
    if did_train:
        enn_change = changes.get("enn.npz", "")
        if "→" in enn_change:
            inv.retrain_ok = True
            inv.retrain_msg = "ENN weights updated after training"
        else:
            inv.retrain_ok = False
            inv.retrain_msg = "CRITICAL: Training ran but enn.npz unchanged! Check checkpoint paths."
    else:
        inv.retrain_msg = "Training skipped this round"

    # 3. Fusion invariant: if ENN changed and fusion ran, fusion should change
    if did_fuse:
        enn_change = changes.get("enn.npz", "")
        fusion_change = changes.get("fusion.npz", "")
        if "→" in enn_change:
            if "→" in fusion_change:
                inv.fusion_ok = True
                inv.fusion_msg = "Fusion updated after ENN change"
            else:
                # This is suspicious but might be OK if changes are tiny
                inv.fusion_ok = True  # Warning, not failure
                inv.fusion_msg = "WARNING: ENN changed but fusion.npz byte-identical (may be OK if delta tiny)"
        else:
            inv.fusion_msg = "ENN unchanged, fusion consistency not checked"
    else:
        inv.fusion_msg = "Fusion skipped this round"

    # 4. Progress invariant: tube variance should decrease OR coverage should increase
    history = load_progress_history(run_dir)
    current_progress = compute_tube_variance(run_dir)

    if history:
        prev = history[-1]
        prev_tube_var = prev.get("tube_var_sum", float("inf"))
        prev_sampled = prev.get("total_sampled", 0)

        curr_tube_var = current_progress["tube_var_sum"]
        curr_sampled = current_progress["total_sampled"]

        var_decreased = curr_tube_var < prev_tube_var
        coverage_increased = curr_sampled > prev_sampled

        if var_decreased or coverage_increased:
            inv.progress_ok = True
            inv.progress_msg = (
                f"Progress made: tube_var {prev_tube_var:.6f}→{curr_tube_var:.6f}, "
                f"sampled {prev_sampled}→{curr_sampled}"
            )
        else:
            inv.progress_ok = False
            inv.progress_msg = (
                f"NO PROGRESS: tube_var {prev_tube_var:.6f}→{curr_tube_var:.6f}, "
                f"sampled {prev_sampled}→{curr_sampled}"
            )
    else:
        inv.progress_ok = True
        inv.progress_msg = f"First round: tube_var={current_progress['tube_var_sum']:.6f}, sampled={current_progress['total_sampled']}"

    # Update history
    current_progress["round_id"] = meta_after.get("run_id", "unknown")
    current_progress["timestamp"] = datetime.now().isoformat()
    history.append(current_progress)
    save_progress_history(history, run_dir)

    return inv


@dataclass
class InvariantViolation(Exception):
    """Raised when critical invariants fail."""
    invariants: RoundInvariants


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"[ActiveRound] Running: {' '.join(cmd)} (cwd={cwd})")
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one adaptive training round")
    parser.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--run-dir", type=str, default=str(Path("artifacts") / "latest_bicep"))
    parser.add_argument("--bicep-results", type=str, help="CSV with columns index,successes,trials")
    parser.add_argument("--train-script", type=str, default="train_enn.py")
    parser.add_argument("--fuse-script", type=str, default="fuse_field.py")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-fuse", action="store_true")
    parser.add_argument("--num-select", type=int, default=64)
    parser.add_argument("--min-distance", type=float, default=0.08)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--plan-path", type=str, default="active_sampling_plan.json")
    parser.add_argument("--train-args", nargs=argparse.REMAINDER, help="Extra args passed to train script")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_dir = (repo_root / args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir does not exist: {run_dir}")

    # Collect metadata BEFORE any changes
    round_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    meta_before = collect_run_metadata(run_dir, round_id=f"{round_id}_pre")
    print(f"[ActiveRound] Starting round {round_id}")
    print(f"[ActiveRound] Pre-round artifact fingerprints:")
    for name, info in meta_before.get("artifacts", {}).items():
        fp = info.get("fingerprint") or info.get("sha256_prefix", "N/A")
        print(f"  {name}: {fp}")

    sampler = AdaptiveSampler(run_dir)
    if args.bicep_results:
        results_path = Path(args.bicep_results)
        if not results_path.is_absolute():
            results_path = (repo_root / results_path).resolve()
        sampler.update_from_csv(results_path)
        print(f"[ActiveRound] Updated Beta posteriors from {results_path}")

    # Rebuild training data (merge static priors + dynamic beta)
    print(f"[ActiveRound] Rebuilding training dataset...")
    merged_csv_path = run_dir / "training_data_merged.csv"
    rebuild_dataset(
        run_dir=run_dir,
        static_csv_path=repo_root / "double_well_data.csv",
        beta_path=run_dir / "beta_posteriors.npz",
        output_path=merged_csv_path
    )

    if not args.skip_train:
        train_cmd = [sys.executable, str(repo_root / args.train_script)]
        # Add data path argument
        train_cmd.extend(["--data-path", str(merged_csv_path)])
        
        if args.train_args:
            train_cmd.extend(args.train_args)
        run_cmd(train_cmd, repo_root)

    if not args.skip_fuse:
        fuse_cmd = [sys.executable, str(repo_root / args.fuse_script)]
        fuse_cmd.extend(["--data-path", str(merged_csv_path)])
        run_cmd(fuse_cmd, repo_root)

    # Collect metadata AFTER training/fusion
    meta_after = collect_run_metadata(run_dir, round_id=f"{round_id}_post")
    changes = compare_fingerprints(meta_before, meta_after)
    print(f"[ActiveRound] Post-round artifact changes:")
    for name, status in changes.items():
        marker = "CHANGED" if "→" in status else "same"
        print(f"  {name}: [{marker}] {status}")

    # Save both metadata files for audit trail
    save_run_metadata(meta_before, run_dir)
    save_run_metadata(meta_after, run_dir)

    # Check all invariants
    invariants = check_invariants(
        run_dir=run_dir,
        meta_before=meta_before,
        meta_after=meta_after,
        changes=changes,
        had_new_data=bool(args.bicep_results),
        did_train=not args.skip_train,
        did_fuse=not args.skip_fuse,
    )
    print(f"[ActiveRound] Invariant check:")
    print(invariants.summary())

    if not invariants.all_passed():
        print(f"[ActiveRound] *** INVARIANT VIOLATIONS DETECTED ***")
        if not invariants.retrain_ok:
            print(f"[ActiveRound] CRITICAL: Stopping due to retrain invariant failure.")
            print(f"[ActiveRound] The ENN checkpoint is not updating. Check:")
            print(f"[ActiveRound]   1. train_simple_enn.py output path")
            print(f"[ActiveRound]   2. Whether it's loading a cached checkpoint")
            print(f"[ActiveRound]   3. Whether training data actually changed")
            sys.exit(1)

    # Re-initialize sampler to pick up refreshed artifacts (ENN/Fusion outputs)
    sampler = AdaptiveSampler(run_dir)
    plan = sampler.compute_plan(
        num_select=args.num_select,
        min_distance=args.min_distance,
        n_bins=args.n_bins,
    )
    plan_path = run_dir / args.plan_path
    sampler.save_plan(plan, plan_path)
    print(f"[ActiveRound] Saved new plan with {len(plan['selected_indices'])} points to {plan_path}")

    # Log score statistics for sanity check
    scores = plan.get("scores", [])
    if scores:
        scores_arr = np.array(scores)
        print(f"[ActiveRound] Score stats: min={scores_arr.min():.4f}, max={scores_arr.max():.4f}, "
              f"mean={scores_arr.mean():.4f}, median={np.median(scores_arr):.4f}")
        if scores_arr.max() > 100:
            print(f"[ActiveRound] WARNING: Max score > 100 indicates possible acquisition blow-up!")

    # Generate comprehensive round report
    try:
        from round_report import RoundReport
        report = RoundReport.load(run_dir, round_id=round_id)
        report_text = report.generate()
        report_path = run_dir / f"round_report_{round_id}.txt"
        report_path.write_text(report_text)
        print(f"[ActiveRound] Full report saved to {report_path}")

        # Also print a summary to console
        print("\n" + "=" * 70)
        print("ROUND SUMMARY".center(70))
        print("=" * 70)
        n_tube = int(report.tube_mask.sum())
        n_tube_sampled = int((report.tube_mask & report.sampled_mask).sum())
        n_selected = len(plan.get("selected_indices", []))
        total_budget = sum(plan.get("budget", []))
        print(f"  Tube coverage: {n_tube_sampled}/{n_tube} ({100*n_tube_sampled/n_tube:.1f}%)")
        print(f"  Targets selected: {n_selected} (budget: {total_budget} trajectories)")
        print(f"  Epistemic in tube: {int((report.high_epistemic_mask & report.tube_mask).sum())} high-uncertainty points")
        if report.progress_history:
            print(f"  Tube variance: {report.progress_history[-1].get('tube_var_sum', 0):.4f}")
        print("=" * 70)
    except Exception as e:
        print(f"[ActiveRound] Could not generate report: {e}")


if __name__ == "__main__":
    main()
