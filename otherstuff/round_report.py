#!/usr/bin/env python3
"""Generate comprehensive round reports for the adaptive sampling loop.

This module produces human-readable reports explaining:
- What the loop is doing
- What each metric means
- What the current state implies
- What should happen next
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RoundReport:
    """Structured round report with all metrics and interpretations."""

    run_dir: Path
    round_id: str

    # ENN outputs
    q_enn: np.ndarray
    var_aleatoric: np.ndarray
    var_epistemic: np.ndarray
    var_total: np.ndarray

    # Beta posteriors
    beta_alpha: np.ndarray
    beta_beta: np.ndarray

    # Fusion (optional)
    q_fused: Optional[np.ndarray]

    # Plan
    plan: Dict[str, Any]

    # Progress history
    progress_history: List[Dict]

    # Grid
    points: np.ndarray

    @classmethod
    def load(cls, run_dir: Path, round_id: str = "current") -> "RoundReport":
        """Load all artifacts and construct report."""
        run_dir = Path(run_dir)

        # Load ENN
        enn = np.load(run_dir / "enn.npz")
        q_enn = enn["q_enn"]
        var_aleatoric = enn.get("var_aleatoric", enn.get("var_enn", np.zeros_like(q_enn)))
        var_epistemic = enn.get("var_epistemic", np.zeros_like(q_enn))
        var_total = enn.get("var_enn", var_aleatoric + var_epistemic)

        # Load Beta posteriors
        beta = np.load(run_dir / "beta_posteriors.npz")
        beta_alpha = beta["alpha"]
        beta_beta = beta["beta"]

        # Load Fusion (optional)
        fusion_path = run_dir / "fusion.npz"
        q_fused = None
        if fusion_path.exists():
            fusion = np.load(fusion_path)
            q_fused = fusion["q_fused"]

        # Load plan
        plan_path = run_dir / "active_sampling_plan.json"
        plan = {}
        if plan_path.exists():
            with open(plan_path) as f:
                plan = json.load(f)

        # Load progress history
        history_path = run_dir / "progress_history.json"
        progress_history = []
        if history_path.exists():
            with open(history_path) as f:
                progress_history = json.load(f)

        # Load grid
        grid = np.load(run_dir / "grid.npz")
        points = np.stack([grid["x"], grid["y"]], axis=1)

        return cls(
            run_dir=run_dir,
            round_id=round_id,
            q_enn=q_enn,
            var_aleatoric=var_aleatoric,
            var_epistemic=var_epistemic,
            var_total=var_total,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            q_fused=q_fused,
            plan=plan,
            progress_history=progress_history,
            points=points,
        )

    # -------------------------------------------------------------------------
    # Computed properties
    # -------------------------------------------------------------------------

    @property
    def n_points(self) -> int:
        return len(self.q_enn)

    @property
    def beta_mean(self) -> np.ndarray:
        return self.beta_alpha / (self.beta_alpha + self.beta_beta)

    @property
    def beta_var(self) -> np.ndarray:
        a, b = self.beta_alpha, self.beta_beta
        return a * b / ((a + b) ** 2 * (a + b + 1))

    @property
    def beta_std(self) -> np.ndarray:
        return np.sqrt(self.beta_var)

    @property
    def beta_samples(self) -> np.ndarray:
        """Number of samples at each point (alpha + beta - 2 for uniform prior)."""
        return self.beta_alpha + self.beta_beta - 2

    @property
    def sampled_mask(self) -> np.ndarray:
        """Points that have been sampled at least once."""
        return self.beta_samples > 0

    @property
    def tube_mask(self) -> np.ndarray:
        """Points in the transition tube (0.4 < q < 0.6)."""
        return (self.q_enn >= 0.4) & (self.q_enn <= 0.6)

    @property
    def high_epistemic_mask(self) -> np.ndarray:
        """Points with epistemic uncertainty above 90th percentile."""
        if self.var_epistemic.max() == 0:
            return np.zeros(self.n_points, dtype=bool)
        threshold = np.percentile(self.var_epistemic, 90)
        return self.var_epistemic > threshold

    # -------------------------------------------------------------------------
    # Report sections
    # -------------------------------------------------------------------------

    def section_header(self, title: str) -> str:
        width = 70
        return f"\n{'=' * width}\n{title.center(width)}\n{'=' * width}\n"

    def section_system_overview(self) -> str:
        """Explain what the system is doing at a high level."""
        return f"""{self.section_header("SYSTEM OVERVIEW")}
This is a COMMITTOR LEARNING MACHINE for rare-event simulation.

GOAL: Learn q(x) = P(reach B before A | start at x)
      where A and B are two metastable states (basins)

The committor q(x) encodes the ENTIRE long-term stochastic behavior:
  - q ≈ 0: Almost certainly goes to basin A
  - q ≈ 1: Almost certainly goes to basin B
  - q ≈ 0.5: The "knife edge" - outcomes genuinely uncertain

WHY THIS MATTERS:
  - The q ≈ 0.5 isosurface IS the transition state ensemble
  - ∇q points toward success (useful for control/steering)
  - Rate constants can be computed from q and the reactive flux

THE LOOP:
  1. BICEP simulates trajectories at selected states → hit/miss data
  2. Beta posteriors track uncertainty at sampled points
  3. ENN learns a smooth global committor field
  4. Fusion enforces dynamical consistency
  5. Acquisition decides where to sample next (this report)
  6. Repeat until the tube (q ≈ 0.5) is sharply resolved
"""

    def section_current_state(self) -> str:
        """Summary of current state."""
        n_sampled = int(self.sampled_mask.sum())
        n_tube = int(self.tube_mask.sum())
        n_tube_sampled = int((self.tube_mask & self.sampled_mask).sum())
        n_tube_unsampled = n_tube - n_tube_sampled

        # Coverage
        total_samples = float(self.beta_samples.sum())
        tube_samples = float(self.beta_samples[self.tube_mask].sum())

        # Uncertainty
        tube_var = float(self.beta_var[self.tube_mask].sum())
        total_var = float(self.beta_var.sum())

        return f"""{self.section_header("CURRENT STATE")}
GRID: {self.n_points} total points

SAMPLING COVERAGE:
  Points sampled:       {n_sampled:>6} / {self.n_points} ({100*n_sampled/self.n_points:.1f}%)
  Total trajectories:   {total_samples:>6.0f}

TRANSITION TUBE (0.4 < q < 0.6):
  Points in tube:       {n_tube:>6} ({100*n_tube/self.n_points:.1f}% of grid)
  Tube sampled:         {n_tube_sampled:>6} ({100*n_tube_sampled/n_tube:.1f}% of tube)
  Tube UNSAMPLED:       {n_tube_unsampled:>6} ← KEY GAP
  Tube trajectories:    {tube_samples:>6.0f} ({100*tube_samples/max(total_samples,1):.1f}% of budget)

UNCERTAINTY (Beta posterior variance):
  Tube variance sum:    {tube_var:>10.4f}
  Total variance sum:   {total_var:>10.4f}
  Tube fraction:        {100*tube_var/max(total_var,1e-9):>10.1f}%

INTERPRETATION:
  {"✓ Good tube coverage" if n_tube_sampled/n_tube > 0.5 else "✗ Poor tube coverage - most tube points unsampled!"}
  {"✓ Budget focused on tube" if tube_samples/max(total_samples,1) > 0.3 else "✗ Budget spread too thin outside tube"}
"""

    def section_uncertainty_decomposition(self) -> str:
        """Explain the three types of uncertainty."""
        # Stats
        aleatoric_mean = float(self.var_aleatoric.mean())
        aleatoric_max = float(self.var_aleatoric.max())
        epistemic_mean = float(self.var_epistemic.mean())
        epistemic_max = float(self.var_epistemic.max())

        n_high_epi = int(self.high_epistemic_mask.sum())
        n_high_epi_in_tube = int((self.high_epistemic_mask & self.tube_mask).sum())

        # Beta uncertainty in tube
        tube_beta_std_mean = float(self.beta_std[self.tube_mask].mean()) if self.tube_mask.any() else 0

        return f"""{self.section_header("UNCERTAINTY DECOMPOSITION")}
Three distinct sources of uncertainty:

1. BETA POSTERIOR (from sampling)
   "How much do I know about q(x) from the trajectories I've run here?"

   In tube: mean std = {tube_beta_std_mean:.4f}

   This shrinks as you sample more at each point.
   CI-based early stopping uses this to prune confident points.

2. ALEATORIC (learned variance head)
   "How noisy is the outcome at this state?"

   Mean: {aleatoric_mean:.6f}    Max: {aleatoric_max:.6f}

   This is INHERENT stochasticity - more sampling won't reduce it.
   High aleatoric near q=0.5 is expected (outcomes are genuinely uncertain).

3. EPISTEMIC (MC dropout)
   "Is my model extrapolating / out-of-distribution here?"

   Mean: {epistemic_mean:.6f}    Max: {epistemic_max:.6f}
   High epistemic points: {n_high_epi} ({100*n_high_epi/self.n_points:.1f}% of grid)
   High epistemic IN TUBE: {n_high_epi_in_tube}

   This signals "model hasn't seen training data like this."
   High epistemic + in tube = HIGH VALUE TARGET for sampling.

INTERPRETATION:
  Aleatoric >> Epistemic: Model has seen the space, dynamics are just noisy
  Epistemic >> Aleatoric: Model is guessing - need more training coverage
  Both low: Confident predictions (but check against Beta!)
"""

    def section_enn_vs_beta(self) -> str:
        """Compare ENN predictions to Beta posteriors."""
        # Only compare where we have samples
        mask = self.sampled_mask
        if not mask.any():
            return f"{self.section_header('ENN VS BETA COMPARISON')}\nNo sampled points yet.\n"

        enn_at_sampled = self.q_enn[mask]
        beta_at_sampled = self.beta_mean[mask]
        beta_std_at_sampled = self.beta_std[mask]

        # Disagreement
        abs_diff = np.abs(enn_at_sampled - beta_at_sampled)
        z_score = abs_diff / np.maximum(beta_std_at_sampled, 0.01)

        mean_diff = float(abs_diff.mean())
        max_diff = float(abs_diff.max())
        mean_z = float(z_score.mean())
        max_z = float(z_score.max())

        # Outliers (z > 2)
        n_outliers = int((z_score > 2).sum())

        return f"""{self.section_header("ENN VS BETA COMPARISON")}
Comparing ENN predictions to empirical Beta posteriors at sampled points:

ABSOLUTE DIFFERENCE |q_enn - q_beta|:
  Mean: {mean_diff:.4f}    Max: {max_diff:.4f}

Z-SCORE (difference / beta_std):
  Mean: {mean_z:.2f}    Max: {max_z:.2f}
  Outliers (z > 2): {n_outliers} points

INTERPRETATION:
  Mean z < 1: ENN is well-calibrated to empirical data ✓
  Mean z > 2: ENN is systematically wrong - check training

  High-z points are where ENN and data strongly disagree.
  These may indicate:
    - Insufficient training data in that region
    - Model architecture limitations
    - Or just sampling noise (if few trajectories)
"""

    def section_plan_analysis(self) -> str:
        """Analyze the current sampling plan."""
        if not self.plan:
            return f"{self.section_header('SAMPLING PLAN')}\nNo plan available.\n"

        n_selected = len(self.plan.get("selected_indices", []))
        scores = self.plan.get("scores", [])
        budgets = self.plan.get("budget", [])
        components = self.plan.get("components", {})
        metadata = self.plan.get("metadata", {})

        if not scores:
            return f"{self.section_header('SAMPLING PLAN')}\nEmpty plan.\n"

        scores = np.array(scores)
        budgets = np.array(budgets)

        # Score stats
        score_stats = metadata.get("score_stats", {})
        threshold = score_stats.get("threshold_used", 0)

        # Component breakdown
        uncertainty = np.array(components.get("uncertainty", []))
        tube = np.array(components.get("tube", []))
        epistemic = np.array(components.get("epistemic", []))
        disagreement = np.array(components.get("disagreement", []))

        # Top targets
        top_indices = np.argsort(scores)[::-1][:5]

        lines = [self.section_header("SAMPLING PLAN")]
        lines.append(f"""
SELECTION SUMMARY:
  Points selected:      {n_selected}
  Points requested:     {metadata.get('quality_threshold', {}).get('requested', 64)}
  Quality threshold:    {threshold:.4f}
  Active candidates:    {metadata.get('active_candidate_count', 'N/A')}
  Skipped (confident):  {len(metadata.get('skipped_confident', []))}

SCORE DISTRIBUTION:
  Min:    {scores.min():.4f}
  Max:    {scores.max():.4f}
  Mean:   {scores.mean():.4f}
  Median: {np.median(scores):.4f}

BUDGET ALLOCATION:
  Total trajectories:   {budgets.sum()}
  Min per point:        {budgets.min()}
  Max per point:        {budgets.max()}
  Mean per point:       {budgets.mean():.1f}
""")

        lines.append("\nTOP 5 TARGETS (highest acquisition score):")
        lines.append(f"{'Rank':<5} {'Idx':<6} {'Score':<8} {'Budget':<7} {'Tube':<8} {'Epist':<8} {'Disagr':<8}")
        lines.append("-" * 60)
        for rank, i in enumerate(top_indices, 1):
            idx = self.plan["selected_indices"][i]
            lines.append(
                f"{rank:<5} {idx:<6} {scores[i]:<8.3f} {budgets[i]:<7} "
                f"{tube[i] if len(tube) > i else 0:<8.4f} "
                f"{epistemic[i] if len(epistemic) > i else 0:<8.4f} "
                f"{disagreement[i] if len(disagreement) > i else 0:<8.4f}"
            )

        lines.append("""
COMPONENT WEIGHTS (in acquisition function):
  uncertainty (0.30): Expected variance reduction from more samples
  tube        (0.25): Gradient magnitude × proximity to q=0.5
  epistemic   (0.25): MC dropout variance (model uncertainty)
  disagreement(0.15): |ENN - Beta| / Beta_std
  fusion_delta(0.05): |Fused - ENN|

HIGH TUBE + HIGH EPISTEMIC = "transition region where model is unsure"
  → These are the most valuable points to sample next.
""")

        return "\n".join(lines)

    def section_progress(self) -> str:
        """Show progress across rounds."""
        if not self.progress_history:
            return f"{self.section_header('PROGRESS HISTORY')}\nNo history yet.\n"

        lines = [self.section_header("PROGRESS HISTORY")]
        lines.append("\nTube variance should DECREASE as we learn the transition region.\n")
        lines.append(f"{'Round':<6} {'Tube Var':<12} {'Tube Pts':<10} {'Sampled':<10} {'Timestamp':<20}")
        lines.append("-" * 65)

        for i, entry in enumerate(self.progress_history):
            lines.append(
                f"{i+1:<6} {entry.get('tube_var_sum', 0):<12.4f} "
                f"{entry.get('tube_points', 0):<10} "
                f"{entry.get('total_sampled', 0):<10} "
                f"{entry.get('timestamp', 'N/A')[:19]:<20}"
            )

        if len(self.progress_history) >= 2:
            first = self.progress_history[0]
            last = self.progress_history[-1]
            var_change = last.get('tube_var_sum', 0) - first.get('tube_var_sum', 0)
            sample_change = last.get('total_sampled', 0) - first.get('total_sampled', 0)
            lines.append(f"\nTOTAL CHANGE:")
            lines.append(f"  Tube variance: {var_change:+.4f} ({'↓ improving' if var_change < 0 else '↑ worsening'})")
            lines.append(f"  Samples: {sample_change:+d}")

        return "\n".join(lines)

    def section_recommendations(self) -> str:
        """Actionable recommendations based on current state."""
        recs = []

        # Check tube coverage
        n_tube = int(self.tube_mask.sum())
        n_tube_sampled = int((self.tube_mask & self.sampled_mask).sum())
        if n_tube_sampled / max(n_tube, 1) < 0.3:
            recs.append("• PRIORITY: Increase tube coverage - most transition region unsampled")

        # Check epistemic
        n_high_epi_tube = int((self.high_epistemic_mask & self.tube_mask).sum())
        if n_high_epi_tube > 10:
            recs.append(f"• {n_high_epi_tube} tube points have high epistemic uncertainty - model extrapolating")

        # Check if plan is too small
        if self.plan:
            n_selected = len(self.plan.get("selected_indices", []))
            if n_selected < 10:
                recs.append("• Only {n_selected} targets selected - consider relaxing quality threshold")

        # Check variance trend
        if len(self.progress_history) >= 2:
            recent = self.progress_history[-1].get('tube_var_sum', 0)
            prev = self.progress_history[-2].get('tube_var_sum', 0)
            if recent >= prev:
                recs.append("• WARNING: Tube variance not decreasing - check if samples are being ingested")

        if not recs:
            recs.append("• System operating normally - continue sampling loop")

        return f"""{self.section_header("RECOMMENDATIONS")}
{chr(10).join(recs)}

NEXT STEPS:
  1. Run BICEP at the selected target points
  2. Ingest results with: python active_round.py --bicep-results <csv>
  3. Check that tube_var_sum decreases in next round
  4. Repeat until tube is sharply resolved (low variance in [0.4, 0.6])
"""

    def generate(self) -> str:
        """Generate the full report."""
        sections = [
            self.section_system_overview(),
            self.section_current_state(),
            self.section_uncertainty_decomposition(),
            self.section_enn_vs_beta(),
            self.section_plan_analysis(),
            self.section_progress(),
            self.section_recommendations(),
        ]
        return "\n".join(sections)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate round report")
    parser.add_argument("--run-dir", default="artifacts/latest_bicep")
    parser.add_argument("--output", help="Write report to file instead of stdout")
    args = parser.parse_args()

    report = RoundReport.load(Path(args.run_dir))
    text = report.generate()

    if args.output:
        Path(args.output).write_text(text)
        print(f"Report written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
