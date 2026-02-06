#!/usr/bin/env python3
"""
Rebuilds the training dataset for the ENN by merging:
1. The static initial dataset (with boundary conditions) from double_well_data.csv
2. The dynamic adaptive samples from beta_posteriors.npz

This ensures the ENN trains on the full history of evidence.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def rebuild_dataset(
    run_dir: Path,
    static_csv_path: Path,
    beta_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Merge static and dynamic data into a new training CSV."""
    
    if not static_csv_path.exists():
        raise FileNotFoundError(f"Static data not found: {static_csv_path}")
    if not beta_path.exists():
        raise FileNotFoundError(f"Beta posteriors not found: {beta_path}")

    # Load static data (Base + Boundaries)
    df = pd.read_csv(static_csv_path)
    
    # Load dynamic data (Adaptive samples)
    beta_data = np.load(beta_path)
    alpha_dynamic = beta_data["alpha"]
    beta_dynamic = beta_data["beta"]

    if len(df) != len(alpha_dynamic):
        raise ValueError(
            f"Dimension mismatch: CSV has {len(df)} rows, "
            f"Beta has {len(alpha_dynamic)} rows."
        )

    # Convert static CSV to pseudo-counts
    # q = alpha / (alpha + beta)  => alpha = q * w
    # w = alpha + beta            => beta = w - alpha
    w_static = df["weight"].to_numpy()
    q_static = df["q_hat"].to_numpy()
    
    alpha_static = q_static * w_static
    beta_static = (1.0 - q_static) * w_static

    # Extract observed counts from dynamic posteriors
    # Subtract 1.0 because beta_posteriors initialized with Uniform(1,1) prior
    # We don't want to double-count the prior if the CSV already implies one,
    # or if we just want to treat the CSV as the "strong prior".
    delta_alpha = np.maximum(alpha_dynamic - 1.0, 0.0)
    delta_beta = np.maximum(beta_dynamic - 1.0, 0.0)

    # Merge
    total_alpha = alpha_static + delta_alpha
    total_beta = beta_static + delta_beta
    
    total_weight = total_alpha + total_beta
    
    # Avoid division by zero (though weight should be > 0 if static exists)
    total_weight = np.maximum(total_weight, 1e-6)
    
    new_q = total_alpha / total_weight
    
    # Compute Beta variance: alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
    # Note: total_weight = alpha + beta
    numerator = total_alpha * total_beta
    denominator = (total_weight ** 2) * (total_weight + 1.0)
    new_var = numerator / np.maximum(denominator, 1e-24)

    # Create new DataFrame
    new_df = df.copy()
    new_df["q_hat"] = new_q
    new_df["weight"] = total_weight
    new_df["var"] = new_var
    
    # Calculate stats for logging
    n_changed = np.sum(total_weight > (w_static + 1e-6))
    added_trials = np.sum(delta_alpha + delta_beta)
    
    if output_path is None:
        output_path = run_dir / "training_data_merged.csv"
        
    new_df.to_csv(output_path, index=False)
    
    print(f"[DatasetBuilder] Merged static + dynamic data.")
    print(f"  Static points: {len(df)}")
    print(f"  Points with new samples: {n_changed}")
    print(f"  Total new trials added: {added_trials:.1f}")
    print(f"  Output: {output_path}")

    return output_path

if __name__ == "__main__":
    # Standalone usage
    import sys
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = Path("artifacts/latest_bicep")
        
    rebuild_dataset(
        run_dir=root,
        static_csv_path=Path("double_well_data.csv").resolve(),
        beta_path=root / "beta_posteriors.npz"
    )
