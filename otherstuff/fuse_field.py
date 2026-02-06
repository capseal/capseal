from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Try to import real FusionAlpha Rust library
FUSION_ALPHA_PATH = Path(__file__).parent / "FusionAlpha/target/release"
sys.path.insert(0, str(FUSION_ALPHA_PATH))

try:
    import fusion_alpha as fa
    USE_RUST_FUSION = True
    print("[FusionAlpha] Using Rust backend")
except ImportError:
    USE_RUST_FUSION = False
    print("[FusionAlpha] Rust backend not available, using Python fallback")
    from fusion_utils import build_knn_graph, conjugate_gradient, estimate_diag_inverse


def build_knn_edges(points: np.ndarray, k: int = 15) -> tuple:
    """Build k-NN graph with RBF edge weights."""
    n = len(points)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree').fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Compute length scale from median distance
    all_dists = distances[:, 1:].flatten()
    length_scale = float(np.median(all_dists))
    sigma = length_scale

    # Build edge list [u, v, weight]
    edges = []
    for i in range(n):
        for j_idx in range(1, k + 1):  # Skip self
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            weight = np.exp(-(dist ** 2) / (sigma ** 2))
            edges.append([i, j, weight])

    return np.array(edges, dtype=np.float32), length_scale


def run_rust_fusion(
    points: np.ndarray,
    q_enn: np.ndarray,
    confidence: np.ndarray,
    k: int = 15,
    t_max: int = 100,
) -> tuple:
    """Run FusionAlpha using Rust backend."""
    edges, length_scale = build_knn_edges(points, k=k)

    # Run propagation
    q_fused = fa.propagate_field(
        nodes=points.astype(np.float32),
        edges=edges,
        priors=q_enn.astype(np.float32),
        confidences=confidence.astype(np.float32),
        severity=0.0,
        t_max=t_max,
    )

    # Estimate variance (simplified - Rust doesn't expose this directly)
    var_fused = np.abs(q_fused - q_enn) + 1e-6

    return q_fused, var_fused, length_scale, t_max


def run_python_fusion(
    points: np.ndarray,
    q_enn: np.ndarray,
    confidence: np.ndarray,
    k: int = 15,
) -> tuple:
    """Run fusion using Python fallback."""
    neighbors, weights, degrees, length_scale = build_knn_graph(points, k=k)

    mu = 0.2
    b_vec = mu * confidence * q_enn
    x0 = q_enn.copy()

    fused, iters, residual = conjugate_gradient(
        neighbors, weights, degrees, mu, confidence, b_vec, x0,
        tol=1e-6, max_iter=4000,
    )

    diag_post = estimate_diag_inverse(
        neighbors, weights, degrees, mu, confidence,
        samples=16, tol=1e-4, max_iter=2000,
    )

    return fused, diag_post, length_scale, iters, residual


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse ENN field with data")
    parser.add_argument("--data-path", type=str, help="Path to training data (merged CSV)")
    args, _ = parser.parse_known_args()

    root = Path(__file__).resolve().parent
    run_dir = root / "artifacts" / "latest_bicep"
    grid = np.load(run_dir / "grid.npz")
    enn = np.load(run_dir / "enn.npz")

    x = grid["x"]
    y = grid["y"]
    q_enn = enn["q_enn"]

    # Get variance - try multiple sources
    if "var_epistemic" in enn.files:
        var = enn["var_epistemic"] + enn.get("var_aleatoric", np.zeros_like(q_enn))
    elif "var_enn" in enn.files:
        var = enn["var_enn"]
    else:
        var = np.ones_like(q_enn) * 0.01

    # Check for data variance from merged CSV or BICEP
    data_var = None
    if args.data_path:
        data_path = Path(args.data_path)
        if data_path.exists():
            print(f"[Fusion] Loading data variance from {data_path}")
            df = pd.read_csv(data_path)
            if "var" in df.columns:
                data_var = df["var"].to_numpy()
    
    if data_var is None:
        # Fallback to legacy bicep.npz
        bicep_path = run_dir / "bicep.npz"
        if bicep_path.exists():
            bicep = np.load(bicep_path)
            if "var_bicep" in bicep.files:
                data_var = bicep["var_bicep"]
    
    if data_var is not None:
         var = np.maximum(var, data_var)

    points = np.stack([x, y], axis=1)

    # Compute confidence from variance
    confidence = 1.0 / (var + 1e-6)
    confidence = np.clip(confidence, 0.1, 1e6).astype(np.float32)

    if USE_RUST_FUSION:
        q_fused, var_fused, length_scale, iters = run_rust_fusion(
            points, q_enn, confidence, k=15, t_max=100
        )
        residual = 0.0  # Rust doesn't expose residual
    else:
        q_fused, var_fused, length_scale, iters, residual = run_python_fusion(
            points, q_enn, confidence, k=15
        )

    np.savez(
        run_dir / "fusion.npz",
        q_fused=q_fused,
        var_fused=var_fused,
        length_scale=length_scale,
        iterations=iters,
        backend="rust" if USE_RUST_FUSION else "python",
    )

    backend = "Rust" if USE_RUST_FUSION else "Python"
    print(
        f"Fused field written to {run_dir}, backend={backend}, "
        f"iterations={iters}, length_scale={length_scale:.3f}"
    )


if __name__ == "__main__":
    main()
