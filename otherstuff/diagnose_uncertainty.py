from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "artifacts" / "latest_bicep"
EPS = 1e-10


def load_bundle() -> Dict[str, np.ndarray]:
    grid = np.load(RUN_DIR / "grid.npz")
    bicep = np.load(RUN_DIR / "bicep.npz")

    data = {
        "x": grid["x"],
        "y": grid["y"],
        "q_bicep": bicep["q_bicep"],
        "var_bicep": bicep["var_bicep"],
    }

    enn_path = RUN_DIR / "enn.npz"
    if enn_path.exists():
        enn = np.load(enn_path)
        data.update({"q_enn": enn["q_enn"], "var_enn": enn["var_enn"]})

    fusion_path = RUN_DIR / "fusion.npz"
    if fusion_path.exists():
        fusion = np.load(fusion_path)
        data.update({"q_fused": fusion["q_fused"], "var_fused": fusion["var_fused"]})

    return data


def standardized_residuals(mean: np.ndarray, var: np.ndarray, truth: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.maximum(var, EPS))
    return (truth - mean) / std


def reliability_curve(mean: np.ndarray, truth: np.ndarray, bins: np.ndarray):
    idx = np.digitize(mean, bins) - 1
    centers = []
    pred_avg = []
    truth_avg = []
    counts = []

    for i in range(len(bins) - 1):
        mask = idx == i
        if not np.any(mask):
            continue
        centers.append(0.5 * (bins[i] + bins[i + 1]))
        pred_avg.append(np.mean(mean[mask]))
        truth_avg.append(np.mean(truth[mask]))
        counts.append(np.sum(mask))
    return np.array(centers), np.array(pred_avg), np.array(truth_avg), np.array(counts)


def binned_abs_error(var: np.ndarray, mean: np.ndarray, truth: np.ndarray, n_bins: int = 10):
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(var, quantiles)
    idx = np.digitize(var, edges, right=True) - 1
    centers = []
    errors = []
    counts = []
    for i in range(n_bins):
        mask = idx == i
        if not np.any(mask):
            continue
        centers.append(np.mean(var[mask]))
        errors.append(np.mean(np.abs(mean[mask] - truth[mask])))
        counts.append(np.sum(mask))
    return np.array(centers), np.array(errors), np.array(counts)


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without SciPy.

    Computes Pearson correlation of the ranks (average ranks for ties).
    """
    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(x), dtype=np.float64)
        # average ranks for ties
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=ranks)
        avg = sums / counts
        return avg[inv]

    ra = ranks(a)
    rb = ranks(b)
    ra = (ra - ra.mean()) / (ra.std() + EPS)
    rb = (rb - rb.mean()) / (rb.std() + EPS)
    return float(np.mean(ra * rb))


def gradient_magnitude(points: np.ndarray, values: np.ndarray, k: int = 12) -> np.ndarray:
    n = len(points)
    grad = np.zeros(n, dtype=np.float64)
    for i in range(n):
        diff = points - points[i]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        idx = np.argpartition(dist2, k + 1)[: k + 1]
        idx = idx[idx != i]
        idx = idx[:k]
        neigh = diff[idx]
        delta = values[idx] - values[i]
        if neigh.size == 0:
            grad[i] = 0.0
            continue
        g, *_ = np.linalg.lstsq(neigh, delta, rcond=None)
        grad[i] = float(np.linalg.norm(g))
    return grad


def plot_residual_histograms(residuals: Dict[str, np.ndarray], out_path: Path) -> None:
    plt.figure()
    bins = np.linspace(-4, 4, 80)
    for label, r in residuals.items():
        plt.hist(r, bins=bins, alpha=0.5, density=True, label=label)
    plt.axvline(0.0, color="k", linewidth=1)
    plt.title("Standardized residuals")
    plt.xlabel("r")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_reliability(curves: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], out_path: Path) -> None:
    plt.figure()
    for label, (centers, pred, truth) in curves.items():
        plt.plot(centers, truth, marker="o", label=f"Empirical ({label})")
        plt.plot(centers, pred, linestyle="--", label=f"Pred ({label})")
    plt.plot([0, 1], [0, 1], color="k", linestyle=":", label="perfect")
    plt.xlabel("Predicted mean")
    plt.ylabel("Empirical mean")
    plt.title("Reliability curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_var_error(binned: Dict[str, tuple[np.ndarray, np.ndarray]], out_path: Path) -> None:
    plt.figure()
    for label, (var_centers, errors) in binned.items():
        plt.plot(var_centers, errors, marker="o", label=label)
    plt.xlabel("Predicted variance")
    plt.ylabel("Mean |error|")
    plt.title("Variance vs absolute error")
    plt.xscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_gradient(tri: Triangulation, values: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure()
    plt.tricontourf(tri, values, levels=40)
    plt.colorbar(label="|grad|")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def main() -> None:
    bundle = load_bundle()
    x = bundle["x"]
    y = bundle["y"]
    truth = bundle["q_bicep"]

    tri = Triangulation(x, y)
    points = np.stack([x, y], axis=1)

    outputs = {}
    residuals = {}
    reliability = {}
    var_error = {}

    if "q_enn" in bundle and "var_enn" in bundle:
        outputs["enn"] = bundle["q_enn"]
        residuals["ENN"] = standardized_residuals(bundle["q_enn"], bundle["var_enn"], truth)
        centers, pred, truth_avg, _ = reliability_curve(bundle["q_enn"], truth, np.linspace(0, 1, 11))
        reliability["ENN"] = (centers, pred, truth_avg)
        vc, err, _ = binned_abs_error(bundle["var_enn"], bundle["q_enn"], truth)
        var_error["ENN"] = (vc, err)

    if "q_fused" in bundle and "var_fused" in bundle:
        outputs["fused"] = bundle["q_fused"]
        residuals["Fusion"] = standardized_residuals(bundle["q_fused"], bundle["var_fused"], truth)
        centers, pred, truth_avg, _ = reliability_curve(bundle["q_fused"], truth, np.linspace(0, 1, 11))
        reliability["Fusion"] = (centers, pred, truth_avg)
        vc, err, _ = binned_abs_error(bundle["var_fused"], bundle["q_fused"], truth)
        var_error["Fusion"] = (vc, err)

    if residuals:
        plot_residual_histograms(residuals, ROOT / "residual_hist.png")
    if reliability:
        plot_reliability(reliability, ROOT / "reliability_curve.png")
    if var_error:
        plot_var_error(var_error, ROOT / "variance_error.png")

    for label, preds in outputs.items():
        grad = gradient_magnitude(points, preds)
        plot_gradient(tri, grad, f"|∇q| ({label})", ROOT / f"gradient_{label}.png")

    # Print numeric summaries
    for label, r in residuals.items():
        print(f"{label} residual mean={np.mean(r):.3f}, std={np.std(r):.3f}")

    # --- Transition-tube diagnostics ---
    # Define tube by BICEP committor in (lo, hi)
    lo, hi = 0.1, 0.9
    tube_mask = (truth > lo) & (truth < hi)

    def masked_stats(name: str, pred_key: str, var_key: str) -> Tuple[float, float, float]:
        if pred_key not in bundle or var_key not in bundle:
            return float("nan"), float("nan"), float("nan")
        mu = bundle[pred_key][tube_mask]
        var = bundle[var_key][tube_mask]
        t = truth[tube_mask]
        r = standardized_residuals(mu, var, t)
        # Spearman against |error|
        sp = rank_correlation(var, np.abs(mu - t))
        print(
            f"[tube] {name}: residual mean={np.mean(r):.3f}, std={np.std(r):.3f}, spearman(var,|err|)={sp:.3f}"
        )
        return float(np.mean(r)), float(np.std(r)), float(sp)

    masked_stats("ENN", "q_enn", "var_enn")
    masked_stats("Fusion", "q_fused", "var_fused")

    # Optional: tube-only plots
    if "q_enn" in bundle and "var_enn" in bundle:
        vc, err, _ = binned_abs_error(bundle["var_enn"][tube_mask], bundle["q_enn"][tube_mask], truth[tube_mask])
        plt.figure()
        plt.plot(vc, err, marker="o")
        plt.xscale("log")
        plt.xlabel("Predicted variance (ENN)")
        plt.ylabel("Mean |error| (tube)")
        plt.title("Variance vs |error| (ENN, tube)")
        plt.tight_layout()
        plt.savefig(ROOT / "variance_error_enn_tube.png", dpi=200)

    if "q_fused" in bundle and "var_fused" in bundle:
        vc, err, _ = binned_abs_error(bundle["var_fused"][tube_mask], bundle["q_fused"][tube_mask], truth[tube_mask])
        plt.figure()
        plt.plot(vc, err, marker="o")
        plt.xscale("log")
        plt.xlabel("Predicted variance (Fusion)")
        plt.ylabel("Mean |error| (tube)")
        plt.title("Variance vs |error| (Fusion, tube)")
        plt.tight_layout()
        plt.savefig(ROOT / "variance_error_fusion_tube.png", dpi=200)


if __name__ == "__main__":
    main()
