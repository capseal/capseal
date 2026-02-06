import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path

A_BOUND = 0.9
B_BOUND = -0.9


def drift(state: np.ndarray) -> np.ndarray:
    x, y = state
    return np.array([-4.0 * x * (x * x - 1.0), -2.0 * y])


def simulate_paths(
    n_paths: int,
    dt: float,
    t_max: float,
    sigma: float,
    low: float,
    high: float,
    seed: int = 1337,
):
    rng = np.random.default_rng(seed)
    steps = int(np.ceil(t_max / dt))
    sqrt_dt = np.sqrt(dt)

    starts = rng.uniform(low, high, size=(n_paths, 2))
    paths = []
    hits = []

    for start in starts:
        state = start.copy()
        states = [state.copy()]
        outcome = None

        for _ in range(steps):
            if state[0] > A_BOUND:
                outcome = 1
                break
            if state[0] < B_BOUND:
                outcome = 0
                break

            z = rng.normal(size=2)
            dw = sqrt_dt * z
            state = state + drift(state) * dt + sigma * dw
            states.append(state.copy())

        if outcome is None:
            outcome = 0

        paths.append(np.stack(states))
        hits.append(outcome)

    return paths, np.array(hits, dtype=np.int8)


def save_paths(paths, hits, out_path: Path) -> None:
    lengths = [p.shape[0] for p in paths]
    offsets = np.zeros(len(paths) + 1, dtype=np.int32)
    for i, length in enumerate(lengths, start=1):
        offsets[i] = offsets[i - 1] + length
    points = np.concatenate(paths, axis=0)
    np.savez(out_path, points=points, offsets=offsets, hits=hits)


def plot_field(tri: Triangulation, values: np.ndarray, title: str, label: str, out_path: Path) -> None:
    plt.figure()
    plt.tricontourf(tri, values, levels=40)
    plt.colorbar(label=label)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_difference(
    tri: Triangulation,
    base: np.ndarray,
    other: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    plt.figure()
    diff = other - base
    vmax = np.max(np.abs(diff))
    levels = np.linspace(-vmax, vmax, 41)
    plt.tricontourf(tri, diff, levels=levels, cmap="coolwarm")
    plt.colorbar(label="difference")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def plot_committor_with_paths(
    tri: Triangulation,
    values: np.ndarray,
    paths,
    hits: np.ndarray,
    out_path: Path,
):
    fig, ax = plt.subplots()
    contour = ax.tricontourf(tri, values, levels=40)
    fig.colorbar(contour, label="q_hat")
    ax.set_title("Committor with sample trajectories")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.axvspan(B_BOUND - 0.05, B_BOUND + 0.05, color="purple", alpha=0.15, label="Set B")
    ax.axvspan(A_BOUND - 0.05, A_BOUND + 0.05, color="gold", alpha=0.15, label="Set A")

    colors = {0: "#3B0A45", 1: "#E0B341"}
    for path, hit in zip(paths, hits):
        ax.plot(path[:, 0], path[:, 1], color=colors[int(hit)], linewidth=1.2, alpha=0.9)
        ax.scatter(path[0, 0], path[0, 1], color=colors[int(hit)], s=10)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def load_bundles(root: Path):
    bundle_dir = root / "artifacts" / "latest_bicep"
    grid = np.load(bundle_dir / "grid.npz")
    bicep = np.load(bundle_dir / "bicep.npz")

    data = {
        "x": grid["x"],
        "y": grid["y"],
        "q_bicep": bicep["q_bicep"],
        "var_bicep": bicep["var_bicep"],
    }

    enn_path = bundle_dir / "enn.npz"
    if enn_path.exists():
        enn = np.load(enn_path)
        data["q_enn"] = enn["q_enn"]
        if "var_enn" in enn:
            data["var_enn"] = enn["var_enn"]

    fusion_path = bundle_dir / "fusion.npz"
    if fusion_path.exists():
        fusion = np.load(fusion_path)
        data["q_fused"] = fusion["q_fused"]
        if "var_fused" in fusion:
            data["var_fused"] = fusion["var_fused"]

    return data


def main() -> None:
    root = Path(__file__).resolve().parent

    if (root / "artifacts" / "latest_bicep" / "grid.npz").exists():
        bundle = load_bundles(root)
        x = bundle["x"]
        y = bundle["y"]
        q = bundle["q_bicep"]
        v = bundle["var_bicep"]
    else:
        df = pd.read_csv(root / "double_well_data.csv")
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        q = df["q_hat"].to_numpy()
        v = df["var"].to_numpy()
        bundle = {"x": x, "y": y, "q_bicep": q, "var_bicep": v}

    tri = Triangulation(x, y)

    plot_field(tri, q, "Double well committor estimate (BICEP MF)", "q_hat", root / "bicep_q_hat.png")
    plot_field(
        tri,
        np.log10(v),
        "BICEP uncertainty (log10 var)",
        "log10(var)",
        root / "bicep_var.png",
    )

    paths, hits = simulate_paths(
        n_paths=20,
        dt=0.01,
        t_max=6.0,
        sigma=0.7,
        low=-1.5,
        high=1.5,
        seed=4242,
    )
    save_paths(paths, hits, root / "double_well_paths.npz")
    plot_committor_with_paths(tri, q, paths, hits, root / "bicep_q_hat_paths.png")

    if "q_enn" in bundle:
        q_enn = bundle["q_enn"]
        plot_field(tri, q_enn, "ENN committor estimate", "q_enn", root / "enn_q_hat.png")
        plot_difference(
            tri,
            q,
            q_enn,
            "ENN - BICEP",
            root / "enn_minus_bicep.png",
        )
        if "var_enn" in bundle:
            plot_field(
                tri,
                np.log10(bundle["var_enn"] + 1e-12),
                "ENN uncertainty (log10 var)",
                "log10(var_enn)",
                root / "enn_var.png",
            )

    if "q_fused" in bundle:
        q_fused = bundle["q_fused"]
        prior = bundle.get("q_enn", q)
        plot_field(tri, q_fused, "FusionAlpha propagated field", "q_fused", root / "fusion_q_hat.png")
        plot_difference(
            tri,
            prior,
            q_fused,
            "Fusion - prior",
            root / "fusion_minus_prior.png",
        )
        if "var_fused" in bundle:
            plot_field(
                tri,
                np.log10(bundle["var_fused"] + 1e-12),
                "FusionAlpha posterior variance (log10)",
                "log10(var_fused)",
                root / "fusion_var.png",
            )

    plt.show()


if __name__ == "__main__":
    main()
