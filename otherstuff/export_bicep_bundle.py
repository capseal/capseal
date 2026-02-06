from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parent
    run_dir = root / "artifacts" / "latest_bicep"
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(root / "double_well_data.csv")

    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()
    q = df["q_hat"].to_numpy()
    var = df["var"].to_numpy()
    weights = df["weight"].to_numpy()
    n_hi = df["n_hi"].to_numpy(dtype=np.int32)
    n_lo = df["n_lo"].to_numpy(dtype=np.int32)
    dt_hi = df["dt_hi"].to_numpy()
    dt_lo = df["dt_lo"].to_numpy()
    sigma = df["sigma"].to_numpy()
    t_max = df["t_max"].to_numpy()

    np.savez(run_dir / "grid.npz", x=xs, y=ys)
    np.savez(
        run_dir / "bicep.npz",
        q_bicep=q,
        var_bicep=var,
        weights=weights,
        n_hi=n_hi,
        n_lo=n_lo,
        dt_hi=dt_hi,
        dt_lo=dt_lo,
        sigma=sigma,
        t_max=t_max,
    )

    print(f"Wrote artifacts to {run_dir}")


if __name__ == "__main__":
    main()
