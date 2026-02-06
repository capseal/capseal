from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

class SimpleENN(nn.Module):
    """Simple ENN with MC dropout for epistemic uncertainty estimation.

    Outputs:
        mean: Predicted committor q(x), sigmoid-bounded to [0, 1]
        log_var: Log aleatoric variance (learned heteroscedastic noise)

    For epistemic uncertainty, run multiple forward passes with dropout
    enabled and compute variance of predictions.
    """

    def __init__(self, dropout_p: float = 0.1) -> None:
        super().__init__()
        self.dropout_p = dropout_p

        # Network with dropout between layers for MC dropout
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.net(x)
        mean_raw, log_var_raw = out.split(1, dim=-1)
        mean = torch.sigmoid(mean_raw)
        log_var = torch.clamp(log_var_raw, min=-10.0, max=3.0)
        return mean, log_var

    def mc_dropout_inference(
        self, x: torch.Tensor, n_samples: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run MC dropout inference to estimate epistemic uncertainty.

        Args:
            x: Input tensor [N, 2]
            n_samples: Number of stochastic forward passes

        Returns:
            mean_pred: Mean of predictions across samples [N, 1]
            aleatoric_var: Mean of learned variance head [N, 1]
            epistemic_var: Variance of mean predictions across samples [N, 1]
            total_var: aleatoric + epistemic [N, 1]
        """
        self.train()  # Enable dropout

        means = []
        log_vars = []

        with torch.no_grad():
            for _ in range(n_samples):
                mean, log_var = self.forward(x)
                means.append(mean)
                log_vars.append(log_var)

        # Stack: [n_samples, N, 1]
        means_stacked = torch.stack(means, dim=0)
        log_vars_stacked = torch.stack(log_vars, dim=0)

        # Mean prediction across MC samples
        mean_pred = means_stacked.mean(dim=0)

        # Epistemic uncertainty: variance of predictions
        epistemic_var = means_stacked.var(dim=0)

        # Aleatoric uncertainty: mean of variance predictions
        aleatoric_var = torch.exp(log_vars_stacked).mean(dim=0)

        # Total uncertainty
        total_var = aleatoric_var + epistemic_var

        self.eval()  # Restore eval mode
        return mean_pred, aleatoric_var, epistemic_var, total_var


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Simple ENN")
    parser.add_argument("--data-path", type=str, default="double_well_data.csv", help="Path to training CSV")
    parser.add_argument("--run-dir", type=str, default="artifacts/latest_bicep")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    root = Path(__file__).resolve().parent
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = root / data_path
        
    print(f"[SimpleENN] Loading data from {data_path}")
    df = pd.read_csv(data_path)

    x = df[["x", "y"]].to_numpy(dtype=np.float32)
    y = df["q_hat"].to_numpy(dtype=np.float32)
    weights = df["weight"].to_numpy(dtype=np.float32)

    w_norm = weights / np.mean(weights)
    w_norm = np.clip(w_norm, 1e-3, 1e3)

    device = torch.device("cpu")
    X = torch.from_numpy(x).to(device)
    Y = torch.from_numpy(y).unsqueeze(1).to(device)
    W = torch.from_numpy(w_norm).unsqueeze(1).to(device)

    # Model with dropout for MC dropout epistemic uncertainty
    dropout_p = 0.1
    model = SimpleENN(dropout_p=dropout_p).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    epochs = args.epochs
    for epoch in range(epochs):
        model.train()
        optim.zero_grad(set_to_none=True)
        mean, log_var = model(X)
        # Heteroscedastic loss: weighted NLL with learned variance
        loss = torch.mean(W * (((mean - Y) ** 2) * torch.exp(-log_var) + log_var))
        loss.backward()
        optim.step()

        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} loss={loss.item():.6f}")

    # MC dropout inference for epistemic uncertainty
    print("Running MC dropout inference (30 samples)...")
    mc_samples = 30
    mean_pred, aleatoric_var, epistemic_var, total_var = model.mc_dropout_inference(X, n_samples=mc_samples)

    q_pred = mean_pred.squeeze(1).cpu().numpy()
    var_aleatoric = aleatoric_var.squeeze(1).cpu().numpy()
    var_epistemic = epistemic_var.squeeze(1).cpu().numpy()
    var_total = total_var.squeeze(1).cpu().numpy()

    # Report uncertainty decomposition
    print(f"Uncertainty decomposition:")
    print(f"  Aleatoric (learned):  mean={var_aleatoric.mean():.6f}, max={var_aleatoric.max():.6f}")
    print(f"  Epistemic (MC drop):  mean={var_epistemic.mean():.6f}, max={var_epistemic.max():.6f}")
    print(f"  Total:                mean={var_total.mean():.6f}, max={var_total.max():.6f}")

    # Check for high epistemic regions (OOD / undersampled)
    high_epistemic = var_epistemic > np.percentile(var_epistemic, 90)
    print(f"  High epistemic (>90th %ile): {high_epistemic.sum()} points")

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = root / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save all uncertainty components
    np.savez(
        run_dir / "enn.npz",
        q_enn=q_pred,
        var_enn=var_total,  # Total for backward compat
        var_aleatoric=var_aleatoric,
        var_epistemic=var_epistemic,
    )
    torch.save(model.state_dict(), run_dir / "enn.pt")
    print(f"Saved ENN predictions + uncertainty decomposition to {run_dir}")


if __name__ == "__main__":
    main()
