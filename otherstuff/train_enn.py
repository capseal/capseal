#!/usr/bin/env python3
"""Full Entangled Neural Network (ENN) implementation in PyTorch.

This implements the real ENN architecture from enn-cpp/:
- EntangledCell with PSD entanglement matrix E = L @ L.T
- Collapse with attention mechanism
- MC dropout for epistemic uncertainty
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EntangledCell(nn.Module):
    """Entangled cell with PSD entanglement matrix.

    Forward: ψ = tanh(W_x @ x + W_h @ h + E @ ψ_in - λ * ψ_in + b)
    where E = L @ L.T guarantees positive semi-definiteness.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int,  # Entanglement dimension
        init_lambda: float = 0.1,
        dropout_p: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k

        # Learnable parameters
        self.Wx = nn.Linear(input_dim, k, bias=False)
        self.Wh = nn.Linear(hidden_dim, k, bias=False)
        self.L = nn.Parameter(torch.randn(k, k) * 0.05)  # For E = L @ L.T
        self.b = nn.Parameter(torch.zeros(k))
        self.log_lambda = nn.Parameter(torch.tensor(np.log(max(init_lambda, 1e-6))))

        # Layer norm and dropout
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(k)
        self.dropout = nn.Dropout(p=dropout_p)

        # Initialize weights
        nn.init.xavier_uniform_(self.Wx.weight)
        nn.init.xavier_uniform_(self.Wh.weight)

    @property
    def lambda_val(self) -> torch.Tensor:
        return torch.exp(self.log_lambda)

    @property
    def E(self) -> torch.Tensor:
        """Entanglement matrix (PSD by construction)."""
        return self.L @ self.L.T

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, psi_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [batch, input_dim]
            h: Hidden state [batch, hidden_dim]
            psi_in: Previous entangled state [batch, k]

        Returns:
            psi: New entangled state [batch, k]
        """
        # Compute pre-activation
        E = self.E
        lambda_val = self.lambda_val

        # ψ = tanh(Wx @ x + Wh @ h + E @ ψ_in - λ * ψ_in + b)
        pre_act = (
            self.Wx(x)
            + self.Wh(h)
            + F.linear(psi_in, E)
            - lambda_val * psi_in
            + self.b
        )

        # Optional layer norm
        if self.use_layer_norm:
            pre_act = self.layer_norm(pre_act)

        # Dropout before activation
        pre_act = self.dropout(pre_act)

        # Activation
        psi = torch.tanh(pre_act)
        return psi


class Collapse(nn.Module):
    """Attention-based collapse mechanism.

    Computes: α = softmax((Wq @ ψ) * ψ / temperature)
              collapsed = α * ψ
              output = Wout @ collapsed + b
    """

    def __init__(
        self,
        k: int,
        output_dim: int = 1,
        init_temperature: float = 1.0,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.k = k
        self.output_dim = output_dim

        self.Wq = nn.Linear(k, k, bias=False)
        self.Wout = nn.Linear(k, output_dim)
        self.log_temp = nn.Parameter(torch.tensor(np.log(init_temperature)))
        self.dropout = nn.Dropout(p=dropout_p)

        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.xavier_uniform_(self.Wout.weight)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

    def forward(self, psi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            psi: Entangled state [batch, k]

        Returns:
            output: Prediction [batch, output_dim]
            alpha: Attention weights [batch, k]
        """
        # Gated attention scores
        gated = self.Wq(psi)  # [batch, k]
        scores = gated * psi  # Element-wise [batch, k]

        # Temperature-scaled softmax
        alpha = F.softmax(scores / self.temperature, dim=-1)  # [batch, k]
        alpha = self.dropout(alpha)

        # Collapse
        collapsed = alpha * psi  # [batch, k]

        # Output projection
        output = self.Wout(collapsed)  # [batch, output_dim]
        return output, alpha


class ENN(nn.Module):
    """Full Entangled Neural Network for committor learning.

    Architecture:
    1. Input embedding (optional)
    2. N EntangledCell layers with recurrent ψ state
    3. Collapse to scalar output (committor prediction)
    4. Optional variance head for heteroscedastic uncertainty
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 64,
        k: int = 32,
        n_layers: int = 2,
        dropout_p: float = 0.1,
        init_lambda: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # Input embedding
        self.input_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout_p),
        )

        # Entangled layers
        self.cells = nn.ModuleList([
            EntangledCell(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                k=k,
                init_lambda=init_lambda,
                dropout_p=dropout_p,
            )
            for _ in range(n_layers)
        ])

        # Collapse for committor
        self.collapse = Collapse(k=k, output_dim=1, dropout_p=dropout_p)

        # Variance head (for aleatoric uncertainty)
        self.var_head = nn.Sequential(
            nn.Linear(k, k // 2),
            nn.SiLU(),
            nn.Linear(k // 2, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input coordinates [batch, input_dim]

        Returns:
            q: Committor prediction [batch, 1]
            log_var: Log variance (aleatoric) [batch, 1]
            alpha: Attention weights from collapse [batch, k]
        """
        batch_size = x.shape[0]

        # Embed input
        h = self.input_embed(x)  # [batch, hidden_dim]

        # Initialize entangled state
        psi = torch.zeros(batch_size, self.k, device=x.device)

        # Process through entangled layers
        for cell in self.cells:
            psi = cell(x, h, psi)

        # Collapse to committor
        q_raw, alpha = self.collapse(psi)
        q = torch.sigmoid(q_raw)  # Bound to [0, 1]

        # Variance prediction
        log_var = self.var_head(psi)
        log_var = torch.clamp(log_var, min=-10.0, max=3.0)

        return q, log_var, alpha

    def mc_dropout_inference(
        self, x: torch.Tensor, n_samples: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """MC dropout inference for epistemic uncertainty.

        Returns:
            mean_q: Mean prediction
            var_aleatoric: Mean of learned variance
            var_epistemic: Variance of predictions (model uncertainty)
            entropy: Mean attention entropy (interpretability)
        """
        self.train()  # Enable dropout

        qs = []
        log_vars = []
        alphas = []

        with torch.no_grad():
            for _ in range(n_samples):
                q, log_var, alpha = self.forward(x)
                qs.append(q)
                log_vars.append(log_var)
                alphas.append(alpha)

        qs = torch.stack(qs, dim=0)  # [n_samples, batch, 1]
        log_vars = torch.stack(log_vars, dim=0)
        alphas = torch.stack(alphas, dim=0)

        mean_q = qs.mean(dim=0)
        var_epistemic = qs.var(dim=0)
        var_aleatoric = torch.exp(log_vars).mean(dim=0)

        # Attention entropy (how focused is the collapse?)
        mean_alpha = alphas.mean(dim=0)
        entropy = -(mean_alpha * torch.log(mean_alpha + 1e-10)).sum(dim=-1, keepdim=True)

        self.eval()
        return mean_q, var_aleatoric, var_epistemic, entropy

    def get_entanglement_matrices(self) -> list:
        """Get entanglement matrices from all layers."""
        return [cell.E.detach().cpu().numpy() for cell in self.cells]

    def get_lambda_values(self) -> list:
        """Get learned lambda values from all layers."""
        return [cell.lambda_val.item() for cell in self.cells]


def train_enn(
    model: ENN,
    X: torch.Tensor,
    Y: torch.Tensor,
    W: torch.Tensor,
    epochs: int = 2000,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    print_every: int = 200,
) -> list:
    """Train the ENN with heteroscedastic loss."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        q, log_var, alpha = model(X)

        # Heteroscedastic NLL loss
        loss = torch.mean(W * (((q - Y) ** 2) * torch.exp(-log_var) + log_var))

        # L2 regularization on entanglement matrices (encourage sparsity)
        for cell in model.cells:
            loss = loss + 1e-6 * cell.L.pow(2).sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if epoch % print_every == 0:
            lambdas = model.get_lambda_values()
            lambda_str = ", ".join(f"{l:.3f}" for l in lambdas)
            print(f"Epoch {epoch:>4}/{epochs} | loss={loss.item():>10.6f} | λ=[{lambda_str}]")

    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ENN for committor prediction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--run-dir", type=str, default="artifacts/latest_bicep", help="Output directory")
    parser.add_argument("--data-path", type=str, default="double_well_data.csv", help="Path to training CSV")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)
    print(f"[ENN] Using seed={args.seed}")

    root = Path(__file__).resolve().parent
    data_path = Path(args.data_path)
    if not data_path.is_absolute():
        data_path = root / data_path
        
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(f"[ENN] Loading data from {data_path}")
    df = pd.read_csv(data_path)

    x = df[["x", "y"]].to_numpy(dtype=np.float32)
    y = df["q_hat"].to_numpy(dtype=np.float32)
    weights = df["weight"].to_numpy(dtype=np.float32)

    w_norm = weights / np.mean(weights)
    w_norm = np.clip(w_norm, 1e-3, 1e3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X = torch.from_numpy(x).to(device)
    Y = torch.from_numpy(y).unsqueeze(1).to(device)
    W = torch.from_numpy(w_norm).unsqueeze(1).to(device)

    # Create full ENN
    model = ENN(
        input_dim=2,
        hidden_dim=64,
        k=32,
        n_layers=2,
        dropout_p=0.1,
        init_lambda=0.1,
    ).to(device)

    print(f"ENN Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Entanglement dim k: {model.k}")
    print(f"  - Hidden dim: {model.hidden_dim}")
    print(f"  - Layers: {model.n_layers}")

    # Train
    print(f"\nTraining ENN (epochs={args.epochs}, lr={args.lr})...")
    losses = train_enn(model, X, Y, W, epochs=args.epochs, lr=args.lr, print_every=200)

    # MC dropout inference
    print("\nRunning MC dropout inference (30 samples)...")
    model.eval()
    mean_q, var_aleatoric, var_epistemic, entropy = model.mc_dropout_inference(X, n_samples=30)

    q_pred = mean_q.squeeze(1).cpu().numpy()
    var_aleatoric_np = var_aleatoric.squeeze(1).cpu().numpy()
    var_epistemic_np = var_epistemic.squeeze(1).cpu().numpy()
    var_total = var_aleatoric_np + var_epistemic_np
    entropy_np = entropy.squeeze(1).cpu().numpy()

    # Report
    print(f"\nUncertainty decomposition:")
    print(f"  Aleatoric (learned):  mean={var_aleatoric_np.mean():.6f}, max={var_aleatoric_np.max():.6f}")
    print(f"  Epistemic (MC drop):  mean={var_epistemic_np.mean():.6f}, max={var_epistemic_np.max():.6f}")
    print(f"  Attention entropy:    mean={entropy_np.mean():.4f}, max={entropy_np.max():.4f}")

    # Report entanglement properties
    print(f"\nEntanglement properties:")
    for i, (E, lam) in enumerate(zip(model.get_entanglement_matrices(), model.get_lambda_values())):
        eigvals = np.linalg.eigvalsh(E)
        print(f"  Layer {i+1}: λ={lam:.4f}, E eigenvalues: min={eigvals.min():.4f}, max={eigvals.max():.4f}")

    # High epistemic points
    high_epi = var_epistemic_np > np.percentile(var_epistemic_np, 90)
    print(f"  High epistemic (>90th %ile): {high_epi.sum()} points")

    # Save
    run_dir = (root / args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        run_dir / "enn.npz",
        q_enn=q_pred,
        var_enn=var_total,
        var_aleatoric=var_aleatoric_np,
        var_epistemic=var_epistemic_np,
        entropy=entropy_np,
    )
    torch.save(model.state_dict(), run_dir / "enn.pt")

    # Save model config for reproducibility
    config = {
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "k": model.k,
        "n_layers": model.n_layers,
        "dropout_p": model.dropout_p,
        "lambda_values": model.get_lambda_values(),
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    with open(run_dir / "enn_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved ENN to {run_dir}")


if __name__ == "__main__":
    main()
