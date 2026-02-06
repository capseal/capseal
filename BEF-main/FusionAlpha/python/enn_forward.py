import json
import numpy as np
from typing import Tuple, Union

def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function"""
    return np.maximum(x, 0.0)

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))  # Clip for numerical stability

def _softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Softmax with temperature"""
    x = x / max(tau, 1e-6)
    x = x - x.max(axis=-1, keepdims=True)  # Numerical stability
    ex = np.exp(x)
    return ex / np.clip(ex.sum(axis=-1, keepdims=True), 1e-6, None)

class ENNForward:
    """
    Mirrors the minimal 'collapse' forward pass from ENN:
      x ∈ R^d (input features)
      z = ReLU(x @ Wf + bf)             # Hidden layer (h)
      logits = z @ Wc + bc              # Collapse logits (k)
      logits = logits @ L^T             # Apply entanglement
      alpha = softmax(logits / tau)     # Mixture weights (k)
      q0 = sigmoid(alpha @ w_out + b)   # Committor value in (0,1)
      
    Expects JSON weights exported by collapse_committor_train --export-weights
    """
    def __init__(self, weights_json_path: str):
        """Load ENN weights from JSON file"""
        with open(weights_json_path, "r") as f:
            d = json.load(f)
            
        # Load weight matrices
        self.Wf = np.asarray(d["feat_w"], dtype=np.float32)      # (d, h)
        self.bf = np.asarray(d["feat_b"], dtype=np.float32)      # (h,)
        self.Wc = np.asarray(d["collapse_w"], dtype=np.float32)  # (h, k)
        self.bc = np.asarray(d["collapse_b"], dtype=np.float32)  # (k,)
        self.L  = np.asarray(d["L"], dtype=np.float32)           # (k, k)
        self.wo = np.asarray(d["readout_w"], dtype=np.float32)   # (k,)
        self.bo = float(d.get("readout_b", 0.0))
        self.tau = float(d.get("temperature", 1.0))
        
        # Store dimensions
        self.d = d["d"]  # Input dimension
        self.h = d["h"]  # Hidden dimension
        self.k = d["k"]  # Number of collapse states
        
        # Sanity checks
        d_in, h = self.Wf.shape
        h2, k = self.Wc.shape
        assert d_in == self.d, f"Expected d={self.d}, got {d_in}"
        assert h == self.h, f"Expected h={self.h}, got {h}"
        assert self.bf.shape == (h,)
        assert h2 == h
        assert k == self.k, f"Expected k={self.k}, got {k}"
        assert self.bc.shape == (k,)
        assert self.L.shape == (k, k)
        assert self.wo.shape == (k,)
        
        print(f"Loaded ENN weights: d={self.d}, h={self.h}, k={self.k}")

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through ENN collapse head
        
        Args:
            X: (n, d) node features
            
        Returns:
            q0: (n,) committor values in (0,1)
            alpha: (n, k) latent mixture weights
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        n, d = X.shape
        assert d == self.d, f"Expected input dim {self.d}, got {d}"
        
        # Feature projection with ReLU
        Z = _relu(X @ self.Wf + self.bf)      # (n, h)
        
        # Collapse head
        logits = Z @ self.Wc + self.bc        # (n, k)
        
        # Apply entanglement matrix
        logits = logits @ self.L.T             # (n, k)
        
        # Softmax to get mixture weights
        alpha = _softmax(logits, tau=self.tau) # (n, k)
        
        # Final readout to committor
        q0 = _sigmoid(alpha @ self.wo + self.bo)  # (n,)
        
        return q0.astype(np.float32), alpha.astype(np.float32)

    def forward_one(self, x: np.ndarray) -> float:
        """
        Convenience method for single node
        
        Args:
            x: (d,) single node features
            
        Returns:
            q0: scalar committor value in (0,1)
        """
        q0, _ = self.forward(x.reshape(1, -1))
        return float(q0[0])
    
    def compute_severity(self, alpha: np.ndarray) -> float:
        """
        Compute severity from alpha entropy
        
        Args:
            alpha: (n, k) mixture weights from forward pass
            
        Returns:
            severity: scalar in [0, 1] based on entropy
        """
        # Compute entropy of mixture weights
        eps = 1e-12
        alpha_entropy = -(alpha * np.clip(np.log(alpha + eps), -30, 30)).sum(axis=1).mean()
        
        # Normalize by max entropy (log k)
        max_entropy = np.log(self.k)
        severity = float(np.clip(alpha_entropy / max_entropy, 0.0, 1.0))
        
        return severity


# Example usage and testing
if __name__ == "__main__":
    import os
    
    # Path to exported weights (update this)
    weights_path = "../ENNsrc/ENNrust/enn/runs/enn_weights.json"
    
    if os.path.exists(weights_path):
        # Load ENN
        enn = ENNForward(weights_path)
        
        # Test on dummy features
        n_nodes = 5
        X = np.random.randn(n_nodes, enn.d).astype(np.float32)
        
        # Forward pass
        q0, alpha = enn.forward(X)
        
        print(f"\nTest forward pass:")
        print(f"Input shape: {X.shape}")
        print(f"q0 shape: {q0.shape}, values: {q0}")
        print(f"alpha shape: {alpha.shape}")
        print(f"alpha sums: {alpha.sum(axis=1)}")  # Should be all ~1.0
        
        # Compute severity
        severity = enn.compute_severity(alpha)
        print(f"Severity: {severity:.3f}")
        
        # Test single node
        x_single = X[0]
        q0_single = enn.forward_one(x_single)
        print(f"\nSingle node q0: {q0_single:.3f}")
        assert abs(q0_single - q0[0]) < 1e-6, "Single node forward mismatch"
        
        print("\n✅ ENN forward pass working!")
    else:
        print(f"⚠️  Weights file not found at {weights_path}")
        print("Run collapse_committor_train with --export-weights first")