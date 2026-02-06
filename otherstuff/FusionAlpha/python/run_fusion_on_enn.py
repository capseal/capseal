#!/usr/bin/env python3
"""
FusionAlpha Integration Script

Reads ENN predictions from CSV, builds a k-NN graph, and runs global committor propagation.
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# Add shared library path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "target", "release"))

try:
    import fusion_alpha as fa
except ImportError as exc:
    print(f"Failed to import fusion_alpha: {exc}")
    print("Build with: cd FusionAlpha && cargo build --release -p fusion-bindings")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_fusion_on_enn.py <enn_output.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    print(f"Loading {csv_path}...")
    
    # Load ENN output
    df = pd.read_csv(csv_path)
    
    # Load original data to get X back
    original_csv = "double_well_data_fixed.csv"
    if not os.path.exists(original_csv):
        print(f"Error: Original data {original_csv} needed to recover X coordinates.")
        sys.exit(1)
        
    print(f"Loading original data {original_csv} to merge coordinates...")
    df_orig = pd.read_csv(original_csv)
    
    # Merge on sequence_id
    df = pd.merge(df, df_orig[['sequence_id', 'state_0']], on='sequence_id', how='left')
    
    # Now we have state_0 (X) and state_mean (Y)
    coords = df[['state_0', 'state_mean']].to_numpy().astype(np.float32)
    q_enn = df['q_pred'].to_numpy().astype(np.float32)
    
    # Confidence from variance
    variance = df['state_std'].to_numpy().astype(np.float32)
    epsilon = 1e-4
    confidence = 1.0 / (variance + epsilon)
    confidence = np.clip(confidence, 0.1, 1000.0) 
    
    print(f"Building k-NN graph for {len(coords)} nodes...")
    k = 10
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Build edge list [u, v, w]
    edges = []
    sigma_graph = 0.5 
    
    for i in range(len(coords)):
        for j_idx, idx in enumerate(indices[i]):
            if i == idx: continue 
            dist = distances[i][j_idx]
            w = np.exp(- (dist**2) / (sigma_graph**2))
            edges.append([i, idx, w])
            
    edges_array = np.array(edges, dtype=np.float32)
    
    print("Running FusionAlpha propagation...")
    
    priors = q_enn
    q_final = fa.propagate_field(
        nodes=coords,
        edges=edges_array,
        priors=priors,
        confidences=confidence.astype(np.float32),
        severity=0.0, 
        t_max=100
    )
    
    print("Propagation complete.")
    
    # Save result
    df['q_fusion'] = q_final
    out_file = "fusion_output.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved fused results to {out_file}")
    
    # === PNG Plotting ===
    print("Generating fusion_surface.png...")
    x = df['state_0'].to_numpy()
    y = df['state_mean'].to_numpy()
    z = df['q_fusion'].to_numpy()
    
    tri = Triangulation(x, y)
    
    plt.figure(figsize=(10, 8))
    plt.tricontourf(tri, z, levels=np.linspace(0, 1, 21), cmap='viridis')
    plt.colorbar(label='Committor Probability (q)')
    plt.scatter(x, y, c=z, s=5, cmap='viridis', edgecolors='none', alpha=0.5)
    plt.title('FusionAlpha: Double Well Committor Field')
    plt.xlabel('X (State 0)')
    plt.ylabel('Y (State Mean)')
    plt.tight_layout()
    plt.savefig('fusion_surface.png', dpi=150)
    print("Saved fusion_surface.png")

if __name__ == "__main__":
    main()
