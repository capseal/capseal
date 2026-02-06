#!/usr/bin/env python3
"""
Demo: ENN → Fusion Alpha integration
Shows how to use ENN forward pass to provide priors for committor propagation
"""

import numpy as np
import sys
import os

# Add path for fusion_alpha module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'target', 'release'))

try:
    import fusion_alpha as fa
except ImportError:
    print("Warning: fusion_alpha not built. Run: cargo build --release -p fusion-bindings")
    fa = None

from enn_forward import ENNForward


def create_simple_maze_graph(width=5, height=5):
    """Create a simple grid maze for testing"""
    nodes = []
    edges = []
    
    # Create grid nodes
    for i in range(height):
        for j in range(width):
            x = float(j)
            y = float(i)
            nodes.append([x, y])
    
    # Create edges (4-connected grid)
    node_id = lambda i, j: i * width + j
    
    for i in range(height):
        for j in range(width):
            current = node_id(i, j)
            
            # Right edge
            if j < width - 1:
                neighbor = node_id(i, j + 1)
                edges.append([current, neighbor, 1.0])
                edges.append([neighbor, current, 1.0])
            
            # Down edge
            if i < height - 1:
                neighbor = node_id(i + 1, j)
                edges.append([current, neighbor, 1.0])
                edges.append([neighbor, current, 1.0])
    
    nodes = np.array(nodes, dtype=np.float32)
    edges = np.array(edges, dtype=np.float32)
    
    return nodes, edges


def node_to_features(nodes, current_idx, goal_idx):
    """
    Convert node positions to features for ENN
    This should match the feature engineering in collapse_committor_train.rs
    """
    current_pos = nodes[current_idx]
    goal_pos = nodes[goal_idx]
    
    features = []
    for i, pos in enumerate(nodes):
        # Distance-based features
        dist_to_current = np.linalg.norm(pos - current_pos)
        dist_to_goal = np.linalg.norm(pos - goal_pos)
        
        # Create feature vector similar to double-well features
        x = dist_to_goal - dist_to_current  # Relative position
        feat = np.array([
            x,                      # position-like
            x**2,                   # quadratic
            x**3,                   # cubic
            (x**2 - 1)**2 / 4,     # potential-like
            x**3 - x,              # force-like
        ], dtype=np.float32)
        
        features.append(feat)
    
    return np.array(features, dtype=np.float32)


def main():
    # Check if weights exist
    enn_weights_path = "../../ENNsrc/ENNrust/enn/runs/enn_weights.json"
    
    if not os.path.exists(enn_weights_path):
        print(f"⚠️  ENN weights not found at {enn_weights_path}")
        print("Please run: cargo run -r -p enn-examples --bin collapse_committor_train -- \\")
        print("  --in ../../BICEPsrc/BICEPrust/bicep/runs/dw.parquet \\")
        print("  --epochs 10 --export-weights runs/enn_weights.json")
        return
    
    if fa is None:
        print("⚠️  fusion_alpha module not available")
        return
    
    print("=== ENN → Fusion Alpha Integration Demo ===\n")
    
    # 1. Create simple maze graph
    nodes, edges = create_simple_maze_graph(5, 5)
    n_nodes = len(nodes)
    
    current_node = 0      # Top-left corner
    goal_node = 24        # Bottom-right corner
    
    print(f"Created {n_nodes}-node grid graph")
    print(f"Start: node {current_node} at {nodes[current_node]}")
    print(f"Goal: node {goal_node} at {nodes[goal_node]}")
    
    # 2. Load ENN and compute features
    enn = ENNForward(enn_weights_path)
    features = node_to_features(nodes, current_node, goal_node)
    
    # 3. Get ENN predictions
    q0_enn, alpha = enn.forward(features)
    severity = enn.compute_severity(alpha)
    
    print(f"\nENN predictions:")
    print(f"  q0 range: [{q0_enn.min():.3f}, {q0_enn.max():.3f}]")
    print(f"  q0[current]: {q0_enn[current_node]:.3f}")
    print(f"  q0[goal]: {q0_enn[goal_node]:.3f}")
    print(f"  Severity: {severity:.3f}")
    
    # 4. Run Fusion Alpha propagation
    print(f"\nRunning Fusion Alpha propagation...")
    
    # Map severity to propagation config
    t_max = int(20 + 80 * severity)
    
    # Use fusion_alpha
    q_fusion = fa.simple_propagate(
        nodes=nodes,
        edges=edges,
        goal_node=goal_node,
        current_node=current_node,
        enn_q_prior=q0_enn[current_node],
        severity=severity,
        t_max=t_max
    )
    
    print(f"  t_max: {t_max} (based on severity)")
    print(f"  Converged values:")
    print(f"  q range: [{q_fusion.min():.3f}, {q_fusion.max():.3f}]")
    print(f"  q[current]: {q_fusion[current_node]:.3f}")
    print(f"  q[goal]: {q_fusion[goal_node]:.3f}")
    
    # 5. Visualize results
    print("\nCommittor field (5x5 grid):")
    for i in range(5):
        row = []
        for j in range(5):
            idx = i * 5 + j
            val = q_fusion[idx]
            if idx == current_node:
                row.append(f"[{val:.2f}]")  # Current position
            elif idx == goal_node:
                row.append(f"<{val:.2f}>")  # Goal
            else:
                row.append(f" {val:.2f} ")
        print("  ".join(row))
    
    # 6. Action selection
    current_neighbors = []
    for edge in edges:
        if int(edge[0]) == current_node:
            neighbor_id = int(edge[1])
            current_neighbors.append((neighbor_id, q_fusion[neighbor_id]))
    
    if current_neighbors:
        best_neighbor = max(current_neighbors, key=lambda x: x[1])
        print(f"\nBest next node: {best_neighbor[0]} (q={best_neighbor[1]:.3f})")
        print(f"Direction: {nodes[best_neighbor[0]] - nodes[current_node]}")
    
    print("\n✅ ENN → Fusion Alpha pipeline working!")


if __name__ == "__main__":
    main()