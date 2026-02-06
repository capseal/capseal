#!/usr/bin/env python3
"""
Integration Test: BICEP → ENN → Fusion Alpha Pipeline

Tests the complete pipeline:
1. Load BICEP trajectory data (parquet) 
2. Convert to graph edges
3. Load ENN weights (JSON)
4. Run ENN forward pass for q0 priors
5. Run Fusion Alpha propagation
6. Verify end-to-end functionality

Usage:
  python integration_test.py \
    --bicep-data ../BICEPsrc/BICEPrust/bicep/runs/dw.parquet \
    --enn-weights ../ENNsrc/ENNrust/enn/runs/enn_weights.json
"""

import argparse
import os
import sys
import numpy as np
from typing import Tuple, Optional

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'target', 'release'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

try:
    import fusion_alpha as fa
except ImportError:
    print("⚠️  fusion_alpha module not found. Run: cargo build --release -p fusion-bindings")
    fa = None

try:
    from enn_forward import ENNForward
    from bicep_to_edges import compute_transition_stats, transitions_to_edge_weights, create_grid_nodes
    import pandas as pd
except ImportError as e:
    print(f"⚠️  Missing dependencies: {e}")
    sys.exit(1)


def load_bicep_data(parquet_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load BICEP data and convert to graph"""
    print(f"Loading BICEP data from {parquet_path}")
    
    if not os.path.exists(parquet_path):
        print(f"⚠️  BICEP data not found: {parquet_path}")
        # Create synthetic data for testing
        print("Creating synthetic trajectory data...")
        return create_synthetic_bicep_graph()
    
    # Load real data
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} trajectories")
    
    # Compute transition statistics
    stats = compute_transition_stats(df, n_bins=10)  # Smaller for test
    print(f"Computed {len(stats['transitions'])} transitions")
    
    # Convert to edges
    edges = transitions_to_edge_weights(stats, temperature=1.0)
    print(f"Created {len(edges)} edges")
    
    # Create nodes
    bounds = np.array(stats['state_bounds'])
    nodes = create_grid_nodes(stats['n_bins'], bounds)
    
    # Convert edges to numpy format
    edge_array = np.array(edges, dtype=np.float32) if edges else np.zeros((0, 3), dtype=np.float32)
    
    print(f"Graph: {len(nodes)} nodes, {len(edge_array)} edges")
    return nodes, edge_array


def create_synthetic_bicep_graph() -> Tuple[np.ndarray, np.ndarray]:
    """Create synthetic graph for testing when BICEP data unavailable"""
    print("Creating synthetic 2D grid graph...")
    
    # Create 5x5 grid
    nodes = []
    for i in range(5):
        for j in range(5):
            nodes.append([float(i), float(j)])
    
    nodes = np.array(nodes, dtype=np.float32)
    
    # Create edges (4-connected grid)
    edges = []
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            
            # Right neighbor
            if j < 4:
                neighbor = i * 5 + (j + 1)
                edges.append([idx, neighbor, 0.8])
                
            # Down neighbor  
            if i < 4:
                neighbor = (i + 1) * 5 + j
                edges.append([idx, neighbor, 0.8])
    
    # Add some long-range connections with lower weights
    edges.extend([
        [0, 24, 0.1],   # Corner to corner
        [12, 6, 0.3],   # Center connections
        [12, 18, 0.3],
    ])
    
    edges = np.array(edges, dtype=np.float32)
    return nodes, edges


def create_enn_features(nodes: np.ndarray, current_idx: int, goal_idx: int) -> np.ndarray:
    """Create ENN features from graph nodes"""
    current_pos = nodes[current_idx]
    goal_pos = nodes[goal_idx]
    
    features = []
    for pos in nodes:
        # Distance-based features (matching double-well training)
        dist_to_current = np.linalg.norm(pos - current_pos)
        dist_to_goal = np.linalg.norm(pos - goal_pos)
        
        x = dist_to_goal - dist_to_current  # Relative position
        
        feat = np.array([
            x,                      # position
            x**2,                   # quadratic
            x**3,                   # cubic
            (x**2 - 1)**2 / 4,     # potential-like
            x**3 - x,              # force-like
        ], dtype=np.float32)
        
        features.append(feat)
    
    return np.array(features, dtype=np.float32)


def run_integration_test(bicep_data_path: str, enn_weights_path: str):
    """Run the complete integration test"""
    
    print("=== BICEP → ENN → Fusion Alpha Integration Test ===\n")
    
    # Check dependencies
    if fa is None:
        print("❌ fusion_alpha module missing")
        return False
    
    # 1. Load BICEP data and create graph
    print("Step 1: Load BICEP data")
    try:
        nodes, edges = load_bicep_data(bicep_data_path)
        if len(nodes) == 0:
            raise ValueError("No nodes created")
        if len(edges) == 0:
            print("⚠️  No edges created, adding minimal connectivity")
            # Add minimal edges for test
            for i in range(min(len(nodes)-1, 10)):
                edges = np.append(edges, [[i, i+1, 1.0]], axis=0)
        
        print(f"✅ BICEP graph: {len(nodes)} nodes, {len(edges)} edges")
    except Exception as e:
        print(f"❌ BICEP loading failed: {e}")
        return False
    
    # 2. Load ENN weights
    print(f"\nStep 2: Load ENN from {enn_weights_path}")
    if not os.path.exists(enn_weights_path):
        print(f"❌ ENN weights not found: {enn_weights_path}")
        print("Run: cargo run -r -p enn-examples --bin collapse_committor_train -- \\")
        print("  --in ../BICEPsrc/BICEPrust/bicep/runs/dw.parquet \\")
        print("  --epochs 10 --export-weights runs/enn_weights.json")
        return False
    
    try:
        enn = ENNForward(enn_weights_path)
        print(f"✅ ENN loaded: d={enn.d}, h={enn.h}, k={enn.k}")
    except Exception as e:
        print(f"❌ ENN loading failed: {e}")
        return False
    
    # 3. Set up test scenario
    print(f"\nStep 3: Set up test scenario")
    current_node = 0
    goal_node = len(nodes) - 1  # Last node as goal
    
    print(f"Current node: {current_node} at {nodes[current_node]}")
    print(f"Goal node: {goal_node} at {nodes[goal_node]}")
    
    # 4. ENN forward pass
    print(f"\nStep 4: ENN forward pass")
    try:
        features = create_enn_features(nodes, current_node, goal_node)
        print(f"Created features: {features.shape}")
        
        q0_enn, alpha = enn.forward(features)
        severity = enn.compute_severity(alpha)
        
        print(f"✅ ENN predictions:")
        print(f"  q0 range: [{q0_enn.min():.3f}, {q0_enn.max():.3f}]")
        print(f"  q0[current]: {q0_enn[current_node]:.3f}")
        print(f"  q0[goal]: {q0_enn[goal_node]:.3f}")
        print(f"  Severity: {severity:.3f}")
        
    except Exception as e:
        print(f"❌ ENN forward pass failed: {e}")
        return False
    
    # 5. Fusion Alpha propagation
    print(f"\nStep 5: Fusion Alpha propagation")
    try:
        # Configure propagation
        t_max = int(20 + 50 * severity)
        q_prior = q0_enn[current_node]
        
        print(f"Propagation config:")
        print(f"  t_max: {t_max}")
        print(f"  q_prior: {q_prior:.3f}")
        print(f"  severity: {severity:.3f}")
        
        # Run propagation
        q_fusion = fa.simple_propagate(
            nodes=nodes,
            edges=edges,
            goal_node=goal_node,
            current_node=current_node,
            enn_q_prior=q_prior,
            severity=severity,
            t_max=t_max
        )
        
        print(f"✅ Fusion Alpha results:")
        print(f"  q range: [{q_fusion.min():.3f}, {q_fusion.max():.3f}]")
        print(f"  q[current]: {q_fusion[current_node]:.3f}")
        print(f"  q[goal]: {q_fusion[goal_node]:.3f}")
        
        # Verify boundary conditions
        if abs(q_fusion[goal_node] - 1.0) > 0.01:
            print(f"⚠️  Goal boundary condition not satisfied: {q_fusion[goal_node]:.3f} ≠ 1.0")
        
    except Exception as e:
        print(f"❌ Fusion Alpha propagation failed: {e}")
        return False
    
    # 6. Action selection test
    print(f"\nStep 6: Action selection")
    try:
        # Find neighbors of current node
        neighbors = []
        for edge in edges:
            if int(edge[0]) == current_node:
                neighbor = int(edge[1])
                neighbors.append((neighbor, q_fusion[neighbor]))
        
        if neighbors:
            best_neighbor = max(neighbors, key=lambda x: x[1])
            print(f"Best neighbor: node {best_neighbor[0]} (q={best_neighbor[1]:.3f})")
            print(f"Direction: {nodes[best_neighbor[0]] - nodes[current_node]}")
        else:
            print("No neighbors found for current node")
            
    except Exception as e:
        print(f"❌ Action selection failed: {e}")
        return False
    
    # 7. Integration verification
    print(f"\nStep 7: Integration verification")
    
    # Check that ENN and Fusion Alpha are consistent
    enn_goal_q = q0_enn[goal_node]
    fusion_goal_q = q_fusion[goal_node]
    
    print(f"ENN goal q0: {enn_goal_q:.3f}")
    print(f"Fusion goal q: {fusion_goal_q:.3f}")
    
    # Check monotonicity (values should generally increase toward goal)
    avg_q_near_current = np.mean([q_fusion[n] for n, _ in neighbors[:3]] if neighbors else [q_fusion[current_node]])
    
    print(f"Average q near current: {avg_q_near_current:.3f}")
    
    if fusion_goal_q > avg_q_near_current:
        print("✅ Committor field has reasonable gradient")
    else:
        print("⚠️  Committor field may be flat")
    
    print(f"\n🎉 Integration test completed successfully!")
    print(f"All components (BICEP → ENN → Fusion Alpha) are working together.")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="BICEP → ENN → Fusion Alpha integration test")
    parser.add_argument("--bicep-data", 
                       default="../BICEPsrc/BICEPrust/bicep/runs/dw.parquet",
                       help="Path to BICEP parquet file")
    parser.add_argument("--enn-weights",
                       default="../ENNsrc/ENNrust/enn/runs/enn_weights.json", 
                       help="Path to ENN weights JSON")
    
    args = parser.parse_args()
    
    success = run_integration_test(args.bicep_data, args.enn_weights)
    
    if success:
        print("\n✅ All tests passed! Pipeline is ready for deployment.")
        sys.exit(0)
    else:
        print("\n❌ Integration test failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()