#!/usr/bin/env python3
"""
Simple Fusion Alpha Demo
Tests the working Python bindings
"""

import numpy as np
import sys
import os

# Add the built library to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'target', 'release'))

try:
    import fusion_alpha as fa
except ImportError as e:
    print(f"Failed to import fusion_alpha: {e}")
    print("Make sure to build with: cargo build --release -p fusion-bindings")
    sys.exit(1)

def main():
    print("Simple Fusion Alpha Demo")
    
    # Create test graph
    nodes, edges, current_node, goal_node = fa.create_simple_graph()
    print(f"Created test graph with {nodes.shape[0]} nodes")
    print(f"Current node: {current_node}, Goal node: {goal_node}")
    
    # Test propagation
    enn_q_prior = 0.5
    severity = 0.7  # High uncertainty
    t_max = 50
    
    print(f"Running propagation with severity={severity}, t_max={t_max}")
    
    q_values = fa.simple_propagate(
        nodes=nodes,
        edges=edges, 
        goal_node=goal_node,
        current_node=current_node,
        enn_q_prior=enn_q_prior,
        severity=severity,
        t_max=t_max
    )
    
    print("Results:")
    for i, q in enumerate(q_values):
        print(f"  Node {i}: q = {q:.3f}")
        
    # Verify gradient
    assert q_values[goal_node] >= 0.99, "Goal should have q ≈ 1"
    assert q_values[0] < q_values[1] < q_values[2], "Should have increasing gradient"
    
    print("✅ Simple demo passed!")

if __name__ == "__main__":
    main()