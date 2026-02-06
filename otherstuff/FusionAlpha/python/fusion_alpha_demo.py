#!/usr/bin/env python3
"""
FusionAlpha Python Demo

Demonstrates the current Python bindings (`simple_propagate` and `create_simple_graph`).
Builds a few small graphs directly in Python and runs the committor propagation pipeline.
"""

import numpy as np
import os
import sys

# Add the shared library produced by cargo build -p fusion-bindings --release
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "target", "release"))

try:
    import fusion_alpha as fa
except ImportError as exc:
    print(f"Failed to import fusion_alpha: {exc}")
    print("Build with: cargo build --release -p fusion-bindings")
    sys.exit(1)


def build_chain_graph(length: int = 3) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Chain of `length` nodes equally spaced on the x-axis."""
    nodes = np.array([[float(x), 0.0] for x in range(length)], dtype=np.float32)
    edges = []
    for i in range(length - 1):
        edges.append([i, i + 1, 1.0])
        edges.append([i + 1, i, 1.0])
    edges = np.array(edges, dtype=np.float32)
    return nodes, edges, 0, length - 1


def build_grid_graph(width: int, height: int, cell_size: float = 1.0) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Grid graph on a rectangular lattice."""
    nodes = []
    for j in range(height):
        for i in range(width):
            nodes.append([i * cell_size, j * cell_size])
    nodes = np.array(nodes, dtype=np.float32)
    edges = []
    def node_id(x: int, y: int) -> int:
        return y * width + x
    for y in range(height):
        for x in range(width):
            u = node_id(x, y)
            if x + 1 < width:
                v = node_id(x + 1, y)
                edges.append([u, v, 1.0])
                edges.append([v, u, 1.0])
            if y + 1 < height:
                v = node_id(x, y + 1)
                edges.append([u, v, 1.0])
                edges.append([v, u, 1.0])
    edges = np.array(edges, dtype=np.float32)
    return nodes, edges, node_id(0, 0), node_id(width - 1, height - 1)


def describe_neighbors(edges: np.ndarray, node: int, q_values: np.ndarray) -> str:
    neighbors = []
    for u, v, w in edges:
        if int(u) == node:
            neighbors.append((int(v), float(w)))
    neighbors.sort(key=lambda item: item[0])
    lines = []
    for v, weight in neighbors:
        lines.append(f"  {v}: q={q_values[v]:.3f}, weight={weight:.2f}")
    return "\n".join(lines) if lines else "  (no outgoing edges)"


def run_demo(title: str, nodes: np.ndarray, edges: np.ndarray, current: int, goal: int, severity: float) -> None:
    print(f"\n=== {title} ===")
    print(f"Nodes: {nodes.shape[0]}, Edges: {edges.shape[0]}")
    print(f"Current node: {current}, Goal node: {goal}")
    q_values = fa.simple_propagate(
        nodes=nodes,
        edges=edges,
        goal_node=goal,
        current_node=current,
        enn_q_prior=0.6,
        severity=severity,
        t_max=60,
    )
    print(f"q[current]={q_values[current]:.3f}, q[goal]={q_values[goal]:.3f}")
    print("Neighbors of current node:")
    print(describe_neighbors(edges, current, q_values))


def demo_risk_sensitive(nodes: np.ndarray, edges: np.ndarray, current: int, goal: int) -> None:
    print("\n=== Risk-Sensitive Comparison ===")
    conservative = fa.simple_propagate(nodes, edges, goal, current, 0.6, severity=0.9, t_max=60)
    aggressive = fa.simple_propagate(nodes, edges, goal, current, 0.6, severity=0.1, t_max=60)
    print(f"Conservative q[current]={conservative[current]:.3f}")
    print(f"Aggressive   q[current]={aggressive[current]:.3f}")


def main():
    chain_nodes, chain_edges, c0, g0 = build_chain_graph(length=3)
    run_demo("Chain Graph", chain_nodes, chain_edges, c0, g0, severity=0.5)

    grid_nodes, grid_edges, c1, g1 = build_grid_graph(width=4, height=4, cell_size=1.0)
    run_demo("Grid Graph", grid_nodes, grid_edges, c1, g1, severity=0.3)
    demo_risk_sensitive(grid_nodes, grid_edges, c1, g1)

if __name__ == "__main__":
    main()
