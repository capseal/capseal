#!/usr/bin/env python3
"""
Example wrapper showing how to embed FusionAlpha's simple Python bindings into a planner class.
This version mirrors what an OGBench integration would do without relying on the removed PyGraph APIs.
"""

from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "target", "release"))

try:
    import fusion_alpha as fa
except ImportError as exc:
    print(f"Warning: fusion_alpha bindings not available: {exc}")
    print("Build with: cargo build --release -p fusion-bindings")
    fa = None


def build_grid_graph(width: int, height: int, cell_size: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
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
    return nodes, edges


def pick_next_node(edges: np.ndarray, q_values: np.ndarray, current: int) -> int:
    candidates = []
    for u, v, _ in edges:
        if int(u) == current:
            candidates.append(int(v))
    if not candidates:
        return current
    candidates.sort()
    best = candidates[0]
    best_q = q_values[best]
    for nid in candidates[1:]:
        if q_values[nid] > best_q + 1e-9 or (abs(q_values[nid] - best_q) <= 1e-9 and nid < best):
            best = nid
            best_q = q_values[nid]
    return best


@dataclass
class GridPlanner:
    width: int = 6
    height: int = 6
    cell_size: float = 1.0
    severity: float = 0.4

    def plan_action(self, obs: Dict[str, float], goal: Dict[str, float]) -> Tuple[int, int, np.ndarray]:
        if fa is None:
            raise RuntimeError("fusion_alpha bindings not available")
        nodes, edges = build_grid_graph(self.width, self.height, self.cell_size)

        def nearest(pos: Tuple[float, float]) -> int:
            dists = np.linalg.norm(nodes - np.array(pos, dtype=np.float32), axis=1)
            return int(np.argmin(dists))

        current_node = nearest((obs['x'], obs['y']))
        goal_node = nearest((goal['x'], goal['y']))

        q_values = fa.simple_propagate(
            nodes=nodes,
            edges=edges,
            goal_node=goal_node,
            current_node=current_node,
            enn_q_prior=0.6,
            severity=self.severity,
            t_max=80,
        )
        next_node = pick_next_node(edges, q_values, current_node)
        action = nodes[next_node] - nodes[current_node]
        return current_node, next_node, action


def main():
    planner = GridPlanner(width=8, height=8, cell_size=0.5)
    obs = {'x': 0.5, 'y': 0.5}
    goal = {'x': 3.5, 'y': 3.5}
    current, nxt, action = planner.plan_action(obs, goal)
    print(f"Current node: {current}, Next node: {nxt}, Suggested displacement: {action}")


if __name__ == "__main__":
    main()
