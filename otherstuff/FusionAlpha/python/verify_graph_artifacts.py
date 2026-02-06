#!/usr/bin/env python3
"""
Verify FusionAlpha graph artifacts for determinism and consistency.

Usage:
  Strict mode (recommended for replay):
    python verify_graph_artifacts.py --strict graph_v1.json

  Compare two graph JSONs (byte-level ordering not required, but sets must match):
    python verify_graph_artifacts.py --compare graph_v1_a.json graph_v1_b.json

Notes:
- Rebuild mode is intentionally omitted here unless a buffer snapshot and builder inputs
  are available; strict verification covers most reproducibility cases in CI.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple


def _is_sorted_neighbors(neighbors: List[Dict]) -> bool:
    last = (-1, float('-inf'))
    for item in neighbors:
        key = (int(item.get('v', -1)), float(item.get('w', 0.0)))
        if key < last:
            return False
        last = key
    return True


def _normalize_edges(edges: List[Dict]) -> List[Tuple[int, int, float]]:
    norm = []
    for e in edges:
        u = int(e.get('u', -1))
        v = int(e.get('v', -1))
        w = float(e.get('w', 0.0))
        norm.append((u, v, w))
    norm.sort(key=lambda t: (t[0], t[1], t[2]))
    return norm


def _adj_from_edges(n_nodes: int, edges: List[Tuple[int, int, float]]) -> Dict[int, List[Dict]]:
    adj: Dict[int, List[Dict]] = {i: [] for i in range(n_nodes)}
    for u, v, w in edges:
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            adj[u].append({"v": v, "w": w})
    for k in adj:
        adj[k].sort(key=lambda item: (int(item['v']), float(item['w'])))
    return adj


def verify_strict(graph_path: str) -> None:
    with open(graph_path, 'r', encoding='utf-8') as fh:
        graph = json.load(fh)

    nodes = graph.get('nodes')
    edges = graph.get('edges')
    adjacency = graph.get('adjacency')
    current = int(graph.get('current_node', -1))
    goal = int(graph.get('goal_node', -1))

    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise AssertionError('nodes/edges must be lists')
    n_nodes = len(nodes)
    norm_edges = _normalize_edges(edges)
    if any(u < 0 or v < 0 or u >= n_nodes or v >= n_nodes for u, v, _ in norm_edges):
        raise AssertionError('Edge indices out of range')

    # Recompute adjacency and compare
    recomputed_adj = _adj_from_edges(n_nodes, norm_edges)
    if not isinstance(adjacency, dict):
        raise AssertionError('adjacency must be present and a dict')

    for key, nbrs in adjacency.items():
        k = int(key)
        if k < 0 or k >= n_nodes:
            raise AssertionError(f'adjacency key {k} out of range')
        if not isinstance(nbrs, list):
            raise AssertionError(f'adjacency for {k} must be list')
        if not _is_sorted_neighbors(nbrs):
            raise AssertionError(f'neighbors list for {k} is not sorted deterministically')
        # Compare with recomputed
        exp = recomputed_adj[k]
        if len(exp) != len(nbrs):
            raise AssertionError(f'neighbor count mismatch for {k}: {len(nbrs)} vs {len(exp)}')
        for a, b in zip(nbrs, exp):
            if int(a['v']) != int(b['v']) or abs(float(a['w']) - float(b['w'])) > 1e-12:
                raise AssertionError(f'neighbor mismatch for {k}: {a} vs {b}')

    # Index checks
    if current != -1 and not (0 <= current < n_nodes):
        raise AssertionError('current_node out of range')
    if goal != -1 and not (0 <= goal < n_nodes):
        raise AssertionError('goal_node out of range')

    print(f'[OK] Graph verification passed for {graph_path} (n_nodes={n_nodes}, n_edges={len(norm_edges)})')


def compare_graphs(a_path: str, b_path: str) -> None:
    with open(a_path, 'r', encoding='utf-8') as fa:
        a = json.load(fa)
    with open(b_path, 'r', encoding='utf-8') as fb:
        b = json.load(fb)

    a_edges = _normalize_edges(a.get('edges', []))
    b_edges = _normalize_edges(b.get('edges', []))

    if len(a.get('nodes', [])) != len(b.get('nodes', [])):
        raise AssertionError('node counts differ')
    if a_edges != b_edges:
        raise AssertionError('edge sets differ')

    # Optional adjacency checks
    n_nodes = len(a.get('nodes', []))
    a_adj = _adj_from_edges(n_nodes, a_edges)
    b_adj = _adj_from_edges(n_nodes, b_edges)
    if a_adj != b_adj:
        raise AssertionError('adjacency differs after normalization')

    print(f'[OK] Graphs are equivalent: {a_path} == {b_path}')


def main() -> None:
    p = argparse.ArgumentParser(description='Verify FusionAlpha graph artifacts')
    sub = p.add_subparsers(dest='mode', required=True)

    s1 = sub.add_parser('strict', help='Strict verification of a single graph JSON')
    s1.add_argument('graph')

    s2 = sub.add_parser('compare', help='Compare two graph JSONs for equality')
    s2.add_argument('graph_a')
    s2.add_argument('graph_b')

    args = p.parse_args()
    if args.mode == 'strict':
        verify_strict(args.graph)
    elif args.mode == 'compare':
        compare_graphs(args.graph_a, args.graph_b)


if __name__ == '__main__':
    main()

