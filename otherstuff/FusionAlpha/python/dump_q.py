#!/usr/bin/env python3
"""Dump deterministic q-values from the simple_propagate binding."""

import argparse
import json
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "target", "release"))

try:
    import fusion_alpha as fa
except ImportError as exc:
    print(f"Failed to import fusion_alpha: {exc}")
    print("Build with: cargo build --release -p fusion-bindings")
    sys.exit(1)


def build_chain(length: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    nodes = np.array([[float(i), 0.0] for i in range(length)], dtype=np.float32)
    edges = []
    for i in range(length - 1):
        edges.append([i, i + 1, 1.0])
        edges.append([i + 1, i, 1.0])
    return nodes, np.array(edges, dtype=np.float32), 0, length - 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump q-values for a chain graph")
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--severity", type=float, default=0.4)
    parser.add_argument("--t-max", type=int, default=60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    nodes, edges, current, goal = build_chain(args.length)
    q_values = fa.simple_propagate(
        nodes=nodes,
        edges=edges,
        goal_node=goal,
        current_node=current,
        enn_q_prior=0.6,
        severity=args.severity,
        t_max=args.t_max,
    )
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump({
            "length": args.length,
            "severity": args.severity,
            "t_max": args.t_max,
            "current": current,
            "goal": goal,
            "q_values": [float(x) for x in q_values],
        }, fh, indent=2)
    print(f"Wrote q-values to {args.output}")


if __name__ == "__main__":
    main()
