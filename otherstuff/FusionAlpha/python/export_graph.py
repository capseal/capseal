#!/usr/bin/env python3
"""Export graph spec and serialized graph for reproducibility."""

import argparse
import json
import numpy as np
from graph_builder import (
    HumanoidMazeGraphBuilder,
    AntSoccerGraphBuilder,
    PuzzleGraphBuilder,
)

BUILDERS = {
    "humanoid": HumanoidMazeGraphBuilder,
    "ant": AntSoccerGraphBuilder,
    "puzzle": PuzzleGraphBuilder,
}


def parse_array(arg: str) -> np.ndarray:
    return np.asarray([float(x) for x in arg.split(',')], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export graph spec + graph JSON")
    parser.add_argument("builder", choices=BUILDERS.keys())
    parser.add_argument("--current", required=True, help="Current state vector, comma-separated")
    parser.add_argument("--goal", required=False, help="Goal state vector, comma-separated")
    parser.add_argument("--spec-out", required=True)
    parser.add_argument("--graph-out", required=True)
    parser.add_argument("--params", nargs="*", help="key=value overrides")
    args = parser.parse_args()

    params = {}
    if args.params:
        for item in args.params:
            key, value = item.split('=', 1)
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value

    builder_cls = BUILDERS[args.builder]
    builder = builder_cls(**{k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in params.items()})
    current = parse_array(args.current)
    goal = parse_array(args.goal) if args.goal else None

    nodes, edges, _, _, spec, graph = builder.build_graph_with_artifacts(current, goal)

    # Attach simple stats for downstream mapping (state_bounds, n_bins)
    try:
        x = np.asarray(nodes, dtype=np.float32)
        x_min = float(np.min(x[:, 0]))
        x_max = float(np.max(x[:, 0]))
        n_bins = int(len(nodes))
        graph.setdefault('stats', {})
        graph['stats']['state_bounds'] = [[x_min], [x_max]]
        graph['stats']['n_bins'] = n_bins
    except Exception:
        pass

    with open(args.spec_out, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
    with open(args.graph_out, "w", encoding="utf-8") as fh:
        json.dump(graph, fh, indent=2)
    print(f"Wrote spec to {args.spec_out} and graph to {args.graph_out}")


if __name__ == "__main__":
    main()
