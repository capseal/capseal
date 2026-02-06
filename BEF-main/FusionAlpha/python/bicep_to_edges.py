#!/usr/bin/env python3
"""
Convert BICEP trajectories to edge weights for Fusion Alpha graphs
Reads BICEP parquet files and computes transition statistics
"""

import numpy as np
import pandas as pd
import json
import ast
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import argparse


def _coerce_state_value(value) -> np.ndarray:
    """Convert misc state representations into a 1D numpy array."""
    if isinstance(value, np.ndarray):
        arr = value.astype(np.float32)
    elif isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=np.float32)
    elif isinstance(value, (float, int, np.floating, np.integer)):
        arr = np.array([value], dtype=np.float32)
    elif isinstance(value, str):
        text = value.strip()
        try:
            if text.startswith('[') and text.endswith(']'):
                parsed = ast.literal_eval(text)
                if isinstance(parsed, (list, tuple)):
                    arr = np.asarray(parsed, dtype=np.float32)
                else:
                    arr = np.array([float(parsed)], dtype=np.float32)
            else:
                arr = np.array([float(text)], dtype=np.float32)
        except Exception:
            arr = np.array([0.0], dtype=np.float32)
    else:
        arr = np.array([float(value)], dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.ravel().astype(np.float32)


def _row_state_to_vec(row, columns) -> np.ndarray:
    """Robustly extract a state vector from a DataFrame row.

    Supports BICEP parity parquet (one row per step with a 1D list),
    flattened columns like 'state_0', or falling back to 'input'.
    """
    # Preferred: 'state' column (list-like or scalar)
    if 'state' in columns:
        s = row['state']
        return _coerce_state_value(s)
    # Next: flattened component
    if 'state_0' in columns:
        return _coerce_state_value(row['state_0'])
    # Fallback: use 'input' as proxy state (parity task)
    if 'input' in columns:
        return _coerce_state_value(row['input'])
    # Last resort: try any numeric columns
    numeric_cols = [c for c in columns if isinstance(row[c], (float, int, np.floating, np.integer))]
    if numeric_cols:
        return np.array([row[numeric_cols[0]]], dtype=np.float32)
    raise ValueError("Could not derive state vector from row; missing 'state', 'state_0', or 'input'.")


def discretize_state(state: np.ndarray, bounds: np.ndarray, n_bins: int = 20) -> int:
    """
    Discretize continuous state to grid index
    
    Args:
        state: continuous state vector
        bounds: (2, d) min/max bounds per dimension
        n_bins: discretization resolution per dimension
    
    Returns:
        grid_index: flattened grid index
    """
    # Normalize to [0, 1] per dimension
    normalized = (state - bounds[0]) / (bounds[1] - bounds[0] + 1e-8)
    normalized = np.clip(normalized, 0, 0.999)
    
    # Convert to grid indices
    indices = (normalized * n_bins).astype(int)
    
    # Flatten to single index (for simplicity, just use first 2 dims)
    if len(indices) >= 2:
        return indices[0] * n_bins + indices[1]
    else:
        return int(indices[0])


def compute_transition_stats(trajectories: pd.DataFrame,
                             n_bins: int = 20,
                             dt: float = 0.01) -> Dict:
    """
    Compute transition statistics from BICEP trajectories.

    Works with step-wise Parquet (one row per step) by grouping on
    'sequence_id' and sorting by 'step'.

    Returns dict with:
      - transitions: {(i,j): count}
      - variances: {(i,j): variance}
      - state_bounds: (2, d) array
    """
    if len(trajectories) == 0:
      return {
        'transitions': {},
        'variances': {},
        'state_bounds': [[0.0], [1.0]],
        'n_bins': n_bins,
        'n_states': n_bins * n_bins,
      }

    # Compute bounds from all rows
    cols = trajectories.columns
    states_list = []
    for _, row in trajectories.iterrows():
        v = _row_state_to_vec(row, cols)
        states_list.append(v)
    states = np.vstack(states_list)  # (N, d)
    state_dim = states.shape[1]

    state_bounds = np.array([
        states.min(axis=0),
        states.max(axis=0)
    ], dtype=np.float32)

    # Count node occupancy across the dataset
    node_counts = defaultdict(int)
    for state in states:
        idx = discretize_state(state, state_bounds, n_bins)
        node_counts[int(idx)] += 1

    # Count transitions by sequence
    transitions = defaultdict(int)
    transition_times = defaultdict(list)

    if 'sequence_id' in cols and 'step' in cols:
        grouped = trajectories.groupby('sequence_id', sort=False)
        for _, df_seq in grouped:
            df_seq = df_seq.sort_values('step')
            prev = None
            for _, row in df_seq.iterrows():
                cur = _row_state_to_vec(row, cols)
                if prev is not None:
                    i = discretize_state(prev, state_bounds, n_bins)
                    j = discretize_state(cur, state_bounds, n_bins)
                    if i != j:
                        transitions[(i, j)] += 1
                        transition_times[(i, j)].append(dt)
                prev = cur
    else:
        # Fallback: treat consecutive rows as a single trajectory
        prev = None
        for _, row in trajectories.iterrows():
            cur = _row_state_to_vec(row, cols)
            if prev is not None:
                i = discretize_state(prev, state_bounds, n_bins)
                j = discretize_state(cur, state_bounds, n_bins)
                if i != j:
                    transitions[(i, j)] += 1
                    transition_times[(i, j)].append(dt)
            prev = cur

    # Compute variances
    variances = {}
    for (i, j), times in transition_times.items():
        if len(times) > 1:
            variances[(i, j)] = float(np.var(times))
        else:
            variances[(i, j)] = 0.1  # Default variance

    return {
        'transitions': dict(transitions),
        'variances': dict(variances),
        'state_bounds': state_bounds.tolist(),
        'n_bins': n_bins,
        'n_states': int(n_bins * n_bins),  # 2D grid indexing even for 1D state
        'node_counts': {int(k): int(v) for k, v in node_counts.items()},
        'state_dim': int(state_dim),
    }


def transitions_to_edge_weights(stats: Dict,
                               temperature: float = 1.0) -> List[Tuple[int, int, float]]:
    """
    Convert transition statistics to edge weights
    
    Uses: w_ij = count_ij / max_count * exp(-var_ij / temperature)
    """
    transitions = stats['transitions']
    variances = stats['variances']
    
    if not transitions:
        return []
    
    max_count = max(transitions.values())
    edges = []
    
    for (i, j), count in transitions.items():
        # Skip self-loops for cleaner graphs
        if i == j:
            continue
            
        # Count-based weight
        count_weight = count / max_count
        
        # Variance penalty
        var = variances.get((i, j), 0.1)
        var_weight = np.exp(-var / temperature)
        
        # Combined weight
        weight = count_weight * var_weight
        
        # Store as (i, j, weight)
        edges.append((int(i), int(j), float(weight)))
    
    return edges


def build_spatial_edges(stats: Dict,
                        nodes: np.ndarray,
                        spatial_k: int = 4,
                        spatial_sigma: float = 0.25,
                        min_count: int = 5,
                        weight_scale: float = 0.3,
                        include_full_grid: bool = False) -> List[Tuple[int, int, float]]:
    if spatial_k <= 0:
        return []
    node_counts = stats.get('node_counts', {})
    if include_full_grid:
        visited = list(range(nodes.shape[0]))
    else:
        visited = [idx for idx, cnt in node_counts.items() if cnt >= min_count]
    if len(visited) < 2:
        return []

    coords = nodes[visited]
    edges: List[Tuple[int, int, float]] = []
    for idx, coord in zip(visited, coords):
        dists = np.linalg.norm(coords - coord, axis=1)
        order = np.argsort(dists)
        neighbor_count = 0
        for neighbor_idx in order[1:]:  # skip self at 0
            neighbor_node = visited[neighbor_idx]
            dist = dists[neighbor_idx]
            if dist <= 0:
                continue
            weight = weight_scale * float(np.exp(-dist / max(1e-6, spatial_sigma)))
            edges.append((int(idx), int(neighbor_node), weight))
            edges.append((int(neighbor_node), int(idx), weight))
            neighbor_count += 1
            if neighbor_count >= spatial_k:
                break
    return edges


def merge_edge_sets(*edge_lists: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    edge_map: Dict[Tuple[int, int], float] = {}
    for edges in edge_lists:
        for i, j, w in edges:
            key = (int(i), int(j))
            if key in edge_map:
                edge_map[key] = max(edge_map[key], w)
            else:
                edge_map[key] = w
    return [(i, j, w) for (i, j), w in edge_map.items()]


def create_grid_nodes(n_bins: int, bounds: np.ndarray) -> np.ndarray:
    """Create node positions for discretized grid"""
    nodes = []
    
    for i in range(n_bins):
        for j in range(n_bins):
            # Map back to continuous space
            x = bounds[0, 0] + (i + 0.5) * (bounds[1, 0] - bounds[0, 0]) / n_bins
            y = bounds[0, 1] + (j + 0.5) * (bounds[1, 1] - bounds[0, 1]) / n_bins if bounds.shape[1] > 1 else 0.0
            nodes.append([x, y])
    
    return np.array(nodes, dtype=np.float32)


def load_bicep_dataframe(parquet_path: Optional[str], csv_path: Optional[str]) -> pd.DataFrame:
    if csv_path:
        print(f"Loading BICEP trajectories from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows")
        return df
    if parquet_path:
        print(f"Loading BICEP trajectories from Parquet: {parquet_path}")
        try:
            import polars as pl  # type: ignore
            df = pl.read_parquet(parquet_path).to_pandas()
        except Exception:
            df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} rows")
        return df
    raise ValueError("Must supply either parquet or csv input")


def main():
    parser = argparse.ArgumentParser(description="Convert BICEP trajectories to graph edges")
    parser.add_argument("--bicep-parquet", help="Path to BICEP trajectory parquet")
    parser.add_argument("--bicep-csv", help="Path to BICEP trajectory CSV (parity_data.csv)")
    parser.add_argument("--output", default="bicep_graph.json", help="Output JSON file")
    parser.add_argument("--n-bins", type=int, default=20, help="Discretization resolution")
    parser.add_argument("--temperature", type=float, default=1.0, help="Variance temperature")
    parser.add_argument("--spatial-k", type=int, default=0, help="Add k-NN spatial edges per node (0=disable)")
    parser.add_argument("--spatial-sigma", type=float, default=0.25, help="Spatial decay length for spatial edges")
    parser.add_argument("--spatial-weight", type=float, default=0.3, help="Scaling factor for spatial edges")
    parser.add_argument("--min-count", type=int, default=5, help="Minimum node visits before spatial edges are added")
    parser.add_argument("--spatial-full-grid", action="store_true", help="Include all grid nodes when building spatial edges")
    
    args = parser.parse_args()

    if not args.bicep_parquet and not args.bicep_csv:
        parser.error("Must supply --bicep-parquet or --bicep-csv")

    df = load_bicep_dataframe(args.bicep_parquet, args.bicep_csv)
    
    # Compute transition statistics
    print(f"Computing transition statistics (n_bins={args.n_bins})...")
    stats = compute_transition_stats(df, n_bins=args.n_bins)
    
    n_transitions = len(stats['transitions'])
    print(f"Found {n_transitions} unique transitions")
    
    # Create node positions
    bounds = np.array(stats['state_bounds'])
    nodes = create_grid_nodes(args.n_bins, bounds)

    # Convert to edge weights
    print("Converting to edge weights...")
    transition_edges = transitions_to_edge_weights(stats, temperature=args.temperature)
    spatial_edges = build_spatial_edges(
        stats,
        nodes,
        spatial_k=args.spatial_k,
        spatial_sigma=args.spatial_sigma,
        min_count=args.min_count,
        weight_scale=args.spatial_weight,
        include_full_grid=args.spatial_full_grid,
    )
    edges = merge_edge_sets(transition_edges, spatial_edges)
    print(f"Created {len(edges)} edges")
    
    # Save results
    stats_json = {
        'n_bins': stats['n_bins'],
        'n_states': stats['n_states'],
        'state_bounds': stats['state_bounds'],
        'transitions': [
            {'source': int(i), 'target': int(j), 'count': int(count)}
            for (i, j), count in stats['transitions'].items()
        ],
        'variances': [
            {'source': int(i), 'target': int(j), 'variance': float(var)}
            for (i, j), var in stats['variances'].items()
        ],
        'node_counts': stats.get('node_counts', {}),
        'state_dim': stats.get('state_dim', 1),
    }

    output = {
        'nodes': nodes.tolist(),
        'edges': edges,
        'stats': stats_json,
        'n_nodes': len(nodes),
        'n_edges': len(edges),
    }
    
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved graph to {args.output}")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    if edges:
        w_min = min(e[2] for e in edges)
        w_max = max(e[2] for e in edges)
        print(f"  Edge weight range: [{w_min:.3f}, {w_max:.3f}]")
    else:
        print("  Edge weight range: [n/a]")
    
    # Print example edges
    print("\nExample edges (top 10 by weight):")
    sorted_edges = sorted(edges, key=lambda e: e[2], reverse=True)[:10]
    for i, j, w in sorted_edges:
        print(f"  {i} → {j}: {w:.3f}")


if __name__ == "__main__":
    main()
