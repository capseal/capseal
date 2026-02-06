# FusionAlpha GraphSpec v1.0 (Draft)

Scope: Deterministic graph construction contract for FusionAlpha planning.

## JSON Format

```
{
  "schema": "fusion_graph_spec_v1",
  "builder_type": "humanoid_maze" | "ant_soccer" | "puzzle" | "custom",
  "params": {
    "k_neighbors": 10,
    "max_nodes": 100,
    "grid_size": 16,
    "cell_size": 0.5,
    "bfs_max_depth": 3
  },
  "buffer_hash": "<hash of the state buffer snapshot>",
  "adjacency_sorted": true,
  "knn_tie_break": "node_id_asc",
  "node_dim": 2,
  "version": "1.0.0",
  "materialized": {
    "nodes": [[x,y], ...],
    "edges": [[u,v,w], ...]
  }
}
```

## Determinism Requirements

- Adjacency lists are vectors sorted by neighbor node id ascending.
- kNN selection is stable-sorted by (distance asc, node_id asc).
- BFS and grid builders iterate neighbors in deterministic order.
- Edge lists sorted (u asc, then v asc) when persisted.

## Replay

- Given `GraphSpec` and buffer snapshot, materialized graph must be bitwise-identical across runs.

