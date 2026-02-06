# STC Polynomial Commitment Backend

This backend exposes STC as a vector commitment suitable for FRI/Σ-style PCs.
It introduces two JSON schemas:

```
// bef_pc_commit_v1
{
  "schema": "bef_pc_commit_v1",
  "length": n,
  "chunk_len": L,
  "num_chunks": K,
  "global_root": "...hex..."
}

// bef_pc_open_v1
{
  "schema": "bef_pc_open_v1",
  "index": i,
  "value": v_i,
  "chunk_index": k,
  "chunk_offset": k*L,
  "leaf_pos": j,
  "leaf_path": ["...", ...],
  "chunk_root": "...",
  "chunk_pos": k,
  "chunk_root_path": ["...", ...]
}
```

The commitment is a two-level Merkle tree:

1. Each chunk (values[k*L : (k+1)*L]) has its own Merkle tree, with leaves hashed as
   `H(offset || local_idx || value)`.
2. Chunk roots are hashed into a top-level Merkle tree whose root is `global_root`.

`stc_pc_cli.py` demonstrates the flow:

```
# Commit a trace
python scripts/stc_pc_cli.py commit-trace code/traces/vm_demo.json pc_commit.json

# Produce an opening using the original trace data
python scripts/stc_pc_cli.py open code/traces/vm_demo.json pc_commit.json 42 pc_open.json

# Verify using only the commitment + opening
python scripts/stc_pc_cli.py verify pc_commit.json pc_open.json
```

The opening size is `O(log L + log K)` hashes, and verification requires only the
commitment + proof. These APIs are the building blocks for the STC+FRI PC used
in the Σ+FS backend.
