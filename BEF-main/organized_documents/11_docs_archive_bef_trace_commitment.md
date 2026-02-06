# BEF Trace Commitment

This note specifies the streaming trace/vector commitment that the BEF stack
exposes to the rest of the system.  The commitment is deliberately
GPU-friendly: all heavy work reduces to hashing chunk roots and evaluating
large dot products Σ v[i]·r^i mod p, which we accelerate with the CUDA
accumulator.

- **Trace:** sequence `v ∈ F_p^n` (field element per time step)
- **State:** `(n, root, {r_j}, {s_j}, {pow_j})`
  - `root`: hash/Merkle commitment to the chunk structure
  - `r_j`: hash-derived challenges (Fiat–Shamir over transcript)
  - `s_j = Σ v[i]·r_j^i`: linear sketches for integrity sampling
  - `pow_j = r_j^n`: carried only to support streaming updates
- **Commitment output:** `(n, root, {r_j}, {s_j})`

## Algorithms

### Setup
Choose:
- Field `F_p` (target: SNARK-friendly prime; prototype uses 2^61-1)
- Hash `H : {0,1}* -> {0,1}^{256}` (BLAKE3/SHA-256) with domain separators
- Chunk length `L`, number of challenges `m`
- Merkle layout (per-chunk trees + top tree or hash chain)

Publish `pp` with these parameters.

### Init
```
state.len   = 0
state.root  = H("bef-init" || context)
for j in 0..m-1:
    r_j        = FE( H(state.root || "bef-challenge" || j) )   // reject 0
    s_j        = 0
    pow_j      = 1
```
Return `(len, root, {r_j}, {s_j}, {pow_j})`.

### Update (append chunk values)
Input chunk `(v_len, …, v_len+ℓ-1)` with ℓ ≤ L.
1. Build chunk Merkle root `chunk_root` (or hash of `(len || values)`).
2. `root' = H(root || "bef-chunk" || encode(len) || chunk_root)`.
3. For each challenge j:
   ```
   s_j'   = s_j
   pow_j' = pow_j          // currently r_j^len
   for each value x in chunk:
       s_j'   = s_j'   + x * pow_j'
       pow_j' = pow_j' * r_j
   ```
4. `len' = len + ℓ`.
Return updated state.

### Commit
Stream the trace through Init+Update, then output
`Commitment = (len, root, {r_j}, {s_j})`.
This is what gets persisted / fed into IVC.

### Open & Verify
To open position i:
- Reveal `value = v[i]` plus Merkle paths
  * `proof_chunk`: leaf→chunk_root
  * `proof_top`: chunk_root→root

Verifier:
1. Check Merkle paths vs `root`.
2. (Optional) recompute sampled contributions and compare with `s_j` for global checks.

The sketches give probabilistic detection of bulk tampering:
with m challenges over a 128-bit field the chance of hiding a
non-zero error vector is ≤ (deg/|F|)^m by Schwartz–Zippel.

## Intended Usage
1. **Trace/DA commitment:** replace KZG/FRI for raw blobs with GPU hashes + sketches.
2. **IVC state:** `(len, root, {r_j}, {s_j})` becomes the public state inside a
   folding/Nova/Protostar relation `R` that enforces `Update`.
3. **PCD payload:** nodes exchange `(state, proof)` pairs where `state` embeds
   this commitment and `proof` is the IVC witness of honest evolution.

## Security Notes
- Binding reduces to hash binding (Merkle) + sketch collision bounds.
- Not hiding; all values can be opened.
- Formal reduction/probability analysis TBD once the field and sampling
  distributions are finalized.

## Chunk Summary Format

The GPU accumulator exports per-chunk metadata used by `Verify_fast` and by the
JSON sketch files:

```
{
  "chunk_index": k,
  "offset": start_index,
  "length": chunk_length,
  "root_hex": "...32-byte hex...",
  "sketch_vec": [s_{k,0}, s_{k,1}, ...]
}
```

Each entry contains:

- `chunk_index`: sequential identifier starting at 0.
- `offset` / `length`: describe the covered range in the global trace.
- `root_hex`: Merkle root for the chunk's raw values.
- `sketch_vec`: the chunk's contribution to each sketch after multiplying by
  `r_j^{offset}`. Summing all vectors reproduces the global `s_j` values.

The JSON also includes a `trace_commitment` object with `(len, root_hex,
challenges, sketches)` plus the `commitment_root`. These fields map directly to
`TraceCommitment` and `ChunkSketch` in the code and are the inputs for
`bef_verify_fast`.
