# Step Relation R for BEF-Stream-Accum

This document describes the step relation `R` used when embedding
BEF-Stream-Accum into an incrementally verifiable computation (IVC) or
PCD framework.

## State

At step `t`, the public state is the BEF trace accumulator:

```
S_t = (n_t, root_t, {r_j}_{j < m}, {s_{t,j}}_{j < m}, {pow_{t,j}}_{j < m})
```

- `n_t`: number of elements processed so far.
- `root_t`: SHA-256 hash/commitment root chaining over chunk roots up to `t`.
- `r_j`: constant challenges derived once from the initial root.
- `s_{t,j}`: sketches `Σ_{i < n_t} v[i] · r_j^i`.
- `pow_{t,j}`: `r_j^{n_t}`, used to stream updates; may be kept as auxiliary
  witness data inside the circuit.

## Step input

Each step consumes a chunk of values `chunk_t = (v_{n_t}, ..., v_{n_t+ℓ-1})`
with `ℓ ≤ L`. The implementation also computes a chunk Merkle root
`chunk_root_t` over these values.

## Relation R

`R(S_t, chunk_t, S_{t+1}) = 1` iff `S_{t+1}` is exactly what
`TraceCommitState::update_with_chunk(S_t, chunk_root_t, chunk_t)` produces:

1. `n_{t+1} = n_t + ℓ`.
2. `root_{t+1} = Poseidon2(root_t, n_t, chunk_root_t)` where `root_t` and
   `chunk_root_t` are interpreted as field elements modulo `2^61-1`.
3. For each challenge `r_j`:
   ```
   s' = s_{t,j}
   pow' = pow_{t,j}
   for value in chunk_t:
       s'   = s'   + value · pow'
       pow' = pow' · r_j
   s_{t+1,j}   = s'
   pow_{t+1,j} = pow'
   ```

In circuit form, this relation costs `O(ℓ · m)` field mul/add gates plus the
hash constraints. Implementing R in an IVC framework means treating this as the
tiny per-step circuit; the heavy data (all v[i]) stays in the BEF accumulator,
which we compute on GPU.
