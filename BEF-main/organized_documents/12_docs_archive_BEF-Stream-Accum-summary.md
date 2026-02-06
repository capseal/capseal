# BEF-Stream-Accum Summary

BEF-Stream-Accum is a streaming accumulation scheme for execution traces. The
accumulator state carries a Merkle-style hash root over chunk commitments plus a
small vector of random linear sketches. Each update consumes one chunk of the
trace and mutates the state via a fixed transition rule (`update_with_chunk`).
This state is what we export from the GPU accumulator and what we plan to treat
as the public accumulator inside an accumulation-based IVC/PCD framework.

## Algorithms

- **Setup** picks the field `F_p`, hash `H`, chunk size, and number of
  challenges `m`.
- **Init** produces the empty state `(len=0, root=H("bef-init"), {r_j}, {s_j=0},
  {pow_j=1})` with challenges derived from the initial root.
- **Accumulate** takes `(state_t, chunk)` and applies the deterministic update
  rule: hash the new chunk root into `root`, update every sketch via
  `s_j += Σ chunk[k] · r_j^{offset+k}`, update `pow_j`, and bump `len`.
- **Output** projects the state to the public trace commitment
  `(len, root, {r_j}, {s_j})`.
- **Verify (reference)** replays the entire sequence, calling `Accumulate` for
  each chunk. This is linear in the total trace length and serves as the ground
  truth for correctness.

## Fast verification (`Verify_fast`)

Given the final commitment `com_T` and metadata about each chunk
`(chunk_index, offset, length, root, sketch_vec)`, we can verify consistency
without reprocessing raw values:

1. Sort chunks by offset and ensure they cover `[0, len)` with no gaps.
2. Recompute the commitment root by iteratively applying `hash_root_update` over
   the chunk roots and check it equals `com_T.root`.
3. Sum the per-chunk sketch vectors and ensure the result matches
   `com_T.sketches`.

This runs in `O(K · m)` time for `K` chunks and `m` challenges. It detects any
chunk-level tampering (bad root, bad sketch vector, broken coverage) while the
reference verifier remains available if we need to replay the trace.

## Implementation status

- The GPU accumulator emits the per-chunk metadata and the final commitment in
  JSON (`sketch_trace.py`).
- `TraceCommitState` and `bef_verify_fast` are implemented in the `bicep-crypto`
  crate.
- Automated tests load the sample JSON (`code/sketches/trace_demo_sketch.json`)
  and ensure the fast verifier accepts the honest transcript and rejects
  tampered chunk roots/sketches.
- Benchmarking hooks are ready: `demo_cuda(...)` reports accumulate timings,
  while `bef_verify_fast` and the reference replay give CPU-side verification
  costs. These metrics will feed into the accumulation-based IVC/PCD story.

The next step is to expose the BEF state/transition as the public accumulator in
an IVC/folding backend, so that the heavy trace processing stays in the GPU
layer while a generic accumulation/PCD framework handles succinct proofs.


## Soundness experiment (informal)

We view BEF-Stream-Accum as an accumulation scheme for the step predicate
“`state_out` is the result of applying `update_with_chunk(state_in, chunk)`.”
Let `Verify` denote the reference verifier that replays every step.

Define the experiment `Exp_BEFSound(A)`:

1. Challenger runs `pp ← Setup(1^λ)`.
2. Adversary outputs a final state `a_T` and a sequence of instances
   `{x_t} = {(state_in_t, chunk_t, state_out_t)}`.
3. Challenger computes `b = Verify(pp, a_T, {x_t})`.

A wins if `b = 1` but there exists some `t` such that
`state_out_t ≠ update_with_chunk(state_in_t, chunk_t)`.

We require that every PPT adversary wins with at most negligible probability in λ,
assuming collision resistance of H (SHA-256) and the usual large-field behavior
for the sketches. `Verify_fast` is a derived verifier operating on chunk metadata;
its correctness is relative to the honesty of those chunk summaries.


## Reference vs Fast Verification

- `Verify(pp, a_T, {x_t})`: the reference verifier that replays all steps (linear in the
  total trace length) and defines the canonical soundness notion.
- `Verify_fast(pp, com_T, {chunk_summaries})`: a metadata-only checker that ensures chunks
  cover `[0, len)`, the recomputed root matches `com_T.root`, and the sum of per-chunk
  sketches matches `com_T.sketches`. It runs in `O(K·m)` for `K` chunks and `m` challenges.

Soundness of BEF-Stream-Accum is defined with respect to `Verify`; `Verify_fast`
provides a practical consistency check when chunk summaries are available.


## Benchmark plan

We benchmark three operations on an NVIDIA A100 (40GB) over the field `F_p` with
`p = 2^61 - 1`:

| Trace length (N) | # chunks (K) | # challenges (m) | GPU accumulate (ms) | Verify (replay, ms) | Verify_fast (ms) |
|-----------------:|-------------:|-----------------:|--------------------:|--------------------:|-----------------:|
| 1e4              | 10           | 2                | TBD                 | TBD                 | TBD             |
| 1e6              | 100          | 4                | TBD                 | TBD                 | TBD             |
| 1e7              | 100          | 4                | TBD                 | TBD                 | TBD             |

- **GPU accumulate**: streaming accumulator on CUDA (r^i build + chunk sketches).
- **Verify (replay)**: CPU replay of `update_with_chunk` for every chunk.
- **Verify_fast**: metadata-only check using chunk roots and sketch vectors.

`Verify_fast` scales with `K·m` rather than `N`, while GPU accumulate leverages
massive parallelism. Actual numbers will be filled in once the A100 runs are
recorded.
