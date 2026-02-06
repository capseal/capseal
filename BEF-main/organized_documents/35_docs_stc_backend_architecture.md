# STC as Vector Commitment and Polynomial Commitment Backend

We treat HSSA/STC as a two-layer commitment system:

1. **Vector commitment (VC):** streaming commitment to a trace/evaluation vector.
2. **Polynomial commitment (PC):** plug the VC into an IOP+FRI stack, exactly as
   Merkle trees back classic STARK PCs.

## VC Interface

- `Commit_vec(pp, w)` streams the evaluation vector `w ∈ F^n` into STC buckets,
  producing the public commitment `(n, root, r⃗, s⃗)` plus bucket metadata.
- `Open_vec(pp, C, i)` returns `(w_i, proof_i)` where `proof_i` contains Merkle
  paths inside the bucket tree and from bucket root to the global root.
- `Verify_vec(pp, C, i, w_i, proof_i)` recomputes leaves and Merkle paths to
  ensure inclusion. Sketches stay global; they detect bucket tampering or DA
  faults, not per-index openings.

The only twist vs. vanilla Merkle VC is that STC buckets are uneven/streaming
and every bucket records per-challenge sketch contributions. All hashing remains
SHA-256 (or any CRH).

## PC Interface (FRI-backed)

To commit to a low-degree polynomial `p(X)`:

1. Evaluate `p` over the FRI domain `H` to obtain `w`.
2. Run `Commit_vec` to get the base commitment `C = C^(0)`.
3. During the FRI prover loop, each round’s codeword is also committed with STC,
   giving `C^(1) … C^(r)`.

To open at a point `z`:

1. Run the standard FRI prover and record all query indices.
2. For every queried index in every round, invoke `Open_vec` on the corresponding
   STC commitment to supply `(w_i, proof_i)`.
3. Return `y = p(z)` together with the full FRI transcript plus all STC openings.

The verifier runs `Verify_vec` for every opened index, then executes the usual
FRI algebraic checks to link the base codeword to the final small-degree
polynomial and to enforce `p(z) = y`.

## What’s new vs. stock Merkle+FRI

- **Streaming/GPU:** `Commit_vec` is identical to the CUDA accumulator, so the
  commitment phase consumes huge traces on GPU without ever materializing the
  entire vector in CPU memory.
- **Uneven buckets + sketches:** per-bucket sketches and SHA-256 roots give cheap
  integrity/DA checks and mesh with the DA sampling protocol in
  `docs/hssa_da_protocol.md`.
- **Hash-only/PQ:** No pairings or trusted setup; the PC inherits STARK-style
  post-quantum security.

The algebraic security story is unchanged from Merkle-backed FRI PCs. The
systems story is what changes: STC provides an optimized VC backend tailored to
streaming traces, GPU acceleration, and DA requirements.
