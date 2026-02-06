Data Availability via Streaming Trace Commitments (STC)

1) Generic DA via STC (abstract)
- Goal: Given a large blob B, a sequencer convinces light clients that (1) data is available and (2) everyone agrees on a single committed version, using a Streaming Trace Commitment (STC) layer.
- STC algorithms: (Init, Update, Commit, Open, VerifyOpen). GlobalCheck is optional for fast probabilistic checks.

- Actors:
  - Sequencer (S): forms blobs and posts commitments.
  - Full nodes: store full blobs, serve openings.
  - Light clients: see on-chain state; run cheap sampling.

- Data model:
  - Encode blob B to field vector v ∈ F^n and partition into chunks chunk_t ∈ F^L for t=0..T−1.

- Commit phase (per blob):
  1) Encoding: B → v ∈ F^n; partition into chunks {chunk_t}.
  2) Streaming commitment: st0 ← Init(pp). For t in order: st_{t+1} ← Update(pp, st_t, chunk_t). Final (C, meta) ← Commit(pp, st_T).
  3) On-chain posting: post C (and minimal metadata), blob ID / index, optional short hash.

- Availability sampling (light client):
  1) Sample k random chunk indices {t_1..t_k} using a public seed (e.g., block hash).
  2) Request openings: for each sampled t, fetch the chunk values and openings π (per-index Merkle paths + chunk inclusion to the committed structure).
  3) Verify: for each i in sampled chunks, VerifyOpen(pp, C, i, v_i, π_i) ?= 1. If any fails → unavailable.
  4) Decision: if all pass, accept availability with soundness error ≤ (1−δ)^k for target corruption fraction δ.

2) HSSA-based DA Instantiation (concrete)
- State after all chunks: st_T = (n, root, (r_j)_{j=1..m}, (s_j)_{j=1..m}, (pow_j)_{j=1..m}).
- Commitment C = (n, root, (r_j), (s_j)).
- Metadata (off-chain or partially on-chain): per-chunk (offset, length, chunkRoot_t, optional per-chunk sketch contributions); global (T, field/param IDs).

- Merkle/hash structure:
  - Each chunk has a Merkle root.
  - Global root is a hash chain (or top Merkle tree) over chunk roots: root_{t+1} = H(root_t || n_t || chunkRoot_t).

- Open / VerifyOpen (index i in chunk t):
  - Open returns (v_i), Merkle path leaf→chunkRoot_t, and chain from chunkRoot_t into root; includes offset/length metadata.
  - VerifyOpen recomputes chunkRoot_t and root, then checks equality with C.root and consistency with n and index i.

- GlobalCheck via sketches (optional, fast):
  - s_j = Σ_{i=0}^{n−1} v_i r_j^i.
  - Reconstruct expected global sketches ŝ_j from per-chunk contributions (or recompute) and offsets.
  - Accept iff ŝ_j = s_j for all j.

3) Security sketch (DA + sketches)
- Sampling error: if δ fraction of chunks are bad and k random chunks are sampled, Pr[miss] ≤ (1−δ)^k.
- Sketch error: for any v≠v′ of length ≤ n_max, with m independent challenges over F_p,
  Pr[(s_j(v)=s_j(v′) for all j)] ≤ ((n_max−1)/p)^m (Schwartz–Zippel).
- Combined (ignoring hash collisions):
  Pr[cheat] ≤ (1−δ)^k + ((n_max−1)/p)^m.
- With a b‑bit CRH for the root chain, add Adv_crh(H) ≈ 2^{−b}:
  Pr[cheat] ≤ Adv_crh(H) + (1−δ)^k + ((n_max−1)/p)^m.

