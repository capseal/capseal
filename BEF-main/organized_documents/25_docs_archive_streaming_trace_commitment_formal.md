Streaming Trace Commitment (HSSA) — Formal Definition and Error Bounds

0) Streaming Trace Commitment (STC) — Abstract Definition
- Algorithms: (Setup, Init, Update, Commit, Open, VerifyOpen, GlobalCheck) over traces v ∈ F^n.
- Setup(1^λ): outputs pp with field F (|F|=p), CRH H, chunk bound L, sketch count m.
- Init(pp): returns initial state st0.
- Update(pp, st, chunk): deterministically updates state with chunk values; returns st′.
- Commit(pp, st): returns commitment C and public metadata (e.g., n, r⃗, s⃗).
- Open(pp, st, i): returns (val_i, π_i) proving v[i] under C.
- VerifyOpen(pp, C, i, val_i, π_i): deterministic check of opening.
- GlobalCheck(pp, C, meta): probabilistic fast check over per‑chunk metadata (offset/length/root, pre‑shifted sketches) ensuring consistency with a unique underlying trace.

1) Model and Parameters
- Field: a prime field Fp with p = 2^61 − 1 (prototype), extendable to a SNARK‑friendly prime.
- Trace: a length‑N vector v ∈ Fp^N indexed from 0.
- Chunking: values are processed in variable‑length chunks (≤ L). The chunker exposes, per chunk t, a Merkle root root_chunk_t over the chunk values (or H(len || values)).
- Hash: H is a collision‑resistant hash (CRH) with domain separation; we use SHA‑256 in the prototype.
- Challenges: m ≥ 1 Fiat–Shamir challenges r0,…,r{m−1} ∈ Fp\{0}, derived from the transcript (random oracle/ROM idealization).

2) Algorithms (HSSA construction)
Inputs shared across algorithms: public params pp = (p, H, m, L, domain tags).

Init()
- state.len ← 0
- state.root ← H("bef-init" || context)
- For j∈[0..m): r_j ← FE(H(state.root || "bef-challenge" || j)) with rejection sampling for 0
- state.s_j ← 0, state.pow_j ← 1 for all j
- Output state S0 = (len, root, {r_j}, {s_j}, {pow_j})

Update(S, chunk)
- Let ℓ = |chunk|, offset = S.len
- Compute root_chunk = MerkleRoot(chunk) (or H(offset || chunk_bytes))
- S.root ← H(S.root || "bef-chunk" || encode(offset) || root_chunk)
- For j in 0..m−1:
  - s ← S.s_j; pow ← S.pow_j; r ← S.r_j
  - For x in chunk (in order): s ← s + x·pow; pow ← pow·r
  - S.s_j ← s; S.pow_j ← pow
- S.len ← offset + ℓ
- Output updated S′

Commit(S)
- Output commitment com = (len=S.len, root=S.root, challenges={r_j}, sketches={s_j})

Open(v, i, proofs)
- Provide opening at index i: value = v[i], chunk index t containing i, Merkle path leaf→root_chunk_t, and path root_chunk_t→root using the hashed update chain (or a top tree).

VerifyOpen(com, i, value, paths)
- Check Merkle inclusion of value at position i inside its chunk to root_chunk_t.
- Recompute the hash chain root′ by iterating Update’s root accumulation across chunk roots up to t.
- Accept if root′ equals com.root and the path checks succeed.

GlobalCheck(com, summaries)
- summaries = list of per‑chunk metadata {offset_t, length_t, root_chunk_t, sketch_vec_t[0..m)}
- 1) Coverage: sort by offset; ensure they are contiguous, start at 0, sum of lengths equals com.len, and have canonical indices.
- 2) Root recompute: root′ ← H("bef-init"); for each t in order: root′ ← H(root′ || encode(offset_t) || root_chunk_t). Require root′ = com.root.
- 3) Sketch aggregate: for each j, set Ŝ_j ← Σ_t sketch_vec_t[j] and require Ŝ = com.sketches.
- Accept iff all hold. Note: per‑chunk sketches are expected to be pre‑shifted by r_j^{offset_t}. If not, the verifier multiplies each per‑chunk sketch by r_j^{offset_t} before summing.

Correctness
- Honest Init/Update/Commit/Open/VerifyOpen/GlobalCheck accept with probability 1.

3) Security Goals

Binding / Trace Integrity (informal)
- Assuming H is collision‑resistant and the transcript‑derived r_j behave as independent random elements in Fp (ROM), the adversary cannot produce two different traces v≠v′ together with proofs that both (a) pass VerifyOpen for all queried indices and (b) pass GlobalCheck for the same commitment com, except with negligible probability. Intuition: Merkle binding fixes chunk contents and order; linear sketches give a global equality check keyed by r⃗.

Global Sketch Soundness (schwartz–zippel bound)
- Let e = v′ − v ∈ Fp^N be the error vector between two traces with the same length and chunk structure. Define the error polynomial in r:
  E(r) = Σ_{i=0}^{N−1} e[i]·r^i ∈ Fp[r]
- If e ≠ 0, then deg(E) ≤ N−1 and the Schwartz–Zippel lemma yields:
  Pr_{r←Fp\{0}}[E(r) = 0] ≤ deg(E)/(p−1) ≤ (N−1)/(p−1)
- With m independent challenges r0,…,r_{m−1} (via ROM/Fiat‑Shamir with domain separation), the failure probability is at most:
  Pr[∀j, E(r_j)=0] ≤ ((N−1)/(p−1))^m
- If the adversary modifies only a δ fraction of coordinates (|supp(e)| = δN), then deg(E) ≤ N−1 still, and the bound remains ≤ ((N−1)/(p−1))^m. Using a coarser but interpretable bound: ≤ (δN/(p−1))^m when the error support is sized δN.

Instantiation Numbers (prototype)
- p ≈ 2.3058×10^18 (2^61−1). For N = 10^7 and m = 4:
  - (N−1)/(p−1) ≈ 1.0×10^7 / 2.3×10^18 ≈ 4.3×10^−12
  - Failure ≤ (4.3×10^−12)^4 ≈ 3.4×10^−46
- Thus even small m provides extremely strong global integrity detection at large N.

4) Remarks on Fiat–Shamir and Independence
- The prototype derives r_j by hashing the initial root with a per‑index domain tag. In the ROM/QROM model, this is modeled as independent uniform samples in Fp\{0}. If a standard model instantiation is required, sample r_j from an explicit PRF keyed by a public seed or from on‑chain randomness.

5) Notes on Verify_fast vs Full Replay
- Verify_fast (GlobalCheck) guarantees internal consistency of (len, root, {r_j}, {s_j}) with the provided chunk summaries without touching raw values. If chunk summaries themselves are adversarial, Verify_fast can accept only if they collude to match both the hash chain and the sketch sums; this is prevented when summaries are computed by trusted GPU workers or are cross‑checked via sampling and Merkle openings (Open/VerifyOpen) against the raw data.

6) Parameter Guidance
- m (sketches): 2–8 in practice; increases detection confidence exponentially.
- Chunk size L: balances Merkle overhead vs streaming overlap; 1–8k values typical; does not affect bounds.
- Field p: any large prime; 64‑bit Mersenne prime is fast for GPUs; SNARK primes align with IVC backends.

7) API Summary Snapshot (for implementers)
- Commit(v) → (p, len, root, r⃗, s⃗) and optional per‑chunk sketch vectors (pre‑shifted by r^offset).
- Open(i) → (v[i], Merkle paths) to root; independent of sketches.
- GlobalCheck(commit, summaries) → {True, False} (O(K·m) time for K chunks).

8) Proof Sketches (binding)
- If an adversary outputs com for two different traces v≠v′ such that both pass GlobalCheck with their own (consistent) chunk roots and both sets of chunk roots hash to the same com.root, then either (a) H is broken (collision on the root chain), or (b) the sketch vectors match on all m challenges, implying E(r_j)=0 for all j which, by the bound above, occurs with probability ≤ ((N−1)/(p−1))^m. Therefore the scheme is binding except with negligible probability under CRH+ROM.

9) Alternative Security Plug-In (n_max bound and combined advantage)
- For a system-level maximum trace length n_max, the sketch soundness advantage obeys
  Adv_sketch(λ) ≤ ((n_max−1)/p)^m.
- With a b-bit CRH (e.g., SHA-256 truncated to 128 bits), the overall binding advantage satisfies
  Adv_bind(λ) ≤ 2^{−b} + ((n_max−1)/p)^m.
- Example (coarse, round-number framing): p=2^61−1, n_max=2^32, m=4 ⇒ ((n_max−1)/p)^m ≈ (2^{−29})^4 = 2^{−116}.
  With a 128-bit CRH this yields ≈ 2^{−116} + 2^{−128} total binding failure probability.

10) Derived Zero-Knowledge Protocols Using STC
- HSSA–STC itself is intentionally transparent: (C,π) encodes sketch vectors and chunk metadata and is therefore distinguishable for different traces. This is the desired behavior for data-availability and auditability.
- To obtain privacy, treat the STC state as a *public input* to a separate zkSNARK/zkSTARK for a richer relation R_geom (e.g., “there exists an execution trace of an AIR that yields Σ and updates STC state S_old→S_new”).
- The zk layer commits to masked polynomial/codeword evaluations only. In our prototype this means:
  * Deriving per-column masks and a composition-polynomial mask via Fiat–Shamir.
  * Committing (via STC) to the masked vectors and running an FRI-style IOP on top.
  * During verification, subtracting the same deterministic masks before checking that the openings satisfy the AIR constraints.
- This composition yields: STC as a post-quantum streaming VC / AoK primitive, plus an outer transparent zk argument establishing that the committed trace satisfies the intended transition system while leaking only the public invariants (Σ, STC endpoints, parameters). The simulator for the zk layer never needs to break STC; it only programs the random oracle for the masking polynomials.
