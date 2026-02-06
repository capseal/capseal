# Spec 06 — Protocol (normative)

## Ordering constraints (DA)
1. Prover commits: publishes `(header_commit_hash, chunk_handles_root)` and (optionally) receives a signed commit receipt with timestamp T_c.
2. Challenger issues challenge: signed JSON with `{capsule_commit_hash, seed, k, chunk_len, chunk_tree_arity, issued_at, key_id}`. Challenge time T_ch > T_c.
3. Verifier derives seed and sampling deterministically; provider serves sampled chunks; verifier checks Merkle paths against `row_root`.

`header_commit_hash` MUST exclude the challenge so commit→challenge remains acyclic.

## Messages (canonical JSON)
- Commit: `{capsule_commit_hash, payload_hash, chunk_handles_root, num_chunks}`
- Challenge v1: `{
    scheme: "da_sample_v1",
    capsule_commit_hash,
    seed, k, chunk_len, chunk_tree_arity,
    issued_at,
    issuer: { key_id },
    sig
  }`

Verifier MUST reject if: missing/expired/unsigned challenge; challenge hash mismatches; registry root mismatch; any sampled chunk missing or Merkle path invalid.
