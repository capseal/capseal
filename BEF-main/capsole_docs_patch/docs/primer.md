# Capsules + STC: Primer (aligned to current code)

This primer explains the core object (the capsule), how it binds traces and proofs, and the end‑to‑end flow including Data Availability (DA) auditing.

This file is **non‑normative**: the normative rules live in `docs/spec/*`. It’s written to support hostile review by making the binding surfaces obvious.

---

## Capsule Anatomy

A capsule is a content‑addressed object: a header, a payload, and canonical hashes that give it a stable identity.

**Canonical encoding (normative default):** `dag_cbor_compact_fields_v1`  
(See `docs/spec/02_canonicalization.md`.)

```
Capsule (content-addressed)
┌───────────────────────────────────────────────────────────────────────┐
│ header: {                                                             │
│   schema,                                                              │
│   trace_spec_hash, statement_hash,                                     │
│   row_index_ref: { root: row_root, row_tree_arity },                   │
│   chunk_meta: { chunk_len, num_chunks, ... },                          │
│   proof_system: { air_params_hash, fri_params_hash, program_hash, vk_hash },│
│   params_hash,                                                        │
│   chunk_meta_hash,                                                    │
│   instance_hash,                                                      │
│   policy_ref, da_ref, artifact_manifest_hash, events_log_hash/len      │
│ }                                                                     │
│                                                                       │
│ payload: {                                                            │
│   trace_spec (canonical),                                             │
│   statement  (canonical),                                             │
│   proof artifacts + references,                                       │
│   row archive manifest pointers,                                      │
│   policy view, da_challenge (optional)                                │
│ }                                                                     │
│                                                                       │
│ payload_hash       = H("BEF_CAPSULE_V1"           || Enc(payload_view))│
│ header_commit_hash = H("CAPSULE_HEADER_COMMIT_V1::"|| Enc(header_commit_view))│
│ capsule_hash       = H("CAPSULE_ID_V2::"          || header_commit_hash || payload_hash)│
└───────────────────────────────────────────────────────────────────────┘
```

Notes:
- The verifier hashes **canonical bytes**, not parsed JSON.
- The DA challenge is excluded from `header_commit_hash` to preserve the commit→challenge acyclicity.

---

## Binding Surfaces

### Trace + statement
```
TraceSpecV1  --hash-->  trace_spec_hash
StatementV1  --hash-->  statement_hash
STC rows     --Merkle--> row_root + chunk_meta (chunk_len, num_chunks, ...)
```

### Proof system pins
The header pins these proof-system identifiers:
- `air_params_hash`
- `fri_params_hash`
- `program_hash`
- `vk_hash`

### Instance binding (what backends MUST bind)

**Current code binds an “extended” instance tuple under the `CAPSULE_INSTANCE_V1::` prefix.**

Conceptually:

```
instance_hash =
  H("CAPSULE_INSTANCE_V1::" ||
    vk_hash || statement_hash || trace_spec_hash || row_root ||
    params_hash || chunk_meta_hash || row_tree_arity ||
    air_params_hash || fri_params_hash || program_hash)
```

This is the value passed into adapters as `binding_hash`, and must be absorbed/checked by every backend verifier.

(For the normative definition, see `docs/spec/04_instance_binding.md`.)

---

## Manifests (policy inputs)

Manifests are hashed into a **manifest anchor** and signed:

```
manifests/*.json  --sha256-->  hashes map
anchor_payload = { schema: "capsule_bench_manifest_anchor_v1", hashes }
anchor_digest  = sha256(Enc(anchor_payload))
```

**Current code verifies a secp256k1 signature over the anchor digest bytes** (no additional domain tag prepended).

---

## DA Commit → Challenge → Open (flow)

```
        (1) commit                          (2) challenge                          (3) open samples
Prover ─────────────► Relay/Registry ─────────────────────────► Verifier ─────────────► Provider
         header_commit_hash,               signed challenge{ relay_pubkey_id,         fetch indices k
         chunk_handles_root                capsule_commit_hash, nonce, expiry }       verify Merkle paths

seed = H("BEF_AUDIT_SEED_V1" || capsule_hash || challenge_bytes)
indices = deterministic_sample(seed, total=num_chunks, k)
```

The verifier accepts only challenges signed by a relay key included in a pinned relay registry root.

---

## End‑to‑End Verification (high level)

1) Capsule integrity: payload/hash/header hashes match; schemas consistent.
2) Spec/statement: recompute `trace_spec_hash`, `statement_hash`; match header.
3) Instance binding: recompute proof-system hashes from the proof object; recompute `instance_hash`; feed to backend verify.
4) Row commitment: chunk_len/num_chunks/tree_arity consistent; sampled chunks open to `row_root` (if DA audited).
5) Manifests/policy: recompute anchor digest; verify manifest signature against pinned registry; evaluate policy.
6) Authorship/ACL: signature over `capsule_hash`; enforce ACL if required by profile.

---

## Threat Model (abridged)

- Adversary can fabricate traces, manifests, and capsules but cannot break SHA‑256 or backend soundness.
- Verifier is honest; trust roots (relay + signer registries) are pinned and not attacker‑controlled.
- DA audit requires commit before unpredictable challenge; relay keys are authentic.
