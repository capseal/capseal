# Capsules + STC: Primer

This primer explains the core primitive (the capsule), how it binds traces and proofs, and the end‑to‑end flow including Data Availability (DA) auditing. It is written to support hostile review: what is bound, in what order, and how it is verified.

## Capsule Anatomy

A capsule is a content‑addressed object: a header, a payload, and canonical hashes that give it a stable identity.

```
Capsule (bef_capsule_v1)
┌───────────────────────────────────────────────────────────────┐
│ header: {                                                     │
│   schema: HEADER_SCHEMA,                                      │
│   trace_spec_hash, statement_hash,                            │
│   row_index_ref: { commitment(root), tree_arity },            │
│   chunk_meta: { chunk_len, num_chunks },                      │
│   proof_system: {                                             │
│     air_params_hash, fri_params_hash, program_hash, vk_hash   │
│   },                                                          │
│   instance_hash = H("CAPSULE_INSTANCE_V1" ||                 │
│                    vk_hash || statement_hash ||               │
│                    trace_spec_hash || row_root),              │
│   policy_ref: { policy_id, version, policy_hash, track_id },  │
│   da_ref, artifact_manifest_hash, events_log_hash/len         │
│ }                                                             │
│                                                               │
│ payload: {                                                    │
│   trace_spec (canonical),                                     │
│   statement  (canonical),                                     │
│   proofs + proof artifacts,                                   │
│   row archive manifest pointers,                              │
│   policy view, da_challenge (optional)                        │
│ }                                                             │
│                                                               │
│ payload_hash        = H("CAPSULE_PAYLOAD_V2" || Enc(payload)) │
│ header_commit_hash  = H("CAPSULE_HEADER_COMMIT_V2" || Enc(..))│
│ capsule_hash        = H("CAPSULE_ID_V2" ||                    │
│                       header_commit_hash || payload_hash)     │
└───────────────────────────────────────────────────────────────┘
```

The capsule may carry an authorship signature (secp256k1 over `capsule_hash`) and an ACL policy for the signer.

## Binding Surfaces

```
TraceSpecV1  --hash-->  trace_spec_hash
StatementV1  --hash-->  statement_hash (binds trace_root, policy_hash, anchors)
STC rows     --merkle--> row_root (k‑ary tree with arity, chunk_len, num_chunks)

Proof System:
  air_params_hash, fri_params_hash, program_hash, vk_hash
  instance_hash = H("CAPSULE_INSTANCE_V1" || vk_hash || statement_hash || trace_spec_hash || row_root)

Backends absorb the binding:
  • STARK (geom): absorb `instance_hash` (or `statement_hash`) into Fiat–Shamir
  • RISC0: journal == binding bytes (equality check in prover/verifier)
```

Manifests (hardware/os/toolchain) are hashed into a *manifest anchor* and then signed:

```
manifests/*.json  --sha256-->  hashes map
anchor_payload = { schema: "capsule_bench_manifest_anchor_v1", hashes }
anchor_ref = "capsulebench_manifest_v1:" + sha256(anchor_payload)

Signature: manifests/manifest_signature.json
  { schema: "capsule_manifest_signature_v1",
    signer_id, signature = secp256k1_recoverable(anchor_ref) }
```

## DA Commit → Challenge → Open (flow)

```
        (1) commit                           (2) challenge                         (3) open samples
Prover ─────────────► Relay/Registry ─────────────────────────► Verifier ─────────────► Provider
         header_commit_hash,               signed challenge{ relay_pubkey_id,         fetch indices k
         chunk_handles_root                capsule_commit_hash, nonce, expiry }       verify Merkle paths

seed = derive_da_seed(capsule_hash, challenge)
indices = deterministic_sample(seed, total=num_chunks, k)
```

The verifier accepts only challenges signed by a relay key included in the pinned relay registry root.

## End‑to‑End Verification

1) Capsule integrity: payload/hash/header hashes match; schemas consistent.
2) Spec/statement: recompute `trace_spec_hash`, `statement_hash`; match header.
3) Instance binding: recompute `air_params_hash`, `fri_params_hash`, `program_hash`, `vk_hash`; build `instance_hash` and feed to backend verifier.
4) Row commitment: chunk_len/num_chunks/tree_arity consistent; sampled chunks open to `row_root` (if DA audited).
5) Manifests/policy: recompute anchor; verify manifest signature against trusted registry; apply policy over signed manifests.
6) Authorship/ACL: signature over `capsule_hash`; evaluate ACL if provided.

The result is a structured verdict: PROOF_ONLY / POLICY_SELF_REPORTED / POLICY_ENFORCED / FULLY_VERIFIED with error codes if any predicate fails.

## Threat Model (abridged)

- Adversary can fabricate traces, manifests, and capsules but cannot break SHA‑256 or STARK/RISC0 soundness.
- Verifier is honest; trust roots (relay + signer registries) are pinned and not attacker‑controlled.
- DA audit requires commit before unpredictable challenge; relay keys are authentic.

## Demo Commands (quick)

```
# Run a signed demo
capsule-bench run \
  --backend geom \
  --policy policies/demo_policy_v1.json \
  --policy-id demo_policy_v1 --policy-version 1.0 \
  --track-id demo_geom_fast \
  --manifest-signer-id test_manifest \
  --manifest-signer-key 3333333333333333333333333333333333333333333333333333333333333333

# Verify
PYTHONPATH=. python scripts/verify_capsule.py \
  out/capsule_runs/<run_id>/pipeline/strategy_capsule.json \
  --policy policies/demo_policy_v1.json \
  --manifest-root out/capsule_runs/<run_id>/manifests
```

For a live demo that shows PASS → TAMPER → REJECT and prints key hashes, use `scripts/demo_run_and_verify.py` (see below).

