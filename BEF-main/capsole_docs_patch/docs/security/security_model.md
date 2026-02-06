# Capsules + STC Security Model (aligned to current code)

This note records the threat model, binding requirements, and security claims that the verifier enforces. It is hostile‑review oriented: predicates are explicit; assumptions are named.

This file is **descriptive of current behavior**. Normative object formats and encodings live in `docs/spec/*`.

---

## 1. Adversary model

* **Malicious prover** – controls adapter, filesystem, OS; can fabricate traces, proofs, manifests and DA responses. Cannot break collision resistance or backend soundness assumptions.
* **Malicious registry / policy author** – may publish conflicting policies or lie about enforcement rules.
* **Malicious verifier** – may locally downgrade requirements, but cannot forge relay signatures or change canonical capsule bytes without changing hashes.
* **Malicious DA provider** – can withhold chunks, serve garbage, replay old archives; cannot break Merkle binding or relay signatures.

Verifier runtime + hashing are assumed honest.

---

## 2. Binding points (what enforces what)

### Capsule identity hashes

Current code uses canonical bytes (`dag_cbor_compact_fields_v1`) and the following domain tags:

* `payload_hash       = H("BEF_CAPSULE_V1"            || Enc(payload_view))`
* `header_commit_hash = H("CAPSULE_HEADER_COMMIT_V1::" || Enc(header_commit_view))`
* `capsule_hash       = H("CAPSULE_ID_V2::"           || header_commit_hash || payload_hash)`

Mutations without recomputing hashes are caught (payload/header drift).

### Proof parameters + instance binding (critical)

The header pins:
- `air_params_hash`, `fri_params_hash`, `program_hash`, `vk_hash`
- plus `params_hash`, `chunk_meta_hash`, and `row_tree_arity`

The verifier recomputes these from the proof object and compares them to the header.

**Instance binding (current):** the verifier recomputes

```
instance_hash =
  H("CAPSULE_INSTANCE_V1::" ||
    vk_hash || statement_hash || trace_spec_hash || row_root ||
    params_hash || chunk_meta_hash || row_tree_arity ||
    air_params_hash || fri_params_hash || program_hash)
```

and requires the backend verifier to absorb/check it (adapter passes it as `binding_hash`).

### Statement binding

The proof carries a full `StatementV1`. The verifier recomputes its hash and ensures it matches the header. Backend transcript absorption binds the proof to the caller-provided binding.

### Policy binding / assurance

Policy enforcement requires:
- `policy_ref` pinned by `(policy_id, version, policy_hash, track_id)`
- a `manifest_signature` that verifies under a pinned manifest-signer registry

**Manifest signature (current code):** secp256k1 signature over the *anchor digest bytes* (no additional DST prefix).

Registry roots are pinned via a hash of a sorted `id → pubkey` map. Overriding a registry requires supplying both the map and the expected root (fail‑closed).

### DA audit binding

FULL verification requires a relay-issued challenge signed by a pinned relay key. The verifier checks:
- relay key id is present in the pinned relay registry root
- challenge signature is valid
- deterministic sampling opens k indices to the committed `row_root`
- commit→challenge→open ordering is respected by construction (DA challenge excluded from `header_commit_hash`)

### Event log

Event logs are treated as **tamper‑evident**, not authenticated. The verifier checks structural integrity (hash chain, length, ordering) and reports warnings; it does not treat logs as ground truth.

---

## 3. Verification predicate

Let `requested_level ∈ {PROOF_ONLY, POLICY_ENFORCED, FULL}` be selected by the verifier/operator.

Define:

```
proof_ok  := header_ok ∧ VerifyProof(instance_hash, proof) = 1
policy_ok := CheckPolicy(policy_ref, policy_doc, manifests, statement)
acl_ok    := (no ACL) ∨ signer ∈ ACL(policy_id)
da_ok     := VerifyChallenge ∧ SampleAndOpen = 1
```

Then:

- If `¬proof_ok` → `REJECTED`
- Else if `requested_level = PROOF_ONLY` → `PROOF_ONLY`
- Else if `requested_level = POLICY_ENFORCED`:
  - `POLICY_ENFORCED` if `policy_ok ∧ acl_ok` with pinned registries
  - else `REJECTED` (fail‑closed)
- Else if `requested_level = FULL`:
  - `FULL` if `policy_ok ∧ acl_ok ∧ da_ok` with pinned registries and relay
  - else `REJECTED` (fail‑closed)

Event log failures are warnings only.

---

## 4. DA sampling bound (FULL)

Let δ be the fraction of chunks unavailable at audit time, k the number of sampled chunks, and let the remaining terms denote standard advantages against SHA‑256 collision resistance, backend soundness, and signature unforgeability. Under commit→unpredictable‑challenge→open:

```
Pr[cheat] ≤ (1 − δ)^k + Adv_H_crh + Adv_backend + Adv_sig + (other explicitly-modeled terms)
```

(Exact algebraic-check terms depend on which consistency checks you enable; keep them in `docs/security/theorems.md`.)

---

## 5. What this system does *not* claim

- It does not prove what the rows “mean” beyond the declared TraceSpec/Statement.
- It does not provide host attestation or storage durability by itself.
- Event logs are not authenticated unless you add attestation.
