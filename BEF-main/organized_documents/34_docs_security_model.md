# Capsules + STC Security Model

This note records the threat model, binding requirements and security claims that the verifier now enforces. It is designed to be hostile-review ready: every predicate is explicit, and every assumption is called out.

## 1. Adversary model

* **Malicious prover** – controls the adapter, local filesystem and OS, can fabricate traces, proofs, manifests and DA responses. Cannot break collision resistance or the STC/FRI soundness assumptions.
* **Malicious policy author / registry** – may attempt to publish conflicting policies or lie about enforcement rules.
* **Malicious verifier** – may downgrade required levels locally, but cannot forge relay signatures or alter the canonical capsule payload without changing the payload hash.
* **Malicious DA provider** – can withhold chunks, serve garbage data or replay old archives, but cannot break Merkle binding or the relay signature scheme.

The verifier runs on an honest machine: we trust its runtime, hashing, and the coincurve implementation when available.

## 2. Binding points and what now enforces them

* **Capsule identity** – three hashes exist:
  * `payload_hash = H("CAPSULE_PAYLOAD_V2" || Enc(payload\ {authorship, da_challenge}))`
  * `header_commit_hash = H("CAPSULE_HEADER_COMMIT_V2" || Enc(header\ {da_ref.challenge_hash}))`
  * `capsule_hash = H("CAPSULE_ID_V2" || header_commit_hash || payload_hash)`
  The header records `payload_hash`; the outer capsule carries both `payload_hash` and `capsule_hash`. Any mutation to the payload or header that isn’t followed by a recompute is caught as `E011/E013`.
* **Header contents** – `trace_spec_hash`, `statement_hash`, row-commitment metadata, proof-system hashes, `policy_ref`, `da_ref`, `artifact_manifest_hash`, and the `(events_log_hash, events_log_len)` pair. The DA challenge is intentionally excluded from `header_commit_hash` so commit-then-challenge remains acyclic.
* **Proof parameters + instance binding** – `air_params_hash`, `fri_params_hash`, `program_hash`, and `vk_hash` live in the header, along with `instance_hash = H("CAPSULE_INSTANCE_V1" || vk_hash || statement_hash || trace_spec_hash || row_root)`. The verifier recomputes all of these from the proof object (`GeomAIRParams`, `FRIConfig`, program descriptor, row commitment) before calling the backend verifier and fails with `E301_PROOF_SYSTEM_MISMATCH` if any hash differs. Because the proof transcript already absorbs `statement_hash` and the row root, this `instance_hash` prevents statement/trace swaps.
* **Statement binding** – the proof object carries a full `StatementV1`. The verifier recomputes its hash and ensures it matches the header; `zk_verify_geom` also absorbs the caller-provided `statement_hash` into the Fiat–Shamir transcript so a receipt tied to one statement cannot be replayed under another without breaking soundness.
* **Row commitments & manifests** – the Merkle root referenced in the statement is re-derived from the chunk roots and compared to the proof’s `row_commitment`. Chunk manifests are canonicalised, hashed, and every handle is confined to the extracted archive root; the verifier recomputes each chunk’s size and SHA-256 before any DA I/O happens.
* **Policy binding / assurance** – `policy_ref` commits to `(policy_id, version, hash, track_id)`. Policy enforcement now requires a `manifest_signature.json` beside the manifests; it must be a secp256k1 recoverable signature over the manifest anchor hash, signed by a trusted manifest authority listed in `config/manifest_signers.json` (or CLI overrides). The verifier re-derives the registry hash, requires the signer id to be allowed, and returns `E106-E109` otherwise. Merkle inclusion proofs against a trusted policy registry root upgrade `policy_assurance` to `ATTESTED`; without them enforcement is capped at `POLICY_SELF_REPORTED`.
* **Event log** – verification treats the log as a tamper-evident transcript. The predicate is purely structural (hash chain, monotone sequence numbers, required event types once each). It enforces non-equivocation/integrity and emits `E201_EVENT_LOG_MISMATCH` warnings but does **not** block FULL verification because the emitter is unauthenticated.
* **DA audit** – FULL verification is a verifier-chosen predicate; it requires a relay-issued challenge signed over `canonical_encode(challenge\{signature})`, hashed via SHA-256 before secp256k1 signing. The verifier only accepts challenges whose `relay_pubkey_id` appears in a pinned registry hash `CAPSULE_TRUSTED_RELAYS_ROOT` (computed as the SHA-256 of the sorted id→pubkey map). A default registry lives at `config/trusted_relays.json`; overriding it requires supplying both the new key map and the matching root. Missing/expired/unsigned challenges or registry hash mismatches raise `E073` instead of silently downgrading.
* **Authorship/ACL** – any capsule claiming more than PROOF_ONLY must include a secp256k1 recoverable signature over `capsule_hash`. ACLs bind signer pubkeys to policy IDs; if signatures are missing or not in the ACL, status drops to PROOF_ONLY.

## 3. Verification predicate

Let `level ∈ {PROOF_ONLY, POLICY_ENFORCED, FULLY_VERIFIED}` be selected by the *verifier/operator* (the capsule merely advertises an intended profile). Define the booleans:

```
proof_ok  := header_ok ∧ VerifyProof(instance_hash, proof) = 1
policy_ok := CheckPolicy(policy_ref, policy_doc, manifests, statement)
acl_ok    := (no ACL) ∨ signer ∈ ACL(policy_id)
da_ok     := DA audit predicate (challenge verified + sampling succeeds)
```

Then:

```
if ¬proof_ok:              status = REJECTED
elif level = PROOF_ONLY:   status = PROOF_ONLY
elif level = POLICY_ENFORCED:
    status = POLICY_ENFORCED      if policy_ok ∧ acl_ok ∧ policy_assurance=ATTESTED
    status = POLICY_SELF_REPORTED if policy_ok ∧ acl_ok but assurance is self-reported
elif level = FULLY_VERIFIED:
    require policy_ok ∧ acl_ok ∧ da_ok (with attested policy)
    status = FULLY_VERIFIED
else:                       status = E302_VERIFICATION_PROFILE_UNSATISFIED
```

Any missing prerequisite short-circuits with the corresponding `E0xx` code (e.g. `E071/E073` for DA). Event log issues are reported as warnings instead of fatal errors because they are not authenticated data sources.

### DA Sampling Bound (FULL)

Let δ be the (adversarial) fraction of rows unavailable at audit time, k the number of sampled chunks, p the field modulus, n_max the maximum row/trace length relevant to the soundness of the sketch/consistency checks, m the number of independent algebraic checks (e.g. random linear constraints), `Adv_H_crh` the adversary’s advantage against SHA‑256 collision resistance, `Adv_backend` the advantage against the backend’s proof soundness with instance absorption, and `Adv_sig` the advantage against the signature scheme (secp256k1 UF‑CMA). Under the commit→unpredictable‑challenge→open ordering, the acceptance probability for a cheating prover is bounded by:

```
Pr[cheat] ≤ (1 − δ)^k + Adv_H_crh + ((p − 1)/n_max − 1)^m + Adv_backend + Adv_sig
```

This isolates the three levers under operator control: (i) availability via k, (ii) algebraic collision via m (and field size p versus problem scale n_max), and (iii) the trust roots (hash, backend, signatures).

## 4. Attack surfaces & mitigations

| Surface | Mitigation |
| --- | --- |
| Payload/header drift | Structural payload view with explicit key list plus `capsule_hash = H(header_commit || payload_hash)`. Any mutation without recomputing both hashes trips `E011/E013`.
| Proof parameter swaps | Header pins `air/fri/program/vk` hashes; verifier recomputes and raises `E301` on mismatch.
| Fake DA randomness | Relay issues commit-then-challenge receipts and signs them. FULL verification requires a trusted relay key; insecure local challenges are rejected.
| Manifest spoofing | `artifact_manifest_hash` recomputed; lack of attestation is surfaced via `policy_assurance = SELF_REPORTED` and downgrades status.
| Event log tampering | Hash chain recomputed and file length checked; failures emit `E201` warnings but do not affect the verification status because the emitter is untrusted.
| Capsulepack traversal / tarbomb | `_extract_capsulepack` validates members (no abs paths, dot-dot, symlinks) and enforces a size cap and hash check via `pack_meta` before verification runs.
| Legacy DA fallback | FULL verification now refuses legacy/insecure challenges even if the capsule declares `verification_profile=FULL`.

## 5. Security claims (informal)

1. **Capsule binding** – Producing two different payload/header pairs that verify under the same `capsule_hash` implies either a SHA-256 collision or a breach of the proof/commitment primitives (identical to the standard commitment “two openings” experiment).
2. **Statement binding** – Any adversary that makes the verifier accept while the header’s `statement_hash` differs from the proof’s statement must break the STC/FRI soundness guarantees or produce a collision in the canonical Statement hash.
3. **DA audit soundness** – Given an honest relay key and unpredictable challenges, the probability that an unavailable fraction ≥δ of the committed chunks evades detection is at most `(1-δ)^k` plus negligible collision probability in the vector-commitment/Merkle layers (the exact bound in HSSA’s DA section).
4. **Event log non-equivocation** – Under collision resistance, no two distinct logs can yield the same `(event_chain_head, events_log_hash, events_log_len)` triple, so tampering always surfaces as `E201_EVENT_LOG_MISMATCH` warnings even though the verifier no longer blocks FULL status on unauthenticated logs.
5. **Policy assurance honesty** – Without an inclusion proof under a trusted registry root, the verifier never claims `POLICY_ENFORCED`/`FULLY_VERIFIED`; all output explicitly labels such runs as `POLICY_SELF_REPORTED` and includes machine-readable warnings.

Future work (attested manifests, alternate backend adapters) fits into this skeleton by adding new hash commitments or elevating `policy_assurance` to `ATTESTED` once trusted attestations are plumbed through.

## 6. Canonical security experiments

Just as in Katz–Lindell and HSSA, each security notion is phrased as an experiment against an adversary with advantage measured relative to standard primitives. The reduction targets for Capsules + STC are:

1. **CapsuleBinding** – Challenger samples a payload/header pair, computes `capsule_hash`, and gives the adversary oracle access to the canonical hash function. The adversary wins if it outputs `(header, payload), (header', payload')` with `(header, payload) ≠ (header', payload')` that both pass `_verify_capsule_core` under the same `capsule_hash`. This reduces to either a collision in SHA-256 (payload hash or header commit hash) or a violation of the proof/statement binding conditions (i.e. STC/Fri soundness). Advantage is bounded by `Adv_bind ≤ Adv_crh + Adv_proof`.
2. **CapsuleAuthenticity** – Challenger exposes a signing oracle for a registered policy id/ACL. The adversary may query capsules to be signed; it wins if it outputs a capsule that `_verify_capsule_core` classifies as `POLICY_ENFORCED`/`FULLY_VERIFIED` without having requested a signature on that capsule hash (UF-CMA). Advantage is bounded by the secp256k1 UF-CMA advantage plus any ACL/policy registry assumptions.
3. **DASoundness** – Challenger commits to a capsule (fixing `payload_hash`, `header_commit_hash`, row roots) and then samples a relay challenge uniformly at random, signs it under the trusted relay key, and reveals it. The adversary controls the DA provider and wins if it convinces the verifier while at least a δ fraction of chunk roots are unavailable. The failure probability is bounded by `(1-δ)^k + Adv_crh + Adv_sketch`, matching the HSSA DA experiment with the additional assumption that the relay key is bound to `CAPSULE_TRUSTED_RELAYS_ROOT`.
4. **EventNonEquivocation** – Challenger exposes the canonical event hash/length predicate; the adversary wins if it produces two different event logs that both verify under the same `(events_log_hash, events_log_len, event_chain_head)` triple. Advantage reduces to finding a collision in SHA-256 over the chained hash or forging `events_log_len` (i.e. a length/tamper check bypass).

Stating the security goal as a combination of these experiments makes it clear that any break in Capsules + STC must either (a) collide SHA-256 / Merkle hashes, (b) forge a signature, (c) break the backend proof system, or (d) subvert the DA hash chain – all standard cryptographic assumptions.
