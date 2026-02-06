# Security Theorems (non‑normative statements)

Capsule Binding (informal)
- Under SHA‑256 collision resistance and backend instance‑binding soundness, any adversary that produces two distinct (header, payload) that verify under the same `capsule_hash` breaks CRH or backend soundness.

Instance Soundness (informal)
- If `_verify_capsule_core` accepts at PROOF_ONLY for a capsule whose backend absorbed the instance tuple (V1/V2), then the statement proven by the backend is true for the bound row commitment.

DA Sampling Bound (FULL)
- Let δ be the unavailable fraction; k be the sample count; p the field modulus; n_max the problem scale; m the number of algebraic checks; Adv terms as in the security model. Under commit→unpredictable‑challenge→open ordering,

```
Pr[cheat] ≤ (1 − δ)^k + Adv_H_crh + ((p − 1)/n_max − 1)^m + Adv_backend + Adv_sig
```

Authorship/Registry Authenticity
- Accept at ≥ POLICY_ENFORCED implies a valid secp256k1 signature over `capsule_hash` by a pubkey authorized in the ACL (if provided) and a manifest anchor signature by a signer in the pinned registry root; otherwise reduces to UF‑CMA or registry‑auth failure.

