# CapSole / CapsuleTech — Documentation Map

CapSole is a **reproducible execution receipt**: a portable “capsule” that binds (1) *what was computed*, (2) *the exact trace format/spec*, (3) *the proof/commitments*, and optionally (4) *policy* and (5) *data availability sampling* into a verifier that runs **fail-closed**.

This repo has four layers of documentation. Each layer has a job. **Only one layer is normative.**

---

## Read-this-first routing

### If you’re an investor / PM / general reader (5–10 minutes)
1) **Primer (human overview):** `docs/primer.md`  
2) **Product wedge + flow:** `docs/guides/integration_guide.md`  
3) **What users get (guarantees + limits):** `docs/spec/05_profiles.md`

### If you’re a systems engineer integrating this (20–40 minutes)
1) `docs/primer.md`  
2) **Integration guide:** `docs/guides/integration_guide.md`  
3) **Adapter contract:** `docs/spec/07_adapter_contract.md`  
4) **Canonicalization + domain tags:** `docs/spec/02_canonicalization.md`, `docs/spec/03_domain_tags.md`  
5) **Registries + trust roots:** `docs/spec/08_registries.md`

### If you’re a cryptographer / “hostile reviewer” (30–60 minutes)
1) **Security model (claims + threat model):** `docs/security/security_model.md`  
2) **Normative spec:** `docs/spec/00_overview.md` → follow links  
3) **Theorems + bounds:** `docs/security/theorems.md`  
4) **Conformance + mutation tests:** `docs/spec/07_adapter_contract.md`

### If you’re working on the proving backend (Geom or future backends)
1) **Backend overview:** `docs/backends/geom.md`  
2) **Instance binding:** `docs/spec/04_instance_binding.md`  
3) **Protocol ordering (commit → challenge → open):** `docs/spec/06_protocol.md`

---

## Normative vs explanatory

- **Normative (MUST/SHOULD):** everything under `docs/spec/`  
  If something conflicts with spec, **spec wins**.
- **Explanatory:** `docs/primer.md`, `docs/guides/*`, `docs/backends/*`  
  These may simplify; they must not redefine.

---

## Verification profiles (promise surface)

- **PROOF_ONLY**  
  Capsule integrity + backend verification bound to `instance_hash` ⇒ statement holds for the committed root.

- **POLICY_ENFORCED**  
  PROOF_ONLY + policy rules evaluated over a **signed manifest anchor** (verified against pinned registries / ACL).

- **FULL**  
  POLICY_ENFORCED + relay challenge verified (pinned relay) + k sample opens to `row_root` under commit→challenge→open.

Authoritative definitions and predicates live in `docs/spec/05_profiles.md`.

---

## Conformance & interoperability

If you implement anything (adapter/backend/verifier), you must pass:

- **Canonicalization vectors:** `docs/spec/test_vectors/canon_vectors.json`
- **Adapter mutation tests:** `docs/spec/test_vectors/adapter_mutation_cases.md`
- **Reason code stability:** `docs/spec/09_reason_codes.md`

No conformance, no claims.
