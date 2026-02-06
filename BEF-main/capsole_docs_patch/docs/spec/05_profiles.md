# Verification profiles (normative)

Profiles are *verifier predicates*: the verifier chooses the requested level and the verification procedure must fail-closed if prerequisites are missing.

## Profiles

### PROOF_ONLY

Verifies:
- Capsule integrity hashes (payload/hash/header)
- `statement_hash`, `trace_spec_hash` recomputation
- `instance_hash` recomputation and backend proof verification bound to it

Guarantee: *the statement holds for the committed `row_root` under the bound proof system parameters.*

### POLICY_ENFORCED

Includes PROOF_ONLY, plus:
- Manifest anchor recomputation
- Manifest signature verification against a pinned signer registry root
- Policy evaluation over the signed anchor + statement
- Authorship signature and ACL checks if required by the policy

Guarantee: PROOF_ONLY + *policy rules were evaluated over a signed manifest anchor under pinned trust roots.*

### FULL

Includes POLICY_ENFORCED, plus:
- Relay challenge verification against a pinned relay registry root
- Deterministic sampling and k sample-open checks to `row_root`
- Enforced commit→challenge→open ordering (challenge excluded from header_commit_hash)

Guarantee: POLICY_ENFORCED + *availability sampling succeeded under a trusted relay challenge.*

## Fail-closed rule (MUST)

A verifier MUST NOT silently downgrade the requested profile due to missing prerequisites. It must return a reject status with machine-readable reason codes.
