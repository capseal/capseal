# Spec 00 — Overview and Normative Scope

This folder is the single source of truth. Every definition here is normative and versioned. Primer and guides MAY summarize but MUST NOT redefine.

Conformance language follows RFC 2119 (MUST/SHOULD/MAY).

Scope:
- Objects/fields that constitute a capsule and its header/payload.
- Canonical encodings and domain tags for hashes/signatures.
- Instance binding (the tuple and how backends MUST absorb it).
- Verifier profiles and acceptance predicates.
- Adapter contract (prove/verify API + mutation tests).
- Registries/trust roots and override rules.
- DA protocol messages and ordering constraints.

Out of scope:
- Performance and parameter tuning; storage SLAs; UI/relay UX.

