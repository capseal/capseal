# Geom backend (non-normative)

Geom is the reference STARK backend used to demonstrate the end-to-end binding contract:
- it produces a STARK proof over a committed row-major table
- it MUST absorb `instance_hash` (adapter input `binding_hash`) into Fiat–Shamir
- it pins AIR/FRI/program/VK identities via header hashes

Geom is not positioned as a competitive prover; it exists to make the capsule binding surface concrete and testable.
