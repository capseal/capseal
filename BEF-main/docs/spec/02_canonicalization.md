# Canonicalization (normative)

All hashes and signatures in CapSole are computed over **canonical bytes**. Implementations MUST hash bytes, not parsed objects.

## Normative encoding: `dag_cbor_compact_fields_v1`

The canonical encoding for all hash/signature inputs is **DAG‑CBOR** with the project’s “compact fields” ruleset:

- Deterministic / canonical CBOR encoding (no floats, no indefinite lengths).
- Stable map key ordering.
- Field normalization (“compact fields”) applied consistently (exact rules belong to the encoder implementation).

If two implementations produce different bytes for the same semantic object, they are **non‑conformant**.

## Non‑normative views

JSON may be used for human inspection, debugging, or UI, but:
- JSON encodings are not authoritative
- JSON bytes MUST NOT be hashed for verification
