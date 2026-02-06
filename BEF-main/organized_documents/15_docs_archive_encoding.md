# Canonical Encoding

This repository ships deterministic binary artifacts for capsules and proofs using
DAG-CBOR based encodings. Two variants exist today:

* `dag_cbor_canonical_v1` – original format (ints encoded as CBOR integers).
* `dag_cbor_compact_fields_v1` – new default; field elements are serialized as
  compact little-endian byte strings whose width matches the actual field size
  (8 bytes for Goldilocks, 32 bytes for larger curves).

Both encodings share the same goals:

1. Stable hashes that do not depend on incidental formatting
2. Compact binary representations (no hex strings) that enable content-addressed
   storage
3. Data that can be round-tripped into JSON for debugging without loss

## Canonicalization rules

* **Encoding** – The encoding is a strict subset of DAG-CBOR (deterministic
  CBOR).  Only the following CBOR types are used: unsigned/signed integers,
  byte strings, UTF‑8 text strings, arrays, maps, booleans, null, and 64‑bit
  IEEE floats.  Indefinite-length items are forbidden.
* **Map ordering** – Map keys are encoded independently, and entries are sorted
  by the lexicographic order of their encoded byte representation before being
  emitted.  This matches the DAG-CBOR canonical ordering rule.
* **Byte formats** – Cryptographic data that is naturally a byte string (hashes,
  commitments, Merkle siblings, etc.) MUST be encoded as raw byte strings
  (`major type 2`).  JSON views may still present these values as lowercase hex,
  but binary artifacts never store textual encodings.
* **Integers** – Integers are encoded using the minimal CBOR representation
  (major types 0/1).  In the compact-fields encoding, integers that are tagged as
  field elements are instead emitted as byte strings whose width equals the
  field’s native bit width (little-endian).
  The serializer tags every field element (including nested FRI openings), so
  this rule applies recursively throughout the proof/capsule structure.
* **Floats** – Floating point values are encoded as IEEE754 binary64 (`0xfb`).
* **Null/default values** – Optional fields MAY be omitted; when present they
  are encoded as CBOR `null` (0xf6).  The Capsule hash is computed over whatever
  explicit representation is emitted, so either omission or explicit null is
  acceptable as long as it is deterministic for the generator.
* **Domain separation** – Every hash derived from canonical bytes uses an
  explicit ASCII prefix, e.g. `BEF_CAPSULE_V1 || canonical_bytes`.

## Sanitization helper

A lightweight sanitization pass (`sanitize_for_encoding`) canonicalizes host
objects before encoding:

* Paths become strings
* Tuples become arrays
* Dictionaries drop keys whose value is `None` only when the calling site
  requests pruning (currently capsules preserve explicit `None`s)

## API

`bef_zk.codec` exposes:

* `ENCODING_ID` – currently `dag_cbor_compact_fields_v1`
* `canonical_encode(obj, encoding_id=ENCODING_ID) -> bytes`
* `canonical_decode(data: bytes) -> object`
* `FieldElement(value, bits)` helper used when tagging field values for compact
  emission.

Higher-level helpers (`encode_capsule`, `encode_proof`, etc.) sanitize
structures before calling these primitives.

## Capsule hash and audit seed

* `capsule_hash = SHA256("BEF_CAPSULE_V1" || canonical_bytes_without_hash)`
* Default audit seed uses the capsule hash plus unbiasable anchors:
  `seed = first_u64( SHA256("BEF_AUDIT_SEED_V1" || capsule_hash || anchor_ref || policy_id) )`

Any change to the capsule contents now deterministically updates the hash,
allowing content-addressed storage and reproducible audit sampling.
