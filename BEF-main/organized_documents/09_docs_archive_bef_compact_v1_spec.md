# BEF Compact Serialization (`bef_compact_v1`)

This document sketches the Phase‑0 specification for a production-grade binary
serialization format that supersedes the current DAG‑CBOR encodings. It focuses
on the core file layout, data types, and security invariants; subsequent phases
will provide the full byte-level schema for every proof/capsule structure.

## 1. Design Goals

1. **Unambiguous by construction** – every valid byte stream corresponds to a
   single semantic object. No optional keys, no duplicate entries, no slack.
2. **Performance-oriented** – optimized for zero-copy reads and minimal heap
   churn in verifiers. No key names, no self-describing overhead.
3. **Security hardened** – parsing must be strictly bounded, rejecting any
   malformed/crafted payload before it can trigger DoS or verification bypass.
4. **Layered abstraction** – canonical encoding is uncompressed; higher layers
   may apply zstd or similar without changing the canonical form.
5. **Backwards compatible** – legacy JSON / CBOR artifacts continue to decode via
   dispatcher logic keyed on magic bytes / tags.

## 2. Threat Model

* **DoS via parsing** – attacker supplies oversized lengths or malformed varints
  to exhaust CPU/RAM. Countermeasure: strict length-prefix validation and
  canonical varint enforcement.
* **Verification bypass** – malformed payload attempts to skip fields or violate
  field ordering. Countermeasure: schema-driven parsing, zero tolerance for
  missing/extra data.
* **Canonicality attacks** – attacker attempts to produce multiple encodings of
  the same proof. Countermeasure: canonical varints, canonical field ordering,
  no optional whitespace/keys.

## 3. File Layout

```
+------------+-------------------+---------------------------+
| Bytes 0-3  | Magic             | 0xBEF0C0DE                |
| Bytes 4-5  | Version           | 0x0100 (major=1, minor=0) |
| Bytes 6-9  | Metadata length N | uint32 LE                 |
| Bytes 10…  | Metadata JSON     | N bytes UTF-8             |
| …          | Proof payload     | schema-defined            |
+------------+-------------------+---------------------------+
```

* **Magic bytes** – allow instant rejection of unrelated files.
* **Version** – enables evolution via `major.minor`. Minor upgrades remain
  backward compatible; major upgrades signal breaking changes.
* **Metadata** – small, human-readable JSON (trace id, air id, timestamp, etc.)
  that is *not* part of the canonical proof hash. This allows operators to store
  auxiliary context without affecting verification.
* **Payload** – canonical proof bytes described in Section 4.

## 4. Payload Encoding Rules

### 4.1 Primitive Types

* **Varint** – unsigned LEB128-like encoding with canonicality checks (no leading
  zero continuation bytes, no overflow above 2^64-1). Used for lengths, counters,
  Goldilocks field elements, etc.
* **Bytes** – raw byte arrays. Used for hashes, Pallas/Vesta field elements,
  Merkle roots, etc.
* **Fixed arrays** – concatenation of child fields in schema order.

### 4.2 Field Elements (`Fp`)

| Field            | Encoding                             |
|------------------|--------------------------------------|
| Goldilocks (64b) | canonical varint (little endian)     |
| Pallas/Vesta     | 32-byte little-endian fixed array    |
| Generic big Fp   | ceil(bits/8) bytes little-endian     |

### 4.3 Structs

Each struct is encoded as a fixed-order sequence of its fields, with no key
names. Example for `FriLayer { commitment: Hash, beta: Fp, length: u32 }`:

```
[32 bytes commitment] [beta (varint or 32-byte)] [length varint]
```

### 4.4 Lists

All lists are `varint(count)` followed by `count` encoded elements in order.

## 5. Security Invariants

* **Length checking** – `BinaryDecoder.read_bytes(n)` fails if `offset+n > len`.
* **Canonical varints** – decoder rejects encodings with redundant leading
  zeros/continuations.
* **No recursion** – decoder iterates with explicit stacks to avoid stack
  exhaustion.
* **Full consumption** – after parsing the payload, `offset` must equal
  `len(buffer)` or decoder fails (prevents trailing garbage).

## 6. Backward Compatibility

* Legacy artifacts are detected as:
  * JSON – leading `{` or `[`.
  * CBOR – self-describe tag 0xD9D9F7 or other CBOR lead bytes.
* New dispatcher reads magic bytes; if `0xBEF0C0DE`, it routes to the
  `BinaryDecoder`; otherwise falls back to legacy decoders.

## 7. Next Steps

1. Complete the byte-level schema for proofs/capsules (field-by-field order).
2. Implement `BinaryEncoder` / `BinaryDecoder` with canonical varint, field-
   specific writers/readers, and hardened bounds checks.
3. Wire encoder into pipeline tooling, gating behind `--encoding-id
   bef_compact_v1`.

This document intentionally stops short of a full implementation; it establishes
the format and threatmodel so Phase 1/2 can proceed with confidence.
