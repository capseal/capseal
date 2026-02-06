# SeedSpec v1.0 (Draft)

Scope: Deterministic, identity-bound random seeding for BICEP paths/sequences.

## Purpose

Bind per-path RNG to semantic identity (instrument/time/ensemble/path) so replays are addressable and invariant to ordering.

## Inputs (Identity)

- `instrument_id` string (domain-specific; may be "synthetic" for lab tasks)
- `date_bucket` string (YYYY-MM-DD) or `epoch_day` int32
- `ensemble_id` uint64
- `path_or_sequence_id` uint64

## Key and Derivation

- `global_seed_key_id`: string (identifier referencing a 32-byte secret key stored out-of-band)
- Hash function: `blake3_keyed_64`
- Canonical JSON serialization of identity (UTF-8, sorted keys, no whitespace, lowercase strings):

```
{"date_bucket":"2025-01-01","ensemble_id":1,"instrument_id":"AAPL","path_or_sequence_id":17}
```

- Derivation: `seed_u64 = little_endian_64( BLAKE3(key=global_seed_key, msg=canonical_json_bytes) )`

## SeedSpec JSON

```
{
  "schema": "bicep_seed_spec_v1",
  "global_seed_key_id": "seedkey-main-01",
  "hash_fn": "blake3_keyed_64",
  "identity_fields": [
    "instrument_id",
    "date_bucket",
    "ensemble_id",
    "path_or_sequence_id"
  ]
}
```

## Replay API Requirements

- Provide a single-path replay API: construct RNG from derived seed and step the integrator without iterating other paths.
- Seeds MUST NOT depend on iteration order or thread scheduling.

