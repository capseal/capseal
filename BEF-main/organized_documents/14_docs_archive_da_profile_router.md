# DA Profile + Router Layer

To keep STC/HSSA agnostic to where the raw trace lives, we bind a tiny **data
availability profile** to every STC commitment and provide a backend-agnostic
router. This adds a DA “menu” without changing the accumulator.

## Profile schema

```
DAProfile = {
  "version": 1,
  "mode": "L1_BLOB" | "EXTERNAL_DA" | "COMMITTEE_SIGS" | "LIGHT_SAMPLING" | "ARCHIVE_ONLY",
  "sampling": {
    "delta_min": 0.1,          # minimum withholding fraction we aim to detect
    "epsilon_max": 1e-6,       # failure probability budget
    "k_min": 96                # required number of chunk samples (computed via scripts/stc_param_table.py)
  },
  "provider": {
    "chain_id": 1,
    "blob_tx": "0x..."
  }
}
```

The block header commitment becomes `H(STC_root || encode(DAProfile))`, so the
da policy is cryptographically bound to the trace it claims to cover.

## Router interface

```
trait DABackend {
    fn publish(stc_root: Hash, profile: DAProfile, data_ref: DataRef) -> DAHandle;
    fn check_available(handle: DAHandle, chunks: &[ChunkId]) -> Result<(), DAError>;
}
```

Examples:

- `L1_BLOB`: data_ref = `{chain_id, blob_tx}`, check = download blob + verify bucket set.
- `EXTERNAL_DA`: data_ref = `{namespace, height}`, check = call DA API + run sampling.
- `COMMITTEE_SIGS`: data_ref = `URI list`, publish collects threshold sigs attesting to `stc_root`.

Profiles can be swapped over time using a `SwapDA` message:

```
SwapDA {
  stc_root,
  old_profile, old_handle,
  new_profile, new_handle,
  attestation: signature or SNARK proving equality of data.
}
```

## Sample profiles

See `da_profiles/minimal.json` and `da_profiles/strong.json` for concrete
configurations. Clients can always run stricter sampling than `k_min`, but the
profile sets a minimum bar (“if you accept this commitment, you must at least
run k_min samples targeting δ and ε”).
