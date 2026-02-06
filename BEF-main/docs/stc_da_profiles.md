# STC Data Availability Profiles

A DA profile is a small policy word bound to every STC sketch via the
`da_profile` field. It specifies how the raw trace is stored and what sampling
parameters verifiers must satisfy.

```
{
  "version": 1,
  "mode": "LOCAL_FILE | L1_BLOB | EXTERNAL_DA | COMMITTEE",
  "sampling": {
    "delta": 0.1,      # targeted withholding fraction
    "epsilon": 1e-6,   # failure probability budget
    "k_min": 96        # minimum random chunk samples (derived)
  },
  "provider": {
    "path": "code/traces/vm_demo.json"
  }
}
```

## Example profiles

### Minimal-Local

- mode: `LOCAL_FILE`
- delta = 0.2, epsilon = 1e-3 → `k_min ≈ 33`
- provider: `{ "path": "code/traces/vm_demo.json" }`
- Use for demos/tests; run `scripts/stc_da_sample.py` to spot-check chunks.

### Rollup-L1Blob (hypothetical)

- mode: `L1_BLOB`
- delta = 0.1, epsilon = 1e-9 → `k_min ≈ 207`
- provider: `{ "chain_id": 1, "blob_tx": "0xdead..." }`
- Intended for rollup posting data to Ethereum blobs; samplers re-fetch blob and
  verify sampled buckets.

### External-DA (Celestia/Eigen)

- mode: `EXTERNAL_DA`
- delta = 0.05, epsilon = 1e-12 → `k_min ≈ 552`
- provider: `{ "namespace": "0xabc...", "height": 123456 }`
- Samplers use the DA backend API to fetch random chunks.

Profiles are stored in `da_profiles/*.json` for reference. Clients may always
run more samples than `k_min`; the profile just encodes the minimum guarantee.
