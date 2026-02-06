# FRI / STC parameter guidance

This backend exposes two sets of knobs:

1. **FRI / AIR parameters** – determine the soundness error of the zk argument.
2. **STC parameters** – determine binding / DA guarantees of the streaming accumulator.

Both must be configured together for a concrete deployment. The table below gives sample settings for the toy geometry AIR; adapt them to your own relation.

| Target steps (T) | Domain size (N) | AIR degree bound (d_max) | FRI rounds | Queries (q) | Estimated FRI error ((d_max / N)^q) |
|------------------|-----------------|--------------------------|-----------|-------------|-------------------------------------|
| 2^10 (≈1k)       | 2^10            | 2^10                     | 4         | 16          | ≈ 2^{-40}                           |
| 2^14 (≈16k)      | 2^15            | 2^14                     | 5         | 24          | ≈ 2^{-60}                           |
| 2^18 (≈260k)     | 2^19            | 2^18                     | 6         | 32          | ≈ 2^{-80}                           |
| 2^22 (≈4M)       | 2^23            | 2^22                     | 7         | 48          | ≈ 2^{-110}                          |

**How to read the table**

- Choose ((N)) as the next power of two above your padded trace length.
- Compute a conservative degree bound ((d_max)) for the composition polynomial (max AIR degree plus masking degree). In this prototype, `build_composition_vector` never exceeds the raw trace degree, so you can over-approximate with the trace length.
- Pick a number of FRI rounds and queries ((q)) so that ((d_max / N)^q) is below your target failure probability.
- Masking polynomials must have degree strictly less than the FRI degree bound—keep mask degree < (d_max).

**STC parameters**

Reuse `docs/stc_parameter_guidance.md` to pick the number of sketch challenges (m) and DA sampling rate. For convenience the existing helper script can be run as:

```bash
python scripts/stc_param_table.py --n-max 16777216 --p 2305843009213693951 --challenges 4 6 8
```

**FRI helper script**

`scripts/fri_param_table.py` prints the FRI soundness estimate for a grid of domain sizes and query counts. Example:

```bash
python scripts/fri_param_table.py --domain 1048576 --degree 262144 --queries 20 30 40
```

This reports ((degree / domain)^q). Use it to sanity-check new AIRs before collecting full benchmarks.

Remember that the final argument’s soundness is the sum of:

- FRI error (as above),
- STC binding failure (from the sketch bound in §3), and
- collision resistance of SHA-256 (≈2^{-128}).

Pick parameters so that each term is comfortably below your overall security target (e.g., 128 bits).
