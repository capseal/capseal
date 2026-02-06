# Operator Flow (Geom demo)

The geometry demo now has one blessed workflow that produces a full Capsule:

```
python scripts/run_pipeline.py --backend geom \
    --steps 64 \
    --num-challenges 2 \
    --output-dir out/demo_geom_
```

It performs:

1. **Trace + STC log**: run `simulate_trace` on `GEOM_PROGRAM`, flatten each row to
   `[pc, opcode, gas, acc, x1, x2, cnt, m11, m12, m22, s[0..m-1], pow[0..m-1]]`, and
generate a `bef_trace_v1` (`stc_trace.json`).

2. **Geom AIR proof**: run `zk_prove_geom`, serialize to `geom_proof.json`, verify
   immediately for sanity.

3. **Nova-over-STC**: call `cargo run -p nova_stc -- prove --chunks stc_trace.json ...`
   which produces both timings and the final STC state. `--stats-out` writes
   `nova_stats.json` (final `(n, root, s, pow)` plus timings and proof sizes).

4. **Capsule construction**: aggregate paths + metadata into
   `strategy_capsule.json` (schema `bef_capsule_v1`). The capsule now embeds the
   STC commitment `(n, root, s, pow)`, the Geom proof summary, and file pointers
   for every artifact.

5. **Pipeline stats**: `pipeline_stats.json` captures the plain-trace runtime,
   Geom proof profile, and Nova timings in one place. Sweeps can consume this via
   `bench/geom_pipeline_sweep.py` to produce reproducible scaling studies.

Operators follow this flow per epoch:

```
# produce trace + Geom proof + Capsule for 2048 steps
python scripts/run_pipeline.py --backend geom --steps 2048 --num-challenges 4 --output-dir out/epoch_42
```

Compression is still optional: Nova state is maintained continuously; the
compressed SNARK (~10 kB, ~20–40 s) is produced on demand (per epoch) via the
`--compressed` flag.

### Capsule verification

To re-check everything later (or share the artifact), run:

```
python scripts/verify_capsule.py out/epoch_42/strategy_capsule.json
```

The command replays the Geom verifier and ensures the capsule's STC commitment
matches the Nova stats. This is the same entry point we use when handing a
Capsule to another team.
