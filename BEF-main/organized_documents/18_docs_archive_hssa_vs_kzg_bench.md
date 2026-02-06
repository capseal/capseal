HSSA vs KZG — Benchmark Plan and Artifacts

1) Goals
- Measure HSSA commit throughput (elements/s, GB/s) for realistic trace sizes.
- Measure HSSA fast verification (CPU) vs trace size, #chunks, m.
- Compare with a basic KZG commitment baseline (same N) to provide directional evidence.

2) Parameter Grid
- Trace sizes N ∈ {2^20, 2^24, 2^26} (≈ 1M, 16M, 64M elements).
- Challenges m ∈ {2, 4, 8}.
- Chunk length L ∈ {2^10, 2^13} (1024, 8192 elements).
- Field p = 2^61 − 1 (prototype), note CPU/GPU models used.

3) HSSA Benchmark Harness (bench_hssa.py)
- For each (N, m, L):
  - Generate random v ∈ F_p^N.
  - Partition into chunks.
 - Run StreamingAccumulatorCUDA.prove() to commit (r^i build or fused kernel; per‑chunk sketches).
 - Record commit wall‑clock time t_commit_gpu and compute throughput: elems/s and GB/s.
 - Run bef_verify_fast on the emitted sketch JSON to get t_verify_fast (CPU).
 - Write CSV: N, m, L, t_commit_gpu_ms, t_verify_fast_ms, elems_per_s, GB_per_s, hardware.
- Script: `python bench/bench_hssa.py --Ns 1048576,16777216 --challenges 2,4,8 --chunk-lens 1024,8192 --repeats 3`
  - Expects `bef_verify_fast_cli` at `BICEPsrc/BICEPrust/bicep/target/release`. Build via `cargo build --release -p bicep-crypto --bin bef_verify_fast_cli`.

4) KZG Baseline Harness (bench/kzg-bench)
- Located at `bench/kzg-bench` (standalone Cargo crate using arkworks KZG10).
- For each N: sample random v over BLS12-381 Fr and measure time to compute Commit(v).
- Compute throughput: elems/s and MB/s (Fr is 32 bytes).
- Run on the same hardware; note HSSA commit uses GPU while KZG runs on CPU.
- Command:
  ```bash
  cd bench/kzg-bench
  cargo run --release > ../kzg_results.csv
  ```

5) Report Template
- Produce markdown table such as:

| N (elems) | m | L (chunk) | HSSA commit (ms) | HSSA thr. (GB/s) | HSSA verify_fast (ms) | KZG commit (ms) | KZG thr. (MB/s) | Hardware |
|----------:|---|-----------:|-----------------:|------------------:|----------------------:|----------------:|------------------:|----------|
| 2^20      | 4 | 2^10       | …                | …                 | …                     | …               | …                 | A100 + CPU XYZ |
| 2^24      | 4 | 2^13       | …                | …                 | …                     | …               | …                 | … |
| 2^26      | 8 | 2^13       | …                | …                 | …                     | …               | …                 | … |

6) Narrative (short)
- On an A100, HSSA achieves X–Y GB/s commit throughput for N ∈ [1M, 64M] and m ∈ [2,8]; CPU bef_verify_fast stays ≤ Z ms.
- A naive KZG baseline on the same machine yields U–V MB/s commitment throughput. Preliminary results illustrate that HSSA is GPU‑native and offers high‑throughput commitment suitable for DA/IVC scenarios.
