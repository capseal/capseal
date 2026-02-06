# GPU Streaming Accumulator

This directory contains a cleaned-up version of the CUDA streaming accumulator
prototype that was previously trapped inside `batch_streaming_accum.ipynb`.

## Layout

```
gpu_accumulator/
├── __init__.py
├── stream_accumulator.py        # Python API (merkle helpers, accumulator class, demo)
└── cuda/
    ├── stream_sketch_extension.cpp
    └── stream_sketch_kernel.cu
```

## Usage

```python
from gpu_accumulator import StreamingAccumulatorCUDA

acc = StreamingAccumulatorCUDA(num_challenges=4)
acc.add_chunk(values_chunk)          # values_chunk is any list/sequence of ints
proof = acc.prove()
print(proof["challenges"], proof["global_sketch_vec"])
```

* `build_rpow_gpu(r, length)` exposes the tiled kernel that builds `[1, r, …, r^{L-1}]` on
  the GPU and can be reused across audits.
* `chunk_dot_cuda(vals, rpow)` computes a chunk sketch if you want to provide your
  own accumulator logic.
* `demo_cuda()` reproduces the benchmark from the notebook (guarded behind
  `if __name__ == "__main__"`).
* `sketch_trace.py` consumes a `bef_trace_v1` JSON (emitted by the BEF simulators)
  and produces a `bef_sketch_v1` JSON containing the challenge vector, the global
  sketch vector, per-chunk Merkle roots, and GPU timing data.

## Integration ideas

* **BEF temporal core** – export flattened per-step values from the Rust
  `CryptoPath` and feed them into `StreamingAccumulatorCUDA` to obtain a
temporal sketch for the entire execution trace.
* **ENN trace encoder** – treat the global sketch (and per-chunk sketches) as
  additional features when training the entangled RNN/attention collapse model.
* **FusionAlpha contradiction graph** – use chunk sketches as node features so
  the graph propagation can focus on windows that disagree with the global
  fingerprint.

The module avoids runtime pip installs, uses a stable extension name, and can be
imported by any Python codebase in the repo.  The `code/traces → code/sketches`
toolchain lets you ship BEF traces + sketches on a flash drive with zero setup.
