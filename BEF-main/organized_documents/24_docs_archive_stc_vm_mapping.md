# VM Trace → STC Encoding

`scripts/build_vm_trace.py` emits a toy VM trace following `bef_trace_v1`. Each
step packs `(pc, opcode, gas, accumulator)` into one field element via

```
value = (pc << 45) | (opcode << 37) | (gas << 25) | acc
```

The script writes `code/traces/vm_demo.json` with `chunk_length=4`. Running

```
python scripts/stc_aok.py prove-trace code/traces/vm_demo.json code/sketches/vm_demo_sketch.json --num-challenges 4 --chunk-len 4
```

produces a CPU-only sketch that matches the GPU layout. `python scripts/stc_aok.py verify code/sketches/vm_demo_sketch.json` runs the pure-Python `verify_fast` to
demonstrate end-to-end trace → STC → verification without CUDA.

## Geometry AIR trace mapping

The real geometry VM rows include more state. Each row is converted into a
length `10 + 2m` vector before feeding the STC accumulator:

```
[pc, opcode, gas, acc, x1, x2, cnt, m11, m12, m22, s[0..m-1], pow[0..m-1]]
```

`air/geom_trace_export.py` exposes helpers to flatten rows and emit a
`bef_trace_v1` JSON with `chunk_length = 10 + 2m`. Running

```
python scripts/export_geom_trace.py --steps 64 --num-challenges 2 --output nova_stc/examples/geom_trace.json
```

produces a real trace chunk file aligned with the STC row backend (each chunk =
one row). This is the format consumed by the `nova_stc` crate so the Nova proof
alibis the exact rows used in the GeomAIR proof.
