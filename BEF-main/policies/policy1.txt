# BICEP-ENN-FusionAlpha Pipeline

A pipeline for stochastic trajectory generation, sequence learning, and graph-based planning.

## Overview

Three-component system:

- **BICEP**: Stochastic differential equation (SDE) trajectory generator (Rust)
- **ENN-C++**: Entangled neural network implementation with backpropagation through time (C++)
- **FusionAlpha**: Graph-based planning using committor functions (Rust/Python)

## Critical Gaps and Limitations

- No standardized benchmarks or third-party validation
- Missing unit tests for BICEP SDE integrators
- No convergence tests for ENN training
- FusionAlpha lacks proper graph algorithm verification
- Performance metrics are from limited toy examples (parity task only)
- No comparison with baseline methods (standard RNNs, LSTMs)
- Memory usage and scaling properties undocumented
- No error bars or confidence intervals on reported accuracies

## Architecture

```
BICEP (Rust)     →     ENN-C++ (C++)     →     FusionAlpha (Rust/Python)
├─ SDE Integration     ├─ Entangled Cells      ├─ Graph Construction
├─ Trajectory Gen      ├─ BPTT Training        ├─ Severity Scaling
└─ Parquet Output      └─ Committor Learning   └─ Decision Making
```

## Quick Start

### Prerequisites
- Rust (1.70+)
- C++17 compiler (GCC/Clang)  
- Python 3.9+
- Eigen3 (auto-downloaded)

### Run Complete Pipeline
```bash
# Make executable and run
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

This will:
1. Build all components
2. Generate BICEP parity trajectories  
3. Train ENN-C++ on the data
4. Run FusionAlpha planning
5. Display results

### Manual Execution

#### 1. Build BICEP
```bash
cd BICEPsrc/BICEPrust/bicep
cargo build --release
```

#### 2. Generate Trajectories
```bash
./target/release/parity_trajectories --sequences 200 --seq-len 15
```

#### 3. Build ENN-C++
```bash
cd ../../../enn-cpp
make all
```

#### 4. Train ENN
```bash
# Convert parquet to CSV first
python3 -c "import polars as pl; df = pl.read_parquet('../BICEPsrc/BICEPrust/bicep/runs/parity_trajectories.parquet'); df.with_columns([pl.col('state').list.get(0).alias('state_0')]).drop('state').write_csv('parity_data.csv')"

# Run ENN training
./apps/bicep_to_enn parity_data.csv
```

#### 5. Run FusionAlpha
```bash
cd ../
python3 pipeline_demo.py
```

## Repository Structure

```
bicep-enn-fusion-clean/
├── BICEPsrc/                    # Rust-based SDE trajectory generator
│   └── BICEPrust/bicep/
│       ├── crates/              # BICEP core modules
│       └── target/release/      # Compiled binaries
├── enn-cpp/                     # High-performance ENN implementation
│   ├── include/enn/             # Header files
│   ├── src/                     # Implementation
│   ├── apps/                    # Applications
│   ├── tests/                   # Validation tests
│   └── Makefile                 # Build system
├── FusionAlpha/                 # Committor-based planning
│   ├── crates/                  # Rust core
│   ├── python/                  # Python bindings
│   └── examples/                # Demo scripts
├── run_full_pipeline.sh         # Complete pipeline runner
├── pipeline_demo.py             # Python integration demo
└── FUSION_ALPHA_USAGE_GUIDE.md  # Detailed usage instructions
```

## Implementation Details

### BICEP
- SDE integrators: Euler-Maruyama, Heun, Milstein
- Parity task trajectory generator
- Parquet output format

### ENN-C++
- Entangled cell update: ψₜ₊₁ = tanh(Wₓxₜ + Wₕhₜ + (E-λI)ψₜ + b)
- Backpropagation through time implementation
- OpenMP parallel batch processing
- Gradient checking (tolerance: 1e-10)

### FusionAlpha
- Committor function computation
- Graph-based planning
- Severity-weighted propagation

## Technical Details

| Component | BICEP | ENN-C++ | FusionAlpha |
|-----------|-------|---------|-------------|
| Language | Rust | C++ | Rust/Python |
| Parallelization | Single-threaded | OpenMP | Single-threaded |
| Data Format | Parquet | CSV | JSON |
| Dependencies | nalgebra, arrow | Eigen3 | nalgebra, serde |

## Potential Use Cases

- Time series prediction with temporal dependencies
- Stochastic process simulation
- Sequential decision making
- Graph-based planning problems

## Known Issues

- Only tested on parity task (toy problem)
- Limited to small sequence lengths (<100 steps)
- No GPU support
- Single-threaded bottlenecks in BICEP and FusionAlpha
- Untested on real-world datasets

## Requirements for Production Use

1. Comprehensive test suite needed
2. Benchmarking against standard methods required
3. GPU acceleration not implemented
4. Distributed processing not supported
5. Memory optimization needed for large-scale data

## License

Not specified. Research prototype only.