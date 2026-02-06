# BICEP - Brownian-Inspired Computation Engine for Paths

A high-performance Rust library for simulating stochastic differential equations (SDEs) with mathematically rigorous support for both Itô and Stratonovich calculus.

## Features

- **Multiple integrators**: Euler-Maruyama, Heun midpoint (Stratonovich), Milstein (planned)
- **Itô ↔ Stratonovich conversion**: Automatic drift correction with Jacobian support
- **Built-in models**: Geometric Brownian Motion (GBM), Ornstein-Uhlenbeck (OU)
- **Reproducible randomness**: Counter-based PRNG with per-path seeding
- **Variance reduction**: Antithetic variates, common random numbers
- **Path management**: Efficient storage with stride saving
- **Future-ready**: Hooks for change-of-measure (Girsanov) and GPU acceleration

## Quick Start

```rust
use bicep_core::{State, Calc, integrators::EulerMaruyama, SdeIntegrator};
use bicep_core::noise::NoiseGenerator;
use bicep_models::GeometricBrownianMotion;

// Define a GBM model
let gbm = GeometricBrownianMotion::new(0.05, 0.2); // μ=5%, σ=20%

// Set up simulation
let mut state = State::new(vec![100.0]); // Initial price
let mut rng = NoiseGenerator::new(42);
let integrator = EulerMaruyama;

// Simulate one step
let dt = 0.01;
let dW = rng.generate_dw(1, dt.sqrt());
state = integrator.step(Calc::Ito, 0.0, &state, dt, &dW, &gbm, &gbm);
```

## Mathematical Foundation

BICEP implements the numerical solution of SDEs in both Itô and Stratonovich forms:

- **Itô**: dX_t = μ(t,X_t)dt + σ(t,X_t)dW_t
- **Stratonovich**: dX_t = μ°(t,X_t)dt + σ(t,X_t)∘dW_t

With the conversion: μ° = μ - ½(∂σ/∂x)·σ

## Running Tests

```bash
cargo test
```

Key tests:
- `gbm_moments`: Verifies GBM moments against analytical solutions
- `ou_moments`: Tests OU process convergence to stationary distribution
- `ito_strat_equivalence`: Confirms Itô-Stratonovich conversion correctness

## Examples

Run the GBM example:
```bash
cargo run --example gbm_simple
```

## Architecture

- `bicep-core`: Core types, traits, and integrators
- `bicep-models`: Standard SDE models (GBM, OU, etc.)
- `bicep-cpu`: CPU-optimized parallel samplers (planned)
- `bicep-gpu`: GPU backends via wgpu/CUDA (planned)
- `bicep-io`: Parquet/Arrow I/O (planned)

## Future Work

- [ ] Milstein integrator with derivative terms
- [ ] GPU kernels for massive parallelism
- [ ] Path-dependent options and stopping times
- [ ] Integration with ENN and Fusion Alpha components