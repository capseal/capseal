# BICEP (Brownian-Inspired Computation Engine for Paths) Source Documentation

## 1. System Overview

**BICEP** is a high-performance, concurrent stochastic differential equation (SDE) simulation engine written in Rust. It is designed to simulate ensembles of paths for multi-dimensional SDEs with precise control over stochastic calculus interpretations (Itô vs. Stratonovich), noise generation, and integration schemes.

The core architecture separates the **mathematical definition** of a process (Drift/Diffusion) from the **numerical method** (Integrator) and the **execution logic** (Sampler).

## 2. Core Primitives & Data Structures

### 2.1. State Representation
*   **`State`**: A wrapper around `nalgebra::DVector<f64>`. It represents the state vector $X_t 	in ℝ>^n$ at a specific time $t$. It supports standard arithmetic operations required for numerical integration.
*   **`Time`**: Type alias for `f64`.

### 2.2. Model Definitions (Traits)
Users define SDEs by implementing two core traits:
*   **`Drift`**: Defines the deterministic component $\mu(t, X_t)$.
    *   Signature: `fn mu(&self, t: Time, x: &State) -> State`
*   **`Diffusion`**: Defines the stochastic volatility component $\sigma(t, X_t)$.
    *   Signature: `fn sigma(&self, t: Time, x: &State) -> DMatrix<f64>`
    *   Returns an $n × m$ matrix where $n$ is state dimension and $m$ is noise dimension.
    *   **Jacobian Support**: Optional `sigma_jacobian` method returns $
abla_x \sigma$ to support higher-order integrators (Milstein) or automatic Stratonovich drift corrections.

## 3. Numerical Integration Engine

BICEP supports interchangeable integrators via the `SdeIntegrator` trait. The engine explicitly handles the conversion between Itô and Stratonovich calculus interpretations.

### 3.1. Calculus Modes (`Calc`)
*   **`Calc::Ito`**: Standard Itô integration.
*   **`Calc::Stratonovich`**: Stratonovich integration, which follows the ordinary chain rule.

### 3.2. Implemented Integrators
1.  **`EulerMaruyama`**: Strong order 0.5.
    *   **Itô Mode**: $X_{t+\Delta t} = X_t + \mu(X_t) \Delta t + \sigma(X_t) \Delta W$.
    *   **Stratonovich Mode**: Automatically computes the drift correction term $-0.5 \sum_j \sigma_j (
abla \sigma_j)$ using the model's Jacobian and subtracts it from the drift before applying the standard step.
2.  **`HeunStratonovich`**: Strong order 1.0 (for additive noise/Stratonovich).
    *   Uses a predictor-corrector (midpoint) scheme.
    *   Predictor: $	ilde{X} = X_t + \mu \Delta t + \sigma \Delta W$.
    *   Corrector: Averages drift and diffusion terms at $t$ and $t+\Delta t$ (using $	ilde{X}$).
    *   Converges to Stratonovich solution naturally without explicit Jacobian calculation.
3.  **`Milstein`**: Strong order 1.0.
    *   Adds the correction term involving Lévy areas: $0.5 \sigma \sigma' ((\Delta W)^2 - \Delta t)$.
    *   Requires `sigma_jacobian` implementation in the `Diffusion` trait.

## 4. Noise Generation & Randomness

### 4.1. `NoiseGenerator`
*   **PRNG**: Uses `rand_chacha::ChaCha20Rng` for cryptographic-strength, reproducible pseudo-randomness.
*   **Seeding**: 
    *   Supports deterministic seeding via `SeedSpec` and `SeedIdentity`.
    *   Path-specific seeding: `global_seed + path_id * mixing_constant`.
*   **Distribution**:
    *   Default: Standard Gaussian (`rand_distr::StandardNormal`).
    *   **`ShockType::StudentT`**: Supports heavy-tailed noise innovations ($t$-distribution with $
u$ degrees of freedom), scaled to unit variance.

### 4.2. EWMA Variance Scaling
The generator supports an optional Exponentially Weighted Moving Average (EWMA) mechanism (`NoiseConfig::ewma_alpha`).
*   Recursively updates an internal variance estimate: $\sigma^2_t = \alpha \sigma^2_{t-1} + (1-\alpha) z_t^2$.
*   Rescales generated noise based on this evolving volatility proxy.

## 5. Execution Flow & Sampling

The simulation is orchestrated by the `Sampler` struct in `bicep-sampler`.

### 5.1. The Sampling Loop
1.  **Configuration**: Accepts a `PathSpec` (steps, $dt$, save stride).
2.  **Parallelization**: Uses `rayon` (`par_iter`) to simulate paths concurrently across available CPU cores.
3.  **Step Cycle**:
    *   Generate noise increment $dW 	in ℤ(0, dt)$ (or configured distribution).
    *   Call `integrator.step(...)` to advance state.
    *   **Boundary Conditions**: Apply constraints (Reflecting, Absorbing, Periodic) via `apply_boundary`.
    *   **Stopping Conditions**: Check first-passage criteria or time limits via `Stopping` struct.
    *   **Striding**: Only push state to the `Path` history if `step % save_stride == 0` to conserve memory.

### 5.2. Output Artifacts
*   **`Ensemble`**: A collection of `Path` objects.
*   **`EnsembleStats`**: Aggregated statistics (Mean, Variance, First Passage Times) computed efficiently over the ensemble.
*   **Parquet Integration**: The `bicep-io` crate and CLI examples demonstrate serializing trajectories to Apache Parquet for downstream analysis.

## 6. Usage Example (Conceptual)

```rust
// 1. Define Model
let model = GeometricBrownianMotion::new(mu, sigma);

// 2. Configure Sampler
let integrator = EulerMaruyama;
let sampler = Sampler::new(integrator, model.clone(), model);

// 3. Define Spec
let spec = PathSpec::new(steps, dt).with_stride(10);

// 4. Run (Parallel)
let ensemble = sampler.run_paths(Calc::Ito, &spec, &initial_states, &Boundary::None, &Stopping::default(), seed);
```
