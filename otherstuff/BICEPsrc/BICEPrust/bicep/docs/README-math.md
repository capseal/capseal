# BICEP Mathematical Contract

This document locks the mathematical foundations for BICEP (Brownian-Inspired Computation Engine for Paths). All implementations must strictly adhere to these definitions.

## SDE Forms Supported (Vector-Valued)

### Itô Interpretation
```
dX_t = μ(t,X_t) dt + σ(t,X_t) dW_t
```

### Stratonovich Interpretation  
```
dX_t = μ°(t,X_t) dt + σ(t,X_t) ∘ dW_t
```

## Itô-Stratonovich Bridge

**Component-wise drift correction:**
```
μ°_i = μ_i - (1/2) ∑_j σ_ij (∂σ_ij/∂x_j)
```

This transforms Itô drift to Stratonovich drift. The correction term arises from the quadratic variation of the Stratonovich integral.

## Numerical Integrators

### Euler-Maruyama (Itô)
- **Weak order**: 1
- **Strong order**: 1/2
- **Update rule**: X_{n+1} = X_n + μ(t_n, X_n)Δt + σ(t_n, X_n)ΔW_n
- **Stratonovich mode**: Apply drift correction μ° = μ - correction

### Milstein (Itô)
- **Strong order**: 1 (when ∂σ/∂x available)
- **Additional term**: (1/2) ∑_j σ_j (∂σ_j/∂x) ((ΔW_j)² - Δt)
- **Requirement**: Jacobian σ_jacobian must be provided

### Stochastic Heun (Stratonovich)
- **Weak order**: 1
- **Method**: Predictor-corrector with midpoint evaluation
- **Natural**: No drift correction needed (native Stratonovich)

## Boundary Conditions

### Absorbing
- **Rule**: Stop path when X_t enters boundary set
- **Application**: First-passage time problems

### Reflecting
- **Rule**: Reflect normal component of velocity at boundary
- **Conservation**: Maintains proper measure

### Periodic
- **Rule**: Wrap coordinates: x → x mod (high - low) + low
- **Topology**: Torus geometry

## Stopping Conditions

### First-Hit Times
- **Target sets**: A, B for transition path sampling
- **Record**: Time τ when first X_τ ∈ Target

### Maximum Time
- **Cutoff**: Stop at T_max regardless of hits
- **Use case**: Prevent infinite simulations

## Determinism Policy

### Reproducible Parallelism
- **Constraint**: Results independent of thread count
- **Implementation**: Counter-based RNG with per-path seeding
- **Seed derivation**: seed_path = hash64(global_seed, path_id)

### Random Number Generation
- **Base PRNG**: ChaCha12 (counter-based, parallel-safe)
- **Normal sampling**: StandardNormal distribution
- **Scaling**: dW ~ N(0, dt) for time step dt

## Variance Reduction Techniques

### Antithetic Variates
- **Pairing**: For each ΔW, also simulate -ΔW
- **Reduction**: ~50% variance for smooth functions

### Common Random Numbers
- **Coupling**: Use identical ΔW across parameter comparisons
- **Application**: A/B testing, sensitivity analysis

### Control Variates
- **Correction**: Subtract known expectations (e.g., GBM moments)
- **Requirement**: Analytical benchmarks available

## Model Requirements

Each model must implement:

### Drift Interface
```rust
trait Drift {
    fn mu(&self, t: Time, x: &State) -> State;
}
```

### Diffusion Interface  
```rust
trait Diffusion {
    fn sigma(&self, t: Time, x: &State) -> DMatrix<F>;
    fn sigma_jacobian(&self, t: Time, x: &State) -> Option<Vec<DMatrix<F>>>;
}
```

## Validation Standards

### Unit Tests
- **Brownian motion**: E[X_T] = 0, Var[X_T] = T
- **GBM moments**: Log-normal distribution properties
- **OU process**: Mean reversion and stationary variance

### Convergence Tests
- **Weak order**: Bias scales as O(Δt^p) for order p
- **Strong order**: RMS error scales as O(Δt^q) for order q

### Calculus Consistency
- **Equivalence**: Itô(corrected) ≈ Stratonovich for same σ
- **Convergence**: KS distance → 0 as Δt → 0

## Error Tolerances

### Floating Point
- **Absolute**: 1e-12 for exact zero expectations
- **Relative**: 1e-10 for ratio tests
- **Statistical**: 3σ confidence intervals for Monte Carlo

### Determinism
- **Reproducibility**: Exact bit-for-bit across runs
- **Thread safety**: No dependence on execution order

This mathematical contract serves as the foundation for all BICEP implementations. Any deviation requires explicit documentation and justification.