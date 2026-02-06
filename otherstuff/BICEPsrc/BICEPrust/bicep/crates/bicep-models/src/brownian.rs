use bicep_core::{State, Time, Drift, Diffusion};
use nalgebra::DMatrix;

/// Standard Brownian Motion: dX_t = 0 dt + σ dW_t
#[derive(Clone, Debug)]
pub struct BrownianMotion {
    pub sigma: f64,
}

impl BrownianMotion {
    pub fn new(sigma: f64) -> Self {
        Self { sigma }
    }
    
    /// Standard Brownian motion with σ = 1
    pub fn standard() -> Self {
        Self::new(1.0)
    }
    
    /// Exact mean for Brownian motion (always zero)
    pub fn exact_mean(&self, _x0: f64, _t: Time) -> f64 {
        0.0
    }
    
    /// Exact variance for Brownian motion
    /// Var[X_t | X_0] = σ²t
    pub fn exact_variance(&self, t: Time) -> f64 {
        self.sigma * self.sigma * t
    }
}

impl Drift for BrownianMotion {
    fn mu(&self, _t: Time, x: &State) -> State {
        // Zero drift
        State::zeros(x.dim())
    }
}

impl Diffusion for BrownianMotion {
    fn sigma(&self, _t: Time, x: &State) -> DMatrix<f64> {
        let n = x.dim();
        // Constant diffusion: σ * I
        DMatrix::from_diagonal_element(n, n, self.sigma)
    }
    
    fn sigma_jacobian(&self, _t: Time, x: &State) -> Option<Vec<DMatrix<f64>>> {
        let n = x.dim();
        // Constant diffusion, so all Jacobians are zero
        Some(vec![DMatrix::zeros(n, n); n])
    }
}