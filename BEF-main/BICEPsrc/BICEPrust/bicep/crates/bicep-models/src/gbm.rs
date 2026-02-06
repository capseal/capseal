use bicep_core::{State, Time, Drift, Diffusion};
use nalgebra::DMatrix;

/// Geometric Brownian Motion: dX_t = μ X_t dt + σ X_t dW_t
#[derive(Clone, Debug)]
pub struct GeometricBrownianMotion {
    pub mu: f64,
    pub sigma: f64,
}

impl GeometricBrownianMotion {
    pub fn new(mu: f64, sigma: f64) -> Self {
        Self { mu, sigma }
    }
    
    /// Exact solution for GBM at time t
    /// X_t = X_0 * exp((μ - σ²/2)t + σ W_t)
    pub fn exact_mean(&self, x0: f64, t: Time) -> f64 {
        x0 * (self.mu * t).exp()
    }
    
    /// Exact variance for GBM
    /// Var[X_t] = X_0² * exp(2μt) * (exp(σ²t) - 1)
    pub fn exact_variance(&self, x0: f64, t: Time) -> f64 {
        let exp_2mu_t = (2.0 * self.mu * t).exp();
        let exp_sigma2_t = (self.sigma * self.sigma * t).exp();
        x0 * x0 * exp_2mu_t * (exp_sigma2_t - 1.0)
    }
}

impl Drift for GeometricBrownianMotion {
    fn mu(&self, _t: Time, x: &State) -> State {
        // μ X_t (elementwise multiplication)
        State(self.mu * &x.0)
    }
}

impl Diffusion for GeometricBrownianMotion {
    fn sigma(&self, _t: Time, x: &State) -> DMatrix<f64> {
        let n = x.dim();
        // Diagonal matrix with σ * x_i on diagonal
        let mut s = DMatrix::zeros(n, n);
        for i in 0..n {
            s[(i, i)] = self.sigma * x.0[i];
        }
        s
    }
    
    fn sigma_jacobian(&self, _t: Time, x: &State) -> Option<Vec<DMatrix<f64>>> {
        let n = x.dim();
        let mut jacobians = Vec::with_capacity(n);
        
        // For GBM, σ_i(x) = σ * x_i * e_i
        // So ∂σ_i/∂x_j = σ * δ_ij (Kronecker delta)
        for i in 0..n {
            let mut jac = DMatrix::zeros(n, n);
            jac[(i, i)] = self.sigma;
            jacobians.push(jac);
        }
        
        Some(jacobians)
    }
}