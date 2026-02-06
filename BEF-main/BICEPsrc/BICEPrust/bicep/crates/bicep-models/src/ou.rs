use bicep_core::{State, Time, Drift, Diffusion};
use nalgebra::DMatrix;

/// Ornstein-Uhlenbeck process: dX_t = θ(μ - X_t) dt + σ dW_t
#[derive(Clone, Debug)]
pub struct OrnsteinUhlenbeck {
    pub theta: f64,    // Mean reversion rate
    pub mu: f64,       // Long-term mean
    pub sigma: f64,    // Volatility
}

impl OrnsteinUhlenbeck {
    pub fn new(theta: f64, mu: f64, sigma: f64) -> Self {
        Self { theta, mu, sigma }
    }
    
    /// Exact mean for OU process
    /// E[X_t | X_0] = μ + (X_0 - μ) * exp(-θt)
    pub fn exact_mean(&self, x0: f64, t: Time) -> f64 {
        self.mu + (x0 - self.mu) * (-self.theta * t).exp()
    }
    
    /// Exact variance for OU process
    /// Var[X_t | X_0] = σ²/(2θ) * (1 - exp(-2θt))
    pub fn exact_variance(&self, t: Time) -> f64 {
        if self.theta.abs() < 1e-10 {
            // Limit as θ → 0: Brownian motion variance
            self.sigma * self.sigma * t
        } else {
            let sigma2_over_2theta = self.sigma * self.sigma / (2.0 * self.theta);
            sigma2_over_2theta * (1.0 - (-2.0 * self.theta * t).exp())
        }
    }
    
    /// Stationary variance (t → ∞)
    /// Var_∞ = σ²/(2θ)
    pub fn stationary_variance(&self) -> f64 {
        if self.theta > 0.0 {
            self.sigma * self.sigma / (2.0 * self.theta)
        } else {
            f64::INFINITY
        }
    }
}

impl Drift for OrnsteinUhlenbeck {
    fn mu(&self, _t: Time, x: &State) -> State {
        // θ(μ - X_t)
        State(self.theta * x.0.map(|xi| self.mu - xi))
    }
}

impl Diffusion for OrnsteinUhlenbeck {
    fn sigma(&self, _t: Time, x: &State) -> DMatrix<f64> {
        let n = x.dim();
        // Constant diffusion: σ * I
        DMatrix::from_diagonal_element(n, n, self.sigma)
    }
    
    fn sigma_jacobian(&self, _t: Time, x: &State) -> Option<Vec<DMatrix<f64>>> {
        let n = x.dim();
        // Since σ is constant, all Jacobians are zero
        Some(vec![DMatrix::zeros(n, n); n])
    }
}