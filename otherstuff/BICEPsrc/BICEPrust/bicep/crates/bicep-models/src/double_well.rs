use bicep_core::{State, Time, Drift, Diffusion};
use nalgebra::DMatrix;

/// Double-well potential model: dX_t = -∇U(X_t) dt + √(2β⁻¹) dW_t
/// where U(x) = a*x^4 - b*x^2 (quartic double-well potential)
#[derive(Clone, Debug)]
pub struct DoubleWell {
    pub a: f64,          // Quartic coefficient (a > 0)
    pub b: f64,          // Quadratic coefficient (b > 0 for double well)
    pub temperature: f64, // β⁻¹ = k_B T (inverse temperature)
}

impl DoubleWell {
    pub fn new(a: f64, b: f64, temperature: f64) -> Self {
        assert!(a > 0.0, "Quartic coefficient a must be positive for stability");
        assert!(b > 0.0, "Quadratic coefficient b must be positive for double well");
        assert!(temperature > 0.0, "Temperature must be positive");
        
        Self { a, b, temperature }
    }
    
    /// Standard symmetric double well with unit barrier
    pub fn standard() -> Self {
        Self::new(1.0, 2.0, 0.1) // Low temperature for clear barriers
    }
    
    /// Potential energy U(x) = a*x^4 - b*x^2
    pub fn potential(&self, x: f64) -> f64 {
        self.a * x.powi(4) - self.b * x * x
    }
    
    /// Force (negative gradient): -dU/dx = -4ax³ + 2bx = 2x(b - 2ax²)
    pub fn force(&self, x: f64) -> f64 {
        -4.0 * self.a * x.powi(3) + 2.0 * self.b * x
    }
    
    /// Second derivative of force (for Jacobian): d²U/dx² = 12ax² - 2b  
    pub fn force_derivative(&self, x: f64) -> f64 {
        -12.0 * self.a * x * x + 2.0 * self.b
    }
    
    /// Metastable minima locations: x = ±√(b/(2a))
    pub fn minima(&self) -> (f64, f64) {
        let x_min = (self.b / (2.0 * self.a)).sqrt();
        (-x_min, x_min)
    }
    
    /// Barrier height: U(0) - U(x_min) = b²/(4a)
    pub fn barrier_height(&self) -> f64 {
        self.b * self.b / (4.0 * self.a)
    }
    
    /// Effective noise strength: √(2β⁻¹)
    pub fn noise_strength(&self) -> f64 {
        (2.0 * self.temperature).sqrt()
    }
}

impl Drift for DoubleWell {
    fn mu(&self, _t: Time, x: &State) -> State {
        // Drift is the negative gradient of potential: -∇U(x)
        let forces: Vec<f64> = x.0.iter()
            .map(|&xi| self.force(xi))
            .collect();
        State::new(forces)
    }
}

impl Diffusion for DoubleWell {
    fn sigma(&self, _t: Time, x: &State) -> DMatrix<f64> {
        let n = x.dim();
        let noise_coeff = self.noise_strength();
        
        // Isotropic diffusion: √(2β⁻¹) * I
        DMatrix::from_diagonal_element(n, n, noise_coeff)
    }
    
    fn sigma_jacobian(&self, _t: Time, x: &State) -> Option<Vec<DMatrix<f64>>> {
        let n = x.dim();
        // Constant diffusion coefficient, so all Jacobians are zero
        Some(vec![DMatrix::zeros(n, n); n])
    }
}

/// Convenience functions for transition path sampling
impl DoubleWell {
    /// Check if state is in left well (A region)
    pub fn in_left_well(&self, x: &State, threshold: f64) -> bool {
        let (x_left, _) = self.minima();
        x.0.iter().all(|&xi| xi < x_left + threshold)
    }
    
    /// Check if state is in right well (B region)  
    pub fn in_right_well(&self, x: &State, threshold: f64) -> bool {
        let (_, x_right) = self.minima();
        x.0.iter().all(|&xi| xi > x_right - threshold)
    }
    
    /// Check if state is near transition region
    pub fn in_transition_region(&self, x: &State, width: f64) -> bool {
        x.0.iter().all(|&xi| xi.abs() < width)
    }
    
    /// Theoretical committor for 1D double well (approximate)
    /// q(x) ≈ (1 + tanh(x/√(2T))) / 2 for high barriers
    pub fn approximate_committor(&self, x: f64) -> f64 {
        let scale = (2.0 * self.temperature).sqrt();
        0.5 * (1.0 + (x / scale).tanh())
    }
}