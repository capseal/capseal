use crate::{State, Time};
use nalgebra::DMatrix;

pub trait Diffusion: Send + Sync {
    /// σ(t,x) as a matrix mapping dW (R^m) -> state (R^n)
    fn sigma(&self, t: Time, x: &State) -> DMatrix<f64>;

    /// Optional Jacobian ∂σ/∂x for Stratonovich correction
    /// Returns a vector of matrices where jacs[j] is the Jacobian of the j-th column of σ
    fn sigma_jacobian(&self, _t: Time, _x: &State) -> Option<Vec<DMatrix<f64>>> {
        None
    }

    /// Number of noise dimensions (columns in σ matrix)
    fn noise_dim(&self, t: Time, x: &State) -> usize {
        self.sigma(t, x).ncols()
    }
}
