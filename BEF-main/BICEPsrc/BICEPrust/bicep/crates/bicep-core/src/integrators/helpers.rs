use crate::diffusion::Diffusion;
use crate::{State, Time};
use nalgebra::DVector;

/// Computes the Stratonovich drift correction: -0.5 * Σ_j σ_j * (∂σ_j/∂x)
pub fn stratonovich_correction(t: Time, x: &State, diffusion: &impl Diffusion) -> Option<State> {
    let jacs = diffusion.sigma_jacobian(t, x)?;
    let sigma = diffusion.sigma(t, x);
    let n = x.dim();
    let mut correction = DVector::zeros(n);

    // For each noise dimension j
    for (j, jac_j) in jacs.iter().enumerate() {
        // σ_j is the j-th column of σ
        let sigma_j = sigma.column(j);
        // correction += jac_j * σ_j
        correction += jac_j * &sigma_j;
    }

    Some(State(0.5 * correction))
}
