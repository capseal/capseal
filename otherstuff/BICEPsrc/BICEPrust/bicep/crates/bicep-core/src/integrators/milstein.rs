use super::helpers::stratonovich_correction;
use super::{Calc, SdeIntegrator};
use crate::diffusion::Diffusion;
use crate::drift::Drift;
use crate::{State, Time};
use nalgebra::DVector;

/// Milstein method for SDEs with higher strong order convergence
/// Requires sigma_jacobian to be implemented for full effectiveness
#[derive(Clone, Copy, Debug)]
pub struct Milstein;

impl SdeIntegrator for Milstein {
    fn step(
        &self,
        calc: Calc,
        t: Time,
        x: &State,
        dt: f64,
        dW: &State,
        drift: &impl Drift,
        diffusion: &impl Diffusion,
    ) -> State {
        // Get base drift
        let mut mu = drift.mu(t, x);

        // Apply Stratonovich correction if needed
        if let Calc::Stratonovich = calc {
            if let Some(correction) = stratonovich_correction(t, x, diffusion) {
                mu = State(&mu.0 - &correction.0);
            }
        }

        // Get diffusion coefficient
        let sigma = diffusion.sigma(t, x);

        // Start with Euler-Maruyama step
        let mut x_next = &x.0 + &mu.0 * dt + &sigma * &dW.0;

        // Add Milstein correction terms if Jacobians are available
        if let Some(jacs) = diffusion.sigma_jacobian(t, x) {
            let n = x.dim();
            let m = sigma.ncols();
            let mut milstein_correction = DVector::zeros(n);

            // Milstein term: 0.5 * Σ_j σ_j * (∂σ_j/∂x) * ((ΔW_j)² - Δt)
            for j in 0..m {
                let sigma_j = sigma.column(j).into_owned();
                let jac_j = &jacs[j];
                let dW_j = dW.0[j];

                // (ΔW_j)² - Δt term
                let levy_area = dW_j * dW_j - dt;

                // Add contribution: σ_j * (∂σ_j/∂x) * levy_area
                milstein_correction += jac_j * sigma_j * (0.5 * levy_area);
            }

            x_next += milstein_correction;
        } else {
            // If no Jacobian available, warn and fall back to Euler-Maruyama
            eprintln!(
                "Warning: Milstein method requires sigma_jacobian. \
                      Falling back to Euler-Maruyama (lower strong order)."
            );
        }

        State(x_next)
    }
}
