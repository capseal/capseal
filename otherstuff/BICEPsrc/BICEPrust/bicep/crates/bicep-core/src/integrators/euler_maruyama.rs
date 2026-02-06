use super::helpers::stratonovich_correction;
use super::{Calc, SdeIntegrator};
use crate::diffusion::Diffusion;
use crate::drift::Drift;
use crate::{State, Time};

#[derive(Clone, Copy, Debug)]
pub struct EulerMaruyama;

impl SdeIntegrator for EulerMaruyama {
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
                // μ° = μ - 0.5 * Σ_j σ_j * (∂σ_j/∂x)
                mu = State(&mu.0 - &correction.0);
            } else {
                // If no Jacobian provided, warn that Heun should be used instead
                eprintln!(
                    "Warning: Stratonovich mode requested but no Jacobian provided. \
                          Consider using HeunStratonovich integrator instead."
                );
            }
        }

        // Get diffusion coefficient
        let sigma = diffusion.sigma(t, x);

        // Euler-Maruyama step: X_{t+dt} = X_t + μ*dt + σ*dW
        State(&x.0 + &mu.0 * dt + sigma * &dW.0)
    }
}
