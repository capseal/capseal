use super::{Calc, SdeIntegrator};
use crate::diffusion::Diffusion;
use crate::drift::Drift;
use crate::{State, Time};

/// Stochastic Heun method (midpoint rule) for Stratonovich SDEs
#[derive(Clone, Copy, Debug)]
pub struct HeunStratonovich;

impl SdeIntegrator for HeunStratonovich {
    fn step(
        &self,
        _calc: Calc, // Always treats as Stratonovich
        t: Time,
        x: &State,
        dt: f64,
        dW: &State,
        drift: &impl Drift,
        diffusion: &impl Diffusion,
    ) -> State {
        // Predictor step
        let mu0 = drift.mu(t, x);
        let sigma0 = diffusion.sigma(t, x);
        let x_tilde = State(&x.0 + &mu0.0 * dt + &sigma0 * &dW.0);

        // Corrector step with midpoint
        let mu1 = drift.mu(t + dt, &x_tilde);
        let sigma1 = diffusion.sigma(t + dt, &x_tilde);

        // Average drift and diffusion (midpoint rule)
        let mu_mid = State(0.5 * (&mu0.0 + &mu1.0));
        let sigma_mid = 0.5 * (sigma0 + sigma1);

        // Final step using midpoint values
        State(&x.0 + &mu_mid.0 * dt + sigma_mid * &dW.0)
    }
}
