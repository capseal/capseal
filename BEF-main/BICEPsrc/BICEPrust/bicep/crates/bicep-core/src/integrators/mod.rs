pub mod euler_maruyama;
pub mod helpers;
pub mod heun_stratonovich;
pub mod milstein;

use crate::diffusion::Diffusion;
use crate::drift::Drift;
use crate::{State, Time};

pub use euler_maruyama::EulerMaruyama;
pub use heun_stratonovich::HeunStratonovich;
pub use milstein::Milstein;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Calc {
    Ito,
    Stratonovich,
}

pub trait SdeIntegrator: Send + Sync {
    fn step(
        &self,
        calc: Calc,
        t: Time,
        x: &State,
        dt: f64,
        dW: &State,
        drift: &impl Drift,
        diffusion: &impl Diffusion,
    ) -> State;
}
