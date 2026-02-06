use crate::{State, Time};

/// Hook for future change-of-measure operations (e.g., Girsanov)
pub trait MeasureChange: Send + Sync {
    fn radon_nikodym_derivative(&self, t: Time, x: &State) -> f64;
}
