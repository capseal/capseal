use crate::{State, Time};

pub trait Drift: Send + Sync {
    fn mu(&self, t: Time, x: &State) -> State;
}
