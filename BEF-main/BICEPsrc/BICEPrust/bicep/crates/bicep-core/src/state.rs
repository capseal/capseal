use nalgebra::DVector;
use serde::{Deserialize, Serialize};

pub type Time = f64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct State(pub DVector<f64>);

impl State {
    pub fn new(values: Vec<f64>) -> Self {
        State(DVector::from_vec(values))
    }

    pub fn zeros(n: usize) -> Self {
        State(DVector::zeros(n))
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }
}

impl std::ops::Deref for State {
    type Target = DVector<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for State {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<DVector<f64>> for State {
    fn from(v: DVector<f64>) -> Self {
        State(v)
    }
}

impl From<Vec<f64>> for State {
    fn from(v: Vec<f64>) -> Self {
        State::new(v)
    }
}
