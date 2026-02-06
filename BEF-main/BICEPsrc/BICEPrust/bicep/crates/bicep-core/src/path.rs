use crate::Time;

#[derive(Clone, Debug)]
pub struct PathSpec {
    pub n_steps: usize,
    pub dt: f64,
    pub save_stride: usize,
}

impl PathSpec {
    pub fn new(n_steps: usize, dt: f64) -> Self {
        Self {
            n_steps,
            dt,
            save_stride: 1,
        }
    }

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.save_stride = stride.max(1);
        self
    }

    pub fn final_time(&self) -> Time {
        self.dt * self.n_steps as f64
    }
}

#[derive(Clone, Debug)]
pub struct Path<S> {
    pub times: Vec<Time>,
    pub states: Vec<S>,
}

impl<S> Path<S> {
    pub fn new() -> Self {
        Self {
            times: Vec::new(),
            states: Vec::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            times: Vec::with_capacity(capacity),
            states: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, t: Time, state: S) {
        self.times.push(t);
        self.states.push(state);
    }

    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    pub fn final_state(&self) -> Option<&S> {
        self.states.last()
    }

    pub fn initial_state(&self) -> Option<&S> {
        self.states.first()
    }
}

#[derive(Clone, Debug)]
pub struct Ensemble<S> {
    pub paths: Vec<Path<S>>,
}

impl<S> Ensemble<S> {
    pub fn new() -> Self {
        Self { paths: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            paths: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, path: Path<S>) {
        self.paths.push(path);
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}
