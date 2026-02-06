use bicep_core::{
    seed::{SeedIdentity, SeedSpec},
    Calc, Diffusion, Drift, NoiseConfig, NoiseGenerator, SdeIntegrator, State, Time, F,
};
use nalgebra::DVector;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Path specification for simulation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PathSpec {
    pub n_steps: usize,
    pub dt: F,
    pub save_stride: usize, // Save every nth step
}

/// Single path trajectory
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Path {
    pub times: Vec<Time>,
    pub states: Vec<State>,
    pub stopped: bool,
    pub stop_time: Option<Time>,
    pub hit_set: Option<String>,
}

/// Collection of paths (ensemble)
#[derive(Clone, Debug)]
pub struct Ensemble {
    pub paths: Vec<Path>,
    pub spec: PathSpec,
}

/// Statistical summary of ensemble
#[derive(Clone, Debug)]
pub struct EnsembleStats {
    pub n_paths: usize,
    pub means: DVector<F>,
    pub variances: DVector<F>,
    pub first_passage_times: Vec<Time>,
}

/// Boundary conditions
pub enum Boundary {
    None,
    Absorbing(Box<dyn Fn(&State) -> bool + Send + Sync>),
    Reflecting(Box<dyn Fn(&State) -> State + Send + Sync>),
    Periodic { low: DVector<F>, high: DVector<F> },
}

/// Stopping conditions
pub struct Stopping {
    pub first_hit: Option<Box<dyn Fn(&State) -> Option<String> + Send + Sync>>,
    pub max_time: Option<F>,
}

/// Main sampler for path generation
pub struct Sampler<I, D, S>
where
    I: SdeIntegrator,
    D: Drift,
    S: Diffusion,
{
    pub integrator: I,
    pub drift: D,
    pub diffusion: S,
    pub noise_config: Option<NoiseConfig>,
}

impl<I, D, S> Sampler<I, D, S>
where
    I: SdeIntegrator + Clone,
    D: Drift + Clone,
    S: Diffusion + Clone,
{
    pub fn new(integrator: I, drift: D, diffusion: S) -> Self {
        Self {
            integrator,
            drift,
            diffusion,
            noise_config: None,
        }
    }

    pub fn with_noise_config(mut self, config: Option<NoiseConfig>) -> Self {
        self.noise_config = config;
        self
    }

    /// Run ensemble of paths in parallel
    pub fn run_paths(
        &self,
        calc: Calc,
        spec: &PathSpec,
        x0s: &[State],
        boundary: &Boundary,
        stopping: &Stopping,
        global_seed: u64,
    ) -> Ensemble {
        let paths: Vec<Path> = x0s
            .par_iter()
            .enumerate()
            .map(|(path_id, x0)| {
                let mut rng = NoiseGenerator::from_path_id(global_seed, path_id as u64);
                if self.noise_config.is_some() { rng.set_config(self.noise_config.clone()); }
                self.run_single_path(calc, spec, x0.clone(), boundary, stopping, &mut rng)
            })
            .collect();

        Ensemble {
            paths,
            spec: spec.clone(),
        }
    }

    /// Run ensemble with explicit SeedSpec + identity per path.
    pub fn run_paths_with_seed_spec(
        &self,
        calc: Calc,
        spec: &PathSpec,
        x0s: &[State],
        boundary: &Boundary,
        stopping: &Stopping,
        seed_spec: &SeedSpec,
        identities: &[SeedIdentity],
    ) -> Ensemble {
        assert_eq!(
            x0s.len(),
            identities.len(),
            "identities must align with x0s"
        );
        let paths: Vec<Path> = x0s
            .par_iter()
            .enumerate()
            .map(|(idx, x0)| {
                let identity = identities.get(idx).expect("identity missing for path");
                let mut rng = NoiseGenerator::from_identity(seed_spec, identity);
                if self.noise_config.is_some() { rng.set_config(self.noise_config.clone()); }
                self.run_single_path(calc, spec, x0.clone(), boundary, stopping, &mut rng)
            })
            .collect();

        Ensemble {
            paths,
            spec: spec.clone(),
        }
    }

    /// Replay a single path deterministically given a SeedSpec and identity.
    pub fn run_single_path_with_seed_spec(
        &self,
        calc: Calc,
        spec: &PathSpec,
        x0: State,
        boundary: &Boundary,
        stopping: &Stopping,
        seed_spec: &SeedSpec,
        identity: &SeedIdentity,
    ) -> Path {
        let mut rng = NoiseGenerator::from_identity(seed_spec, identity);
        if self.noise_config.is_some() { rng.set_config(self.noise_config.clone()); }
        self.run_single_path(calc, spec, x0, boundary, stopping, &mut rng)
    }

    /// Run a single path (called by run_paths)
    fn run_single_path(
        &self,
        calc: Calc,
        spec: &PathSpec,
        mut x: State,
        boundary: &Boundary,
        stopping: &Stopping,
        rng: &mut NoiseGenerator,
    ) -> Path {
        let mut path = Path {
            times: Vec::new(),
            states: Vec::new(),
            stopped: false,
            stop_time: None,
            hit_set: None,
        };

        let mut t = 0.0;
        let noise_dim = self.diffusion.noise_dim(t, &x);
        let sqrt_dt = spec.dt.sqrt();

        for step in 0..spec.n_steps {
            // Save state if on stride boundary
            if step % spec.save_stride == 0 {
                path.times.push(t);
                path.states.push(x.clone());
            }

            // Check stopping conditions
            if let Some(ref hit_fn) = stopping.first_hit {
                if let Some(hit_label) = hit_fn(&x) {
                    path.stopped = true;
                    path.stop_time = Some(t);
                    path.hit_set = Some(hit_label);
                    break;
                }
            }

            if let Some(max_time) = stopping.max_time {
                if t >= max_time {
                    path.stopped = true;
                    path.stop_time = Some(t);
                    break;
                }
            }

            // Generate noise
            let dw = rng.generate_dw(noise_dim, sqrt_dt);

            // Take integration step
            x = self
                .integrator
                .step(calc, t, &x, spec.dt, &dw, &self.drift, &self.diffusion);

            // Apply boundary conditions
            x = apply_boundary(boundary, x);

            t += spec.dt;
        }

        // Save final state if not already saved
        if path.times.is_empty() || path.times.last() != Some(&t) {
            path.times.push(t);
            path.states.push(x);
        }

        path
    }
}

/// Apply boundary conditions to state
pub fn apply_boundary(boundary: &Boundary, mut state: State) -> State {
    match boundary {
        Boundary::None => state,

        Boundary::Absorbing(absorb_fn) => {
            if absorb_fn(&state) {
                state
            } else {
                state
            }
        }

        Boundary::Reflecting(reflect_fn) => reflect_fn(&state),

        Boundary::Periodic { low, high } => {
            for i in 0..state.dim() {
                let range = high[i] - low[i];
                if range > 0.0 {
                    let wrapped = ((state.0[i] - low[i]) % range + range) % range + low[i];
                    state.0[i] = wrapped;
                }
            }
            state
        }
    }
}

// Implementations
impl PathSpec {
    pub fn new(n_steps: usize, dt: F, save_stride: usize) -> Self {
        Self {
            n_steps,
            dt,
            save_stride,
        }
    }

    pub fn total_time(&self) -> Time {
        self.n_steps as F * self.dt
    }

    pub fn saved_steps(&self) -> usize {
        (self.n_steps + self.save_stride - 1) / self.save_stride
    }
}

impl Path {
    pub fn new() -> Self {
        Self {
            times: Vec::new(),
            states: Vec::new(),
            stopped: false,
            stop_time: None,
            hit_set: None,
        }
    }

    pub fn final_state(&self) -> Option<&State> {
        self.states.last()
    }

    pub fn final_time(&self) -> Option<Time> {
        self.times.last().copied()
    }

    pub fn first_passage_time(&self) -> Option<Time> {
        self.stop_time
    }
}

impl Ensemble {
    pub fn new(paths: Vec<Path>, spec: PathSpec) -> Self {
        Self { paths, spec }
    }

    pub fn n_paths(&self) -> usize {
        self.paths.len()
    }

    pub fn final_statistics(&self) -> EnsembleStats {
        let final_states: Vec<&State> = self.paths.iter().filter_map(|p| p.final_state()).collect();

        if final_states.is_empty() {
            return EnsembleStats::empty();
        }

        let dim = final_states[0].dim();
        let mut means = vec![0.0; dim];
        let mut variances = vec![0.0; dim];

        // Compute means
        for state in &final_states {
            for i in 0..dim {
                means[i] += state.0[i];
            }
        }
        for mean in &mut means {
            *mean /= final_states.len() as F;
        }

        // Compute variances
        for state in &final_states {
            for i in 0..dim {
                let diff = state.0[i] - means[i];
                variances[i] += diff * diff;
            }
        }
        for variance in &mut variances {
            *variance /= (final_states.len() - 1).max(1) as F;
        }

        EnsembleStats {
            n_paths: final_states.len(),
            means: DVector::from_vec(means),
            variances: DVector::from_vec(variances),
            first_passage_times: self.first_passage_statistics(),
        }
    }

    fn first_passage_statistics(&self) -> Vec<Time> {
        self.paths
            .iter()
            .filter_map(|p| p.first_passage_time())
            .collect()
    }
}

impl EnsembleStats {
    fn empty() -> Self {
        Self {
            n_paths: 0,
            means: DVector::zeros(0),
            variances: DVector::zeros(0),
            first_passage_times: Vec::new(),
        }
    }
}

impl Default for Stopping {
    fn default() -> Self {
        Self {
            first_hit: None,
            max_time: None,
        }
    }
}

impl Stopping {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_time(mut self, max_time: F) -> Self {
        self.max_time = Some(max_time);
        self
    }

    pub fn with_first_hit<F>(mut self, hit_fn: F) -> Self
    where
        F: Fn(&State) -> Option<String> + Send + Sync + 'static,
    {
        self.first_hit = Some(Box::new(hit_fn));
        self
    }
}

impl Boundary {
    pub fn absorbing_sphere(center: DVector<F>, radius: F) -> Self {
        Boundary::Absorbing(Box::new(move |state| {
            let dist_sq = (&state.0 - &center).norm_squared();
            dist_sq <= radius * radius
        }))
    }

    pub fn reflecting_box(low: DVector<F>, high: DVector<F>) -> Self {
        Boundary::Reflecting(Box::new(move |state| {
            let mut reflected = state.clone();
            for i in 0..state.dim() {
                if reflected.0[i] < low[i] {
                    reflected.0[i] = 2.0 * low[i] - reflected.0[i];
                } else if reflected.0[i] > high[i] {
                    reflected.0[i] = 2.0 * high[i] - reflected.0[i];
                }
            }
            reflected
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bicep_core::BrownianMotion;
    use bicep_core::EulerMaruyama;

    #[test]
    fn test_path_spec() {
        let spec = PathSpec::new(1000, 0.01, 10);
        assert_eq!(spec.total_time(), 10.0);
        assert_eq!(spec.saved_steps(), 100);
    }

    #[test]
    fn test_single_brownian_path() {
        let bm = BrownianMotion::standard();
        let integrator = EulerMaruyama;
        let sampler = Sampler::new(integrator, bm.clone(), bm);

        let spec = PathSpec::new(100, 0.01, 1);
        let x0 = State::new(vec![0.0]);
        let boundary = Boundary::None;
        let stopping = Stopping::default();

        let ensemble = sampler.run_paths(Calc::Ito, &spec, &[x0], &boundary, &stopping, 42);

        assert_eq!(ensemble.n_paths(), 1);
        assert_eq!(ensemble.paths[0].states.len(), 100);

        // Check that times are properly spaced
        let path = &ensemble.paths[0];
        for i in 1..path.times.len() {
            assert!((path.times[i] - path.times[i - 1] - spec.dt).abs() < 1e-12);
        }
    }
}
