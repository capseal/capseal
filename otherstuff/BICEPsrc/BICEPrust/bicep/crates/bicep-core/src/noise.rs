use crate::{
    seed::{SeedIdentity, SeedSpec},
    State,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal, StudentT};

#[derive(Clone, Debug)]
pub enum ShockType {
    Gaussian,
    StudentT,
}

#[derive(Clone, Debug)]
pub struct NoiseConfig {
    pub shock_type: ShockType,
    pub nu: Option<f64>,          // df for Student-t
    pub ewma_alpha: Option<f64>,  // smoothing factor in [0,1)
}

pub struct NoiseGenerator {
    rng: ChaCha20Rng,
    config: Option<NoiseConfig>,
    scale2: Option<Vec<f64>>, // EWMA variance per-dimension
}

impl NoiseGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
            config: None,
            scale2: None,
        }
    }

    pub fn from_path_id(global_seed: u64, path_id: u64) -> Self {
        // Combine seeds deterministically (legacy behavior)
        let seed = global_seed.wrapping_add(path_id.wrapping_mul(0x9e3779b97f4a7c15));
        Self::new(seed)
    }

    pub fn from_identity(seed_spec: &SeedSpec, identity: &SeedIdentity) -> Self {
        let seed = seed_spec.derive_seed(identity);
        Self::new(seed)
    }

    pub fn set_config(&mut self, cfg: Option<NoiseConfig>) {
        self.config = cfg;
        self.scale2 = None; // reset EWMA state
    }

    pub fn generate_dw(&mut self, n: usize, sqrt_dt: f64) -> State {
        if self.config.is_none() {
            // Backward-compatible Gaussian
            let values: Vec<f64> = (0..n)
                .map(|_| {
                    let sample: f64 = StandardNormal.sample(&mut self.rng);
                    sample * sqrt_dt
                })
                .collect();
            return State::new(values);
        }

        // Configured path: Student‑t support and EWMA scaling
        let cfg = self.config.as_ref().unwrap();
        let mut values = Vec::with_capacity(n);
        if self.scale2.is_none() {
            self.scale2 = Some(vec![1.0; n]);
        }
        let scale2 = self.scale2.as_mut().unwrap();

        // Precompute standardization for Student-t if needed
        let (use_t, std_t): (bool, f64) = match (&cfg.shock_type, cfg.nu) {
            (ShockType::StudentT, Some(nu)) if nu > 2.0 => (true, (nu / (nu - 2.0)).sqrt()),
            (ShockType::StudentT, _) => (true, (5.0f64 / 3.0f64).sqrt()), // default nu=5 fallback
            _ => (false, 1.0),
        };
        let t_dist: Option<StudentT<f64>> = if use_t {
            Some(StudentT::new(cfg.nu.unwrap_or(5.0)).unwrap())
        } else {
            None
        };

        for j in 0..n {
            let mut z = if let Some(ref t) = t_dist {
                // Standardize to unit variance
                t.sample(&mut self.rng) / std_t
            } else {
                StandardNormal.sample(&mut self.rng)
            };
            if let Some(alpha) = cfg.ewma_alpha {
                let a = alpha.max(0.0).min(0.9999);
                scale2[j] = a * scale2[j] + (1.0 - a) * (z * z);
                let s = scale2[j].max(1e-12).sqrt();
                z *= s;
            }
            values.push(z * sqrt_dt);
        }
        State::new(values)
    }

    pub fn generate_antithetic_pair(&mut self, n: usize, sqrt_dt: f64) -> (State, State) {
        let dw1 = self.generate_dw(n, sqrt_dt);
        let dw2 = State(dw1.0.map(|x| -x));
        (dw1, dw2)
    }
}
