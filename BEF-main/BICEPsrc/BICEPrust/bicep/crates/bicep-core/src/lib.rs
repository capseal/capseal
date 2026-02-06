pub mod diffusion;
pub mod drift;
pub mod integrators;
pub mod measure;
pub mod noise;
pub mod path;
pub mod seed;
pub mod state;

// Core types
pub type F = f64;
pub use noise::{NoiseGenerator, NoiseConfig, ShockType};
pub use state::{State, Time};

// SDE traits
pub use diffusion::Diffusion;
pub use drift::Drift;

// Integrators
pub use integrators::{Calc, EulerMaruyama, HeunStratonovich, Milstein, SdeIntegrator};

// Path and ensemble types
pub use path::{Ensemble, Path, PathSpec};
pub use seed::{SeedIdentity, SeedSpec};

// Convenience aliases for SDE-specific usage
pub type SdePath = Path<State>;
pub type SdeEnsemble = Ensemble<State>;
