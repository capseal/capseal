pub mod actions;
pub mod graph;
pub mod priors;
pub mod propagation;

pub use actions::{pick_next_node, ActionDecoder};
pub use graph::{Edge, Graph, NodeFeat};
pub use priors::{PriorSource, Priors};
pub use propagation::{propagate_committor, PropConfig, StepPolicy};

/// Core types
pub type NodeId = usize;
pub type F = f32;
pub type Severity = f32; // [0,1] from ENN
pub type Committor = f32; // [0,1] value

const RELIABILITY_OOD_THRESHOLD: F = 0.05;
const DEFAULT_RISK_FALLBACK: F = 1.0;

/// Integration with BICEP/ENN
#[derive(Clone, Debug)]
pub struct FusionState {
    pub q_prior_enn: F,      // ENN's q prediction
    pub obs_reliability: F,  // Calibrated reliability [0,1]
    pub risk_aversion: F,    // Policy knob [0,1]
    pub bicep_confidence: F, // BICEP path reliability
}

impl FusionState {
    pub fn new(q_prior_enn: F, obs_reliability: F, risk_aversion: F, bicep_confidence: F) -> Self {
        Self {
            q_prior_enn,
            obs_reliability: obs_reliability.clamp(0.0, 1.0),
            risk_aversion: risk_aversion.clamp(0.0, 1.0),
            bicep_confidence,
        }
    }

    /// Backwards-compatible constructor from severity (legacy semantics).
    pub fn from_severity(q_prior_enn: F, severity: Severity, bicep_confidence: F) -> Self {
        let s = severity.clamp(0.0, 1.0);
        let obs_reliability = (1.0 - s).clamp(0.0, 1.0);
        Self::new(q_prior_enn, obs_reliability, s, bicep_confidence)
    }

    pub fn effective_risk(&self) -> F {
        if self.obs_reliability < RELIABILITY_OOD_THRESHOLD {
            DEFAULT_RISK_FALLBACK
        } else {
            self.risk_aversion
        }
    }

    pub fn propagation_steps(&self, config: &PropConfig) -> usize {
        let risk = self.effective_risk();
        match config.step_policy {
            StepPolicy::Fixed => config.t_max.max(1),
            StepPolicy::RiskScaled => {
                if config.t_max <= 1 {
                    1
                } else {
                    let span = (config.t_max - 1) as F;
                    1 + (span * risk).floor() as usize
                }
            }
        }
    }
}
