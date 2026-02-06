use crate::{NodeId, F};
use serde::{Deserialize, Serialize};

/// Prior information for committor propagation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Priors {
    pub q0: Vec<Option<F>>, // Initial committor estimates (None = no prior)
    pub eta: Vec<F>,        // Confidence weights per node
}

impl Priors {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            q0: vec![None; num_nodes],
            eta: vec![0.0; num_nodes],
        }
    }

    /// Set hard boundary condition (goal/fail)
    pub fn set_boundary(&mut self, node: NodeId, value: F, confidence: F) {
        if node < self.q0.len() {
            self.q0[node] = Some(value);
            self.eta[node] = confidence;
        }
    }

    /// Set soft prior (from ENN/BICEP)
    pub fn set_prior(&mut self, node: NodeId, value: F, confidence: F) {
        if node < self.q0.len() {
            self.q0[node] = Some(value);
            self.eta[node] = confidence;
        }
    }

    /// Set prior using reliability mapping to eta
    pub fn set_prior_with_reliability(
        &mut self,
        node: NodeId,
        value: F,
        obs_reliability: F,
        eta_min: F,
        eta_max: F,
    ) {
        if node < self.q0.len() {
            let eta = confidence::from_reliability(obs_reliability, eta_min, eta_max);
            self.set_prior(node, value, eta);
        }
    }

    /// Mark goal nodes (q = 1, high confidence)
    pub fn set_goals(&mut self, goal_nodes: &[NodeId]) {
        for &node in goal_nodes {
            self.set_boundary(node, 1.0, 1e9);
        }
    }

    /// Mark fail nodes (q = 0, high confidence)
    pub fn set_fails(&mut self, fail_nodes: &[NodeId]) {
        for &node in fail_nodes {
            self.set_boundary(node, 0.0, 1e9);
        }
    }

    /// Clear all priors
    pub fn clear(&mut self) {
        for i in 0..self.q0.len() {
            self.q0[i] = None;
            self.eta[i] = 0.0;
        }
    }

    pub fn len(&self) -> usize {
        self.q0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.q0.is_empty()
    }
}

/// Source of prior information
#[derive(Clone, Debug)]
pub enum PriorSource {
    ENN {
        q_pred: F,     // ENN's q prediction
        confidence: F, // Based on ENN uncertainty
    },
    BICEP {
        success_rate: F, // Empirical success rate from rollouts
        n_paths: usize,  // Number of BICEP paths
        variance: F,     // Variance across paths
    },
    Manual {
        value: F,
        confidence: F,
    },
}

impl PriorSource {
    /// Convert to (value, confidence) pair
    pub fn to_prior(&self) -> (F, F) {
        match self {
            PriorSource::ENN { q_pred, confidence } => (*q_pred, *confidence),
            PriorSource::BICEP {
                success_rate,
                n_paths,
                variance,
            } => {
                // Confidence increases with more paths, decreases with variance
                let confidence = (*n_paths as F) / ((*n_paths as F) + 16.0);
                let confidence = confidence * (1.0 - variance.min(0.5) * 2.0);
                (*success_rate, confidence.max(0.01))
            }
            PriorSource::Manual { value, confidence } => (*value, *confidence),
        }
    }

    /// Create ENN-based prior with severity scaling
    pub fn from_enn(q_pred: F, severity: F, base_confidence: F) -> Self {
        // Higher severity = lower confidence in prediction
        let confidence = base_confidence * (1.0 - severity * 0.5);
        Self::ENN { q_pred, confidence }
    }

    /// Create ENN prior using calibrated reliability directly
    pub fn from_enn_reliability(q_pred: F, obs_reliability: F, eta_min: F, eta_max: F) -> Self {
        let confidence = confidence::from_reliability(obs_reliability, eta_min, eta_max);
        Self::ENN { q_pred, confidence }
    }

    /// Create BICEP-based prior from simulation results
    pub fn from_bicep_paths(outcomes: &[bool]) -> Self {
        let n_paths = outcomes.len();
        let success_count = outcomes.iter().filter(|&&x| x).count();
        let success_rate = if n_paths > 0 {
            success_count as F / n_paths as F
        } else {
            0.5
        };

        // Estimate variance (binomial)
        let variance = if n_paths > 1 {
            success_rate * (1.0 - success_rate) / (n_paths as F)
        } else {
            0.25
        };

        Self::BICEP {
            success_rate,
            n_paths,
            variance,
        }
    }
}

/// Builder for constructing priors from multiple sources
pub struct PriorBuilder {
    priors: Priors,
}

impl PriorBuilder {
    pub fn new(num_nodes: usize) -> Self {
        Self {
            priors: Priors::new(num_nodes),
        }
    }

    /// Add prior from any source
    pub fn add_prior(mut self, node: NodeId, source: PriorSource) -> Self {
        let (value, confidence) = source.to_prior();
        self.priors.set_prior(node, value, confidence);
        self
    }

    /// Set goal nodes
    pub fn with_goals(mut self, goal_nodes: &[NodeId]) -> Self {
        self.priors.set_goals(goal_nodes);
        self
    }

    /// Set fail nodes
    pub fn with_fails(mut self, fail_nodes: &[NodeId]) -> Self {
        self.priors.set_fails(fail_nodes);
        self
    }

    /// Build final priors
    pub fn build(self) -> Priors {
        self.priors
    }
}

/// Confidence estimation utilities
pub mod confidence {
    use super::F;

    /// Confidence from BICEP path statistics
    pub fn from_bicep(n_paths: usize, variance: F) -> F {
        let path_factor = (n_paths as F) / ((n_paths as F) + 16.0);
        let variance_factor = 1.0 - (variance * 2.0).min(0.9);
        (path_factor * variance_factor).max(0.01)
    }

    /// Confidence from ENN entropy/severity
    pub fn from_enn_entropy(entropy: F, severity: F, base_conf: F) -> F {
        // Lower entropy = higher confidence
        let entropy_factor = (-entropy).exp().min(1.0);
        // Lower severity = higher confidence
        let severity_factor = 1.0 - severity.min(0.9);
        (base_conf * entropy_factor * severity_factor).max(0.01)
    }

    /// Adaptive confidence based on local graph structure
    pub fn adaptive_spatial(
        distance_to_goal: F,
        local_density: F, // Number of nearby nodes
        max_distance: F,
    ) -> F {
        let dist_factor = 1.0 - (distance_to_goal / max_distance).min(1.0);
        let density_factor = (local_density / 10.0).min(1.0);
        (0.1 + 0.9 * dist_factor * density_factor).min(1.0)
    }

    /// Confidence derived from calibrated reliability
    pub fn from_reliability(obs_reliability: F, eta_min: F, eta_max: F) -> F {
        let rel = obs_reliability.clamp(0.0, 1.0);
        eta_min + (eta_max - eta_min) * rel
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_priors_basic() {
        let mut priors = Priors::new(5);

        priors.set_boundary(0, 1.0, 1e9); // Goal
        priors.set_boundary(4, 0.0, 1e9); // Fail
        priors.set_prior(2, 0.7, 0.5); // ENN prior

        assert_eq!(priors.q0[0], Some(1.0));
        assert_eq!(priors.q0[4], Some(0.0));
        assert_eq!(priors.q0[2], Some(0.7));
        assert_eq!(priors.q0[1], None);

        assert!(priors.eta[0] > 1e6);
        assert!(priors.eta[4] > 1e6);
        assert!((priors.eta[2] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_prior_source_enn() {
        let source = PriorSource::from_enn(0.8, 0.3, 1.0);
        let (value, confidence) = source.to_prior();

        assert!((value - 0.8).abs() < 1e-6);
        assert!(confidence < 1.0); // Should be reduced due to severity
        assert!(confidence > 0.5); // But not too much
    }

    #[test]
    fn test_prior_source_bicep() {
        let outcomes = vec![true, true, false, true, false, true]; // 4/6 success
        let source = PriorSource::from_bicep_paths(&outcomes);

        let (value, confidence) = source.to_prior();
        assert!((value - 2.0 / 3.0).abs() < 1e-2); // ~0.67
        assert!(confidence > 0.0);
        assert!(confidence < 1.0);
    }

    #[test]
    fn test_builder() {
        let priors = PriorBuilder::new(4)
            .with_goals(&[3])
            .with_fails(&[0])
            .add_prior(
                1,
                PriorSource::Manual {
                    value: 0.6,
                    confidence: 0.8,
                },
            )
            .build();

        assert_eq!(priors.q0[3], Some(1.0));
        assert_eq!(priors.q0[0], Some(0.0));
        assert_eq!(priors.q0[1], Some(0.6));
        assert_eq!(priors.q0[2], None);
    }

    #[test]
    fn test_confidence_functions() {
        // BICEP confidence
        let conf1 = confidence::from_bicep(100, 0.1);
        let conf2 = confidence::from_bicep(10, 0.1);
        assert!(conf1 > conf2); // More paths = higher confidence

        let conf3 = confidence::from_bicep(100, 0.01);
        let conf4 = confidence::from_bicep(100, 0.3);
        assert!(conf3 > conf4); // Lower variance = higher confidence

        // ENN confidence
        let conf5 = confidence::from_enn_entropy(0.1, 0.2, 1.0);
        let conf6 = confidence::from_enn_entropy(1.0, 0.8, 1.0);
        assert!(conf5 > conf6); // Lower entropy + severity = higher confidence
    }
}
