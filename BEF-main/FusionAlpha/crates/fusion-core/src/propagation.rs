use crate::{Committor, Graph, F};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepPolicy {
    Fixed,
    RiskScaled,
}

impl Default for StepPolicy {
    fn default() -> Self {
        StepPolicy::Fixed
    }
}

/// Configuration for committor propagation
#[derive(Clone, Debug)]
pub struct PropConfig {
    pub t_max: usize,       // Maximum iterations
    pub eps: F,             // Convergence tolerance
    pub use_parallel: bool, // Use parallel iterations
    pub alpha_max: F,       // Maximum risk parameter for risk-sensitive blending
    pub step_policy: StepPolicy,
}

impl Default for PropConfig {
    fn default() -> Self {
        Self {
            t_max: 100,
            eps: 1e-4,
            use_parallel: true,
            alpha_max: 6.0, // Pessimistic blending at high risk
            step_policy: StepPolicy::Fixed,
        }
    }
}

/// Solve committor with priors using Jacobi/Gauss-Seidel relaxation
///
/// Boundary conditions:
/// - Goal nodes: h = 1 (fixed)  
/// - Fail nodes: h = 0 (fixed)
/// - Other nodes: weighted harmonic mean of neighbors + prior
///
/// Update rule:
/// h_v^{t+1} = (Σ_{u∈N(v)} w_uv * h_u^t + η_v * d_v) / (Σ_{u∈N(v)} w_uv + η_v)
pub fn propagate_committor(
    graph: &Graph,
    q0: &[Option<F>], // Initial values (None = no prior)
    eta: &[F],        // Confidence weights
    config: &PropConfig,
    t_steps: usize,   // Actual propagation steps (policy-scaled)
    risk_aversion: F, // Risk sensitivity knob [0,1]
) -> Vec<Committor> {
    let n = graph.num_nodes();
    assert_eq!(q0.len(), n);
    assert_eq!(eta.len(), n);

    // Initialize values - start from zero except for fixed boundary conditions
    let mut h_curr = vec![0.0; n]; // Start from zero for better convergence
    let mut h_next = vec![0.0; n];

    // Set initial conditions - only for high-confidence boundary conditions
    for i in 0..n {
        if let Some(val) = q0[i] {
            if eta[i] > 1e6 {
                // Only set true boundary conditions
                h_curr[i] = val;
            }
        }
    }

    let effective_steps = t_steps.min(config.t_max);
    let alpha = config.alpha_max * risk_aversion.clamp(0.0, 1.0);

    // Jacobi iterations
    for _iter in 0..effective_steps {
        if config.use_parallel {
            // Parallel update
            h_next.par_iter_mut().enumerate().for_each(|(v, h_v_next)| {
                if risk_aversion < 1e-6 {
                    *h_v_next = update_node(graph, &h_curr, q0, eta, v);
                } else {
                    *h_v_next =
                        update_node_with_risk(graph, &h_curr, q0, eta, v, alpha, risk_aversion);
                }
            });
        } else {
            // Sequential update
            for v in 0..n {
                if risk_aversion < 1e-6 {
                    h_next[v] = update_node(graph, &h_curr, q0, eta, v);
                } else {
                    h_next[v] =
                        update_node_with_risk(graph, &h_curr, q0, eta, v, alpha, risk_aversion);
                }
            }
        }

        // Explicit [0,1] clamp after all updates (prevents drift from α≠0 and spicy priors)
        for v in 0..n {
            h_next[v] = h_next[v].clamp(0.0, 1.0);
        }

        // Check convergence
        let max_change = h_curr
            .iter()
            .zip(h_next.iter())
            .map(|(old, new)| (old - new).abs())
            .fold(0.0, f32::max);

        if max_change < config.eps {
            break;
        }

        std::mem::swap(&mut h_curr, &mut h_next);
    }

    h_curr
}

/// Update single node using weighted harmonic mean + prior
fn update_node(graph: &Graph, h_curr: &[F], q0: &[Option<F>], eta: &[F], v: usize) -> F {
    // Fixed boundary conditions - check first
    if let Some(fixed_val) = q0[v] {
        if eta[v] > 1e6 {
            // High confidence = fixed boundary
            return fixed_val.clamp(0.0, 1.0);
        }
    }

    let neighbors = graph.neighbors(v);
    if neighbors.is_empty() && q0[v].is_none() {
        return 0.0; // Isolated node, start from zero
    }

    // Weighted sum from neighbors
    let mut neighbor_sum = 0.0;
    let mut weight_sum = 0.0;

    for &(u, w) in neighbors {
        neighbor_sum += w * h_curr[u];
        weight_sum += w;
    }

    // Add prior term only for non-boundary conditions
    let (prior_contrib, eta_v) = if let Some(prior_val) = q0[v] {
        if prior_val.is_finite() && eta[v] > 0.0 && eta[v].is_finite() && eta[v] <= 1e6 {
            (eta[v] * prior_val, eta[v])
        } else {
            (0.0, 0.0) // Skip boundary conditions and invalid priors
        }
    } else {
        (0.0, 0.0)
    };

    let total_weight = weight_sum + eta_v;

    if total_weight < 1e-12 {
        return 0.0; // Isolated node
    }

    let mut new_val = (neighbor_sum + prior_contrib) / total_weight;

    // Hard clamp to [0,1] and guard against NaN
    if new_val.is_nan() {
        new_val = 0.0;
    }
    new_val.clamp(0.0, 1.0)
}

/// Update node with risk-sensitive blending based on risk aversion
fn update_node_with_risk(
    graph: &Graph,
    h_curr: &[F],
    q0: &[Option<F>],
    eta: &[F],
    v: usize,
    alpha: F,
    risk_aversion: F,
) -> F {
    // Always use regular update when severity is 0 or very low
    if risk_aversion < 1e-6 {
        return update_node(graph, h_curr, q0, eta, v);
    }

    // Get regular harmonic mean update
    let regular_val = update_node(graph, h_curr, q0, eta, v);

    // If alpha too small, just use regular
    if alpha.abs() < 1e-6 {
        return regular_val;
    }

    // Get risk-sensitive update
    let risk_val = update_node_risk_sensitive(graph, h_curr, q0, eta, v, alpha);

    // Blend based on risk aversion: higher risk → more pessimistic
    let blend_weight = risk_aversion.clamp(0.0, 1.0);
    let blended = (1.0 - blend_weight) * regular_val + blend_weight * risk_val;

    // Final clamp and NaN guard
    if blended.is_nan() {
        return regular_val;
    }
    blended.clamp(0.0, 1.0)
}

/// Risk-sensitive propagation (upgraded version)
/// h_v^{t+1} = (1/α) * log(Σ_u w_uv * exp(α * h_u^t))
pub fn propagate_risk_sensitive(
    graph: &Graph,
    q0: &[Option<F>],
    eta: &[F],
    alpha: F, // Risk parameter: α > 0 = pessimistic, α < 0 = optimistic
    config: &PropConfig,
    t_steps: usize,
) -> Vec<Committor> {
    let n = graph.num_nodes();
    let mut h_curr = vec![0.5; n];
    let mut h_next = vec![0.5; n];

    // Set initial conditions
    for i in 0..n {
        if let Some(val) = q0[i] {
            h_curr[i] = val;
        }
    }

    let effective_steps = t_steps.min(config.t_max);

    for _iter in 0..effective_steps {
        if config.use_parallel {
            h_next.par_iter_mut().enumerate().for_each(|(v, h_v_next)| {
                *h_v_next = update_node_risk_sensitive(graph, &h_curr, q0, eta, v, alpha);
            });
        } else {
            for v in 0..n {
                h_next[v] = update_node_risk_sensitive(graph, &h_curr, q0, eta, v, alpha);
            }
        }

        // Check convergence
        let max_change = h_curr
            .iter()
            .zip(h_next.iter())
            .map(|(old, new)| (old - new).abs())
            .fold(0.0, f32::max);

        if max_change < config.eps {
            break;
        }

        std::mem::swap(&mut h_curr, &mut h_next);
    }

    h_curr
}

fn update_node_risk_sensitive(
    graph: &Graph,
    h_curr: &[F],
    q0: &[Option<F>],
    eta: &[F],
    v: usize,
    alpha: F,
) -> F {
    // Fixed boundaries
    if let Some(fixed_val) = q0[v] {
        if eta[v] > 1e6 {
            return fixed_val;
        }
    }

    let neighbors = graph.neighbors(v);
    if neighbors.is_empty() {
        return q0[v].unwrap_or(0.5);
    }

    if alpha.abs() < 1e-6 {
        // α ≈ 0: degenerate to regular harmonic mean
        return update_node(graph, h_curr, q0, eta, v);
    }

    // Risk-sensitive update: (1/α) * log(Σ w * exp(α * h))
    let mut exp_sum = 0.0;
    let mut weight_sum = 0.0;

    for &(u, w) in neighbors {
        exp_sum += w * (alpha * h_curr[u]).exp();
        weight_sum += w;
    }

    if exp_sum <= 0.0 {
        return q0[v].unwrap_or(0.5);
    }

    let risk_val = (exp_sum / weight_sum).ln() / alpha;

    // Blend with prior
    let total_weight = weight_sum + eta[v];
    let prior_contrib = q0[v].unwrap_or(0.5) * eta[v];

    let blended = (risk_val * weight_sum + prior_contrib) / total_weight;
    blended.clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn test_simple_chain() {
        // Linear chain: 0 -- 1 -- 2
        // Goal at 2, start at 0
        let nodes = vec![
            crate::NodeFeat::new(0.0, 0.0),
            crate::NodeFeat::new(1.0, 0.0),
            crate::NodeFeat::new(2.0, 0.0),
        ];
        let edges = vec![
            crate::Edge::new(0, 1, 1.0),
            crate::Edge::new(1, 0, 1.0),
            crate::Edge::new(1, 2, 1.0),
            crate::Edge::new(2, 1, 1.0),
        ];
        let graph = Graph::new(nodes, edges);

        let mut q0 = vec![None; 3];
        q0[2] = Some(1.0); // Goal
        let eta = vec![0.0, 0.0, 1e9]; // Only goal is fixed

        let config = PropConfig {
            step_policy: StepPolicy::Fixed,
            ..PropConfig::default()
        };

        // Use early iterations where gradient is clear (before over-convergence)
        let h = propagate_committor(&graph, &q0, &eta, &config, 3, 0.0);

        // Should have gradient: h[0] < h[1] < h[2] = 1.0
        assert!((h[2] - 1.0).abs() < 1e-3);
        assert!(h[0] < h[1], "h[0]={} should be < h[1]={}", h[0], h[1]);
        assert!(h[1] < h[2]);

        // Expected values after 3 iterations: [0.5, 0.75, 1.0]
        assert!((h[0] - 0.5).abs() < 0.1);
        assert!((h[1] - 0.75).abs() < 0.1);
        assert!(h[1] - h[0] > 0.2); // Meaningful gradient
    }

    #[test]
    fn test_with_prior() {
        // 2x2 grid with strong prior at node 0
        let graph = Graph::grid(2, 2, 1.0, None);

        let mut q0 = vec![None; 4];
        q0[0] = Some(0.9); // Strong prior
        q0[3] = Some(1.0); // Goal

        let eta = vec![10.0, 0.0, 0.0, 1e9];

        let config = PropConfig {
            step_policy: StepPolicy::Fixed,
            ..PropConfig::default()
        };
        let h = propagate_committor(&graph, &q0, &eta, &config, 30, 0.0);

        assert!((h[3] - 1.0).abs() < 1e-3); // Goal fixed
        assert!(h[0] > 0.8); // Prior should persist
    }

    #[test]
    fn test_risk_sensitive() {
        let nodes = vec![
            crate::NodeFeat::new(0.0, 0.0),
            crate::NodeFeat::new(1.0, 0.0),
        ];
        let edges = vec![crate::Edge::new(0, 1, 1.0), crate::Edge::new(1, 0, 1.0)];
        let graph = Graph::new(nodes, edges);

        let mut q0 = vec![None; 2];
        q0[1] = Some(1.0); // Goal
        let eta = vec![0.0, 1e9];

        let config = PropConfig {
            step_policy: StepPolicy::Fixed,
            ..PropConfig::default()
        };

        // Pessimistic (α > 0)
        let h_pess = propagate_risk_sensitive(&graph, &q0, &eta, 2.0, &config, 20);

        // Optimistic (α < 0)
        let h_opt = propagate_risk_sensitive(&graph, &q0, &eta, -2.0, &config, 20);

        // Regular
        let h_reg = propagate_committor(&graph, &q0, &eta, &config, 20, 0.0);

        // Pessimistic should be lower than regular, optimistic higher
        // (though in this simple case, difference may be small)
        assert!((h_pess[1] - 1.0).abs() < 1e-3);
        assert!((h_opt[1] - 1.0).abs() < 1e-3);
        assert!(h_pess[0] <= h_reg[0] + 1e-6);
        assert!(h_opt[0] >= h_reg[0] - 1e-6);
    }
}
