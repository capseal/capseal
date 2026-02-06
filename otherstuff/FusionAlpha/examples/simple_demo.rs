use fusion_core::*;
use fusion_core::actions::pick_next_node_weighted;
use fusion_core::propagation::propagate_risk_sensitive;
use fusion_envs::*;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn main() -> anyhow::Result<()> {
    println!("Fusion Alpha Demo: Committor Planning");
    
    // Create simple maze
    let maze = MazeConfig::simple_maze();
    println!("Created {}x{} maze", maze.width, maze.height);
    
    // Mock humanoid observation
    let obs = HumanoidObs {
        x: 1.0,
        y: 1.0,
        angle: 0.0,
        velocity: (0.0, 0.0),
    };
    
    // Goal in opposite corner
    let goal = GoalSpec::new(8.0, 8.0, 1.0);
    println!("Goal at ({:.1}, {:.1}) with radius {:.1}", goal.x, goal.y, goal.radius);
    
    // Build graph
    let (graph, current_node, goal_node) = build_graph_humanoid(&obs, &goal, &maze, None);
    println!("Built graph: {} nodes, {} edges", graph.num_nodes(), graph.num_edges());
    println!("Current node: {}, Goal node: {}", current_node, goal_node);
    
    // Mock ENN state (high uncertainty)
    let enn_state = FusionState::from_severity(0.6, 0.8, 0.7);
    println!(
        "ENN state: q_prior={:.2}, obs_reliability={:.2}, risk_aversion={:.2}",
        enn_state.q_prior_enn,
        enn_state.obs_reliability,
        enn_state.risk_aversion
    );
    
    // Build priors
    let priors = build_humanoid_priors(&graph, current_node, goal_node, Some(&enn_state), None);
    println!("Set priors for {} nodes", priors.len());
    
    // Propagation config
    let prop_config = PropConfig {
        t_max: 50,
        eps: 1e-4,
        use_parallel: true,
        alpha_max: 6.0,
        step_policy: StepPolicy::RiskScaled,
    };
    
    // Severity-scaled propagation steps
    let t_steps = enn_state.propagation_steps(&prop_config);
    println!(
        "Propagation steps: {} (risk-scaled from max {})",
        t_steps, prop_config.t_max
    );
    
    // Propagate committor values
    let q_values = propagate_committor(
        &graph,
        &priors.q0,
        &priors.eta,
        &prop_config,
        t_steps,
        enn_state.effective_risk(),
    );
    println!("Computed committor values");
    
    // Display results
    println!("\nCommittor values:");
    println!("Current node {}: q = {:.3}", current_node, q_values[current_node]);
    println!("Goal node {}: q = {:.3}", goal_node, q_values[goal_node]);
    
    // Show neighbors of current node
    let neighbors = graph.neighbors(current_node);
    println!("\nNeighbors of current node:");
    for &(neighbor_id, weight) in neighbors {
        println!("  Node {} (weight {:.2}): q = {:.3}", 
                 neighbor_id, weight, q_values[neighbor_id]);
    }
    
    // Pick next action
    if let Some(next_node) = pick_next_node(&graph, &q_values, current_node) {
        println!("\nSelected next node: {} (q = {:.3})", next_node, q_values[next_node]);
        
        let current_pos = &graph.nodes[current_node];
        let target_pos = &graph.nodes[next_node];
        println!("Move from ({:.1}, {:.1}) to ({:.1}, {:.1})", 
                 current_pos.x, current_pos.y, target_pos.x, target_pos.y);
    } else {
        println!("No valid next node found!");
    }
    
    // Test with exploration
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    println!("\nExploration test (10 samples with temperature 0.5):");
    for i in 0..10 {
        if let Some(next) = pick_next_node_weighted(&graph, &q_values, current_node, 0.5, &mut rng) {
            println!("  Sample {}: node {} (q = {:.3})", i+1, next, q_values[next]);
        }
    }
    
    // Test risk-sensitive propagation
    println!("\nRisk-sensitive comparison:");
    let alpha_pessimistic = 2.0;
    let alpha_optimistic = -2.0;
    
    let q_pess = propagate_risk_sensitive(&graph, &priors.q0, &priors.eta, alpha_pessimistic, &prop_config, t_steps);
    let q_opt = propagate_risk_sensitive(&graph, &priors.q0, &priors.eta, alpha_optimistic, &prop_config, t_steps);
    
    println!("Current node committor values:");
    println!("  Regular: {:.3}", q_values[current_node]);
    println!("  Pessimistic (α={}): {:.3}", alpha_pessimistic, q_pess[current_node]);
    println!("  Optimistic (α={}): {:.3}", alpha_optimistic, q_opt[current_node]);
    
    Ok(())
}
