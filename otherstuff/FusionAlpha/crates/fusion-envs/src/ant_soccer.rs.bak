use fusion_core::{Graph, NodeFeat, Edge, Priors, PriorSource, F};
use fusion_core::priors::PriorBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Ant soccer field configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldConfig {
    pub width: F,           // Field width (meters)
    pub height: F,          // Field height (meters) 
    pub cell_size: F,       // Ball position discretization
    pub goal_width: F,      // Goal width
    pub obstacles: Vec<Obstacle>, // Field obstacles
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Obstacle {
    pub x: F,
    pub y: F,
    pub radius: F,
}

/// Ant soccer observation
#[derive(Clone, Debug)]
pub struct AntObs {
    pub ant_x: F,
    pub ant_y: F,
    pub ant_angle: F,
    pub ball_x: F,
    pub ball_y: F,
    pub ball_velocity: (F, F),
}

/// Soccer goal specification
#[derive(Clone, Debug)]
pub struct SoccerGoal {
    pub center_x: F,
    pub center_y: F,
    pub width: F,
    pub direction: F, // Goal direction (0 = right, π = left)
}

impl SoccerGoal {
    pub fn new(center_x: F, center_y: F, width: F, direction: F) -> Self {
        Self { center_x, center_y, width, direction }
    }
    
    /// Check if ball position scores
    pub fn scores(&self, ball_x: F, ball_y: F) -> bool {
        let dy = (ball_y - self.center_y).abs();
        
        if dy > self.width / 2.0 {
            return false;
        }
        
        // Check if ball crossed goal line
        match self.direction {
            d if d.abs() < 0.1 => ball_x > self.center_x,  // Right goal
            d if (d - std::f32::consts::PI).abs() < 0.1 => ball_x < self.center_x, // Left goal
            _ => false,
        }
    }
}

/// Build graph for ant soccer (ball-centric nodes)
pub fn build_graph_ant_soccer(
    obs: &AntObs,
    goal: &SoccerGoal,
    field: &FieldConfig,
    bicep_paths: Option<&SoccerBICEPPaths>,
) -> (Graph, usize, usize) {
    let grid_width = (field.width / field.cell_size) as usize;
    let grid_height = (field.height / field.cell_size) as usize;
    
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    
    // Create ball position nodes
    let mut node_map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut node_id = 0;
    
    for i in 0..grid_height {
        for j in 0..grid_width {
            let x = j as F * field.cell_size + field.cell_size * 0.5;
            let y = i as F * field.cell_size + field.cell_size * 0.5;
            
            // Skip positions inside obstacles
            if is_inside_obstacle(x, y, &field.obstacles) {
                continue;
            }
            
            // Add control state information
            let ant_dist = ((x - obs.ant_x).powi(2) + (y - obs.ant_y).powi(2)).sqrt();
            let in_control = ant_dist < 1.5; // Ant can influence ball
            
            let mut extra = vec![if in_control { 1.0 } else { 0.0 }];
            extra.push(ant_dist); // Distance to ant
            
            nodes.push(NodeFeat::with_extra(x, y, extra));
            node_map.insert((i, j), node_id);
            node_id += 1;
        }
    }
    
    // Create edges (ball movement transitions)
    for ((i, j), &from_id) in &node_map {
        let from_x = nodes[from_id].x;
        let from_y = nodes[from_id].y;
        
        // Local ball movements (adjacent cells)
        let neighbors = [
            (*i as i32 - 1, *j as i32),     // up
            (*i as i32 + 1, *j as i32),     // down  
            (*i as i32, *j as i32 - 1),     // left
            (*i as i32, *j as i32 + 1),     // right
            (*i as i32 - 1, *j as i32 - 1), // diagonals
            (*i as i32 - 1, *j as i32 + 1),
            (*i as i32 + 1, *j as i32 - 1),
            (*i as i32 + 1, *j as i32 + 1),
        ];
        
        for (ni, nj) in neighbors {
            if ni >= 0 && nj >= 0 && (ni as usize) < grid_height && (nj as usize) < grid_width {
                if let Some(&to_id) = node_map.get(&(ni as usize, nj as usize)) {
                    let to_x = nodes[to_id].x;
                    let to_y = nodes[to_id].y;
                    
                    // Weight based on ant positioning cost
                    let move_dist = ((to_x - from_x).powi(2) + (to_y - from_y).powi(2)).sqrt();
                    let ant_repositioning_cost = compute_ant_cost(obs, from_x, from_y, to_x, to_y);
                    
                    let base_weight = 1.0 / (1.0 + move_dist);
                    let weight = base_weight / (1.0 + ant_repositioning_cost);
                    
                    edges.push(Edge::new(from_id, to_id, weight));
                }
            }
        }
        
        // Long-range ball shots (if ant is in control)
        let ant_dist = nodes[from_id].extra[1];
        if ant_dist < 1.0 {
            add_shot_edges(&mut edges, from_id, &nodes, from_x, from_y, field, goal);
        }
    }
    
    let mut graph = Graph::new(nodes, edges);
    
    // Find current ball node
    let current_ball_node = find_nearest_ball_node(&graph, obs.ball_x, obs.ball_y);
    
    // Add goal node
    let goal_node = graph.add_goal(goal.center_x, goal.center_y);
    
    // Connect goal to scoring positions
    add_goal_connections(&mut graph, goal_node, goal, field);
    
    // Add BICEP-discovered transitions
    if let Some(paths) = bicep_paths {
        add_soccer_bicep_edges(&mut graph, paths, field);
    }
    
    (graph, current_ball_node, goal_node)
}

/// Check if position is inside any obstacle
fn is_inside_obstacle(x: F, y: F, obstacles: &[Obstacle]) -> bool {
    obstacles.iter().any(|obs| {
        let dx = x - obs.x;
        let dy = y - obs.y;
        dx * dx + dy * dy <= obs.radius * obs.radius
    })
}

/// Compute cost for ant to reposition for ball movement
fn compute_ant_cost(obs: &AntObs, from_x: F, from_y: F, to_x: F, to_y: F) -> F {
    // Desired ant position to push ball from 'from' to 'to'
    let push_angle = (to_y - from_y).atan2(to_x - from_x) + std::f32::consts::PI;
    let desired_ant_x = from_x + 0.8 * push_angle.cos();
    let desired_ant_y = from_y + 0.8 * push_angle.sin();
    
    // Distance ant needs to move
    let ant_move_dist = ((desired_ant_x - obs.ant_x).powi(2) + (desired_ant_y - obs.ant_y).powi(2)).sqrt();
    
    // Angular cost (ant reorientation)
    let current_ant_angle = obs.ant_angle;
    let desired_ant_angle = (to_y - from_y).atan2(to_x - from_x);
    let mut angle_diff = (desired_ant_angle - current_ant_angle).abs();
    if angle_diff > std::f32::consts::PI {
        angle_diff = 2.0 * std::f32::consts::PI - angle_diff;
    }
    
    ant_move_dist * 0.5 + angle_diff * 0.3
}

/// Add long-range shot edges when ant is in control
fn add_shot_edges(
    edges: &mut Vec<Edge>,
    from_id: usize,
    nodes: &[NodeFeat],
    from_x: F,
    from_y: F,
    field: &FieldConfig,
    goal: &SoccerGoal,
) {
    let max_shot_dist = 4.0;
    let shot_angle_spread = std::f32::consts::PI / 3.0; // 60 degrees
    
    // Direction towards goal
    let goal_angle = (goal.center_y - from_y).atan2(goal.center_x - from_x);
    
    // Try shots in goal direction ± spread
    let num_shots = 5;
    for i in 0..num_shots {
        let angle_offset = (i as F - 2.0) * shot_angle_spread / 4.0;
        let shot_angle = goal_angle + angle_offset;
        
        for dist in [1.0, 2.0, 3.0, 4.0] {
            let target_x = from_x + dist * shot_angle.cos();
            let target_y = from_y + dist * shot_angle.sin();
            
            // Check bounds
            if target_x < 0.0 || target_x >= field.width || target_y < 0.0 || target_y >= field.height {
                continue;
            }
            
            // Find nearest node to target
            if let Some(target_id) = find_nearest_node(nodes, target_x, target_y) {
                if target_id != from_id {
                    // Weight decreases with distance and angle deviation
                    let angle_penalty = (angle_offset.abs() / shot_angle_spread).min(1.0);
                    let dist_penalty = (dist / max_shot_dist).min(1.0);
                    let shot_weight = 0.8 * (1.0 - angle_penalty) * (1.0 - dist_penalty);
                    
                    if shot_weight > 0.1 {
                        edges.push(Edge::new(from_id, target_id, shot_weight));
                    }
                }
            }
        }
    }
}

/// Find nearest node to ball position
fn find_nearest_ball_node(graph: &Graph, ball_x: F, ball_y: F) -> usize {
    graph.nearest_node(ball_x, ball_y).unwrap_or(0)
}

/// Find nearest node to coordinates
fn find_nearest_node(nodes: &[NodeFeat], x: F, y: F) -> Option<usize> {
    let target = NodeFeat::new(x, y);
    nodes.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            target.distance_to(a).partial_cmp(&target.distance_to(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
}

/// Connect goal node to scoring positions
fn add_goal_connections(graph: &mut Graph, goal_node: usize, goal: &SoccerGoal, _field: &FieldConfig) {
    let goal_approach_dist = 1.5; // Distance from goal line for high-value connections
    
    // Connect nodes near the goal line
    for node_id in 0..graph.nodes.len() - 1 { // Exclude the goal node itself
        let node = &graph.nodes[node_id];
        
        // Check if node is in scoring position
        if goal.scores(node.x, node.y) {
            // Direct scoring connection (very high weight)
            graph.edges.push(Edge::new(node_id, goal_node, 100.0));
            continue;
        }
        
        // Check if node is near goal and aligned
        let dist_to_goal = ((node.x - goal.center_x).powi(2) + (node.y - goal.center_y).powi(2)).sqrt();
        let y_alignment = (node.y - goal.center_y).abs();
        
        if dist_to_goal < goal_approach_dist && y_alignment < goal.width * 0.6 {
            let weight = 10.0 * (1.0 - dist_to_goal / goal_approach_dist);
            graph.edges.push(Edge::new(node_id, goal_node, weight));
        }
    }
}

/// Add edges from BICEP soccer simulations
fn add_soccer_bicep_edges(graph: &mut Graph, paths: &SoccerBICEPPaths, field: &FieldConfig) {
    let mut transition_counts: HashMap<(usize, usize), usize> = HashMap::new();
    
    // Count ball transitions in BICEP paths
    for path in &paths.paths {
        for window in path.ball_states.windows(2) {
            let from_node = ball_to_node_id(window[0].x, window[0].y, field);
            let to_node = ball_to_node_id(window[1].x, window[1].y, field);
            
            if let (Some(from), Some(to)) = (from_node, to_node) {
                if from != to {
                    *transition_counts.entry((from, to)).or_insert(0) += 1;
                }
            }
        }
    }
    
    // Add frequently observed transitions
    let min_observations = 2;
    for ((from, to), count) in transition_counts {
        if count >= min_observations && from < graph.num_nodes() && to < graph.num_nodes() {
            let weight = (count as F * 0.1).min(1.0);
            graph.edges.push(Edge::new(from, to, weight));
        }
    }
}

/// Convert ball position to node ID
fn ball_to_node_id(ball_x: F, ball_y: F, field: &FieldConfig) -> Option<usize> {
    let grid_x = (ball_x / field.cell_size) as usize;
    let grid_y = (ball_y / field.cell_size) as usize;
    let grid_width = (field.width / field.cell_size) as usize;
    let grid_height = (field.height / field.cell_size) as usize;
    
    if grid_x < grid_width && grid_y < grid_height {
        Some(grid_y * grid_width + grid_x)
    } else {
        None
    }
}

/// Build priors for ant soccer
pub fn build_soccer_priors(
    graph: &Graph,
    current_node: usize,
    goal_node: usize,
    enn_state: Option<&fusion_core::FusionState>,
    bicep_success_rate: Option<F>,
) -> Priors {
    let mut builder = PriorBuilder::new(graph.num_nodes())
        .with_goals(&[goal_node]);
    
    // ENN prior for current ball state
    if let Some(state) = enn_state {
        let enn_source = PriorSource::from_enn(
            state.q_prior_enn,
            state.severity,
            state.bicep_confidence,
        );
        builder = builder.add_prior(current_node, enn_source);
    }
    
    // BICEP success rate
    if let Some(success_rate) = bicep_success_rate {
        let bicep_source = PriorSource::Manual {
            value: success_rate,
            confidence: 0.7_f32.clamp(0.05, 0.95),
        };
        builder = builder.add_prior(current_node, bicep_source);
    }
    
    // Add distance-based priors (closer to goal = higher prior)
    for (node_id, node) in graph.nodes.iter().enumerate() {
        if node_id != current_node && node_id != goal_node {
            let goal_node_pos = &graph.nodes[goal_node];
            let dist_to_goal = node.distance_to(goal_node_pos);
            
            if dist_to_goal < 3.0 {
                let distance_prior = 1.0 - (dist_to_goal / 3.0);
                let spatial_source = PriorSource::Manual {
                    value: distance_prior,
                    confidence: 0.2_f32.clamp(0.05, 0.95),
                };
                builder = builder.add_prior(node_id, spatial_source);
            }
        }
    }
    
    builder.build()
}

/// BICEP paths for soccer (mock)
#[derive(Clone, Debug)]
pub struct SoccerBICEPPaths {
    pub paths: Vec<SoccerBICEPPath>,
}

#[derive(Clone, Debug)]
pub struct SoccerBICEPPath {
    pub ball_states: Vec<BallState>,
    pub ant_states: Vec<AntState>,
    pub scored: bool,
}

#[derive(Clone, Debug)]
pub struct BallState {
    pub x: F,
    pub y: F,
    pub vx: F,
    pub vy: F,
    pub t: F,
}

#[derive(Clone, Debug)]
pub struct AntState {
    pub x: F,
    pub y: F,
    pub angle: F,
    pub t: F,
}

impl SoccerBICEPPaths {
    pub fn success_rate(&self) -> F {
        if self.paths.is_empty() {
            return 0.1;
        }
        
        let goals = self.paths.iter().filter(|p| p.scored).count();
        goals as F / self.paths.len() as F
    }
}

/// Default field configurations
impl FieldConfig {
    /// Standard soccer field
    pub fn standard() -> Self {
        Self {
            width: 12.0,
            height: 8.0,
            cell_size: 0.4,
            goal_width: 2.0,
            obstacles: Vec::new(),
        }
    }
    
    /// Field with obstacles
    pub fn with_obstacles() -> Self {
        let mut field = Self::standard();
        field.obstacles = vec![
            Obstacle { x: 6.0, y: 4.0, radius: 0.8 }, // Center obstacle
            Obstacle { x: 3.0, y: 2.0, radius: 0.5 }, // Side obstacle
            Obstacle { x: 9.0, y: 6.0, radius: 0.5 }, // Side obstacle
        ];
        field
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_soccer_goal() {
        let goal = SoccerGoal::new(12.0, 4.0, 2.0, 0.0); // Right goal
        
        assert!(goal.scores(12.1, 4.0)); // Ball past goal line, centered
        assert!(goal.scores(12.1, 3.2)); // Ball past goal line, within width
        assert!(!goal.scores(12.1, 2.0)); // Ball past goal line, outside width
        assert!(!goal.scores(11.9, 4.0)); // Ball before goal line
    }
    
    #[test]
    fn test_obstacle_detection() {
        let obstacles = vec![
            Obstacle { x: 5.0, y: 5.0, radius: 1.0 },
        ];
        
        assert!(is_inside_obstacle(5.0, 5.0, &obstacles)); // Center
        assert!(is_inside_obstacle(5.8, 5.0, &obstacles)); // Edge
        assert!(!is_inside_obstacle(6.2, 5.0, &obstacles)); // Outside
    }
    
    #[test]
    fn test_ant_cost_computation() {
        let obs = AntObs {
            ant_x: 3.0,
            ant_y: 3.0,
            ant_angle: 0.0,
            ball_x: 5.0,
            ball_y: 3.0,
            ball_velocity: (0.0, 0.0),
        };
        
        // Moving ball right should have low cost (ant is already behind ball)
        let cost_right = compute_ant_cost(&obs, 5.0, 3.0, 6.0, 3.0);
        
        // Moving ball left should have higher cost (ant needs to reposition)
        let cost_left = compute_ant_cost(&obs, 5.0, 3.0, 4.0, 3.0);
        
        assert!(cost_right < cost_left);
    }
    
    #[test]
    fn test_build_soccer_graph() {
        let field = FieldConfig::standard();
        let obs = AntObs {
            ant_x: 6.0,
            ant_y: 4.0,
            ant_angle: 0.0,
            ball_x: 6.0,
            ball_y: 4.0,
            ball_velocity: (0.0, 0.0),
        };
        let goal = SoccerGoal::new(12.0, 4.0, 2.0, 0.0);
        
        let (graph, current, goal_node) = build_graph_ant_soccer(&obs, &goal, &field, None);
        
        assert!(graph.num_nodes() > 0);
        assert!(graph.num_edges() > 0);
        
        // Goal should have high committor value after propagation
        let priors = build_soccer_priors(&graph, current, goal_node, None, None);
        assert_eq!(priors.q0[goal_node], Some(1.0));
    }
}