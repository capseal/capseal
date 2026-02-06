use crate::{Committor, Graph, NodeFeat, NodeId, F};
use std::cmp::Ordering;

/// Action selection from committor values
pub fn pick_next_node(graph: &Graph, q: &[Committor], current: NodeId) -> Option<NodeId> {
    if current >= graph.num_nodes() {
        return None;
    }

    let neighbors = graph.neighbors(current);
    if neighbors.is_empty() {
        return None;
    }

    // Find neighbor with highest committor value
    neighbors
        .iter()
        .max_by(|(u, _), (v, _)| match q[*u].partial_cmp(&q[*v]) {
            Some(Ordering::Equal) | None => u.cmp(v),
            Some(ordering) => ordering,
        })
        .map(|(node_id, _)| *node_id)
}

/// Improved action selection with exploration
pub fn pick_next_node_weighted(
    graph: &Graph,
    q: &[Committor],
    current: NodeId,
    temperature: F, // Higher = more exploration
    rng: &mut impl rand::Rng,
) -> Option<NodeId> {
    if current >= graph.num_nodes() {
        return None;
    }

    let neighbors = graph.neighbors(current);
    if neighbors.is_empty() {
        return None;
    }

    if temperature <= 1e-6 {
        // Greedy selection
        return pick_next_node(graph, q, current);
    }

    // Softmax over neighbor committor values
    let mut exp_vals = Vec::new();
    let mut sum_exp = 0.0;

    for &(neighbor, _) in neighbors {
        let exp_val = (q[neighbor] / temperature).exp();
        exp_vals.push(exp_val);
        sum_exp += exp_val;
    }

    // Sample from distribution
    let mut cumsum = 0.0;
    let sample = rng.gen::<f32>() * sum_exp;

    for (i, &exp_val) in exp_vals.iter().enumerate() {
        cumsum += exp_val;
        if sample <= cumsum {
            return Some(neighbors[i].0);
        }
    }

    // Fallback
    neighbors.last().map(|(node_id, _)| *node_id)
}

/// Environment-specific action decoders
pub trait ActionDecoder<Obs, Action> {
    fn to_action(&self, obs: &Obs, current_node: &NodeFeat, target_node: &NodeFeat) -> Action;
}

/// Waypoint-based action for spatial navigation
#[derive(Clone, Debug)]
pub struct WaypointDecoder {
    pub max_velocity: F,
    pub lookahead_steps: usize,
}

impl WaypointDecoder {
    pub fn new(max_velocity: F, lookahead_steps: usize) -> Self {
        Self {
            max_velocity,
            lookahead_steps,
        }
    }

    /// Convert target waypoint to velocity command
    pub fn waypoint_to_velocity(&self, current_pos: (F, F), target_pos: (F, F)) -> (F, F) {
        let dx = target_pos.0 - current_pos.0;
        let dy = target_pos.1 - current_pos.1;
        let dist = (dx * dx + dy * dy).sqrt();

        if dist < 1e-6 {
            return (0.0, 0.0);
        }

        let scale = (self.max_velocity / dist).min(1.0);
        (dx * scale, dy * scale)
    }

    /// Multi-step lookahead for smoother paths
    pub fn multi_step_target(
        &self,
        graph: &Graph,
        q: &[Committor],
        current: NodeId,
    ) -> Option<NodeId> {
        let mut node = current;

        // Follow gradient for multiple steps
        for _ in 0..self.lookahead_steps {
            if let Some(next) = pick_next_node(graph, q, node) {
                node = next;
            } else {
                break;
            }
        }

        if node != current {
            Some(node)
        } else {
            None
        }
    }
}

/// Humanoid maze actions
#[derive(Clone, Debug)]
pub struct HumanoidAction {
    pub forward_velocity: F,
    pub angular_velocity: F,
}

impl ActionDecoder<HumanoidObs, HumanoidAction> for WaypointDecoder {
    fn to_action(
        &self,
        obs: &HumanoidObs,
        _current: &NodeFeat,
        target: &NodeFeat,
    ) -> HumanoidAction {
        let current_pos = (obs.x, obs.y);
        let target_pos = (target.x, target.y);
        let current_angle = obs.angle;

        // Compute desired heading
        let dx = target_pos.0 - current_pos.0;
        let dy = target_pos.1 - current_pos.1;
        let target_angle = dy.atan2(dx);

        // Angular control
        let mut angle_diff = target_angle - current_angle;
        while angle_diff > std::f32::consts::PI {
            angle_diff -= 2.0 * std::f32::consts::PI;
        }
        while angle_diff < -std::f32::consts::PI {
            angle_diff += 2.0 * std::f32::consts::PI;
        }

        let angular_velocity = (angle_diff * 2.0).clamp(-self.max_velocity, self.max_velocity);

        // Forward velocity - reduce when turning
        let forward_scale = 1.0 - (angle_diff.abs() / std::f32::consts::PI) * 0.7;
        let forward_velocity = self.max_velocity * forward_scale.max(0.1);

        HumanoidAction {
            forward_velocity,
            angular_velocity,
        }
    }
}

/// Ant soccer actions  
#[derive(Clone, Debug)]
pub struct AntAction {
    pub torques: Vec<F>, // Joint torques
}

impl ActionDecoder<AntObs, AntAction> for WaypointDecoder {
    fn to_action(&self, obs: &AntObs, _current: &NodeFeat, target: &NodeFeat) -> AntAction {
        // Simple ball-chasing behavior
        // In practice, this would be a learned policy conditioned on target

        let ball_pos = (obs.ball_x, obs.ball_y);
        let ant_pos = (obs.ant_x, obs.ant_y);
        let target_pos = (target.x, target.y);

        // Compute desired ball displacement
        let desired_ball_dx = target_pos.0 - ball_pos.0;
        let desired_ball_dy = target_pos.1 - ball_pos.1;

        // Ant should position to push ball toward target
        let push_angle = desired_ball_dy.atan2(desired_ball_dx) + std::f32::consts::PI;
        let push_distance = 0.5; // Stay close to ball

        let desired_ant_x = ball_pos.0 + push_distance * push_angle.cos();
        let desired_ant_y = ball_pos.1 + push_distance * push_angle.sin();

        // Convert to joint torques (simplified PD controller)
        let ant_dx = desired_ant_x - ant_pos.0;
        let ant_dy = desired_ant_y - ant_pos.1;

        // Mock torque computation - in practice use learned controller
        let n_joints = 8;
        let mut torques = vec![0.0; n_joints];

        for i in 0..n_joints {
            let phase = i as F * 2.0 * std::f32::consts::PI / n_joints as F;
            torques[i] = 0.3 * (ant_dx * phase.cos() + ant_dy * phase.sin());
            torques[i] = torques[i].clamp(-1.0, 1.0);
        }

        AntAction { torques }
    }
}

/// Puzzle button press action
#[derive(Clone, Debug)]
pub struct PuzzleAction {
    pub button_id: usize, // Which button to press (0-19)
}

pub struct PuzzleDecoder;

impl ActionDecoder<PuzzleObs, PuzzleAction> for PuzzleDecoder {
    fn to_action(&self, _obs: &PuzzleObs, _current: &NodeFeat, _target: &NodeFeat) -> PuzzleAction {
        // For puzzle, the "action" is encoded in the edge between current and target
        // This is handled by pick_puzzle_button function
        PuzzleAction { button_id: 0 } // Placeholder
    }
}

/// Special case for puzzle: button selection
pub fn pick_puzzle_button(
    current_config: u32, // Current light configuration (bitmask)
    target_config: u32,  // Target light configuration
) -> Option<usize> {
    // Find which button press transforms current -> target
    for button in 0..20 {
        let mask = puzzle_button_mask(button);
        if current_config ^ mask == target_config {
            return Some(button);
        }
    }
    None
}

/// Get toggle mask for puzzle button
fn puzzle_button_mask(button: usize) -> u32 {
    // 4x5 grid, button affects itself + cross neighbors
    let row = button / 5;
    let col = button % 5;

    let mut mask = 0u32;

    // Toggle button itself
    mask |= 1 << button;

    // Toggle neighbors (cross pattern)
    let neighbors = [
        (row.wrapping_sub(1), col), // up
        (row + 1, col),             // down
        (row, col.wrapping_sub(1)), // left
        (row, col + 1),             // right
    ];

    for (r, c) in neighbors {
        if r < 4 && c < 5 {
            let neighbor_id = r * 5 + c;
            mask |= 1 << neighbor_id;
        }
    }

    mask
}

// Mock observation types for compilation
#[derive(Clone, Debug)]
pub struct HumanoidObs {
    pub x: F,
    pub y: F,
    pub angle: F,
}

#[derive(Clone, Debug)]
pub struct AntObs {
    pub ant_x: F,
    pub ant_y: F,
    pub ball_x: F,
    pub ball_y: F,
}

#[derive(Clone, Debug)]
pub struct PuzzleObs {
    pub config: u32, // 20-bit light configuration
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, Graph};
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn test_pick_next_node() {
        let nodes = vec![
            NodeFeat::new(0.0, 0.0), // 0
            NodeFeat::new(1.0, 0.0), // 1
            NodeFeat::new(2.0, 0.0), // 2
        ];
        let edges = vec![Edge::new(0, 1, 1.0), Edge::new(0, 2, 1.0)];
        let graph = Graph::new(nodes, edges);

        let q = vec![0.3, 0.7, 0.9]; // Node 2 has highest value

        let next = pick_next_node(&graph, &q, 0).unwrap();
        assert_eq!(next, 2); // Should pick node 2 (highest q)
    }

    #[test]
    fn test_weighted_selection() {
        let nodes = vec![
            NodeFeat::new(0.0, 0.0),
            NodeFeat::new(1.0, 0.0),
            NodeFeat::new(2.0, 0.0),
        ];
        let edges = vec![Edge::new(0, 1, 1.0), Edge::new(0, 2, 1.0)];
        let graph = Graph::new(nodes, edges);

        let q = vec![0.0, 0.2, 0.8];
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        // With high temperature, should sometimes pick suboptimal node 1
        let mut picks = [0; 3];
        for _ in 0..100 {
            if let Some(next) = pick_next_node_weighted(&graph, &q, 0, 2.0, &mut rng) {
                picks[next] += 1;
            }
        }

        assert!(picks[2] > picks[1]); // Node 2 should be picked more often
        assert!(picks[1] > 0); // But node 1 should sometimes be picked too
        assert_eq!(picks[0], 0); // Node 0 is not a neighbor
    }

    #[test]
    fn test_waypoint_decoder() {
        let decoder = WaypointDecoder::new(2.0, 3);

        // Test velocity computation
        let (vx, vy) = decoder.waypoint_to_velocity((0.0, 0.0), (3.0, 4.0));
        let speed = (vx * vx + vy * vy).sqrt();
        assert!((speed - 2.0).abs() < 1e-6); // Should be clamped to max_velocity

        // Direction should be correct
        assert!((vx / vy - 3.0 / 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_puzzle_button_mask() {
        // Button 0 (top-left) should toggle itself + right + down
        let mask = puzzle_button_mask(0);
        let expected = (1 << 0) | (1 << 1) | (1 << 5); // positions 0, 1, 5
        assert_eq!(mask, expected);

        // Button 12 (middle) should toggle cross pattern
        let mask = puzzle_button_mask(12);
        let expected = (1 << 12) | (1 << 7) | (1 << 17) | (1 << 11) | (1 << 13);
        assert_eq!(mask, expected);
    }

    #[test]
    fn test_pick_puzzle_button() {
        let current = 0b00000_00000_00000_00000; // All lights off

        // Calculate what button 12 actually produces (light 12 + its cross)
        let mask12 = puzzle_button_mask(12);
        let target = current ^ mask12; // This is what button 12 creates

        // Try to find the button that transforms current -> target
        let button = pick_puzzle_button(current, target);

        // Should find button 12
        assert!(button.is_some());
        assert_eq!(button.unwrap(), 12);

        // Verify the transformation works
        let mask = puzzle_button_mask(button.unwrap());
        assert_eq!(current ^ mask, target);
    }
}
