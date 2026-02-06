use fusion_core::{Graph, NodeFeat, Edge, Priors, PriorSource, F};

const ETA_MIN: F = 0.1;
const ETA_MAX: F = 10.0;
use fusion_core::priors::PriorBuilder;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Puzzle 4x5 configuration (20 lights, 20 buttons)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PuzzleConfig {
    pub width: usize,   // 5 
    pub height: usize,  // 4
    pub initial_state: u32, // Initial light configuration (20-bit)
    pub goal_state: u32,    // Target light configuration
}

/// Puzzle observation
#[derive(Clone, Debug)]
pub struct PuzzleObs {
    pub config: u32,    // Current 20-bit light configuration
    pub moves_left: usize, // Remaining moves (for time pressure)
}

/// Build graph for puzzle (local subgraph around current state)
pub fn build_graph_puzzle(
    current_config: u32,
    goal_config: u32,
    depth: usize,
    bicep_paths: Option<&PuzzleBICEPPaths>,
) -> (Graph, usize, usize) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut config_to_node: HashMap<u32, usize> = HashMap::new();
    let mut visited: HashSet<u32> = HashSet::new();
    
    // BFS to build local subgraph with capacity limits
    let mut queue = VecDeque::new();
    queue.push_back((current_config, 0)); // (config, depth)
    visited.insert(current_config);
    
    // Add current node
    let current_node_id = 0;
    nodes.push(config_to_node_feat(current_config));
    config_to_node.insert(current_config, current_node_id);
    
    const MAX_NODES: usize = 2000; // Cap graph size for performance
    
    while let Some((config, current_depth)) = queue.pop_front() {
        if current_depth >= depth || nodes.len() >= MAX_NODES {
            continue;
        }
        
        let from_id = config_to_node[&config];
        
        // Try all 20 button presses
        for button in 0..20 {
            let mask = puzzle_button_mask(button);
            let next_config = config ^ mask;
            
            // Skip if already processed - deduplication by bitboard config
            if visited.contains(&next_config) {
                // Still add edge to existing node
                if let Some(&existing_id) = config_to_node.get(&next_config) {
                    edges.push(Edge::new(from_id, existing_id, button_to_weight(button)));
                }
                continue;
            }
            
            // Create new node if under capacity
            if nodes.len() < MAX_NODES {
                let new_id = nodes.len();
                nodes.push(config_to_node_feat(next_config));
                config_to_node.insert(next_config, new_id);
                
                // Add edge
                edges.push(Edge::new(from_id, new_id, button_to_weight(button)));
                
                // Add to frontier if within depth
                if current_depth + 1 < depth {
                    visited.insert(next_config);
                    queue.push_back((next_config, current_depth + 1));
                }
            }
        }
    }
    
    // Add goal node if not already present
    let goal_node_id = if let Some(&existing_id) = config_to_node.get(&goal_config) {
        existing_id
    } else {
        let goal_id = nodes.len();
        nodes.push(config_to_node_feat(goal_config));
        config_to_node.insert(goal_config, goal_id);
        
        // Connect goal to its predecessors (configs that can reach goal in 1 move)
        for button in 0..20 {
            let mask = puzzle_button_mask(button);
            let prev_config = goal_config ^ mask; // Reverse the button press
            
            if let Some(&prev_id) = config_to_node.get(&prev_config) {
                edges.push(Edge::new(prev_id, goal_id, button_to_weight(button)));
            }
        }
        
        goal_id
    };
    
    let mut graph = Graph::new(nodes, edges);
    
    // Add BICEP-discovered move patterns
    if let Some(paths) = bicep_paths {
        add_puzzle_bicep_edges(&mut graph, paths, &config_to_node);
    }
    
    (graph, current_node_id, goal_node_id)
}

/// Convert configuration to node features
fn config_to_node_feat(config: u32) -> NodeFeat {
    // Use Hamming distance to goal as spatial coordinates
    let x = (config.count_ones() as f32) / 20.0; // Density of lights
    let y = hamming_distance_normalized(config, 0); // Distance from all-off
    
    // Pack full configuration into extra features
    let mut extra = Vec::new();
    for i in 0..20 {
        extra.push(if (config >> i) & 1 == 1 { 1.0 } else { 0.0 });
    }
    
    NodeFeat::with_extra(x, y, extra)
}

/// Normalized Hamming distance between two configurations
fn hamming_distance_normalized(a: u32, b: u32) -> f32 {
    (a ^ b).count_ones() as f32 / 20.0
}

/// Convert button ID to edge weight (uniform for now)
fn button_to_weight(_button: usize) -> F {
    1.0 // All button presses equally likely
}

/// Get toggle mask for puzzle button (cross pattern)
pub fn puzzle_button_mask(button: usize) -> u32 {
    let row = button / 5;
    let col = button % 5;
    
    let mut mask = 0u32;
    
    // Toggle button itself
    mask |= 1 << button;
    
    // Toggle cross neighbors
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

/// Find which button press transforms config A to config B
pub fn find_button_press(from_config: u32, to_config: u32) -> Option<usize> {
    let diff = from_config ^ to_config;
    
    for button in 0..20 {
        if puzzle_button_mask(button) == diff {
            return Some(button);
        }
    }
    
    None
}

/// Build priors for puzzle
pub fn build_puzzle_priors(
    graph: &Graph,
    current_node: usize,
    goal_node: usize,
    goal_config: u32,
    enn_state: Option<&fusion_core::FusionState>,
) -> Priors {
    let mut builder = PriorBuilder::new(graph.num_nodes())
        .with_goals(&[goal_node]);
    
    // ENN prior for current state
    if let Some(state) = enn_state {
        let enn_source = PriorSource::from_enn_reliability(
            state.q_prior_enn,
            state.obs_reliability,
            ETA_MIN,
            ETA_MAX,
        );
        builder = builder.add_prior(current_node, enn_source);
    }
    
    // Heuristic priors based on Hamming distance to goal
    for (node_id, node) in graph.nodes.iter().enumerate() {
        if node_id != current_node && node_id != goal_node {
            let config = node_feat_to_config(node);
            let hamming_dist = hamming_distance_normalized(config, goal_config);
            
            // Closer to goal = higher prior (but low confidence)
            let heuristic_prior = 1.0 - hamming_dist;
            if heuristic_prior > 0.5 {
                let heuristic_source = PriorSource::Manual {
                    value: heuristic_prior,
                    confidence: 0.1_f32.clamp(0.05, 0.95), // Low confidence heuristic
                };
                builder = builder.add_prior(node_id, heuristic_source);
            }
        }
    }
    
    builder.build()
}

/// Extract configuration from node features
fn node_feat_to_config(node: &NodeFeat) -> u32 {
    let mut config = 0u32;
    
    // Reconstruct from extra features
    for (i, &bit) in node.extra.iter().enumerate() {
        if i < 20 && bit > 0.5 {
            config |= 1 << i;
        }
    }
    
    config
}

/// Add edges from BICEP puzzle-solving paths
fn add_puzzle_bicep_edges(
    graph: &mut Graph,
    paths: &PuzzleBICEPPaths,
    config_to_node: &HashMap<u32, usize>,
) {
    let mut move_counts: HashMap<(usize, usize), usize> = HashMap::new();
    
    // Count move transitions
    for path in &paths.paths {
        for window in path.moves.windows(2) {
            let from_config = window[0].config;
            let to_config = window[1].config;
            
            if let (Some(&from_id), Some(&to_id)) = (
                config_to_node.get(&from_config),
                config_to_node.get(&to_config),
            ) {
                *move_counts.entry((from_id, to_id)).or_insert(0) += 1;
            }
        }
    }
    
    // Add frequently used moves with higher weight
    for ((from, to), count) in move_counts {
        if count >= 2 {
            let boosted_weight = 1.0 + (count as F * 0.2).min(1.0);
            
            // Find existing edge and boost its weight
            let existing_edge = graph.edges.iter_mut()
                .find(|e| e.u == from && e.v == to);
                
            if let Some(edge) = existing_edge {
                edge.w = edge.w.max(boosted_weight);
            }
        }
    }
}

/// Solve puzzle with A* for comparison/validation
pub fn solve_puzzle_astar(
    initial: u32,
    goal: u32,
    max_moves: usize,
) -> Option<Vec<usize>> {
    let mut open_set = std::collections::BinaryHeap::new();
    let mut came_from: HashMap<u32, (u32, usize)> = HashMap::new();
    let mut g_score: HashMap<u32, usize> = HashMap::new();
    
    g_score.insert(initial, 0);
    open_set.push(AStarNode {
        config: initial,
        f_score: hamming_distance(initial, goal),
        g_score: 0,
    });
    
    while let Some(current) = open_set.pop() {
        if current.config == goal {
            return Some(reconstruct_path(&came_from, goal));
        }
        
        if current.g_score >= max_moves {
            continue;
        }
        
        for button in 0..20 {
            let mask = puzzle_button_mask(button);
            let neighbor = current.config ^ mask;
            let tentative_g = current.g_score + 1;
            
            if tentative_g < *g_score.get(&neighbor).unwrap_or(&usize::MAX) {
                came_from.insert(neighbor, (current.config, button));
                g_score.insert(neighbor, tentative_g);
                
                let f_score = tentative_g + hamming_distance(neighbor, goal);
                open_set.push(AStarNode {
                    config: neighbor,
                    f_score,
                    g_score: tentative_g,
                });
            }
        }
    }
    
    None
}

/// Hamming distance between configurations
fn hamming_distance(a: u32, b: u32) -> usize {
    (a ^ b).count_ones() as usize
}

/// Reconstruct A* path
fn reconstruct_path(came_from: &HashMap<u32, (u32, usize)>, mut current: u32) -> Vec<usize> {
    let mut path = Vec::new();
    
    while let Some(&(prev_config, button)) = came_from.get(&current) {
        path.push(button);
        current = prev_config;
    }
    
    path.reverse();
    path
}

/// A* node for priority queue
#[derive(Eq, PartialEq)]
struct AStarNode {
    config: u32,
    f_score: usize,
    g_score: usize,
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse for min-heap
        other.f_score.cmp(&self.f_score)
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// BICEP paths for puzzle solving
#[derive(Clone, Debug)]
pub struct PuzzleBICEPPaths {
    pub paths: Vec<PuzzleBICEPPath>,
}

#[derive(Clone, Debug)]
pub struct PuzzleBICEPPath {
    pub moves: Vec<PuzzleMove>,
    pub solved: bool,
    pub final_distance: usize, // Hamming distance to goal at end
}

#[derive(Clone, Debug)]
pub struct PuzzleMove {
    pub config: u32,
    pub button: Option<usize>,
    pub t: F,
}

impl PuzzleBICEPPaths {
    pub fn success_rate(&self) -> F {
        if self.paths.is_empty() {
            return 0.1;
        }
        
        let solved = self.paths.iter().filter(|p| p.solved).count();
        solved as F / self.paths.len() as F
    }
    
    pub fn average_final_distance(&self) -> F {
        if self.paths.is_empty() {
            return 10.0;
        }
        
        let total_dist: usize = self.paths.iter()
            .map(|p| p.final_distance)
            .sum();
        
        total_dist as F / self.paths.len() as F
    }
}

/// Default puzzle configurations
impl PuzzleConfig {
    /// Simple puzzle (few lights on)
    pub fn simple() -> Self {
        Self {
            width: 5,
            height: 4,
            initial_state: 0b00000_00000_00000_00000, // All off
            goal_state:    0b11111_11111_11111_11111, // All on
        }
    }
    
    /// Cross pattern puzzle
    pub fn cross_pattern() -> Self {
        Self {
            width: 5,
            height: 4,
            initial_state: 0b00000_00000_00000_00000, // All off
            goal_state:    0b00100_01110_01110_00100, // Cross shape
        }
    }
    
    /// Random intermediate puzzle
    pub fn intermediate() -> Self {
        Self {
            width: 5,
            height: 4,
            initial_state: 0b10101_01010_10101_01010, // Checkerboard
            goal_state:    0b01010_10101_01010_10101, // Inverted checkerboard
        }
    }
    
    /// Hard random configuration
    pub fn hard_random(seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;
        
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        
        Self {
            width: 5,
            height: 4,
            initial_state: rng.gen_range(0..=0xFFFFF), // Random 20-bit
            goal_state: rng.gen_range(0..=0xFFFFF),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_button_mask() {
        // Button 0 (top-left): toggles self + right + down
        let mask = puzzle_button_mask(0);
        let expected = (1 << 0) | (1 << 1) | (1 << 5);
        assert_eq!(mask, expected);
        
        // Button 12 (center): toggles self + all 4 neighbors
        let mask = puzzle_button_mask(12);
        let expected = (1 << 12) | (1 << 7) | (1 << 17) | (1 << 11) | (1 << 13);
        assert_eq!(mask, expected);
        
        // Button 4 (top-right): toggles self + left + down (no right/up neighbors)
        let mask = puzzle_button_mask(4);
        let expected = (1 << 4) | (1 << 3) | (1 << 9);
        assert_eq!(mask, expected);
    }
    
    #[test]
    fn test_find_button_press() {
        let from = 0b00000_00000_00000_00000;
        let to = 0b00000_00000_00100_00110; // Button 12 pattern
        
        let button = find_button_press(from, to);
        assert_eq!(button, Some(12));
        
        // Invalid transition
        let invalid_to = 0b10000_00000_00000_00001;
        assert_eq!(find_button_press(from, invalid_to), None);
    }
    
    #[test]
    fn test_hamming_distance() {
        let a = 0b11110000_11110000_11110000;
        let b = 0b11111111_11111111_11111111;
        
        assert_eq!(hamming_distance(a, b), 12);
        assert!((hamming_distance_normalized(a, b) - 0.6).abs() < 1e-6);
    }
    
    #[test]
    fn test_build_puzzle_graph() {
        let config = PuzzleConfig::simple();
        let (graph, current, goal) = build_graph_puzzle(
            config.initial_state,
            config.goal_state,
            3,
            None,
        );
        
        assert!(graph.num_nodes() > 1);
        assert!(graph.num_edges() > 0);
        
        // Current should be different from goal (unless trivial case)
        if config.initial_state != config.goal_state {
            assert_ne!(current, goal);
        }
        
        // All nodes should have button press edges (degree >= 1 unless isolated)
        let has_neighbors = graph.neighbors(current).len() > 0;
        assert!(has_neighbors);
    }
    
    #[test]
    fn test_astar_solver() {
        // Simple case: all off -> all on
        let solution = solve_puzzle_astar(0, 0xFFFFF, 30);
        
        // There should be a solution (may not be optimal)
        assert!(solution.is_some());
        
        if let Some(moves) = solution {
            // Verify solution by applying moves
            let mut config = 0u32;
            for button in moves {
                let mask = puzzle_button_mask(button);
                config ^= mask;
            }
            assert_eq!(config, 0xFFFFF);
        }
    }
    
    #[test]
    fn test_config_conversion() {
        let original_config = 0b10101_01010_11001_00110;
        let node = config_to_node_feat(original_config);
        let recovered_config = node_feat_to_config(&node);
        
        assert_eq!(original_config, recovered_config);
    }
}
