use fusion_core::{Graph, Edge, Priors, PriorSource, F};

const ETA_MIN: F = 0.1;
const ETA_MAX: F = 10.0;
use fusion_core::priors::PriorBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Humanoid maze environment configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MazeConfig {
    pub width: usize,
    pub height: usize,
    pub cell_size: F,        // Meters per cell
    pub walls: Vec<bool>,    // True = wall, False = free space
    pub teleporters: Vec<Teleporter>,
    pub white_holes: Vec<WhiteHole>, // Dead-end teleport destinations
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Teleporter {
    pub from_x: usize,
    pub from_y: usize,
    pub to_x: usize,
    pub to_y: usize,
    pub success_prob: F, // Probability of successful teleport
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WhiteHole {
    pub x: usize,
    pub y: usize,
    pub penalty: F, // Negative reward for ending up here
}

/// Humanoid observation
#[derive(Clone, Debug)]
pub struct HumanoidObs {
    pub x: F,
    pub y: F,
    pub angle: F,
    pub velocity: (F, F),
}

/// Goal specification
#[derive(Clone, Debug)]
pub struct GoalSpec {
    pub x: F,
    pub y: F,
    pub radius: F,
}

impl GoalSpec {
    pub fn new(x: F, y: F, radius: F) -> Self {
        Self { x, y, radius }
    }
    
    pub fn contains(&self, x: F, y: F) -> bool {
        let dx = x - self.x;
        let dy = y - self.y;
        dx * dx + dy * dy <= self.radius * self.radius
    }
}

/// Build graph for humanoid maze navigation
pub fn build_graph_humanoid(
    obs: &HumanoidObs,
    goal: &GoalSpec, 
    maze: &MazeConfig,
    bicep_paths: Option<&BICEPPaths>,
) -> (Graph, usize, usize) {
    // Create grid-based graph
    let mut graph = Graph::grid(maze.width, maze.height, maze.cell_size, Some(&maze.walls));
    
    // Add teleporter edges
    for teleporter in &maze.teleporters {
        let from_id = teleporter.from_y * maze.width + teleporter.from_x;
        let to_id = teleporter.to_y * maze.width + teleporter.to_x;
        
        // Check if destination is a white hole (dead end)
        let mut weight = teleporter.success_prob;
        for white_hole in &maze.white_holes {
            let white_hole_id = white_hole.y * maze.width + white_hole.x;
            if to_id == white_hole_id {
                // Strong penalty for known dead-end white holes
                // Apply both penalty factor and severity-based reduction
                weight *= 0.1 * (1.0 - white_hole.penalty.abs().min(0.8)); // 90% penalty + severity
                break;
            }
        }
        
        // Add stochastic teleport edge (downweighted by success probability and white holes)
        graph.edges.push(Edge::new(from_id, to_id, weight));
    }
    
    // Rebuild adjacency after adding teleporter edges
    let mut new_graph = Graph::new(graph.nodes, graph.edges);
    
    // Find current node (snap to nearest)
    let current_node = new_graph.nearest_node(obs.x, obs.y).unwrap_or(0);
    
    // Add goal node
    let goal_node = new_graph.add_goal(goal.x, goal.y);
    
    // Connect goal to nearby free cells
    add_goal_connections(&mut new_graph, goal_node, goal, maze);
    
    // Add BICEP-discovered edges if available
    if let Some(paths) = bicep_paths {
        add_bicep_edges(&mut new_graph, paths, maze);
    }
    
    (new_graph, current_node, goal_node)
}

/// Connect goal node to nearby reachable cells
fn add_goal_connections(graph: &mut Graph, goal_node: usize, goal: &GoalSpec, maze: &MazeConfig) {
    let goal_cell_x = (goal.x / maze.cell_size) as usize;
    let goal_cell_y = (goal.y / maze.cell_size) as usize;
    
    // Connect to cells within goal radius
    let radius_cells = ((goal.radius / maze.cell_size) as usize).max(1);
    
    for dy in 0..=radius_cells {
        for dx in 0..=radius_cells {
            for &(sx, sy) in &[(goal_cell_x + dx, goal_cell_y + dy),
                               (goal_cell_x.saturating_sub(dx), goal_cell_y + dy),
                               (goal_cell_x + dx, goal_cell_y.saturating_sub(dy)),
                               (goal_cell_x.saturating_sub(dx), goal_cell_y.saturating_sub(dy))] {
                
                if sx < maze.width && sy < maze.height {
                    let cell_id = sy * maze.width + sx;
                    let cell_center_x = sx as F * maze.cell_size + maze.cell_size * 0.5;
                    let cell_center_y = sy as F * maze.cell_size + maze.cell_size * 0.5;
                    
                    // Check if cell is within goal and not a wall
                    if goal.contains(cell_center_x, cell_center_y) && !maze.walls[cell_id] {
                        // Bidirectional connection with high weight
                        graph.edges.push(Edge::new(cell_id, goal_node, 10.0));
                        graph.edges.push(Edge::new(goal_node, cell_id, 10.0));
                    }
                }
            }
        }
    }
}

/// Add edges discovered by BICEP rollouts
fn add_bicep_edges(graph: &mut Graph, paths: &BICEPPaths, maze: &MazeConfig) {
    let mut edge_counts: HashMap<(usize, usize), usize> = HashMap::new();
    
    // Count transitions in BICEP paths
    for path in &paths.paths {
        for window in path.states.windows(2) {
            let from_cell = world_to_cell(window[0].x, window[0].y, maze);
            let to_cell = world_to_cell(window[1].x, window[1].y, maze);
            
            if let (Some(from), Some(to)) = (from_cell, to_cell) {
                if from != to { // Avoid self-loops
                    *edge_counts.entry((from, to)).or_insert(0) += 1;
                }
            }
        }
    }
    
    // Add edges with weights proportional to frequency
    let min_count = 3; // Minimum observations to add edge
    let max_weight = 2.0;
    
    for ((from, to), count) in edge_counts {
        if count >= min_count {
            let weight = (count as F / 10.0).min(max_weight);
            graph.edges.push(Edge::new(from, to, weight));
        }
    }
}

/// Convert world coordinates to grid cell
fn world_to_cell(x: F, y: F, maze: &MazeConfig) -> Option<usize> {
    let cell_x = (x / maze.cell_size) as usize;
    let cell_y = (y / maze.cell_size) as usize;
    
    if cell_x < maze.width && cell_y < maze.height {
        let cell_id = cell_y * maze.width + cell_x;
        if !maze.walls[cell_id] {
            Some(cell_id)
        } else {
            None
        }
    } else {
        None
    }
}

/// Build priors for humanoid maze
pub fn build_humanoid_priors(
    graph: &Graph,
    current_node: usize,
    goal_node: usize,
    enn_state: Option<&fusion_core::FusionState>,
    bicep_success_rate: Option<F>,
) -> Priors {
    let mut builder = PriorBuilder::new(graph.num_nodes())
        .with_goals(&[goal_node]);
    
    // Add ENN prior for current state
    if let Some(state) = enn_state {
        let enn_source = PriorSource::from_enn_reliability(
            state.q_prior_enn,
            state.obs_reliability,
            ETA_MIN,
            ETA_MAX,
        );
        builder = builder.add_prior(current_node, enn_source);
    }
    
    // Add BICEP success rate if available
    if let Some(success_rate) = bicep_success_rate {
        let bicep_source = PriorSource::Manual {
            value: success_rate,
            confidence: 0.8_f32.clamp(0.05, 0.95),
        };
        builder = builder.add_prior(current_node, bicep_source);
    }
    
    // Add white hole penalties
    // (In practice, would parse from maze config and add fail nodes)
    
    builder.build()
}

/// BICEP path integration (mock structure)
#[derive(Clone, Debug)]
pub struct BICEPPaths {
    pub paths: Vec<BICEPPath>,
}

#[derive(Clone, Debug)]  
pub struct BICEPPath {
    pub states: Vec<BICEPState>,
    pub success: bool,
}

#[derive(Clone, Debug)]
pub struct BICEPState {
    pub x: F,
    pub y: F,
    pub t: F,
}

impl BICEPPaths {
    pub fn success_rate(&self) -> F {
        if self.paths.is_empty() {
            return 0.5;
        }
        
        let successes = self.paths.iter().filter(|p| p.success).count();
        successes as F / self.paths.len() as F
    }
    
    pub fn confidence(&self) -> F {
        let n = self.paths.len();
        if n == 0 {
            return 0.1;
        }
        
        // Confidence increases with sample size, decreases with variance
        let success_rate = self.success_rate();
        let variance = success_rate * (1.0 - success_rate);
        
        let sample_factor = (n as F) / ((n as F) + 16.0);
        let variance_factor = 1.0 - variance * 0.5;
        
        // Cap eta to [0.05, 0.95] to prevent bad priors from overriding graph
        let raw_confidence = sample_factor * variance_factor;
        raw_confidence.clamp(0.05, 0.95)
    }
}

/// Default maze configurations for testing
impl MazeConfig {
    pub fn empty_room(width: usize, height: usize) -> Self {
        Self {
            width,
            height, 
            cell_size: 0.5,
            walls: vec![false; width * height],
            teleporters: Vec::new(),
            white_holes: Vec::new(),
        }
    }
    
    pub fn simple_maze() -> Self {
        let width = 10;
        let height = 10;
        let mut walls = vec![false; width * height];
        
        // Add walls around perimeter
        for i in 0..height {
            for j in 0..width {
                if i == 0 || i == height - 1 || j == 0 || j == width - 1 {
                    walls[i * width + j] = true;
                }
            }
        }
        
        // Add some internal walls (create maze structure)
        for i in 2..height-2 {
            for j in 2..width-2 {
                if (i + j) % 4 == 0 {
                    walls[i * width + j] = true;
                }
            }
        }
        
        Self {
            width,
            height,
            cell_size: 0.5,
            walls,
            teleporters: Vec::new(),
            white_holes: Vec::new(),
        }
    }
    
    pub fn teleport_maze() -> Self {
        let mut maze = Self::simple_maze();
        
        // Add teleporter from (2,2) to (7,7) with 80% success
        maze.teleporters.push(Teleporter {
            from_x: 2,
            from_y: 2,
            to_x: 7,
            to_y: 7,
            success_prob: 0.8,
        });
        
        // Add white hole at (8,8)
        maze.white_holes.push(WhiteHole {
            x: 8,
            y: 8, 
            penalty: -10.0,
        });
        
        maze
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_maze_config() {
        let maze = MazeConfig::simple_maze();
        assert_eq!(maze.width, 10);
        assert_eq!(maze.height, 10);
        assert_eq!(maze.walls.len(), 100);
        
        // Check that perimeter is walls
        assert!(maze.walls[0]); // Top-left corner
        assert!(maze.walls[9]); // Top-right corner
        assert!(maze.walls[90]); // Bottom-left corner
        assert!(maze.walls[99]); // Bottom-right corner
        
        // Check that some interior cells are free
        assert!(!maze.walls[11]); // Should be free
    }
    
    #[test]
    fn test_goal_contains() {
        let goal = GoalSpec::new(5.0, 5.0, 2.0);
        
        assert!(goal.contains(5.0, 5.0)); // Center
        assert!(goal.contains(6.0, 5.0)); // Within radius
        assert!(!goal.contains(8.0, 5.0)); // Outside radius
    }
    
    #[test]
    fn test_world_to_cell() {
        let maze = MazeConfig::empty_room(10, 10);
        
        assert_eq!(world_to_cell(0.25, 0.25, &maze), Some(0)); // Top-left
        assert_eq!(world_to_cell(1.25, 0.25, &maze), Some(2)); // Cell (2, 0)
        assert_eq!(world_to_cell(0.25, 1.25, &maze), Some(20)); // Cell (0, 2)
        
        // Out of bounds
        assert_eq!(world_to_cell(-1.0, 0.0, &maze), None);
        assert_eq!(world_to_cell(10.0, 0.0, &maze), None);
    }
    
    #[test]
    fn test_build_graph_simple() {
        let maze = MazeConfig::empty_room(3, 3);
        let obs = HumanoidObs {
            x: 0.25,
            y: 0.25,
            angle: 0.0,
            velocity: (0.0, 0.0),
        };
        let goal = GoalSpec::new(2.25, 2.25, 0.5);
        
        let (graph, current, goal_node) = build_graph_humanoid(&obs, &goal, &maze, None);
        
        assert_eq!(graph.num_nodes(), 10); // 9 maze cells + 1 goal node
        assert_eq!(current, 0); // Should snap to cell (0,0)
        assert_eq!(goal_node, 9); // Goal should be last node added
        
        // Check that goal has connections
        assert!(!graph.neighbors(goal_node).is_empty());
    }
    
    #[test]
    fn test_bicep_paths() {
        let paths = BICEPPaths {
            paths: vec![
                BICEPPath { states: vec![], success: true },
                BICEPPath { states: vec![], success: false },
                BICEPPath { states: vec![], success: true },
                BICEPPath { states: vec![], success: true },
            ],
        };
        
        assert!((paths.success_rate() - 0.75).abs() < 1e-6);
        assert!(paths.confidence() > 0.1);
        assert!(paths.confidence() < 1.0);
    }
}
