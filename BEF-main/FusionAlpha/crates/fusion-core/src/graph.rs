use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Generic node features (spatial coords + extra)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeFeat {
    pub x: f32,
    pub y: f32,
    pub extra: Vec<f32>,
}

impl NodeFeat {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            extra: Vec::new(),
        }
    }

    pub fn with_extra(x: f32, y: f32, extra: Vec<f32>) -> Self {
        Self { x, y, extra }
    }

    pub fn distance_to(&self, other: &NodeFeat) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        let mut v = vec![self.x, self.y];
        v.extend(&self.extra);
        v
    }
}

/// Weighted edge between nodes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Edge {
    pub u: usize, // source node id
    pub v: usize, // target node id
    pub w: f32,   // edge weight (reliability/frequency)
}

impl Edge {
    pub fn new(u: usize, v: usize, w: f32) -> Self {
        Self { u, v, w }
    }
}

/// Task graph for committor propagation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<NodeFeat>,
    pub edges: Vec<Edge>,
    adjacency: Option<Vec<Vec<(usize, f32)>>>, // cached adjacency lists
}

impl Graph {
    pub fn new(nodes: Vec<NodeFeat>, edges: Vec<Edge>) -> Self {
        let mut g = Self {
            nodes,
            edges,
            adjacency: None,
        };
        g.build_adjacency();
        g
    }

    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Build adjacency list representation for fast neighbor queries
    fn build_adjacency(&mut self) {
        let n = self.nodes.len();
        let mut adj = vec![Vec::new(); n];

        for edge in &self.edges {
            if edge.u < n && edge.v < n {
                adj[edge.u].push((edge.v, edge.w));
            }
        }

        for neighbors in adj.iter_mut() {
            neighbors.sort_by(|(a, wa), (b, wb)| {
                a.cmp(b)
                    .then_with(|| wa.partial_cmp(wb).unwrap_or(Ordering::Equal))
            });
        }

        self.edges
            .sort_by(|a, b| a.u.cmp(&b.u).then_with(|| a.v.cmp(&b.v)));

        self.adjacency = Some(adj);
    }

    /// Get neighbors of node u with edge weights
    pub fn neighbors(&self, u: usize) -> &[(usize, f32)] {
        match &self.adjacency {
            Some(adj) => adj.get(u).map(|v| v.as_slice()).unwrap_or(&[]),
            None => &[],
        }
    }

    /// Add goal node at specified coordinates
    pub fn add_goal(&mut self, x: f32, y: f32) -> usize {
        let goal_id = self.nodes.len();
        self.nodes.push(NodeFeat::new(x, y));
        self.build_adjacency(); // rebuild adjacency
        goal_id
    }

    /// Add fail node(s) - absorbing boundaries
    pub fn add_fail(&mut self, x: f32, y: f32) -> usize {
        let fail_id = self.nodes.len();
        self.nodes.push(NodeFeat::new(x, y));
        self.build_adjacency();
        fail_id
    }

    /// Find nearest node to given coordinates
    pub fn nearest_node(&self, x: f32, y: f32) -> Option<usize> {
        let target = NodeFeat::new(x, y);
        self.nodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                target
                    .distance_to(a)
                    .partial_cmp(&target.distance_to(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Create grid-based graph for spatial domains
    pub fn grid(width: usize, height: usize, cell_size: f32, walls: Option<&[bool]>) -> Self {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Create nodes
        for i in 0..height {
            for j in 0..width {
                let x = j as f32 * cell_size;
                let y = i as f32 * cell_size;
                nodes.push(NodeFeat::new(x, y));
            }
        }

        // Create edges (4-connected grid)
        for i in 0..height {
            for j in 0..width {
                let node_id = i * width + j;

                // Skip if current cell is wall
                if let Some(w) = walls {
                    if w.get(node_id) == Some(&true) {
                        continue;
                    }
                }

                // Connect to neighbors
                let neighbors = [
                    (i.wrapping_sub(1), j), // up
                    (i + 1, j),             // down
                    (i, j.wrapping_sub(1)), // left
                    (i, j + 1),             // right
                ];

                for (ni, nj) in neighbors {
                    if ni < height && nj < width {
                        let neighbor_id = ni * width + nj;

                        // Skip if neighbor is wall
                        if let Some(w) = walls {
                            if w.get(neighbor_id) == Some(&true) {
                                continue;
                            }
                        }

                        edges.push(Edge::new(node_id, neighbor_id, 1.0));
                    }
                }
            }
        }

        Self::new(nodes, edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_feat() {
        let n1 = NodeFeat::new(0.0, 0.0);
        let n2 = NodeFeat::new(3.0, 4.0);
        assert!((n1.distance_to(&n2) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_grid_graph() {
        let g = Graph::grid(3, 3, 1.0, None);
        assert_eq!(g.num_nodes(), 9);

        // Each interior node has 4 neighbors, boundary nodes fewer
        let center = 4; // middle of 3x3
        assert_eq!(g.neighbors(center).len(), 4);

        let corner = 0;
        assert_eq!(g.neighbors(corner).len(), 2);
    }

    #[test]
    fn test_walls() {
        let mut walls = vec![false; 9];
        walls[4] = true; // block center

        let g = Graph::grid(3, 3, 1.0, Some(&walls));

        // Center node should have no edges
        assert_eq!(g.neighbors(4).len(), 0);

        // Neighbors of center should not connect to center
        for &(neighbor, _) in g.neighbors(1) {
            assert_ne!(neighbor, 4);
        }
    }
}
