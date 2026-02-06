#!/usr/bin/env python3
"""
Graph builders for OGBench environments
Creates local k-NN graphs around current state for Fusion Alpha
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from sklearn.neighbors import NearestNeighbors
import json


class StateGraphBuilder:
    """Base class for building state-space graphs"""
    
    def __init__(self, k_neighbors: int = 10, max_nodes: int = 100):
        self.k_neighbors = k_neighbors
        self.max_nodes = max_nodes
        self.state_buffer = []  # Replay buffer of visited states
        
    def add_state(self, state: np.ndarray):
        """Add state to replay buffer"""
        self.state_buffer.append(state.copy())
        # Keep buffer size reasonable
        if len(self.state_buffer) > 10000:
            self.state_buffer = self.state_buffer[-5000:]
    
    def build_graph(self, current_state: np.ndarray, 
                   goal_state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Build k-NN graph around current state
        
        Returns:
            nodes: (N, d) array of states
            edges: (M, 3) array of [i, j, weight]
            current_idx: index of current state in nodes
            goal_idx: index of goal state in nodes (-1 if no goal)
        """
        raise NotImplementedError


class HumanoidMazeGraphBuilder(StateGraphBuilder):
    """Graph builder for Humanoid/Ant maze environments"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.position_dim = 2  # (x, y) position
        
    def build_graph(self, current_state: np.ndarray, 
                   goal_state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
        
        # Extract position from full state
        current_pos = current_state[:self.position_dim]
        
        # Start with current state
        nodes = [current_state]
        node_positions = [current_pos]
        current_idx = 0
        
        # Add goal if provided
        goal_idx = -1
        if goal_state is not None:
            nodes.append(goal_state)
            node_positions.append(goal_state[:self.position_dim])
            goal_idx = 1
        
        # Add nearby states from buffer
        if len(self.state_buffer) > 0:
            buffer_positions = np.array([s[:self.position_dim] for s in self.state_buffer])
            
            # Find k nearest neighbors
            nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(buffer_positions)))
            nbrs.fit(buffer_positions)
            
            distances, indices = nbrs.kneighbors([current_pos])
            
            for idx in indices[0]:
                if len(nodes) < self.max_nodes:
                    nodes.append(self.state_buffer[idx])
                    node_positions.append(buffer_positions[idx])
        
        # Convert to arrays
        nodes = np.array(nodes, dtype=np.float32)
        node_positions = np.array(node_positions, dtype=np.float32)
        
        # Build edges based on position distance
        edges = []
        n_nodes = len(nodes)
        
        for i in range(n_nodes):
            # Find k nearest neighbors for each node
            if n_nodes > 1:
                distances = np.linalg.norm(node_positions - node_positions[i], axis=1)
                nearest = np.argsort(distances)[1:self.k_neighbors+1]  # Exclude self
                
                for j in nearest:
                    if j < n_nodes:
                        # Weight inversely proportional to distance
                        dist = distances[j]
                        weight = np.exp(-dist / 2.0)  # Gaussian kernel
                        edges.append([i, j, weight])
                        edges.append([j, i, weight])  # Bidirectional
        
        edges = np.array(edges, dtype=np.float32) if edges else np.zeros((0, 3), dtype=np.float32)
        
        return nodes, edges, current_idx, goal_idx


class AntSoccerGraphBuilder(StateGraphBuilder):
    """Graph builder for AntSoccer - focuses on ball dynamics"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ant_pos_dim = 2
        self.ball_pos_dim = 2
        
    def build_graph(self, current_state: np.ndarray, 
                   goal_state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
        
        # Extract positions (ant_x, ant_y, ball_x, ball_y, ...)
        current_ball_pos = current_state[2:4]  # Ball position
        
        # Build graph around ball positions
        nodes = [current_state]
        ball_positions = [current_ball_pos]
        current_idx = 0
        
        # Add goal (typically a target ball position)
        goal_idx = -1
        if goal_state is not None:
            nodes.append(goal_state)
            ball_positions.append(goal_state[2:4])
            goal_idx = 1
        
        # Sample reachable ball positions
        # Create a grid of potential ball positions
        grid_size = int(np.sqrt(self.max_nodes))
        x_range = np.linspace(current_ball_pos[0] - 2, current_ball_pos[0] + 2, grid_size)
        y_range = np.linspace(current_ball_pos[1] - 2, current_ball_pos[1] + 2, grid_size)
        
        for x in x_range:
            for y in y_range:
                if len(nodes) < self.max_nodes:
                    # Create synthetic state with ball at (x, y)
                    new_state = current_state.copy()
                    new_state[2] = x
                    new_state[3] = y
                    nodes.append(new_state)
                    ball_positions.append([x, y])
        
        nodes = np.array(nodes, dtype=np.float32)
        ball_positions = np.array(ball_positions, dtype=np.float32)
        
        # Build edges based on ball reachability
        edges = []
        n_nodes = len(nodes)
        
        for i in range(n_nodes):
            distances = np.linalg.norm(ball_positions - ball_positions[i], axis=1)
            nearest = np.argsort(distances)[1:self.k_neighbors+1]
            
            for j in nearest:
                if j < n_nodes:
                    dist = distances[j]
                    # Prefer small moves
                    weight = np.exp(-dist / 0.5)
                    edges.append([i, j, weight])
        
        edges = np.array(edges, dtype=np.float32) if edges else np.zeros((0, 3), dtype=np.float32)
        
        return nodes, edges, current_idx, goal_idx


class PuzzleGraphBuilder(StateGraphBuilder):
    """Graph builder for Puzzle - uses exact toggle dynamics"""
    
    def __init__(self, width: int = 5, height: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.height = height
        self.n_lights = width * height
        
    def _get_button_mask(self, button: int) -> int:
        """Get toggle mask for button press (cross pattern)"""
        row = button // self.width
        col = button % self.width
        
        mask = 1 << button  # Button itself
        
        # Cross neighbors
        neighbors = [
            (row-1, col), (row+1, col),  # up, down
            (row, col-1), (row, col+1),  # left, right
        ]
        
        for r, c in neighbors:
            if 0 <= r < self.height and 0 <= c < self.width:
                neighbor_id = r * self.width + c
                mask |= (1 << neighbor_id)
        
        return mask
    
    def _state_to_config(self, state: np.ndarray) -> int:
        """Convert observation vector to bit configuration"""
        config = 0
        for i in range(self.n_lights):
            if state[i] > 0.5:
                config |= (1 << i)
        return config
    
    def _config_to_state(self, config: int, template: np.ndarray) -> np.ndarray:
        """Convert bit configuration to state vector"""
        state = template.copy()
        for i in range(self.n_lights):
            state[i] = 1.0 if (config & (1 << i)) else 0.0
        return state
    
    def build_graph(self, current_state: np.ndarray, 
                   goal_state: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, int, int]:
        
        current_config = self._state_to_config(current_state)
        goal_config = self._state_to_config(goal_state) if goal_state is not None else None
        
        # BFS to build local graph
        from collections import deque
        
        nodes = [current_state]
        configs = [current_config]
        config_to_idx = {current_config: 0}
        current_idx = 0
        
        queue = deque([(current_config, 0)])  # (config, depth)
        max_depth = 3  # Limited depth for tractability
        
        while queue and len(nodes) < self.max_nodes:
            config, depth = queue.popleft()
            
            if depth >= max_depth:
                continue
            
            # Try all button presses
            for button in range(self.n_lights):
                mask = self._get_button_mask(button)
                next_config = config ^ mask
                
                if next_config not in config_to_idx:
                    idx = len(nodes)
                    config_to_idx[next_config] = idx
                    configs.append(next_config)
                    nodes.append(self._config_to_state(next_config, current_state))
                    
                    if depth + 1 < max_depth:
                        queue.append((next_config, depth + 1))
        
        # Add goal if not already present
        goal_idx = -1
        if goal_config is not None and goal_config not in config_to_idx:
            goal_idx = len(nodes)
            nodes.append(goal_state)
            configs.append(goal_config)
            config_to_idx[goal_config] = goal_idx
        elif goal_config is not None:
            goal_idx = config_to_idx[goal_config]
        
        nodes = np.array(nodes, dtype=np.float32)
        
        # Build edges based on button presses
        edges = []
        for i, config in enumerate(configs):
            for button in range(self.n_lights):
                mask = self._get_button_mask(button)
                next_config = config ^ mask
                
                if next_config in config_to_idx:
                    j = config_to_idx[next_config]
                    # All button presses have equal weight
                    edges.append([i, j, 1.0])
        
        edges = np.array(edges, dtype=np.float32) if edges else np.zeros((0, 3), dtype=np.float32)
        
        return nodes, edges, current_idx, goal_idx


# Factory function
def create_graph_builder(env_name: str, **kwargs) -> StateGraphBuilder:
    """Create appropriate graph builder for environment"""
    
    env_lower = env_name.lower()
    
    if "humanoid" in env_lower or "ant" in env_lower and "maze" in env_lower:
        return HumanoidMazeGraphBuilder(**kwargs)
    elif "soccer" in env_lower:
        return AntSoccerGraphBuilder(**kwargs)
    elif "puzzle" in env_lower:
        return PuzzleGraphBuilder(**kwargs)
    else:
        raise ValueError(f"Unknown environment: {env_name}")


# Example usage
if __name__ == "__main__":
    print("Testing graph builders...")
    
    # Test HumanoidMaze
    print("\n=== HumanoidMaze ===")
    builder = HumanoidMazeGraphBuilder(k_neighbors=5, max_nodes=20)
    
    # Add some states to buffer
    for _ in range(50):
        state = np.random.randn(10)  # Mock state
        builder.add_state(state)
    
    current = np.array([1.0, 2.0] + [0.0] * 8)  # Position at (1, 2)
    goal = np.array([5.0, 5.0] + [0.0] * 8)     # Goal at (5, 5)
    
    nodes, edges, curr_idx, goal_idx = builder.build_graph(current, goal)
    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    print(f"Current: {curr_idx}, Goal: {goal_idx}")
    
    # Test Puzzle
    print("\n=== Puzzle ===")
    builder = PuzzleGraphBuilder(width=5, height=4, max_nodes=50)
    
    current = np.zeros(20)
    current[5] = 1.0  # One light on
    
    goal = np.zeros(20)
    goal[10] = 1.0
    goal[15] = 1.0  # Two different lights on
    
    nodes, edges, curr_idx, goal_idx = builder.build_graph(current, goal)
    print(f"Nodes: {len(nodes)}, Edges: {len(edges)}")
    print(f"Current: {curr_idx}, Goal: {goal_idx}")
    
    print("\n✅ Graph builders working!")