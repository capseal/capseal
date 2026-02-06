#!/usr/bin/env python3
"""
Fusion Alpha Agent for OGBench
Implements the main agent that uses committor planning for OGBench environments
"""

import numpy as np
import sys
import os
from typing import Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'target', 'release'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    import fusion_alpha as fa
except ImportError as e:
    print(f"Warning: Could not import fusion_alpha: {e}")
    print("Build with: cargo build --release -p fusion-bindings")
    fa = None

# Import our graph builder and ENN forward
try:
    from graph_builder import create_graph_builder
    from enn_forward import ENNForward
except ImportError as e:
    print(f"Warning: Could not import graph_builder or enn_forward: {e}")
    create_graph_builder = None
    ENNForward = None

class BaseAgent(ABC):
    """Base agent interface for OGBench"""
    
    @abstractmethod
    def reset(self):
        """Reset agent state"""
        pass
    
    @abstractmethod
    def act(self, obs, info: Dict) -> Any:
        """Select action given observation"""
        pass

class RandomAgent(BaseAgent):
    """Random baseline agent"""
    
    def __init__(self, action_space):
        self.action_space = action_space
        
    def reset(self):
        pass
        
    def act(self, obs, info: Dict):
        return self.action_space.sample()

class FusionAlphaAgent(BaseAgent):
    """Fusion Alpha committor planning agent"""
    
    def __init__(self, 
                 env_name: str,
                 action_space,
                 observation_space,
                 # Fusion Alpha parameters
                 k_neighbors: int = 10,
                 max_nodes: int = 100,
                 t_max_base: int = 20,
                 severity_scale: float = 1.0,
                 # ENN parameters
                 enn_weights_path: Optional[str] = None,
                 # BICEP parameters  
                 bicep_graph_path: Optional[str] = None):
        
        self.env_name = env_name
        self.action_space = action_space
        self.observation_space = observation_space
        
        # Fusion Alpha config
        self.k_neighbors = k_neighbors
        self.max_nodes = max_nodes
        self.t_max_base = t_max_base
        self.severity_scale = severity_scale
        
        # Agent state
        self.step_count = 0
        self.episode_count = 0
        self.success_history = []
        self.q_predictions = []  # For tracking ENN predictions
        
        # Initialize graph builder
        if create_graph_builder is not None:
            self.graph_builder = create_graph_builder(
                env_name, 
                k_neighbors=k_neighbors,
                max_nodes=max_nodes
            )
        else:
            self.graph_builder = None
            
        # Load ENN if available
        self.enn = None
        if enn_weights_path and os.path.exists(enn_weights_path) and ENNForward is not None:
            try:
                self.enn = ENNForward(enn_weights_path)
                print(f"Loaded ENN weights from {enn_weights_path}")
            except Exception as e:
                print(f"Failed to load ENN weights: {e}")
        
        # Load BICEP graph if available
        self.bicep_edges = None
        if bicep_graph_path and os.path.exists(bicep_graph_path):
            try:
                import json
                with open(bicep_graph_path, 'r') as f:
                    data = json.load(f)
                    self.bicep_edges = data.get('edges', [])
                print(f"Loaded {len(self.bicep_edges)} BICEP edges")
            except Exception as e:
                print(f"Failed to load BICEP graph: {e}")
        
        # Environment-specific setup
        self._setup_env_specific()
        
    def _setup_env_specific(self):
        """Set up environment-specific parameters"""
        if "humanoid" in self.env_name.lower():
            self.env_type = "humanoid_maze"
            self.cell_size = 1.0  # Default cell size
            
        elif "ant" in self.env_name.lower() and "soccer" in self.env_name.lower():
            self.env_type = "ant_soccer"
            
        elif "puzzle" in self.env_name.lower():
            self.env_type = "puzzle"
            self.puzzle_depth = 6
            
        else:
            raise ValueError(f"Unsupported environment: {self.env_name}")
    
    def reset(self):
        """Reset agent for new episode"""
        self.step_count = 0
        self.episode_count += 1
        
    def act(self, obs, info: Dict) -> Any:
        """Select action using Fusion Alpha planning"""
        self.step_count += 1
        
        if fa is None:
            # Fallback to random if Fusion Alpha not available
            return self.action_space.sample()
            
        try:
            if self.env_type == "humanoid_maze":
                return self._act_humanoid_maze(obs, info)
            elif self.env_type == "ant_soccer":
                return self._act_ant_soccer(obs, info)
            elif self.env_type == "puzzle":
                return self._act_puzzle(obs, info)
            else:
                return self.action_space.sample()
                
        except Exception as e:
            print(f"Fusion Alpha planning failed: {e}")
            return self.action_space.sample()
    
    def _act_humanoid_maze(self, obs, info: Dict) -> np.ndarray:
        """Action selection for HumanoidMaze"""
        # Extract observation
        agent_x, agent_y, agent_angle = obs[0], obs[1], obs[2]
        goal_x, goal_y = obs[3], obs[4]
        
        # Build current state vector
        current_state = obs.copy()
        goal_state = np.array([goal_x, goal_y] + [0.0] * (len(obs) - 2))
        
        # Add state to graph builder buffer
        if self.graph_builder is not None:
            self.graph_builder.add_state(current_state)
        
        # Use real graph builder or fallback
        if self.graph_builder is not None:
            nodes, edges, current_node, goal_node = self.graph_builder.build_graph(
                current_state, goal_state
            )
        else:
            # Fallback to simple graph
            nodes, edges, current_node, goal_node = self._create_humanoid_graph(
                agent_x, agent_y, goal_x, goal_y
            )
        
        # Get ENN predictions if available
        if self.enn is not None:
            # Convert nodes to ENN features
            enn_features = self._nodes_to_enn_features(
                nodes, current_node, goal_node, info
            )
            
            # Get ENN forward pass
            q0_enn, alpha = self.enn.forward(enn_features)
            severity = self.enn.compute_severity(alpha)
            q_prior_enn = q0_enn[current_node]
        else:
            # Fallback to mock predictions
            q_prior_enn = 0.5  # Neutral prior
            severity = 0.5
        
        # Dynamic t_max based on severity
        t_max = int(self.t_max_base + 50 * severity * self.severity_scale)
        
        # Run Fusion Alpha propagation
        q_values = fa.simple_propagate(
            nodes=nodes,
            edges=edges,
            goal_node=goal_node,
            current_node=current_node,
            enn_q_prior=q_prior_enn,
            severity=severity,
            t_max=t_max,
        )
        
        # Convert committor values to action
        action = self._committor_to_humanoid_action(
            q_values, nodes, current_node, agent_angle
        )
        
        # Store for learning
        self.q_predictions.append(q_prior_enn)
        
        return action
    
    def _act_ant_soccer(self, obs, info: Dict) -> np.ndarray:
        """Action selection for AntSoccer"""
        ant_x, ant_y = obs[0], obs[1]
        ball_x, ball_y = obs[2], obs[3]
        goal_x, goal_y = obs[6], obs[7]
        
        # Build state vectors
        current_state = obs.copy()
        # Goal state has ball at goal position
        goal_state = obs.copy()
        goal_state[2] = goal_x
        goal_state[3] = goal_y
        
        # Add state to buffer
        if self.graph_builder is not None:
            self.graph_builder.add_state(current_state)
        
        # Use real graph builder or fallback
        if self.graph_builder is not None:
            nodes, edges, current_node, goal_node = self.graph_builder.build_graph(
                current_state, goal_state
            )
        else:
            nodes, edges, current_node, goal_node = self._create_soccer_graph(
                ball_x, ball_y, goal_x, goal_y
            )
        
        # Get ENN predictions
        if self.enn is not None:
            enn_features = self._nodes_to_enn_features(
                nodes, current_node, goal_node, info
            )
            q0_enn, alpha = self.enn.forward(enn_features)
            severity = self.enn.compute_severity(alpha)
            q_prior_enn = q0_enn[current_node]
        else:
            q_prior_enn = 0.3  # Lower prior for soccer (harder)
            severity = 0.6
        
        # Dynamic t_max
        t_max = int(self.t_max_base + 50 * severity * self.severity_scale)
        
        # Run propagation
        q_values = fa.simple_propagate(
            nodes=nodes,
            edges=edges,
            goal_node=goal_node,
            current_node=current_node,
            enn_q_prior=q_prior_enn,
            severity=severity,
            t_max=t_max,
        )
        
        # Convert to ant joint torques
        action = self._committor_to_soccer_action(
            q_values, nodes, current_node, ant_x, ant_y, ball_x, ball_y
        )
        
        return action
    
    def _act_puzzle(self, obs, info: Dict) -> int:
        """Action selection for Puzzle"""
        # Current state is observation
        current_state = obs.copy()
        
        # Goal state from info
        goal_config = info.get('goal_config', 0)
        goal_state = np.zeros_like(obs)
        for i in range(len(obs)):
            if goal_config & (1 << i):
                goal_state[i] = 1.0
        
        # Use real graph builder
        if self.graph_builder is not None:
            nodes, edges, current_node, goal_node = self.graph_builder.build_graph(
                current_state, goal_state
            )
        else:
            # Fallback
            config = 0
            for i, light in enumerate(obs):
                if light > 0.5:
                    config |= (1 << i)
            nodes, edges, current_node, goal_node = self._create_puzzle_graph(
                config, goal_config
            )
        
        # Get ENN predictions
        if self.enn is not None:
            enn_features = self._nodes_to_enn_features(
                nodes, current_node, goal_node, info
            )
            q0_enn, alpha = self.enn.forward(enn_features)
            severity = self.enn.compute_severity(alpha)
            q_prior_enn = q0_enn[current_node]
        else:
            hamming_dist = info.get('hamming_distance', 10)
            q_prior_enn = max(0.1, 1.0 - hamming_dist / 20.0)
            severity = min(0.9, hamming_dist / 10.0)
        
        # Dynamic t_max
        t_max = int(self.t_max_base + 30 * severity * self.severity_scale)
        
        # Run propagation
        q_values = fa.simple_propagate(
            nodes=nodes,
            edges=edges,
            goal_node=goal_node,
            current_node=current_node,
            enn_q_prior=q_prior_enn,
            severity=severity,
            t_max=t_max,
        )
        
        # Find best button press
        action = self._committor_to_puzzle_action(q_values, current_node, nodes.shape[0])
        
        return action
    
    def _nodes_to_enn_features(self, nodes: np.ndarray, 
                               current_idx: int, goal_idx: int, 
                               info: Dict) -> np.ndarray:
        """Convert graph nodes to ENN features"""
        # This should match the feature engineering used during ENN training
        # For now, using distance-based features similar to enn_fusion_demo.py
        
        current_pos = nodes[current_idx][:2]  # Use first 2 dims as position
        goal_pos = nodes[goal_idx][:2] if goal_idx >= 0 else current_pos
        
        features = []
        for i, node in enumerate(nodes):
            pos = node[:2]
            
            # Distance features
            dist_to_current = np.linalg.norm(pos - current_pos)
            dist_to_goal = np.linalg.norm(pos - goal_pos)
            
            # Relative position feature
            x = dist_to_goal - dist_to_current
            
            # Create feature vector (matching double-well style)
            feat = np.array([
                x,                      # position-like
                x**2,                   # quadratic
                x**3,                   # cubic  
                (x**2 - 1)**2 / 4,     # potential-like
                x**3 - x,              # force-like
            ], dtype=np.float32)
            
            features.append(feat)
        
        return np.array(features, dtype=np.float32)
    
    def _create_humanoid_graph(self, agent_x, agent_y, goal_x, goal_y) -> Tuple:
        """Create simple spatial graph for humanoid"""
        # Simple 3x3 grid around agent
        grid_size = 3
        spacing = 1.5
        
        nodes = []
        # Current position
        nodes.append([agent_x, agent_y])
        
        # Neighboring positions  
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                x = agent_x + i * spacing
                y = agent_y + j * spacing
                nodes.append([x, y])
                
        # Goal position
        nodes.append([goal_x, goal_y])
        
        nodes = np.array(nodes, dtype=np.float32)
        
        # Simple connectivity (all-to-all for small graph)
        edges = []
        n_nodes = len(nodes)
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    dist = np.linalg.norm(nodes[i] - nodes[j])
                    weight = 1.0 / (1.0 + dist)  # Distance-based weights
                    edges.append([i, j, weight])
                    
        edges = np.array(edges, dtype=np.float32)
        
        current_node = 0  # Agent position is first node
        goal_node = n_nodes - 1  # Goal is last node
        
        return nodes, edges, current_node, goal_node
    
    def _create_soccer_graph(self, ball_x, ball_y, goal_x, goal_y) -> Tuple:
        """Create ball-centric graph for soccer"""
        # Grid around ball position
        grid_size = 2
        spacing = 1.0
        
        nodes = []
        # Current ball position
        nodes.append([ball_x, ball_y])
        
        # Surrounding positions
        for i in range(-grid_size, grid_size + 1):
            for j in range(-grid_size, grid_size + 1):
                if i == 0 and j == 0:
                    continue
                x = ball_x + i * spacing
                y = ball_y + j * spacing
                # Keep within field bounds roughly
                x = np.clip(x, 0.5, 11.5)
                y = np.clip(y, 0.5, 7.5)
                nodes.append([x, y])
                
        # Goal position
        nodes.append([goal_x, goal_y])
        
        nodes = np.array(nodes, dtype=np.float32)
        
        # Connect nodes with goal bias
        edges = []
        n_nodes = len(nodes)
        goal_node = n_nodes - 1
        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    dist = np.linalg.norm(nodes[i] - nodes[j])
                    weight = 1.0 / (1.0 + dist)
                    
                    # Boost edges toward goal
                    if j == goal_node:
                        weight *= 2.0
                        
                    edges.append([i, j, weight])
                    
        edges = np.array(edges, dtype=np.float32)
        
        return nodes, edges, 0, goal_node
    
    def _create_puzzle_graph(self, config, goal_config) -> Tuple:
        """Create local puzzle graph"""
        # BFS around current config
        from collections import deque
        
        visited = set()
        queue = deque([config])
        visited.add(config)
        configs = [config]
        
        depth = min(3, self.puzzle_depth)  # Limit for mock version
        
        for level in range(depth):
            level_size = len(queue)
            for _ in range(level_size):
                if not queue:
                    break
                    
                current = queue.popleft()
                
                # Try each button press
                for button in range(20):
                    mask = self._get_puzzle_mask(button)
                    next_config = current ^ mask
                    
                    if next_config not in visited:
                        visited.add(next_config)
                        queue.append(next_config)
                        configs.append(next_config)
                        
                        if len(configs) >= 20:  # Limit graph size
                            break
                            
                if len(configs) >= 20:
                    break
                    
        # Add goal if not present
        if goal_config not in configs:
            configs.append(goal_config)
            
        # Convert configs to spatial nodes (using Hamming distance as coordinates)
        nodes = []
        for cfg in configs:
            x = bin(cfg).count('1') / 20.0  # Light density
            y = bin(cfg ^ goal_config).count('1') / 20.0  # Distance to goal
            nodes.append([x, y])
            
        nodes = np.array(nodes, dtype=np.float32)
        
        # Connect configs that differ by one button press
        edges = []
        for i, cfg1 in enumerate(configs):
            for j, cfg2 in enumerate(configs):
                if i != j:
                    # Check if one button press transforms cfg1 to cfg2
                    diff = cfg1 ^ cfg2
                    if bin(diff).count('1') <= 7:  # Plausible button press
                        weight = 1.0 / (1.0 + bin(diff).count('1'))
                        edges.append([i, j, weight])
                        
        edges = np.array(edges, dtype=np.float32)
        
        current_node = 0  # Current config is first
        try:
            goal_node = configs.index(goal_config)
        except ValueError:
            goal_node = len(configs) - 1  # Use last node as proxy
            
        return nodes, edges, current_node, goal_node
    
    def _get_puzzle_mask(self, button: int) -> int:
        """Get button press mask"""
        row = button // 5
        col = button % 5
        
        mask = 1 << button  # Button itself
        
        # Cross neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 4 and 0 <= nc < 5:
                neighbor = nr * 5 + nc
                mask |= (1 << neighbor)
                
        return mask
    
    def _committor_to_humanoid_action(self, q_values, nodes, current_node, agent_angle) -> np.ndarray:
        """Convert committor values to humanoid action"""
        # Find neighbor with highest q-value
        current_pos = nodes[current_node]
        
        best_q = -1
        best_target = current_pos
        
        for i, q in enumerate(q_values):
            if i != current_node and q > best_q:
                best_q = q
                best_target = nodes[i]
        
        # Compute desired direction
        dx = best_target[0] - current_pos[0]
        dy = best_target[1] - current_pos[1]
        target_angle = np.arctan2(dy, dx)
        
        # Compute angular difference
        angle_diff = target_angle - agent_angle
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
            
        # Action: [forward_velocity, angular_velocity]
        forward_vel = 0.8 if abs(angle_diff) < 0.5 else 0.3
        angular_vel = np.clip(angle_diff * 2.0, -1.0, 1.0)
        
        return np.array([forward_vel, angular_vel], dtype=np.float32)
    
    def _committor_to_soccer_action(self, q_values, nodes, current_node, ant_x, ant_y, ball_x, ball_y) -> np.ndarray:
        """Convert committor values to ant action"""
        # Find best ball target position
        best_q = -1
        best_target = nodes[current_node]
        
        for i, q in enumerate(q_values):
            if i != current_node and q > best_q:
                best_q = q
                best_target = nodes[i]
                
        # Compute desired ant position to push ball toward target
        push_dir_x = best_target[0] - ball_x
        push_dir_y = best_target[1] - ball_y
        push_length = np.sqrt(push_dir_x**2 + push_dir_y**2) + 1e-6
        
        push_dir_x /= push_length
        push_dir_y /= push_length
        
        # Ant should be behind ball relative to target
        desired_ant_x = ball_x - 0.8 * push_dir_x
        desired_ant_y = ball_y - 0.8 * push_dir_y
        
        # Simple controller: move ant toward desired position
        ant_force_x = np.clip((desired_ant_x - ant_x) * 5.0, -1.0, 1.0)
        ant_force_y = np.clip((desired_ant_y - ant_y) * 5.0, -1.0, 1.0)
        
        # Convert to joint torques (mock)
        action = np.zeros(8, dtype=np.float32)
        action[:4] = ant_force_x * 0.5  # Front legs
        action[4:] = ant_force_y * 0.5  # Back legs
        
        # Add some noise for realism
        action += np.random.normal(0, 0.1, 8)
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def _committor_to_puzzle_action(self, q_values, current_node, n_nodes) -> int:
        """Convert committor values to button press"""
        # Find node with highest q-value (excluding current)
        best_q = -1
        best_node = current_node
        
        for i, q in enumerate(q_values):
            if i != current_node and q > best_q:
                best_q = q
                best_node = i
        
        # For puzzle with exact graph builder, edges encode the button press
        # Try to find which button leads to best_node
        if self.graph_builder is not None and hasattr(self.graph_builder, '_get_button_mask'):
            # Get current config from node
            current_config = self.graph_builder.configs[current_node]
            target_config = self.graph_builder.configs[best_node] if best_node < len(self.graph_builder.configs) else current_config
            
            # Find button that transforms current -> target
            for button in range(20):
                mask = self.graph_builder._get_button_mask(button)
                if (current_config ^ mask) == target_config:
                    return button
        
        # Fallback: prefer buttons with higher impact
        if best_q > q_values[current_node]:
            weights = []
            for button in range(20):
                mask = self._get_puzzle_mask(button)
                impact = bin(mask).count('1')
                weights.append(impact)
                
            weights = np.array(weights, dtype=np.float32)
            weights = weights / weights.sum()
            
            return np.random.choice(20, p=weights)
        else:
            return np.random.randint(20)
    
    def update(self, obs, action, reward, next_obs, done, info: Dict):
        """Update agent after step (for learning)"""
        if done:
            success = reward > 50  # Success threshold varies by env
            self.success_history.append(success)
            
            # Keep history bounded
            if len(self.success_history) > 50:
                self.success_history = self.success_history[-50:]
                
            if len(self.q_predictions) > 10:
                self.q_predictions = self.q_predictions[-10:]
            
            # Note: Real ENN training would happen offline with collected trajectories

def create_agent(env_name: str, action_space, observation_space, 
                agent_type: str = "fusion_alpha", **kwargs) -> BaseAgent:
    """Factory function to create agents"""
    if agent_type == "random":
        return RandomAgent(action_space)
    elif agent_type == "fusion_alpha":
        return FusionAlphaAgent(env_name, action_space, observation_space, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

if __name__ == "__main__":
    # Quick test
    from mock_ogbench import create_mock_env
    
    print("Testing Fusion Alpha Agent")
    
    env = create_mock_env("HumanoidMaze-v1")
    agent = create_agent("HumanoidMaze-v1", env.action_space, env.observation_space, "fusion_alpha")
    
    obs, info = env.reset(seed=42)
    agent.reset()
    
    for step in range(10):
        action = agent.act(obs, info)
        next_obs, reward, term, trunc, next_info = env.step(action)
        
        agent.update(obs, action, reward, next_obs, term or trunc, next_info)
        
        print(f"Step {step}: reward={reward:.3f}, distance={next_info.get('distance_to_goal', 0):.3f}")
        
        if term or trunc:
            break
            
        obs, info = next_obs, next_info
    
    print("✅ Fusion Alpha agent test complete!")