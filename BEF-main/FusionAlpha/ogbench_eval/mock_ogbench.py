#!/usr/bin/env python3
"""
Mock OGBench Environment for Fusion Alpha Testing
Simulates the OGBench API without requiring the full installation
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any, List, Optional
import time
import json
import os

class MockHumanoidMaze(gym.Env):
    """Mock HumanoidMaze environment following OGBench API"""
    
    def __init__(self, maze_type="small", render_mode=None):
        super().__init__()
        
        self.maze_type = maze_type
        self.render_mode = render_mode
        
        # Environment parameters
        if maze_type == "small":
            self.maze_size = (10, 10)
            self.cell_size = 0.5
        elif maze_type == "giant":
            self.maze_size = (20, 20)  
            self.cell_size = 0.5
        else:
            self.maze_size = (15, 15)
            self.cell_size = 0.5
            
        self.width, self.height = self.maze_size
        
        # Action space: [forward_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32)
        )
        
        # Observation space: [x, y, angle, goal_x, goal_y]
        max_coord = max(self.width, self.height) * self.cell_size
        self.observation_space = spaces.Box(
            low=np.array([-max_coord, -max_coord, -np.pi, -max_coord, -max_coord], dtype=np.float32),
            high=np.array([max_coord, max_coord, np.pi, max_coord, max_coord], dtype=np.float32)
        )
        
        # Create maze walls (simple pattern)
        self.walls = self._create_maze_walls()
        
        self.reset()
        
    def _create_maze_walls(self) -> np.ndarray:
        """Create maze wall pattern"""
        walls = np.zeros((self.height, self.width), dtype=bool)
        
        # Perimeter walls
        walls[0, :] = True
        walls[-1, :] = True  
        walls[:, 0] = True
        walls[:, -1] = True
        
        # Internal maze structure
        for i in range(2, self.height-2, 3):
            for j in range(2, self.width-2, 3):
                if np.random.rand() > 0.3:  # 70% chance of wall
                    walls[i, j] = True
                    
        return walls
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        # Random start position (avoid walls)
        while True:
            start_i = np.random.randint(1, self.height-1)
            start_j = np.random.randint(1, self.width-1)
            if not self.walls[start_i, start_j]:
                break
                
        self.agent_x = start_j * self.cell_size + self.cell_size/2
        self.agent_y = start_i * self.cell_size + self.cell_size/2
        self.agent_angle = np.random.uniform(-np.pi, np.pi)
        
        # Random goal position (avoid walls, far from start)
        while True:
            goal_i = np.random.randint(1, self.height-1)
            goal_j = np.random.randint(1, self.width-1)
            goal_x = goal_j * self.cell_size + self.cell_size/2
            goal_y = goal_i * self.cell_size + self.cell_size/2
            
            dist = np.sqrt((goal_x - self.agent_x)**2 + (goal_y - self.agent_y)**2)
            if not self.walls[goal_i, goal_j] and dist > 3.0:
                break
                
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_radius = 0.8
        
        self.steps = 0
        self.max_steps = 500 if self.maze_type == "small" else 1000
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment"""
        self.steps += 1
        
        # Apply action
        forward_vel, angular_vel = action
        dt = 0.05  # 20 Hz
        
        # Update agent state
        self.agent_angle += angular_vel * dt
        self.agent_angle = ((self.agent_angle + np.pi) % (2 * np.pi)) - np.pi  # Wrap angle
        
        # Move forward
        dx = forward_vel * np.cos(self.agent_angle) * dt
        dy = forward_vel * np.sin(self.agent_angle) * dt
        
        new_x = self.agent_x + dx
        new_y = self.agent_y + dy
        
        # Check wall collision
        if not self._is_in_wall(new_x, new_y):
            self.agent_x = new_x
            self.agent_y = new_y
            
        # Compute reward
        reward = self._compute_reward(action)
        
        # Check termination
        terminated = self._is_goal_reached()
        truncated = self.steps >= self.max_steps
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
        
    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        return np.array([
            self.agent_x,
            self.agent_y, 
            self.agent_angle,
            self.goal_x,
            self.goal_y,
        ], dtype=np.float32)
        
    def _get_info(self) -> Dict:
        """Get info dict"""
        return {
            'distance_to_goal': np.sqrt((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2),
            'steps': self.steps,
            'success': self._is_goal_reached(),
        }
        
    def _is_in_wall(self, x: float, y: float) -> bool:
        """Check if position is in a wall"""
        i = int(y / self.cell_size)
        j = int(x / self.cell_size)
        
        if i < 0 or i >= self.height or j < 0 or j >= self.width:
            return True
            
        return self.walls[i, j]
        
    def _is_goal_reached(self) -> bool:
        """Check if goal is reached"""
        dist = np.sqrt((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)
        return dist <= self.goal_radius
        
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward"""
        # Distance reward
        dist_to_goal = np.sqrt((self.agent_x - self.goal_x)**2 + (self.agent_y - self.goal_y)**2)
        dist_reward = -0.01 * dist_to_goal
        
        # Goal reward
        if self._is_goal_reached():
            return 100.0 + dist_reward
            
        # Step penalty
        step_penalty = -0.1
        
        # Control penalty
        control_penalty = -0.001 * (action[0]**2 + action[1]**2)
        
        return dist_reward + step_penalty + control_penalty
        
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
            
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Draw maze
        for i in range(self.height):
            for j in range(self.width):
                if self.walls[i, j]:
                    rect = plt.Rectangle((j*self.cell_size, i*self.cell_size), 
                                       self.cell_size, self.cell_size, 
                                       facecolor='black')
                    ax.add_patch(rect)
                    
        # Draw goal
        goal_circle = plt.Circle((self.goal_x, self.goal_y), self.goal_radius, 
                               color='green', alpha=0.5)
        ax.add_patch(goal_circle)
        
        # Draw agent
        agent_circle = plt.Circle((self.agent_x, self.agent_y), 0.2, color='red')
        ax.add_patch(agent_circle)
        
        # Draw agent direction
        arrow_length = 0.4
        arrow_dx = arrow_length * np.cos(self.agent_angle)
        arrow_dy = arrow_length * np.sin(self.agent_angle)
        ax.arrow(self.agent_x, self.agent_y, arrow_dx, arrow_dy, 
                head_width=0.1, head_length=0.1, fc='red', ec='red')
        
        ax.set_xlim(0, self.width * self.cell_size)
        ax.set_ylim(0, self.height * self.cell_size)
        ax.set_aspect('equal')
        ax.set_title(f'HumanoidMaze ({self.maze_type})')
        
        plt.tight_layout()
        plt.show()

class MockAntSoccer(gym.Env):
    """Mock AntSoccer environment"""
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Field dimensions
        self.field_width = 12.0
        self.field_height = 8.0
        self.goal_width = 2.0
        
        # Action space: ant joint torques (8 joints)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        
        # Observation space: [ant_pos(2), ball_pos(2), ball_vel(2), goal_info(2)]
        self.observation_space = spaces.Box(
            low=-20.0, high=20.0, shape=(8,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        # Reset positions
        self.ant_x = np.random.uniform(1.0, 3.0)
        self.ant_y = np.random.uniform(2.0, 6.0)
        self.ball_x = np.random.uniform(4.0, 8.0)
        self.ball_y = np.random.uniform(2.0, 6.0)
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        
        # Goal position (right side)
        self.goal_x = 12.0
        self.goal_y = 4.0
        
        self.steps = 0
        self.max_steps = 1000
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.steps += 1
        
        # Simplified ant dynamics
        dt = 0.02
        
        # Convert joint torques to ant movement (simplified)
        ant_force_x = np.mean(action[:4]) * 2.0
        ant_force_y = np.mean(action[4:]) * 2.0
        
        self.ant_x += ant_force_x * dt
        self.ant_y += ant_force_y * dt
        
        # Keep ant in bounds
        self.ant_x = np.clip(self.ant_x, 0.5, self.field_width - 0.5)
        self.ant_y = np.clip(self.ant_y, 0.5, self.field_height - 0.5)
        
        # Ball interaction (if ant is close)
        ant_ball_dist = np.sqrt((self.ant_x - self.ball_x)**2 + (self.ant_y - self.ball_y)**2)
        
        if ant_ball_dist < 1.0:  # Ant can push ball
            push_strength = 0.5
            push_dir_x = (self.ball_x - self.ant_x) / (ant_ball_dist + 1e-6)
            push_dir_y = (self.ball_y - self.ant_y) / (ant_ball_dist + 1e-6)
            
            self.ball_vx += push_strength * push_dir_x * dt
            self.ball_vy += push_strength * push_dir_y * dt
            
        # Ball physics
        friction = 0.9
        self.ball_vx *= friction
        self.ball_vy *= friction
        
        self.ball_x += self.ball_vx * dt
        self.ball_y += self.ball_vy * dt
        
        # Ball boundaries
        if self.ball_x <= 0:
            self.ball_x = 0.1
            self.ball_vx = abs(self.ball_vx) * 0.5
        elif self.ball_x >= self.field_width:
            self.ball_x = self.field_width - 0.1
            self.ball_vx = -abs(self.ball_vx) * 0.5
            
        if self.ball_y <= 0:
            self.ball_y = 0.1
            self.ball_vy = abs(self.ball_vy) * 0.5
        elif self.ball_y >= self.field_height:
            self.ball_y = self.field_height - 0.1
            self.ball_vy = -abs(self.ball_vy) * 0.5
            
        # Check goal
        scored = (self.ball_x >= self.goal_x - 0.5 and 
                 abs(self.ball_y - self.goal_y) <= self.goal_width/2)
                 
        reward = self._compute_reward(action, scored)
        
        terminated = scored
        truncated = self.steps >= self.max_steps
        
        obs = self._get_obs()
        info = self._get_info()
        info['scored'] = scored
        
        return obs, reward, terminated, truncated, info
        
    def _get_obs(self) -> np.ndarray:
        return np.array([
            self.ant_x, self.ant_y,
            self.ball_x, self.ball_y,
            self.ball_vx, self.ball_vy,
            self.goal_x, self.goal_y,
        ], dtype=np.float32)
        
    def _get_info(self) -> Dict:
        ball_goal_dist = np.sqrt((self.ball_x - self.goal_x)**2 + (self.ball_y - self.goal_y)**2)
        return {
            'ball_goal_distance': ball_goal_dist,
            'steps': self.steps,
            'ant_ball_distance': np.sqrt((self.ant_x - self.ball_x)**2 + (self.ant_y - self.ball_y)**2),
        }
        
    def _compute_reward(self, action: np.ndarray, scored: bool) -> float:
        if scored:
            return 1000.0
            
        # Distance rewards
        ball_goal_dist = np.sqrt((self.ball_x - self.goal_x)**2 + (self.ball_y - self.goal_y)**2)
        ant_ball_dist = np.sqrt((self.ant_x - self.ball_x)**2 + (self.ant_y - self.ball_y)**2)
        
        # Reward being close to ball and ball being close to goal
        reward = -0.1 * ball_goal_dist - 0.05 * ant_ball_dist
        
        # Control penalty
        reward -= 0.001 * np.sum(action**2)
        
        # Step penalty
        reward -= 0.01
        
        return reward

class MockPuzzle(gym.Env):
    """Mock Puzzle-4x5 environment"""
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.width = 5
        self.height = 4
        self.n_buttons = 20
        
        # Action space: discrete button press (0-19)
        self.action_space = spaces.Discrete(self.n_buttons)
        
        # Observation space: 20-bit light configuration
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(20,), dtype=np.float32
        )
        
        self.reset()
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            
        # Random initial configuration
        self.config = np.random.randint(0, 2**20, dtype=np.uint32)
        
        # Random goal configuration (but solvable)
        self.goal_config = np.random.randint(0, 2**20, dtype=np.uint32)
        
        self.steps = 0
        self.max_steps = 50
        
        obs = self._config_to_obs(self.config)
        info = self._get_info()
        
        return obs, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.steps += 1
        
        # Apply button press
        mask = self._get_button_mask(action)
        self.config ^= mask
        
        # Check if solved
        solved = (self.config == self.goal_config)
        
        # Compute reward
        if solved:
            reward = 100.0
        else:
            # Hamming distance reward
            hamming_dist = bin(self.config ^ self.goal_config).count('1')
            reward = -hamming_dist - 0.1
            
        terminated = solved
        truncated = self.steps >= self.max_steps
        
        obs = self._config_to_obs(self.config)
        info = self._get_info()
        info['solved'] = solved
        
        return obs, reward, terminated, truncated, info
        
    def _config_to_obs(self, config: np.uint32) -> np.ndarray:
        """Convert bit configuration to observation array"""
        obs = np.zeros(20, dtype=np.float32)
        for i in range(20):
            obs[i] = float((config >> i) & 1)
        return obs
        
    def _get_button_mask(self, button: int) -> np.uint32:
        """Get toggle mask for button press (cross pattern)"""
        row = button // 5
        col = button % 5
        
        mask = 0
        # Toggle button itself
        mask |= (1 << button)
        
        # Toggle cross neighbors
        neighbors = [
            (row-1, col), (row+1, col),  # up, down
            (row, col-1), (row, col+1),  # left, right
        ]
        
        for r, c in neighbors:
            if 0 <= r < self.height and 0 <= c < self.width:
                neighbor_id = r * self.width + c
                mask |= (1 << neighbor_id)
                
        return np.uint32(mask)
        
    def _get_info(self) -> Dict:
        hamming_dist = bin(self.config ^ self.goal_config).count('1')
        return {
            'hamming_distance': hamming_dist,
            'steps': self.steps,
            'current_config': int(self.config),
            'goal_config': int(self.goal_config),
        }

def create_mock_env(env_name: str, **kwargs):
    """Factory function to create mock environments"""
    if "humanoid" in env_name.lower() and "maze" in env_name.lower():
        maze_type = "small"
        if "giant" in env_name.lower():
            maze_type = "giant"
        elif "teleport" in env_name.lower():
            maze_type = "teleport"
        return MockHumanoidMaze(maze_type=maze_type, **kwargs)
        
    elif "ant" in env_name.lower() and "soccer" in env_name.lower():
        return MockAntSoccer(**kwargs)
        
    elif "puzzle" in env_name.lower():
        return MockPuzzle(**kwargs)
        
    else:
        raise ValueError(f"Unknown environment: {env_name}")

if __name__ == "__main__":
    # Quick test of mock environments
    print("Testing Mock OGBench Environments")
    
    # Test HumanoidMaze
    print("\n=== HumanoidMaze Test ===")
    env = create_mock_env("HumanoidMaze-v1")
    obs, info = env.reset(seed=42)
    print(f"Initial obs: {obs}")
    print(f"Initial info: {info}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, distance={info['distance_to_goal']:.3f}")
        if term or trunc:
            break
            
    # Test AntSoccer
    print("\n=== AntSoccer Test ===")
    env = create_mock_env("AntSoccer-v1") 
    obs, info = env.reset(seed=42)
    print(f"Initial obs: {obs}")
    
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, ball_dist={info['ball_goal_distance']:.3f}")
        if term or trunc:
            break
            
    # Test Puzzle
    print("\n=== Puzzle Test ===")
    env = create_mock_env("Puzzle-4x5-v1")
    obs, info = env.reset(seed=42)
    print(f"Initial config: {info['current_config']:020b}")
    print(f"Goal config:    {info['goal_config']:020b}")
    print(f"Hamming dist: {info['hamming_distance']}")
    
    for i in range(3):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i+1}: button={action}, hamming={info['hamming_distance']}")
        if term or trunc:
            break
            
    print("✅ Mock OGBench environments working!")