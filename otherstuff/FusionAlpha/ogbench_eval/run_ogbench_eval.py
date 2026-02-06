#!/usr/bin/env python3
"""
OGBench Evaluation Suite for Fusion Alpha
Runs comprehensive evaluation comparing Fusion Alpha to baseline methods
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import time
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

from mock_ogbench import create_mock_env
from fusion_alpha_agent import create_agent

@dataclass
class EvalConfig:
    """Evaluation configuration"""
    envs: List[str]
    agents: List[str]
    n_episodes: int
    max_steps_per_episode: int
    seeds: List[int]
    save_dir: str

@dataclass
class EpisodeResult:
    """Single episode results"""
    env_name: str
    agent_name: str
    episode: int
    seed: int
    success: bool
    total_reward: float
    steps: int
    final_distance: float
    solve_time: float

class OGBenchEvaluator:
    """Main evaluation class"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.results: List[EpisodeResult] = []
        
        # Create save directory
        os.makedirs(config.save_dir, exist_ok=True)
        
    def run_evaluation(self):
        """Run complete evaluation suite"""
        print("🚀 Starting OGBench Evaluation for Fusion Alpha")
        print(f"Environments: {self.config.envs}")
        print(f"Agents: {self.config.agents}")
        print(f"Episodes per config: {self.config.n_episodes}")
        
        start_time = time.time()
        
        for env_name in self.config.envs:
            print(f"\n{'='*50}")
            print(f"Environment: {env_name}")
            print(f"{'='*50}")
            
            for agent_name in self.config.agents:
                print(f"\n--- Running {agent_name} on {env_name} ---")
                
                self._eval_agent_on_env(env_name, agent_name)
                
        total_time = time.time() - start_time
        print(f"\n✅ Evaluation complete! Total time: {total_time:.1f}s")
        
        # Analyze and save results
        self._analyze_results()
        self._save_results()
        
    def _eval_agent_on_env(self, env_name: str, agent_name: str):
        """Evaluate single agent on single environment"""
        episode_results = []
        
        for episode in range(self.config.n_episodes):
            for seed in self.config.seeds:
                result = self._run_single_episode(env_name, agent_name, episode, seed)
                episode_results.append(result)
                self.results.append(result)
                
                # Print progress
                if (episode * len(self.config.seeds) + len(self.config.seeds)) % 10 == 0:
                    success_rate = sum(r.success for r in episode_results) / len(episode_results)
                    avg_reward = np.mean([r.total_reward for r in episode_results])
                    print(f"  Episodes {len(episode_results):3d}: "
                          f"Success={success_rate:.2%}, "
                          f"Avg Reward={avg_reward:.1f}")
        
        # Episode summary
        success_rate = sum(r.success for r in episode_results) / len(episode_results)
        avg_reward = np.mean([r.total_reward for r in episode_results])
        avg_steps = np.mean([r.steps for r in episode_results])
        
        print(f"\n{agent_name} on {env_name} Summary:")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average Steps: {avg_steps:.1f}")
        
    def _run_single_episode(self, env_name: str, agent_name: str, episode: int, seed: int) -> EpisodeResult:
        """Run a single episode"""
        
        # Create environment and agent
        env = create_mock_env(env_name, render_mode=None)
        agent = create_agent(env_name, env.action_space, env.observation_space, agent_name)
        
        # Reset
        obs, info = env.reset(seed=seed)
        agent.reset()
        
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        for step in range(self.config.max_steps_per_episode):
            # Agent acts
            action = agent.act(obs, info)
            
            # Environment step
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            
            # Update agent
            if hasattr(agent, 'update'):
                agent.update(obs, action, reward, next_obs, terminated or truncated, next_info)
            
            total_reward += reward
            steps += 1
            
            # Check termination
            if terminated or truncated:
                break
                
            obs, info = next_obs, next_info
            
        solve_time = time.time() - start_time
        
        # Determine success and final distance
        success = info.get('success', False) or info.get('solved', False) or info.get('scored', False)
        if not success and terminated:
            success = True  # Terminated successfully
            
        # Get final distance/metric
        final_distance = info.get('distance_to_goal', 
                                info.get('ball_goal_distance',
                                       info.get('hamming_distance', 0)))
        
        return EpisodeResult(
            env_name=env_name,
            agent_name=agent_name,
            episode=episode,
            seed=seed,
            success=success,
            total_reward=total_reward,
            steps=steps,
            final_distance=final_distance,
            solve_time=solve_time,
        )
    
    def _analyze_results(self):
        """Analyze and visualize results"""
        print("\n📊 Analyzing Results...")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame([
            {
                'env': r.env_name,
                'agent': r.agent_name,
                'episode': r.episode,
                'seed': r.seed,
                'success': r.success,
                'reward': r.total_reward,
                'steps': r.steps,
                'distance': r.final_distance,
                'time': r.solve_time,
            }
            for r in self.results
        ])
        
        # Aggregate statistics
        stats = df.groupby(['env', 'agent']).agg({
            'success': ['mean', 'std'],
            'reward': ['mean', 'std'],
            'steps': ['mean', 'std'],
            'distance': ['mean', 'std'],
            'time': ['mean', 'std'],
        }).round(3)
        
        print("\n📈 Performance Summary:")
        print(stats)
        
        # Create visualizations
        self._create_plots(df)
        
        return df, stats
        
    def _create_plots(self, df: pd.DataFrame):
        """Create evaluation plots"""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'seaborn')
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fusion Alpha OGBench Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Success Rate by Environment
        ax1 = axes[0, 0]
        success_rates = df.groupby(['env', 'agent'])['success'].mean().unstack()
        success_rates.plot(kind='bar', ax=ax1, rot=45)
        ax1.set_title('Success Rate by Environment')
        ax1.set_ylabel('Success Rate')
        ax1.legend(title='Agent')
        ax1.grid(True, alpha=0.3)
        
        # 2. Average Reward by Environment  
        ax2 = axes[0, 1]
        avg_rewards = df.groupby(['env', 'agent'])['reward'].mean().unstack()
        avg_rewards.plot(kind='bar', ax=ax2, rot=45)
        ax2.set_title('Average Reward by Environment')
        ax2.set_ylabel('Average Reward')
        ax2.legend(title='Agent')
        ax2.grid(True, alpha=0.3)
        
        # 3. Steps to Completion
        ax3 = axes[1, 0]
        # Box plot of steps by agent
        sns.boxplot(data=df, x='agent', y='steps', hue='env', ax=ax3)
        ax3.set_title('Steps to Completion Distribution')
        ax3.set_ylabel('Steps')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Success Rate Over Episodes (Learning Curves)
        ax4 = axes[1, 1]
        for env in df['env'].unique():
            for agent in df['agent'].unique():
                env_agent_df = df[(df['env'] == env) & (df['agent'] == agent)]
                if not env_agent_df.empty:
                    # Rolling success rate
                    rolling_success = env_agent_df.groupby('episode')['success'].mean().rolling(5, min_periods=1).mean()
                    ax4.plot(rolling_success.index, rolling_success.values, 
                           label=f'{agent}-{env}', marker='o', markersize=3)
                    
        ax4.set_title('Learning Curves (Rolling Success Rate)')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Success Rate (Rolling Avg)')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.save_dir, 'evaluation_results.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional detailed plots
        self._create_detailed_plots(df)
        
    def _create_detailed_plots(self, df: pd.DataFrame):
        """Create additional detailed analysis plots"""
        
        # Environment-specific analysis
        for env in df['env'].unique():
            env_df = df[df['env'] == env]
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'Detailed Analysis: {env}', fontsize=14, fontweight='bold')
            
            # Success rate comparison
            ax1 = axes[0, 0]
            success_by_agent = env_df.groupby('agent')['success'].mean()
            success_by_agent.plot(kind='bar', ax=ax1, color=['skyblue', 'lightcoral', 'lightgreen'][:len(success_by_agent)])
            ax1.set_title('Success Rate by Agent')
            ax1.set_ylabel('Success Rate')
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            
            # Reward distribution
            ax2 = axes[0, 1]
            sns.boxplot(data=env_df, x='agent', y='reward', ax=ax2)
            ax2.set_title('Reward Distribution')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            
            # Episode success over time
            ax3 = axes[1, 0]
            for agent in env_df['agent'].unique():
                agent_df = env_df[env_df['agent'] == agent]
                episode_success = agent_df.groupby('episode')['success'].mean()
                ax3.plot(episode_success.index, episode_success.values, 
                        label=agent, marker='o', markersize=2)
            ax3.set_title('Success Rate by Episode')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Success Rate')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Final distance/quality metric
            ax4 = axes[1, 1]
            sns.boxplot(data=env_df, x='agent', y='distance', ax=ax4)
            ax4.set_title('Final Distance/Error Distribution')
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
            
            plt.tight_layout()
            env_clean = env.replace('/', '_').replace('-', '_')
            plt.savefig(os.path.join(self.config.save_dir, f'{env_clean}_detailed.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
    def _save_results(self):
        """Save results to files"""
        print(f"\n💾 Saving results to {self.config.save_dir}")
        
        # Save raw results
        results_data = [
            {
                'env_name': r.env_name,
                'agent_name': r.agent_name,
                'episode': r.episode,
                'seed': r.seed,
                'success': r.success,
                'total_reward': r.total_reward,
                'steps': r.steps,
                'final_distance': r.final_distance,
                'solve_time': r.solve_time,
            }
            for r in self.results
        ]
        
        with open(os.path.join(self.config.save_dir, 'raw_results.json'), 'w') as f:
            json.dump(results_data, f, indent=2)
            
        # Save aggregated statistics
        df = pd.DataFrame(results_data)
        
        summary_stats = {}
        for env in df['env_name'].unique():
            summary_stats[env] = {}
            env_df = df[df['env_name'] == env]
            
            for agent in env_df['agent_name'].unique():
                agent_df = env_df[env_df['agent_name'] == agent]
                
                summary_stats[env][agent] = {
                    'success_rate': float(agent_df['success'].mean()),
                    'success_rate_std': float(agent_df['success'].std()),
                    'avg_reward': float(agent_df['total_reward'].mean()),
                    'avg_reward_std': float(agent_df['total_reward'].std()),
                    'avg_steps': float(agent_df['steps'].mean()),
                    'avg_steps_std': float(agent_df['steps'].std()),
                    'avg_final_distance': float(agent_df['final_distance'].mean()),
                    'avg_solve_time': float(agent_df['solve_time'].mean()),
                    'total_episodes': len(agent_df),
                }
                
        with open(os.path.join(self.config.save_dir, 'summary_stats.json'), 'w') as f:
            json.dump(summary_stats, f, indent=2)
            
        # Save CSV for further analysis
        df.to_csv(os.path.join(self.config.save_dir, 'results.csv'), index=False)
        
        print("Results saved:")
        print(f"  - Raw results: raw_results.json")
        print(f"  - Summary stats: summary_stats.json") 
        print(f"  - CSV data: results.csv")
        print(f"  - Plots: evaluation_results.png + detailed plots")

def main():
    """Main evaluation script"""
    
    # Evaluation configuration
    config = EvalConfig(
        envs=[
            "HumanoidMaze-small-v1",
            "AntSoccer-v1", 
            "Puzzle-4x5-v1",
        ],
        agents=[
            "random",
            "fusion_alpha",
        ],
        n_episodes=20,  # Episodes per seed
        max_steps_per_episode=500,
        seeds=[42, 123, 456],  # Multiple seeds for robustness
        save_dir="ogbench_results"
    )
    
    # Run evaluation
    evaluator = OGBenchEvaluator(config)
    evaluator.run_evaluation()
    
    print("\n🎉 OGBench evaluation complete!")
    print("Check the 'ogbench_results' directory for detailed results and plots.")

if __name__ == "__main__":
    main()