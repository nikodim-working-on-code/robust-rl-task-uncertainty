"""Predictive Control Algorithm for Task-Uncertain RL"""

import numpy as np


class PredictiveControlAgent:
    """
    Predictive Control (MPC-style) Algorithm.
    
    At each step, simulates forward rollouts for each expert policy
    and selects the one with highest cumulative reward in the true environment.
    """
    
    def __init__(self, models, horizon=50):
        """
        Initialize Predictive Control agent.
        
        Args:
            models: List of pre-trained expert policies
            horizon: Simulation horizon for rollouts (T0)
        """
        self.models = models
        self.horizon = horizon
        self.m = len(models)
    
    def execute(self, env, num_steps=1000):
        """
        Execute Predictive Control algorithm.
        
        Args:
            env: Environment to execute in
            num_steps: Maximum number of steps
            
        Returns:
            trajectory: Agent trajectory (full state with theta)
            policy_selections: History of selected policy indices
        """
        trajectory = []
        policy_selections = []
        
        obs, info = env.reset()
        current_state = info['info']
        
        for step in range(num_steps):
            # Simulate rollouts for each policy
            rewards = []
            first_actions = []
            
            for i, policy in enumerate(self.models):
                # Save environment state
                env_obstacles = [obs.copy() for obs in env.obstacles]
                env_goal = env.goal_state.copy()
                env_state_backup = env.state.copy()
                
                # Simulate rollout
                total_reward = 0
                first_action = None
                
                for t in range(self.horizon):
                    action, _ = policy.predict(obs, deterministic=True)
                    
                    if t == 0:
                        first_action = action
                    
                    obs_sim, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    
                    if terminated or truncated:
                        break
                
                rewards.append(total_reward)
                first_actions.append(first_action)
                
                # Restore environment state
                env.obstacles = env_obstacles
                env.goal_state = env_goal
                env.state = env_state_backup
            
            # Select best policy
            best_idx = np.argmax(rewards)
            policy_selections.append(best_idx + 1)
            
            # Execute first action of best policy
            best_action = first_actions[best_idx]
            obs, reward, terminated, truncated, info = env.step(best_action)
            
            trajectory.append(info['state'])
            
            if terminated or truncated:
                break
        
        return np.array(trajectory), policy_selections
