"""Dual-Lambda Algorithm for Task-Uncertain RL"""

import numpy as np
from src.utils.metrics import project_to_simplex, normalize_action, limit_state_change


class DualLambdaAgent:
    """
    Dual-Lambda Algorithm based on Lagrangian duality.
    
    Implements the primal-dual optimization approach for robust RL
    under task uncertainty. The algorithm maintains a mixture weight
    vector λ over expert policies and updates it using gradient descent.
    """
    
    def __init__(self, models, environments, eta_lambda=0.05, T0=80):
        """
        Initialize Dual-Lambda agent.
        
        Args:
            models: List of pre-trained expert policies
            environments: List of training environments
            eta_lambda: Learning rate for dual variables
            T0: Rollout horizon length
        """
        self.models = models
        self.environments = environments
        self.eta_lambda = eta_lambda
        self.T0 = T0
        self.m = len(models)
        
        # Initialize lambda uniformly on simplex
        self.lambda_ = np.ones(self.m) / self.m
    
    def execute(self, radius_ind, epochs=80):
        """
        Execute Dual-Lambda algorithm for given environment.
        
        Args:
            radius_ind: Index of environment (1, 2, or 3)
            epochs: Number of epochs to run
            
        Returns:
            trajectory: Agent trajectory (positions)
            lambda_history: History of selected policies
        """
        trajectory = []
        lambda_history = []
        
        chosen_env = self.environments[radius_ind - 1]
        obs, info_start_point = chosen_env.reset()
        
        for k in range(epochs):
            # Sample policy according to current lambda
            chosen_policy_idx = np.random.choice(self.m, p=self.lambda_)
            chosen_policy = self.models[chosen_policy_idx]
            
            # Rollout for T0 steps
            states, actions = [], []
            obs_current = obs
            
            for t in range(self.T0):
                action, _ = chosen_policy.predict(obs_current, deterministic=True)
                action = normalize_action(action)
                obs_current, reward, terminated, truncated, info = chosen_env.step(action)
                
                states.append(obs_current)
                actions.append(action)
                
                if terminated or truncated:
                    break
            
            # Evaluate on all environments
            rewards_all_envs = []
            for env in self.environments:
                total_reward = 0
                for state, action in zip(states, actions):
                    _, reward, _, _, _ = env.step(action)
                    total_reward += reward
                rewards_all_envs.append(total_reward)
            
            # Update dual variables
            lambda_half = self.lambda_ - (self.eta_lambda / self.T0) * np.array(rewards_all_envs)
            self.lambda_ = project_to_simplex(lambda_half)
            
            # Select best policy based on current lambda
            best_policy_idx = np.argmax(self.lambda_)
            lambda_history.append(best_policy_idx + 1)
            best_model = self.models[best_policy_idx]
            
            # Execute one step with best policy
            if k == 0:
                trajectory.append(info_start_point['info'][:2])
            
            action, _ = best_model.predict(obs, deterministic=True)
            action = normalize_action(action)
            obs, reward, terminated, truncated, info_step = chosen_env.step(action)
            
            if len(trajectory) > 0:
                last_state = trajectory[-1]
                info_step['state'][:2] = limit_state_change(
                    info_step['state'][:2], last_state
                )
            
            if k != 0:
                trajectory.append(info_step['state'][:2])
        
        return np.array(trajectory), lambda_history
