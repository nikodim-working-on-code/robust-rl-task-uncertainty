"""
Phase 2: Complex Environment Experiments
Test algorithms in cluttered environments with multiple goals
"""

from src.environment.custom_env import CustomEnv
from stable_baselines3 import PPO
from src.algorithms.dual_lambda import DualLambdaAgent
from src.algorithms.predictive_control import PredictiveControlAgent
from src.utils.visualization import plot_trajectory
import numpy as np


def create_complex_environment(num_goals=2):
    """Create complex environment with multiple goals and mixed obstacles"""
    env = CustomEnv(min_obstacles=10, max_obstacles=15)
    # Add custom logic for multiple goals if needed
    return env


def test_complex_scenarios():
    """Test algorithms in complex scenarios"""
    print("Phase 2: Testing in Complex Environments...")
    
    # Load pre-trained models
    model_r1 = PPO.load("models/ppo_rad_1")
    model_r2 = PPO.load("models/ppo_rad_2")
    model_r3 = PPO.load("models/ppo_rad_3")
    
    models = [model_r1, model_r2, model_r3]
    
    # Test with 2, 3, and 4 goals
    for num_goals in [2, 3, 4]:
        print(f"\nTesting with {num_goals} goals...")
        env = create_complex_environment(num_goals=num_goals)
        
        # Test Dual-Lambda
        print("  Running Dual-Lambda...")
        # Add execution logic here
        
        # Test Predictive Control
        print("  Running Predictive Control...")
        # Add execution logic here
    
    print("\nPhase 2 completed!")


if __name__ == "__main__":
    test_complex_scenarios()
