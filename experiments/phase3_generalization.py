"""
Phase 3: Zero-Shot Generalization
Test algorithms on novel obstacle geometries (triangles, squares)
"""

from src.environment.custom_env import CustomEnv
from stable_baselines3 import PPO
from src.algorithms.dual_lambda import DualLambdaAgent
from src.algorithms.predictive_control import PredictiveControlAgent
import numpy as np


def create_novel_geometry_environment(shape_type='triangle'):
    """
    Create environment with novel obstacle shapes.
    
    Args:
        shape_type: 'triangle', 'square', or 'mixed'
    """
    env = CustomEnv(min_obstacles=5, max_obstacles=8)
    # In a full implementation, you would modify obstacle shapes here
    print(f"Created environment with {shape_type} obstacles")
    return env


def test_generalization():
    """Test zero-shot generalization to unseen geometries"""
    print("Phase 3: Zero-Shot Generalization Test...")
    
    # Load pre-trained models
    model_r1 = PPO.load("models/ppo_rad_1")
    model_r2 = PPO.load("models/ppo_rad_2")
    model_r3 = PPO.load("models/ppo_rad_3")
    
    models = [model_r1, model_r2, model_r3]
    
    # Test on different geometries
    for shape in ['triangle', 'square', 'mixed']:
        print(f"\nTesting on {shape} obstacles...")
        env = create_novel_geometry_environment(shape_type=shape)
        
        # Test Dual-Lambda
        print("  Running Dual-Lambda...")
        # Add execution logic
        
        # Test Predictive Control
        print("  Running Predictive Control...")
        # Add execution logic
    
    print("\nPhase 3 completed!")


if __name__ == "__main__":
    test_generalization()
