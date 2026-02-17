"""
Phase 1: Baseline Experiments
Train and test expert policies on simple environments with r=1,2,3
"""

from src.environment.custom_env import CustomEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from src.algorithms.dual_lambda import DualLambdaAgent
from src.algorithms.predictive_control import PredictiveControlAgent
from src.utils.visualization import plot_trajectory, plot_lambda_history
import matplotlib.pyplot as plt


def train_expert_policies():
    """Train three expert policies for different obstacle radii"""
    print("Training Expert Policies...")
    
    # Create environments for each radius
    env_r1 = make_vec_env(
        lambda: CustomEnv(min_obstacles=5, max_obstacles=7), 
        n_envs=12, 
        seed=0
    )
    env_r2 = make_vec_env(
        lambda: CustomEnv(min_obstacles=5, max_obstacles=7), 
        n_envs=12, 
        seed=1
    )
    env_r3 = make_vec_env(
        lambda: CustomEnv(min_obstacles=5, max_obstacles=7), 
        n_envs=12, 
        seed=2
    )
    
    # Train models
    print("Training model for radius 1...")
    model_r1 = PPO("MlpPolicy", env_r1, verbose=1)
    model_r1.learn(total_timesteps=1000000, progress_bar=True)
    model_r1.save("models/ppo_rad_1")
    
    print("Training model for radius 2...")
    model_r2 = PPO("MlpPolicy", env_r2, verbose=1)
    model_r2.learn(total_timesteps=1000000, progress_bar=True)
    model_r2.save("models/ppo_rad_2")
    
    print("Training model for radius 3...")
    model_r3 = PPO("MlpPolicy", env_r3, verbose=1)
    model_r3.learn(total_timesteps=1000000, progress_bar=True)
    model_r3.save("models/ppo_rad_3")
    
    return model_r1, model_r2, model_r3, env_r1, env_r2, env_r3


def test_dual_lambda(models, envs):
    """Test Dual-Lambda algorithm on baseline environments"""
    print("\nTesting Dual-Lambda Algorithm...")
    
    agent = DualLambdaAgent(
        models=list(models),
        environments=list(envs),
        eta_lambda=0.05,
        T0=80
    )
    
    for radius_idx in [1, 2, 3]:
        print(f"\nTesting on environment with radius {radius_idx}...")
        trajectory, lambda_history = agent.execute(
            radius_ind=radius_idx, 
            epochs=80
        )
        
        # Visualize results
        env = CustomEnv(min_obstacles=5, max_obstacles=7)
        env_state = {'obstacles': env.obstacles, 'goal_state': env.goal_state}
        
        plot_trajectory(
            env, trajectory, env_state,
            f"Dual-Lambda_Radius_{radius_idx}"
        )
        plot_lambda_history(
            lambda_history, 80,
            f"Lambda_History_Radius_{radius_idx}"
        )


def main():
    """Main execution function"""
    # Check environment
    print("Checking environment validity...")
    env_test = CustomEnv(min_obstacles=3, max_obstacles=7)
    check_env(env_test)
    print("Environment check passed!")
    
    # Train or load models
    try:
        print("\nAttempting to load pre-trained models...")
        model_r1 = PPO.load("models/ppo_rad_1")
        model_r2 = PPO.load("models/ppo_rad_2")
        model_r3 = PPO.load("models/ppo_rad_3")
        print("Models loaded successfully!")
        
        # Create environments
        env_r1 = make_vec_env(lambda: CustomEnv(min_obstacles=5, max_obstacles=7), n_envs=12, seed=0)
        env_r2 = make_vec_env(lambda: CustomEnv(min_obstacles=5, max_obstacles=7), n_envs=12, seed=1)
        env_r3 = make_vec_env(lambda: CustomEnv(min_obstacles=5, max_obstacles=7), n_envs=12, seed=2)
        
    except:
        print("\nPre-trained models not found. Training new models...")
        model_r1, model_r2, model_r3, env_r1, env_r2, env_r3 = train_expert_policies()
    
    # Test algorithms
    models = (model_r1, model_r2, model_r3)
    envs = (env_r1, env_r2, env_r3)
    
    test_dual_lambda(models, envs)
    
    print("\nPhase 1 experiments completed!")


if __name__ == "__main__":
    main()
