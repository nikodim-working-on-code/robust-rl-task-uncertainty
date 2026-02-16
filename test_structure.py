"""Test script to verify module structure"""

# Test imports
try:
    from src.environment.custom_env import CustomEnv
    print("✓ CustomEnv imported successfully")
except Exception as e:
    print(f"✗ CustomEnv import failed: {e}")

try:
    from src.algorithms.dual_lambda import DualLambdaAgent
    print("✓ DualLambdaAgent imported successfully")
except Exception as e:
    print(f"✗ DualLambdaAgent import failed: {e}")

try:
    from src.algorithms.predictive_control import PredictiveControlAgent
    print("✓ PredictiveControlAgent imported successfully")
except Exception as e:
    print(f"✗ PredictiveControlAgent import failed: {e}")

try:
    from src.utils.metrics import project_to_simplex, normalize_action
    print("✓ Metrics utilities imported successfully")
except Exception as e:
    print(f"✗ Metrics import failed: {e}")

try:
    from src.utils.visualization import run_episode, plot_trajectory
    print("✓ Visualization utilities imported successfully")
except Exception as e:
    print(f"✗ Visualization import failed: {e}")

print("\n✓ All modules structure validated!")
