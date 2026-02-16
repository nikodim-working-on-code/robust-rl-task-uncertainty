"""Utility functions for mathematical operations"""

import numpy as np


def project_to_simplex(x):
    """
    Project vector x onto the probability simplex.
    
    Ensures that the result satisfies:
    - All elements are non-negative
    - Sum of elements equals 1
    
    Args:
        x: Input vector (numpy array)
        
    Returns:
        Projected vector on the simplex
    """
    x_sorted = np.sort(x)[::-1]
    cumsum = np.cumsum(x_sorted)
    rho = np.where(x_sorted * (np.arange(1, len(x) + 1)) > (cumsum - 1))[0][-1]
    theta = (cumsum[rho] - 1) / (rho + 1)
    return np.maximum(x - theta, 0)


def normalize_action(action, max_step_size=0.15):
    """
    Normalize action to ensure it doesn't exceed maximum step size.
    
    Args:
        action: Action vector (numpy array)
        max_step_size: Maximum allowed norm of action
        
    Returns:
        Normalized action vector
    """
    action_norm = np.linalg.norm(action)
    if action_norm > max_step_size:
        action = action / action_norm * max_step_size
    return action


def limit_state_change(new_state, last_state, max_step_size=0.15):
    """
    Limit the change between consecutive states.
    
    Args:
        new_state: New state vector
        last_state: Previous state vector
        max_step_size: Maximum allowed change
        
    Returns:
        Limited new state
    """
    delta = new_state - last_state
    delta_norm = np.linalg.norm(delta)
    if delta_norm > max_step_size:
        new_state = last_state + delta / delta_norm * max_step_size
    return new_state
