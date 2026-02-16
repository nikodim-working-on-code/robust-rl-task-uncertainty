"""Visualization utilities for trajectories and analysis"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint


def run_episode(env, model, num_steps=1000):
    """
    Run a single episode with a given model.
    
    Args:
        env: Environment instance
        model: Trained policy model
        num_steps: Maximum number of steps
        
    Returns:
        trajectory: Array of states [x, y, theta]
        env_state: Dictionary with obstacles and goal
    """
    obs, _ = env.reset()
    trajectory = []
    
    env_state = {
        'obstacles': env.obstacles.copy(),
        'goal_state': env.goal_state.copy()
    }
    
    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, info = env.step(action)
        trajectory.append(info['state'])
        
        if terminated:
            break
            
    return np.array(trajectory), env_state


def plot_trajectory(env, trajectory, env_state, title):
    """
    Plot simple trajectory without triangles.
    
    Args:
        env: Environment instance
        trajectory: Trajectory array
        env_state: Environment state dict
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    plt.axis('equal')
    plt.axis([0, env.map_size[0], 0, env.map_size[1]])

    # Draw obstacles
    for center, radius in env_state['obstacles']:
        circle = Circle((center[0], center[1]), radius, 
                       edgecolor='r', facecolor='none', alpha=0.5)
        plt.gca().add_patch(circle)

    # Draw goal
    plt.plot(env_state['goal_state'][0], env_state['goal_state'][1], 
             'go', markersize=10, label='Goal')

    # Draw trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1, label='Trajectory')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'bo', markersize=3)

    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/figures/{title.replace(" ", "_")}.png', dpi=150)
    plt.show()


def plot_trajectory_with_triangles(env, trajectory, env_state, title):
    """
    Plot trajectory with sensor visibility triangles.
    
    Args:
        env: Environment instance
        trajectory: Trajectory array [x, y, theta]
        env_state: Environment state dict
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    plt.axis('equal')
    plt.axis([0, env.map_size[0], 0, env.map_size[1]])

    # Draw obstacles
    for center, radius in env_state['obstacles']:
        circle = Circle((center[0], center[1]), radius, 
                       edgecolor='r', facecolor='none', alpha=0.5)
        plt.gca().add_patch(circle)

    # Draw goal
    plt.plot(env_state['goal_state'][0], env_state['goal_state'][1], 
             'go', markersize=10, label='Goal')

    # Draw trajectory
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=1, label='Trajectory')
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'bo', markersize=3)

    # Draw triangles and intersections every 20 steps
    for i, (x, y, theta) in enumerate(trajectory):
        if i % 20 == 0 and i < len(trajectory)-1:
            plt.text(x, y + 0.1, str(i), fontsize=6, color='black', 
                    ha='center', va='bottom')
            
            triangle_points = env._get_triangle_points([x, y, theta])
            triangle_patch = Polygon(
                triangle_points,
                closed=True,
                edgecolor='blue',
                facecolor='blue',
                alpha=0.1
            )
            plt.gca().add_patch(triangle_patch)

            shapely_triangle = ShapelyPolygon(triangle_points)
            total_area = 0.0
            
            for center, radius in env_state['obstacles']:
                shapely_circle = ShapelyPoint(center).buffer(radius)
                intersection = shapely_triangle.intersection(shapely_circle)
                
                if not intersection.is_empty:
                    area = intersection.area
                    total_area += area
                    
                    if intersection.geom_type == 'Polygon':
                        intersection_patch = Polygon(
                            list(intersection.exterior.coords),
                            closed=True,
                            color='orange',
                            alpha=0.5
                        )
                        plt.gca().add_patch(intersection_patch)
                    elif intersection.geom_type == 'MultiPolygon':
                        for geom in intersection.geoms:
                            if geom.geom_type == 'Polygon':
                                intersection_patch = Polygon(
                                    list(geom.exterior.coords),
                                    closed=True,
                                    color='orange',
                                    alpha=0.5
                                )
                                plt.gca().add_patch(intersection_patch)

            plt.text(x, y - 0.2, f"{total_area:.2f}", fontsize=6, 
                    color='black', ha='center', va='bottom')

    plt.title(title)
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/figures/{title.replace(" ", "_")}.png', dpi=150)
    plt.show()


def plot_lambda_history(lambda_history, epochs, title):
    """
    Plot history of selected policies (lambda indices).
    
    Args:
        lambda_history: List of selected policy indices
        epochs: Number of epochs
        title: Plot title
    """
    plt.figure()
    plt.plot(range(epochs), lambda_history, 'b-', label='Selected Policy')
    plt.xlabel('Step (Epoch)')
    plt.ylabel('Policy Index')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/figures/{title.replace(" ", "_")}.png', dpi=150)
    plt.show()
