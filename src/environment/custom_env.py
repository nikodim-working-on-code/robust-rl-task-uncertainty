"""Custom 2D Navigation Environment for Task-Uncertain RL"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import random
from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint
from matplotlib.patches import Circle, Polygon


class CustomEnv(gym.Env):
    """Custom environment for robot navigation with extended sensors"""

    def __init__(self, min_obstacles=3, max_obstacles=10, map_size=[10, 10], 
                 sensor_range=2, sensor_angle=60):
        super().__init__()
        
        # Environment parameters
        self.min_obstacles = min_obstacles
        self.max_obstacles = max_obstacles
        self.map_size = map_size
        self.sensor_range = sensor_range
        self.sensor_angle = sensor_angle

        # Movement parameters
        self.linear_step = 0.1
        self.angular_step = 0.5

        # Action space: [linear_velocity, angular_velocity]
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(2,), 
            dtype=np.float32
        )

        # Observation space: [distance_to_goal, angle_to_goal, 
        #                     distance_to_obstacle, angle_to_obstacle]
        max_distance = np.linalg.norm(map_size)
        self.observation_space = spaces.Box(
            low=np.array([0, -np.pi, 0, -np.pi], dtype=np.float32),
            high=np.array([max_distance, np.pi, sensor_range, np.pi], dtype=np.float32),
            shape=(4,),
            dtype=np.float32
        )

        self.state = None
        self.obstacles = []
        self.goal_state = None
        self._generate_environment()

    def _generate_environment(self):
        """Generate random obstacles and goal, allowing obstacle overlap"""
        self.obstacles = []
        
        num_obstacles = random.randint(self.min_obstacles, self.max_obstacles)
        
        # Generate obstacles with different radii
        for _ in range(num_obstacles):
            radius = random.choice([1, 2, 3])
            center_x = random.uniform(radius, self.map_size[0] - radius)
            center_y = random.uniform(radius, self.map_size[1] - radius)
            center = np.array([center_x, center_y])
            self.obstacles.append((center, radius))
        
        # Generate goal position
        valid_goal = False
        attempts = 0
        
        while not valid_goal and attempts < 50:
            goal_x = random.uniform(1, self.map_size[0] - 1)
            goal_y = random.uniform(1, self.map_size[1] - 1)
            self.goal_state = np.array([goal_x, goal_y])
            
            valid_goal = True
            for obs_center, obs_radius in self.obstacles:
                if np.linalg.norm(self.goal_state - obs_center) < (obs_radius + 0.5):
                    valid_goal = False
                    break
            
            attempts += 1
            
            if attempts >= 50:
                goal_x = random.uniform(1, self.map_size[0] - 1)
                goal_y = random.uniform(1, self.map_size[1] - 1)
                self.goal_state = np.array([goal_x, goal_y])
                break
        
        # Generate starting position
        valid_start = False
        attempts = 0
        
        while not valid_start and attempts < 50:
            start_x = random.uniform(1, self.map_size[0] - 1)
            start_y = random.uniform(1, self.map_size[1] - 1)
            start_point = np.array([start_x, start_y])
            
            valid_start = True
            for obs_center, obs_radius in self.obstacles:
                if np.linalg.norm(start_point - obs_center) < (obs_radius + 0.5):
                    valid_start = False
                    break
            
            attempts += 1
            
            if attempts >= 50:
                start_x = random.uniform(1, self.map_size[0] - 1)
                start_y = random.uniform(1, self.map_size[1] - 1)
                start_point = np.array([start_x, start_y])
                break
        
        # State includes position and orientation
        self.state = np.array([start_point[0], start_point[1], 0.0], dtype=np.float32)

    def _get_obs(self):
        """Compute observation vector"""
        x, y, theta = self.state
        agent_point = ShapelyPoint(x, y)
        
        # Goal parameters
        dx_goal = self.goal_state[0] - x
        dy_goal = self.goal_state[1] - y
        distance_goal = np.hypot(dx_goal, dy_goal)
        angle_goal = np.arctan2(dy_goal, dx_goal) - theta
        angle_goal = (angle_goal + np.pi) % (2 * np.pi) - np.pi
        
        # Default obstacle parameters
        distance_obstacle = self.sensor_range
        angle_obstacle = 0.0
        
        # Build sensor visibility zone
        half_angle = np.radians(self.sensor_angle) / 2
        point_left = (
            x + self.sensor_range * np.cos(theta + half_angle),
            y + self.sensor_range * np.sin(theta + half_angle)
        )
        point_right = (
            x + self.sensor_range * np.cos(theta - half_angle),
            y + self.sensor_range * np.sin(theta - half_angle)
        )
        sensor_polygon = ShapelyPolygon([(x, y), point_left, point_right])
        
        # Check intersection with obstacles
        closest_point = None
        min_dist = float('inf')
        
        for center, radius in self.obstacles:
            obstacle_circle = ShapelyPoint(center).buffer(radius)
            intersection = sensor_polygon.intersection(obstacle_circle.boundary)
            
            if not intersection.is_empty:
                if intersection.geom_type == 'MultiPoint':
                    for p in intersection.geoms:
                        d = agent_point.distance(p)
                        if d < min_dist:
                            min_dist = d
                            closest_point = p
                else:
                    temp_point = intersection.interpolate(intersection.project(agent_point))
                    d = agent_point.distance(temp_point)
                    if d < min_dist:
                        min_dist = d
                        closest_point = temp_point
        
        if closest_point:
            dx = closest_point.x - x
            dy = closest_point.y - y
            distance_obstacle = min(np.hypot(dx, dy), self.sensor_range)
            angle_obstacle = np.arctan2(dy, dx) - theta
            angle_obstacle = (angle_obstacle + np.pi) % (2 * np.pi) - np.pi
        
        return np.array([
            distance_goal,
            angle_goal,
            distance_obstacle,
            angle_obstacle
        ], dtype=np.float32)

    def step(self, action):
        """Execute one time step"""
        linear_vel = action[0] * self.linear_step
        angular_vel = action[1] * self.angular_step

        theta = (self.state[2] + angular_vel) % (2 * np.pi)
        dx = linear_vel * np.cos(theta)
        dy = linear_vel * np.sin(theta)
        
        x = np.clip(self.state[0] + dx, 0, self.map_size[0])
        y = np.clip(self.state[1] + dy, 0, self.map_size[1])

        self.state = np.array([x, y, theta], dtype=np.float32)
        obs = self._get_obs()
        reward = self._compute_reward(self.state, action)
        terminated = bool(np.linalg.norm(self.state[:2] - self.goal_state) < 0.5)
        
        return obs, reward, terminated, False, {'state': self.state}

    def _compute_reward(self, state, action):
        """Compute reward: distance to goal + collision penalty"""
        reward_goal = -np.linalg.norm(state[:2] - self.goal_state)
        
        reward_avoid = 0
        for center, radius in self.obstacles:
            dist_to_obstacle = np.linalg.norm(state[:2] - center)
            if dist_to_obstacle <= radius:
                reward_avoid = -15
                break
                
        return reward_goal + reward_avoid

    def reset(self, seed=None, options=None):
        """Reset environment"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self._generate_environment()
        return self._get_obs(), {'info': self.state}

    def render(self):
        """Render environment"""
        plt.cla()
        plt.xlim(0, self.map_size[0])
        plt.ylim(0, self.map_size[1])
        
        # Robot
        plt.arrow(
            self.state[0], self.state[1],
            0.5*np.cos(self.state[2]), 0.5*np.sin(self.state[2]),
            head_width=0.3, fc='blue'
        )
        
        # Goal
        plt.plot(self.goal_state[0], self.goal_state[1], 'go', markersize=10)
        
        # Obstacles
        for center, radius in self.obstacles:
            plt.gca().add_patch(plt.Circle((center[0], center[1]), radius, color='r', alpha=0.3))
        
        # Sensor field of view
        half_angle = np.radians(self.sensor_angle)/2
        points = [
            [self.state[0], self.state[1]],
            [self.state[0] + self.sensor_range*np.cos(self.state[2] + half_angle),
             self.state[1] + self.sensor_range*np.sin(self.state[2] + half_angle)],
            [self.state[0] + self.sensor_range*np.cos(self.state[2] - half_angle),
             self.state[1] + self.sensor_range*np.sin(self.state[2] - half_angle)]
        ]
        plt.gca().add_patch(Polygon(points, closed=True, color='blue', alpha=0.1))
        
        plt.grid(True)
        plt.pause(0.01)

    def close(self):
        """Close environment"""
        plt.close()
        
    def _get_triangle_points(self, state):
        """Return coordinates of sensor visibility zone vertices"""
        x, y, theta = state
        half_angle = np.radians(self.sensor_angle) / 2
        
        return [
            (x, y),
            (x + self.sensor_range * np.cos(theta + half_angle),
             y + self.sensor_range * np.sin(theta + half_angle)),
            (x + self.sensor_range * np.cos(theta - half_angle),
             y + self.sensor_range * np.sin(theta - half_angle))
        ]
