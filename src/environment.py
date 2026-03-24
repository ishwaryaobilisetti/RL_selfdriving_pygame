import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import yaml
import os

from .car import Car
from .utils import load_track

class CarRacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.num_rays = config['env_params']['num_rays']
        self.max_steps = config['env_params']['max_steps_per_episode']
        
        # Actions: 0=No-op, 1=Accel, 2=Brake, 3=Left, 4=Right
        self.action_space = spaces.Discrete(5)
        # Observations: normalize rays + velocity
        # rays max=300, v max=8
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_rays + 1,), dtype=np.float32)
        
        track_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tracks', 'track_1.txt')
        self.walls, self.checkpoints = load_track(track_path)
        
        self.car = None
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = 600
        
        self.current_step = 0
        self.current_checkpoint = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Start in middle of gate 0
        if len(self.checkpoints) > 0:
            cp = self.checkpoints[0]
            start_x = (cp[0] + cp[2]) / 2
            start_y = (cp[1] + cp[3]) / 2
            # Calculate angle towards next checkpoint roughly
            cp_next = self.checkpoints[1] if len(self.checkpoints) > 1 else cp
            nx = (cp_next[0] + cp_next[2]) / 2
            ny = (cp_next[1] + cp_next[3]) / 2
            angle = np.arctan2(ny - start_y, nx - start_x)
        else:
            start_x, start_y, angle = 0, -75, 0
            
        self.car = Car(start_x, start_y, angle, num_rays=self.num_rays)
        self.current_step = 0
        self.current_checkpoint = 1 % len(self.checkpoints) # Next target
        self.laps = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        distances, self.last_ray_intersections = self.car.cast_rays(self.walls)
        # Normalize
        obs = [d / self.car.ray_length for d in distances]
        obs.append(max(0.0, self.car.velocity / self.car.max_velocity))
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        self.current_step += 1
        
        # Apply action
        if action == 1:
            self.car.accelerate()
        elif action == 2:
            self.car.brake()
        elif action == 3:
            self.car.turn(-1)
        elif action == 4:
            self.car.turn(1)
            
        self.car.update()
        
        # Dense reward: reward for moving forward!
        velocity_reward = (self.car.velocity / self.car.max_velocity) * 0.5
        reward = -0.05 + velocity_reward
        
        terminated = False
        truncated = False
        
        # Check collision
        if self.car.check_collision(self.walls):
            reward -= 10.0
            terminated = True
            
        # Check checkpoint mapping
        nxt_cp = self.car.check_checkpoint(self.checkpoints, self.current_checkpoint)
        if nxt_cp > self.current_checkpoint:
            reward += 10.0 # Huge incentive to cross checkpoints
            self.current_checkpoint = nxt_cp
            if self.current_checkpoint >= len(self.checkpoints):
                # Lap complete
                reward += 50.0
                self.laps += 1
                self.current_checkpoint = 0
                
        if self.current_step >= self.max_steps:
            truncated = True
            
        obs = self._get_obs()
        
        if self.render_mode == "human":
            self.render()
            
        return obs, float(reward), terminated, truncated, {}

    def render(self):
        if self.window is None and self.render_mode is not None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode((self.window_size, self.window_size))
            else:
                s_surface = pygame.Surface((self.window_size, self.window_size))
                self.window = s_surface
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((50, 50, 50))
        
        # Shift coordinate center for rendering (0,0 in math -> 300,300 in UI)
        def tr(p):
            return (int(p[0] + self.window_size/2), int(p[1] + self.window_size/2))
            
        # Draw track walls
        for w in self.walls:
            pygame.draw.line(canvas, (255, 255, 255), tr((w[0], w[1])), tr((w[2], w[3])), 3)
            
        # Draw checkpoints
        for cp in self.checkpoints:
            pygame.draw.line(canvas, (0, 255, 0), tr((cp[0], cp[1])), tr((cp[2], cp[3])), 1)

        # Draw rays
        if hasattr(self, 'last_ray_intersections'):
            for intersect in self.last_ray_intersections:
                pygame.draw.line(canvas, (255, 0, 0), tr((self.car.x, self.car.y)), tr(intersect), 1)
                
        # Draw car
        corners = self.car.get_corners()
        tr_corners = [tr(c) for c in corners]
        pygame.draw.polygon(canvas, (0, 100, 255), tr_corners)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            self.window = canvas
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
