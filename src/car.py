import math
import pygame
from .utils import line_intersection, dist

class Car:
    def __init__(self, x, y, angle=0, num_rays=8):
        self.x = x
        self.y = y
        self.angle = angle
        self.velocity = 0
        self.acceleration = 0.5
        self.max_velocity = 8.0
        self.friction = 0.2
        self.turn_speed = math.radians(5)
        self.width = 20
        self.height = 10
        self.num_rays = num_rays
        self.ray_length = 300

    def turn(self, direction):
        # direction: 1 for right, -1 for left
        if self.velocity != 0:
            self.angle += direction * self.turn_speed
            self.angle %= (2 * math.pi)

    def accelerate(self):
        self.velocity = min(self.velocity + self.acceleration, self.max_velocity)

    def brake(self):
        self.velocity = max(self.velocity - self.acceleration, -self.max_velocity / 2)

    def update(self):
        if self.velocity > 0:
            self.velocity = max(self.velocity - self.friction, 0)
        elif self.velocity < 0:
            self.velocity = min(self.velocity + self.friction, 0)
            
        self.x += math.cos(self.angle) * self.velocity
        self.y += math.sin(self.angle) * self.velocity

    def get_corners(self):
        corners = []
        for ix, iy in [(1,1), (1,-1), (-1,-1), (-1,1)]:
            cx = self.x + ix * (self.height / 2) * math.sin(self.angle) + iy * (self.width / 2) * math.cos(self.angle)
            cy = self.y - ix * (self.height / 2) * math.cos(self.angle) + iy * (self.width / 2) * math.sin(self.angle)
            corners.append((cx, cy))
        return corners

    def check_collision(self, walls):
        corners = self.get_corners()
        car_segments = [(corners[i], corners[(i+1)%4]) for i in range(4)]
        
        for w in walls:
            w_p1 = (w[0], w[1])
            w_p2 = (w[2], w[3])
            for cp1, cp2 in car_segments:
                if line_intersection(cp1, cp2, w_p1, w_p2):
                    return True
        return False

    def check_checkpoint(self, checkpoints, current_checkpoint):
        if current_checkpoint >= len(checkpoints):
            return current_checkpoint
            
        cp = checkpoints[current_checkpoint]
        w_p1 = (cp[0], cp[1])
        w_p2 = (cp[2], cp[3])
        
        corners = self.get_corners()
        car_segments = [(corners[i], corners[(i+1)%4]) for i in range(4)]
        
        for cp1, cp2 in car_segments:
            if line_intersection(cp1, cp2, w_p1, w_p2):
                return current_checkpoint + 1
        return current_checkpoint

    def cast_rays(self, walls):
        distances = []
        intersections = []
        
        # Rays from front: spread from -90 to +90 degrees relative to car front
        start_angle = -math.pi / 2
        angle_step = math.pi / (self.num_rays - 1) if self.num_rays > 1 else 0
        
        for i in range(self.num_rays):
            ray_angle = self.angle + start_angle + i * angle_step
            end_x = self.x + math.cos(ray_angle) * self.ray_length
            end_y = self.y + math.sin(ray_angle) * self.ray_length
            
            p1 = (self.x, self.y)
            p2 = (end_x, end_y)
            
            min_dist = self.ray_length
            closest_intersect = None
            for w in walls:
                w_p1 = (w[0], w[1])
                w_p2 = (w[2], w[3])
                intersect = line_intersection(p1, p2, w_p1, w_p2)
                if intersect:
                    d = dist(p1, intersect)
                    if d < min_dist:
                        min_dist = d
                        closest_intersect = intersect
            distances.append(min_dist)
            intersections.append(closest_intersect if closest_intersect else p2)
            
        return distances, intersections
