"""
The coverage path planning problem implemented by Jaegoo Choy et al.
"""
import random
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class IRLEnv(gym.Env):
    """
    Description:
        A robot can move four direction based on a grid map. The robot starts at the ranndom point, and the goal is to coverage all area with the shortest path.

    Observation: 
        Type: Discrete(11)
        Num	Observation               Type                MIN     MAX
        0	Cart Position             Discrete(11)       
        
    Actions:
        Type: Discrete(4)
        Num	Action
        0	Move left
        1	Move right

    Reward:
        #Reward is -1 for every step taken, including the termination step

    Starting State:

    Episode Termination:
        The robot crash into a wall
        Episode length is greater than 200
        Solved Requirements
        
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):

        self.dim = 11
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.dim)
        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        
        x_pos = state
        
        done = False
        if action == 0:
            x_pos = x_pos - 1
            if x_pos == 0:
                done = True
        else:
            x_pos = x_pos + 1
            if x_pos == self.dim - 1:
                done = True
        
        reward = -1
        if done:
            reward = 20
            
        self.state = x_pos
        
        return self.state, reward, done, {}

    def reset(self):
        pos = random.randint(1, self.dim - 2)
        self.state = pos
        return self.state
    
    def render(self, mode='human'):
        screen_width = 60 * self.dim
        screen_height = 60
        robot_radius = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            
            robot = rendering.make_circle(robot_radius)
            robot.set_color(1.0, 0.0, 0.0)
            self.robotrans = rendering.Transform()
            robot.add_attr(self.robotrans)
            self.viewer.add_geom(robot)
            
            for i in range(1, self.dim):
                line = rendering.Line((60 * i, 0), (60 * i, screen_height))
                self.viewer.add_geom(line)

        pos = self.state
        x_pos = 30 + 60 * pos
        y_pos = screen_height / 2
        self.robotrans.set_translation(x_pos, y_pos)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None