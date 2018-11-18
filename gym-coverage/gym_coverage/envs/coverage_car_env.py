"""
The coverage path planning problem implemented by Jaegoo Choy et al.
"""
import random
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CoverageCarEnv(gym.Env):
    """
    Description:
        A robot can move four direction based on a grid map. The robot starts at the ranndom point, and the goal is to coverage all area with the shortest path.

    Observation: 
        Type: Tuple(2)
        Num	Observation               Type                MIN     MAX
        0	Cart Position             Discrete(100)       
        1	Coverage Map              Box(100)            -1      1
        
    Actions:
        Type: Discrete(4)
        Num	Action
        0	Move left
        1	Move up
        2	Move right
        3	Move down       

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

        self.world_width = 10
        self.world_height = 10
        self.action_space = spaces.Discrete(4)
        self.low = np.array([-self.world_width, -self.world_height, -math.pi])
        self.high = np.array([self.world_width, self.world_height, math.pi])
        self.observation_space = spaces.Box(self.low, self.high)
        self.seed()
        self.viewer = None
        #self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x_pos, y_pos, theta_pos = state
        
        col = False
        if action == 0:
            x_pos = x_pos - 1
            theta_pos = -math.pi
            if x_pos < 0:
                col = True
                x_pos = 0
        elif action == 1:
            y_pos = y_pos + 1
            theta_pos = math.pi / 2 
            if y_pos > self.world_height - 1:
                col = True
                y_pos = self.world_height - 1
        elif action == 2:
            x_pos = x_pos + 1
            theta_pos = 0
            if x_pos > self.world_width - 1:
                col = True
                x_pos = self.world_width - 1
        else:
            y_pos = y_pos - 1
            theta_pos = -math.pi / 2 
            if y_pos < 0:
                col = True
                y_pos = 0
                
        done = col
        if done:
            reward = -1000
        else:
            reward = -1
        
        self.state = (x_pos, y_pos, theta_pos)
        return self.state, reward, done, {}

    def reset(self):
        #x_pos = self.world_width / 2 * (random.random() - 1)
        #y_pos = self.world_height / 2 * (random.random() - 1)
        #theta_pos = 2 * math.pi * (random.random() - 1)
        x_pos = random.randint(0, self.world_width - 1)
        y_pos = random.randint(0, self.world_height - 1)
        theta_pos = 2 * math.pi * (random.random() - 1)
        self.state = (x_pos, y_pos, theta_pos)
        return self.state
    
    def render(self, mode='human'):
        screen_width = 60 * self.world_width
        screen_height = 60 * self.world_height
        carwidth = 40
        carheight = 20
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            clearance = 10
            
            l, r, t, b = -carwidth/2, carwidth/2, carheight/2, -carheight/2
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, 0)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            
            l_w, r_w, t_w, b_w = -carwidth/8, carwidth/8, carheight*5/8, -carheight*5/8
            frontwheel = rendering.FilledPolygon([(l_w, b_w), (l_w, t_w), (r_w, t_w), (r_w, b_w)])
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth/4, 0)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            
            backwheel = rendering.FilledPolygon([(l_w, b_w), (l_w, t_w), (r_w, t_w), (r_w, b_w)])
            backwheel.set_color(.5, .5, .5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth/4, 0)))
            backwheel.add_attr(self.cartrans)
            self.viewer.add_geom(backwheel)
            
            self.viewer.add_geom(car)
            
            for i in range(1, self.world_width):
                line = rendering.Line((60*i, 0), (60*i, 60*self.world_height))
                self.viewer.add_geom(line) 
            for i in range(1, self.world_height):
                line = rendering.Line((0, 60*i), (60*self.world_width, 60*i))
                self.viewer.add_geom(line) 
            
        #x_pos = screen_width / 2 + 60 * self.state[0]
        #y_pos = screen_height / 2 + 60 * self.state[1]
        x_pos = 30 + 60 * self.state[0]
        y_pos = 30 + 60 * self.state[1]
        theta_pos = self.state[2]
        self.cartrans.set_translation(x_pos, y_pos)
        self.cartrans.set_rotation(theta_pos)
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

