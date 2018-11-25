"""
The coverage path planning problem implemented by Jaegoo Choy et al.
"""
import random
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class Coverage_v1(gym.Env):
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

        self.dim = 8
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple([spaces.Discrete(self.dim * self.dim), 
                                               spaces.Box(low=-1.0, high=1.0, shape=(self.dim, self.dim))])
        self.seed()
        self.cover1 = False
        self.viewer = None
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        pos, map_curr = state
        x_pos = pos // self.dim
        y_pos = pos % self.dim
        col = False
        map_next = np.copy(map_curr)
        
        if action == 0:
            x_pos = x_pos - 1
        elif action == 1:
            y_pos = y_pos + 1
        elif action == 2:
            x_pos = x_pos + 1
        else:
            y_pos = y_pos - 1
                
        if map_curr[x_pos, y_pos] == -1.0:
            col = True
                
        done = col
        pos = self.dim * x_pos + y_pos
        map_next[x_pos, y_pos] = 1
        
        reward = -1
        if done:
            reward = -300.0
        elif np.sum(map_next) > np.sum(map_curr):
            reward = 20
        '''    
        if np.sum(map_next) == -16.0 and self.cover1 == False:
            self.cover1 = True
            print("half cover")
            reward = 500.0
        ''' 
        if np.sum(map_next) == self.dim * self.dim - 2 * (2 * 2 + 4 * self.dim - 4):
            print("full cover")
            reward = 1000.0
            done = True
        self.state = (pos, map_next)
        return self.state, reward, done, {}

    def reset(self):
        while 1:
            pos = random.randint(0, self.dim * self.dim - 1)     
            x_pos = pos // self.dim
            y_pos = pos % self.dim
            if x_pos == 0 or x_pos == self.dim - 1:
                continue
            if y_pos == 0 or y_pos == self.dim - 1:
                continue
            if x_pos > self.dim / 2 - 2 and x_pos < self.dim / 2 + 1:
                continue
            if y_pos > self.dim / 2 - 2 and y_pos < self.dim / 2 + 1:
                continue
            else:
                break
            
        map_curr = self.np_random.uniform(low=-0.0, high=0.0, size=(self.dim, self.dim))
        
        for x in range(self.dim / 2 - 1, self.dim / 2 + 1):
            for y in range(self.dim / 2 - 1, self.dim / 2 + 1):
                map_curr[x, y] = -1.0
            
        for x in range(0, self.dim):
            for y in range(0, self.dim):
                if x == 0 or x == self.dim - 1:
                    map_curr[x, y] = -1.0
                if y == 0 or y == self.dim - 1:
                    map_curr[x, y] = -1.0    
                    
        map_curr[x_pos, y_pos] = 1.0
        self.state = (pos, map_curr)
        self.cover1 = False
        return self.state
    
    def render(self, mode='human'):
        screen_width = 60 * self.dim
        screen_height = 60 * self.dim
        robot_radius = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
           
            self.cell_color = []
            
            for x in range(0, self.dim):
                for y in range(0, self.dim):
                    cell = rendering.FilledPolygon([(60 * x, 60 * y), (60 * (x + 1), 60 * y), 
                                                    (60 * (x + 1), 60 * (y + 1)), (60 * x, 60 * (y + 1))])
                    self.cell_color.append(cell.attrs[0])
                    self.viewer.add_geom(cell)         
            robot = rendering.make_circle(robot_radius)
            robot.set_color(1.0, 0.0, 0.0)
            self.robotrans = rendering.Transform()
            robot.add_attr(self.robotrans)
            self.viewer.add_geom(robot)
            
            for i in range(1, self.dim):
                line = rendering.Line((60 * i, 0), (60 * i, 60 * self.dim))
                self.viewer.add_geom(line) 
            for i in range(1, self.dim):
                line = rendering.Line((0, 60 * i), (60 * self.dim, 60 * i))
                self.viewer.add_geom(line) 

        pos = self.state[0]
        x_pos = 30 + 60 * (pos // self.dim)
        y_pos = 30 + 60 * (pos % self.dim)
        self.robotrans.set_translation(x_pos, y_pos)
        
        map_curr = self.state[1]
        
        for x in range(0, self.dim):
            for y in range(0, self.dim):
                transparency = (map_curr[x, y] + 1) / 2
                self.cell_color[self.dim * x + y].vec4 = ((transparency, transparency, transparency, 1))
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

