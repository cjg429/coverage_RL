"""
The coverage path planning problem implemented by Jaegoo Choy et al.
"""
import random
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CoverageObsEnv(gym.Env):
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

        self.dim = 4
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple([spaces.Discrete(self.dim*self.dim), spaces.Discrete(self.dim*self.dim), spaces.Box(low=-1.0, high=1.0, shape=(self.dim,self.dim))])
        self.seed()
        self.viewer = None
        #self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        pos, obs_pos, map_curr = state
        x_pos = pos // self.dim
        y_pos = pos % self.dim
        obs_x_pos = obs_pos // self.dim
        obs_y_pos = obs_pos % self.dim
        col = False
        map_next = map_curr
        
        if obs_x_pos == 0:
            if obs_y_pos == 0:
                obs_action = random.sample([1,2], 1)
            elif obs_y_pos == self.dim - 1:
                obs_action = random.sample([2,3], 1)
            else:
                obs_action = random.sample([1,2,3], 1)
        elif obs_y_pos == 0:
            if obs_x_pos == self.dim - 1:
                obs_action = random.sample([0,1], 1)
            else:    
                obs_action = random.sample([0,1,2], 1)
        elif obs_x_pos == self.dim - 1:
            if obs_y_pos == self.dim - 1:
                obs_action = random.sample([0,3], 1)
            else:
                obs_action = random.sample([0,1,3], 1)
        elif obs_y_pos == self.dim - 1:
            obs_action = random.sample([0,2,3], 1)
        else:
            obs_action = random.randint(0, 3)
        
        if action == 0 and obs_action == 2 and y_pos == obs_y_pos and x_pos == obs_x_pos + 1:
            col = True
        elif action == 2 and obs_action == 0 and y_pos == obs_y_pos and x_pos == obs_x_pos - 1:
            col = True
        elif action == 1 and obs_action == 3 and x_pos == obs_x_pos and y_pos == obs_y_pos - 1:
            col = True
        elif action == 3 and obs_action == 1 and x_pos == obs_x_pos and y_pos == obs_y_pos + 1:
            col = True
            
        if action == 0:
            x_pos = x_pos - 1
            if x_pos < 0:
                col = True
                x_pos = 0
        elif action == 1:
            y_pos = y_pos + 1
            if y_pos > self.dim - 1:
                col = True
                y_pos = self.dim - 1
        elif action == 2:
            x_pos = x_pos + 1
            if x_pos > self.dim - 1:
                col = True
                x_pos = self.dim - 1
        else:
            y_pos = y_pos - 1
            if y_pos < 0:
                col = True
                y_pos = 0
                
        if obs_action == 0:
            obs_x_pos = obs_x_pos - 1
            if obs_x_pos < 0:
                obs_x_pos = 0
        elif obs_action == 1:
            obs_y_pos = obs_y_pos + 1
            if obs_y_pos > self.dim - 1:
                obs_y_pos = self.dim - 1
        elif obs_action == 2:
            obs_x_pos = obs_x_pos + 1
            if obs_x_pos > self.dim - 1:
                obs_x_pos = self.dim - 1
        else:
            obs_y_pos = obs_y_pos -1
            if obs_y_pos < 0:
                obs_y_pos = 0
                
        if x_pos == obs_x_pos and y_pos == obs_y_pos:
            col = True
            
        done = col
        pos = self.dim * x_pos + y_pos
        obs_pos = self.dim * obs_x_pos + obs_y_pos
        map_next[x_pos, y_pos] = 1

        if not done:
            reward = -1
            
        else:
            reward = -1000.0
        if np.sum(map_next) == self.dim * self.dim:
            print("full cover")
            reward = 1000.0
            done = True
        self.state = (pos, obs_pos, map_next)
        return self.state, reward, done, {}

    def reset(self):
        pos = random.randint(0, self.dim * self.dim - 1)
        map_curr = self.np_random.uniform(low=-0.0, high=0.0, size=(self.dim,self.dim))
        x_pos = pos // self.dim
        y_pos = pos % self.dim
        map_curr[x_pos, y_pos] = 1.0
        obs_pos = random.randint(0, self.dim * self.dim - 1)
        while pos == obs_pos:
            obs_pos = random.randint(0, self.dim * self.dim - 1)
        self.state = (pos, obs_pos, map_curr)
        return self.state
    
    def render(self, mode='human'):
        screen_width = 60 * self.dim
        screen_height = 60 * self.dim
        robot_radius = 20
        obs_radius = 20
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
           
            self.cell_color = []
            for x in range(0, self.dim):
                for y in range(0, self.dim):
                    cell = rendering.FilledPolygon([(60*x,60*y), (60*(x+1),60*y), (60*(x+1),60*(y+1)), (60*x,60*(y+1))])
                    self.cell_color.append(cell.attrs[0])
                    self.viewer.add_geom(cell)    
                    
            robot = rendering.make_circle(robot_radius)
            robot.set_color(1.0, 0.0, 0.0)
            self.robotrans = rendering.Transform()
            robot.add_attr(self.robotrans)
            self.viewer.add_geom(robot)
            
            obs = rendering.make_circle(obs_radius)
            obs.set_color(0.0, 0.0, 1.0)
            self.obstrans = rendering.Transform()
            obs.add_attr(self.obstrans)
            self.viewer.add_geom(obs)
            
            for i in range(1, self.dim):
                line = rendering.Line((60*i, 0), (60*i, 60*self.dim))
                self.viewer.add_geom(line) 
            for i in range(1, self.dim):
                line = rendering.Line((0, 60*i), (60*self.dim, 60*i))
                self.viewer.add_geom(line) 

        pos = self.state[0]
        x_pos = 30 + 60 * (pos // self.dim)
        y_pos = 30 + 60 * (pos % self.dim)
        self.robotrans.set_translation(x_pos, y_pos)
        
        obs_pos = self.state[1]
        x_pos = 30 + 60 * (obs_pos // self.dim)
        y_pos = 30 + 60 * (obs_pos % self.dim)
        self.obstrans.set_translation(x_pos, y_pos)
        
        map_curr = self.state[2]    
        for x in range(0, self.dim):
            for y in range(0, self.dim):
                transparency = (map_curr[x, y] + 1) / 2
                self.cell_color[self.dim*x+y].vec4 = ((transparency, transparency, transparency, 1))
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

