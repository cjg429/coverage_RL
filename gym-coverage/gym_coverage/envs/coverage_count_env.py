"""
The coverage path planning problem implemented by Jaegoo Choy et al.
"""
import random
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class CoverageCountEnv(gym.Env):
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

        self.dim = 6
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple([spaces.Discrete(self.dim * self.dim), spaces.Box(low=-1.0, high=1.0, shape=(self.dim, self.dim))])       
        self.visit_count = np.zeros((256, 1), dtype=int)
        self.K = np.random.normal(0, 1, (8, 2 * self.dim * self.dim))
        self.beta = 10
        self.cover1 = False
        self.cover2 = False
        self.cover2 = False
        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, count=False):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        pos, map_curr = state
        x_pos = pos // self.dim
        y_pos = pos % self.dim
        col = False
        map_next = map_curr
            
        if count == True:
            map_col = map_curr.reshape((self.dim * self.dim, 1))
            pos_col = np.eye(self.dim * self.dim)[pos].reshape((self.dim * self.dim, 1))
            state_cat = np.concatenate((pos_col, map_col), axis=0)
            hash_out = np.sign(np.matmul(self.K, state_cat))
            binary = 0
            for i in range(0, 8):
                if hash_out[i] >= 0:
                    binary = (binary + 1)
                binary = binary * 2    
            binary = binary / 2
            self.visit_count[binary] += 1
            count = self.visit_count[binary]
        reward = -1
        
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
            y_pos = y_pos -1
            if y_pos < 0:
                col = True
                y_pos = 0
                
        done = col
        pos = self.dim * x_pos + y_pos
        np.clip(map_next, 0, 1, out=map_next)
        map_next[x_pos, y_pos] = 1
        
        if done:
            reward = -1000.0
        
        if np.sum(map_next) > np.sum(map_curr):
            reward += np.sum(map_next) * 10
        '''     
        if np.sum(map_next) == self.dim * self.dim / 4 and self.cover1 == False:
            reward = 250.0
            self.cover1 = True
            #print("1/4 cover")
        elif np.sum(map_next) == self.dim * self.dim * 2 / 4 and self.cover2 == False:
            reward = 500.0
            self.cover2 = True
            #print("1/2 cover")
        elif np.sum(map_next) == self.dim * self.dim * 3 / 4 and self.cover3 == False:
            reward = 750.0
            self.cover3 = True
            #print("3/4 cover")
        ''' 
        if np.sum(map_next) == self.dim * self.dim:
            print("full cover")
            reward = 1000.0
            done = True
        
        if count == True:
            reward += self.beta / math.sqrt(count)
        self.state = (pos, map_next)
        return self.state, reward, done, {}

    def reset(self):
        pos = random.randint(0, self.dim * self.dim - 1)
        map_curr = self.np_random.uniform(low=-0.0, high=0.0, size=(self.dim, self.dim))
        x_pos = pos // self.dim
        y_pos = pos % self.dim
        map_curr[x_pos, y_pos] = 1.0
        self.cover1 = False
        self.cover2 = False
        self.cover3 = False
        self.state = (pos, map_curr)
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
                    cell = rendering.FilledPolygon([(60*x,60*y), (60*(x+1),60*y), (60*(x+1),60*(y+1)), (60*x,60*(y+1))])
                    self.cell_color.append(cell.attrs[0])
                    self.viewer.add_geom(cell)         
            robot = rendering.make_circle(robot_radius)
            robot.set_color(1.0, 0.0, 0.0)
            self.robotrans = rendering.Transform()
            robot.add_attr(self.robotrans)
            self.viewer.add_geom(robot)
            
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
        
        map_curr = self.state[1]
        
        for x in range(0, self.dim):
            for y in range(0, self.dim):
                transparency = (map_curr[x, y] + 1) / 2
                self.cell_color[self.dim*x+y].vec4 = ((transparency, transparency, transparency, 1))
        
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

