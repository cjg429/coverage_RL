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

        self.world_width = 10.0
        self.world_height = 10.0
        self.min_speed = 0.0
        self.max_speed = 1.0
        self.min_angle = -math.pi / 4
        self.max_angle = math.pi / 4
        
        self.act_low = np.array([self.min_speed, self.min_angle])
        self.act_high = np.array([self.max_speed, self.max_angle])
        self.action_space = spaces.Box(self.act_low, self.act_high)
        
        self.obs_low = np.array([-self.world_width, -self.world_height, -math.pi])
        self.obs_high = np.array([self.world_width, self.world_height, math.pi])
        self.observation_space = spaces.Tuple([spaces.Box(self.obs_low, self.obs_high), 
                                               spaces.Box(low=-1.0, high=1.0, shape=(int(self.world_width), int(self.world_height)))])
        
        self.seed()
        self.viewer = None
        #self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x_pos = state[0][0]
        y_pos = state[0][1]
        theta_pos = state[0][2]
        map_curr = state[1]
        map_next = map_curr
        
        col = False
        
        speed = action[0]
        angle = action[1]
        if speed > self.max_speed:
            speed = self.max_speed
        elif speed < self.min_speed:
            speed = self.min_speed
        
        if angle > self.max_angle:
            angle = self.max_angle
        elif angle < self.min_angle:
            angle = self.min_angle
            
        L = 1 / 2.0
        x_pos = x_pos + L / math.tan(angle) * (math.sin(theta_pos + speed * math.tan(angle) / L) - math.sin(theta_pos))
        y_pos = y_pos + L / math.tan(angle) * (math.cos(theta_pos) - math.cos(theta_pos + speed * math.tan(angle) / L))       
        theta_pos = theta_pos + speed / L * math.tan(angle)
        if theta_pos > math.pi:
            theta_pos = theta_pos - 2 * math.pi
        if theta_pos < -math.pi:
            theta_pos = theta_pos + 2 * math.pi
        
        if map_curr[int(x_pos), int(y_pos)] == -1.0:
            col = True
                
        done = col
        map_next[int(x_pos), int(y_pos)] = 1
        
        if done:
            reward = -50
        else:
            reward = -1
        
        if np.sum(map_next) == int(self.world_width) * int(self.world_height) - 2 * (2 * 2 + 2 * int(self.world_width) + 2 * int(self.world_height) - 4):
            #print("full cover")
            reward = 1000.0
            done = True
            
        self.state = ([x_pos, y_pos, theta_pos], map_next)
        return self.state, reward, done, {}

    def reset(self):
        while 1:
            x_pos = random.uniform(1.0, self.world_width - 1.0)
            y_pos = random.uniform(1.0, self.world_height - 1.0)
            theta_pos = random.uniform(-math.pi, math.pi)
            if x_pos > self.world_width / 2 - 2 and x_pos < self.world_width / 2 + 1:
                continue
            if y_pos > self.world_height / 2 - 2 and y_pos < self.world_height / 2 + 1:
                continue
            else:
                break
                
        map_curr = self.np_random.uniform(low=0.0, high=0.0, size=(int(self.world_width), int(self.world_height)))
        
        for x in range(int(self.world_width / 2) - 1, int(self.world_width / 2) + 1):
            for y in range(int(self.world_width / 2) - 1, int(self.world_width / 2) + 1):
                map_curr[x, y] = -1.0
            
        for x in range(0, int(self.world_width)):
            for y in range(0, int(self.world_height)):
                if x == 0 or x == int(self.world_width) - 1:
                    map_curr[x, y] = -1.0
                if y == 0 or y == int(self.world_height) - 1:
                    map_curr[x, y] = -1.0    
                    
        map_curr[int(x_pos), int(y_pos)] = 1.0                                  
                                          
        self.state = ([x_pos, y_pos, theta_pos], map_curr)
        return self.state
    
    def render(self, mode='human'):
        screen_width = 60 * int(self.world_width)
        screen_height = 60 * int(self.world_height)
        carwidth = 40
        carheight = 20
        
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            self.cell_color = []
            
            for x in range(0, int(self.world_width)):
                for y in range(0, int(self.world_height)):
                    cell = rendering.FilledPolygon([(60 * x, 60 * y), (60 * (x + 1), 60 * y), 
                                                    (60 * (x + 1), 60 * (y + 1)), (60 * x, 60 * (y + 1))])
                    self.cell_color.append(cell.attrs[0])
                    self.viewer.add_geom(cell)
            
            l, r, t, b = -carwidth / 2, carwidth / 2, carheight / 2, -carheight / 2
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, 0)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            
            l, r, t, m, b = 0, carwidth / 4, carheight / 2, 0, -carheight / 2
            front = rendering.FilledPolygon([(l, b), (r, m), (l, t)])
            front.set_color(0.0, 0.0, 1.0)
            front.add_attr(rendering.Transform(translation=(carwidth * 1 / 2 , 0)))
            front.add_attr(self.cartrans)
            self.viewer.add_geom(front)
            
            l_w, r_w, t_w, b_w = -carwidth / 8, carwidth / 8, carheight * 5 / 8, -carheight * 5 / 8
            frontwheel = rendering.FilledPolygon([(l_w, b_w), (l_w, t_w), (r_w, t_w), (r_w, b_w)])
            frontwheel.set_color(1.0, 0.0, 0.0)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, 0)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            
            backwheel = rendering.FilledPolygon([(l_w, b_w), (l_w, t_w), (r_w, t_w), (r_w, b_w)])
            backwheel.set_color(1.0, 0.0, 0.0)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, 0)))
            backwheel.add_attr(self.cartrans)
            self.viewer.add_geom(backwheel)
            
            self.viewer.add_geom(car)
            
            for i in range(1, int(self.world_width)):
                line = rendering.Line((60 * i, 0), (60 * i, 60 * int(self.world_height)))
                self.viewer.add_geom(line) 
            for i in range(1, int(self.world_height)):
                line = rendering.Line((0, 60 * i), (60 * int(self.world_width), 60 * i))
                self.viewer.add_geom(line) 
            
        x_pos = 60 * self.state[0][0]
        y_pos = 60 * self.state[0][1]
        theta_pos = self.state[0][2]
        self.cartrans.set_translation(x_pos, y_pos)
        self.cartrans.set_rotation(theta_pos)
        
        map_curr = self.state[1]
        
        for x in range(0, int(self.world_width)):
            for y in range(0, int(self.world_height)):
                transparency = (map_curr[x, y] + 1) / 2
                self.cell_color[int(self.world_width) * x + y].vec4 = ((transparency, transparency, transparency, 1))
           
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

