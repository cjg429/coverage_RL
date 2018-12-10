import random
import numpy as np
import tensorflow as tf
import torch
from torch.autograd import Variable

from collections import deque

class GCLAgent:
    def __init__(self, n_state, n_action, gamma = 0.999,
                 seed = 0, learning_rate = 1e-3, # STEP SIZE
                 batch_size = 64, memory_size = 10000, hidden_unit_size = 64):
        self.seed = seed 
        self.n_state = n_state
        self.n_action = n_action
        self.gamma = gamma

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_start = 0

        self.memory = deque(maxlen=memory_size)
            
        self.hidden_unit_size = hidden_unit_size
        
        self.build_model()
    
    def build_model(self):
        hid1_size = self.hidden_unit_size  # 10 empirically determined
        hid2_size = self.hidden_unit_size
        
        tanh = torch.nn.Tanh()
        
        self.fc1 = torch.nn.Linear(self.n_state, hid1_size, bias=True)
        self.fc2 = torch.nn.Linear(hid1_size, hid2_size, bias=True)
        self.fc3 = torch.nn.Linear(hid2_size, self.n_action, bias=True)
        
        self.reward = torch.nn.Sequential(self.fc1, tanh, self.fc2, tanh, self.fc3)
            
    def get_reward(self, obs, action):
        obs_pos = torch.from_numpy(obs).float()
        obs_pos = Variable(obs_pos)
        action_pos = torch.from_numpy(action).float()
        action_pos = Variable(action_pos)
        reward = torch.mul(self.reward(obs_pos), action_pos)
        return reward
    
    def add_experience(self, trajectory):
        for i in range(len(trajectory)):
            self.memory.append(trajectory[i])
        #mini_batch = random.sample(self.memory, self.batch_size)
        #print(mini_batch)

    def train_model(self):
        output = np.nan
            
        n_entries = len(self.memory)
            
        if n_entries > self.train_start:
            mini_batch = random.sample(self.memory, self.batch_size)
        
        cost = 0
        traj_states = []
        traj_actions = []
        for i in range(self.batch_size):
            for step in mini_batch[i]:
                traj_states.append(np.eye(self.n_state)[step.cur_state])
                traj_actions.append(np.eye(self.n_action)[step.action])
                #cost += self.get_reward(np.eye(self.n_state)[step.cur_state], step.action)
        traj_states = np.array(traj_states)
        traj_actions = np.array(traj_actions)
        reward = self.get_reward(traj_states, traj_actions)
        reward = -torch.sum(reward)
        
        loss = torch.nn.MSELoss()
        zero_variable =  Variable(torch.zeros([1]).float())
        output = 0.5 * loss(reward, zero_variable)
        optimizer = torch.optim.Adam(self.reward.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        output.backward()
        optimizer.step()
            
        output = output.data.numpy()    
        
        all_state = np.identity(self.n_state)
        all_state = torch.from_numpy(all_state).float()
        all_state = Variable(all_state)
        all_reward = self.reward(all_state)
        print(all_reward)
        return output