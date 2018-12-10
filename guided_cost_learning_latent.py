import random
import numpy as np
import tensorflow as tf

from collections import deque

class GCLAgent:
    def __init__(self, n_state, n_action, gamma = 0.999, T = 15,
                 seed=0, learning_rate = 1e-3, # STEP SIZE
                 batch_size = 64, 
                 memory_size = 10000, hidden_unit_size = 64):
        self.seed = seed 
        self.n_state = n_state
        self.n_action = n_action
        self.T = T
        self.gamma = gamma

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_start = 5

        self.memory = deque(maxlen=memory_size)
        self.rand_memory = deque(maxlen=memory_size)
            
        self.hidden_unit_size = hidden_unit_size
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.build_placeholders()
            self.build_model()
            self.build_loss()
            self.init_session()        
            
    def build_placeholders(self):
        self.states_ph = tf.placeholder(tf.float32, (None, self.n_state), 'states')
        self.rand_states_ph = tf.placeholder(tf.float32, (None, self.n_state), 'rand_states')
        self.latents_ph = tf.placeholder(tf.float32, (None, 1), 'latents')
        self.rand_latents_ph = tf.placeholder(tf.float32, (None, 1), 'rand_latents')
        self.weights_ph = tf.placeholder(tf.float32, (), 'weights')
        self.rand_weights_ph = tf.placeholder(tf.float32, (), 'rand_weights')
        self.actions_ph = tf.placeholder(tf.float32, (None, self.n_action), 'actions')
        self.rand_actions_ph = tf.placeholder(tf.float32, (None, self.n_action), 'rand_actions')
        self.learning_rate_ph = tf.placeholder(tf.float32, (), 'lr') 
    
    def build_model(self):
        hid1_size = self.hidden_unit_size  # 10 empirically determined
        hid2_size = self.hidden_unit_size
        
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_state + 1, hid1_size], stddev=0.01, seed=self.seed)),
            'h2': tf.Variable(tf.random_normal([hid1_size, hid2_size], stddev=0.01, seed=self.seed)),
            'out': tf.Variable(tf.random_normal([hid2_size, self.n_action], stddev=0.01, seed=self.seed))
        }

        self.biases = {
            'b1': tf.Variable(tf.random_normal([hid1_size], stddev=0.01, seed=self.seed)),
            'b2': tf.Variable(tf.random_normal([hid2_size], stddev=0.01, seed=self.seed)),
            'out': tf.Variable(tf.random_normal([self.n_action], stddev=0.01, seed=self.seed))
        }
        
        with tf.variable_scope('cost'):
            x = tf.concat(axis=1, values=[self.states_ph, self.latents_ph])
            out_x = tf.tanh(tf.add(tf.matmul(x, self.weights['h1']), self.biases['b1']))
            out_x = tf.tanh(tf.add(tf.matmul(out_x, self.weights['h2']), self.biases['b2']))
            self.cost = tf.matmul(out_x, self.weights['out']) + self.biases['out']
            
            y = tf.concat(axis=1, values=[self.rand_states_ph, self.rand_latents_ph])
            out_y = tf.tanh(tf.add(tf.matmul(y, self.weights['h1']), self.biases['b1']))
            out_y = tf.tanh(tf.add(tf.matmul(out_y, self.weights['h2']), self.biases['b2']))
            self.rand_cost = tf.matmul(out_y, self.weights['out']) + self.biases['out']
            
        self.weights = tf.trainable_variables(scope='cost')
        
    def init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
    def build_loss(self):
        cost = tf.multiply(self.cost, self.actions_ph)
        self.cost_sum = tf.reduce_sum(cost) * self.weights_ph
        rand_cost_reshape = tf.reshape(self.rand_cost, [-1, self.T, self.n_action])
        rand_action_reshape = tf.reshape(self.rand_actions_ph, [-1, self.T, self.n_action])
        rand_cost = tf.reduce_sum(tf.reduce_sum(tf.multiply(rand_cost_reshape, rand_action_reshape), axis=2), axis=1)
        self.cost_sum = self.cost_sum + tf.reduce_logsumexp(-rand_cost) + tf.log(self.rand_weights_ph)
        #tf.multiply(self.rand_cost, self.rand_actions_ph)
        #self.rand_cost_total = tf.redeuce_sum(self.rand_cost_total, axis=1)
        #self.rand_cost_sum = tf.
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).minimize(self.cost_sum)
        
    def save_model(self, path):
        self.saver.save(self.sess, path)
        print("Model saved.")
            
    def restore_model(self, path):
        self.saver.restore(self.sess, path)
        print("Model restored.")
    
    def get_costs(self, states, latents):
        cost_predict = self.sess.run(self.cost, feed_dict={self.states_ph:states, self.latents_ph:latents})
        return cost_predict
    
    def add_experience(self, trajectory, rand_trajectory):
        for i in range(len(trajectory)):
            self.memory.append(trajectory[i])   
        for i in range(len(rand_trajectory)):
            self.rand_memory.append(rand_trajectory[i])    
        
    def train_model(self):
        loss = np.nan
        
        n_entries = len(self.memory)
        if n_entries > self.train_start:
            mini_batch = random.sample(self.memory, self.batch_size)
        
        traj_states = []
        traj_actions = []
        traj_latents = []
        for i in range(self.batch_size):
            for step in mini_batch[i]:
                traj_states.append(np.eye(self.n_state)[step.cur_state])
                traj_actions.append(np.eye(self.n_action)[step.action])
                traj_latents.append(step.encode)
        traj_states = np.array(traj_states)
        traj_actions = np.array(traj_actions)
        traj_latents = np.array(traj_latents).reshape((-1, 1))        
        traj_weights = 1.0 / self.batch_size
        
        rand_traj_states = np.zeros((len(self.rand_memory), self.T, self.n_state))
        rand_traj_actions = np.zeros((len(self.rand_memory), self.T, self.n_action))
        rand_traj_latents = np.zeros((len(self.rand_memory), self.T, 1))
        for i in range(len(self.rand_memory)):
            j = 0
            for step in self.rand_memory[i]:
                rand_traj_states[i, j, :] = np.eye(self.n_state)[step.cur_state]
                rand_traj_actions[i, j, :] = np.eye(self.n_action)[step.action]
                rand_traj_latents[i, j, :] = 0.0
        '''
        rand_traj_states = np.zeros((2 * len(self.rand_memory), self.T, self.n_state))
        rand_traj_actions = np.zeros((2 * len(self.rand_memory), self.T, self.n_action))
        rand_traj_latents = np.zeros((2 * len(self.rand_memory), self.T, 1))
        for i in range(len(self.rand_memory)):
            j = 0
            for step in self.rand_memory[i]:
                rand_traj_states[2 * i, j, :] = np.eye(self.n_state)[step.cur_state]
                rand_traj_actions[2 * i, j, :] = np.eye(self.n_action)[step.action]
                rand_traj_latents[2 * i, j, :] = 0.0
                rand_traj_states[2 * i + 1, j, :] = np.eye(self.n_state)[step.cur_state]
                rand_traj_actions[2 * i + 1, j, :] = np.eye(self.n_action)[step.action]
                rand_traj_latents[2 * i + 1, j, :] = 1.0
        '''
        rand_traj_states = rand_traj_states.reshape((-1, self.n_state))
        rand_traj_actions = rand_traj_actions.reshape((-1, self.n_action))
        rand_traj_latents = rand_traj_latents.reshape((-1, 1))
        rand_traj_weights = 1.0 / len(self.rand_memory)
        '''
        for i in range(self.batch_size):
            for step in self.rand_memory[i]:
                traj_states.append(np.eye(self.n_state)[step.cur_state])
                traj_actions.append(np.eye(self.n_action)[step.action])
                traj_latents.append(1.0)
                traj_weights.append(-1.0)
        '''       
        
        cost_sum, _ = self.sess.run([self.cost_sum, self.optim], 
                                      feed_dict={self.states_ph:traj_states, self.latents_ph:traj_latents,
                                                 self.rand_states_ph:rand_traj_states, self.rand_latents_ph:rand_traj_latents,
                                                 self.weights_ph:traj_weights, self.rand_weights_ph:rand_traj_weights,
                                                 self.actions_ph:traj_actions, self.rand_actions_ph:rand_traj_actions, 
                                                 self.learning_rate_ph:self.learning_rate})
        
        all_states = np.identity(self.n_state)
        all_latents = np.zeros((self.n_state, 1))
        #all_states = np.concatenate((all_states, all_latents), axis=1)
        all_rewards = self.get_costs(all_states, all_latents)
        return cost_sum, all_rewards