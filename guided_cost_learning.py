import random
import numpy as np
import tensorflow as tf

from collections import deque

class GCLAgent:
    def __init__(self, n_state, n_action, gamma = 0.999,
                 seed=0, learning_rate = 1e-3, # STEP SIZE
                 batch_size = 64, 
                 memory_size = 10000, hidden_unit_size = 64):
        self.seed = seed 
        self.n_state = n_state
        self.n_action = n_action
        self.gamma = gamma

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_start = 5

        self.memory = deque(maxlen=memory_size)
            
        self.hidden_unit_size = hidden_unit_size
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.build_placeholders()
            self.build_model()
            self.build_loss()
            self.init_session()        
            
    def build_placeholders(self):
        self.states_ph = tf.placeholder(tf.float32, (None, self.n_state), 'states')
        self.actions_ph = tf.placeholder(tf.float32, (None, self.n_action), 'actions')
        self.learning_rate_ph = tf.placeholder(tf.float32, (), 'lr') 
    
    def build_model(self):
        hid1_size = self.hidden_unit_size  # 10 empirically determined
        hid2_size = self.hidden_unit_size
        
        with tf.variable_scope('reward'):
            out = tf.layers.dense(self.states_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            self.reward = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='reward')
        
        self.weights = tf.trainable_variables(scope='reward')
        
        '''with tf.variable_scope('dynamics'):
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            self.dynamics_predict = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='dynamics_predict')
        
        self.weights = tf.trainable_variables(scope='dynamics')'''
        
    def init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
    def build_loss(self):
        self.reward_total = tf.multiply(self.reward, self.actions_ph)
        self.reward_sum = tf.reduce_sum(self.reward_total)
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).minimize(-self.reward_sum)
        
    def save_model(self, path):
        self.saver.save(self.sess, path)
        print("Model saved.")
            
    def restore_model(self, path):
        self.saver.restore(self.sess, path)
        print("Model restored.")
    
    def get_rewards(self, states):
        reward_predict = self.sess.run(self.reward, feed_dict={self.states_ph:states})
        return reward_predict
    
    def add_experience(self, trajectory):
        for i in range(len(trajectory)):
            self.memory.append(trajectory[i])      
        
    def train_model(self):
        loss = np.nan
        
        n_entries = len(self.memory)
        if n_entries > self.train_start:
            mini_batch = random.sample(self.memory, self.batch_size)
        
        traj_states = []
        traj_actions = []
        for i in range(self.batch_size):
            for step in mini_batch[i]:
                traj_states.append(np.eye(self.n_state)[step.cur_state])
                traj_actions.append(np.eye(self.n_action)[step.action])
        traj_states = np.array(traj_states)
        traj_actions = np.array(traj_actions)
        
        reward_sum, _ = self.sess.run([self.reward_sum, self.optim], 
                                      feed_dict={self.states_ph:traj_states, self.actions_ph:traj_actions, 
                                                 self.learning_rate_ph:self.learning_rate})
        
        all_states = np.identity(self.n_state)
        all_rewards = self.get_rewards(all_states)
        return reward_sum, all_rewards