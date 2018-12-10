import random
import numpy as np
import tensorflow as tf
from replay import ReplayMemory

from collections import deque

class MENTAgent:
    def __init__(self, n_state, n_action, P, T = 30, l2 = 10,
                 seed=0, learning_rate = 1e-3, # STEP SIZE
                 batch_size = 64, gamma = 0.999, error = 0.01,
                 memory_size = 10000, hidden_unit_size = 64):
        self.seed = seed 
        self.n_state = n_state
        self.n_action = n_action
        self.P = P
        self.T = T
        self.l2 = l2
        self.gamma = gamma
        self.error = error

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
        self.input_s = tf.placeholder(tf.float32, (None, self.n_state), 'obs')
        self.latent = tf.placeholder(tf.float32, (None, 1), 'latent')
        self.grad_r = tf.placeholder(tf.float32, (None, self.n_action), 'gradr')
    
    def build_model(self):
        hid1_size = self.hidden_unit_size  # 10 empirically determined
        hid2_size = self.hidden_unit_size
        
        with tf.variable_scope('theta'):
            out = tf.layers.dense(tf.concat(axis=1, values=[self.input_s, self.latent]), hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            self.reward = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='reward')
        
        self.theta = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='theta')
        #self.weights = tf.trainable_variables(scope='theta')
        
    def build_loss(self):
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate) 
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.theta]) 
        self.grad_l2 = tf.gradients(self.l2_loss, self.theta) 
        self.grad_theta = tf.gradients(self.reward, self.theta, -self.grad_r) 
        self.grad_theta = [tf.add(self.l2 * self.grad_l2[i], self.grad_theta[i]) for i in range(len(self.grad_l2))] 
        self.grad_theta, _ = tf.clip_by_global_norm(self.grad_theta, 100.0) 
        self.grad_norms = tf.global_norm(self.grad_theta) 
        self.optimize = self.optimizer.apply_gradients(zip(self.grad_theta, self.theta)) 
        
    def apply_grads(self, feat_map, grad_r): 
        #grad_r = np.reshape(grad_r, [-1, 1]) 
        #feat_map = np.reshape(feat_map, [-1, self.n_state]) 
        _, grad_theta, l2_loss, grad_norms = self.sess.run([self.optimize, self.grad_theta, self.l2_loss, self.grad_norms],
                                                            feed_dict={self.grad_r: grad_r, self.input_s: feat_map}) 
        return grad_theta, l2_loss, grad_norms 
        
    def init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
    def save_model(self, path):
        self.saver.save(self.sess, path)
        print("Model saved.")
            
    def restore_model(self, path):
        self.saver.restore(self.sess, path)
        print("Model restored.")

    def get_rewards(self, states): 
        rewards = self.sess.run(self.reward, feed_dict={self.input_s: states}) 
        return rewards 
        
    def value_iteration(self):
        value = np.full([self.n_state], -1e10)
        #value = np.full([self.n_state], np.NINF)
        feat_map = np.identity(self.n_state)
        rewards = self.get_rewards(feat_map)
        
        while True:
            temp_value = np.copy(value)
            value[0] = 0
            value[self.n_state - 1] = 0
            q = np.zeros([self.n_state, self.n_action])
            for s in range(self.n_state):
                for a in range(self.n_action):
                    q[s, a] = sum([self.P[s, s1, a]*(rewards[s, a] + self.gamma * value[s1]) for s1 in range(self.n_state)])
                value[s] = max(q[s, :])
            if max([abs(value[s] - temp_value[s]) for s in range(self.n_state)]) < self.error:
                break

        policy = np.argmax(q, axis=1)
        return value, policy
  
    def demo_svf(self, trajs):
        '''
        p = np.zeros((self.n_state, 1))
        for traj in trajs:
            for step in traj:
                p[step.cur_state, 0] += 1
        p = p / len(trajs)
        '''
        p = np.zeros((self.n_state, self.n_action))
        for traj in trajs:
            for step in traj:
                p[step.cur_state, step.action] += 1
        p = p / len(trajs)
        return p
    
    def compute_state_visition_freq(self, trajs, policy):
        
        mu = np.zeros((self.n_state, self.T))
        for traj in trajs:
            mu[traj[0].cur_state, 0] += 1
        mu[:, 0] = mu[:, 0] / len(trajs)
        for t in range(self.T - 1):
            for s in range(self.n_state):
                mu[s, t + 1] = sum([mu[pre_s, t] * self.P[pre_s, s, int(policy[pre_s])] for pre_s in range(self.n_state)])
        p = np.sum(mu, 1) / self.T
        '''
        mu = np.zeros((self.n_state, self.n_action, self.T))
        for traj in trajs:
            mu[traj[0].cur_state, traj[0].action, 0] += 1
        mu[:, 0] = mu[:, 0] / len(trajs)
        for t in range(self.T - 1):
            for s in range(self.n_state):
                mu[s, int(policy[s]), t + 1] = sum([sum(mu[pre_s, :, t]) * self.P[pre_s, s, int(policy[pre_s])] for pre_s in range(self.n_state)])
        p = np.sum(mu, axis=2) / self.T
        '''
        return p

    def train_model(self, trajs, n_iters):
        feat_map = np.identity(self.n_state)
        
        mu_D = self.demo_svf(trajs)

        for iteration in range(n_iters):
            
            value, policy = self.value_iteration()
            mu_exp = self.compute_state_visition_freq(trajs, policy)
            mu_exp = mu_exp.reshape((-1, 1))
            grad_r = mu_D - np.tile(mu_exp, (1, 2))
            #grad_r = mu_D - mu_exp
            #grad_r = mu_D
            #grad_r = np.tile(grad_r, (1, 2))
            grad_theta, l2_loss, grad_norm = self.apply_grads(feat_map, grad_r)
        
        rewards = self.get_rewards(feat_map)
        print(rewards)
        print(policy)