import random
import numpy as np
import tensorflow as tf
from replay import ReplayMemory

from collections import deque

class DQNAgent:
    def __init__(self, obs_dim, n_action, seed=0,
                 discount_factor = 0.995, epsilon_decay = 0.999, epsilon_min = 0.01,
                 learning_rate = 1e-3, # STEP SIZE
                 batch_size = 64, 
                 memory_size = 10000, hidden_unit_size = 64,
                 target_mode = 'DDQN', memory_mode = 'PER'):
        self.seed = seed 
        self.obs_dim = obs_dim
        self.n_action = n_action

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = 5000

        self.target_mode = target_mode
        self.memory_mode = memory_mode
        if memory_mode == 'PER':
            self.memory = ReplayMemory(memory_size=memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
            
        self.hidden_unit_size = hidden_unit_size
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.build_placeholders()
            self.build_model()
            self.build_loss()
            self.build_update_operation()
            self.init_session()        
    
    def build_placeholders(self):
        self.obs_ph = tf.placeholder(tf.float32, (None, self.obs_dim), 'obs')
        self.target_ph = tf.placeholder(tf.float32, (None, self.n_action), 'target')
        self.batch_weights_ph = tf.placeholder(tf.float32,(None, self.n_action), name="batch_weights")
        self.learning_rate_ph = tf.placeholder(tf.float32, (), 'lr')        
    
    def build_model(self):
        hid1_size = self.hidden_unit_size  # 10 empirically determined
        hid2_size = self.hidden_unit_size
        
        with tf.variable_scope('q_func'):
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            self.q_predict = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_predict')
                        
        with tf.variable_scope('q_func_old'):
            out = tf.layers.dense(self.obs_ph, hid1_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden1')
            out = tf.layers.dense(out, hid2_size, tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='hidden2')
            self.q_predict_old = tf.layers.dense(out, self.n_action,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01,seed=self.seed), name='q_predict')
        
        self.weights = tf.trainable_variables(scope='q_func')
        self.weights_old = tf.trainable_variables(scope='q_func_old')

    def build_loss(self):
        self.errors = self.target_ph - self.q_predict
        self.loss = 0.5*tf.reduce_mean(tf.square(self.target_ph - self.q_predict))
        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).minimize(self.loss)
            
    def build_update_operation(self):
        update_ops = []
        for var, var_old in zip(self.weights, self.weights_old):
            update_ops.append(var_old.assign(var))
        self.update_ops = update_ops
        
    def init_session(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config,graph=self.g)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.update_ops)
        self.saver = tf.train.Saver()
        
    def save_model(self, path):
        self.saver.save(self.sess, path)
        #with self.sess as sess:
        #    tf.train.Saver().save(sess, "model/coverage_6_model.ckpt")
        print("Model saved.")
        
    def restore_model(self, path):
        self.saver.restore(self.sess, path)
        #with self.sess as sess:
        #    tf.train.Saver().restore(sess, "model/coverage_6_model.ckpt")
        print("Model restored.")    
        
    def update_target(self):
        self.sess.run(self.update_ops)
    
    def update_memory(self, step, max_step):
        if self.memory_mode == 'PER':
            self.memory.anneal_per_importance_sampling(step, max_step)
        
    def update_policy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def get_prediction_old(self, obs): 
        q_value_old = self.sess.run(self.q_predict_old, feed_dict={self.obs_ph:obs})        
        return q_value_old
        
    def get_prediction(self, obs):
        q_value = self.sess.run(self.q_predict, feed_dict={self.obs_ph:obs})        
        return q_value
    
    def get_action(self, obs):
        if np.random.rand() <= self.epsilon:
            #return random.randint(0, 3)
            return random.randint(0, self.n_action - 1)
        else:
            q_value = self.get_prediction(obs)
            return np.argmax(q_value[0])

    def add_experience(self, obs, action, reward, next_obs, done):
        if self.memory_mode == 'PER':
            self.memory.save_experience(obs, action, reward, next_obs, done)
        else:
            self.memory.append((obs, action, reward, next_obs, done))

    def train_model(self):
        loss = np.nan
        
        if self.memory_mode == 'PER':
            n_entries = self.memory.memory.n_entries
        else:
            n_entries = len(self.memory)
            
        if n_entries > self.train_start:
            
            if self.memory_mode == 'PER':
                # PRIORITIZED EXPERIENCE REPLAY
                idx, priorities, w, mini_batch = self.memory.retrieve_experience(self.batch_size)
                batch_weights = np.transpose(np.tile(w, (self.n_action, 1)))
            else:
                mini_batch = random.sample(self.memory, self.batch_size)
                batch_weights = np.ones((self.batch_size,self.n_action))

            observations = np.zeros((self.batch_size, self.obs_dim))
            next_observations = np.zeros((self.batch_size, self.obs_dim))
            actions, rewards, dones = [], [], []

            for i in range(self.batch_size):
                observations[i] = mini_batch[i][0]
                actions.append(mini_batch[i][1])
                rewards.append(mini_batch[i][2])
                next_observations[i] = mini_batch[i][3]
                dones.append(mini_batch[i][4])

            target = self.get_prediction(observations)
            if self.target_mode == 'DDQN':
                bast_a = np.argmax(self.get_prediction(next_observations),axis=1)
            next_q_value = self.get_prediction_old(next_observations)

            # BELLMAN UPDATE RULE 
            for i in range(self.batch_size):
                if dones[i]:
                    target[i][actions[i]] = rewards[i]
                else:
                    if self.target_mode == 'DDQN':
                        target[i][actions[i]] = rewards[i] + self.discount_factor * next_q_value[i][bast_a[i]]
                    else:
                        target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(next_q_value[i]))

            loss, errors, _ = self.sess.run([self.loss, self.errors, self.optim], 
                                 feed_dict={self.obs_ph:observations,self.target_ph:target,self.learning_rate_ph:self.learning_rate,self.batch_weights_ph:batch_weights})
            errors = errors[np.arange(len(errors)), actions]
            
            if self.memory_mode == 'PER':
                # PRIORITIZED EXPERIENCE REPLAY
                self.memory.update_experience_weight(idx, errors)
            
        return loss  