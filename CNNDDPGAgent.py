import random
import numpy as np
import tensorflow as tf
from replay import ReplayMemory

from collections import deque

class CNNDDPGAgent:
    def __init__(self, pos_dim, map_dim, act_dim, seed=0,
                 discount_factor = 0.995, epsilon_decay = 0.999, epsilon_min = 0.01,
                 learning_rate = 1e-3, # STEP SIZE
                 batch_size = 64, tau = 0.5,
                 memory_size = 10000, hidden_unit_size = 64, filter_size = 32,
                 memory_mode = 'PER'):
        self.seed = seed 
        self.pos_dim = pos_dim
        self.map_dim = map_dim
        self.obs_dim = pos_dim + map_dim * map_dim
        self.act_dim = act_dim

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = 5000
        self.keep_conv = 0.8
        self.tau = tau

        self.memory_mode = memory_mode
        if memory_mode == 'PER':
            self.memory = ReplayMemory(memory_size=memory_size)
        else:
            self.memory = deque(maxlen=memory_size)
            
        self.hidden_unit_size = hidden_unit_size
        self.filter_size = filter_size
        
        self.g = tf.Graph()
        with self.g.as_default():
            self.build_placeholders()
            self.build_model()
            self.build_loss()
            self.build_update_operation()
            self.init_session()        
    
    def build_placeholders(self):
        self.pos_ph = tf.placeholder(tf.float32, (None, self.pos_dim), 'pose')
        self.map_ph = tf.placeholder(tf.float32, (None, self.map_dim, self.map_dim, 1), 'gridmap')
        self.act_ph = tf.placeholder(tf.float32, (None, self.act_dim), 'action')
        self.target_ph = tf.placeholder(tf.float32, (None, 1), 'target')
        self.batch_weights_ph = tf.placeholder(tf.float32,(None, 1), name="batch_weights")
        self.learning_rate_ph = tf.placeholder(tf.float32, (), 'lr')        
    
    def build_model(self):
        fit_size = self.filter_size
        hid1_size = self.hidden_unit_size  # 10 empirically determined
        hid2_size = self.hidden_unit_size
        
        with tf.variable_scope('source_network'):
            map_out = tf.layers.conv2d(self.map_ph, fit_size, 3, padding='same', activation=tf.tanh,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='conv_hidden1')
            map_out = tf.layers.max_pooling2d(map_out, 2, 2, name='conv_pool1')
            map_out = tf.layers.dropout(map_out, rate=self.keep_conv, name='conv_dropout1') 
            map_out = tf.reshape(map_out, [-1, self.map_dim * self.map_dim * self.filter_size / 4]) 
            map_out = tf.layers.dense(map_out, hid1_size, activation=tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='conv_hidden2')        
            
            out = tf.layers.dense(self.pos_ph, hid1_size, activation=tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='hidden1')
            out = tf.concat(axis=1, values=[out, map_out])
            out = tf.layers.dense(out, hid2_size,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='hidden2')
            
            act_out = tf.layers.dense(self.act_ph, hid2_size,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='act_hidden')
                                      
            self.q_predict_source = tf.layers.dense(tf.tanh(tf.add(out, act_out)), 1,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='q_predict')
            
            critic_out = tf.layers.dense(tf.tanh(out), hid1_size, activation=tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='actor_hidden')
            
            self.critic_source = tf.layers.dense(critic_out, self.act_dim,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='critic_out')
                        
        with tf.variable_scope('target_network'):
            map_out = tf.layers.conv2d(self.map_ph, fit_size, 3, padding='same', activation=tf.tanh,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='conv_hidden1')
            map_out = tf.layers.max_pooling2d(map_out, 2, 2, name='conv_pool1')
            map_out = tf.layers.dropout(map_out, rate=self.keep_conv, name='conv_dropout1') 
            map_out = tf.reshape(map_out, [-1, self.map_dim * self.map_dim * self.filter_size / 4]) 
            map_out = tf.layers.dense(map_out, hid1_size, activation=tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='conv_hidden2')
            
            out = tf.layers.dense(self.pos_ph, hid1_size, activation=tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='hidden1')
            out = tf.concat(axis=1, values=[out, map_out])
            out = tf.layers.dense(out, hid2_size,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='hidden2')
            
            act_out = tf.layers.dense(self.act_ph, hid2_size,
                                      kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='act_hidden')
                                      
            self.q_predict_target = tf.layers.dense(tf.tanh(tf.add(out, act_out)), 1,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='q_predict')
            
            critic_out = tf.layers.dense(tf.tanh(out), hid1_size, activation=tf.tanh,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='actor_hidden')
            
            self.critic_target = tf.layers.dense(critic_out, self.act_dim,
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01, seed=self.seed), name='critic_out')
        
        self.weights_source = tf.trainable_variables(scope='source_network')
        self.weights_target = tf.trainable_variables(scope='target_network')

    def build_loss(self):
        self.errors = self.target_ph - self.q_predict_source
        self.loss = 0.5 * tf.reduce_mean(tf.square(self.target_ph - self.q_predict_source))
        self.optim_critic = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).minimize(self.loss)
        self.q_gradient_input = tf.gradients(self.q_predict_source, self.act_ph)
        self.parameters_gradients = tf.gradients(self.critic_source, self.weights_source)
        self.optim_actor = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph).apply_gradients(zip(self.parameters_gradients, self.weights_source))

    def build_update_operation(self):
        update_ops = []
        for var_source, var_target in zip(self.weights_source, self.weights_target):
            update_ops.append(var_target.assign(self.tau * var_source + (1 - self.tau) * var_target))
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
        print("Model saved.")
        
    def restore_model(self, path):
        self.saver.restore(self.sess, path)
        print("Model restored.")    
        
    def update_target(self):
        self.sess.run(self.update_ops)
    
    def update_memory(self, step, max_step):
        if self.memory_mode == 'PER':
            self.memory.anneal_per_importance_sampling(step, max_step)
        
    def update_policy(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def get_prediction_target(self, obs, act):
        pos_obs = obs[:, 0:self.pos_dim]
        map_obs = obs[:, self.pos_dim:obs.shape[1]].reshape((-1, self.map_dim, self.map_dim, 1))
        q_value_old = self.sess.run(self.q_predict_target, 
                                    feed_dict={self.pos_ph:pos_obs, self.map_ph:map_obs, self.act_ph: act})        
        return q_value_old
        
    def get_prediction_source(self, obs, act):
        pos_obs = obs[:, 0:self.pos_dim]
        map_obs = obs[:, self.pos_dim:obs.shape[1]].reshape((-1, self.map_dim, self.map_dim, 1))
        q_value = self.sess.run(self.q_predict_source, 
                                feed_dict={self.pos_ph:pos_obs, self.map_ph:map_obs, self.act_ph: act})    
        return q_value
    
    def get_action(self, obs):
        pos_obs = obs[:, 0:self.pos_dim]
        map_obs = obs[:, self.pos_dim:obs.shape[1]].reshape((-1, self.map_dim, self.map_dim, 1))
        action_source = self.sess.run(self.critic_source, feed_dict={self.pos_ph:pos_obs, self.map_ph:map_obs})  
        return action_source
    
    def get_action_target(self, obs):
        pos_obs = obs[:, 0:self.pos_dim]
        map_obs = obs[:, self.pos_dim:obs.shape[1]].reshape((-1, self.map_dim, self.map_dim, 1))
        action_target = self.sess.run(self.critic_target, feed_dict={self.pos_ph:pos_obs, self.map_ph:map_obs})  
        return action_target
    
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
                batch_weights = np.transpose(w).reshape(-1, 1)
            else:
                mini_batch = random.sample(self.memory, self.batch_size)
                batch_weights = np.ones((self.batch_size, 1))

            observations = np.zeros((self.batch_size, self.obs_dim))
            next_observations = np.zeros((self.batch_size, self.obs_dim))
            actions, rewards, dones = [], [], []

            for i in range(self.batch_size):
                observations[i] = mini_batch[i][0]
                actions.append(mini_batch[i][1])
                rewards.append(mini_batch[i][2])
                next_observations[i] = mini_batch[i][3]
                dones.append(mini_batch[i][4])

            target = self.get_prediction_source(observations, actions)
            next_actions = self.get_action_target(next_observations)
            next_q_value = self.get_prediction_target(next_observations, next_actions)

            # BELLMAN UPDATE RULE 
            for i in range(self.batch_size):
                if dones[i]:
                    target[i] = rewards[i]
                else:
                    target[i] = rewards[i] + self.discount_factor * next_q_value[i]
            
            pos_obs = observations[:, 0:self.pos_dim]
            map_obs = observations[:, self.pos_dim:observations.shape[1]].reshape((-1, self.map_dim, self.map_dim, 1))
            loss, errors, _, _ = self.sess.run([self.loss, self.errors, self.optim_critic, self.optim_actor], 
                                 feed_dict={self.pos_ph:pos_obs, self.map_ph:map_obs, self.target_ph:target, self.act_ph: actions, self.learning_rate_ph:self.learning_rate, self.batch_weights_ph:batch_weights})
           
            #errors = errors[np.arange(len(errors)), actions]
            
            if self.memory_mode == 'PER':
                # PRIORITIZED EXPERIENCE REPLAY
                self.memory.update_experience_weight(idx, errors)
                
            
        return loss  