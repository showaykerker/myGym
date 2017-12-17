"""
The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf
import random
from collections import deque
import matplotlib.pyplot as plt
import time

np.random.seed(1)
tf.set_random_seed(1)


class Color():
	W = '\033[0m'
	DR = '\033[31m'
	DG = '\033[32m'
	BG = '\033[1;32m'
	DY = '\033[33m'
	BY = '\033[1;33m'
	DB = '\033[34m'
	BB = '\033[1;34m'
	DP = '\033[35m'
	BP = '\033[1;35m'



class DuelingDQN:
	def __init__(
			self,
			n_actions,
			n_features,
			learning_rate=0.001,
			gamma=0.5,
			e_greedy=0.9,
			replace_target_iter=200,
			memory_size=500,
			batch_size=32,
			e_greedy_increment=0.0005,
			output_graph=False,
			dueling=True,
			sess=None,
	):
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = gamma
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.epsilon_increment = 0.0005
		self.epsilon = 0 

		self.dueling = dueling	  # decide to use dueling DQN or not

		self.learn_step_counter = 0
		self.memory = np.zeros((self.memory_size, n_features*2+2))
		self._build_net()
		t_params = tf.get_collection('target_net_params')
		e_params = tf.get_collection('eval_net_params')
		self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

		if sess is None:
			self.sess = tf.Session()
			self.sess.run(tf.global_variables_initializer())
		else:
			self.sess = sess
		if output_graph:
			tf.summary.FileWriter("logs/", self.sess.graph)
		self.cost_his = []
		self.saver = tf.train.Saver()


		# for plots
		self.fig = plt.figure()
		self.r_saver = []
		self.step_saver = []
		self.n_step_saver = deque(maxlen=100)
		self.n_r_saver = deque(maxlen=100)
		self.quartile_saver = [[],[],[]]
		self.total_time_step = 0
		self.action_taken=[0]*n_actions


	def _build_net(self):
		def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
			with tf.variable_scope('l1'):
				w1 = tf.get_variable('w1', [self.n_features, 128], initializer=w_initializer, collections=c_names)
				b1 = tf.get_variable('b1', [1, 128], initializer=b_initializer, collections=c_names)
				l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

			
			with tf.variable_scope('l2'):
				w2 = tf.get_variable('w2', [128, n_l1], initializer=w_initializer, collections=c_names)
				b2 = tf.get_variable('b2', [1, n_l1], initializer=b_initializer, collections=c_names)
				l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
			

			if self.dueling:
				# Dueling DQN
				with tf.variable_scope('Value'):
					w3 = tf.get_variable('w3', [n_l1, 1], initializer=w_initializer, collections=c_names)
					b3 = tf.get_variable('b3', [1, 1], initializer=w_initializer, collections=c_names)
					self.V = tf.nn.tanh(tf.matmul(l2, w3) + b3)


				with tf.variable_scope('Advantage'):
					w3 = tf.get_variable('w3', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
					b3 = tf.get_variable('b3', [1, self.n_actions], initializer=w_initializer, collections=c_names)
					self.A = tf.matmul(l2, w3) + b3

				with tf.variable_scope('Q'):
					out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))	 # Q = V(s) + A(s,a)
			else:
				with tf.variable_scope('Q'):
					#w2 = tf.get_variable('w2', [n_l1, 100], initializer=w_initializer, collections=c_names)
					#b2 = tf.get_variable('b2', [1, 100], initializer=b_initializer, collections=c_names)
					w3 = tf.get_variable('w3', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
					b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
					out = tf.matmul(l2, w3) + b3


			return out

		# ------------------ build evaluate_net ------------------
		self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
		self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
		with tf.variable_scope('eval_net'):
			c_names, n_l1, w_initializer, b_initializer = \
				['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
				tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

			self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
		with tf.variable_scope('train'):
			self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

		# ------------------ build target_net ------------------
		self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')	# input
		with tf.variable_scope('target_net'):
			c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

			self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

	def store_transition(self, s, a, r, s_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0
		transition = np.hstack((s, [a, r], s_))
		index = self.memory_counter % self.memory_size
		self.memory[index, :] = transition
		self.memory_counter += 1

	def choose_action(self, observation):
		observation = observation[np.newaxis, :]
		action = None
		if np.random.random() < self.epsilon:  # choosing action
			actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
			action = np.argmax(actions_value)
		else:
			if np.random.random() < 0.2:
				if np.random.random() > 0.5: action = self.n_actions-1
				else : action = 0
			else: action = random.randrange(1, self.n_actions-1)
		self.action_taken[action] += 1
		return action

	def learn(self):
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.replace_target_op)
			print('\ntarget_params_replaced\n')

		sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]

		q_next = self.sess.run(self.q_next, feed_dict={self.s_: batch_memory[:, -self.n_features:]}) # next observation
		q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

		q_target = q_eval.copy()

		batch_index = np.arange(self.batch_size, dtype=np.int32)
		eval_act_index = batch_memory[:, self.n_features].astype(int)
		reward = batch_memory[:, self.n_features + 1]

		q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

		_, self.cost = self.sess.run([self._train_op, self.loss],
									 feed_dict={self.s: batch_memory[:, :self.n_features],
												self.q_target: q_target})
		self.cost_his.append(self.cost)

		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		self.learn_step_counter += 1


	def append_data(self, ep_reward = None, ep_steps = None):
		if ep_reward is not None : 
			self.r_saver.append(ep_reward)
			self.n_r_saver.append(ep_reward)
			self.quartile_saver[0].append(np.percentile(self.n_r_saver, 25))
			self.quartile_saver[1].append(np.percentile(self.n_r_saver, 50))
			self.quartile_saver[2].append(np.percentile(self.n_r_saver, 75))
		if ep_steps is not None : 
			self.step_saver.append(ep_steps)
			self.total_time_step += ep_steps
			


	def plot(self):

		self.fig.suptitle(self.description)

		p = self.fig.add_subplot(3,1,1) 
		p.clear()
		p.set_yscale('symlog', basey=1000)
		p.set_ylabel('cost')
		p.grid(True)
		p.plot(self.cost_his, 'bo', ms=0.1)

		s = self.fig.add_subplot(3,1,2)
		s.clear()
		s.set_ylabel('reward per ep')
		#s.set_yscale('symlog', basey=10)
		s.grid(True)
		#s.plot(self.r_saver, 'yo', ms=0.2)
		s.plot(self.quartile_saver[0], 'g-', lw=0.4)
		s.plot(self.quartile_saver[1], 'b-', lw=0.4)
		s.plot(self.quartile_saver[2], 'r-', lw=0.4)

		q = self.fig.add_subplot(3,1,3)
		q.clear()
		q.set_ylabel('action taken')
		q.grid(True)
		x = range(len(self.action_taken))
		q.bar(x, self.action_taken, 0.2, color='green')
		plt.pause(0.01)


	def get_description(self):

		import os

		self.description = input(Color.BP+'Description: '+Color.W)
		self.test_name = input(Color.BP+'Test Name: '+Color.W)

		directory = 'log/' + self.test_name
		if not os.path.exists(directory): 
			os.makedirs(directory)
		if not os.path.exists(directory + '/models'): 
			os.makedirs(directory + '/models')
		if not os.path.exists(directory + '/plots'): 
			os.makedirs(directory + '/plots')
		
		#plot_model(self.Q_eval, to_file='log/'+self.test_name+'/'+self.description+'.png', show_shapes=True)



	def save_model(self, ep, apd):
		
		import os
		directory = 'log/' + self.test_name
		if not os.path.exists(directory): 
			os.makedirs(directory)
		if not os.path.exists(directory + '/models'): 
			os.makedirs(directory + '/models')
		if not os.path.exists(directory + '/plots'): 
			os.makedirs(directory + '/plots')

		import time

		while (not hasattr(self, 'test_name') or not hasattr(self, 'description')):
			time.sleep(0.5)

		time_ = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
		model_name = time_ + '_E%d' % (ep) + '_' + str(apd) + '.h5' 

		
		fig_name = time_ + '_E%d' % (ep) + '_' + str(apd) + '.png' 
		self.fig.savefig('log/' + self.test_name + '/plots/' + fig_name , dpi=self.fig.dpi)
		print( Color.BY + "Saving Figure: " + 'log/' +  self.test_name + '/plots/' + fig_name + Color.W )

		save_path = self.saver.save(self.sess, 'log/' + self.test_name + '/models/' + model_name)
		print( Color.BY + "Saving Model: " + save_path + Color.W )


