import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential, clone_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import TruncatedNormal, glorot_normal, Ones, lecun_uniform, Zeros

class DQNCartPoleSolver():
    def __init__(self, n_episodes=10000, n_win_ticks=5000, max_env_steps=None, gamma=1.0, epsilon=1.0, epsilon_min=0.0,
                 epsilon_log_decay=0.9, alpha=0.03, alpha_decay=0.005, monitor=False):
        self.env = gym.make('CartPole-v2')
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.model = Sequential()
        self.model.add(Dense(72, input_dim=4, activation='tanh', kernel_initializer='Ones',bias_initializer='Ones'))
        self.model.add(Dense(144, activation='tanh', kernel_initializer='glorot_normal'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

    def choose_action(self, state):
        return_value = self.env.action_space.sample() if (np.random.random() <= self.epsilon) else np.argmax(self.model.predict(np.reshape(state, [1, 4])))
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min: self.epsilon = self.epsilon_min
        return return_value

    def fitting(self, state, action, reward, next_state, done):

        y_target = self.model.predict(np.reshape(state, [1, 4]))
        y_target[0][action] = reward if done else reward + self.gamma * self.model.predict(np.reshape(next_state, [1,4]))[0][action]


        self.model.fit(np.reshape(state,[1,4]), y_target, batch_size=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self):

        for e in range(self.n_episodes):

            state = np.reshape(self.env.reset(), [1,4])
            done = False
            i = 0

            while not done:

                if i>=100 or i == 0: self.env.render()
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                

                # Modify Reward
                #reward += 0.45*(abs(next_state[0][0])-abs(state[0][0])) # x
                #reward += 0.7*(abs(next_state[0][3])-abs(state[0][3])) # theta_dot
                #reward -= 1.4*abs(next_state[0][2])
                if done: reward = -1


                self.fitting(state, action, reward, next_state, done)
                state = next_state

                # Display on Cmd Line
                i += 1
                if i % 2000 == 0 and i != 0 and not done:
                    print('... Current Time Step', i)

            print('Ep No. %d, score %d' %(e+1, i))



if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()