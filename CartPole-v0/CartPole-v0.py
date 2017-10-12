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
                 epsilon_log_decay=0.9, alpha=0.03, alpha_decay=0.005, batch_size=64, monitor=False, quiet=False):
        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v2')
        if monitor: self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size
        self.quiet = quiet
        if max_env_steps is not None: self.env._max_episode_steps = max_env_steps

        # Init model
        self.model = Sequential()
        #self.model.add(Dense(72, input_dim=5, activation='tanh', kernel_initializer='Ones', bias_initializer='Zeros')) # Best try 6 episodes
        self.model.add(Dense(72, input_dim=5, activation='tanh', kernel_initializer='Ones',bias_initializer='Ones'))
        #self.model.add(Dense(72, input_dim=5, activation='tanh', kernel_initializer='glorot_normal', bias_initializer='glorot_normal')) # Best try 19 episodes
        #self.model.add(Dense(72, input_dim=5, activation='tanh', kernel_initializer='Ones', bias_initializer='glorot_normal'))# Work Well Almost Everytime
        self.model.add(Dense(144, activation='tanh', kernel_initializer='glorot_normal'))
        #self.model.add(Dense(144, activation='tanh', kernel_initializer='Ones'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

        self.target_net = clone_model(self.model)  # old network, used to evaluate actions
        self.target_net.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

        self.i_ = deque(maxlen=5)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(
            self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        state=np.insert(state, 4, sum(self.i_))
        return np.reshape(state, [1, 5])

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)

            y_target[0][action] = reward if done else reward + self.gamma * self.target_net.predict(next_state)[0][action]
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replace_target(self):
        print('replace_target!!')
        models_weights = self.model.get_weights()
        self.target_net.set_weights(models_weights)

    def run(self):

        train_target = 5

        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            #print('init:', state)
            done = False
            i = 0

            while not done:

                if i>=100 or i == 0: self.env.render()
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)

                self.i_.append(next_state[0][2])

                # x, x_dot, theta, theta_dot
                #reward -= abs(next_state[0][0]) # x

                #if abs(next_state[0][2]) <= abs(state[0][2]):
                #reward += 0.7*(abs(next_state[0][2])-abs(state[0][2])) # theta
                #else: reward -= 0.2
                #if abs(next_state[0][0]) <= abs(state[0][0]):
                reward += 0.45*(abs(next_state[0][0])-abs(state[0][0])) # x
                reward += 0.7*(abs(next_state[0][3])-abs(state[0][3])) # theta_dot
                reward -= 1.4*abs(next_state[0][4])
                #print(1.2*abs(next_state[0][4]))
                #else: reward -= 0.1
                if done: reward = -1

                #print(state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if i % 2000 == 0 and i != 0 and not done:
                    print('... Current Time Step', i)

            print('Ep No. %d, score %d' %(e+1, i))

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e+1 >= 100:
                if not self.quiet: print('Ran {} episodes. Solved after {} trials ✔'.format(e+1, e+1 - 100))
                input()
                return e - 100
            if (e+1) % 100 == 0 and not self.quiet:
                print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e+1, mean_score))


            self.replay(self.batch_size)

            if e + 1 > 1000: train_target = 250
            elif e + 1 > 800: train_target = 200
            elif e + 1 > 500: train_target = 100
            elif e + 1 > 300: train_target = 50
            elif e + 1 > 200: train_target = 35
            elif e + 1 > 100: train_target = 20
            elif e + 1 > 50: train_target = 10

            if (e+1) % train_target == 0:
                self.replace_target()

        if not self.quiet: print('Did not solve after {} episodes 😞'.format(e))
        return e


if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()