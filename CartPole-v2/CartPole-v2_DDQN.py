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
    def __init__(self, n_episodes=10000, n_win_ticks=5000, gamma=1.0, epsilon=1.0, epsilon_min=0.0,
                 epsilon_log_decay=0.9, alpha=0.03, alpha_decay=0.005, batch_size=64):

        self.memory = deque(maxlen=100000)
        self.env = gym.make('CartPole-v2')
        self.env = gym.wrappers.Monitor(self.env, '../data/cartpole-1', force=True)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_log_decay
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.n_episodes = n_episodes
        self.n_win_ticks = n_win_ticks
        self.batch_size = batch_size

        # Init model
        self.model = Sequential()
        self.model.add(Dense(72, input_dim=4, activation='tanh', kernel_initializer='Ones',bias_initializer='Ones'))
        self.model.add(Dense(144, activation='tanh', kernel_initializer='glorot_normal'))
        self.model.add(Dense(2, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))

        # This NN's parameters (or called neuron, weight) will freeze for a while 
        self.target_net = clone_model(self.model)  # old network, used to evaluate actions ( argmaxQ term )
        self.target_net.compile(loss='mse', optimizer=Adam(lr=self.alpha, decay=self.alpha_decay))


    def remember(self, state, action, reward, next_state, done):
        # save the tuple in the memory queue
        self.memory.append((state, action, reward, next_state, done))


    def choose_action(self, state, epsilon):
        # Using Epsilon-Greedy Method
        if np.random.random() <= epsilon : 
            return self.env.action_space.sample() 
        else: 
            return np.argmax(self.model.predict(state))


    def get_epsilon(self, t):
        # Epsilon decay because we want our agent to explore more in the beginning 
        # and perfom more stable in late episodes
        # The math below is to make decay of epsilon smoother
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))


    def preprocess_state(self, state):
        # If you want to add a I term from PID or something, 
        # You can do it here, then modify structure of model in __init__
        return np.reshape(state, [1, 4])


    def replay(self, batch_size):

        x_batch, y_batch = [], []

        # Randomly pick certain amount of data from self.memory to fit the model
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(state)
            y_target[0][action] = reward if done else reward + self.gamma * self.target_net.predict(next_state)[0][action]
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)


    def replace_target(self):
        print('replace_target!!')
        models_weights = self.model.get_weights()
        self.target_net.set_weights(models_weights)


    def run(self):

        train_target = 5 # Replace target_net every 5 episodes in the beginning
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            #print('init:', state)
            done = False
            step_ = 0

            while not done:

                # gym.render is to display the cart
                if step_ >= 100 or step_ == 0: self.env.render() 

                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)

                # You can modify reward function here for better performance.
                # States are in the format of [[ x, x_dot, theta, theta_dot ]],
                # so you'll have to get the value through state[0][__] .
                # You can either give punishment/reward by comparing next_state[0] to state[0]
                # or decide only by next_state[0], try to explain what makes the difference performance.
                


                # your reward function here




                # Add current data to memory
                self.remember(state, action, reward, next_state, done)
                state = next_state

                # Simply shows current progress
                step_ += 1
                if step_ % 2000 == 0 and step_ != 0 and not done: print('... Current Time Step', step_)

            print('Ep No. %d, score %d' %(e+1, step_))


            # memory replay
            self.replay(self.batch_size)

            # As more episodes of trainning, the performance become more stable,
            # so we make it longer to freeze target_net
            if e + 1 > 1000: train_target = 250
            elif e + 1 > 800: train_target = 200
            elif e + 1 > 500: train_target = 100
            elif e + 1 > 300: train_target = 50
            elif e + 1 > 200: train_target = 35
            elif e + 1 > 100: train_target = 20
            elif e + 1 > 50: train_target = 10

            if (e+1) % train_target == 0: self.replace_target()

        return e


if __name__ == '__main__':
    agent = DQNCartPoleSolver()
    agent.run()
