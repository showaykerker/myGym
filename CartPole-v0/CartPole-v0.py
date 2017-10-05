#/usr/bin/python3
import gym
import random as rand
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

'''
env = gym.make('CartPole-v0')
for i_epsidoe in range(50):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done: 
            print("Episode %d finished after %d steps"%(i_epsidoe+1, t+1))
            break
'''

class DQNagent():
    def __init__(self, n_episodes=1000, n_win_ticks=195, max_env_steps=None, gamma=1.0, 
                 epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995, 
                 alpha=0.01, alpha_decay=0.01,
                 batch_size=64, monitor=False, quiet=False
                 ):
        self.env = gym.make('CartPole-v0')
        self.memory = deque(maxlen=100000)
        if monitor: self.env=gym.wrappers.Monitor(self.env, 'data/', force=True)
        self.alpha, self.alpha_decay, self.gamma = alpha, alpha_decay, gamma
        self.epsilon, self.epsilon_min, self.epsilon_log_decay = epsilon, epsilon_min, epsilon_log_decay
        

        pass
    
    def state_preprocessing(self, state):
        return np.reshape(state, [1,4])

    def remember(self, state, action, reward, next_state, done):
        pass

    def choose_action(self, state):
        if np.rand.random()<=self.epsilon: return self.env.action.sample()
        else: return np.argmax(self.model.predict(state))

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
        

if __name__ == '__main__':
