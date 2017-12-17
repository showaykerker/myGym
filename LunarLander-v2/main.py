import gym
from DuelingDQN import DuelingDQN
from collections import deque
env = gym.make('LunarLander-v2')

agent = DuelingDQN(
			n_actions = 4,
			n_features = 8,
			learning_rate=0.001,
			gamma=0.5,
			e_greedy=0.9,
			replace_target_iter=100,
			memory_size=16000,
			batch_size=64,
			e_greedy_increment=0.0005,
			output_graph=True,
			dueling=True,
			sess=None,
			)


train_target = 5 # Replace target_net every 5 episodes in the beginning

scores = deque(maxlen=100)

for e in range(10000):
	state = env.reset()
	#print('init:', state)
	done = False
	step_ = 0
	total_reward = 0

	while not done:

		# gym.render is used to display the cart
		if step_ >= 100 or step_ == 0: env.render() 

		action = agent.choose_action(state)
		next_state, reward, done, _ = env.step(action)
		next_state = next_state

		# You can modify reward function here for better performance.
		# States are in the format of [[ x, x_dot, theta, theta_dot ]],
		# so you'll have to get the value through state[0][__] .
		# You can either give punishment/reward by comparing next_state[0] to state[0]
		# or decide only by next_state[0], try to explain what makes the difference performance.
		


		# your reward function here
		#print(reward)




		# Add current data to memory
		agent.store_transition(state, action, reward, next_state)
		state = next_state

		# Simply shows current progress
		step_ += 1
		total_reward += reward
		if step_ % 2000 == 0 and step_ != 0 and not done: print('... Current Time Step', step_)

	print('Ep No. %d, steps %d' %(e+1, step_), ' reward = ', total_reward)


	# memory replay
	agent.learn()




