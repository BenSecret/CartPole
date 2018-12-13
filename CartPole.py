import torch
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


class Simulation(torch.nn.Module):
	def __init__(self):
		super(Simulation, self).__init__()
		HIDDEN = 400
		self.linear1 = torch.nn.Linear(5, HIDDEN)
		self.linear2 = torch.nn.Linear(HIDDEN, HIDDEN)
		self.linear3 = torch.nn.Linear(HIDDEN, 4)

	def forward(self, x):
		x = torch.tanh(self.linear1(x))
		x = torch.tanh(self.linear2(x))
		return self.linear3(x)

def train_simulation(predicted, states):
	loss = loss_fn(predicted, torch.FloatTensor(states))
	optim.zero_grad()
	loss.backward()
	optim.step()
	scheduler.step()

def run_simulation(action, state, momentum):
	errors = []
	look_ahead_mask = torch.FloatTensor([0.1,0.2,0.4,1])
	for i in range(momentum):
		state = simulation(torch.FloatTensor(np.hstack(([action], state.detach()))))
		errors.append(F.mse_loss(torch.mul(state, states_mask), torch.zeros(env.observation_space.shape[0])))
	return torch.sum(torch.mul(torch.tensor(errors), look_ahead_mask))


env = gym.make('CartPole-v0')
env.seed(8); torch.manual_seed(8); np.random.seed(8)
episodes = 1000
steps = 200

simulation = Simulation()
optim = torch.optim.Adam(simulation.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1080, gamma=0.5)
loss_fn = torch.nn.MSELoss()

ave_reward = deque(maxlen=100)
states_mask = torch.FloatTensor([0.26,0,0.7,1])
exploration = 1
momentum = 4
results = []


for episode in range(episodes):
	episode_reward = 0	
	state = env.reset()
	action = np.random.randint(0,2)

	for s in range(steps):
		env.render()
		prediction_l = run_simulation(0, torch.tensor(state), momentum)
		prediction_r = run_simulation(1, torch.tensor(state), momentum)

		if np.random.rand(1) < exploration:
			action = np.random.randint(0,2)
		else: 
			_, action = torch.min(torch.FloatTensor([prediction_l, prediction_r]), 0)

		prediction = simulation(torch.FloatTensor(np.hstack(([action], state))))
		state, reward, done, _ = env.step(int(action))
		train_simulation(prediction, state)
		episode_reward += reward
		
		if done:
			if episode % 10 == 0:
				print('Episode {}\tAverage reward: {:.2f}'.format(episode, np.mean(ave_reward)))
			break

	results.append(episode_reward)
	exploration *= 0.93
	ave_reward.append(episode_reward)
	if np.mean(ave_reward) >= env.spec.reward_threshold:
		print('Completed at episode: ', episode)
		break

plt.plot(results)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
