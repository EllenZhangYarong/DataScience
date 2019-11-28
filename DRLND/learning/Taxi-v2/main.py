from agent import Agent
from monitor import interact
import gym
import numpy as np

env = gym.make('Taxi-v2')
agent = Agent(nA=6, eps=0.001, gamma=1, alpha=0.245)
avg_rewards, best_avg_reward = interact(env, agent)