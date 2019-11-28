# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg_agent import Agent, ReplayBuffer
import numpy as np
import random
import copy
from collections import namedtuple, deque

UPDATE_EVERY = 2        
BUFFER_SIZE = int(2e5)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor

class MAgent():
    def __init__(self, state_size, action_size, num_agents, random_seed, shared_replay_buffer):
        
        
        
        self.state_size =state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.shared_replay_buffer = shared_replay_buffer
        
        self.t_step = 0
        
        if shared_replay_buffer:
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
            shared_memory = self.memory
        else:
            shared_memory = None
            self.memory = None
            
            
        print("ma shared_memory -> ", shared_memory)
            
        self.ddpg_agents = [Agent(state_size, action_size,random_seed, shared_memory) for _ in range(num_agents)]
#         print("MAgent: number of agents: ->", num_agents)
#         print("Enter into ddpg Agent")
         
             
    def reset(self):
        for agent in self.ddpg_agents:
            agent.reset()
    
    def act(self, all_states):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(np.expand_dims(states, axis=0)) for agent, states in zip(self.ddpg_agents, all_states)]
        return actions
    
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
            
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY

        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
#             print(len(self.memory))
            if len(self.memory) > BATCH_SIZE:
                for agent in self.ddpg_agents:
                    if self.shared_replay_buffer:
                        experiences = self.memory.sample()
                    else:
                        experiences = agent.memory.sample()
                    
                    agent.learn(experiences, GAMMA)
