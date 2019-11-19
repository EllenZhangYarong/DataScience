import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6, eps=0.05, gamma=1, alpha=0.2):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random()>self.eps:
            return np.argmax(self.Q[state])
        else:
            return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        next_action = self.select_action(next_state) 
        self.Q[state][action] = self.Q[state][action] + self.alpha * (reward + self.gamma * (self.Q[next_state][next_action]) - self.Q[state][action]) 