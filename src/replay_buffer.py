#!/usr/bin/python3

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=10000, state_dim=3, action_dim=2, next_state_dim=3):
        self.states = np.empty((capacity, state_dim))
        self.actions = np.empty((capacity, action_dim))

        self.capacity = capacity
        self.full = False
        self.idx = 0

    def add(self, state, action):
        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)

        self.idx = (self.idx + 1) % self.capacity
        if self.idx == 0:
            self.full = True
    
    def sample(self, sample_size):
        n_stored = self.capacity if self.full else self.idx
        sample_size = min(n_stored, sample_size)
        valid_idxs = np.arange(n_stored)
        if self.idx > 0:
            valid_idxs = np.delete(valid_idxs, self.idx - 1)

        sample_idx = np.random.choice(valid_idxs, sample_size)
        states = self.states[sample_idx]
        actions = self.actions[sample_idx]
        next_states = self.states[(sample_idx + 1) % self.capacity]

        return states, actions, next_states
