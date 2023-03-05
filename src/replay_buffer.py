#!/usr/bin/python

import numpy as np
import os
from datetime import datetime
from glob import glob


STATE_DIM = 3
ACTION_DIM = 2


class ReplayBuffer:
    # def __init__(self, capacity=10000, state_dim=3, action_dim=2):
    def __init__(self, params, random=False, precollecting=False):
        self.params = params

        state_dim = STATE_DIM * (self.n_robots + self.use_object)
        action_dim = ACTION_DIM * self.n_robots

        # states always stored as [robot_0, robot_1, ..., robot_n, object]
        self.states = np.empty((self.buffer_capacity, state_dim))
        self.next_states = np.empty((self.buffer_capacity, state_dim))
        self.actions = np.empty((self.buffer_capacity, action_dim))

        self.full = False
        self.idx = 0

        if self.exp_name is None:
            now = datetime.now()
            self.exp_name = now.strftime("%d_%m_%Y_%H_%M_%S")

        if precollecting:
            self.restore_dir = None

            if random:
                self.save_dir = os.path.expanduser("~/kamigami_data/replay_buffers/random_buffers/")
            else:
                self.save_dir = os.path.expanduser("~/kamigami_data/replay_buffers/meshgrid_buffers/")
        else:
            self.save_dir = os.path.expanduser("~/kamigami_data/replay_buffers/online_buffers/")

            if random:
                self.restore_dir = os.path.expanduser("~/kamigami_data/replay_buffers/random_buffers/")
            else:
                self.restore_dir = os.path.expanduser("~/kamigami_data/replay_buffers/meshgrid_buffers/")

        self.save_path = self.save_dir + f"{self.exp_name}.npz"

    def __getattr__(self, key):
        return self.params[key]

    @property
    def size(self):
        return self.buffer_capacity if self.full else self.idx

    def add(self, state, action, next_state):
        np.copyto(self.states[self.idx], state)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.next_states[self.idx], next_state)

        self.idx = (self.idx + 1) % self.buffer_capacity
        if self.idx == 0:
            self.full = True

    def sample(self, sample_size):
        sample_size = min(self.size, sample_size)
        stored_idxs = np.arange(self.size)
        sample_idx = np.random.choice(stored_idxs, sample_size)

        states = self.states[sample_idx]
        actions = self.actions[sample_idx]
        next_states = self.next_states[sample_idx]

        return states, actions, next_states

    # assumes buffer never fills up/wraps
    def sample_recent(self, sample_size):
        idxs = np.arange(self.size)
        sample_idx = idxs[-sample_size:]

        states = self.states[sample_idx]
        actions = self.actions[sample_idx]
        next_states = self.next_states[sample_idx]

        return states, actions, next_states

    def dump(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        np.savez_compressed(self.save_path,
                            states=self.states[:self.size],
                            actions = self.actions[:self.size],
                            next_states=self.next_states[:self.size],
                            idx=self.idx)

    def restore(self, restore_path=None):
        if restore_path is None:
            # get latest file in save directory
            list_of_files = glob(self.restore_dir + "*.npz")
            self.restore_path = max(list_of_files, key=os.path.getctime)
        else:
            self.restore_path = restore_path

        data = np.load(self.restore_path)
        n_samples = len(data["states"])

        for i in range(n_samples):
            idx = (i + data["idx"]) % n_samples
            self.add(data["states"][idx], data["actions"][idx], data["next_states"][idx])
