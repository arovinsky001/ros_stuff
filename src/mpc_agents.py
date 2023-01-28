#!/usr/bin/python3

import os
from datetime import datetime
import numpy as np
import torch

from replay_buffer import STATE_DIM, ACTION_DIM
from dynamics_network import DynamicsNetwork
from utils import DataUtils

device = "cpu"


class MPCAgent:
    def __init__(self, params):
        self.params = params
        assert self.ensemble_size > 0

        if not self.robot_goals:
            assert self.use_object

        # weights for MPC cost terms
        self.cost_weights_dict = {
            "distance": 1.,
            "heading": 0.,
            "action_norm": 0.,
            "distance_bonus": 0.,
            "separation": 0.,
            "heading_difference": 0.,
        }

        self.state_dim = STATE_DIM * (self.n_robots + self.use_object)
        self.action_dim = ACTION_DIM * self.n_robots
        self.dtu = DataUtils(params)

        self.models = [DynamicsNetwork(params) for _ in range(self.ensemble_size)]
        for model in self.models:
            model.to(device)

        self.trained = False

        now = datetime.now()
        date_time = now.strftime("%d_%m_%Y_%H_%M_%S")
        self.save_dir = os.path.expanduser(f"~/kamigami_data/agents/")

        self.state_dict_save_dir = self.save_dir + date_time + "/"
        self.save_paths = [self.state_dict_save_dir + f"state_dict{i}.npz" for i in range(self.ensemble_size)]

    def __getattr__(self, key):
        return self.params[key]

    def simulate(self, initial_state, action_sequence):
        mpc_samples, mpc_horizon = action_sequence.shape[0], action_sequence.shape[1]
        initial_state = np.tile(initial_state, (mpc_samples, 1))
        pred_state_sequence = np.empty((len(self.models), mpc_samples, mpc_horizon, self.state_dim))

        for i, model in enumerate(self.models):
            model.eval()

            for t in range(mpc_horizon):
                action = action_sequence[:, t]

                with torch.no_grad():
                    if t == 0:
                        pred_state_sequence[i, :, t] = model(initial_state, action, sample=False, delta=False)
                    else:
                        pred_state_sequence[i, :, t] = model(pred_state_sequence[i, :, t-1], action, sample=False, delta=False)

        return pred_state_sequence

    def compute_trajectory_costs(self, predicted_state_sequence, sampled_actions, goals, robot_goals):
        cost_dict = self.dtu.cost_dict(predicted_state_sequence, sampled_actions, goals, robot_goals=robot_goals)

        ensemble_costs = np.zeros((self.ensemble_size, self.mpc_samples, self.mpc_horizon))
        for cost_type in cost_dict:
            ensemble_costs += cost_dict[cost_type] * self.cost_weights_dict[cost_type]

        # discount costs through time
        # discount = (1 - 1 / (4 * self.mpc_horizon)) ** np.arange(self.mpc_horizon)

        discount = 0.8 ** np.arange(self.mpc_horizon)
        ensemble_costs *= discount[None, None, :]

        # average over ensemble and horizon dimensions to get per-sample cost
        total_costs = ensemble_costs.mean(axis=(0, 2))
        total_costs -= total_costs.min()
        total_costs /= total_costs.max()

        return total_costs

    def dump(self):
        os.makedirs(self.state_dict_save_dir)

        for model, path in zip(self.models, self.save_paths):
            torch.save(model.state_dict(), path)

    def restore(self, restore_dir=None):
        if restore_dir is None:
            # get latest subdir in save directory (state_dicts saved in subdir)
            all_subdirs = [d for d in os.listdir(restore_dir) if os.path.isdir(d)]
            restore_dir = max(all_subdirs, key=os.path.getctime)

        sort_fn = lambda path: int(path.split(".")[-1][-1])
        restore_paths = os.listdir(restore_dir).sort(key=sort_fn)

        for i in range(self.ensemble_size):
            self.models[i].load_state_dict(torch.load(restore_paths[i]))

    def get_action(self, init_state, goals):
        return None, None


class RandomShootingAgent(MPCAgent):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    def get_action(self, initial_state, goals):
        sampled_actions = np.random.uniform(-1, 1, size=(self.mpc_samples, self.mpc_horizon, self.action_dim))
        predicted_state_sequence = self.simulate(initial_state, sampled_actions)
        total_costs = self.compute_trajectory_costs(predicted_state_sequence, sampled_actions, goals, self.robot_goals)

        best_idx = total_costs.argmin()
        best_action = sampled_actions[best_idx, 0]
        predicted_next_state = predicted_state_sequence[:, best_idx, 0].mean(axis=0).squeeze()

        return best_action, predicted_next_state


class CEMAgent(MPCAgent):
    def __init__(self, params, **kwargs):
        return super().__init__(params, **kwargs)

    def get_action(self, initial_state, goals):
        action_trajectory_dim = self.action_dim * self.mpc_horizon

        trajectory_mean = np.zeros(action_trajectory_dim)
        trajectory_std = np.zeros(action_trajectory_dim)
        sampled_actions = np.random.uniform(-1, 1, size=(self.mpc_samples, self.mpc_horizon, self.action_dim))

        for i in range(self.refine_iters):
            if i > 0:
                sampled_actions = np.random.normal(loc=trajectory_mean, scale=trajectory_std, size=(self.mpc_samples, action_trajectory_dim))
                sampled_actions = sampled_actions.reshape(self.mpc_samples, self.mpc_horizon, self.action_dim)

            predicted_state_sequence = self.simulate(initial_state, sampled_actions)
            total_costs = self.compute_trajectory_costs(predicted_state_sequence, sampled_actions, goals, self.robot_goals)

            action_trajectories = sampled_actions.reshape((self.mpc_samples, action_trajectory_dim))
            best_costs_idx = np.argsort(-total_costs)[-self.n_best:]
            best_trajectories = action_trajectories[best_costs_idx]
            best_trajectories_mean = best_trajectories.mean(axis=0)
            best_trajectories_std = best_trajectories.std(axis=0)

            trajectory_mean = self.alpha * best_trajectories_mean + (1 - self.alpha) * trajectory_mean
            trajectory_std = self.alpha * best_trajectories_std + (1 - self.alpha) * trajectory_std

            if trajectory_std.max() < 0.02:
                break

        best_action = trajectory_mean[:self.action_dim]
        predicted_next_state = self.simulate(initial_state, best_action[None, None, :]).mean(axis=0).squeeze()

        return best_action, predicted_next_state


class MPPIAgent(MPCAgent):
    def __init__(self, params, **kwargs):
        self.trajectory_mean = None
        return super().__init__(params, **kwargs)

    def get_action(self, initial_state, goals):
        if self.trajectory_mean is None:
            self.trajectory_mean = np.zeros((self.mpc_horizon, self.action_dim))

        just_executed_action = self.trajectory_mean[0].copy()
        self.trajectory_mean[:-1] = self.trajectory_mean[1:]

        sampled_actions = np.empty((self.mpc_samples, self.mpc_horizon, self.action_dim))
        noise = np.random.normal(loc=0, scale=self.noise_std, size=(self.mpc_samples, self.mpc_horizon, self.action_dim))

        for t in range(self.mpc_horizon):
            if t == 0:
                sampled_actions[:, t] = self.beta * (self.trajectory_mean[t] + noise[:, t]) \
                                            + (1 - self.beta) * just_executed_action
            else:
                sampled_actions[:, t] = self.beta * (self.trajectory_mean[t] + noise[:, t]) \
                                            + (1 - self.beta) * sampled_actions[:, t-1]

        sampled_actions = np.clip(sampled_actions, -1, 1)
        predicted_state_sequence = self.simulate(initial_state, sampled_actions)
        total_costs = self.compute_trajectory_costs(predicted_state_sequence, sampled_actions, goals, self.robot_goals)

        action_trajectories = sampled_actions.reshape((self.mpc_samples, -1))

        weights = np.exp(self.gamma * -total_costs)
        weighted_trajectories = (weights[:, None] * action_trajectories).sum(axis=0)
        self.trajectory_mean = weighted_trajectories / weights.sum()

        best_action = self.trajectory_mean[:self.action_dim]
        predicted_next_state = self.simulate(initial_state, best_action[None, None, :]).mean(axis=0).squeeze()

        # if np.linalg.norm(best_action) < np.sqrt(2) * 0.5:
        #     import pdb;pdb.set_trace()

        return best_action, predicted_next_state
