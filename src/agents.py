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
            "target_heading": 0.,
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

        if self.exp_name is None:
            now = datetime.now()
            self.exp_name = now.strftime("%d_%m_%Y_%H_%M_%S")

        self.save_dir = os.path.expanduser(f"~/kamigami_data/agents/")

        self.state_dict_save_dir = self.save_dir + self.exp_name + "/"
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

        discount = self.discount_factor ** np.arange(self.mpc_horizon)
        ensemble_costs *= discount[None, None, :]

        # average over ensemble and horizon dimensions to get per-sample cost
        total_costs = ensemble_costs.mean(axis=(0, 2))
        total_costs -= total_costs.min()
        total_costs /= total_costs.max()

        return total_costs

    def dump(self):
        if not os.path.exists(self.state_dict_save_dir):
            os.makedirs(self.state_dict_save_dir)

        for model, path in zip(self.models, self.save_paths):
            torch.save(model.state_dict(), path)

    def restore(self, restore_dir=None, recency=1):
        if restore_dir is None:
            # get latest subdir in save directory (state_dicts saved in subdir)
            all_subdirs = [os.path.join(self.save_dir, d) for d in os.listdir(self.save_dir) if os.path.isdir(os.path.join(self.save_dir, d))]
            subdirs_sorted_recency = sorted(all_subdirs, key=os.path.getctime)
            restore_dir = subdirs_sorted_recency[-recency]

        sort_fn = lambda path: int(path.split(".")[0][-1])
        restore_paths = [os.path.join(restore_dir, f) for f in os.listdir(restore_dir)]
        restore_paths.sort(key=sort_fn)

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
                sampled_actions = np.clip(sampled_actions, -1, 1)

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

        return best_action, predicted_next_state


class DifferentialDriveAgent:
    def __init__(self, params):
        self.params = params
        self.dtu = DataUtils(params)

    def __getattr__(self, key):
        return self.params[key]

    def get_action(self, state, goals):
        goal = goals[0]
        xy_vector_to_goal = (goal - state)[:2]
        xy_vector_to_goal_angle = np.arctan2(xy_vector_to_goal[1], xy_vector_to_goal[0])

        angle_diff_side = (xy_vector_to_goal_angle - state[2]) % (2 * np.pi)
        angle_diff_dir = np.stack((angle_diff_side, 2 * np.pi - angle_diff_side)).min(axis=0)

        left = (angle_diff_side < np.pi)
        forward = (angle_diff_dir > np.pi / 2)

        left = left * 2 - 1
        forward = forward * 2 - 1

        heading_cost = angle_diff_dir
        heading_cost *= left * forward

        dist_cost = np.linalg.norm(xy_vector_to_goal)
        dist_cost *= forward

        ctrl_array = np.array([[0.5, 0.5], [0.5, -0.5]])
        error_array = np.array([dist_cost * 3., heading_cost * 0.17]) * 9

        action = ctrl_array @ error_array
        action /= max(1, max(abs(action)))

        return action, np.zeros(3)


        # our best attempt at no-ribbon
                # if t == 0:
                #     object_state_temp = initial_state[..., -3:]
                #     new_state = initial_state.copy()

                #     new_state[..., :3] = object_state_temp
                #     new_state[..., 3:] = initial_state[..., :-3]
                # else:
                #     object_state_temp = pred_state_sequence[i, :, t-1, -3:].copy()
                #     new_state = pred_state_sequence[i, :, t-1].copy()

                #     new_state[..., :3] = object_state_temp
                #     new_state[..., 3:] = pred_state_sequence[i, :, t-1, :-3]

                # new_state_relative = self.dtu.state_to_model_input(new_state).numpy()
                # robot0_relative_xy = new_state_relative[..., :2]
                # robot1_relative_xy = new_state_relative[..., 4:6]

                # robot0_norm = np.linalg.norm(robot0_relative_xy, axis=-1)
                # robot1_norm = np.linalg.norm(robot1_relative_xy, axis=-1)

                # limit = 0.25

                # robot0_norm[robot0_norm > limit] = limit
                # robot1_norm[robot1_norm > limit] = limit

                # robot0_norm = robot0_norm[:, None]
                # robot1_norm = robot1_norm[:, None]

                # robot0_relative_xy = robot0_relative_xy / np.linalg.norm(robot0_relative_xy, axis=-1, keepdims=True) * robot0_norm
                # robot1_relative_xy = robot1_relative_xy / np.linalg.norm(robot1_relative_xy, axis=-1, keepdims=True) * robot1_norm

                # robot0_xy = object_state_temp[..., :2] + robot0_relative_xy
                # robot1_xy = object_state_temp[..., :2] + robot1_relative_xy

                # if t == 0:
                #     initial_state[..., :2] = robot0_xy
                #     initial_state[..., 3:5] = robot1_xy
                # else:
                #     pred_state_sequence[i, :, t, :2] = robot0_xy
                #     pred_state_sequence[i, :, t, 3:5] = robot1_xy