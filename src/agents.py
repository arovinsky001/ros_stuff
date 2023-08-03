#!/usr/bin/python3

import os
from datetime import datetime
import numpy as np
import torch

from replay_buffer import STATE_DIM, ACTION_DIM
from dynamics_network import DynamicsNetwork
from utils import DataUtils


class MPCAgent:
    def __init__(self, params):
        self.params = params
        assert self.ensemble_size > 0

        if not self.robot_goals:
            assert self.use_object

        # weights for MPC cost terms
        self.cost_weights_dict = {
            "distance": self.cost_distance_weight,
            # "heading": 0.,
            # "target_heading": 0.,
            "action_norm": self.cost_action_weight,
            "distance_bonus": self.cost_distance_bonus_weight,
            "separation": self.cost_separation_weight,
            # "heading_difference": 0.,
            "to_object_heading": self.cost_to_object_heading_weight,
            "goal_object_robot_angle": self.cost_goal_angle_weight,
            "realistic": self.cost_realistic_weight,
            "object_delta": self.cost_object_delta_weight,
        }

        self.state_dim = STATE_DIM * (self.n_robots + self.use_object)
        self.action_dim = ACTION_DIM * self.n_robots
        self.dtu = DataUtils(params)

        self.models = [DynamicsNetwork(params) for _ in range(self.ensemble_size)]
        self.trained = False

        if self.exp_name is None:
            now = datetime.now()
            self.exp_name = now.strftime("%d_%m_%Y_%H_%M_%S")

        self.save_dir = os.path.expanduser(f"~/kamigami_data/agents/")

        self.state_dict_save_dir = self.save_dir + self.exp_name + "/"
        self.save_paths = [self.state_dict_save_dir + f"robot_{'_'.join([str(id) for id in self.robot_ids])}_object_{self.use_object}_state_dict_{i}.pt" for i in range(self.ensemble_size)]
        self.scale_idx_path = self.state_dict_save_dir + f"robot_{'_'.join([str(id) for id in self.robot_ids])}_object_{self.use_object}_scale_idx_.npy"

    def __getattr__(self, key):
        return self.params[key]

    def simulate(self, initial_state, action_sequence):
        mpc_samples, mpc_horizon = action_sequence.shape[0], action_sequence.shape[1]
        initial_state = torch.tile(torch.tensor(initial_state, dtype=torch.float), (mpc_samples, 1))
        pred_state_sequence = torch.empty((len(self.models), mpc_samples, mpc_horizon, self.state_dim))

        for i, model in enumerate(self.models):
            model.eval()

            for t in range(mpc_horizon):
                action = action_sequence[:, t]

                with torch.no_grad():
                    if t == 0:
                        pred_state_sequence[i, :, t] = model(initial_state, action, sample=False, delta=False)                      #.cpu()
                    else:
                        pred_state_sequence[i, :, t] = model(pred_state_sequence[i, :, t-1], action, sample=False, delta=False)     #.cpu()

        if torch.any(torch.isnan(pred_state_sequence) | torch.isinf(pred_state_sequence)):
            import pdb;pdb.set_trace()

        return pred_state_sequence

    def compute_trajectory_costs(self, predicted_state_sequence, sampled_actions, goals, robot_goals, initial_state):
        cost_dict = self.dtu.cost_dict(predicted_state_sequence, sampled_actions, goals, initial_state, robot_goals=robot_goals)

        ensemble_costs = torch.zeros((self.ensemble_size, self.mpc_samples, self.mpc_horizon))
        for cost_type, cost_weight in self.cost_weights_dict.items():
            # if cost_type != 'distance':
            ensemble_costs += cost_dict[cost_type] * cost_weight

        # ensemble_costs = cost_dict['distance'] * (ensemble_costs + self.cost_weights_dict['distance'])

        discount = self.discount_factor ** torch.arange(self.mpc_horizon)
        ensemble_costs *= discount[None, None, :]

        # average over ensemble and horizon dimensions to get per-sample cost
        ensemble_costs_per_sample = ensemble_costs.mean(axis=2)
        ensemble_costs_mean = ensemble_costs_per_sample.mean(axis=0)
        ensemble_costs_std = ensemble_costs_per_sample.std(axis=0)

        assert ensemble_costs_mean.shape == ensemble_costs_std.shape == (self.mpc_samples,)

        total_costs = ensemble_costs.mean(axis=(0, 2))
        costs_std = ensemble_costs.std(axis=(0, 2))

        if self.use_object:
            total_costs += costs_std * self.cost_std_weight

        total_costs -= total_costs.min()
        total_costs /= total_costs.max()

        if torch.any(torch.isnan(total_costs) | torch.isinf(total_costs)):
            import pdb;pdb.set_trace()

        return total_costs.detach().cpu().numpy()

    def dump(self, scale_idx=0):
        if not os.path.exists(self.state_dict_save_dir):
            os.makedirs(self.state_dict_save_dir)

        for model, path in zip(self.models, self.save_paths):
            torch.save(model.state_dict(), path)

        np.save(self.scale_idx_path, np.array(scale_idx))

    def restore(self, restore_dir=None, recency=1):
        restore_paths = self.save_paths

        for i in range(self.ensemble_size):
            self.models[i].load_state_dict(torch.load(restore_paths[i]))

        self.scale_idx = np.load(self.scale_idx_path)

    def get_action(self, init_state, goals):
        return None, None


class RandomShootingAgent(MPCAgent):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)

    def get_action(self, initial_state, goals):
        sampled_actions = np.random.uniform(-1, 1, size=(self.mpc_samples, self.mpc_horizon, self.action_dim))
        predicted_state_sequence = self.simulate(initial_state, sampled_actions)
        total_costs = self.compute_trajectory_costs(predicted_state_sequence, sampled_actions, goals, self.robot_goals, initial_state)

        best_idx = total_costs.argmin()
        best_action = sampled_actions[best_idx, 0]
        predicted_next_state = predicted_state_sequence[:, best_idx, 0].mean(axis=0).squeeze().detach().cpu().numpy()

        return best_action, predicted_next_state


class CEMAgent(MPCAgent):
    def __init__(self, params, **kwargs):
        return super().__init__(params, **kwargs)

    def get_action(self, initial_state, goals):
        trajectory_mean = np.zeros((self.mpc_horizon, self.action_dim))
        trajectory_std = np.zeros((self.mpc_horizon, self.action_dim))
        sampled_actions = np.random.uniform(-1, 1, size=(self.mpc_samples, self.mpc_horizon, self.action_dim))

        for i in range(self.refine_iters):
            if i > 0:
                sampled_actions = np.random.normal(loc=trajectory_mean, scale=trajectory_std, size=(self.mpc_samples, self.mpc_horizon, self.action_dim))
                sampled_actions = np.clip(sampled_actions, -1., 1.)

            predicted_state_sequence = self.simulate(initial_state, sampled_actions)
            total_costs = self.compute_trajectory_costs(predicted_state_sequence, sampled_actions, goals, self.robot_goals, initial_state)

            best_costs_idx = np.argsort(-total_costs)[-self.n_best:]
            best_trajectories = sampled_actions[best_costs_idx]
            best_trajectories_mean = best_trajectories.mean(axis=0)
            best_trajectories_std = best_trajectories.std(axis=0)

            trajectory_mean = self.alpha * best_trajectories_mean + (1 - self.alpha) * trajectory_mean
            trajectory_std = self.alpha * best_trajectories_std + (1 - self.alpha) * trajectory_std

            if trajectory_std.max() < 0.02:
                break

        best_action = trajectory_mean[0]
        predicted_next_state = self.simulate(initial_state, best_action[None, None, :]).mean(axis=0).squeeze().detach().cpu().numpy()

        return best_action, predicted_next_state


class MPPIAgent(MPCAgent):
    def __init__(self, params, **kwargs):
        self.trajectory_mean = None
        return super().__init__(params, **kwargs)

    def get_action(self, initial_state, goals):
        if self.trajectory_mean is None:
            self.trajectory_mean = np.random.uniform(low=-1., high=1., size=(self.mpc_horizon, self.action_dim))
            # self.trajectory_mean = np.zeros((self.mpc_horizon, self.action_dim))

        just_executed_action = self.trajectory_mean[0].copy()
        self.trajectory_mean[:-1] = self.trajectory_mean[1:]

        sampled_actions = np.empty((self.mpc_samples, self.mpc_horizon, self.action_dim))
        noise = np.random.normal(loc=0, scale=self.noise_std, size=(self.mpc_samples, self.mpc_horizon, self.action_dim))

        for t in range(self.mpc_horizon):
            previous_action = just_executed_action if t == 0 else sampled_actions[:, t-1]
            sampled_actions[:, t] = self.beta * (self.trajectory_mean[t] + noise[:, t]) + (1 - self.beta) * previous_action

        sampled_actions = np.clip(sampled_actions, -1, 1)
        predicted_state_sequence = self.simulate(initial_state, sampled_actions)
        total_costs = self.compute_trajectory_costs(predicted_state_sequence, sampled_actions, goals, self.robot_goals, initial_state)
        total_rewards = -total_costs

        weights = np.exp(self.gamma * (total_rewards - total_rewards.max()))
        weighted_trajectories = (weights[:, None, None] * sampled_actions).sum(axis=0)
        self.trajectory_mean = weighted_trajectories / (weights.sum() + 1e-10)

        best_action = self.trajectory_mean[0]
        predicted_next_state = self.simulate(initial_state, best_action[None, None, :]).mean(axis=0).squeeze().detach().cpu().numpy()

        # weights = np.exp(gamma * (total_rewards - total_rewards.max())); weighted_trajectories = (weights[:, None, None] * sampled_actions).sum(axis=0); trajectory_mean = weighted_trajectories / (weights.sum() + 1e-10); print(trajectory_mean[0]); print(weights[np.argsort(weights)[-10:]])
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
