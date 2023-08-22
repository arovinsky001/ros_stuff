#!/usr/bin/python3

import os
from datetime import datetime
import numpy as np
import torch

from replay_buffer import STATE_DIM, ACTION_DIM
from dynamics_network import DynamicsNetwork
from utils import DataUtils, as_tensor


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
            "distance_penalty": self.cost_distance_penalty_weight,
            "big_distance_penalty": self.cost_big_distance_penalty_weight,
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

        self.n_particles = 20
        assert self.n_particles % self.ensemble_size == 0

    def __getattr__(self, key):
        return self.params[key]

    def simulate(self, initial_state, action_sequence):
        initial_state, action_sequence = as_tensor(initial_state, action_sequence)
        mpc_samples, mpc_horizon = action_sequence.shape[0], action_sequence.shape[1]
        # initial_state = torch.tile(initial_state, (mpc_samples, self.n_particles // self.ensemble_size, 1))
        initial_state = torch.tile(initial_state, (self.ensemble_size, mpc_samples, self.n_particles // self.ensemble_size, 1))
        pred_state_sequence = torch.empty((self.ensemble_size, mpc_samples, self.n_particles // self.ensemble_size, mpc_horizon, self.state_dim))

        for model in self.models:
            model.eval()

        for t in range(mpc_horizon):
            action = torch.tile(action_sequence[:, t], (1, self.n_particles // self.ensemble_size)).reshape(mpc_samples * self.n_particles // self.ensemble_size, self.action_dim)
            state = initial_state if t == 0 else pred_state_sequence[:, :, :, t-1]
            state = state.transpose(0, 1).reshape(mpc_samples, self.n_particles, self.state_dim)

            sort_idx = torch.rand(mpc_samples, self.n_particles).argsort(dim=-1).reshape(-1)
            arange = torch.tile(torch.arange(mpc_samples)[:, None], (1, self.n_particles)).reshape(-1)

            original_idx = torch.tile(torch.arange(self.n_particles)[None, :], (mpc_samples, 1))
            unsort_idx = original_idx[arange, sort_idx].reshape(mpc_samples, self.n_particles).argsort(dim=-1).reshape(-1)

            state = state[arange, sort_idx].reshape(mpc_samples, self.n_particles, self.state_dim)

            pred_next_states = torch.empty(mpc_samples, self.n_particles, self.state_dim)

            with torch.no_grad():
                for i, model in enumerate(self.models):
                    start, end = i*(self.n_particles // self.ensemble_size), (i+1)*(self.n_particles // self.ensemble_size)
                    cur_state = state[:, start:end]
                    cur_state = cur_state.reshape(mpc_samples * self.n_particles // self.ensemble_size, self.state_dim)
                    pred_next_states[:, start:end] = model(cur_state, action, sample=True, delta=False).reshape(mpc_samples, self.n_particles // self.ensemble_size, self.state_dim)

                pred_state_sequence[:, :, :, t] = pred_next_states[arange, unsort_idx].reshape(mpc_samples, self.ensemble_size, self.n_particles // self.ensemble_size, self.state_dim).transpose(0, 1)

        # for i, model in enumerate(self.models):
        #     model.eval()

        #     for t in range(mpc_horizon):
        #         action = torch.tile(action_sequence[:, t], (1, self.n_particles // self.ensemble_size)).reshape(mpc_samples * self.n_particles // self.ensemble_size, self.action_dim)
        #         state = initial_state if t == 0 else pred_state_sequence[i, :, :, t-1]
        #         state = state.reshape(mpc_samples * self.n_particles // self.ensemble_size, self.state_dim)

        #         with torch.no_grad():
        #             # Note: sample to be true
        #             pred_state_sequence[i, :, :, t] = model(state, action, sample=True, delta=False).reshape(mpc_samples, self.n_particles // self.ensemble_size, self.state_dim)

        if torch.any(torch.isnan(pred_state_sequence) | torch.isinf(pred_state_sequence)):
            import pdb;pdb.set_trace()

        return pred_state_sequence

    def compute_trajectory_costs(self, predicted_state_sequence, sampled_actions, goals, initial_state):
        with torch.no_grad():
            cost_dict = self.dtu.cost_dict(predicted_state_sequence, sampled_actions, goals, initial_state, robot_goals=self.robot_goals)

        ensemble_costs_per_step = torch.zeros((self.ensemble_size, self.mpc_samples, self.n_particles // self.ensemble_size, self.mpc_horizon))
        for cost_type, cost_weight in self.cost_weights_dict.items():
            if cost_weight != 0.:
                ensemble_costs_per_step += cost_dict[cost_type] * cost_weight

        # if self.use_object:
        #     # ensemble_costs_per_step (ens_size, mpc_samples, n_particles / ens_size, mpc_horizon) -> ensemble_costs_std (mpc_samples, mpc_horizon)
        #     ensemble_costs_std = ensemble_costs_per_step.std(dim=(0, 2))
        #     ensemble_costs_per_step += ensemble_costs_std[:, None, :] * self.cost_std_weight

        discount = self.discount_factor ** torch.arange(self.mpc_horizon)
        ensemble_costs_per_step *= discount[None, None, None, :]

        # average over ensemble and horizon dimensions to get per-sample cost
        # ensemble_costs_per_step (ens_size, mpc_samples, n_particles / ens_size, mpc_horizon) -> ensemble_costs_per_sample (ens_size, mpc_samples, n_particles / ens_size)
        ensemble_costs_per_sample = ensemble_costs_per_step.mean(dim=3)

        # ensemble_costs_per_sample (ens_size, mpc_samples, n_particles / ens_size) -> costs_per_sample (mpc_samples)
        costs_per_sample = ensemble_costs_per_sample.mean(dim=(0, 2))

        if self.use_object:
            ensemble_costs_std = ensemble_costs_per_sample.std(dim=(0, 2))
            costs_per_sample += ensemble_costs_std * self.cost_std_weight

        # print("\nCOSTS STD:", costs_per_sample.std())
        # power = 0.1
        # a = (1.5) ** (1 - power)
        # fx = (1.5 * a * costs_per_sample.std() ** power)

        # costs_per_sample /= (costs_per_sample.std() * costs_per_sample.mean().abs() / 4.)

        # costs_per_sample /= (costs_per_sample.std() + costs_per_sample.mean() / costs_per_sample.std())

        # costs_per_sample = costs_per_sample * 5. / costs_per_sample.mean().abs()

        # costs_per_sample /= costs_per_sample.std()

        # cx + d / x
        # c = 0.3
        # d = 1.6
        # fx = c * costs_per_sample.std() + d / costs_per_sample.std()
        # costs_per_sample /= fx

        # costs_per_sample -= costs_per_sample.min()
        # costs_per_sample /= costs_per_sample.max()

        if torch.any(torch.isnan(costs_per_sample) | torch.isinf(costs_per_sample)):
            import pdb;pdb.set_trace()

        return costs_per_sample.detach().cpu().numpy()

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
        total_costs = self.compute_trajectory_costs(predicted_state_sequence, sampled_actions, goals, initial_state)

        best_idx = total_costs.argmin()
        best_action = sampled_actions[best_idx, 0]
        predicted_next_state = predicted_state_sequence[:, best_idx, 0].mean(axis=0).squeeze().detach().cpu().numpy()

        return best_action, predicted_next_state


class CEMAgent(MPCAgent):
    def __init__(self, params, **kwargs):
        return super().__init__(params, **kwargs)

    def get_action(self, initial_state, goals):
        trajectory_mean = np.zeros((self.mpc_horizon, self.action_dim))
        # trajectory_mean = np.random.uniform(-1., 1., (self.mpc_horizon, self.action_dim))
        trajectory_std = np.zeros((self.mpc_horizon, self.action_dim))
        sampled_actions = np.random.uniform(-1., 1., size=(self.mpc_samples, self.mpc_horizon, self.action_dim))

        for i in range(self.refine_iters):
            if i > 0:
                sampled_actions = np.random.normal(loc=trajectory_mean, scale=trajectory_std, size=(self.mpc_samples, self.mpc_horizon, self.action_dim))
                sampled_actions = np.clip(sampled_actions, -1., 1.)

            predicted_state_sequence = self.simulate(initial_state, sampled_actions)
            total_costs = self.compute_trajectory_costs(predicted_state_sequence, sampled_actions, goals, initial_state)

            best_costs_idx = np.argsort(total_costs)[:self.n_best]
            best_trajectories = sampled_actions[best_costs_idx]
            best_trajectories_mean = best_trajectories.mean(axis=0)
            best_trajectories_std = best_trajectories.std(axis=0)

            trajectory_mean = self.alpha * best_trajectories_mean + (1 - self.alpha) * trajectory_mean
            trajectory_std = self.alpha * best_trajectories_std + (1 - self.alpha) * trajectory_std

            # print(trajectory_std.max())
            if trajectory_std.max() < 0.02:
                break

        best_action = trajectory_mean[0]
        predicted_next_state = self.simulate(initial_state, best_action[None, None, :]).squeeze().mean(dim=(0, 1)).detach().cpu().numpy()

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
            sampled_actions_t = self.beta * (self.trajectory_mean[t] + noise[:, t]) + (1 - self.beta) * previous_action
            sampled_actions[:, t] = np.clip(sampled_actions_t, -1., 1.)

        predicted_state_sequence = self.simulate(initial_state, sampled_actions)
        total_costs = self.compute_trajectory_costs(predicted_state_sequence, sampled_actions, goals, initial_state)
        total_rewards = -total_costs

        # best_rewards_idx = total_rewards.argsort()[-int(0.3*self.mpc_samples):]
        # total_rewards = total_rewards[best_rewards_idx]
        # sampled_actions = sampled_actions[best_rewards_idx]

        weights = np.exp(self.gamma * (total_rewards - total_rewards.max()))
        weighted_trajectories = (weights[:, None, None] * sampled_actions).sum(axis=0)
        self.trajectory_mean = weighted_trajectories / (weights.sum() + 1e-10)

        best_action = self.trajectory_mean[0]
        predicted_next_state = self.simulate(initial_state, best_action[None, None, :]).squeeze().mean(dim=(0, 1)).detach().cpu().numpy()

        # print('\n')
        # gammas = [0.1, 0.5, 1., 2., 3., 5.]
        # gammas = [5., 8., 10., 12., 15., 18.]
        # # gammas = [10., 15., 20., 25., 30.]
        # for gamma in gammas:
        #     weights = np.exp(gamma * (total_rewards - total_rewards.max()))
        #     weighted_trajectories = (weights[:, None, None] * sampled_actions).sum(axis=0)
        #     trajectory_mean = weighted_trajectories / (weights.sum() + 1e-10)
        #     print('\nGAMMA =', gamma)
        #     print('BEST ACTION:', trajectory_mean[0])
        #     print('TOP WEIGHTS:', weights[np.argsort(weights)[-6:]])
        # print('\n')

        # import pdb;pdb.set_trace()
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
