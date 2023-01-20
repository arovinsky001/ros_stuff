#!/usr/bin/python

import numpy as np
import torch

from dynamics_network import DynamicsNetwork
from utils import DataUtils, signed_angle_difference, dimensions

device = "cpu"


class MPCPolicy:
    def __init__(self, hidden_dim, hidden_depth, lr, std, dist, scale, ensemble, use_object, params, cost_weights_dict):
        self.action_dim = dimensions["action_dim"]
        self.params = params
        self.cost_weights_dict = cost_weights_dict

        assert ensemble > 0

        input_dim = dimensions["action_dim"]
        output_dim = dimensions["robot_output_dim"]
        self.state_dim = dimensions["state_dim"]

        if use_object:
            input_dim += dimensions["object_input_dim"]
            output_dim += dimensions["object_output_dim"]
            self.state_dim += dimensions["state_dim"]

        self.dtu = DataUtils(use_object=use_object)
        self.models = [DynamicsNetwork(input_dim, output_dim, self.dtu, hidden_dim=hidden_dim, hidden_depth=hidden_depth,
                                       lr=lr, std=std, dist=dist, use_object=use_object, scale=scale)
                        for _ in range(ensemble)]
        for model in self.models:
            model.to(device)

        self.scale = scale
        self.ensemble = ensemble
        self.use_object = use_object
        self.trained = False

    def update_params_and_weights(self, params, cost_weights_dict):
        self.params = params
        self.cost_weights_dict = cost_weights_dict

    def simulate(self, initial_state, action_sequence):
        n_samples, horizon, _ = action_sequence.shape
        initial_state = np.tile(initial_state, (n_samples, 1))
        pred_state_sequence = np.empty((len(self.models), n_samples, horizon, self.state_dim))

        for i, model in enumerate(self.models):
            model.eval()

            for t in range(horizon):
                action = action_sequence[:, t]
                with torch.no_grad():
                    if t == 0:
                        pred_state_sequence[i, :, t] = model(initial_state, action, sample=False, delta=False)
                    else:
                        pred_state_sequence[i, :, t] = model(pred_state_sequence[i, :, t-1], action, sample=False, delta=False)

        return pred_state_sequence

    def compute_costs(self, state, action, goals, robot_goals=False, signed=False):
        state_dim = dimensions["state_dim"]
        if self.use_object:
            robot_state = state[:, :, :, :state_dim]
            object_state = state[:, :, :, state_dim:2*state_dim]

            effective_state = robot_state if robot_goals else object_state
        else:
            effective_state = state[:, :, :, :state_dim]

        # distance to goal position
        state_to_goal_xy = (goals - effective_state)[:, :, :, :-1]
        dist_cost = np.linalg.norm(state_to_goal_xy, axis=-1)
        if signed:
            dist_cost *= forward

        # difference between current and goal heading
        current_angle = effective_state[:, :, :, 2]
        target_angle = np.arctan2(state_to_goal_xy[:, :, :, 1], state_to_goal_xy[:, :, :, 0])
        heading_cost = signed_angle_difference(target_angle, current_angle)

        left = (heading_cost > 0) * 2 - 1
        forward = (np.abs(heading_cost) < np.pi / 2) * 2 - 1

        heading_cost[forward == -1] = (heading_cost[forward == -1] + np.pi) % (2 * np.pi)
        heading_cost = np.stack((heading_cost, 2 * np.pi - heading_cost)).min(axis=0)

        if signed:
            heading_cost *= left * forward
        else:
            heading_cost = np.abs(heading_cost)

        # object-robot separation distance
        if self.use_object:
            object_to_robot_xy = (robot_state - object_state)[:, :, :, :-1]
            sep_cost = np.linalg.norm(object_to_robot_xy, axis=-1)
        else:
            sep_cost = np.array([0.])

        # object-robot heading difference
        if self.use_object:
            robot_theta, object_theta = robot_state[:, :, :, -1], object_state[:, :, :, -1]
            heading_diff = (robot_theta - object_theta) % (2 * np.pi)
            heading_diff_cost = np.stack((heading_diff, 2 * np.pi - heading_diff), axis=1).min(axis=1)
        else:
            heading_diff_cost = np.array([0.])

        # action magnitude
        norm_cost = -np.linalg.norm(action, axis=-1)

        cost_dict = {
            "distance": dist_cost,
            "heading": heading_cost,
            "separation": sep_cost,
            "heading_difference": heading_diff_cost,
            "action_norm": norm_cost,
        }

        return cost_dict

    def compute_total_costs(self, predicted_state_sequence, sampled_actions, goals, robot_goals):
        cost_dict = self.compute_costs(predicted_state_sequence, sampled_actions, goals, robot_goals=robot_goals)
        ensemble_size, n_samples, horizon, _ = predicted_state_sequence.shape

        ensemble_costs = np.zeros((ensemble_size, n_samples, horizon))
        for cost_type in cost_dict:
            ensemble_costs += cost_dict[cost_type] * self.cost_weights_dict[cost_type]

        # discount costs through time
        # discount = (1 - 1 / (4 * horizon)) ** np.arange(horizon)

        discount = 0.95 ** np.arange(horizon)
        ensemble_costs *= discount[None, None, :]

        # average over ensemble and horizon dimensions to get per-sample cost
        total_costs = ensemble_costs.mean(axis=(0, 2))
        total_costs -= total_costs.min()
        total_costs /= total_costs.max()

        return total_costs

    def get_action(self):
        return None, None


class RandomShootingPolicy(MPCPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, initial_state, goals):
        horizon = self.params["horizon"]
        n_samples = self.params["sample_trajectories"]
        robot_goals = self.params["robot_goals"]

        sampled_actions = np.random.uniform(-1, 1, size=(n_samples, horizon, self.action_dim))
        predicted_state_sequence = self.simulate(initial_state, sampled_actions)
        total_costs = self.compute_total_costs(predicted_state_sequence, sampled_actions, goals, robot_goals)

        best_idx = total_costs.argmin()
        best_action = sampled_actions[best_idx, 0]
        predicted_next_state = predicted_state_sequence[:, best_idx, 0].squeeze()

        return best_action, predicted_next_state


class CEMPolicy(MPCPolicy):
    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    def get_action(self, initial_state, goals):
        horizon = self.params["horizon"]
        n_samples = self.params["sample_trajectories"]
        refine_iters = self.params["refine_iters"]
        alpha = self.params["alpha"]
        n_best = self.params["n_best"]
        robot_goals = self.params["robot_goals"]
        action_trajectory_dim = self.action_dim * horizon

        trajectory_mean = np.zeros(action_trajectory_dim)
        trajectory_std = np.zeros(action_trajectory_dim)
        sampled_actions = np.random.uniform(-1, 1, size=(n_samples, horizon, self.action_dim))

        for i in range(refine_iters):
            if i > 0:
                sampled_actions = np.random.normal(loc=trajectory_mean, scale=trajectory_std, size=(n_samples, action_trajectory_dim))
                sampled_actions = sampled_actions.reshape(n_samples, horizon, self.action_dim)

            predicted_state_sequence = self.simulate(initial_state, sampled_actions)
            total_costs = self.compute_total_costs(predicted_state_sequence, sampled_actions, goals, robot_goals)

            action_trajectories = sampled_actions.reshape((n_samples, action_trajectory_dim))
            best_costs_idx = np.argsort(-total_costs)[-n_best:]
            best_trajectories = action_trajectories[best_costs_idx]
            best_trajectories_mean = best_trajectories.mean(axis=0)
            best_trajectories_std = best_trajectories.std(axis=0)

            trajectory_mean = alpha * best_trajectories_mean + (1 - alpha) * trajectory_mean
            trajectory_std = alpha * best_trajectories_std + (1 - alpha) * trajectory_std

            if trajectory_std.max() < 0.02:
                break

        best_action = trajectory_mean[:self.action_dim]
        predicted_next_state = self.simulate(initial_state, best_action[None, None, :]).squeeze()

        return best_action, predicted_next_state


class MPPIPolicy(MPCPolicy):
    def __init__(self, **kwargs):
        self.trajectory_mean = None
        return super().__init__(**kwargs)

    def get_action(self, initial_state, goals):
        horizon = self.params["horizon"]
        n_samples = self.params["sample_trajectories"]
        beta = self.params["beta"]
        gamma = self.params["gamma"]
        noise_std = self.params["noise_std"]
        robot_goals = self.params["robot_goals"]

        if self.trajectory_mean is None:
            self.trajectory_mean = np.zeros((horizon, self.action_dim))

        just_executed_action = self.trajectory_mean[0].copy()
        self.trajectory_mean[:-1] = self.trajectory_mean[1:]

        sampled_actions = np.empty((n_samples, horizon, self.action_dim))
        noise = np.random.normal(loc=0, scale=noise_std, size=(n_samples, horizon, self.action_dim))

        for t in range(horizon):
            if t == 0:
                sampled_actions[:, t] = beta * (self.trajectory_mean[t] + noise[:, t]) \
                                            + (1 - beta) * just_executed_action
            else:
                sampled_actions[:, t] = beta * (self.trajectory_mean[t] + noise[:, t]) \
                                            + (1 - beta) * sampled_actions[:, t-1]

        sampled_actions = np.clip(sampled_actions, -1, 1)
        predicted_state_sequence = self.simulate(initial_state, sampled_actions)
        total_costs = self.compute_total_costs(predicted_state_sequence, sampled_actions, goals, robot_goals)

        action_trajectories = sampled_actions.reshape((n_samples, -1))

        weights = np.exp(gamma * -total_costs)
        weighted_trajectories = (weights[:, None] * action_trajectories).sum(axis=0)
        self.trajectory_mean = weighted_trajectories / weights.sum()

        best_action = self.trajectory_mean[:self.action_dim]
        predicted_next_state = self.simulate(initial_state, best_action[None, None, :]).mean(axis=0).squeeze()

        # if np.linalg.norm(best_action) < np.sqrt(2) * 0.5:
        #     import pdb;pdb.set_trace()

        return best_action, predicted_next_state
