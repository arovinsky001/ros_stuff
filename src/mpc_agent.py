#!/usr/bin/python3

import numpy as np
import torch

from dynamics_network import DynamicsNetwork
from mpc_policies import MPPIPolicy, CEMPolicy, RandomShootingPolicy
from utils import DataUtils, signed_angle_difference, dimensions


class MPCAgent:
    def __init__(self, seed=1, mpc_method='mppi', hidden_dim=200, hidden_depth=2, lr=0.001,
                 std=0.01, dist=True, scale=True, ensemble=1, use_object=False,
                 action_range=None, device=torch.device("cpu")):
        assert ensemble > 0

        input_dim = dimensions["action_dim"]
        output_dim = dimensions["robot_output_dim"]
        self.state_dim = dimensions["state_dim"]

        if use_object:
            output_dim += dimensions["object_output_dim"]
            self.state_dim += dimensions["state_dim"]

        self.dtu = DataUtils(use_object=use_object)
        self.models = [DynamicsNetwork(input_dim, output_dim, self.dtu, hidden_dim=hidden_dim, hidden_depth=hidden_depth, lr=lr, std=std, dist=dist, use_object=use_object, scale=scale)
                                for _ in range(ensemble)]
        for model in self.models:
            model.to(device)

        if mpc_method == 'mppi':
            policy = MPPIPolicy
        elif mpc_method == 'cem':
            policy = CEMPolicy
        elif mpc_method == 'shooting':
            policy = RandomShootingPolicy
        else:
            raise NotImplementedError
        self.policy = policy(action_range=action_range, simulate_fn=self.simulate, cost_fn=self.compute_costs)

        self.seed = seed
        self.scale = scale
        self.ensemble = ensemble
        self.use_object = use_object

    @property
    def model(self):
        return self.models[0]

    def get_action(self, state, prev_goal, goal, cost_weights, params):
        return self.policy.get_action(state, prev_goal, goal, cost_weights, params)

    def simulate(self, initial_state, action_sequence):
        n_samples, horizon, _ = action_sequence.shape
        initial_state = np.tile(initial_state, (n_samples, 1))
        state_sequence = np.empty((len(self.models), n_samples, horizon, self.state_dim))

        for i, model in enumerate(self.models):
            for t in range(horizon):
                action = action_sequence[:, t]
                with torch.no_grad():
                    if t == 0:
                        state_sequence[i, :, t] = model(initial_state, action, sample=False, delta=False)
                    else:
                        state_sequence[i, :, t] = model(state_sequence[i, :, t-1], action, sample=False, delta=False)

        return state_sequence

    def compute_costs(self, state, action, prev_goal, goal, robot_goals=False, signed=False):
        # separation cost
        if self.use_object:
            robot_state = state[:, :, :, :dimensions["state_dim"]]
            object_state = state[:, :, :, dimensions["state_dim"]:2*dimensions["state_dim"]]
            object_to_robot_xy = (robot_state - object_state)[:, :, :, :-1]
            sep_cost = np.linalg.norm(object_to_robot_xy, axis=-1)

            effective_state = robot_state if robot_goals else object_state
        else:
            effective_state = state[:, :, :, :dimensions["state_dim"]]
            sep_cost = np.array([0.])

        x0, y0, current_angle = effective_state.transpose(3, 0, 1, 2)
        vec_to_goal = (goal - effective_state)[:, :, :, :-1]

        # dist cost
        dist_cost = np.linalg.norm(vec_to_goal, axis=-1)
        if signed:
            dist_cost *= forward

        # heading cost
        target_angle = np.arctan2(vec_to_goal[:, :, :, 1], vec_to_goal[:, :, :, 0])
        heading_cost = signed_angle_difference(target_angle, current_angle)

        left = (heading_cost > 0) * 2 - 1
        forward = (np.abs(heading_cost) < np.pi / 2) * 2 - 1

        heading_cost[forward == -1] = (heading_cost[forward == -1] + np.pi) % (2 * np.pi)
        heading_cost = np.stack((heading_cost, 2 * np.pi - heading_cost)).min(axis=0)

        if signed:
            heading_cost *= left * forward
        else:
            heading_cost = np.abs(heading_cost)

        # heading_cost[np.abs(dist_cost) < 0.015] = 0.

        # perp cost
        x1, y1, _ = prev_goal
        x2, y2, _ = goal
        perp_denom = np.linalg.norm((goal - prev_goal)[:2])

        perp_cost = (((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / perp_denom)

        if not signed:
            perp_cost = np.abs(perp_cost)

        # object vs. robot heading cost
        if self.use_object:
            robot_theta, object_theta = robot_state[:, :, :, -1], object_state[:, :, :, -1]
            heading_diff = (robot_theta - object_theta) % (2 * np.pi)
            heading_diff_cost = np.stack((heading_diff, 2 * np.pi - heading_diff), axis=1).min(axis=1)
        else:
            heading_diff_cost = np.array([0.])

        # action norm cost
        norm_cost = -np.linalg.norm(action, axis=-1)

        cost_dict = {
            "distance": dist_cost,
            "heading": heading_cost,
            "perpendicular": perp_cost,
            "action_norm": norm_cost,
            "separation": sep_cost,
            "heading_difference": heading_diff_cost,
        }

        return cost_dict
