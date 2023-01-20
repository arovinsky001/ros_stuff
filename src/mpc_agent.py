#!/usr/bin/python

import numpy as np
import torch
import hydra

from dynamics_network import DynamicsNetwork
from mpc_policies import MPPIPolicy, CEMPolicy, RandomShootingPolicy
from utils import DataUtils, signed_angle_difference, dimensions


class MPCAgent:
    def __init__(self, hidden_dim, hidden_depth, lr, std, dist, scale, ensemble, use_object, mpc_config):
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

        self.policy = hydra.utils.instantiate(mpc_config)

        self.policy(simulate_fn=self.simulate, cost_fn=self.compute_costs,
                             params=mpc_params, cost_weights_dict=cost_weights_dict)

        self.scale = scale
        self.ensemble = ensemble
        self.use_object = use_object
        self.trained = False

    @property
    def model(self):
        return self.models[0]

    def get_action(self, state, goals):
        return self.policy.get_action(state, goals)

    def simulate(self, initial_state, action_sequence):
        n_samples, horizon, _ = action_sequence.shape
        initial_state = np.tile(initial_state, (n_samples, 1))
        state_sequence = np.empty((len(self.models), n_samples, horizon, self.state_dim))

        for i, model in enumerate(self.models):
            model.eval()

            for t in range(horizon):
                action = action_sequence[:, t]
                with torch.no_grad():
                    if t == 0:
                        state_sequence[i, :, t] = model(initial_state, action, sample=False, delta=False)
                    else:
                        state_sequence[i, :, t] = model(state_sequence[i, :, t-1], action, sample=False, delta=False)

        return state_sequence
