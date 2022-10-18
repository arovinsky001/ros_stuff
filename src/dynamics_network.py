#!/usr/bin/python3

import torch
from torch import nn
from torch.nn import functional as F

import data_utils as dtu


class DynamicsNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, global_dtu, hidden_dim=256, hidden_depth=2, lr=1e-3, dropout=0.5, std=0.02, dist=True, use_object=False, scale=True):
        super(DynamicsNetwork, self).__init__()
        assert hidden_depth >= 1
        input_layer = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        hidden_layers = []
        for _ in range(hidden_depth):
            hidden_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            hidden_layers += [nn.BatchNorm1d(hidden_dim, momentum=0.1)]
        output_layer = [nn.Linear(hidden_dim, output_dim)]
        layers = input_layer + hidden_layers + output_layer
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.scale = scale
        self.dist = dist
        self.use_object = use_object
        self.std = std
        self.input_scaler = None
        self.output_scaler = None
        self.net.apply(self._init_weights)
        self.dtu = global_dtu

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

    def forward(self, state, action, sample=False, return_dist=False, delta=True):
        state, action = dtu.as_tensor(state, action)

        if len(state.shape) == 1:
            state = state[None, :]
        if len(action.shape) == 1:
            action = action[None, :]

        # state comes in as (x, y, theta)
        input_state = self.dtu.state_to_model_input(state)
        state_action = torch.cat([input_state, action], dim=1).float()
        if self.scale:
            state_action = self.standardize_input(state_action)

        if self.dist:
            mean = self.net(state_action)
            std = torch.ones_like(mean) * self.std
            dist = torch.distributions.normal.Normal(mean, std)
            if return_dist:
                return dist
            state_delta_model = dist.rsample() if sample else mean
        else:
            state_delta_model = self.net(state_action)

        if self.scale:
            state_delta_model = self.unstandardize_output(state_delta_model)

        if delta:
            return state_delta_model

        next_state_model = self.dtu.compute_next_state(state, state_delta_model)
        return next_state_model

    def update(self, state, action, next_state):
        self.train()
        state, action, next_state = dtu.as_tensor(state, action, next_state)

        state_delta = self.dtu.state_delta_xysc(state, next_state).detach()

        if self.dist:
            if self.scale:
                state_delta = self.standardize_output(state_delta)
            dist = self(state, action, return_dist=True)
            loss = -dist.log_prob(state_delta).mean()
        else:
            pred_state_delta = self(state, action)
            loss = F.mse_loss(pred_state_delta, state_delta, reduction='mean')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return dtu.dcn(loss)

    def set_scalers(self, state, action, next_state):
        state, action, next_state = dtu.as_tensor(state, action, next_state)
        input_state = self.dtu.state_to_model_input(state)

        state_action = torch.cat([input_state, action], axis=1)
        state_delta = self.dtu.state_delta_xysc(state, next_state)

        self.input_mean = state_action.mean(dim=0)
        self.input_std = state_action.std(dim=0)

        self.output_mean = state_delta.mean(dim=0)
        self.output_std = state_delta.std(dim=0)

    def standardize_input(self, model_input):
        return (model_input - self.input_mean) / self.input_std

    def standardize_output(self, model_output):
        return (model_output - self.output_mean) / self.output_std

    def unstandardize_output(self, model_output):
        return model_output * self.output_std + self.output_mean
