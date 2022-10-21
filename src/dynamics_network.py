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
        self.hidden_depth = hidden_depth
        self.scale = scale
        self.dist = dist
        self.use_object = use_object
        self.std = std
        self.input_scaler = None
        self.output_scaler = None
        self.net.apply(self._init_weights)
        self.dtu = global_dtu
        self.update_lr = torch.tensor(lr)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.lr_optimizer = torch.optim.Adam([self.update_lr], lr=lr)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

    def forward(self, states, actions, sample=False, return_dist=False, delta=True):
        states, actions = dtu.as_tensor(states, actions)

        if len(states.shape) == 1:
            states = states[None, :]
        if len(actions.shape) == 1:
            actions = actions[None, :]

        # state comes in as (x, y, theta)
        input_states = self.dtu.state_to_model_input(states)
        inputs = torch.cat([input_states, actions], dim=1).float()
        if self.scale:
            inputs = self.standardize_input(inputs)

        if self.dist:
            mean = self.net(inputs)
            std = torch.ones_like(mean) * self.std
            dist = torch.distributions.normal.Normal(mean, std)
            if return_dist:
                return dist
            state_deltas_model = dist.rsample() if sample else mean
        else:
            state_deltas_model = self.net(inputs)

        if self.scale:
            state_deltas_model = self.unstandardize_output(state_deltas_model)

        if delta:
            return state_deltas_model

        next_states_model = self.dtu.compute_next_state(states, state_deltas_model)
        return next_states_model

    def forward_from_params(self, updated_params, states, actions, sample=False, return_dist=False, delta=True):
        states, actions = dtu.as_tensor(states, actions)

        if len(states.shape) == 1:
            states = states[None, :]
        if len(actions.shape) == 1:
            actions = actions[None, :]

        # state comes in as (x, y, theta)
        input_states = self.dtu.state_to_model_input(states)
        x = torch.cat([input_states, actions], dim=1).float()
        if self.scale:
            x = self.standardize_input(x)

        for i in range(self.hidden_depth + 1):
            w, b = updated_params[2*(i+1)], updated_params[2*(i+1)+1]
            x = F.relu(F.linear(x, w, b))

        w, b = updated_params[-2], updated_params[-1]

        if self.dist:
            mean = F.linear(x, w, b)
            std = torch.ones_like(mean) * self.std
            dist = torch.distributions.normal.Normal(mean, std)
            if return_dist:
                return dist
            state_deltas_model = dist.rsample() if sample else mean
        else:
            state_deltas_model = F.linear(x, w, b)

        if self.scale:
            state_deltas_model = self.unstandardize_output(state_deltas_model)

        if delta:
            return state_deltas_model

        next_states_model = self.dtu.compute_next_state(states, state_deltas_model)
        return next_states_model

    def update(self, state, action, next_state):
        self.train()
        state, action, next_state = dtu.as_tensor(state, action, next_state)

        state_delta = self.dtu.compute_state_delta(state, next_state).detach()

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

    def update_meta(self, state, action, next_state):
        self.train()
        state, action, next_state = dtu.as_tensor(state, action, next_state)
        state_delta = self.dtu.compute_state_delta(state_delta)

        # chunk into "tasks" i.e. batches continuous in time
        tasks = 10
        meta_update_steps = 3
        batched_states = torch.stack(state.chunk(tasks, dim=0))
        batched_actions = torch.stack(action.chunk(tasks, dim=0))
        batched_state_deltas = torch.stack(state_delta.chunk(tasks, dim=0))
        test_losses = 0

        for i in tasks:
            train_states, test_states = batched_states[i].chunk(2, dim=0)
            train_actions, test_actions = batched_actions[i].chunk(2, dim=0)
            train_state_deltas, test_state_deltas = batched_state_deltas[i].chunk(2, dim=0)

            for j in range(meta_update_steps):
                # compute prediction and corresponding training loss
                if self.dist:
                    if self.scale:
                        train_state_deltas = self.standardize_output(train_state_deltas)
                        test_state_deltas = self.standardize_output(test_state_deltas)
                    dist = self(train_states, train_actions, return_dist=True)
                    loss = -dist.log_prob(train_state_deltas).mean()
                else:
                    pred_state_delta = self(train_states, train_actions)
                    loss = F.mse_loss(pred_state_delta, train_state_deltas, reduction='mean')

                # compute gradient and simulate gradient step
                params = self.net.parameters() if j == 0 else updated_params
                grad = torch.autograd.grad(loss, params)
                updated_params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, params)))

                # only consider last-step losses for main network update
                if j == meta_update_steps - 1:
                    if self.dist:
                        test_dist = self.forward_from_params(updated_params, test_states, test_actions, return_dist=True)
                        test_losses -= test_dist.log_prob(test_state_deltas)
                    else:
                        pred_state_delta = self.forward_from_params(updated_params, test_states, test_actions)
                        test_losses += F.mse_loss(pred_state_delta, test_state_deltas)

        # update network
        self.optimizer.zero_grad()
        test_losses.backward(retain_graph=True)
        self.optimizer.step()

        # update learning rate
        self.lr_optimizer.zero_grad()
        test_losses.backward()
        self.lr_optimizer.zero_grad()

        return dtu.dcn(test_losses)

    def set_scalers(self, state, action, next_state):
        state, action, next_state = dtu.as_tensor(state, action, next_state)
        input_state = self.dtu.state_to_model_input(state)

        state_action = torch.cat([input_state, action], axis=1)
        state_delta = self.dtu.compute_state_delta(state, next_state)

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
