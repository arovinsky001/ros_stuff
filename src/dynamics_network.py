#!/usr/bin/python3

from matplotlib import test
import torch
from torch import nn
from torch.nn import functional as F

import data_utils as dtu


class DynamicsNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, global_dtu, hidden_dim=200, hidden_depth=1, lr=0.001, std=0.01, dist=True, use_object=False, scale=True):
        super(DynamicsNetwork, self).__init__()

        assert hidden_depth >= 1

        hidden_layers = []
        for _ in range(hidden_depth):
            hidden_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]

        input_layer = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
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
        self.update_lr = nn.parameter.Parameter(torch.tensor(lr))

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.lr_optimizer = torch.optim.Adam([self.update_lr], lr=lr)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

    def forward(self, states, actions, sample=False, return_dist=False, delta=False, params=None):
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

        if params is not None:
            # metalearning forward pass
            x = inputs
            for i in range(self.hidden_depth + 1):
                w, b = params[2*i], params[2*i+1]
                x = F.relu(F.linear(x, w, b))

            w, b = params[-2], params[-1]
            outputs = F.linear(x, w, b)
        else:
            # standard forward pass
            outputs = self.net(inputs)

        if self.dist:
            mean = outputs
            std = torch.ones_like(mean) * self.std
            dist = torch.distributions.normal.Normal(mean, std)
            if return_dist:
                return dist
            state_deltas_model = dist.rsample() if sample else mean
        else:
            state_deltas_model = outputs

        if self.scale:
            state_deltas_model = self.unstandardize_output(state_deltas_model)

        if delta:
            return state_deltas_model

        next_states_model = self.dtu.compute_next_state(states, state_deltas_model)
        return next_states_model

    def update(self, state, action, next_state):
        self.train()
        state, action, next_state = dtu.as_tensor(state, action, next_state)

        state_delta = self.dtu.state_delta_xysc(state, next_state).detach()

        ### TEMP ###
        offsets_per_sample = 40
        heading_offset = torch.rand(offsets_per_sample) * 2 * torch.pi
        offset_state = state.repeat(1, offsets_per_sample).reshape(len(state), offsets_per_sample, -1)
        offset_state[:, :, 2] = (offset_state[:, :, 2] + heading_offset) % (2 * torch.pi)

        sin, cos = torch.sin(heading_offset), torch.cos(heading_offset)
        rotation = torch.stack((torch.stack((cos, -sin)),
                                torch.stack((sin, cos)))).transpose(0, 2)
        tmp = rotation[:, 0, 1].clone()
        rotation[:, 0, 1] = rotation[:, 1, 0].clone()
        rotation[:, 1, 0] = tmp
        offset_state_delta = state_delta.repeat(1, offsets_per_sample).reshape(len(state_delta), offsets_per_sample, -1)
        offset_state_delta[:, :, :2] = (rotation @ offset_state_delta[:, :, :2, None]).squeeze()

        state_offset_heading = offset_state[:, :, 2].reshape(-1)
        next_state_repeat = next_state.repeat(1, offsets_per_sample).reshape(len(next_state), offsets_per_sample, -1)
        next_state_offset_heading = ((next_state_repeat[:, :, 2] + heading_offset) % (2 * torch.pi)).reshape(-1)
        offset_sin, offset_cos = torch.sin(state_offset_heading), torch.cos(state_offset_heading)
        next_offset_sin, next_offset_cos = torch.sin(next_state_offset_heading), torch.cos(next_state_offset_heading)
        sin_delta, cos_delta = next_offset_sin - offset_sin, next_offset_cos - offset_cos

        state = offset_state.reshape(-1, state.shape[-1])
        state_delta = offset_state_delta.reshape(-1, state_delta.shape[-1])
        action = action.repeat(1, offsets_per_sample).reshape(-1, action.shape[-1])

        state_delta[:, 2] = sin_delta
        state_delta[:, 3] = cos_delta
        ### TEMP ###

        if self.dist:
            if self.scale:
                state_delta = self.standardize_output(state_delta)
            dist = self(state, action, return_dist=True)
            loss = -dist.log_prob(state_delta).mean()
        else:
            pred_state_delta = self(state, action, delta=True)
            loss = F.mse_loss(pred_state_delta, state_delta, reduction='mean')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return dtu.dcn(loss)

    def update_meta(self, state, action, next_state):
        self.train()
        state, action, next_state = dtu.as_tensor(state, action, next_state)
        state_delta = self.dtu.state_delta_xysc(state, next_state)

        # chunk into "tasks" i.e. batches continuous in time
        tasks = 20
        task_steps = 100
        meta_update_steps = 2

        first_idxs = torch.randint(len(state) - 2 * task_steps + 1, (tasks,))
        tiled_first_idxs = first_idxs.reshape(-1, 1).repeat(1, task_steps)
        all_idxs = tiled_first_idxs + torch.arange(task_steps)

        batched_states = state[all_idxs]
        batched_actions = action[all_idxs]
        batched_state_deltas = state_delta[all_idxs]

        test_loss_sum = 0

        if self.dist and self.scale:
            batched_state_deltas = self.standardize_output(batched_state_deltas)

        for i in range(tasks):
            train_states, test_states = batched_states[i].chunk(2, dim=0)
            train_actions, test_actions = batched_actions[i].chunk(2, dim=0)
            train_state_deltas, test_state_deltas = batched_state_deltas[i].chunk(2, dim=0)

            params = list(self.net.parameters())

            for _ in range(meta_update_steps):
                # compute prediction and corresponding training loss
                if self.dist:
                    dist = self(train_states, train_actions, return_dist=True, params=params)
                    loss = -dist.log_prob(train_state_deltas).mean()
                else:
                    pred_state_deltas = self(train_states, train_actions, delta=True, params=params)
                    loss = F.mse_loss(pred_state_deltas, train_state_deltas, reduction='mean')

                # compute gradient and simulate gradient step
                grad = torch.autograd.grad(loss, params)
                params = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, params)))

            # only consider last-step losses for main network update
            if self.dist:
                test_dist = self(test_states, test_actions, return_dist=True, params=params)
                test_loss_sum -= test_dist.log_prob(test_state_deltas).mean()
            else:
                pred_state_deltas = self(test_states, test_actions, delta=True, params=params)
                test_loss_sum += F.mse_loss(pred_state_deltas, test_state_deltas)

        loss = test_loss_sum / tasks

        # update network
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        # update learning rate
        self.lr_optimizer.zero_grad()
        loss.backward()
        self.lr_optimizer.step()

        return dtu.dcn(test_loss_sum)

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
