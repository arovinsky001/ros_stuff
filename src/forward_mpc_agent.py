import argparse
from cmath import nan
import pickle as pkl
from pdb import set_trace

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sympy import Q

import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
from torchcontrib.optim import SWA
from tqdm import trange, tqdm
from time import time

from sim.scripts.generate_data import *

pi = torch.pi

def dcn(*args):
    ret = []
    for arg in args:
        ret.append(arg.detach().cpu().numpy())
    return ret if len(ret) > 1 else ret[0]

def to_device(*args):
    ret = []
    for arg in args:
        ret.append(arg.to(device))
    return ret if len(ret) > 1 else ret[0]

def to_tensor(*args):
    ret = []
    for arg in args:
        if type(arg) == np.ndarray:
            ret.append(tensor(arg.astype('float32'), requires_grad=True))
        else:
            ret.append(arg)
    return ret if len(ret) > 1 else ret[0]

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class DynamicsNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, lr=7e-4, std=0.0, dropout=0.5, delta=False):
        super(DynamicsNetwork, self).__init__()
        input_dim = state_dim + action_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.1),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim, momentum=0.1),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.model.apply(init_weights)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        # self.optimizer = SWA(adam, swa_start=10, swa_freq=5, swa_lr=0.05)
        # self.optimizer.param_groups = self.optimizer.optimizer.param_groups
        # self.optimizer.state = self.optimizer.optimizer.state
        # self.optimizer.defaults=self.optimizer.optimizer.defaults

        self.loss_fn = nn.MSELoss(reduction='none')
        self.delta = delta
        self.dist = (std > 0)
        self.std = torch.ones(state_dim).to(device) * std
        self.input_scaler = None
        self.output_scaler = None

    def forward(self, state, action):
        state, action = to_tensor(state, action)
        if len(state.shape) == 1:
            state = state[None, :]
        if len(action.shape) == 1:
            action = action[None, :]

        state_action = torch.cat([state, action], dim=-1).float()
        if self.dist:
            mean = self.model(state_action)
            std = self.std
            return torch.distributions.normal.Normal(mean, std)
        else:
            pred = self.model(state_action)
            return pred
        

    def update(self, state, action, next_state, retain_graph=False):
        self.train()
        state, action, next_state = to_tensor(state, action, next_state)
        
        if self.dist:
            dist = self(state, action)
            prediction = dist.rsample()
            if self.delta:
                # pred_next_state = state + prediction
                losses = self.loss_fn(state + prediction, next_state)
            else:
                # pred_next_state = prediction
                losses = self.loss_fn(prediction, next_state)
        else:
            if self.delta:
                state_delta = self(state, action)
                # pred_next_state = state + state_delta
                losses = self.loss_fn(state + state_delta, next_state)
            else:
                pred_next_state = self(state, action)
                losses = self.loss_fn(pred_next_state, next_state)
        # losses[:, :2] *= 4
        # angle = torch.atan2(next_state[:, 2], next_state[:, 3])
        # pred_angle = torch.atan2(pred_next_state[:, 2], pred_next_state[:, 3])
        # angle_diff = (angle - pred_angle) % (2 * pi)
        # pos_err = (next_state - pred_next_state)[:, :2].norm(dim=-1)
        # angle_err = torch.stack((angle_diff, 2 * pi - angle_diff)).min(dim=0)[0]
        # losses = 100 * pos_err + angle_err
        loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
        return dcn(losses)

    def set_scalers(self, states, actions, next_states):
        with torch.no_grad():
            self.input_scaler = StandardScaler().fit(np.append(states, actions, axis=-1))
            self.output_scaler = StandardScaler().fit(next_states)
    
    def get_scaled(self, *args):
        np_type = True
        arglist = list(args)
        for i, arg in enumerate(arglist):
            if not isinstance(arg, np.ndarray):
                np_type = False
                arglist[i] = dcn(arg)
        if len(arglist) == 2:
            states, actions = arglist
            if len(states.shape) == 1:
                states = states[None, :]
            if len(actions.shape) == 1:
                actions = actions[None, :]
            states_actions = np.append(states, actions, axis=-1)
            states_actions_scaled = self.input_scaler.transform(states_actions)
            states_scaled = states_actions_scaled[:, :states.shape[-1]]
            actions_scaled = states_actions_scaled[:, states.shape[-1]:]
            if not np_type:
                states_scaled, actions_scaled = to_tensor(states_scaled, actions_scaled)
            return states_scaled, actions_scaled
        else:
            next_states = arglist[0]
            next_states_scaled = self.output_scaler.transform(next_states)
            if not np_type:
                next_states_scaled = to_tensor(next_states_scaled)
            return next_states_scaled


class MPCAgent:
    def __init__(self, state_dim, action_dim, seed=1, hidden_dim=512, lr=7e-4, std=0.0, dropout=0.5, delta=True, scale=True):
        self.model = DynamicsNetwork(state_dim, action_dim, hidden_dim=hidden_dim, lr=lr, std=std, dropout=dropout, delta=delta)
        self.model.to(device)
        self.seed = seed
        self.action_dim = action_dim
        self.mse_loss = nn.MSELoss(reduction='none')
        self.neighbors = []
        self.state = None
        self.delta = delta
        self.scale = scale
        self.time = 0

    def mpc_action(self, state, init, goal, state_range, action_range, n_steps=10, n_samples=1000,
                   swarm=False, swarm_weight=0.1, perp_weight=0.0, angle_weight=0.0, forward_weight=0.0):
        state, init, goal, state_range = to_tensor(state, init, goal, state_range)
        self.state = state
        all_actions = torch.empty(n_steps, n_samples, self.action_dim).uniform_(*action_range)
        states = torch.tile(state, (n_samples, 1))
        goals = torch.tile(goal, (n_samples, 1))
        states, all_actions, goals = to_tensor(states, all_actions, goals)
        x1, y1, _ = init
        x2, y2, _ = goal
        vec_to_goal = (goal - init)[:-1]
        optimal_dot = vec_to_goal / vec_to_goal.norm()
        perp_denom = vec_to_goal.norm()
        all_losses = torch.empty(n_steps, n_samples)

        for i in range(n_steps):
            actions = all_actions[i]
            with torch.no_grad():
                states = self.get_prediction(states, actions)
            states = torch.clamp(states, *state_range)

            x0, y0, _ = states.T
            vecs_to_goal = (goals - states)[:, :-1]
            actual_dot = vecs_to_goal.T / vecs_to_goal.norm(dim=-1)
            
            dist_loss = torch.norm((goals - states)[:, :-1], dim=-1)
            perp_loss = torch.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / perp_denom / dist_loss.mean()
            angle_loss = torch.arccos(torch.clamp(optimal_dot @ actual_dot, -1., 1.))
            forward_loss = torch.abs(optimal_dot @ vecs_to_goal.T)
            swarm_loss = self.swarm_loss(states, goals) if swarm else 0

            perp_loss = perp_loss.reshape(dist_loss.shape)
            angle_loss = angle_loss.reshape(dist_loss.shape)
            forward_loss = forward_loss.reshape(dist_loss.shape)
            print("perp:", perp_loss.mean(), "|| forward:", forward_loss.mean(), "|| dist:", dist_loss.mean())

            all_losses[i] = dist_loss * 0 + forward_weight * forward_loss + perp_weight * perp_loss \
                                + angle_weight * angle_loss + swarm_weight * swarm_loss
        
        # best_idx = all_losses.sum(dim=0).argmin()
        best_idx = all_losses[-1].argmin()
        # import pdb;pdb.set_trace()
        return all_actions[0, best_idx]
    
    def get_prediction(self, states, actions, scale=True):
        if self.scale and scale:
            states, actions = self.model.get_scaled(states, actions)
        states, actions = to_tensor(states, actions)
        states, actions = to_device(states, actions)
        with torch.no_grad():
            model_output = self.model(states, actions)
        if self.model.dist:
            if self.delta:
                states_delta = model_output.loc
                next_states = states_delta + states
            else:
                next_states = model_output.loc
        else:
            if self.delta:
                states_delta = model_output
                next_states = states_delta + states
            else:
                next_states = model_output
        next_states = dcn(next_states)
        if self.scale and scale:
            next_states = self.model.output_scaler.inverse_transform(next_states)
        return next_states

    def swarm_loss(self, states, goals):
        neighbor_dists = torch.empty(len(self.neighbors), states.shape[0])
        for i, neighbor in enumerate(self.neighbors):
            neighbor_states = torch.tile(neighbor.state, (states.shape[0], 1))
            distance = torch.norm(states - neighbor_states, dim=-1)
            neighbor_dists[i] = distance
        goal_term = torch.norm(goals - states, dim=-1)
        loss = neighbor_dists.mean(dim=0) * goal_term.mean()
        return loss

    def train(self, states, actions, next_states, epochs=5, batch_size=256):
        if self.scale:
            states, actions = self.model.get_scaled(states, actions)
            next_states = self.model.get_scaled(next_states)
        states, actions, next_states = to_tensor(states, actions, next_states)
        train_states, test_states, train_actions, test_actions, train_next_states, test_next_states \
            = train_test_split(states, actions, next_states, test_size=0.2, random_state=self.seed)

        training_losses = []
        test_losses = []
        test_idx = []
        n_batches = np.ceil(len(train_states) / batch_size).astype("int")
        idx = np.arange(len(train_states))

        for i in tqdm(range(-1, epochs), desc="Epoch", position=0, leave=False):
            np.random.shuffle(idx)
            train_states, train_actions, train_next_states = train_states[idx], train_actions[idx], train_next_states[idx]
            
            if i == -1:
                self.model.eval()
                with torch.no_grad():
                    pred_next_states = to_tensor(self.get_prediction(test_states, test_actions))
                test_loss = self.mse_loss(pred_next_states, test_next_states)
                test_loss_mean = dcn(test_loss.mean())
                training_losses.append(np.zeros(1))
                test_losses.append(test_loss_mean)
                test_idx.append(i)
                tqdm.write(f"{i}: mean training loss: {nan} | mean test loss: {test_loss_mean}")
                self.model.train()

            for j in tqdm(range(n_batches), desc="Batch", position=1, leave=False):
                batch_states = torch.autograd.Variable(train_states[j*batch_size:(j+1)*batch_size])
                batch_actions = torch.autograd.Variable(train_actions[j*batch_size:(j+1)*batch_size])
                batch_next_states = torch.autograd.Variable(train_next_states[j*batch_size:(j+1)*batch_size])
                batch_states, batch_actions, batch_next_states = to_device(batch_states, batch_actions, batch_next_states)
                training_loss = self.model.update(batch_states, batch_actions, batch_next_states)
                if type(training_loss) != float:
                    while len(training_loss.shape) > 1:
                        training_loss = training_loss.mean(axis=-1)
        
            training_loss_mean = training_loss.mean()
            training_losses.append(training_loss_mean)
            self.model.eval()
            with torch.no_grad():
                pred_next_states = to_tensor(self.get_prediction(test_states, test_actions))
            test_loss = self.mse_loss(pred_next_states, test_next_states)
            test_loss_mean = dcn(test_loss.mean())
            test_losses.append(test_loss_mean)
            test_idx.append(i)
            tqdm.write(f"{i}: mean training loss: {training_loss_mean} | mean test loss: {test_loss_mean}")
            self.model.train()
        
        # self.model.optimizer.swap_swa_sgd()    
        self.model.eval()
        return training_losses, test_losses, test_idx, test_states, test_actions, test_next_states

    def optimal_policy(self, state, goal, table, swarm=False, swarm_weight=0.3):
        if swarm:
            vec = goal - state
            states = tensor(state + table[:, 1, None])
            neighbor_dists = []
            for neighbor in self.neighbors:
                neighbor_states = torch.tile(neighbor.state, (states.shape[0], 1))
                distance = self.mse_loss(states, neighbor_states)
                neighbor_dists.append(dcn(distance))
            neighbor_dists = np.array(neighbor_dists)
            mean_dists = neighbor_dists.mean(axis=0)
            goals = np.tile(goal, (len(states), 1))
            goal_dists = self.mse_loss(states, goals)
            costs = goal_dists + swarm_weight * mean_dists
        else:
            vec = goal - state
            diff = abs(vec - table[:, 1, None])
            min_idx = diff.argmin(axis=0)
        return table[min_idx, 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/load agent and do MPC.')
    parser.add_argument('-load_agent_path', type=str,
                        help='path/file to load old agent from')
    parser.add_argument('-save_agent_path', type=str,
                        help='path/file to save newly-trained agent to')
    parser.add_argument('-new_agent', '-n', action='store_true',
                        help='flag to train new agent')
    parser.add_argument('-hidden_dim', type=int, default=512,
                        help='hidden layers dimension')
    parser.add_argument('-epochs', type=int, default=10,
                        help='number of training epochs for new agent')
    parser.add_argument('-batch_size', type=int, default=128,
                        help='batch size for training new agent')
    parser.add_argument('-learning_rate', type=float, default=7e-4,
                        help='batch size for training new agent')
    parser.add_argument('-seed', type=int, default=1,
                        help='random seed for numpy and pytorch')
    parser.add_argument('-stochastic', action='store_true',
                        help='flag to use stochastic transition data')
    parser.add_argument('-std', type=float, default=0.0,
                        help='standard deviation for model distribution')
    parser.add_argument('-delta', action='store_true',
                        help='flag to output state delta')
    parser.add_argument('-real', action='store_true',
                        help='flag to use real data')
    parser.add_argument('-dropout', type=float, default=0.5,
                        help='dropout probability')
    parser.add_argument('-generate_data', type=int, default=0,
                        help='how many iterations of data generation to do')
    parser.add_argument('-scale', action='store_true',
                        help='flag to preprocess data with standard scaler')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.hidden_dim > 1024:
        if torch.backends.mps.is_available:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if args.real:
        agent_path = 'agents/real.pkl'
        data = np.load("sim/data/real_data.npz")
    else:
        agent_path = 'agents/'
        if args.stochastic:
            data = np.load("sim/data/data_stochastic.npz")
        else:
            data = np.load("sim/data/data_deterministic.npz")
    
    states = data['states']
    actions = data['actions']
    next_states = data['next_states']

    shift = 1
    # actions = actions[shift:]
    # states = states[:-shift]
    # next_states = next_states[:-shift]

    # actions = actions[:-shift]
    # states = states[shift:]
    # next_states = next_states[shift:]
    # set_trace()

    plotstart = 10
    plotend = plotstart + 20

    actions_plot = actions[plotstart:plotend]

    states_x = states[plotstart:plotend, 0]
    states_y = states[plotstart:plotend, 1]
    next_states_x = next_states[plotstart:plotend, 0]
    next_states_y = next_states[plotstart:plotend, 1]

    states_sin = states[plotstart:plotend, 2]
    states_cos = states[plotstart:plotend, 3]
    next_states_sin = next_states[plotstart:plotend, 2]
    next_states_cos = next_states[plotstart:plotend, 3]
    states_theta = np.arctan2(states_sin, states_cos)
    states_theta += pi
    states_sin = np.sin(states_theta)
    states_cos = np.cos(states_theta)
    next_states_theta = np.arctan2(next_states_sin, next_states_cos)
    next_states_theta += pi
    next_states_sin = np.sin(next_states_theta)
    next_states_cos = np.cos(next_states_theta)
    # next_states_theta = np.arctan2(next_states_sin, next_states_cos)

    plt.quiver(states_x[1:], states_y[1:], -states_cos[1:], -states_sin[1:], color="green")
    plt.quiver(next_states_x[:-1], next_states_y[:-1], -next_states_cos[:-1], -next_states_sin[:-1], color="purple")
    plt.plot(states_x[1:], states_y[1:], color="green", linewidth=1.0)
    plt.plot(next_states_x[:-1], next_states_y[:-1], color="purple", linewidth=1.0)
    for i, (x, y) in enumerate(zip(states_x, states_y)):
        if i == 0:
            continue
        plt.annotate(f"{i-1}", # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
        
        plt.annotate(str(actions_plot[i]), # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(-10,-10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    for i, (x, y) in enumerate(zip(next_states_x, next_states_y)):
        if i == len(next_states_x) - 1:
            continue
        label = f"{i}"
        plt.annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.show()

    # n_actions = 3
    # length = len(actions) - n_actions + 1
    # new_actions = actions
    # for i in range(n_actions):
    #     new_actions = np.append(new_actions[:-1], actions[i+1:], axis=-1)
    # actions = new_actions
    # states = states[n_actions:]
    # next_states = next_states[n_actions:]
    
    # length = len(states) if len(states) % n_actions == 0 else len(states) - (len(states) % n_actions)
    # states = states[:length:n_actions]
    # actions = actions[:length].reshape(-1, actions.shape[-1] * n_actions)
    # next_states = next_states[:length:n_actions]

    new_states = states.copy()
    new_actions = actions.copy()
    new_next_states = next_states.copy()

    for _ in range(args.generate_data):
        state_shift = np.random.uniform(low=-1.0, high=1.0, size=(len(states), 2))
        shifted_states = states[:, :2] + state_shift
        shifted_next_states = next_states[:, :2] + state_shift

        shifted_states = np.append(shifted_states, states[:, 2:], axis=1)
        shifted_next_states = np.append(shifted_next_states, next_states[:, 2:], axis=1)

        new_states = np.append(new_states, shifted_states, axis=0)
        new_actions = np.append(new_actions, actions, axis=0)
        new_next_states = np.append(new_next_states, shifted_next_states, axis=0)
        
    states = new_states
    actions = new_actions
    next_states = new_next_states

    # normalize = True
    # if normalize:
    #     states_mean = states.mean(axis=0)
    #     states_std = states.std(axis=0)
    #     actions_mean = actions.mean(axis=0)
    #     actions_std = actions.std(axis=0)
    #     next_states_mean = next_states.mean(axis=0)
    #     next_states_std = next_states.std(axis=0)

    #     states = (states - states.mean(axis=0)) / states.std(axis=0)
    #     actions = (actions - actions.mean(axis=0)) / actions.std(axis=0)
    #     next_states = (next_states - next_states.mean(axis=0)) / next_states.std(axis=0)

    # next_states_min = next_states.min(axis=0)
    # next_states_max = next_states.max(axis=0) - next_states_min

    # states -= states.min(axis=0)
    # actions -= actions.min(axis=0)
    # next_states -= next_states.min(axis=0)

    # states /= states.max(axis=0)
    # actions /= actions.max(axis=0)
    # next_states /= next_states.max(axis=0)
    np.set_printoptions(suppress=True)

    # idx = np.argwhere(np.all(abs(actions) <= 0.4, axis=-1)).squeeze()
    # zero_diff = abs(states[idx] - next_states[idx])
    # bad_idx = np.argwhere(np.linalg.norm(zero_diff, axis=-1) > zero_diff.mean()).squeeze()

    # states = np.delete(states, bad_idx, axis=0)
    # actions = np.delete(actions, bad_idx, axis=0)
    # next_states = np.delete(next_states, bad_idx, axis=0)

    # diff_idx = np.argwhere(actions[:, 0] * actions[:, 1] < 0).squeeze()
    # same_idx = np.argwhere(actions[:, 0] * actions[:, 1] > 0).squeeze()
    # np.linalg.norm((states - next_states)[diff_idx, :2], axis=-1).mean()
    # np.linalg.norm((states - next_states)[same_idx, :2], axis=-1).mean()

    # bad_idx1 = np.argwhere(np.linalg.norm((states - next_states)[diff_idx, :2], axis=-1) > 0.02)
    # bad_idx2 = np.argwhere(np.linalg.norm((states - next_states)[same_idx, :2], axis=-1) < 0.05)
    # bad_idx = np.unique(np.append(bad_idx1, bad_idx2, axis=0))
    # set_trace()
    # states = np.delete(states, bad_idx, axis=0)
    # actions = np.delete(actions, bad_idx, axis=0)
    # next_states = np.delete(next_states, bad_idx, axis=0)

    # idx = np.argwhere(np.all(actions > 0, axis=-1)).squeeze()
    # states = states[idx]
    # actions = actions[idx]
    # next_states = next_states[idx]
    # import pdb;pdb.set_trace()

    # if args.real:
    #     bad_idx1 = np.linalg.norm((states - next_states)[:, :2], axis=-1) > 0.2
    #     bad_idx2 = np.linalg.norm((states - next_states)[:, :2], axis=-1) < 0.005
    #     bad_idx = bad_idx1 | bad_idx2
    #     set_trace()
    #     states = states[~bad_idx]
    #     actions = actions[~bad_idx]
    #     next_states = next_states[~bad_idx]
    
    print('\nDATA LOADED\n')

    if not args.real:
        agent_path += f"epochs{args.epochs}"
        agent_path += f"_dim{args.hidden_dim}"
        agent_path += f"_batch{args.batch_size}"
        agent_path += f"_lr{args.learning_rate}"
        if args.dropout > 0:
            agent_path += f"_dropout{args.dropout}"
        if args.std > 0:
            agent_path += f"_std{args.std}"
        if args.stochastic:
            agent_path += "_stochastic"
        if args.delta:
            agent_path += "_delta"
        agent_path += ".pkl"

    if args.new_agent:
        agent = MPCAgent(states.shape[-1], actions.shape[-1], seed=args.seed, std=args.std,
                         delta=args.delta, scale=args.scale, hidden_dim=args.hidden_dim,
                         lr=args.learning_rate, dropout=args.dropout)
        
        if args.scale:
            agent.model.set_scalers(states, actions, next_states)
        
        # if normalize:
        #     agent.states_mean = states_mean
        #     agent.states_std = states_std
        #     agent.actions_mean = actions_mean
        #     agent.actions_std = actions_std
        #     agent.next_states_mean = next_states_mean
        #     agent.next_states_std = next_states_std

        training_losses, test_losses, test_idx, test_states, test_actions, test_next_states = agent.train(
                        states, actions, next_states,
                        epochs=args.epochs, batch_size=args.batch_size)

        training_losses = np.array(training_losses).squeeze()
        test_losses = np.array(test_losses).squeeze()
        ignore = 20
        print("\nMIN TEST LOSS EPOCH:", test_losses[ignore:].argmin() + ignore)
        print("MIN TEST LOSS:", test_losses[ignore:].min())
        plt.plot(np.arange(len(training_losses)), training_losses, label="Training Loss")
        plt.plot(test_idx, test_losses, label="Test Loss")
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Dynamics Model Loss')
        plt.legend()
        plt.grid()
        plt.show()
        
        agent_path = args.save_agent_path if args.save_agent_path else agent_path
        with open(agent_path, "wb") as f:
            pkl.dump(agent, f)
    else:
        agent_path = args.load_agent_path if args.load_agent_path else agent_path
        with open(agent_path, "rb") as f:
            agent = pkl.load(f)

    # np.where((np.all(abs(actions) > 0.7, axis=1) & (actions[:, 0] * actions[:, 1] > 0)))[0]
    # agent.get_prediction(np.array([[-0.5, -0.5, 0.0, 1.0]]), np.array([[0.8, 0.8]]))

    # 50 epochs, normalized, no action thing, dropout 0.5
    # ERROR MEAN: tensor([0.0236, 0.0250, 0.3964, 0.3574], dtype=torch.float64)
    # ERROR STD: tensor([0.0209, 0.0204, 0.3397, 0.2594], dtype=torch.float64)
    # ERROR MAX: tensor([0.0889, 0.0935, 1.4369, 1.1629], dtype=torch.float64)
    # ERROR MIN: tensor([2.4030e-06, 4.3542e-04, 7.1151e-03, 6.9733e-03], dtype=torch.float64)

    agent.model.eval()
    test_states, test_actions, test_next_states = \
        dcn(test_states, test_actions, test_next_states)
    diffs = []
    pred_next_states = agent.get_prediction(test_states, test_actions, scale=False)

    # if normalize:
    #     pred_next_states = pred_next_states * next_states_std + next_states_mean
    #     test_states = test_states.detach() * states_std + states_mean
    #     test_next_states = test_next_states.detach() * next_states_std + next_states_mean

    if args.scale:
        pred_next_states = agent.model.output_scaler.inverse_transform(pred_next_states)
        test_next_states = agent.model.output_scaler.inverse_transform(test_next_states)
    
    error = (pred_next_states - test_next_states) ** 2
    print("\nERROR MEAN:", error.mean(axis=0))
    print("ERROR STD:", error.std(axis=0))
    print("ERROR MAX:", error.max(axis=0)[0])
    print("ERROR MIN:", error.min(axis=0)[0])

    # diffs = abs(test_states - test_next_states)
    diffs = (states - next_states) ** 2
    print("\nACTUAL MEAN:", diffs.mean(axis=0))
    print("ACTUAL STD:", diffs.std(axis=0))
    # set_trace()

    state = states[0]
    slist = []
    alist = []
    l = 10
    for i in range(l):
        action = actions[i]
        slist.append(state.squeeze())
        alist.append(action.squeeze())
        state = agent.get_prediction(state, action)

    slist = np.array(slist)
    alist = np.array(alist)
    
    set_trace()
    plt.quiver(slist[:, 0], slist[:, 1], -slist[:, 2], -slist[:, 3], color="green")
    plt.quiver(states[:l, 0], states[:l, 1], -states[:l, 2], -states[:l, 3], color="purple")
    plt.plot(slist[:, 0], slist[:, 1], color="green", linewidth=1.0)
    plt.plot(states[:l, 0], states[:l, 1], color="purple", linewidth=1.0)

    for i, (x, y) in enumerate(zip(slist[:, 0], slist[:, 1])):
        plt.annotate(f"{i}", # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center
        
        # plt.annotate(str(actions_plot[i]), # this is the text
        #             (x,y), # these are the coordinates to position the label
        #             textcoords="offset points", # how to position the text
        #             xytext=(-10,-10), # distance from text to points (x,y)
        #             ha='center') # horizontal alignment can be left, right or center

    for i, (x, y) in enumerate(zip(states[:l, 0], states[:l, 1])):
        label = f"{i}"
        plt.annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.show()
    
    state_range = np.array([MIN_STATE, MAX_STATE])
    action_range = np.array([MIN_ACTION, MAX_ACTION])

    potential_actions = np.linspace(MIN_ACTION, MAX_ACTION, 10000)
    potential_deltas = FUNCTION(potential_actions)
    TABLE = np.block([potential_actions.reshape(-1, 1), potential_deltas.reshape(-1, 1)])

    # MPC parameters
    n_steps = 1         # length per sample trajectory
    n_samples = 100    # number of trajectories to sample
    forward_weight = 0.0
    perp_weight = 0.0
    angle_weight = 0.0

    # run trials testing the MPC policy against the optimal policy
    n_trials = 1000
    max_steps = 200
    success_threshold = 1.0
    plot = False

    # n_trials = 2
    # max_steps = 120
    # start = np.array([11.1, 18.7])
    # goal = np.array([90.3, 71.5])
    # optimal_losses = np.empty(max_steps)
    # actual_losses = np.empty((n_trials, max_steps))
    # noises = np.random.normal(loc=0.0, scale=NOISE_STD, size=(max_steps, 2))
    # state = start.copy()

    # i = 0
    # while not np.linalg.norm(goal - state) < 0.2:
    #     optimal_losses[i] = np.linalg.norm(goal - state)
    #     noise = noises[i]
    #     state += FUNCTION(agent.optimal_policy(state, goal, TABLE)) + noise
    #     state = np.clip(state, MIN_STATE, MAX_STATE)
    #     i += 1
    # optimal_losses[i:] = np.linalg.norm(goal - state)

    # for k in trange(n_trials):
    #     state = start.copy()
    #     i = 0
    #     while not np.linalg.norm(goal - state) < 0.2:
    #         actual_losses[k, i] = np.linalg.norm(goal - state)
    #         noise = noises[i]
    #         action = agent.mpc_action(state, goal, state_range, action_range,
    #                                 n_steps=n_steps, n_samples=n_samples).detach().numpy()
    #         state += FUNCTION(action) + noise
    #         state = np.clip(state, MIN_STATE, MAX_STATE)
    #         i += 1
    #     actual_losses[k, i:] = np.linalg.norm(goal - state)
    
    # plt.plot(np.arange(max_steps), optimal_losses, 'g-', label="Optimal Controller")
    # plt.plot(np.arange(max_steps), actual_losses.mean(axis=0), 'b-', label="MPC Controller")
    # plt.title("Optimal vs MPC Controller Performance")
    # plt.legend()
    # plt.xlabel("Step\n\nstart = [11.1, 18.7], goal = [90.3, 71.5]\nAveraged over 20 MPC runs")
    # plt.ylabel("Distance to Goal")
    # plt.grid()
    # # plt.text(0.5, 0.01, "start = [11.1, 18.7], goal = [90.3, 71.5]", wrap=True, ha='center', fontsize=12)
    # plt.show()
    # set_trace()



    # perp_weights = np.linspace(0, 5., 3)
    # angle_weights = np.linspace(0, 1., 3)
    # forward_weights = np.linspace(0, 1., 3)
    # init_min, init_max = 20., 22.
    # goal_min, goal_max = 70., 72.
    # init_states = np.random.rand(n_trials, 2) * (init_max - init_min) + init_min
    # goals = np.random.rand(n_trials, 2) * (goal_max - goal_min) + goal_min
    # noises = np.random.normal(loc=0.0, scale=NOISE_STD, size=(n_trials, max_steps, 2))
    # results = []

    # optimal_lengths = np.empty(n_trials)
    # for trial in trange(n_trials, leave=False):
    #     init_state = init_states[trial]
    #     goal = goals[trial]

    #     state = init_state.copy()
    #     i = 0
    #     while not np.linalg.norm(goal - state) < success_threshold:
    #         noise = noises[trial, i] if args.stochastic else 0.0
    #         state += FUNCTION(agent.optimal_policy(state, goal, TABLE)) + noise
    #         if LIMIT:
    #             state = np.clip(state, MIN_STATE, MAX_STATE)
    #         i += 1
    #     optimal_lengths[trial] = i

    # count = 0
    # while True:
    #     for p in perp_weights:
    #         for a in angle_weights:
    #             for f in forward_weights:
    #                 print(f"{count}: f{f}, p{p}, a{a}")
    #                 count += 1
    #                 actual_lengths = np.empty(n_trials)
    #                 optimal = 0
    #                 # all_states = []
    #                 # all_actions = []
    #                 # all_goals = []
    #                 for trial in trange(n_trials):
    #                     init_state = init_states[trial]
    #                     goal = goals[trial]
                        
    #                     state = init_state.copy()
    #                     i = 0
    #                     states, actions = [], []
    #                     while not np.linalg.norm(goal - state) < success_threshold:
    #                         states.append(state)
    #                         if i == max_steps:
    #                             break
    #                         action = agent.mpc_action(state, goal, state_range, action_range,
    #                                             n_steps=n_steps, n_samples=n_samples, perp_weight=p,
    #                                             angle_weight=a, forward_weight=f).detach().numpy()
    #                         noise = noises[trial, i] if args.stochastic else 0.0
    #                         state += FUNCTION(action) + noise
    #                         if LIMIT:
    #                             state = np.clip(state, MIN_STATE, MAX_STATE)
    #                         i += 1
    #                         actions.append(action)

    #                     # all_states.append(states)
    #                     # all_actions.append(actions)
    #                     # all_goals.append(goal)
    #                     actual_lengths[trial] = i

    #                     if i <= optimal_lengths[trial]:
    #                         optimal += 1
                    
    #                 actual_lengths = np.array(actual_lengths)
    #                 results.append(np.abs(optimal_lengths.mean() - actual_lengths.mean()) / optimal_lengths.mean())
        
    #     # print("\noptimal mean:", optimal_lengths.mean())
    #     # print("optimal std:", optimal_lengths.std(), "\n")
    #     # print("actual mean:", actual_lengths.mean())
    #     # print("actual std:", actual_lengths.std(), "\n")
    #     # print("mean error:", np.abs(optimal_lengths.mean() - actual_lengths.mean()) / optimal_lengths.mean())
    #     # print("optimality rate:", optimal / float(n_trials))
    #     # print("timeout rate:", (actual_lengths == max_steps).sum() / float(n_trials), "\n")
        
    #     # if plot:
    #     #     plt.hist(optimal_lengths)
    #     #     plt.plot(optimal_lengths, actual_lengths, 'bo')
    #     #     plt.xlabel("Optimal Steps to Reach Goal")
    #     #     plt.ylabel("Actual Steps to Reach Goal")
    #     #     plt.show()
    #     print(np.argsort(results))
    #     set_trace()



    init_min, init_max = 20., 21.
    goal_min, goal_max = 70., 71.
    init_states = np.random.rand(n_trials, 2) * (init_max - init_min) + init_min
    goals = np.random.rand(n_trials, 2) * (goal_max - goal_min) + goal_min
    noises = np.random.normal(loc=0.0, scale=NOISE_STD, size=(n_trials, max_steps, 2))

    optimal_lengths = np.empty(n_trials)
    for trial in trange(n_trials):
        init_state = init_states[trial]
        goal = goals[trial]

        state = init_state.copy()
        i = 0
        while not np.linalg.norm(goal - state) < success_threshold:
            noise = noises[trial, i] if args.stochastic else 0.0
            state += FUNCTION(agent.optimal_policy(state, goal, TABLE)) + noise
            if LIMIT:
                state = np.clip(state, MIN_STATE, MAX_STATE)
            i += 1
        optimal_lengths[trial] = i

    while True:
        actual_lengths = np.empty(n_trials)
        optimal = 0
        # all_states = []
        # all_actions = []
        # all_goals = []
        for trial in trange(n_trials, leave=False):
            init_state = init_states[trial]
            goal = goals[trial]
            
            state = init_state.copy()
            i = 0
            states, actions = [], []
            while not np.linalg.norm(goal - state) < success_threshold:
                states.append(state)
                if i == max_steps:
                    break
                action = agent.mpc_action(state, init_state, goal, state_range, action_range,
                                    n_steps=n_steps, n_samples=n_samples, perp_weight=perp_weight,
                                    angle_weight=angle_weight, forward_weight=forward_weight).detach().numpy()
                noise = noises[trial, i] if args.stochastic else 0.0
                state += FUNCTION(action) + noise
                if LIMIT:
                    state = np.clip(state, MIN_STATE, MAX_STATE)
                i += 1
                actions.append(action)

            # all_states.append(states)
            # all_actions.append(actions)
            # all_goals.append(goal)
            actual_lengths[trial] = i

            if i <= optimal_lengths[trial]:
                optimal += 1
        
        optimal_lengths, actual_lengths = np.array(optimal_lengths), np.array(actual_lengths)
        print("\n------------------------")
        print("optimal mean:", optimal_lengths.mean())
        print("optimal std:", optimal_lengths.std(), "\n")
        print("actual mean:", actual_lengths.mean())
        print("actual std:", actual_lengths.std(), "\n")
        print("mean error:", np.abs(optimal_lengths.mean() - actual_lengths.mean()) / optimal_lengths.mean())
        print("optimality rate:", optimal / float(n_trials))
        print("timeout rate:", (actual_lengths == max_steps).sum() / float(n_trials))
        print("------------------------\n")

        if plot:
            plt.hist(optimal_lengths)
            plt.plot(optimal_lengths, actual_lengths, 'bo')
            plt.xlabel("Optimal Steps to Reach Goal")
            plt.ylabel("Actual Steps to Reach Goal")
            plt.show()
        set_trace()
