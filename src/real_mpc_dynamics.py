import argparse
from cmath import nan
import pickle as pkl
from pdb import set_trace

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
from torchcontrib.optim import SWA
from tqdm import trange, tqdm
from time import time

from sim.scripts.generate_data import *

pi = torch.pi
device = torch.device("cpu")

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

def to_tensor(*args, requires_grad=True):
    ret = []
    for arg in args:
        if type(arg) == np.ndarray:
            ret.append(tensor(arg.astype('float32'), requires_grad=requires_grad))
        else:
            ret.append(arg)
    return ret if len(ret) > 1 else ret[0]

class DynamicsNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, lr=1e-3, dropout=0.5, entropy_weight=0.02, dist=True):
        super(DynamicsNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.1),
            nn.GELU(),
            nn.Dropout(p=dropout),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            # nn.BatchNorm1d(hidden_dim, momentum=0.1),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * output_dim if dist else output_dim),
        )
        self.output_dim = output_dim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.dist = dist
        self.entropy_weight = entropy_weight
        self.input_scaler = None
        self.output_scaler = None
        self._init_weights()

    def _init_weights(self):
        if isinstance(self.model, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            self.model.bias.data.fill_(0.01)

    def forward(self, state, action):
        state, action = to_tensor(state, action)
        if len(state.shape) == 1:
            state = state[None, :]
        if len(action.shape) == 1:
            action = action[None, :]

        state_action = torch.cat([state, action], dim=-1).float()
        if self.dist:
            mean_std = self.model(state_action)
            mean = mean_std[:, :self.output_dim]
            std = mean_std[:, self.output_dim:]
            std = torch.clamp(std, min=1e-6)
            try:
                return torch.distributions.normal.Normal(mean, std)
            except:
                set_trace()
        else:
            pred = self.model(state_action)
            return pred
        

    def update(self, state, action, state_delta, retain_graph=False):
        self.train()
        state, action, state_delta = to_tensor(state, action, state_delta)
        
        if self.dist:
            dist = self(state, action)
            pred_state_delta = dist.rsample()
            losses = self.loss_fn(pred_state_delta, state_delta)
            losses -= dist.entropy() * self.entropy_weight
        else:
            pred_state_delta = self(state, action)
            losses = self.loss_fn(pred_state_delta, state_delta)
        loss = losses.mean()
        # sin = pred_state_delta[:, 2] + state[:, 0]
        # cos = pred_state_delta[:, 3] + state[:, 1]
        # loss += (self.loss_fn(sin**2 + cos**2, torch.ones_like(sin)) * 0.2).mean()

        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
        return dcn(losses)

    def set_scalers(self, states, actions, states_delta):
        with torch.no_grad():
            self.input_scaler = StandardScaler().fit(np.append(states, actions, axis=-1))
            self.output_scaler = StandardScaler().fit(states_delta)
    
    def get_scaled(self, *args):
        np_type = True
        arglist = list(args)
        for i, arg in enumerate(arglist):
            if not isinstance(arg, np.ndarray):
                np_type = False
                arglist[i] = dcn(arg)
        if len(arglist) == 2:
            # This means args are state and action
            states, actions = arglist
            if len(states.shape) == 1:
                states = states[None, :]
            if len(actions.shape) == 1:
                actions = actions[None, :]
            try:
                states_actions = np.append(states, actions, axis=-1)
            except:
                set_trace()
            states_actions_scaled = self.input_scaler.transform(states_actions)
            states_scaled = states_actions_scaled[:, :states.shape[-1]]
            actions_scaled = states_actions_scaled[:, states.shape[-1]:]
            if not np_type:
                states_scaled, actions_scaled = to_tensor(states_scaled, actions_scaled)
            return states_scaled, actions_scaled
        else:
            # This means args is states_delta
            states_delta = arglist[0]
            states_delta_scaled = self.output_scaler.transform(states_delta)
            if not np_type:
                states_delta_scaled = to_tensor(states_delta)
            return states_delta_scaled


class MPCAgent:
    def __init__(self, input_dim, output_dim, seed=1, hidden_dim=512, lr=7e-4, dropout=0.5, entropy_weight=0.02, dist=True, scale=True):
        self.model = DynamicsNetwork(input_dim, output_dim, hidden_dim=hidden_dim, lr=lr, dropout=dropout, entropy_weight=entropy_weight, dist=dist)
        self.model.to(device)
        self.seed = seed
        self.mse_loss = nn.MSELoss(reduction='none')
        self.neighbors = []
        self.state = None
        self.scale = scale
        self.time = 0

    def mpc_action(self, state, init, goal, prev_actions, state_range, action_range, swarm=False, n_steps=10, n_samples=1000,
                   swarm_weight=0.0, perp_weight=0.4, heading_weight=0.17, forward_weight=0.0, dist_weight=1.0, norm_weight=0.1):
        state, init, goal, prev_actions = to_tensor(state, init, goal, prev_actions)
        self.state = state      # for multi-robot (swarming)
        all_actions = torch.empty(n_steps, n_samples, 2).uniform_(*action_range)
        states = torch.tile(state, (n_samples, 1))
        goals = torch.tile(goal, (n_samples, 1))
        prev_actions = torch.tile(prev_actions.flatten(), (n_samples, 1))
        x1, y1, _ = init
        x2, y2, _ = goal
        vec_to_goal = (goal - init)[:2]
        optimal_dot = vec_to_goal / vec_to_goal.norm()
        perp_denom = vec_to_goal.norm()
        all_losses = torch.empty(n_steps, n_samples)

        for i in range(n_steps):
            actions = all_actions[i]
            actions = torch.cat((prev_actions, actions), dim=-1)
            with torch.no_grad():
                states = to_tensor(self.get_prediction(states, actions), requires_grad=False)

            # heading computations
            x0, y0, current_angle = states.T
            vecs_to_goal = (goals - states)[:, :2]
            target_angle1 = torch.atan2(vecs_to_goal[:, 1], vecs_to_goal[:, 0])
            target_angle2 = torch.atan2(-vecs_to_goal[:, 1], -vecs_to_goal[:, 0])
            angle_diff1 = (target_angle1 - current_angle) % (2 * torch.pi)
            angle_diff2 = (target_angle2 - current_angle) % (2 * torch.pi)
            angle_diff1 = torch.stack((angle_diff1, 2 * torch.pi - angle_diff1)).min(dim=0)[0]
            angle_diff2 = torch.stack((angle_diff2, 2 * torch.pi - angle_diff2)).min(dim=0)[0]
            
            # compute losses
            dist_loss = torch.norm((goals - states)[:, :2], dim=-1).squeeze()
            norm_const = dist_loss.mean() / vec_to_goal.norm()
            # dist_loss[dist_loss < 0.15] *= 0.2
            heading_loss = torch.stack((angle_diff1, angle_diff2)).min(dim=0)[0].squeeze()
            perp_loss = (torch.abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / perp_denom).squeeze()
            forward_loss = torch.abs(optimal_dot @ vecs_to_goal.T).squeeze()
            norm_loss = -all_actions[i, :, :-1].norm(dim=-1).squeeze() if i == 0 else 0.0
            swarm_loss = self.swarm_loss(states, goals).squeeze() if swarm else 0.0

            # normalize appropriate losses and compute total loss
            all_losses[i] = norm_const * (perp_weight * perp_loss + heading_weight * heading_loss \
                                + swarm_weight * swarm_loss + norm_weight * norm_loss) \
                                + dist_weight * dist_loss + forward_weight * forward_loss
        
        # find index of best trajectory and return corresponding first action
        best_idx = all_losses.sum(dim=0).argmin()
        return all_actions[0, best_idx]
    
    def get_prediction(self, states_xy, actions, scale=True, sample=True, delta=False):
        """
        Accepts state [x, y, theta]
        Returns next_state [x, y, theta]
        """
        states_xy, actions = to_tensor(states_xy, actions)
        thetas = states_xy[:, -1]
        sc = torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=1)

        if self.scale and scale:
            sc, actions = self.model.get_scaled(sc, actions)

        sc, actions = to_tensor(sc, actions)
        sc, actions = to_device(sc, actions)

        with torch.no_grad():
            model_output = self.model(sc, actions)

        if self.model.dist:
            if sample:
                states_delta = model_output.rsample()
            else:
                states_delta = model_output.loc
        else:
            states_delta = model_output

        if self.scale and scale:
            states_delta = self.model.output_scaler.inverse_transform(states_delta)

        if delta:
            return states_delta

        states_sc = torch.cat([states_xy[:, :-1], sc], dim=-1)
        next_states_sc = states_sc + states_delta
        n_thetas = torch.atan2(next_states_sc[:, 2], next_states_sc[:, 3]).reshape(-1, 1)
        next_states = torch.cat([next_states_sc[:, :2], n_thetas], dim=-1)
        next_states = dcn(next_states)
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

    def train(self, states, actions, next_states, epochs=5, batch_size=256, set_scalers=False):
        states, actions, next_states = to_tensor(states, actions, next_states)
        sc = torch.stack([torch.sin(states[:, -1]), torch.cos(states[:, -1])], dim=1)
        states_sc = torch.cat([states[:, :-1], sc], dim=-1)
        sc = torch.stack([torch.sin(next_states[:, -1]), torch.cos(next_states[:, -1])], dim=1)
        next_states_sc = torch.cat([next_states[:, :-1], sc], dim=-1)
        states_delta = next_states_sc - states_sc

        idx = np.arange(len(states))
        train_states, test_states, train_actions, test_actions, train_states_delta, test_states_delta, \
                train_idx, test_idx = train_test_split(states, actions, states_delta, idx, test_size=0.1, random_state=self.seed)
        
        train_states = torch.stack([torch.sin(train_states[:, -1]), torch.cos(train_states[:, -1])], dim=1)

        if set_scalers:
            agent.model.set_scalers(train_states, train_actions, train_states_delta)

        if self.scale:
            train_states, train_actions = self.model.get_scaled(train_states, train_actions)
            train_states_delta = self.model.get_scaled(train_states_delta)
            # train_states += np.random.normal(0.0, 0.05, size=train_states.shape)
            # train_actions += np.random.normal(0.0, 0.05, size=train_actions.shape)
            # train_next_states += np.random.normal(0.0, 0.05, size=train_next_states.shape)
        train_states, train_actions, train_states_delta = to_tensor(train_states, train_actions, train_states_delta)
        test_states, test_actions, test_states_delta = to_tensor(test_states, test_actions, test_states_delta)

        training_losses = []
        test_losses = []
        n_batches = np.ceil(len(train_states) / batch_size).astype("int")
        idx = np.arange(len(train_states))

        self.model.eval()
        with torch.no_grad():
            pred_states_delta = to_tensor(self.get_prediction(test_states, test_actions, sample=False, delta=True))
        test_loss = self.mse_loss(pred_states_delta, test_states_delta)
        test_loss_mean = dcn(test_loss.mean())
        test_losses.append(test_loss_mean)
        tqdm.write(f"Pre-Train: mean test loss: {test_loss_mean}")
        self.model.train()

        for i in tqdm(range(-1, epochs), desc="Epoch", position=0, leave=False):
            np.random.shuffle(idx)
            train_states, train_actions, train_states_delta = train_states[idx], train_actions[idx], train_states_delta[idx]                

            for j in tqdm(range(n_batches), desc="Batch", position=1, leave=False):
                batch_states = torch.autograd.Variable(train_states[j*batch_size:(j+1)*batch_size])
                batch_actions = torch.autograd.Variable(train_actions[j*batch_size:(j+1)*batch_size])
                batch_states_delta = torch.autograd.Variable(train_states_delta[j*batch_size:(j+1)*batch_size])
                batch_states, batch_actions, batch_states_delta = to_device(batch_states, batch_actions, batch_states_delta)
                training_loss = self.model.update(batch_states, batch_actions, batch_states_delta)
                if type(training_loss) != float:
                    while len(training_loss.shape) > 1:
                        training_loss = training_loss.mean(axis=-1)
        
            training_loss_mean = training_loss.mean()
            training_losses.append(training_loss_mean)
            self.model.eval()
            with torch.no_grad():
                pred_states_delta = to_tensor(self.get_prediction(test_states, test_actions, sample=False, delta=True))
            test_loss = self.mse_loss(pred_states_delta, test_states_delta)
            test_loss_mean = dcn(test_loss.mean())
            test_losses.append(test_loss_mean)
            tqdm.write(f"{i+1}: mean training loss: {training_loss_mean} | mean test loss: {test_loss_mean}")
            self.model.train()
        
        self.model.eval()
        return training_losses, test_losses, test_idx

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
    parser.add_argument('-dist', action='store_true',
                        help='flag to have the model output a distribution')
    parser.add_argument('-real', action='store_true',
                        help='flag to use real data')
    parser.add_argument('-dropout', type=float, default=0.5,
                        help='dropout probability')
    parser.add_argument('-scale', action='store_true',
                        help='flag to preprocess data with standard scaler')
    parser.add_argument('-save', action='store_true',
                        help='flag to save model after training')
    parser.add_argument('-retrain', action='store_true',
                        help='flag to load existing model and continue training')
    parser.add_argument('-entropy', type=float, default=0.02,
                        help='weight for entropy term in training loss function')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.hidden_dim >= 1024:
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
    
    print('\nDATA LOADED\n')
    
    states = data['states']
    actions = data['actions']
    next_states = data['next_states']

    # TODO: TEMPORARY
    thetas = np.arctan2(states[:, -2], states[:, -1])
    states = np.append(states[:, :2], thetas[:, None], axis=-1)
    thetas = np.arctan2(next_states[:, -2], next_states[:, -1])
    next_states = np.append(next_states[:, :2], thetas[:, None], axis=-1)
    actions = actions[:, -3:]

    if args.retrain:
        online_data = np.load("sim/data/real_data_online.npz")
    
        online_states = online_data['states']
        online_actions = online_data['actions']
        online_next_states = online_data['next_states']

        n_repeat = int(len(states) / len(online_states))
        n_repeat = 1 if n_repeat == 0 else n_repeat
        online_states = np.tile(online_states, (n_repeat, 1))
        online_actions = np.tile(online_actions, (n_repeat, 1))
        online_next_states = np.tile(online_next_states, (n_repeat, 1))

        states = np.append(states, online_states, axis=0)
        actions = np.append(actions, online_actions, axis=0)
        next_states = np.append(next_states, online_next_states, axis=0)

        states = online_states
        actions = online_actions
        next_states = online_next_states

    plot_data = False
    if plot_data:
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
    
    if not args.real:
        agent_path += f"epochs{args.epochs}"
        agent_path += f"_dim{args.hidden_dim}"
        agent_path += f"_batch{args.batch_size}"
        agent_path += f"_lr{args.learning_rate}"
        if args.dropout > 0:
            agent_path += f"_dropout{args.dropout}"
        if args.dist:
            agent_path += "_dist"
        if args.stochastic:
            agent_path += "_stochastic"
        agent_path += ".pkl"

    if args.new_agent:
        agent = MPCAgent(2 + actions.shape[-1], 4, seed=args.seed, dist=args.dist,
                         scale=args.scale, hidden_dim=args.hidden_dim, lr=args.learning_rate,
                         dropout=args.dropout, entropy_weight=args.entropy)

        batch_sizes = [args.batch_size, args.batch_size * 10, args.batch_size * 100, args.batch_size * 1000]
        batch_sizes = [args.batch_size]
        for batch_size in batch_sizes:
            training_losses, test_losses, test_idx = agent.train(states, actions, next_states, set_scalers=True,
                                                       epochs=args.epochs, batch_size=batch_size)

            training_losses = np.array(training_losses).squeeze()
            test_losses = np.array(test_losses).squeeze()
            print("\nMIN TEST LOSS EPOCH:", test_losses.argmin())
            print("MIN TEST LOSS:", test_losses.min())
            plt.plot(np.arange(len(training_losses)), training_losses, label="Training Loss")
            plt.plot(np.arange(-1, len(test_losses)-1), test_losses, label="Test Loss")
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Dynamics Model Loss')
            plt.legend()
            plt.grid()
            plt.show()
        
        if args.save:
            print("\nSAVING MPC AGENT\n")
            agent_path = args.save_agent_path if args.save_agent_path else agent_path
            with open(agent_path, "wb") as f:
                pkl.dump(agent, f)
    else:
        agent_path = args.load_agent_path if args.load_agent_path else agent_path
        with open(agent_path, "rb") as f:
            agent = pkl.load(f)

        agent.model.eval()
        diffs = []
        pred_next_states = agent.get_prediction(test_states, test_actions)
        
        error = abs(pred_next_states - test_states_delta)
        print("\nERROR MEAN:", error.mean(axis=0))
        print("ERROR STD:", error.std(axis=0))
        print("ERROR MAX:", error.max(axis=0))
        print("ERROR MIN:", error.min(axis=0))

        diffs = abs(test_states - test_states_delta)
        print("\nACTUAL MEAN:", diffs.mean(axis=0))
        print("ACTUAL STD:", diffs.std(axis=0))
        set_trace()
        
        if args.retrain:
            training_losses, test_losses = agent.train(train_states, train_actions, train_states_delta,
                                                   test_states, test_actions, test_states_delta,
                                                   epochs=args.epochs, batch_size=args.batch_size)

            training_losses = np.array(training_losses).squeeze()
            test_losses = np.array(test_losses).squeeze()
            ignore = 20
            print("\nMIN TEST LOSS EPOCH:", test_losses[ignore:].argmin() + ignore)
            print("MIN TEST LOSS:", test_losses[ignore:].min())
            plt.plot(np.arange(len(training_losses)), training_losses, label="Training Loss")
            plt.plot(np.arange(-1, len(test_losses)-1), test_losses, label="Test Loss")
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Dynamics Model Loss')
            plt.legend()
            plt.grid()
            plt.show()

            if args.save:
                print("\nSAVING MPC AGENT\n")
                agent_path = args.save_agent_path if args.save_agent_path else agent_path
                with open(agent_path, "wb") as f:
                    pkl.dump(agent, f)

    agent.model.eval()

    states, actions, next_states = to_tensor(states, actions, next_states)
    sc = torch.stack([torch.sin(states[:, -1]), torch.cos(states[:, -1])], dim=1)
    states_sc = torch.cat([states[:, :-1], sc], dim=-1)
    sc = torch.stack([torch.sin(next_states[:, -1]), torch.cos(next_states[:, -1])], dim=1)
    next_states_sc = torch.cat([next_states[:, :-1], sc], dim=-1)
    states_delta = next_states_sc - states_sc

    test_states, test_actions = states[test_idx], actions[test_idx]
    # set_trace()
    test_states_delta = dcn(states_delta[test_idx])
    pred_states_delta = agent.get_prediction(test_states, test_actions, sample=False, delta=True)
    
    error = (pred_states_delta - test_states_delta)
    print("\nERROR MEAN:", error.mean(axis=0))
    print("ERROR STD:", error.std(axis=0))
    print("ERROR MAX:", error.max(axis=0))
    print("ERROR MIN:", error.min(axis=0))

    diffs = abs(test_states_delta)
    print("\nACTUAL MEAN:", diffs.mean(axis=0))
    print("ACTUAL STD:", diffs.std(axis=0))

    set_trace()

    for k in range(20):
        slist = []
        alist = []
        start, end = 10 * k, 10 * k + 10
        state = states[start]
        for i in range(start, end):
            action = actions[i]
            slist.append(state.squeeze())
            alist.append(action.squeeze())
            state = agent.get_prediction(state, action)
            state[2:] = np.clip(state[2:], -1., 1.)

        slist = np.array(slist)
        alist = np.array(alist)
        
        plt.quiver(slist[:, 0], slist[:, 1], -slist[:, 2], -slist[:, 3], color="green")
        plt.quiver(states[start:end, 0], states[start:end, 1], -states[start:end, 2], -states[start:end, 3], color="purple")
        plt.plot(slist[:, 0], slist[:, 1], color="green", linewidth=1.0, label="Predicted Trajectory")
        plt.plot(states[start:end, 0], states[start:end, 1], color="purple", linewidth=1.0, label="Actual Trajectory")

        for i, (x, y) in enumerate(zip(slist[:, 0], slist[:, 1])):
            plt.annotate(f"{i}", # this is the text
                        (x,y), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center

        for i, (x, y) in enumerate(zip(states[start:end, 0], states[start:end, 1])):
            label = f"{i}"
            plt.annotate(label, # this is the text
                        (x,y), # these are the coordinates to position the label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center

        plt.legend()
        plt.show()
        set_trace()
