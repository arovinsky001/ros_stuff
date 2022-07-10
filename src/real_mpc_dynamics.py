import argparse
import pickle as pkl
from pdb import set_trace

import numpy as np
from matplotlib import pyplot as plt
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch import tensor
from tqdm import trange, tqdm

pi = torch.pi
# device = torch.device("cpu")

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
    def __init__(self, input_dim, output_dim, hidden_dim=256, lr=1e-3, dropout=0.5, entropy_weight=0.02, dist=True, multi=False):
        super(DynamicsNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.1),
            nn.GELU(),
            nn.Dropout(p=dropout),
            # nn.Linear(hidden_dim, hidden_dim),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            # nn.Dropout(p=dropout),
            # nn.BatchNorm1d(hidden_dim, momentum=0.1),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * output_dim if dist else output_dim),
            # nn.utils.spectral_norm(nn.Linear(hidden_dim, 2 * output_dim if dist else output_dim)),
        )
        self.output_dim = output_dim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.dist = dist
        self.entropy_weight = entropy_weight
        self.multi = multi
        self.input_scaler = None
        self.output_scaler = None
        self.model.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, state, action, id=None):
        if self.multi:
            id = to_tensor(id)
        state, action = to_tensor(state, action)

        if len(state.shape) == 1:
            state = state[None, :]
        if len(action.shape) == 1:
            action = action[None, :]
        if self.multi and len(id.shape) == 1:
            id = id[None, :]

        state_action = torch.cat([state, action], dim=-1).float()
        if self.multi:
            state_action = torch.cat([state_action, id], dim=-1).float()

        if self.dist:
            mean_std = self.model(state_action)
            mean = mean_std[:, :self.output_dim]
            std = mean_std[:, self.output_dim:]
            std = torch.clamp(std, min=1e-6)
            return torch.distributions.normal.Normal(mean, std)
        else:
            pred = self.model(state_action)
            return pred
        

    def update(self, state, action, state_delta, id=None):
        self.train()
        
        if self.multi:
            id = to_tensor(id)
        state, action, state_delta = to_tensor(state, action, state_delta)
        
        if self.dist:
            dist = self(state, action, id)
            pred_state_delta = dist.rsample()
            losses = self.loss_fn(pred_state_delta, state_delta)
            # losses = -dist.log_prob(state_delta)
            losses -= dist.entropy() * self.entropy_weight
        else:
            pred_state_delta = self(state, action, id)
            losses = self.loss_fn(pred_state_delta, state_delta)
        loss = losses.mean()
        # sin = pred_state_delta[:, 2] + state[:, 0]
        # cos = pred_state_delta[:, 3] + state[:, 1]
        # loss += (self.loss_fn(sin**2 + cos**2, torch.ones_like(sin))).mean() * 0.05

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return dcn(losses)

    def set_scalers(self, states, actions, states_delta):
        states, actions, states_delta = dcn(states, actions, states_delta)
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
            
            states_actions = np.append(states, actions, axis=-1)
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
    def __init__(self, input_dim, output_dim, seed=1, hidden_dim=512, lr=7e-4, dropout=0.5, entropy_weight=0.02, dist=True, scale=True, multi=False, ensemble=0):
        assert ensemble > 0
        self.models = [DynamicsNetwork(input_dim, output_dim, hidden_dim=hidden_dim, lr=lr, dropout=dropout, entropy_weight=entropy_weight, dist=dist, multi=multi)
                                for _ in range(ensemble)]
        for model in self.models:
            model.to(device)
        self.seed = seed
        self.mse_loss = nn.MSELoss(reduction='none')
        self.neighbors = []
        self.state = None
        self.scale = scale
        self.multi = multi
        self.ensemble = ensemble

    def mpc_action(self, state, prev_goal, goal, action_range, swarm=False, n_steps=10, n_samples=1000,
                   swarm_weight=0.0, perp_weight=0.0, heading_weight=0.0, forward_weight=0.0, dist_weight=0.0, norm_weight=0.0,
                   which=-1):
        all_actions = np.random.uniform(*action_range, size=(n_steps, n_samples, action_range.shape[-1]))
        state, prev_goal, goal, all_actions = to_tensor(state, prev_goal, goal, all_actions)
        self.state = state      # for multi-robot
        states = torch.tile(state, (n_samples, 1))
        all_losses = torch.empty(n_steps, n_samples)
        
        for i in range(n_steps):
            actions = all_actions[i]
            with torch.no_grad():
                states[:, -1] %= 2 * torch.pi
                if self.multi:
                    if which == 0:
                        ids = torch.stack((torch.ones(len(states)), torch.zeros(len(states))), dim=1)
                    elif which == 2:
                        ids = torch.stack((torch.zeros(len(states)), torch.ones(len(states))), dim=1)
                    else:
                        raise ValueError
                    states = to_tensor(self.get_prediction(states, actions, ids, sample=False), requires_grad=False)
                else:
                    print("SINGLE")
                    states = to_tensor(self.get_prediction(states, actions, sample=False), requires_grad=False)
   
            dist_loss, heading_loss, perp_loss, forward_loss, norm_loss, swarm_loss, norm_const = self.compute_losses(states, prev_goal, goal, actions=actions, swarm=swarm)

            # normalize appropriate losses and compute total loss
            all_losses[i] = norm_const * (perp_weight * perp_loss + heading_weight * heading_loss \
                                + swarm_weight * swarm_loss + norm_weight * norm_loss) \
                                + dist_weight * dist_loss + forward_weight * forward_loss
        
        # find index of best trajectory and return corresponding first action
        best_idx = all_losses.sum(dim=0).argmin()
        return all_actions[0, best_idx]
    
    def compute_losses(self, states, prev_goal, goal, actions=None, current=False, signed=False, swarm=False):
        states, prev_goal, goal = to_tensor(states, prev_goal, goal)
        n_samples = len(states)
        goals = torch.tile(goal, (n_samples, 1))
        vec_to_goal = (goal - prev_goal)[:2]
        vecs_to_goal = torch.tile(vec_to_goal, (n_samples, 1))
        optimal_dot = vec_to_goal / vec_to_goal.norm()
        perp_denom = vec_to_goal.norm()
        x1, y1, _ = prev_goal
        x2, y2, _ = goal

        if current:
            states = states[None, :]
            goals = goal[None, :]
            vecs_to_goal = vec_to_goal[None, :]

        # heading computations
        x0, y0, current_angle = states.T
        # vecs_to_goal = (goals - states)[:, :2]
        # vecs_to_goal = torch.tile(vec_to_goal, (len(states), 1))
        # target_angle1 = torch.atan2(vecs_to_goal[:, 1], vecs_to_goal[:, 0]) + torch.pi
        # target_angle2 = torch.atan2(-vecs_to_goal[:, 1], -vecs_to_goal[:, 0]) + torch.pi
        target_angle = torch.atan2(vecs_to_goal[:, 1], vecs_to_goal[:, 0]) + torch.pi

        angle_diff_side = (target_angle - current_angle) % (2 * torch.pi)
        angle_diff_dir = torch.stack((angle_diff_side, 2 * torch.pi - angle_diff_side)).min(axis=0) 

        left = (angle_diff_side > torch.pi)
        forward = (angle_diff_dir < torch.pi / 2)

        heading_loss = angle_diff_dir

        if signed:
            heading_loss[~forward] = (heading_loss[~forward] + torch.pi) % (2 * torch.pi)
            heading_loss *= left * forward

        #     angle_diff_opposite = angle_diff_dir + torch.pi, angle_diff_dir - torch.pi
        #     # heading_idx = torch.stack((angle_diff_dir, 

        # if signed:
        #     signed_angle_diff = target_angle - current_angle
        #     # signed_angle_diff1 = target_angle1 - current_angle
        #     # signed_angle_diff2 = target_angle2 - current_angle
        # angle_diff1 = (target_angle1 - current_angle) % (2 * torch.pi)
        # angle_diff2 = (target_angle2 - current_angle) % (2 * torch.pi)
        # angle_diff1 = torch.stack((angle_diff1, 2 * torch.pi - angle_diff1)).min(dim=0)[0]
        # angle_diff2 = torch.stack((angle_diff2, 2 * torch.pi - angle_diff2)).min(dim=0)[0]
        # angle_diffs_stacked = torch.stack((angle_diff1, angle_diff2))
        
        # compute losses
        dist_loss = torch.norm((goals - states)[:, :2], dim=-1).squeeze()
        if signed:
            dist_loss *= forward

        # heading_idx = angle_diffs_stacked.argmin(dim=0)[0].squeeze()
        # if signed:
        #     heading_loss = torch.stack((signed_angle_diff1, signed_angle_diff2)).T[heading_idx].squeeze()
        # else:
        #     heading_loss = angle_diffs_stacked.min(dim=0)[0].squeeze()

        perp_loss = (((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / perp_denom).squeeze()
        if not signed:
            perp_loss = torch.abs(perp_loss)

        if current:
            if torch.any(torch.isnan(perp_loss)):
                import pdb;pdb.set_trace()

            return dist_loss, heading_loss, perp_loss

        forward_loss = torch.abs(optimal_dot @ vecs_to_goal.T).squeeze()
        norm_loss = -actions[:, :-1].norm(dim=-1).squeeze()
        swarm_loss = self.swarm_loss(states, goals).squeeze() if swarm else 0.0
        norm_const = dist_loss.mean()
        # norm_const = dist_loss.mean() / vecs_to_goal.mean(dim=0).norm()
        # norm_const = 1

        return dist_loss, heading_loss, perp_loss, forward_loss, norm_loss, swarm_loss, norm_const
    
    def get_prediction(self, states_xy, actions, ids=None, scale=False, sample=False, delta=False, use_ensemble=False, model_no=0):
        """
        Accepts state [x, y, theta]
        Returns next_state [x, y, theta]
        """
        states_xy, actions = to_tensor(states_xy, actions)
        thetas = states_xy[:, -1]
        sc = torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=1)

        if use_ensemble:
            states_delta_sum = np.zeros((len(states_xy), 4))
            for model in self.models:
                model.eval()
                cur_sc, cur_actions = sc, actions
                if self.scale and scale:
                    cur_sc, cur_actions = model.get_scaled(cur_sc, cur_actions)

                if self.multi:
                    ids = to_device(to_tensor(ids))
                cur_sc, cur_actions = to_tensor(cur_sc, cur_actions)
                cur_sc, cur_actions = to_device(cur_sc, cur_actions)

                with torch.no_grad():
                    model_output = model(cur_sc, cur_actions, ids)

                if model.dist:
                    if sample:
                        states_delta = model_output.rsample()
                    else:
                        states_delta = model_output.loc
                else:
                    states_delta = model_output

                states_delta = dcn(states_delta)
                if self.scale and scale:
                    states_delta_sum += model.output_scaler.inverse_transform(states_delta)
                else:
                    states_delta_sum += states_delta
            states_delta = states_delta_sum / len(self.models)
        else:
            model = self.models[model_no]
            if self.scale and scale:
                sc, actions = model.get_scaled(sc, actions)

            if self.multi:
                ids = to_device(to_tensor(ids))
            sc, actions = to_tensor(sc, actions)
            sc, actions = to_device(sc, actions)

            with torch.no_grad():
                model_output = model(sc, actions, ids)

            if model.dist:
                if sample:
                    states_delta = model_output.rsample()
                else:
                    states_delta = model_output.loc
            else:
                states_delta = model_output

            states_delta = dcn(states_delta)
            if self.scale and scale:
                states_delta = model.output_scaler.inverse_transform(states_delta)

        if delta:
            return states_delta

        states_sc = torch.cat([states_xy[:, :-1], sc], dim=-1)
        next_states_sc = states_sc + states_delta
        # next_states_sc[:, 2:] = torch.clamp(next_states_sc[:, 2:], min=-0.9999999, max=0.9999999)
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

        if self.multi:
            n_robots = states.shape[1]
            ids = torch.eye(n_robots)
            all_ids = torch.tile(ids, (len(states), 1))

            states = states.reshape(-1, states.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
            next_states = next_states.reshape(-1, next_states.shape[-1])

        sc = torch.stack([torch.sin(states[:, -1]), torch.cos(states[:, -1])], dim=1)
        states_sc = torch.cat([states[:, :-1], sc], dim=-1)
        sc = torch.stack([torch.sin(next_states[:, -1]), torch.cos(next_states[:, -1])], dim=1)
        next_states_sc = torch.cat([next_states[:, :-1], sc], dim=-1)
        states_delta = next_states_sc - states_sc

        # n_train = 300 if self.multi else 160
        n_test = 50
        idx = np.arange(len(states))
        train_states, test_states, train_actions, test_actions, train_states_delta, test_states_delta, \
                train_idx, test_idx = train_test_split(states, actions, states_delta, idx, test_size=n_test, random_state=self.seed)
        
        test_states, test_actions, test_states_delta = to_device(*to_tensor(test_states, test_actions, test_states_delta))

        for k, model in enumerate(self.models):
            train_states = states[train_idx]
            train_actions = actions[train_idx]
            train_states_delta = states_delta[train_idx]

            if self.ensemble > 1:
                rand_idx = torch.randint(0, len(train_states), (len(train_states),))
                train_states = train_states[rand_idx]
                train_actions = train_actions[rand_idx]
                train_states_delta = train_states_delta[rand_idx]

# new
# no scale 500
# ERROR MEAN: [0.0255191  0.02171914 0.22240782 0.22136173]
# ERROR MEAN: [0.01725932 0.01438991 0.21782211 0.212989  ]
# ERROR MEAN: [0.01827347 0.01415389 0.23176826 0.20695211]

# scale 500
# ERROR MEAN: [0.02181106 0.02391889 0.22154254 0.30314121]
# ERROR MEAN: [0.02159274 0.02388438 0.22843561 0.30461138]
# ERROR MEAN: [0.02172161 0.02395391 0.23818665 0.31035521]

# no scale 1000
# ERROR MEAN: [0.02447312 0.01933563 0.19355452 0.19447084]
# ERROR MEAN: [0.01673436 0.01550124 0.2096357  0.20053395]


# old
# no scale 500
# ERROR MEAN: [0.02368923 0.02050197 0.3393739  0.26279337]
# ERROR MEAN: [0.01386134 0.01548586 0.32144944 0.27274371]
# ERROR MEAN: [0.01654358 0.01508239 0.36146308 0.26831685]

# no scale 100
# ERROR MEAN: [0.02046056 0.03318009 0.33179109 0.28025044]
# ERROR MEAN: [0.0137791  0.02152438 0.33826193 0.28235782]
# ERROR MEAN: [0.01556799 0.01955299 0.34199481 0.28132791]

# scale 500
# ERROR MEAN: [0.02043447 0.02752197 0.27281989 0.30779337]
# ERROR MEAN: [0.02044041 0.02753705 0.2701623  0.31270161]
# ERROR MEAN: [0.02060534 0.02781321 0.29706241 0.31846031]

            if self.multi:
                train_ids, test_ids = all_ids[train_idx], all_ids[test_idx]
            else:
                train_ids, test_ids = None, None

            train_states = torch.stack([torch.sin(train_states[:, -1]), torch.cos(train_states[:, -1])], dim=1)

            if set_scalers:
                model.set_scalers(train_states, train_actions, train_states_delta)

            if self.scale:
                train_states, train_actions = model.get_scaled(train_states, train_actions)
                train_states_delta = model.get_scaled(train_states_delta)

            train_states, train_actions, train_states_delta = to_tensor(train_states, train_actions, train_states_delta)

            training_losses = []
            test_losses = []
            n_batches = np.ceil(len(train_states) / batch_size).astype("int")
            idx = np.arange(len(train_states))

            model.eval()
            with torch.no_grad():
                pred_states_delta = to_device(to_tensor(self.get_prediction(test_states, test_actions, test_ids, sample=False, delta=True, scale=True, model_no=k)))
            test_loss = self.mse_loss(pred_states_delta, test_states_delta)
            test_loss_mean = dcn(test_loss.mean())
            test_losses.append(test_loss_mean)
            tqdm.write(f"Pre-Train: mean test loss: {test_loss_mean}")
            model.train()

            for i in tqdm(range(-1, epochs), desc="Epoch", position=0, leave=False):
                np.random.shuffle(idx)
                train_states, train_actions, train_states_delta = train_states[idx], train_actions[idx], train_states_delta[idx]
                if self.multi:
                    train_ids = train_ids[idx]

                for j in tqdm(range(n_batches), desc="Batch", position=1, leave=False):
                    start, end = j * batch_size, (j + 1) * batch_size
                    batch_states = torch.autograd.Variable(train_states[start:end])
                    batch_actions = torch.autograd.Variable(train_actions[start:end])
                    batch_states_delta = torch.autograd.Variable(train_states_delta[start:end])
                    if self.multi:
                        batch_ids = to_device(torch.autograd.Variable(train_ids[start:end]))
                    else:
                        batch_ids = None
                    batch_states, batch_actions, batch_states_delta = to_device(batch_states, batch_actions, batch_states_delta)
                    
                    training_loss = model.update(batch_states, batch_actions, batch_states_delta, batch_ids)
                    if type(training_loss) != float:
                        while len(training_loss.shape) > 1:
                            training_loss = training_loss.mean(axis=-1)
            
                training_loss_mean = training_loss.mean()
                training_losses.append(training_loss_mean)
                model.eval()
                with torch.no_grad():
                    pred_states_delta = to_device(to_tensor(self.get_prediction(test_states, test_actions, test_ids, sample=False, delta=True, model_no=k)))
                test_loss = self.mse_loss(pred_states_delta, test_states_delta)
                test_loss_mean = dcn(test_loss.mean())
                test_losses.append(test_loss_mean)
                tqdm.write(f"{i+1}: mean training loss: {training_loss_mean} | mean test loss: {test_loss_mean}")
                model.train()
            
            model.eval()
        return training_losses, test_losses, test_idx


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
    parser.add_argument('-multi', action='store_true',
                        help='flag to use data from multiple robots')
    parser.add_argument('-ensemble', type=int, default=1,
                        help='how many networks to use for an ensemble')
    parser.add_argument('-naive', action='store_true')
    parser.add_argument('-robot', type=int)

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
        data = np.load("../../sim/data/real_data.npz")
        # data = np.load("../../sim/data/real_data_single2.npz")
        if args.multi:
            agent_path = '../../agents/real_multi.pkl'
        else:
            if args.naive:
                # agent_path = f'../../agents/real_single100_retrain100.pkl'
                agent_path = '../../agents/real_multi_naive.pkl'
            else:
                agent_path = f'../../agents/real_single{args.robot}.pkl'
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

    if states.shape[1] == 2:
        assert np.all(states[:, 1, -1] == 2) and np.all(actions[:, 1, -1] == 2) and np.all(next_states[:, 1, -1] == 2)
        assert np.all(states[:, 0, -1] == 0) and np.all(actions[:, 0, -1] == 0) and np.all(next_states[:, 0, -1] == 0)
    states = states[:, :, :-1]
    actions = actions[:, :, :-1]
    next_states = next_states[:, :, :-1]

    if not args.multi:
        if len(states.shape) == len(states.squeeze().shape):
            states = states.reshape(-1, states.shape[-1])
            actions = actions.reshape(-1, actions.shape[-1])
            next_states = next_states.reshape(-1, next_states.shape[-1])

            if not args.naive:
                if args.robot == 0:
                    robot = 0
                elif args.robot == 2:
                    robot = 1
                else:
                    raise ValueError
                states = states[robot::2]
                actions = actions[robot::2]
                next_states = next_states[robot::2]
        else:
            states = states.squeeze()
            actions = actions.squeeze()
            next_states = next_states.squeeze()

            # states = states[:-2]
            # actions_shift_0 = actions[:-2]
            # actions_shift_1 = actions[1:-1]
            # actions_shift_2 = actions[2:]
            # actions = np.concatenate([actions_shift_0, actions_shift_1, actions_shift_2], axis=-1)
            # next_states = next_states[2:]

            # states = states[:-1]
            # actions_shift_0 = actions[:-1]
            # actions_shift_1 = actions[1:]
            # actions = np.concatenate([actions_shift_0, actions_shift_1], axis=-1)
            # next_states = next_states[1:]

    if args.retrain:
        online_data = np.load("../../sim/data/real_data_online400.npz")
    
        online_states = online_data['states']
        online_actions = online_data['actions']
        online_next_states = online_data['next_states']

        n_repeat = int(len(states) / len(online_states))
        n_repeat = 1 if n_repeat == 0 else n_repeat
        online_states = np.tile(online_states, (n_repeat, 1))
        online_actions = np.tile(online_actions, (n_repeat, 1))
        online_next_states = np.tile(online_next_states, (n_repeat, 1))

        # states = np.append(states, online_states, axis=0)
        # actions = np.append(actions, online_actions, axis=0)
        # next_states = np.append(next_states, online_next_states, axis=0)

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

        states_theta = states[plotstart:plotend, -1]
        states_theta += pi
        states_sin = np.sin(states_theta)
        states_cos = np.cos(states_theta)

        next_states_theta = next_states[plotstart:plotend, -1]
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
        input_dim = actions.shape[-1] + 2
        output_dim = 4
        if args.multi:
            input_dim += 2
        
        agent = MPCAgent(input_dim, output_dim, seed=args.seed, dist=args.dist,
                         scale=args.scale, multi=args.multi, hidden_dim=args.hidden_dim,
                         lr=args.learning_rate, dropout=args.dropout, entropy_weight=args.entropy,
                         ensemble=args.ensemble)

        # batch_sizes = [args.batch_size, args.batch_size * 10, args.batch_size * 100, args.batch_size * 1000]
        batch_sizes = [args.batch_size]
        for batch_size in batch_sizes:
            training_losses, test_losses, test_idx = agent.train(states, actions, next_states, set_scalers=True,
                                                       epochs=args.epochs, batch_size=batch_size)

            # training_losses = np.array(training_losses).squeeze()
            # test_losses = np.array(test_losses).squeeze()
            # print("\nMIN TEST LOSS EPOCH:", test_losses.argmin())
            # print("MIN TEST LOSS:", test_losses.min())
            # plt.plot(np.arange(len(training_losses)), training_losses, label="Training Loss")
            # plt.plot(np.arange(-1, len(test_losses)-1), test_losses, label="Test Loss")
            # plt.yscale('log')
            # plt.xlabel('Epoch')
            # plt.ylabel('Loss')
            # plt.title('Dynamics Model Loss')
            # plt.legend()
            # plt.grid()
            # plt.show()
        
        if args.save:
            agent_path = args.save_agent_path if args.save_agent_path else agent_path
            print(f"\nSAVING MPC AGENT: {agent_path}\n")
            with open(agent_path, "wb") as f:
                pkl.dump(agent, f)
    else:
        agent_path = args.load_agent_path if args.load_agent_path else agent_path
        with open(agent_path, "rb") as f:
            agent = pkl.load(f)

        # agent.model.eval()
        # diffs = []
        # pred_next_states = agent.get_prediction(test_states, test_actions)
        
        # error = abs(pred_next_states - test_states_delta)
        # print("\nERROR MEAN:", error.mean(axis=0))
        # print("ERROR STD:", error.std(axis=0))
        # print("ERROR MAX:", error.max(axis=0))
        # print("ERROR MIN:", error.min(axis=0))

        # diffs = abs(test_states - test_states_delta)
        # print("\nACTUAL MEAN:", diffs.mean(axis=0))
        # print("ACTUAL STD:", diffs.std(axis=0))
        # set_trace()
        
        if args.retrain:
            training_losses, test_losses, test_idx = agent.train(states, actions, next_states,
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

    for model in agent.models:
        model.eval()

    if args.multi:
        n_robots = states.shape[1]
        ids = torch.eye(n_robots)
        all_ids = torch.tile(ids, (len(states), 1))

        states = states.reshape(-1, states.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        next_states = next_states.reshape(-1, next_states.shape[-1])

    states, actions, next_states = to_tensor(states, actions, next_states)
    sc = torch.stack([torch.sin(states[:, -1]), torch.cos(states[:, -1])], dim=1)
    states_sc = torch.cat([states[:, :-1], sc], dim=-1)
    sc = torch.stack([torch.sin(next_states[:, -1]), torch.cos(next_states[:, -1])], dim=1)
    next_states_sc = torch.cat([next_states[:, :-1], sc], dim=-1)
    states_delta = next_states_sc - states_sc

    test_states, test_actions = states[test_idx], actions[test_idx]
    if args.multi:
        test_ids = all_ids[test_idx]

    test_states_delta = dcn(states_delta[test_idx])
    if args.multi:
        pred_states_delta = agent.get_prediction(test_states, test_actions, ids=test_ids, sample=False, scale=args.scale, delta=True, use_ensemble=True)
    else:
        pred_states_delta = agent.get_prediction(test_states, test_actions, sample=False, scale=args.scale, delta=True, use_ensemble=True)
    
    error = abs(pred_states_delta - test_states_delta)
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
