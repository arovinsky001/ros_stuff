import argparse
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
from tqdm import trange, tqdm
from time import time

from generate_data import *

def to_tensor(*args):
    ret = []
    for arg in args:
        if type(arg) == np.ndarray:
            ret.append(tensor(arg.astype('float32'), requires_grad=True))
        else:
            ret.append(arg)
    return ret

class DynamicsNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, lr=7e-4, delta=False, dist=False):
        super(DynamicsNetwork, self).__init__()
        input_dim = state_dim + action_dim
        output_dim = state_dim * 2 if dist else state_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # nn.Dropout(p=0.007),
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.delta = delta
        self.dist = dist
        self.trained = False
        self.input_scaler = None
        self.output_scaler = None

    def forward(self, state, action):
        state, action = to_tensor(state, action)
        if len(state.shape) == 1:
            state = state[:, None]
        if len(action.shape) == 1:
            action = action[:, None]

        state_action = torch.cat([state, action], dim=-1).float()
        if self.dist:
            mean_std = self.model(state_action)
            mean = mean_std[:, :int(mean_std.shape[1]/2)]
            std = mean_std[:, int(mean_std.shape[1]/2):]
            std = torch.clamp(std, min=1e-6)
            return torch.distributions.normal.Normal(mean, std)
        else:
            pred = self.model(state_action)
            return pred
        

    def update(self, state, action, next_state, retain_graph=False):
        state, action, next_state = to_tensor(state, action, next_state)
        
        if self.dist:
            dist = self(state, action)
            prediction = dist.rsample()
            if self.delta:
                losses = self.loss_fn(state + prediction, next_state)
            else:
                losses = self.loss_fn(prediction, next_state)
        else:
            if self.delta:
                state_delta = self(state, action)
                losses = self.loss_fn(state + state_delta, next_state)
            else:
                pred_next_state = self(state, action)
                losses = self.loss_fn(pred_next_state, next_state)
        loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
        return losses.detach()
    
    def set_scalers(self, states, actions, next_states):
        with torch.no_grad():
            self.input_scaler = StandardScaler().fit(np.append(states, actions, axis=-1))
            self.output_scaler = StandardScaler().fit(next_states)
    
    def get_scaled(self, *args):
        if len(args) == 2:
            states, actions = args
            states_actions = np.append(states, actions, axis=-1)
            states_actions_scaled = self.input_scaler.transform(states_actions)
            states_scaled = states_actions_scaled[:, :states.shape[-1]]
            actions_scaled = states_actions_scaled[:, states.shape[-1]:]
            return states_scaled, actions_scaled
        else:
            next_states = args[0]
            next_states_scaled = self.output_scaler.transform(next_states)
            return next_states_scaled

class MPCAgent:
    def __init__(self, state_dim, action_dim, seed=1, delta=False, dist=False, hidden_dim=512, lr=7e-4):
        self.model = DynamicsNetwork(state_dim, action_dim, hidden_dim=hidden_dim, lr=lr, delta=delta, dist=dist)
        self.seed = seed
        self.action_dim = action_dim
        self.mse_loss = nn.MSELoss(reduction='none')
        self.neighbors = []
        self.state = None
        self.delta = delta
        self.dist = dist
        self.time = 0

    def mpc_action(self, state, goal, state_range, action_range, n_steps=10, n_samples=1000, swarm=False, swarm_weight=0.1):
        self.state = tensor(state)
        all_actions = np.random.uniform(low=action_range[0], high=action_range[1],
                                        size=(n_steps, n_samples, self.action_dim))
        states = np.tile(state, (n_samples, 1))
        goals = np.tile(goal, (n_samples, 1))
        states, all_actions, goals = to_tensor(states, all_actions, goals)
        all_losses = []

        for i in range(n_steps):
            actions = all_actions[i]
            with torch.no_grad():
                states = self.get_prediction(states, actions)
                if type(states) == np.ndarray:
                    states = tensor(states)
            states = np.clip(states, *state_range)
            if swarm:
                loss = self.swarm_loss(states, goals, swarm_weight)
            else:
                states[:, -1] %= 2 * np.pi
                loss = torch.norm(states[:, :-1] - goals[:, :-1], dim=-1)
                # loss = self.mse_loss(states[:, :-1], goals[:, :-1])
                # loss = loss.mean(axis=-1)
                # vec = goals - states
                # import pdb;pdb.set_trace()
                # vec /= 5
                # tdist = states[:, -1] - goals[:, -1]
                # tdist1 = tdist % (2 * np.pi)
                # tdist2 = 2 * np.pi - tdist % (2 * np.pi)
                # dists_theta = torch.stack((tdist2, tdist2)).min(axis=0)[0].reshape(-1, 1)
                # loss = torch.cat((loss, dists_theta * 0.01), dim=-1)
            # all_losses.append(loss.detach().numpy().mean(axis=-1))
            all_losses.append(loss.detach().numpy())
        
        # best_idx = np.array(all_losses).sum(axis=0).argmin()
        best_idx = loss.argmin()
        return all_actions[0, best_idx]
    
    def get_prediction(self, states, actions):
        states_scaled, actions_scaled = self.model.get_scaled(states, actions)
        model_output = self.model(states_scaled, actions_scaled)
        if self.dist:
            prediction_scaled = model_output.rsample()
            prediction = self.model.output_scaler.inverse_transform(prediction_scaled.detach())
        else:
            prediction = self.model.output_scaler.inverse_transform(model_output.detach())
        return states + prediction if self.delta else prediction

    def swarm_loss(self, states, goals, swarm_weight):
        neighbor_dists = []
        for neighbor in self.neighbors:
            neighbor_states = torch.tile(neighbor.state, (states.shape[0], 1))
            distance = self.mse_loss(states, neighbor_states)
            neighbor_dists.append(distance)
        goal_term = self.mse_loss(states, goals)
        neighbor_term = torch.stack(neighbor_dists).mean(dim=0) * goal_term.mean() / goals.mean()
        loss = goal_term + neighbor_term * swarm_weight
        return loss

    def train(self, states, actions, next_states, epochs=5, batch_size=256, correction=False, error_weight=4, n_tests=500):
        states, actions, next_states = to_tensor(states, actions, next_states)
        train_states, test_states, train_actions, test_actions, train_next_states, test_next_states \
            = train_test_split(states, actions, next_states, test_size=0.05, random_state=self.seed)

        training_losses = []
        test_losses = []
        test_idx = []
        idx = np.arange(len(train_states))

        n_batches = len(train_states) // batch_size + 1
        test_interval = epochs * n_batches // n_tests

        i = 0
        for _ in tqdm(range(epochs), desc="Epoch", position=0, leave=False):
            np.random.shuffle(idx)
            train_states, train_actions, train_next_states = train_states[idx], train_actions[idx], train_next_states[idx]
            
            for j in tqdm(range(n_batches), desc="Batch", position=1, leave=False):
                batch_states = torch.autograd.Variable(train_states[j*batch_size:(j+1)*batch_size])
                batch_actions = torch.autograd.Variable(train_actions[j*batch_size:(j+1)*batch_size])
                batch_next_states = torch.autograd.Variable(train_next_states[j*batch_size:(j+1)*batch_size])

                training_loss = self.model.update(batch_states, batch_actions, batch_next_states)
                if type(training_loss) != float:
                    while len(training_loss.shape) > 1:
                        training_loss = training_loss.sum(axis=-1)

                # if correction:
                #     training_loss = self.correct(states, actions, next_states, data_idx, training_loss,
                #                         batch_size=batch_size, error_weight=error_weight)

                training_loss_mean = training_loss.mean().detach()
                training_losses.append(training_loss_mean)
                
                if i % test_interval == 0:
                    with torch.no_grad():
                        pred_next_states = tensor(self.get_prediction(test_states, test_actions))
                    test_loss = self.mse_loss(pred_next_states, test_next_states)
                    test_loss_mean = test_loss.mean().detach()
                    test_losses.append(test_loss_mean)
                    test_idx.append(i)
                
                i += 1
            
            tqdm.write(f"mean training loss: {training_loss_mean} | mean test loss: {test_loss_mean}")

        self.model.trained = True
        return training_losses, test_losses, test_idx

    def correct(self, states, actions, next_states, data_idx, loss, batch_size=256, error_weight=4):
        worst_idx = torch.topk(loss.squeeze(), int(batch_size / 10))[1].detach().numpy()
        train_idx = np.append(data_idx, np.tile(data_idx[worst_idx], (1, error_weight)))
        train_states, train_actions, train_next_states = states[train_idx], actions[train_idx], next_states[train_idx]
        loss = self.model.update(train_states, train_actions, train_next_states).detach().numpy()
        
        if type(loss) != float:
            while len(loss.shape) > 1:
                loss = loss.sum(axis=-1)

        return loss

    def optimal_policy(self, state, goal, table, swarm=False, swarm_weight=0.3):
        if swarm:
            vec = goal - state
            states = tensor(state + table[:, 1, None])
            neighbor_dists = []
            for neighbor in self.neighbors:
                neighbor_states = torch.tile(neighbor.state, (states.shape[0], 1))
                distance = self.mse_loss(states, neighbor_states)
                neighbor_dists.append(distance.detach().numpy())
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
    parser.add_argument('-correction', action='store_true',
                        help='flag to retrain on mistakes during training')
    parser.add_argument('-correction_weight', type=int, default=4,
                        help='number of times to retrain on mistaken data')
    parser.add_argument('-stochastic', action='store_true',
                        help='flag to use stochastic transition data')
    parser.add_argument('-distribution', action='store_true',
                        help='flag to have the model return a distribution')
    parser.add_argument('-delta', action='store_true',
                        help='flag to output state delta')
    parser.add_argument('-real', action='store_true',
                        help='flag to use real data')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.real:
        agent_path = 'agents/real_BEST.pkl'
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

    print('\nDATA LOADED\n')

    if not args.real:
        agent_path += f"epochs{args.epochs}"
        agent_path += f"_dim{args.hidden_dim}"
        agent_path += f"_batch{args.batch_size}"
        agent_path += f"_lr{args.learning_rate}"
        if args.distribution:
            agent_path += "_distribution"
        if args.stochastic:
            agent_path += "_stochastic"
        if args.delta:
            agent_path += "_delta"
        if args.correction:
            agent_path += f"_correction{args.correction_weight}"
        agent_path += ".pkl"

    if args.new_agent:
        agent = MPCAgent(states.shape[-1], actions.shape[-1], seed=args.seed,
                         delta=args.delta, dist=args.distribution,
                         hidden_dim=args.hidden_dim, lr=args.learning_rate)

        agent.model.set_scalers(states, actions, next_states)
        states_scaled, actions_scaled = agent.model.get_scaled(states, actions)
        next_states_scaled = agent.model.get_scaled(next_states)

        training_losses, test_losses, test_idx = agent.train(
                        states_scaled, actions_scaled, next_states_scaled,
                        epochs=args.epochs, batch_size=args.batch_size,
                        correction=args.correction, error_weight=args.correction_weight,
                        n_tests=500)

        training_losses = np.array(training_losses).squeeze()
        test_losses = np.array(test_losses).squeeze()
        plt.plot(np.arange(len(training_losses)), training_losses, label="Training Loss")
        plt.plot(test_idx, test_losses, label="Test Loss")
        plt.yscale('log')
        plt.xlabel('Batch')
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

    states_min = np.ones(2) * MIN_STATE
    states_max = np.ones(2) * MAX_STATE
    state_range = np.block([[states_min], [states_max]])

    actions_min = np.ones(2) * MIN_ACTION
    actions_max = np.ones(2) * MAX_ACTION
    action_range = np.block([[actions_min], [actions_max]])

    potential_actions = np.linspace(MIN_ACTION, MAX_ACTION, 10000)
    potential_deltas = FUNCTION(potential_actions)
    TABLE = np.block([potential_actions.reshape(-1, 1), potential_deltas.reshape(-1, 1)])

    # MPC parameters
    n_steps = 1         # length per sample trajectory
    n_samples = 5000    # number of trajectories to sample

    # run trials testing the MPC policy against the optimal policy
    n_trials = 100
    max_steps = 200
    success_threshold = 1.0
    plot = False

    n_trials = 2
    max_steps = 120
    start = np.array([11.1, 18.7])
    goal = np.array([90.3, 71.5])
    optimal_losses = np.empty(max_steps)
    actual_losses = np.empty((n_trials, max_steps))
    noises = np.random.normal(loc=0.0, scale=NOISE_STD, size=(max_steps, 2))
    state = start.copy()

    i = 0
    while not np.linalg.norm(goal - state) < 0.2:
        optimal_losses[i] = np.linalg.norm(goal - state)
        noise = noises[i]
        state += FUNCTION(agent.optimal_policy(state, goal, TABLE)) + noise
        state = np.clip(state, MIN_STATE, MAX_STATE)
        i += 1
    optimal_losses[i:] = np.linalg.norm(goal - state)

    for k in trange(n_trials):
        state = start.copy()
        i = 0
        while not np.linalg.norm(goal - state) < 0.2:
            actual_losses[k, i] = np.linalg.norm(goal - state)
            noise = noises[i]
            action = agent.mpc_action(state, goal, state_range, action_range,
                                    n_steps=n_steps, n_samples=n_samples).detach().numpy()
            state += FUNCTION(action) + noise
            state = np.clip(state, MIN_STATE, MAX_STATE)
            i += 1
        actual_losses[k, i:] = np.linalg.norm(goal - state)
    
    plt.plot(np.arange(max_steps), optimal_losses, 'g-', label="Optimal Controller")
    plt.plot(np.arange(max_steps), actual_losses.mean(axis=0), 'b-', label="MPC Controller")
    plt.title("Optimal vs MPC Controller Performance")
    plt.legend()
    plt.xlabel("Step\n\nstart = [11.1, 18.7], goal = [90.3, 71.5]\nAveraged over 20 MPC runs")
    plt.ylabel("Distance to Goal")
    plt.grid()
    # plt.text(0.5, 0.01, "start = [11.1, 18.7], goal = [90.3, 71.5]", wrap=True, ha='center', fontsize=12)
    plt.show()
    set_trace()

    while True:
        optimal_lengths = []
        actual_lengths = []
        optimal = 0
        all_states = []
        all_actions = []
        all_goals = []
        for trial in trange(n_trials):
            init_state = np.random.rand(2) * (MAX_STATE - MIN_STATE) + MIN_STATE
            goal = np.random.rand(2) * (MAX_STATE - MIN_STATE) + MIN_STATE
            noises = np.random.normal(loc=0.0, scale=NOISE_STD, size=(max_steps, 2))

            state = init_state.copy()
            i = 0
            while not np.linalg.norm(goal - state) < success_threshold:
                noise = noises[i] if args.stochastic else 0.0
                state += FUNCTION(optimal_policy(state, goal, TABLE)) + noise
                if LIMIT:
                    state = np.clip(state, MIN_STATE, MAX_STATE)
                i += 1
            optimal_lengths.append(i)
            
            state = init_state.copy()
            j = 0
            states, actions = [], []
            while not np.linalg.norm(goal - state) < success_threshold:
                states.append(state)
                if j == max_steps:
                    break
                action = agent.mpc_action(state, goal, state_range, action_range,
                                    n_steps=n_steps, n_samples=n_samples).detach().numpy()
                noise = noises[j] if args.stochastic else 0.0
                state += FUNCTION(action) + noise
                if LIMIT:
                    state = np.clip(state, MIN_STATE, MAX_STATE)
                j += 1
                actions.append(action)

            all_states.append(states)
            all_actions.append(actions)
            all_goals.append(goal)
            actual_lengths.append(j)

            if j <= i:
                optimal += 1
        
        optimal_lengths, actual_lengths = np.array(optimal_lengths), np.array(actual_lengths)
        print("\noptimal mean:", optimal_lengths.mean())
        print("optimal std:", optimal_lengths.std(), "\n")
        print("actual mean:", actual_lengths.mean())
        print("actual std:", actual_lengths.std(), "\n")
        print("mean error:", np.abs(optimal_lengths.mean() - actual_lengths.mean()) / optimal_lengths.mean())
        print("optimality rate:", optimal / float(n_trials))
        print("timeout rate:", (actual_lengths == max_steps).sum() / float(n_trials), "\n")
        
        if plot:
            plt.hist(optimal_lengths)
            plt.plot(optimal_lengths, actual_lengths, 'bo')
            plt.xlabel("Optimal Steps to Reach Goal")
            plt.ylabel("Actual Steps to Reach Goal")
            plt.show()
        set_trace()
