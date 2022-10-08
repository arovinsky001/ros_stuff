import argparse
import pickle as pkl
from pdb import set_trace

import numpy as np
from matplotlib import pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import data_utils as dtu

device = torch.device("cpu")


class DynamicsNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, dtu=None, hidden_dim=256, hidden_depth=2, lr=1e-3, dropout=0.5, std=0.02, dist=True, use_object=False, scale=True):
        super(DynamicsNetwork, self).__init__()
        assert hidden_depth > 1
        layers = []
        for i in range(hidden_depth - 1):
            layers += [nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)]
            layers += [nn.ReLU()]
            layers += [nn.BatchNorm1d(hidden_dim, momentum=0.1)]
            layers += [nn.Dropout(p=dropout)]
        layers += [nn.Linear(hidden_dim, output_dim)]
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
        if dtu is None:
            self.dtu = dtu.DataUtils(use_object=use_object)
        else:
            self.dtu = dtu

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.0)

    def forward(self, state, action, sample=False, return_dist=False):
        state, action = dtu.as_tensor(state, action)

        if len(state.shape) == 1:
            state = state[None, :]
        if len(action.shape) == 1:
            action = action[None, :]

        input_state = self.dtu.state_to_model_input(state)
        state_action = torch.cat([input_state, action], dim=1).float()
        if self.scale:
            try:
                state_action = self.standardize_input(state_action)
            except:
                import pdb;pdb.set_trace()

        if self.dist:
            mean = self.net(state_action)
            std = torch.ones_like(mean) * 0.1
            dist = torch.distributions.normal.Normal(mean, std)
            if return_dist:
                return dist
            state_delta_model = dist.rsample() if sample else mean
        else:
            state_delta_model = self.net(state_action)

        if self.scale:
            state_delta_model = self.unstandardize_output(state_delta_model)

        return state_delta_model

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


class MPCAgent:
    def __init__(self, seed=1, hidden_dim=512, hidden_depth=2, lr=7e-4, dropout=0.5, std=0.02,
                 dist=True, scale=True, ensemble=0, use_object=False, use_velocity=False):
        assert ensemble > 0
        if use_object:
            if use_velocity:
                input_dim = 12
                output_dim = 14
            else:
                input_dim = 8
                output_dim = 8
        else:
            if use_velocity:
                input_dim = 7
                output_dim = 7
            else:
                input_dim = 4
                output_dim = 4

        self.dtu = dtu.DataUtils(use_object=use_object, use_velocity=use_velocity)
        self.models = [DynamicsNetwork(input_dim, output_dim, dtu=self.dtu, hidden_dim=hidden_dim, hidden_depth=hidden_depth, lr=lr, dropout=dropout, std=std, dist=dist, use_object=use_object, scale=scale)
                                for _ in range(ensemble)]
        for model in self.models:
            model.to(device)
        self.seed = seed
        self.scale = scale
        self.ensemble = ensemble
        self.use_object = use_object


    @property
    def model(self):
        return self.models[0]

    def mpc_action(self, state, prev_goal, goal, action_range, n_steps=10, n_samples=1000, perp_weight=0.0,
                   heading_weight=0.0, dist_weight=1.0, norm_weight=0.0, sep_weight=0.0, heading_diff_weight=0.0,
                   dist_bonus_weight=0.0, robot_goals=False, mpc_softmax=False, mpc_refine_iters=5, softmax_gamma=50000):
        original_state = state

        alpha = 0.9
        beta = 0.6
        n_best_losses = 30
        action_dim = action_range.shape[-1]
        action_trajectory_dim = n_steps * action_dim
        all_actions = np.random.uniform(*action_range, size=(n_samples, n_steps, action_dim))

        trajectory_mean = np.zeros(action_trajectory_dim)
        trajectory_std = np.zeros(action_trajectory_dim)
        all_losses = np.empty((n_samples, n_steps))

        for k in range(mpc_refine_iters):
            state = np.tile(original_state, (n_samples, 1))
            if k > 0:
                if mpc_softmax:
                    u = np.random.randn(n_samples, action_trajectory_dim).reshape(n_samples, n_steps, action_dim) * 0.2
                    n = u
                    for i in range(1, n_steps):
                        n[:, i] = beta * u[:, i] + (1 - beta) * n[:, i-1]
                    all_actions = (trajectory_mean + n.reshape(n_samples, action_trajectory_dim)).reshape(n_samples, n_steps, action_dim)
                else:
                    all_actions = np.random.normal(loc=trajectory_mean, scale=trajectory_std, size=(n_samples, action_trajectory_dim)).reshape(n_samples, n_steps, action_dim)
                all_actions = np.clip(all_actions, *action_range)

            for i in range(n_steps):
                action = all_actions[:, i]
                state = self.get_prediction(state, action, sample=False, use_ensemble=False)

                dist_loss, heading_loss, perp_loss, norm_loss, sep_loss, heading_diff_loss = \
                        self.compute_losses(state, prev_goal, goal, action=action, robot_goals=robot_goals)

                dist_bonus = -(dist_loss < 0.2).astype('float')
                dist_bonus2 = -(dist_loss < 0.1).astype('float')
                dist_bonus3 = -(dist_loss < 0.04).astype('float')
                # normalize appropriate losses and compute total loss
                all_losses[:, i] = (dist_weight + perp_weight * perp_loss + norm_weight * norm_loss + heading_weight * heading_loss
                                    + sep_weight * sep_loss + heading_diff_weight * heading_diff_loss) * dist_loss \
                                    + 0 * (dist_bonus + 10 * dist_bonus2 + 100 * dist_bonus3)

            # trajectory_losses = all_losses.sum(axis=1)
            # trajectory_losses = trajectory_losses / trajectory_losses.sum()

            trajectory_losses = all_losses.sum(axis=1)
            if mpc_softmax:
                # print("softmax")
                trajectory_losses -= trajectory_losses.min()
                trajectory_losses = trajectory_losses / trajectory_losses.sum()

            if mpc_refine_iters > 1:
                action_trajectories = all_actions.reshape(n_samples, action_trajectory_dim)

                if mpc_softmax:
                    # print("SOFTMAX")
                    weights = np.exp(softmax_gamma * -trajectory_losses)
                    weights_sum = weights.sum()
                    weighted_trajectories = (weights[:, None] * action_trajectories).sum(axis=0)
                    # weighted_std =
                    trajectory_mean = weighted_trajectories / weights.sum()
                else:
                    best_losses_idx = np.argsort(-trajectory_losses)[-n_best_losses:]
                    action_trajectories = all_actions.reshape(n_samples, action_trajectory_dim)
                    best_trajectories = action_trajectories[best_losses_idx]
                    best_trajectories_mean = best_trajectories.mean(axis=0)
                    best_trajectories_std = best_trajectories.std(axis=0)

                    trajectory_mean = alpha * best_trajectories_mean + (1 - alpha) * trajectory_mean
                    trajectory_std = alpha * best_trajectories_std + (1 - alpha) * trajectory_std

                trajectory_std = np.stack((trajectory_std, np.ones_like(trajectory_std) * 0.02)).max(axis=0)
            else:
                best_idx = trajectory_losses.argmin()
                return all_actions[best_idx, 0]

        # return trajectory_mean[:action_range.shape[-1]]
        return action_trajectories[trajectory_losses.argmin()][:action_range.shape[-1]]

    def compute_losses(self, state, prev_goal, goal, action=None, robot_goals=False, current=False, signed=False):
        vec_to_goal = (goal - prev_goal)[:2]
        if current:
            state = state[None, :]
            goals = goal[None, :]
            vecs_to_goal = vec_to_goal[None, :]
        else:
            n_samples = len(state)
            goals = np.tile(goal, (n_samples, 1))
            vecs_to_goal = np.tile(vec_to_goal, (n_samples, 1))

        if self.use_object:
            # separation loss
            robot_state, object_state = state[:, :3], state[:, 3:6]
            object_to_robot_xy = (robot_state - object_state)[:, :2]
            sep_loss = np.linalg.norm(object_to_robot_xy, axis=1)

            effective_state = robot_state if robot_goals else object_state
        else:
            effective_state = state[:, :3]     # ignore velocity
            sep_loss = 0.

        x0, y0, current_angle = effective_state.T

        # heading loss
        target_angle = np.arctan2(vecs_to_goal[:, 1], vecs_to_goal[:, 0])
        angle_diff_side = (target_angle - current_angle) % (2 * np.pi)
        angle_diff_dir = np.stack((angle_diff_side, 2 * np.pi - angle_diff_side)).min(axis=0)

        left = (angle_diff_side < np.pi) * 2 - 1.
        forward = (angle_diff_dir < np.pi / 2) * 2 - 1.

        heading_loss = angle_diff_dir
        heading_loss[forward == -1] = (heading_loss[forward == -1] + np.pi) % (2 * np.pi)
        heading_loss = np.stack((heading_loss, 2 * np.pi - heading_loss)).min(axis=0)

        if signed:
            heading_loss *= left * forward

        # dist loss
        dist_loss = np.linalg.norm((goals - effective_state)[:, :2], axis=1)

        if signed:
            vecs_to_goal_state = (goals - effective_state)[:, :2]
            target_angle_state = np.arctan2(vecs_to_goal_state[:, 1], vecs_to_goal_state[:, 0])

            angle_diff_side = (target_angle_state - current_angle) % (2 * np.pi)
            angle_diff_dir = np.stack((angle_diff_side, 2 * np.pi - angle_diff_side)).min(axis=0)
            forward = (angle_diff_dir < np.pi / 2) * 2 - 1.
            dist_loss *= forward

        # perp loss
        x1, y1, _ = prev_goal
        x2, y2, _ = goal
        perp_denom = np.linalg.norm(vec_to_goal)

        perp_loss = (((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / perp_denom)

        if not signed:
            perp_loss = np.abs(perp_loss)

        # return relevant losses for current state
        if current:
            return dist_loss.squeeze(), heading_loss.squeeze(), perp_loss.squeeze()

        # action norm loss
        norm_loss = -np.linalg.norm(action, axis=1).squeeze()

        # object vs. robot heading loss
        if self.use_object:
            robot_theta, object_theta = robot_state[:, -1], object_state[:, -1]
            heading_diff = (robot_theta + np.pi - object_theta) % (2 * np.pi)
            heading_diff_loss = np.stack((heading_diff, 2 * np.pi - heading_diff), axis=1).min(axis=1).squeeze()
            # print(f"Robot: {robot_theta}, Object: {object_theta}")
            # print("Object Robot Heading Diff: ", heading_diff_loss)
        else:
            heading_diff_loss = 0.

        return dist_loss, heading_loss, perp_loss, norm_loss, sep_loss, heading_diff_loss

    def get_prediction(self, state, action, sample=False, delta=False, use_ensemble=False, model_no=0):
        """
        Accepts state [x, y, theta] or [x, y, theta, x, y, theta]
        Returns next_state [x, y, theta] or [x, y, theta, x, y, theta]
        """
        if use_ensemble:
            state_delta_sum = torch.zeros((len(state), 8 if self.use_object else 4))
            next_state_sum = torch.zeros((len(state), 6 if self.use_object else 3))

            for model in self.models:
                model.eval()

                cur_state, cur_action = dtu.as_tensor(state, action)
                cur_state, cur_action = dtu.to_device(cur_state, cur_action)

                with torch.no_grad():
                    state_delta_model = model(state, action, sample=sample)
                    next_state_model = self.dtu.compute_next_state(state, state_delta_model)

                state_delta_sum += state_delta_model
                next_state_sum += next_state_model

            state_delta_model = state_delta_sum / len(self.models)
            next_state_model = next_state_sum / len(self.models)
        else:
            model = self.models[model_no]
            model.eval()

            with torch.no_grad():
                state_delta_model = model(state, action, sample=sample)
                next_state_model = self.dtu.compute_next_state(state, state_delta_model)

        if delta:
            return dtu.dcn(state_delta_model)

        next_state_model[:, 2] %= 2 * np.pi
        if self.use_object:
            next_state_model[:, 5] %= 2 * np.pi

        return dtu.dcn(next_state_model)

    def train(self, state, action, next_state, epochs=5, batch_size=256, set_scalers=False, use_all_data=False):
        state, action, next_state = dtu.as_tensor(state, action, next_state)

        n_test = int(len(state) * 0.1)
        all_idx = torch.arange(len(state))
        train_idx, test_idx = train_test_split(all_idx, test_size=n_test, random_state=self.seed)
        test_state, test_action, test_next_state = state[test_idx], action[test_idx], next_state[test_idx]
        test_state_delta = self.dtu.state_delta_xysc(test_state, test_next_state)

        if use_all_data:
            train_idx = all_idx

        for k, model in enumerate(self.models):
            train_state = state[train_idx]
            train_action = action[train_idx]
            train_next_state = next_state[train_idx]

            if set_scalers:
                model.set_scalers(train_state, train_action, train_next_state)

            train_losses, test_losses = [], []
            n_batches = int(np.ceil(len(train_state) / batch_size))

            with torch.no_grad():
                model.eval()
                pred_state_delta = model(test_state, test_action, sample=False)
                test_loss_mean = F.mse_loss(pred_state_delta, test_state_delta, reduction='mean')

            test_losses.append(dtu.dcn(test_loss_mean))
            tqdm.write(f"Pre-Train: mean test loss: {test_loss_mean}")

            print("\n\nTRAINING MODEL\n")
            for i in tqdm(range(-1, epochs), desc="Epoch", position=0, leave=False):
                random_idx = np.random.permutation(len(train_state))
                train_state, train_action, train_next_state = train_state[random_idx], train_action[random_idx], train_next_state[random_idx]

                for j in tqdm(range(n_batches), desc="Batch", position=1, leave=False):
                    start, end = j * batch_size, (j + 1) * batch_size
                    batch_state, batch_action, batch_next_state = train_state[start:end], train_action[start:end], train_next_state[start:end]
                    train_loss_mean = model.update(batch_state, batch_action, batch_next_state)

                with torch.no_grad():
                    model.eval()
                    pred_state_delta = model(test_state, test_action, sample=False)
                    test_loss_mean = F.mse_loss(pred_state_delta, test_state_delta, reduction='mean')

                train_losses.append(train_loss_mean)
                test_losses.append(dtu.dcn(test_loss_mean))
                tqdm.write(f"{i+1}: train loss: {train_loss_mean:.5f} | test loss: {test_loss_mean:.5f}")

        model.eval()
        return train_losses, test_losses, test_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train/load agent and do MPC.')
    parser.add_argument('-load_agent_path', type=str,
                        help='path/file to load old agent from')
    parser.add_argument('-save_agent_path', type=str,
                        help='path/file to save newly-trained agent to')
    parser.add_argument('-new_agent', '-n', action='store_true',
                        help='flag to train new agent')
    parser.add_argument('-hidden_dim', type=int, default=512,
                        help='dimension of hidden layers for dynamics network')
    parser.add_argument('-hidden_depth', type=int, default=2,
                        help='number of hidden layers for dynamics network')
    parser.add_argument('-epochs', type=int, default=10,
                        help='number of training epochs for new agent')
    parser.add_argument('-batch_size', type=int, default=128,
                        help='batch size for training new agent')
    parser.add_argument('-learning_rate', type=float, default=7e-4,
                        help='batch size for training new agent')
    parser.add_argument('-seed', type=int, default=1,
                        help='random seed for numpy and pytorch')
    parser.add_argument('-dist', action='store_true',
                        help='flag to have the model output a distribution')
    parser.add_argument('-dropout', type=float, default=0.5,
                        help='dropout probability')
    parser.add_argument('-scale', action='store_true',
                        help='flag to preprocess data with standard scaler')
    parser.add_argument('-save', action='store_true',
                        help='flag to save model after training')
    parser.add_argument('-retrain', action='store_true',
                        help='flag to load existing model and continue training')
    parser.add_argument('-std', type=float, default=0.02,
                        help='standard deviation for model distribution')
    parser.add_argument('-ensemble', type=int, default=1,
                        help='how many networks to use for an ensemble')
    parser.add_argument('-use_all_data', action='store_true')
    parser.add_argument('-use_object', action='store_true')

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.hidden_dim * args.hidden_depth >= 4000:
        if torch.backends.mps.is_available:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    if args.use_object:
        with open("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl", "rb") as f:
            buffer = pkl.load(f)

        states = buffer.states[:buffer.idx-1]
        action = buffer.action[:buffer.idx-1]
        next_states = buffer.states[1:buffer.idx]
    else:
        data = np.load("/Users/Obsidian/Desktop/eecs106b/projects/MPCDynamicsKamigami/sim/data/real_data.npz")
        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]

        states = states[:, 0, :-1]
        actions = actions[:, 0, :-1]
        next_states = next_states[:, 0, :-1]

    if args.retrain:
        online_data = np.load("../../sim/data/real_data_online400.npz")

        online_states = online_data['states']
        online_action = online_data['actions']
        online_next_states = online_data['next_states']

        n_repeat = int(len(states) / len(online_states))
        n_repeat = 1 if n_repeat == 0 else n_repeat
        online_states = np.tile(online_states, (n_repeat, 1))
        online_actions = np.tile(online_action, (n_repeat, 1))
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
        states_theta += np.pi
        states_sin = np.sin(states_theta)
        states_cos = np.cos(states_theta)

        next_states_theta = next_states[plotstart:plotend, -1]
        next_states_theta += np.pi
        next_states_sin = np.sin(next_states_theta)
        next_states_cos = np.cos(next_states_theta)

        plt.quiver(states_x[1:], states_y[1:], -states_cos[1:], -states_sin[1:], color="green")
        plt.quiver(next_states_x[:-1], next_states_y[:-1], -next_states_cos[:-1], -next_states_sin[:-1], color="purple")
        plt.plot(states_x[1:], states_y[1:], color="green", linewidth=1.0)
        plt.plot(next_states_x[:-1], next_states_y[:-1], color="purple", linewidth=1.0)
        for i, (x, y) in enumerate(zip(states_x, states_y)):
            if i == 0:
                continue
            plt.annotate(f"{i-1}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
            plt.annotate(str(actions_plot[i]), textcoords="offset points", xytext=(-10, -10), ha='center')

        for i, (x, y) in enumerate(zip(next_states_x, next_states_y)):
            if i == len(next_states_x) - 1:
                continue
            label = f"{i}"
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.show()

    if args.new_agent:
        agent = MPCAgent(seed=args.seed, dist=args.dist,
                         scale=args.scale, hidden_dim=args.hidden_dim,
                         hidden_depth=args.hidden_depth, lr=args.learning_rate,
                         dropout=args.dropout, std=args.std, ensemble=args.ensemble,
                         use_object=args.use_object)

        # batch_sizes = [args.batch_size, args.batch_size * 10, args.batch_size * 100, args.batch_size * 1000]
        batch_sizes = [args.batch_size]
        for batch_size in batch_sizes:
            training_losses, test_losses, test_idx = agent.train(
                        states, actions, next_states, set_scalers=True, epochs=args.epochs,
                        batch_size=batch_size, use_all_data=args.use_all_data
                        )

            if args.ensemble == 1:
                training_losses = np.array(training_losses).squeeze()
                test_losses = np.array(test_losses).squeeze()

                print("\nMIN TEST LOSS EPOCH:", test_losses.argmin())
                print("MIN TEST LOSS:", test_losses.min())

                fig, axes = plt.subplots(1, 2)
                axes[0].plot(np.arange(len(training_losses)), training_losses, label="Training Loss")
                axes[1].plot(np.arange(-1, len(test_losses)-1), test_losses, label="Test Loss")

                axes[0].set_yscale('log')
                axes[1].set_yscale('log')

                axes[0].set_title('Training Loss')
                axes[1].set_title('Test Loss')

                for ax in axes:
                    ax.grid()

                axes[0].set_ylabel('Loss')
                axes[1].set_xlabel('Epoch')
                fig.set_size_inches(15, 5)

                plt.show()

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

        # error = abs(pred_next_states - test_state_delta)
        # print("\nERROR MEAN:", error.mean(axis=0))
        # print("ERROR STD:", error.std(axis=0))
        # print("ERROR MAX:", error.max(axis=0))
        # print("ERROR MIN:", error.min(axis=0))

        # diffs = abs(test_states - test_state_delta)
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

    state_delta = agent.dtu.state_delta_xysc(states, next_states)

    test_state, test_action = states[test_idx], actions[test_idx]
    test_state_delta = dtu.dcn(state_delta[test_idx])

    pred_state_delta = dtu.dcn(model(test_state, test_action, sample=False))
    # pred_state_delta = agent.get_prediction(test_states, test_actions, sample=False, scale=args.scale, delta=True, use_ensemble=False)

    error = abs(pred_state_delta - test_state_delta)
    print("\nERROR MEAN:", error.mean(axis=0))
    print("ERROR STD:", error.std(axis=0))
    print("ERROR MAX:", error.max(axis=0))
    print("ERROR MIN:", error.min(axis=0))

    diffs = abs(test_state_delta)
    print("\nACTUAL MEAN:", diffs.mean(axis=0))
    print("ACTUAL STD:", diffs.std(axis=0))

    set_trace()

    for k in range(20):
        slist = []
        alist = []
        start, end = 10 * k, 10 * k + 10
        state = states[start]
        for i in range(start, end):
            action = action[i]
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
            plt.annotate(f"{i}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        for i, (x, y) in enumerate(zip(states[start:end, 0], states[start:end, 1])):
            label = f"{i}"
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.legend()
        plt.show()
        set_trace()
