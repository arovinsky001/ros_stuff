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
from tqdm import tqdm

import data_utils as dtu

pi = torch.pi
device = torch.device("cpu")


class DynamicsNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, hidden_depth=2, lr=1e-3, dropout=0.5, entropy_weight=0.02, dist=True):
        super(DynamicsNetwork, self).__init__()
        assert hidden_depth > 1
        layers = []
        for i in range(hidden_depth - 1):
            layers += [nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)]
            # layers += [nn.utils.spectral_norm(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.BatchNorm1d(hidden_dim, momentum=0.1)]
            layers += [nn.Dropout(p=dropout)]
        layers += [nn.Linear(hidden_dim, 2 * output_dim if dist else output_dim)]
        self.model = nn.Sequential(*layers)
        self.output_dim = output_dim
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, amsgrad=True, weight_decay=1e-4)
        self.loss_fn = nn.MSELoss(reduction='none')
        self.dist = dist
        self.entropy_weight = entropy_weight
        self.input_scaler = None
        self.output_scaler = None
        self.model.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
        #     nn.init.orthogonal_(m.weight.data)
        #     if hasattr(m.bias, 'data'):
        #         m.bias.data.fill_(0.0)
            torch.nn.init.xavier_uniform_(m.weight)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.01)

    def forward(self, state, action):
        state, action = dtu.as_tensor(state, action)

        if len(state.shape) == 1:
            state = state[None, :]
        if len(action.shape) == 1:
            action = action[None, :]

        state_action = torch.cat([state, action], dim=-1).float()

        if self.dist:
            mean, std = self.model(state_action).chunk(2, dim=-1)
            std = torch.clamp(std, min=1e-6)
            return torch.distributions.normal.Normal(mean, std)
        else:
            pred = self.model(state_action)
            return pred

    def update(self, state, action, state_delta):
        self.train()
        state, action, state_delta = dtu.as_tensor(state, action, state_delta)

        if self.dist:
            dist = self(state, action)
            pred_state_delta = dist.rsample()
            losses = self.loss_fn(pred_state_delta, state_delta)
            # losses = -dist.log_prob(state_delta)
            losses -= dist.entropy() * self.entropy_weight
        else:
            pred_state_delta = self(state, action)
            losses = self.loss_fn(pred_state_delta, state_delta)
        loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return dtu.dcn(losses)

    def set_scalers(self, states, actions, states_delta):
        states, actions, states_delta = dtu.dcn(states, actions, states_delta)
        state_action = np.concatenate([states, actions], axis=1)
        self.input_scaler = StandardScaler().fit(state_action)
        self.output_scaler = StandardScaler().fit(states_delta)

    def get_scaled_input(self, state, action):
        model_input = np.concatenate([state, action], axis=1)
        scaled_input = self.input_scaler.transform(model_input)
        scaled_state = scaled_input[:, :state.shape[-1]]
        scaled_action = scaled_input[:, state.shape[-1]:]
        return scaled_state, scaled_action

    def get_scaled_output(self, state_delta):
        scaled_state_delta = self.output_scaler.transform(state_delta)
        return scaled_state_delta


class MPCAgent:
    def __init__(self, input_dim, output_dim, seed=1, hidden_dim=512, hidden_depth=2, lr=7e-4, dropout=0.5, entropy_weight=0.02, dist=True, scale=True, ensemble=0):
        assert ensemble > 0
        self.models = [DynamicsNetwork(input_dim, output_dim, hidden_dim=hidden_dim, hidden_depth=hidden_depth, lr=lr, dropout=dropout, entropy_weight=entropy_weight, dist=dist)
                                for _ in range(ensemble)]
        for model in self.models:
            model.to(device)
        self.seed = seed
        self.mse_loss = nn.MSELoss(reduction='none')
        self.neighbors = []
        self.state = None
        self.scale = scale
        self.ensemble = ensemble

    def mpc_action(self, state, prev_goal, goal, action_range, n_steps=10, n_samples=1000, perp_weight=0.0,
                   heading_weight=0.0, dist_weight=0.0, norm_weight=0.0, robot_goals=False):
        all_actions = np.random.uniform(*action_range, size=(n_steps, n_samples, action_range.shape[-1]))
        state, prev_goal, goal, all_actions = dtu.as_tensor(state, prev_goal, goal, all_actions)
        self.state = state      # for multi-robot
        states = torch.tile(state, (n_samples, 1))
        all_losses = torch.empty(n_steps, n_samples)

        for i in range(n_steps):
            actions = all_actions[i]
            with torch.no_grad():
                states[:, 2] %= 2 * torch.pi
                states[:, 5] %= 2 * torch.pi
                states = dtu.as_tensor(self.get_prediction(states, actions, sample=False, scale=True, use_ensemble=False), requires_grad=False)

            dist_loss, heading_loss, perp_loss, norm_loss = self.compute_losses(states, prev_goal, goal,
                                                                                actions=actions, robot_goals=robot_goals)

            # normalize appropriate losses and compute total loss
            all_losses[i] = (dist_weight + perp_weight * perp_loss + heading_weight * heading_loss) * dist_loss

        # find index of best trajectory and return corresponding first action
        best_idx = all_losses.sum(dim=0).argmin()
        return all_actions[0, best_idx]

    def compute_losses(self, states, prev_goal, goal, actions=None, robot_goals=False, current=False, signed=False):
        states, prev_goal, goal = dtu.as_tensor(states, prev_goal, goal, requires_grad=False)
        vec_to_goal = (goal - prev_goal)[:2]

        if current:
            states = states[None, :]
            goals = goal[None, :]
            vecs_to_goal = vec_to_goal[None, :]
        else:
            n_samples = len(states)
            goals = torch.tile(goal, (n_samples, 1))
            vecs_to_goal = torch.tile(vec_to_goal, (n_samples, 1))

        states = states[:, :3] if robot_goals else states[:, 3:]

        perp_denom = vec_to_goal.norm()
        x1, y1, _ = prev_goal
        x2, y2, _ = goal

        # heading loss
        x0, y0, current_angle = states.T
        target_angle = torch.atan2(vecs_to_goal[:, 1], vecs_to_goal[:, 0]) + torch.pi

        angle_diff_side = (target_angle - current_angle) % (2 * torch.pi)
        angle_diff_dir = torch.stack((angle_diff_side, 2 * torch.pi - angle_diff_side)).min(axis=0)[0]

        left = (angle_diff_side < torch.pi) * 2 - 1.
        forward = (angle_diff_dir < torch.pi / 2) * 2 - 1.

        heading_loss = angle_diff_dir

        heading_loss[forward == -1] = (heading_loss[forward == -1] + torch.pi) % (2 * torch.pi)
        heading_loss, _ = torch.stack((heading_loss, 2 * torch.pi - heading_loss)).min(dim=0)
        if signed:
            heading_loss *= left * forward

        # dist loss
        dist_loss = torch.norm((goals - states)[:, :2], dim=-1)
        if signed:
            vecs_to_goal_states = (goals - states)[:, :2]
            target_angle = torch.atan2(vecs_to_goal_states[:, 1], vecs_to_goal_states[:, 0]) + torch.pi

            angle_diff_side = (target_angle - current_angle) % (2 * torch.pi)
            angle_diff_dir = torch.stack((angle_diff_side, 2 * torch.pi - angle_diff_side)).min(axis=0)[0]
            forward = (angle_diff_dir < torch.pi / 2) * 2 - 1.
            dist_loss *= forward

        # perp loss
        perp_loss = (((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / perp_denom)
        if not signed:
            perp_loss = torch.abs(perp_loss)

        if current:
            return dist_loss.squeeze(), heading_loss.squeeze(), perp_loss.squeeze()

        norm_loss = -actions[:, :-1].norm(dim=-1).squeeze()

        return dist_loss, heading_loss, perp_loss, norm_loss

    def get_prediction(self, states_original, actions, scale=True, sample=False, delta=False, use_ensemble=False, model_no=0):
        """
        Accepts state [x, y, theta]
        Returns next_state [x, y, theta]
        """
        states_converted = dtu.convert_state(states_original)     # convert state to robot sc, object-to-robot xysc
        # states_converted = dtu.convert_state(states_original)[:, :2]     # convert state to robot sc

        if use_ensemble:
            states_delta_sum = np.zeros((len(states_original), 8))
            for model in self.models:
                model.eval()
                cur_states, cur_actions = states_converted, actions
                if self.scale and scale:
                    cur_states, cur_actions = model.get_scaled_input(cur_states, cur_actions)

                cur_states, cur_actions = dtu.as_tensor(cur_states, cur_actions)
                cur_states, cur_actions = dtu.to_device(cur_states, cur_actions)

                with torch.no_grad():
                    model_output = model(cur_states, cur_actions)

                if model.dist:
                    if sample:
                        states_delta = model_output.rsample()
                    else:
                        states_delta = model_output.loc
                else:
                    states_delta = model_output

                states_delta = dtu.dcn(states_delta)
                if self.scale and scale:
                    states_delta_sum += model.output_scaler.inverse_transform(states_delta)
                else:
                    states_delta_sum += states_delta
            states_delta = states_delta_sum / len(self.models)
        else:
            model = self.models[model_no]
            model.eval()
            if self.scale and scale:
                states, actions = model.get_scaled_input(states_converted, actions)
            else:
                states = states_converted

            states, actions = dtu.as_tensor(states, actions)
            states, actions = dtu.to_device(states, actions)

            with torch.no_grad():
                model_output = model(states, actions)

            if model.dist:
                if sample:
                    states_delta = model_output.rsample()
                else:
                    states_delta = model_output.loc
            else:
                states_delta = model_output

            states_delta = dtu.dcn(states_delta)
            if self.scale and scale:
                states_delta = model.output_scaler.inverse_transform(states_delta)

        if delta:
            return states_delta

        robot_state, object_state = states_original[:, :3], states_original[:, 3:]
        robot_delta, object_delta = states_delta[:, :4], states_delta[:, 4:]

        next_robot_state = dtu.compute_next_state(robot_state, robot_delta)
        next_object_state = dtu.compute_next_state(object_state, object_delta)
        next_state = torch.cat((next_robot_state, next_object_state), dim=-1)

        # robot_state, robot_delta = states_original[:, :3], states_delta[:, :4]
        # next_state = dtu.compute_next_state(robot_state, robot_delta)

        next_state = dtu.dcn(next_state)
        return next_state

    def compute_next_state(self, state, state_delta):
        """
        Inputs:
            state = [x, y, theta]
            state_delta = [x, y, sin(theta), cos(theta)]
        Output:
            next_state = [x, y, theta]
        """
        xy, theta = state[:, :-1], state[:, -1]
        sc = torch.stack((torch.sin(theta), torch.cos(theta)), dim=1)
        xysc = torch.cat((xy, sc), dim=1)
        next_xysc = xysc + state_delta
        next_xy, next_sin, next_cos = next_xysc[:, :2], next_xysc[:, 2], next_xysc[:, 3]
        next_theta = torch.atan2(next_sin, next_cos)
        next_state = torch.cat((next_xy, next_theta[:, None]), dim=-1)
        return next_state

    def convert_state(self, states, xysc=False):
        states = dtu.as_tensor(states)
        robot_xy, object_xy = states[:, :2], states[:, 3:5]
        robot_theta, object_theta = states[:, 2], states[:, 5]

        robot_sc = torch.stack((torch.sin(robot_theta), torch.cos(robot_theta)), dim=1)
        object_sc = torch.stack((torch.sin(object_theta), torch.cos(object_theta)), dim=1)
        robot_xysc = torch.cat((robot_xy, robot_sc), dim=-1)
        object_xysc = torch.cat((object_xy, object_sc), dim=-1)

        obj_to_robot_xysc = robot_xysc - object_xysc
        full_states = torch.cat((robot_sc, obj_to_robot_xysc), dim=-1)
        if xysc:
            return full_states, robot_xysc, object_xysc
        return full_states

    def convert_state_delta(self, states, next_states):
        full_states, robot_xysc, object_xysc = dtu.convert_state(states, xysc=True)
        next_states = dtu.as_tensor(next_states)

        robot_next_theta, object_next_theta = next_states[:, 2], next_states[:, 5]
        robot_next_xy, object_next_xy = next_states[:, :2], next_states[:, 3:5]

        robot_next_sc = torch.stack([torch.sin(robot_next_theta), torch.cos(robot_next_theta)], dim=1)
        object_next_sc = torch.stack([torch.sin(object_next_theta), torch.cos(object_next_theta)], dim=1)
        robot_next_xysc = torch.cat([robot_next_xy, robot_next_sc], dim=-1)
        object_next_xysc = torch.cat([object_next_xy, object_next_sc], dim=-1)

        robot_states_delta = robot_next_xysc - robot_xysc
        object_states_delta = object_next_xysc - object_xysc
        states_delta = torch.cat((robot_states_delta, object_states_delta), dim=-1)

        return full_states, states_delta

    def train(self, states, actions, next_states, epochs=5, batch_size=256, set_scalers=False, use_all_data=False):
        states, actions, next_states = dtu.as_tensor(states, actions, next_states)

        n_test = 100
        idx = np.arange(len(states))
        train_states, test_states, train_actions, test_actions, train_next_states, test_next_states, \
                train_idx, test_idx = train_test_split(states, actions, next_states, idx, test_size=n_test, random_state=self.seed)

        test_states, test_actions, test_next_states = dtu.to_device(*dtu.as_tensor(test_states, test_actions, test_next_states))
        _, test_states_delta = dtu.convert_state_delta(test_states, test_next_states)

        for k, model in enumerate(self.models):
            if use_all_data:
                train_states = states
                train_actions = actions
                train_next_states = next_states
            else:
                train_states = states[train_idx]
                train_actions = actions[train_idx]
                train_next_states = next_states[train_idx]

            train_states, train_states_delta = dtu.convert_state_delta(train_states, train_next_states)

            if set_scalers:
                model.set_scalers(train_states, train_actions, train_states_delta)

            if self.scale:
                train_states, train_actions = model.get_scaled_input(train_states, train_actions)
                train_states_delta = model.get_scaled_output(train_states_delta)

            train_states, train_actions, train_states_delta = dtu.as_tensor(train_states, train_actions, train_states_delta)

            training_losses = []
            test_losses = []
            n_batches = np.ceil(len(train_states) / batch_size).astype("int")
            idx = np.arange(len(train_states))

            model.eval()
            with torch.no_grad():
                pred_states_delta = dtu.to_device(dtu.as_tensor(self.get_prediction(test_states, test_actions, sample=False, delta=True, scale=True, model_no=k)))
            test_loss = self.mse_loss(pred_states_delta, test_states_delta)
            test_loss_mean = dtu.dcn(test_loss.mean())
            test_losses.append(test_loss_mean)
            tqdm.write(f"Pre-Train: mean test loss: {test_loss_mean}")
            model.train()

            for i in tqdm(range(-1, epochs), desc="Epoch", position=0, leave=False):
                np.random.shuffle(idx)
                train_states, train_actions, train_states_delta = train_states[idx], train_actions[idx], train_states_delta[idx]

                for j in tqdm(range(n_batches), desc="Batch", position=1, leave=False):
                    start, end = j * batch_size, (j + 1) * batch_size
                    batch_states = torch.autograd.Variable(train_states[start:end])
                    batch_actions = torch.autograd.Variable(train_actions[start:end])
                    batch_states_delta = torch.autograd.Variable(train_states_delta[start:end])
                    batch_states, batch_actions, batch_states_delta = dtu.to_device(batch_states, batch_actions, batch_states_delta)

                    training_loss = model.update(batch_states, batch_actions, batch_states_delta)
                    if type(training_loss) != float:
                        while len(training_loss.shape) > 1:
                            training_loss = training_loss.mean(axis=-1)

                training_loss_mean = training_loss.mean()
                training_losses.append(training_loss_mean)
                model.eval()
                with torch.no_grad():
                    pred_states_delta = dtu.to_device(dtu.as_tensor(self.get_prediction(test_states, test_actions, sample=False, delta=True, scale=True, model_no=k)))
                test_loss = self.mse_loss(pred_states_delta, test_states_delta)
                test_loss_mean = dtu.dcn(test_loss.mean())
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
    parser.add_argument('-entropy', type=float, default=0.02,
                        help='weight for entropy term in training loss function')
    parser.add_argument('-ensemble', type=int, default=1,
                        help='how many networks to use for an ensemble')
    parser.add_argument('-use_all_data', action='store_true')

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

    with open("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl", "rb") as f:
        buffer = pkl.load(f)

    states = buffer.states[:buffer.idx-1]
    actions = buffer.actions[:buffer.idx-1]
    next_states = buffer.states[1:buffer.idx]

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
            plt.annotate(f"{i-1}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
            plt.annotate(str(actions_plot[i]), textcoords="offset points", xytext=(-10, -10), ha='center')

        for i, (x, y) in enumerate(zip(next_states_x, next_states_y)):
            if i == len(next_states_x) - 1:
                continue
            label = f"{i}"
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.show()

    if args.new_agent:
        input_dim = actions.shape[-1] + 6
        output_dim = 8

        agent = MPCAgent(input_dim, output_dim, seed=args.seed, dist=args.dist,
                         scale=args.scale, hidden_dim=args.hidden_dim,
                         hidden_depth=args.hidden_depth, lr=args.learning_rate,
                         dropout=args.dropout, entropy_weight=args.entropy, ensemble=args.ensemble)

        # batch_sizes = [args.batch_size, args.batch_size * 10, args.batch_size * 100, args.batch_size * 1000]
        batch_sizes = [args.batch_size]
        for batch_size in batch_sizes:
            training_losses, test_losses, test_idx = agent.train(states, actions, next_states, set_scalers=True,
                                                       epochs=args.epochs, batch_size=batch_size, use_all_data=args.use_all_data)

            if args.ensemble == 1:
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

    _, states_delta = agent.convert_state_delta(states, next_states)

    test_states, test_actions = states[test_idx], actions[test_idx]
    test_states_delta = dtu.dcn(states_delta[test_idx])
    pred_states_delta = agent.get_prediction(test_states, test_actions, sample=False, scale=args.scale, delta=True, use_ensemble=False)

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
            plt.annotate(f"{i}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        for i, (x, y) in enumerate(zip(states[start:end, 0], states[start:end, 1])):
            label = f"{i}"
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.legend()
        plt.show()
        set_trace()
