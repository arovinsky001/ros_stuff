import numpy as np


class MPCPolicy:
    def __init__(self, action_dim=2, action_range=None, simulate_fn=None, cost_fn=None):
        self.action_dim = action_dim
        self.simulate = simulate_fn
        self.compute_costs = cost_fn
        if action_range is None:
            self.action_range = np.array([[-1, -1], [1, 1]]) * 0.999
        else:
            self.action_range = action_range

    def compute_total_costs(self, cost_weights_dict, predicted_state_sequence, sampled_actions, prev_goal, goal, robot_goals):
        cost_dict = self.compute_costs(predicted_state_sequence, sampled_actions, prev_goal, goal, robot_goals=robot_goals)

        ensemble_costs = np.zeros(predicted_state_sequence.shape[:-1])
        for cost_type in cost_dict:
            if cost_type != "distance":
                ensemble_costs += cost_dict[cost_type] * cost_weights_dict[cost_type]
        ensemble_costs = (ensemble_costs + cost_weights_dict["distance"]) * cost_dict["distance"]

        # discount costs through time
        # discount = 0.9 ** np.arange(horizon)
        # ensemble_costs *= discount[None, None, :]

        # average over ensemble and horizon dimensions to get per-sample cost
        total_costs = ensemble_costs.mean(axis=(0, 2))
        total_costs -= total_costs.min()
        total_costs /= total_costs.max()

        return total_costs

    def get_action(self):
        return


class RandomShootingPolicy(MPCPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_action(self, initial_state, prev_goal, goal, cost_weights_dict, params):
        horizon = params["horizon"]
        n_samples = params["sample_trajectories"]
        robot_goals = params["robot_goals"]

        sampled_actions = np.random.uniform(*self.action_range, size=(n_samples, horizon, self.action_dim))
        predicted_state_sequence = self.simulate(initial_state, sampled_actions)
        total_costs = self.compute_total_costs(cost_weights_dict, predicted_state_sequence, sampled_actions, prev_goal, goal, robot_goals)

        return sampled_actions[total_costs.argmin(), 0]


class CEMPolicy(MPCPolicy):
    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    def get_action(self, initial_state, prev_goal, goal, cost_weights_dict, params):
        horizon = params["horizon"]
        n_samples = params["sample_trajectories"]
        refine_iters = params["refine_iters"]
        alpha = params["alpha"]
        n_best = params["n_best"]
        robot_goals = params["robot_goals"]
        action_trajectory_dim = self.action_dim * horizon

        trajectory_mean = np.zeros(action_trajectory_dim)
        trajectory_std = np.zeros(action_trajectory_dim)
        sampled_actions = np.random.uniform(*self.action_range, size=(n_samples, horizon, self.action_dim))

        for i in range(refine_iters):
            if i > 0:
                sampled_actions = np.random.normal(loc=trajectory_mean, scale=trajectory_std, size=(n_samples, action_trajectory_dim))
                sampled_actions = sampled_actions.reshape(n_samples, horizon, self.action_dim)

            predicted_state_sequence = self.simulate(initial_state, sampled_actions)
            total_costs = self.compute_total_costs(cost_weights_dict, predicted_state_sequence, sampled_actions, prev_goal, goal, robot_goals)

            action_trajectories = sampled_actions.reshape((n_samples, action_trajectory_dim))
            best_costs_idx = np.argsort(-total_costs)[-n_best:]
            best_trajectories = action_trajectories[best_costs_idx]
            best_trajectories_mean = best_trajectories.mean(axis=0)
            best_trajectories_std = best_trajectories.std(axis=0)

            trajectory_mean = alpha * best_trajectories_mean + (1 - alpha) * trajectory_mean
            trajectory_std = alpha * best_trajectories_std + (1 - alpha) * trajectory_std

            if trajectory_std.max() < 0.02:
                break

        return trajectory_mean[:self.action_dim]


class MPPIPolicy(MPCPolicy):
    def __init__(self, **kwargs):
        self.trajectory_mean = None
        return super().__init__(**kwargs)

    def get_action(self, initial_state, prev_goal, goal, cost_weights_dict, params):
        horizon = params["horizon"]
        n_samples = params["sample_trajectories"]
        beta = params["beta"]
        gamma = params["gamma"]
        noise_std = params["noise_std"]
        robot_goals = params["robot_goals"]

        if self.trajectory_mean is None:
            self.trajectory_mean = np.zeros((horizon, self.action_dim))

        just_executed_action = self.trajectory_mean[0].copy()
        self.trajectory_mean[:-1] = self.trajectory_mean[1:]

        sampled_actions = np.empty((n_samples, horizon, self.action_dim))
        noise = np.random.normal(loc=0, scale=noise_std, size=(n_samples, horizon, self.action_dim))

        for t in range(horizon):
            if t == 0:
                sampled_actions[:, t] = beta * (self.trajectory_mean[t] + noise[:, t]) \
                                            + (1 - beta) * just_executed_action
            else:
                sampled_actions[:, t] = beta * (self.trajectory_mean[t] + noise[:, t]) \
                                            + (1 - beta) * sampled_actions[:, t-1]

        sampled_actions = np.clip(sampled_actions, *self.action_range)
        predicted_state_sequence = self.simulate(initial_state, sampled_actions)
        total_costs = self.compute_total_costs(cost_weights_dict, predicted_state_sequence, sampled_actions, prev_goal, goal, robot_goals)

        action_trajectories = sampled_actions.reshape((n_samples, -1))

        weights = np.exp(gamma * -total_costs)
        weighted_trajectories = (weights[:, None] * action_trajectories).sum(axis=0)
        self.trajectory_mean = weighted_trajectories / weights.sum()

        return self.trajectory_mean[:self.action_dim]
