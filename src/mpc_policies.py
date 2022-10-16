import numpy as np


class MPCPolicy:
    def __init__(self, action_dim=2, action_range=None, simulate_fn=None, cost_fn=None, **kwargs):
        self.action_dim = action_dim
        self.simulate = simulate_fn
        self.compute_costs = cost_fn
        if action_range is None:
            self.action_range = np.array([[-1, -1], [1, 1]]) * 0.999
        else:
            self.action_range = action_range

    def get_action(self):
        return


class MPPIPolicy(MPCPolicy):
    def __init__(self, **kwargs):
        self.trajectory_mean = None
        return super().__init__(**kwargs)

    def get_action(self, state, prev_goal, goal, cost_weights_dict, params):
        horizon = params["horizon"]
        n_samples = params["sample_trajectories"]
        beta = params["beta"]
        gamma = params["gamma"]
        robot_goals = params["robot_goals"]

        if self.trajectory_mean is None:
            self.trajectory_mean = np.zeros((horizon, self.action_dim))

        just_executed_action = self.trajectory_mean[0].copy()
        self.trajectory_mean[:-1] = self.trajectory_mean[1:]

        sampled_actions = np.empty((n_samples, horizon, self.action_dim))
        noise = np.random.normal(loc=0, scale=1., size=(n_samples, horizon, self.action_dim))

        for t in range(horizon):
            if t == 0:
                sampled_actions[:, t] = beta * (self.trajectory_mean[t] + noise[:, t]) \
                                            + (1 - beta) * just_executed_action
            else:
                sampled_actions[:, t] = beta * (self.trajectory_mean[t] + noise[:, t]) \
                                            + (1 - beta) * sampled_actions[:, t-1]

        sampled_actions = np.clip(sampled_actions, *self.action_range)
        predicted_state_sequence = self.simulate(state, sampled_actions)
        cost_dict = self.compute_costs(predicted_state_sequence, sampled_actions, prev_goal, goal, robot_goals=robot_goals)

        ensemble_costs = np.zeros(predicted_state_sequence.shape[:-1])
        for cost_type in cost_dict:
            if cost_type != "distance":
                ensemble_costs += cost_dict[cost_type] * cost_weights_dict[cost_type]
        ensemble_costs = (ensemble_costs + 1) * cost_dict["distance"]

        # average over ensemble and horizon dimensions to get per-sample cost
        print("ENSEMBLE:", ensemble_costs.shape[0])
        total_costs = ensemble_costs.mean(axis=(0, 2))
        total_costs -= total_costs.min()
        total_costs /= total_costs.max()
        action_sequences = sampled_actions.reshape((n_samples, -1))

        weights = np.exp(gamma * -total_costs)
        weighted_trajectories = (weights[:, None] * action_sequences).sum(axis=0)
        self.trajectory_mean = weighted_trajectories / weights.sum()

        return self.trajectory_mean[:self.action_dim]
