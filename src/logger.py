#!/usr/bin/python

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import csv
from collections import OrderedDict
import wandb
from utils import signed_angle_difference, dimensions


class Logger:
    def __init__(self, params):
        self.params = params

        if exp_name is None:
            exp_name = f"{'robot' if self.robot_goals else 'object'}goals"
            if self.pretrain or self.load_agent:
                exp_name += f"_pretrain{self.pretrain_samples}"
            if self.online:
                exp_name += "_online"

        exp_name = f"{'object' if self.use_object else 'robot'}_{exp_name}"

        wandb.init(project=exp_name, entity="kamigami")

        self.exp_path = f"/home/bvanbuskirk/Desktop/experiments/{'object' if self.use_object else 'robot'}/{exp_name}/"
        self.buffer_path = "/home/bvanbuskirk/Desktop/experiments/buffers/"
        self.plot_path = self.exp_path + "plots/"
        self.state_path = self.exp_path + "states/"
        self.agent_path = self.exp_path + "agents/"

        Path(self.buffer_path).mkdir(parents=True, exist_ok=True)
        Path(self.plot_path).mkdir(parents=True, exist_ok=True)
        Path(self.state_path).mkdir(parents=True, exist_ok=True)
        Path(self.agent_path).mkdir(parents=True, exist_ok=True)

        self.robot_states = []
        self.object_states = []
        self.goal_states = []
        self.robot_model_errors = []
        self.object_model_errors = []

        self.logged_costs = self.logged_actions = False
        self.object_or_robot = 'object' if self.use_object else 'robot'

        self.episode_step = 0
        self.costs_dict = {}

    def __getattr__(self, key):
        return self.params[key]

    def log_step(self, state, action, goal, reset=False):
        self.states.append(state)
        self.actions.append(action)
        self.goals.append(goal)

        if reset:
            self.episode_step = 0
        else:
            self.episode_step += 1

    def log_prediction_errors(self, state, next_state, predicted_next_state, step):
        k = 2 if self.use_object else 1

        for i in range(k):
            start, end = i * dimensions["state_dim"], (i+1) * dimensions["state_dim"]
            x_state, x_next_state, x_pred_next_state = state[start:end], next_state[start:end], predicted_next_state[start:end]

            distance_travelled = np.linalg.norm((x_next_state - x_state)[:2])
            heading_travelled = signed_angle_difference(x_next_state[2], x_state[2])
            distance_error = np.linalg.norm((x_next_state - x_pred_next_state)[:2])
            heading_error = signed_angle_difference(x_next_state[2], x_pred_next_state[2])

            prediction_error = OrderedDict()
            robot_or_object = "robot" if i == 0 else "object"
            prediction_error[f"{robot_or_object}_error/distance"] = distance_error
            prediction_error[f"{robot_or_object}_error/normalized_distance"] = distance_error / (distance_travelled + 1e-8)
            prediction_error[f"{robot_or_object}_error/heading"] = heading_error
            prediction_error[f"{robot_or_object}_error/normalized_heading"] = np.abs(heading_error / (heading_travelled + 1e-8))
            wandb.log(prediction_error, step=step)

            print(f"\{robot_or_object} prediction errors:")
            for error_type, error_value in prediction_error.items():
                print(f"{error_type}: {error_value}")

    def log_mpc_costs(self, state, goal, step):
        cost_dict = self.agent.compute_costs(
                state[None, None, None, :], np.array([[[[0., 0.]]]]), goal[None, :], robot_goals=self.robot_goals
                )

        # initialize cost dict upon first step
        if len(self.costs_dict) == 0:
            self.costs_dict = {key: [value] for key, value in cost_dict.items()}

        total_cost = 0
        for cost_type, cost in cost_dict.items():
            cost_squeezed = cost.squeeze()
            cost_dict[cost_type] = cost_squeezed
            self.costs_dict[cost_type].append(cost_squeezed)
            total_cost += cost_squeezed * self.agent.policy.cost_weights_dict[cost_type]

        cost_dict["total"] = total_cost
        self.costs_dict["total"].append(total_cost)
        wandb.log(cost_dict, step=step)

    def log_img_and_plot(self):
        states = np.array(self.states)
        actions = np.array(self.actions)
        goals = np.array(self.goals)

        fig, ax = plt.subplots()

        ax.plot(goals[:, 0], goals[:, 1], color="green", linewidth=1.5, marker="*", label="Goal Trajectory")
        ax.plot(states[:, 0], states[:, 1], color="red", linewidth=1.5, marker=">", label="Robot Trajectory")

        if self.use_object:
            ax.plot(states[:, 3], states[:, 4], color="blue", linewidth=1.5, marker=".", label="Object Trajectory")

        ax.axis('equal')
        ax.set_xlim((self.corner[0], 0))
        ax.set_ylim((self.corner[1], 0))
        ax.legend()
        fig.show()



        if self.episode_step > 0:
            ...

        plt.plot(plot_robot_states[:, 0], plot_robot_states[:, 1], color="red", linewidth=1.5, marker=">", label="Robot Trajectory")

        if self.use_object:
            plt.plot(plot_object_states[:, 0], plot_object_states[:, 1], color="blue", linewidth=1.5, marker=".", label="Object Trajectory")

        if len(self.goal_states) == 1 or not self.plot:
            ax = plt.gca()
            ax.axis('equal')
            plt.xlim((self.corner[0], 0))
            plt.ylim((self.corner[1], 0))
            plt.legend()
            plt.ion()
            plt.show()

        if save:
            plt.plot(start_state[0], start_state[1], color="orange", marker="D", label="Starting Point", markersize=6)
            plt.title(f"{'Reversed' if reverse_lap else 'Standard'} Trajectory")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.legend()
            plt.draw()
            plt.pause(1.)

            plt.savefig(self.plot_path + f"lap{laps}_rb{replay_buffer.size}.png")





    def log_performance_metrics(self, costs, actions):
        dist_costs, heading_costs, total_costs = costs.T
        data = np.array([[dist_costs.mean(), dist_costs.std(), dist_costs.min(), dist_costs.max()],
                         [heading_costs.mean(), heading_costs.std(), heading_costs.min(), heading_costs.max()],
                         [total_costs.mean(), total_costs.std(), total_costs.min(), total_costs.max()]])

        print("rows: (dist, heading, total)")
        print("cols: (mean, std, min, max)")
        print("DATA:", data, "\n")

        self.log_costs(dist_costs, heading_costs, total_costs, costs)
        self.log_actions(actions)

    def log_costs(self, dist_costs, heading_costs, total_costs, costs_np):
        if not self.logged_costs:
            write_option = "w"
            self.logged_costs = True
        else:
            write_option = "a"

        with open(self.exp_path + "costs.csv", write_option, newline="") as csvfile:
            fwriter = csv.writer(csvfile, delimiter=',')
            for total_loss, dist_loss, heading_loss in zip(total_costs, dist_costs, heading_costs):
                fwriter.writerow([total_loss, dist_loss, heading_loss])
            fwriter.writerow([])

        with open(self.exp_path + "costs.npy", "wb") as f:
            np.save(f, costs_np)

    def reset_plot_states(self):
        self.robot_states = []
        self.object_states = []
        self.goal_states = []

    def log_states(self, state_dict, goal_pos, started):
        robot_pos, object_pos = state_dict["robot"], state_dict["object"]
        if started or self.plot:
            self.robot_states.append(robot_pos.copy())
            self.object_states.append(object_pos.copy())
            self.goal_states.append(goal_pos.copy())

        if self.plot:
            self.plot_states(save=False)

    def log_model_errors(self, error, object=False):
        if object:
            assert self.use_object
            self.object_model_errors.append(error)
        else:
            self.robot_model_errors.append(error)

    def reset_model_errors(self):
        self.robot_model_errors = []
        self.object_model_errors = []

    def plot_model_errors(self):
        robot_dist_errors, robot_dist_norm_errors, robot_heading_errors, robot_heading_norm_errors = np.array(self.robot_model_errors).T

        plt.figure()
        plt.hist(robot_dist_errors[robot_dist_errors < 0.03] * 100, bins=15)
        plt.title("Robot XY Prediction Error, PDF")
        plt.xlabel("Prediction Error (cm)")

        plt.figure()
        plt.hist(robot_dist_errors * 100, density=True, cumulative=True, bins=15)
        plt.title("Robot XY Prediction Error, CDF")
        plt.xlabel("Prediction Error (cm)")

        # plt.figure()
        # plt.hist(robot_dist_norm_errors, density=True, cumulative=True, bins=15)
        # plt.title("Robot XY Prediction Error per Distance Travelled, CDF")
        # plt.xlabel("Normalized Error (cm/cm)")

        # plt.figure()
        # plt.hist(robot_dist_norm_errors[robot_dist_norm_errors < 1], density=True, cumulative=True, bins=15)
        # plt.title("Robot XY Prediction Error per Distance Travelled, Zoomed CDF")
        # plt.xlabel("Normalized Error (cm/cm)")

        plt.figure()
        plt.hist(np.abs(robot_heading_errors * 180 / np.pi), density=True, cumulative=True, bins=15)
        plt.title("Robot Absolute Heading Prediction Error, CDF")
        plt.xlabel("Prediction Error (deg)")

        # plt.figure()
        # plt.hist(robot_heading_norm_errors, density=True, cumulative=True, bins=15)
        # plt.title("Robot Absolute Heading Prediction Error Normalized, CDF")
        # plt.xlabel("Normalized Error (deg/deg)")

        # plt.figure()
        # plt.hist(robot_heading_norm_errors[robot_heading_norm_errors < 1], density=True, cumulative=True, bins=15)
        # plt.title("Robot Absolute Heading Prediction Error Normalized, Zoomed CDF")
        # plt.xlabel("Normalized Error (deg/deg)")

        plt.figure()
        plt.hist(robot_heading_errors * 180 / np.pi, bins=15)
        plt.title("Robot Signed Heading Prediction Error, PDF")
        plt.xlabel("Prediction Error (deg)")

        import pdb;pdb.set_trace()

        if self.use_object:
            object_dist_errors, object_dist_norm_errors, object_heading_errors = np.array(self.object_model_errors).T

            plt.figure()
            plt.hist(object_dist_errors * 100)
            plt.title("Object XY Prediction Error Distribution")
            plt.xlabel("Prediction Error (cm)")

            plt.figure()
            plt.hist(object_heading_errors * 180 / np.pi)
            plt.title("Object Heading Prediction Error Distribution")
            plt.xlabel("Prediction Error (deg)")

    def dump_agent(self, agent, laps, replay_buffer):
        with open(self.agent_path + f"lap{laps}_rb{replay_buffer.size}.npy", "wb") as f:
            pkl.dump(agent, f)

    def dump_buffer(self, replay_buffer):
        with open(self.buffer_path + f"{self.object_or_robot}_buffer.pkl", "wb") as f:
            pkl.dump(replay_buffer, f)

    def load_buffer(self, robot_id):
        pkl_path = self.buffer_path + f"{self.object_or_robot}_buffer.pkl"
        validation_path = self.buffer_path + f"{self.object_or_robot}{robot_id}_validation_buffer.pkl"

        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                replay_buffer = pkl.load(f)
        else:
            replay_buffer = None

        if os.path.exists(validation_path):
            with open(validation_path, 'rb') as f:
                validation_buffer = pkl.load(f)
        else:
            validation_buffer = None

        return replay_buffer, validation_buffer

    def plot_states(self, save=False, laps=None, replay_buffer=None, start_state=None, reverse_lap=False):
        plot_goals = np.array(self.goal_states)
        plot_robot_states = np.array(self.robot_states)
        plot_object_states = np.array(self.object_states)

        if self.plot and len(self.goal_states) != 1:
            plot_goals = plot_goals[-2:]
            plot_robot_states = plot_robot_states[-2:]
            plot_object_states = plot_object_states[-2:]

        plt.plot(plot_goals[:, 0], plot_goals[:, 1], color="green", linewidth=1.5, marker="*", label="Goal Trajectory")
        plt.plot(plot_robot_states[:, 0], plot_robot_states[:, 1], color="red", linewidth=1.5, marker=">", label="Robot Trajectory")

        if self.use_object:
            plt.plot(plot_object_states[:, 0], plot_object_states[:, 1], color="blue", linewidth=1.5, marker=".", label="Object Trajectory")

        if len(self.goal_states) == 1 or not self.plot:
            ax = plt.gca()
            ax.axis('equal')
            plt.xlim((self.corner[0], 0))
            plt.ylim((self.corner[1], 0))
            plt.legend()
            plt.ion()
            plt.show()

        if save:
            plt.plot(start_state[0], start_state[1], color="orange", marker="D", label="Starting Point", markersize=6)
            plt.title(f"{'Reversed' if reverse_lap else 'Standard'} Trajectory")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.legend()
            plt.draw()
            plt.pause(1.)

            plt.savefig(self.plot_path + f"lap{laps}_rb{replay_buffer.size}.png")
            plt.close()

            state_dict = {"robot": plot_robot_states, "object": plot_object_states, "goal": plot_goals}
            for name in ["robot", "object", "goal"]:
                with open(self.state_path + f"/{name}_lap{laps}.npy", "wb") as f:
                    np.save(f, state_dict[name])

        else:
            plt.draw()
            plt.pause(0.001)