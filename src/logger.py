#!/usr/bin/python3

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import csv


class Logger:
    def __init__(self, experiment, plot):
        exp_title = f"{'robot' if experiment.robot_goals else 'object'}goals"
        if experiment.pretrain or experiment.load_agent:
            exp_title += f"_pretrain{experiment.pretrain_samples}"
        if experiment.online:
            exp_title += "_online"

        self.exp_path = f"/home/bvanbuskirk/Desktop/experiments/{'object' if experiment.use_object else 'robot'}/{exp_title}/"
        self.buffer_path = "/home/bvanbuskirk/Desktop/experiments/buffers/"
        self.plot_path = self.exp_path + "plots/"
        self.state_path = self.exp_path + "states/"
        self.agent_path = self.exp_path = "agents/"

        Path(self.buffer_path).mkdir(parents=True, exist_ok=True)
        Path(self.plot_path).mkdir(parents=True, exist_ok=True)
        Path(self.state_path).mkdir(parents=True, exist_ok=True)
        Path(self.agent_path).mkdir(parents=True, exist_ok=True)

        self.robot_states = []
        self.object_states = []
        self.goal_states = []

        self.use_object = experiment.use_object
        self.plot = plot

    def log_performance_metrics(self, costs, actions):
        dist_costs, heading_costs, perp_costs, total_costs = costs.T
        data = np.array([[dist_costs.mean(), dist_costs.std(), dist_costs.min(), dist_costs.max()],
                         [perp_costs.mean(), perp_costs.std(), perp_costs.min(), perp_costs.max()],
                         [heading_costs.mean(), heading_costs.std(), heading_costs.min(), heading_costs.max()],
                         [total_costs.mean(), total_costs.std(), total_costs.min(), total_costs.max()]])

        print("rows: (dist, perp, heading, total)")
        print("cols: (mean, std, min, max)")
        print("DATA:", data, "\n")

        self.log_costs(dist_costs, heading_costs, total_costs, costs)
        self.log_actions(actions)

    def log_costs(self, dist_costs, heading_costs, total_costs, costs_np):
        with open(self.exp_path + "costs.csv", "a", newline="") as csvfile:
            fwriter = csv.writer(csvfile, delimiter=',')
            for total_loss, dist_loss, heading_loss in zip(total_costs, dist_costs, heading_costs):
                fwriter.writerow([total_loss, dist_loss, heading_loss])
            fwriter.writerow([])

        with open(self.exp_path + "costs.npy", "wb") as f:
            np.save(f, costs_np)

    def log_actions(self, actions):
        with open(self.exp_path + "actions.csv", "a", newline="") as csvfile:
            fwriter = csv.writer(csvfile, delimiter=',')
            for action in actions:
                fwriter.writerow(action)
            fwriter.writerow([])

    def log_states(self, robot_pos, object_pos, goal_pos):
        self.robot_states.append(robot_pos.copy())
        self.object_states.append(object_pos.copy())
        self.goal_states.append(goal_pos.copy())

        if self.plot:
            self.plot_states(save=False)

    def dump_agent(self, agent, laps, replay_buffer):
        with open(self.agent_path + f"lap{laps}_rb{replay_buffer.size}.npy", "wb") as f:
            pkl.dump(agent, f)

    def dump_buffer(self, replay_buffer):
        with open(self.buffer_path + f"{'object' if self.use_object else 'robot'}_buffer.pkl", "wb") as f:
            pkl.dump(replay_buffer, f)

    def load_buffer(self):
        pkl_path = self.buffer_path + f"{'object' if self.use_object else 'robot'}_buffer.pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                return pkl.load(f)
        else:
            return None

    def plot_states(self, corners, save=False, laps=None):
        plot_goals = np.array(self.goal_states)
        plot_robot_states = np.array(self.robot_states)
        plot_object_states = np.array(self.object_states)
        plt.plot(plot_goals[:, 0], plot_goals[:, 1], color="green", linewidth=1.5, marker="*", label="Goal Trajectory")
        plt.plot(plot_robot_states[:, 0], plot_robot_states[:, 1], color="red", linewidth=1.5, marker=">", label="Robot Trajectory")
        plt.plot(plot_object_states[:, 0], plot_object_states[:, 1], color="blue", linewidth=1.5, marker=".", label="Object Trajectory")

        if len(self.goal_states) == 1 or not self.plot:
            plt.xlim((corners[0], 0))
            plt.ylim((corners[1], 0))
            plt.legend()
            plt.ion()
            plt.show()

        plt.draw()
        plt.pause(0.0001)

        if save:
            plt.savefig(self.plot_path + f"lap{laps}_rb{self.replay_buffer.size}.png")
            plt.pause(5.)
            plt.close()

            state_dict = {"robot": plot_robot_states, "object": plot_object_states, "goal": plot_goals}
            for name in ["robot", "object", "goal"]:
                with open(self.state_path + f"/{name}_lap{laps}.npy", "wb") as f:
                    np.save(f, state_dict[name])
