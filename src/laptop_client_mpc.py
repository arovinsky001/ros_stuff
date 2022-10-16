#!/usr/bin/python3

import os
import argparse
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import rospy
import csv
from pathlib import Path

from real_mpc_dynamics import *
from replay_buffer import ReplayBuffer

from ros_stuff.msg import RobotCmd, ProcessedStates
from ros_stuff.srv import CommandAction

# seed for reproducibility
SEED = 0
import torch; torch.manual_seed(SEED)
np.random.seed(SEED)

AGENT_PATH = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/agent.pkl"

class RealMPC():
    def __init__(self, robot_id, object_id, mpc_horizon, mpc_samples, n_rollouts, tolerance, lap_time, calibrate, plot,
                 new_buffer, pretrain, robot_goals, scale, mpc_softmax, save_freq, online, mpc_refine_iters,
                 pretrain_samples, random_steps, rate, use_all_data, debug, robot_pos, object_pos, corner_pos,
                 robot_vel, object_vel, state_timestamp, save_agent, load_agent, use_velocity, train_epochs, mpc_gamma,
                 ensemble):
        # flags for different stages of eval
        self.started = False
        self.done = False

        # counters
        self.steps = 0

        # AR tag ids for state lookup
        self.robot_id = robot_id
        self.object_id = object_id

        # states
        self.robot_pos = robot_pos
        self.object_pos = object_pos
        self.corner_pos = corner_pos
        self.robot_vel = robot_vel
        self.object_vel = object_vel
        self.state_timestamp = state_timestamp

        # MPC params
        self.mpc_refine_iters = mpc_refine_iters
        self.use_object = (self.object_id >= 0)

        # action params
        max_pwm = 0.999
        self.action_range = np.array([[-max_pwm, -max_pwm], [max_pwm, max_pwm]])
        self.duration = 0.2

        # online data collection/learning params
        self.random_steps = 0 if pretrain or load_agent else random_steps
        self.gradient_steps = 5
        self.online = online
        self.save_freq = save_freq
        self.pretrain_samples = pretrain_samples

        # system params
        self.n_clip = 3
        self.rate = rate

        # misc
        self.n_rollouts = n_rollouts
        self.tolerance = tolerance
        self.lap_time = lap_time
        self.plot = plot
        self.robot_goals = robot_goals
        self.scale = scale
        self.pretrain = pretrain
        self.save_agent = save_agent
        self.load_agent = load_agent
        self.use_all_data = use_all_data
        self.debug = debug
        self.use_velocity = use_velocity
        self.train_epochs = train_epochs
        self.last_action_time = 0.

        # set experiment title and setup for logging
        exp_title = f"{'object' if self.robot_goals else 'robot'}_goals"
        if self.pretrain or self.load_agent:
            exp_title += f"_pretrain{self.pretrain_samples}"
        if self.online:
            exp_title += "_online"

        self.exp_path = f"/home/bvanbuskirk/Desktop/experiments/{exp_title}/"
        self.plot_path = self.exp_path + "plots"
        self.state_path = self.exp_path + "states"
        self.agent_path = self.exp_path = "agents"
        Path(self.exp_path).mkdir(parents=True, exist_ok=True)
        Path(self.plot_path).mkdir(parents=True, exist_ok=True)
        Path(self.state_path).mkdir(parents=True, exist_ok=True)
        Path(self.agent_path).mkdir(parents=True, exist_ok=True)
        Path("/home/bvanbuskirk/Desktop/experiments/buffers/").mkdir(parents=True, exist_ok=True)

        #data for dynamics plot
        self.all_actions = []
        if os.path.exists("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl") and not new_buffer:
            with open("/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/replay_buffers/buffer.pkl", "rb") as f:
                self.replay_buffer = pkl.load(f)
        else:
            state_dim = 6 if self.use_object else 3
            state_dim *= 2 if self.use_velocity else 1
            self.replay_buffer = ReplayBuffer(capacity=100000, state_dim=state_dim, action_dim=2)

        if not self.debug:
            print(f"waiting for robot {self.robot_id} service")
            rospy.wait_for_service(f"/kami{self.robot_id}/server")
            self.service_proxy = rospy.ServiceProxy(f"/kami{self.robot_id}/server", CommandAction)
            print("connected to robot service")

        self.yaw_offset_path = "/home/bvanbuskirk/Desktop/MPCDynamicsKamigami/sim/data/yaw_offsets.npy"
        if not os.path.exists(self.yaw_offset_path) or calibrate:
            self.yaw_offsets = np.zeros(10)
            self.calibrate()

        self.yaw_offsets = np.load(self.yaw_offset_path)

        self.define_goal_trajectory()

        # weights for MPC cost terms
        self.cost_weights = {
            "heading": 0.15,
            "perpendicular": 0.,
            "action_norm": 0.,
            "distance_bonus": 0.,
            "separation": 0.,
            "heading_difference": 0.,
        }

        self.mpc_params = {
            "beta": 0.5,
            "gamma": mpc_gamma,
            "horizon": mpc_horizon,
            "sample_trajectories": mpc_samples,
            "robot_goals": robot_goals,
        }

        self.plot_robot_states = []
        self.plot_object_states = []
        self.plot_goals = []

        self.costs = np.empty((0, 4))      # dist, heading, perp, total

        self.time_elapsed = 0.
        self.logged_transitions = 0
        self.laps = 0
        self.n_prints = 0

        if load_agent:
            with open(AGENT_PATH, "rb") as f:
                self.agent = pkl.load(f)
        else:
            self.agent = MPCAgent(seed=SEED, dist=True, scale=self.scale, hidden_dim=200, hidden_depth=1,
                                  lr=0.001, dropout=0.0, std=0.1, ensemble=ensemble, use_object=self.use_object,
                                  use_velocity=args.use_velocity)
            if pretrain:
                self.train_model()
        np.set_printoptions(suppress=True)

    def calibrate(self):
        yaw_offsets = np.zeros(10)

        input(f"Place robot/object on the left calibration point, aligned with the calibration line and hit enter.")
        left_state = self.get_state()
        input(f"Place robot/object on the right calibration point, aligned with the calibration line and hit enter.")
        right_state = self.get_state()

        robot_left_state, robot_right_state = left_state[:3], right_state[:3]
        true_robot_vector = (robot_left_state - robot_right_state)[:2]
        true_robot_angle = np.arctan2(true_robot_vector[1], true_robot_vector[0])
        measured_robot_angle = robot_left_state[2]
        yaw_offsets[self.robot_id] = true_robot_angle - measured_robot_angle

        if self.use_object:
            object_left_state, object_right_state = left_state[3:6], right_state[3:6]
            true_object_vector = (object_left_state - object_right_state)[:2]
            true_object_angle = np.arctan2(true_object_vector[1], true_object_vector[0])
            measured_object_angle = object_left_state[2]
            yaw_offsets[self.object_id] = true_object_angle - measured_object_angle

        np.save(self.yaw_offset_path, yaw_offsets)

    def train_model(self):
        rb = self.replay_buffer

        if self.pretrain:
            n_samples = min(self.pretrain_samples, rb.size)

            states = rb.states[:n_samples]
            actions = rb.actions[:n_samples]
            next_states = rb.next_states[:n_samples]
        else:
            states, actions, next_states = rb.sample(rb.size)

        training_losses, test_losses, test_idx = self.agent.train(
                states, actions, next_states, set_scalers=True, epochs=self.train_epochs, batch_size=5000, use_all_data=self.use_all_data)

        if self.save_agent:
            with open(AGENT_PATH, "wb") as f:
                pkl.dump(self.agent, f)

        training_losses = np.array(training_losses).squeeze()
        test_losses = np.array(test_losses).squeeze()

        print("\nMIN TEST LOSS EPOCH:", test_losses.argmin())
        print("MIN TEST LOSS:", test_losses.min())

        state_delta = self.agent.dtu.state_delta_xysc(states, next_states)

        test_state, test_action = states[test_idx], actions[test_idx]
        test_state_delta = dtu.dcn(state_delta[test_idx])
        pred_state_delta = self.agent.get_prediction(test_state, test_action, self.agent.model, sample=False, delta=True)

        error = abs(pred_state_delta - test_state_delta)
        print("\nERROR MEAN:", error.mean(axis=0))
        print("ERROR STD:", error.std(axis=0))
        print("ERROR MAX:", error.max(axis=0))
        print("ERROR MIN:", error.min(axis=0))

        diffs = abs(test_state_delta)
        print("\nACTUAL MEAN:", diffs.mean(axis=0))
        print("ACTUAL STD:", diffs.std(axis=0))

        fig, axes = plt.subplots(1, 4)
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

        # for g in self.agent.models[0].optimizer.param_groups:
        #     g['lr'] = 1e-4

    def define_goal_trajectory(self):
        rospy.sleep(0.2)        # wait for states to be published and set

        back_circle_center_rel = np.array([0.7, 0.5])
        front_circle_center_rel = np.array([0.4, 0.5])

        self.back_circle_center = back_circle_center_rel * self.corner_pos[:2]
        self.front_circle_center = front_circle_center_rel * self.corner_pos[:2]
        self.radius = np.linalg.norm(self.back_circle_center - self.front_circle_center) / 2

    def record_costs(self, last_goal, goal):
        cost_dict = self.agent.compute_costs(
                self.get_state()[None, None, None, :], np.array([[[[0., 0.]]]]), last_goal, goal, robot_goals=self.robot_goals
                )

        total_cost = 0
        for cost_type, cost in cost_dict.items():
            cost_dict[cost_type] = cost.squeeze()
            if cost_type != "distance":
                total_cost += cost.squeeze() * self.cost_weights[cost_type]
        total_cost *= cost_dict["distance"]

        if self.started:
            costs_to_record = np.array([[cost_dict["distance"], cost_dict["heading"], cost_dict["perpendicular"], total_cost]])
            self.costs = np.append(self.costs, costs_to_record, axis=0)

        return cost_dict, total_cost

    def record_plot_states(self):
        robot_state_to_plot = self.robot_pos.copy()
        object_state_to_plot = self.object_pos.copy()
        self.plot_robot_states.append(robot_state_to_plot)
        self.plot_object_states.append(object_state_to_plot)
        self.plot_goals.append(self.last_goal.copy())

        if self.plot:
            self.plot_states_and_goals()

    def run(self):
        self.first_plot = True
        self.init_goal = self.get_goal()

        r = rospy.Rate(self.rate)
        while not rospy.is_shutdown():
            t = rospy.get_time()
            if self.started:
                self.record_plot_states()
            elif self.dist_to_start() < self.tolerance and (self.pretrain or self.replay_buffer.full or self.replay_buffer.idx >= self.random_steps):
                self.started = True
                self.last_goal = self.get_goal()
                self.time_elapsed = self.duration / 2

            state, action = self.step()

            if self.done:
                rospy.signal_shutdown("Finished! All robots reached goal.")
                return

            # if rospy.get_time() - t < 1 / self.rate:
            #     rospy.sleep(1 / self.rate - (rospy.get_time() - t))
            r.sleep()
            print("TIME:", rospy.get_time() - t)

            next_state = self.get_state()
            self.collect_training_data(state, action, next_state)
            if self.started:
                self.all_actions.append(action.tolist())

    def plot_states_and_goals(self):
        # negative to flip for perspective
        plot_goals = np.array(self.plot_goals)
        plot_robot_states = np.array(self.plot_robot_states)
        plot_object_states = np.array(self.plot_object_states)
        plt.plot(plot_goals[:, 0], plot_goals[:, 1], color="green", linewidth=1.5, marker="*", label="Goal Trajectory")
        plt.plot(plot_robot_states[:, 0], plot_robot_states[:, 1], color="red", linewidth=1.5, marker=">", label="Robot Trajectory")
        plt.plot(plot_object_states[:, 0], plot_object_states[:, 1], color="blue", linewidth=1.5, marker=".", label="Object Trajectory")
        if len(self.plot_goals) == 1 or not self.plot:
            plt.xlim((0, self.corner_pos[0]))
            plt.ylim((0, self.corner_pos[1]))
            plt.legend()
            plt.ion()
            plt.show()
        plt.draw()
        plt.pause(0.0001)

    def dist_to_start(self):
        state = self.get_state().squeeze()
        state = state[:3] if self.robot_goals else state[3:]
        return np.linalg.norm((state - self.init_goal)[:2])

    def step(self):
        if not self.pretrain and not self.load_agent and self.replay_buffer.size == self.random_steps:
            self.train_model()

        state = self.get_state()
        action = self.get_take_action(state)

        if self.online:
            self.update_model_online()

        self.check_rollout_finished()

        self.steps += 1
        return state, action

    def get_state(self):
        robot_pos = self.robot_pos.copy()
        robot_pos[2] = (robot_pos[2] + self.yaw_offsets[self.robot_id]) % (2 * np.pi)

        if self.use_object:
            object_pos = self.object_pos.copy()
            object_pos[2] = (object_pos[2] + self.yaw_offsets[self.object_id]) % (2 * np.pi)

            if self.use_velocity:
                return np.concatenate((robot_pos, object_pos, self.robot_vel, self.object_vel), axis=0)
            else:
                return np.concatenate((robot_pos, object_pos), axis=0)
        else:
            if self.use_velocity:
                return np.concatenate((robot_pos, self.robot_vel), axis=0)
            else:
                return robot_pos

    def get_take_action(self, state):
        goal = self.get_goal()
        state_for_last_goal = state[:3] if self.robot_goals else state[3:]
        last_goal = self.last_goal if self.started else state_for_last_goal
        last_goal = state_for_last_goal

        if self.replay_buffer.size >= self.random_steps:
            action = self.agent.get_action(state, last_goal, goal, cost_weights=self.cost_weights, params=self.mpc_params)
            self.time_elapsed += self.duration if self.started else 0
        else:
            print("TAKING RANDOM ACTION")
            action = np.random.uniform(*self.action_range, size=(1, self.action_range.shape[-1])).squeeze()

        action = np.clip(action, *self.action_range)
        action_req = RobotCmd()
        action_req.left_pwm = action[0]
        action_req.right_pwm = action[1]
        action_req.duration = self.duration

        if not self.debug:
            print("SENDING ACTION")
            # t1 = rospy.get_time()
            self.service_proxy(action_req, f"kami{self.robot_id}")
            # print(f"\n\nSERVICE TIME: {rospy.get_time() - t1}\n\n")
            t = rospy.get_time()

        self.n_prints += 1
        print(f"\n\n\n\nNO. {self.n_prints}")
        print("/////////////////////////////////////////////////")
        print("=================================================")
        print("GOAL:", goal)
        print("STATE:", state)
        print("ACTION:", action)
        print("ACTION NORM:", np.linalg.norm(action) / np.sqrt(2), "\n")

        cost_dict, total_cost = self.record_costs(last_goal, goal)

        for cost_type, cost in cost_dict.items():
            print(f"{cost_type}: {cost}")
        print("TOTAL:", total_cost)

        print("=================================================")
        print("/////////////////////////////////////////////////")

        self.last_goal = goal.copy() if self.started else None
        self.last_action_time = rospy.get_time()

        if not self.debug:
            post_action_sleep_time = 0.2
            if rospy.get_time() - t < post_action_sleep_time:
                rospy.sleep(post_action_sleep_time - (rospy.get_time() - t))

        return action

    def get_goal(self):
        t_rel = (self.time_elapsed % self.lap_time) / self.lap_time

        if t_rel < 0.5:
            theta = t_rel * 2 * 2 * np.pi
            center = self.front_circle_center
        else:
            theta = np.pi - ((t_rel - 0.5) * 2 * 2 * np.pi)
            center = self.back_circle_center

        goal = center + np.array([np.cos(theta), np.sin(theta)]) * self.radius
        return np.block([goal, 0.0])

    def check_rollout_finished(self):
        if self.time_elapsed > self.lap_time:
            self.laps += 1
            # Print current cumulative loss per lap completed
            dist_costs, heading_costs, perp_costs, total_costs = self.costs.T
            data = np.array([[dist_costs.mean(), dist_costs.std(), dist_costs.min(), dist_costs.max()],
                         [perp_costs.mean(), perp_costs.std(), perp_costs.min(), perp_costs.max()],
                         [heading_costs.mean(), heading_costs.std(), heading_costs.min(), heading_costs.max()],
                         [total_costs.mean(), total_costs.std(), total_costs.min(), total_costs.max()]])
            print("lap:", self.laps)
            print("rows: (dist, perp, heading, total)")
            print("cols: (mean, std, min, max)")
            print("DATA:", data, "\n")
            self.time_elapsed = 0.
            self.started = False
            self.plot_states_and_goals()

            plt.savefig(self.plot_path + f"lap{self.laps}_rb{self.replay_buffer.size}.png")
            plt.pause(1.0)
            plt.close()

            state_dict = {"robot": self.plot_robot_states, "object": self.plot_object_states, "goal": self.plot_goals}
            for name in ["robot", "object", "goal"]:
                with open(self.state_path + f"/{name}_lap{self.laps}.npy", "wb") as f:
                    np.save(f, state_dict[name])

            self.plot_robot_states = []
            self.plot_object_states = []
            self.plot_goals = []

            self.dump_performance_metrics()

            if self.online:
                with open(self.agent_path + f"lap{self.laps}_rb{self.replay_buffer.size}.npy", "wb") as f:
                    pkl.dump(self.agent, f)
                self.costs = np.empty((0, 4))

        if self.laps == self.n_rollouts:
            self.done = True

    def dump_performance_metrics(self):
        with open(self.exp_path + "costs.npy", "wb") as f:
            np.save(f, self.costs)

        dist_costs, heading_costs, perp_costs, total_costs = self.costs.T
        data = np.array([[dist_costs.mean(), dist_costs.std(), dist_costs.min(), dist_costs.max()],
                         [perp_costs.mean(), perp_costs.std(), perp_costs.min(), perp_costs.max()],
                         [heading_costs.mean(), heading_costs.std(), heading_costs.min(), heading_costs.max()],
                         [total_costs.mean(), total_costs.std(), total_costs.min(), total_costs.max()]])

        #log average performance per random step subset sizes
        with open(self.exp_path + "costs.csv", "a", newline="") as csvfile:
            fwriter = csv.writer(csvfile, delimiter=',')
            for total_loss, dist_loss, heading_loss in zip(total_costs, dist_costs, heading_costs):
                fwriter.writerow([total_loss, dist_loss, heading_loss])
            fwriter.writerow([])

        #log all actions made(revise this)
        with open(self.exp_path + "actions.csv", "a", newline="") as csvfile:
            fwriter = csv.writer(csvfile, delimiter=',')
            for action in self.all_actions:
                fwriter.writerow(action)
            fwriter.writerow([])

        print("rows: (dist, perp, heading, total)")
        print("cols: (mean, std, min, max)")
        print("DATA:", data, "\n")

    def collect_training_data(self, state, action, next_state):
        self.replay_buffer.add(state, action, next_state)

        if self.replay_buffer.idx % self.save_freq == 0:
            print(f"\nSAVING REPLAY BUFFER WITH {self.replay_buffer.capacity if self.replay_buffer.full else self.replay_buffer.idx} TRANSITIONS\n")
            with open("/home/bvanbuskirk/Desktop/experiments/buffers/buffer.pkl", "wb") as f:
                pkl.dump(self.replay_buffer, f)

    def update_model_online(self):
        if self.replay_buffer.size >= self.random_steps:
            for model in self.agent.models:
                for _ in range(self.gradient_steps):
                    states, actions, next_states = self.replay_buffer.sample(1000)
                    model.update(states, actions, next_states)

def update_state(msg):
    rs, os, cs = msg.robot_state, msg.object_state, msg.corner_state

    robot_pos[:] = np.array([rs.x, rs.y, rs.yaw])
    object_pos[:] = np.array([os.x, os.y, os.yaw])
    corner_pos[:] = np.array([cs.x, cs.y, cs.yaw])

    robot_vel[:] = np.array([rs.x_vel, rs.y_vel, rs.yaw_vel])
    object_vel[:] = np.array([os.x_vel, os.y_vel, os.yaw_vel])

    secs, nsecs = msg.header.stamp.secs, msg.header.stamp.nsecs
    state_timestamp[:] = secs + nsecs / 1e9


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do MPC.')
    parser.add_argument('-robot_id', type=int, help='robot id for rollout')
    parser.add_argument('-object_id', type=int, default=-1, help='object id for rollout')
    parser.add_argument('-mpc_horizon', type=int)
    parser.add_argument('-mpc_samples', type=int)
    parser.add_argument('-n_rollouts', type=int)
    parser.add_argument('-tolerance', type=float, default=0.05)
    parser.add_argument('-lap_time', type=float)
    parser.add_argument('-calibrate', action='store_true')
    parser.add_argument('-plot', action='store_true')
    parser.add_argument('-new_buffer', action='store_true')
    parser.add_argument('-pretrain', action='store_true')
    parser.add_argument('-robot_goals', action='store_true')
    parser.add_argument('-scale', action='store_true')
    parser.add_argument('-mpc_softmax', action='store_true')
    parser.add_argument('-online', action='store_true')
    parser.add_argument('-save_freq', type=int, default=50)
    parser.add_argument('-mpc_refine_iters', type=int, default=1)
    parser.add_argument('-pretrain_samples', type=int, default=500)
    parser.add_argument('-random_steps', type=int, default=500)
    parser.add_argument('-rate', type=float, default=1.)
    parser.add_argument('-use_all_data', action='store_true')
    parser.add_argument('-debug', action='store_true')
    parser.add_argument('-save_agent', action='store_true')
    parser.add_argument('-load_agent', action='store_true')
    parser.add_argument('-use_velocity', action='store_true')
    parser.add_argument('-train_epochs', type=int, default=200)
    parser.add_argument('-mpc_gamma', type=float, default=50000)
    parser.add_argument('-ensemble', type=int, default=1)
    args = parser.parse_args()

    pos_dim = 3
    vel_dim = 3
    robot_pos = np.empty(pos_dim)
    object_pos = np.empty(pos_dim)
    corner_pos = np.empty(pos_dim)
    robot_vel = np.empty(vel_dim)
    object_vel = np.empty(vel_dim)
    state_timestamp = np.empty(1)

    rospy.init_node("laptop_client_mpc")

    print("waiting for /processed_state topic from state publisher")
    rospy.Subscriber("/processed_state", ProcessedStates, update_state, queue_size=1)
    print("subscribed to /processed_state")

    r = RealMPC(args.robot_id, args.object_id, args.mpc_horizon, args.mpc_samples, args.n_rollouts, args.tolerance,
                args.lap_time, args.calibrate, args.plot, args.new_buffer, args.pretrain, args.robot_goals, args.scale,
                args.mpc_softmax, args.save_freq, args.online, args.mpc_refine_iters, args.pretrain_samples,
                args.random_steps, args.rate, args.use_all_data, args.debug, robot_pos, object_pos, corner_pos,
                robot_vel, object_vel, state_timestamp, args.save_agent, args.load_agent, args.use_velocity,
                args.train_epochs, args.mpc_gamma, args.ensemble)
    r.run()
